"""
10_cross_validation.py
=======================
5-fold cross-validation of the HITL v3 configuration across different
seed/pool splits. Each fold uses a different random seed to split the
training data into seed (70%) and pool (30%) sets, then runs the full
v3 HITL pipeline (lr=5e-6, 50 corrections/round, 1 epoch, replay=100)
for PubMedBERT, ClinicalBERT, and RoBERTa.

This directly addresses the concern that the 70/30 split used in the
main experiments was arbitrary (a single split) — 5-fold CV reports
mean ± std across five independent splits, providing a much stronger
robustness claim than a single run.

Resumable: skips any (fold, model) combination already completed.

Usage
-----
    python src/10_cross_validation.py

Output
------
    results/cv/fold_<seed>_<model_key>.json     (per fold-model curve)
    results/cv_summary.json                      (aggregated mean ± std)
    results/cv_summary.txt                       (human-readable table)
"""

import inspect
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import (AutoModelForSequenceClassification,
                          AutoTokenizer, Trainer, TrainingArguments)

from config import (BATCH_SIZE, DATA_DIR, FP16, HITL_FINETUNE_EPOCHS,
                    HITL_LEARNING_RATE, HITL_MODELS, HITL_ROUNDS,
                    CORRECTIONS_PER_ROUND, REPLAY_BUFFER_SIZE, ID2LABEL,
                    LABEL2ID, MAX_INPUT_LENGTH, MODELS_DIR, NUM_LABELS,
                    RESULTS_DIR, TRANSFORMER_MODELS)
from utils import (ClinicalDataset, build_input_text, compute_metrics,
                   load_split, make_hf_compute_metrics, save_results,
                   set_seed)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

# ── Fold configuration ─────────────────────────────────────────────────────
FOLD_SEEDS = [42, 123, 456, 789, 2024]
SEED_FRACTION_CV = 0.70   # consistent with the main v3 (seed70) experiments

CV_DIR = RESULTS_DIR / "cv"
CV_DIR.mkdir(parents=True, exist_ok=True)


# ── Version-agnostic TrainingArguments helper ─────────────────────────────────

def make_training_args(output_dir, **kwargs):
    """
    Builds TrainingArguments while auto-detecting whether this transformers
    version expects 'eval_strategy' or 'evaluation_strategy'.
    """
    params = inspect.signature(TrainingArguments.__init__).parameters
    eval_key = "eval_strategy" if "eval_strategy" in params else "evaluation_strategy"
    kwargs[eval_key] = kwargs.pop("eval_strategy_value", "no")
    return TrainingArguments(output_dir=str(output_dir), **kwargs)


# ── Tokenisation helper ────────────────────────────────────────────────────────

def tokenize_texts(tokenizer, texts):
    return tokenizer(texts, padding=True, truncation=True,
                     max_length=MAX_INPUT_LENGTH, return_tensors="pt")


# ── Single fold-model HITL run ──────────────────────────────────────────────────

def run_fold(fold_seed: int, model_key: str,
            train_df: pd.DataFrame, val_df: pd.DataFrame,
            test_df: pd.DataFrame) -> dict:
    """
    Run the full v3 HITL pipeline for one model under one fold's seed/pool split.
    Returns the round-by-round curve dict.
    """
    model_name = TRANSFORMER_MODELS[model_key]
    logger.info("=" * 60)
    logger.info("Fold seed=%d | Model=%s", fold_seed, model_key)
    logger.info("=" * 60)

    set_seed(fold_seed)

    # ── Split into seed / pool using this fold's seed ─────────────────────────
    shuffled  = train_df.sample(frac=1.0, random_state=fold_seed).reset_index(drop=True)
    n_seed    = int(len(shuffled) * SEED_FRACTION_CV)
    seed_df   = shuffled.iloc[:n_seed].reset_index(drop=True)
    pool_df   = shuffled.iloc[n_seed:].reset_index(drop=True)
    logger.info("Seed: %d | Pool: %d", len(seed_df), len(pool_df))

    # ── Load fresh pretrained model (not reusing other folds' weights) ────────
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=NUM_LABELS,
        id2label=ID2LABEL, label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    test_texts = [build_input_text(r) for _, r in test_df.iterrows()]
    test_enc   = tokenize_texts(tokenizer, test_texts)
    test_dataset = ClinicalDataset(test_enc, test_df["label"].tolist())

    output_dir = MODELS_DIR / "cv_tmp" / f"fold{fold_seed}_{model_key}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Baseline training on seed set ──────────────────────────────────────────
    # FIX: previously trained for 3 fixed epochs with no validation monitoring,
    # silently diverging from 04_hitl.py's load_best_model_at_end behaviour.
    # Now mirrors the original methodology exactly: evaluate each epoch on the
    # held-out validation set, select the best checkpoint by macro F1.
    seed_texts = [build_input_text(r) for _, r in seed_df.iterrows()]
    seed_enc   = tokenize_texts(tokenizer, seed_texts)
    seed_dataset = ClinicalDataset(seed_enc, seed_df["label"].tolist())

    val_texts = [build_input_text(r) for _, r in val_df.iterrows()]
    val_enc   = tokenize_texts(tokenizer, val_texts)
    val_dataset = ClinicalDataset(val_enc, val_df["label"].tolist())

    args = make_training_args(
        output_dir / "seed_ckpt",
        num_train_epochs=3,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=FP16 and torch.cuda.is_available(),
        seed=fold_seed,
        logging_steps=100,
        report_to="none",
        eval_strategy_value="epoch",
    )
    trainer = Trainer(
        model=model, args=args,
        train_dataset=seed_dataset,
        eval_dataset=val_dataset,
        compute_metrics=make_hf_compute_metrics(),
    )
    trainer.train()

    # ── Evaluate seed model (round 0) ────────────────────────────────────────
    def evaluate_on_test():
        raw = trainer.predict(test_dataset)
        logits = raw.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        y_pred = np.argmax(logits, axis=-1)
        y_prob = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        y_true = test_df["label"].to_numpy()
        return compute_metrics(y_true, y_pred, y_prob)

    curve = {"round": [], "accuracy": [], "macro_f1": [], "auc": []}
    m0 = evaluate_on_test()
    curve["round"].append(0)
    curve["accuracy"].append(m0["accuracy"])
    curve["macro_f1"].append(m0["macro_f1"])
    curve["auc"].append(m0["auc"])
    logger.info("Round 0 (seed): acc=%.4f auc=%.4f", m0["accuracy"], m0["auc"])

    remaining_pool = pool_df.copy().reset_index(drop=True)

    # ── HITL rounds ───────────────────────────────────────────────────────────
    for rnd in range(1, HITL_ROUNDS + 1):
        if len(remaining_pool) == 0:
            logger.info("Pool exhausted at round %d — stopping.", rnd)
            break

        # Predict on remaining pool
        pool_texts = [build_input_text(r) for _, r in remaining_pool.iterrows()]
        pool_enc   = tokenize_texts(tokenizer, pool_texts)
        pool_dataset = ClinicalDataset(pool_enc, remaining_pool["label"].tolist())

        raw_pool = trainer.predict(pool_dataset)
        logits   = raw_pool.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        pool_pred = np.argmax(logits, axis=-1)
        pool_true = remaining_pool["label"].to_numpy()

        wrong_idx = np.where(pool_pred != pool_true)[0]
        if len(wrong_idx) == 0:
            logger.info("No misclassifications at round %d — stopping.", rnd)
            break

        n_corr = min(CORRECTIONS_PER_ROUND, len(wrong_idx))
        chosen_idx = wrong_idx[:n_corr]
        corrections_df = remaining_pool.iloc[chosen_idx].reset_index(drop=True)

        # Remove corrected examples from pool
        remaining_pool = remaining_pool.drop(
            remaining_pool.index[chosen_idx]
        ).reset_index(drop=True)

        # Sample replay buffer from seed set
        replay_df = seed_df.sample(
            n=min(REPLAY_BUFFER_SIZE, len(seed_df)),
            random_state=fold_seed * 1000 + rnd
        )

        combined_df = pd.concat([corrections_df, replay_df], ignore_index=True)
        combined_texts = [build_input_text(r) for _, r in combined_df.iterrows()]
        combined_enc   = tokenize_texts(tokenizer, combined_texts)
        combined_dataset = ClinicalDataset(combined_enc, combined_df["label"].tolist())

        ft_args = make_training_args(
            output_dir / f"round{rnd}_ckpt",
            num_train_epochs=HITL_FINETUNE_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            learning_rate=HITL_LEARNING_RATE,
            save_strategy="no",
            fp16=FP16 and torch.cuda.is_available(),
            seed=fold_seed,
            logging_steps=100,
            report_to="none",
            eval_strategy_value="no",
        )
        ft_trainer = Trainer(model=model, args=ft_args, train_dataset=combined_dataset)
        ft_trainer.train()
        trainer = ft_trainer   # carry forward for next round's predict/eval

        m = evaluate_on_test()
        curve["round"].append(rnd)
        curve["accuracy"].append(m["accuracy"])
        curve["macro_f1"].append(m["macro_f1"])
        curve["auc"].append(m["auc"])
        logger.info("Round %d: acc=%.4f auc=%.4f (pool remaining: %d)",
                   rnd, m["accuracy"], m["auc"], len(remaining_pool))

    # Clean up temp checkpoints to save disk space
    import shutil
    shutil.rmtree(output_dir, ignore_errors=True)

    return curve


# ── Aggregation ──────────────────────────────────────────────────────────────

def aggregate_cv_results() -> dict:
    """
    Load all completed fold-model result files and compute mean ± std
    for seed accuracy, final accuracy, min accuracy, and AUC range.
    """
    summary = {}
    for model_key in HITL_MODELS:
        seed_accs, final_accs, min_accs, auc_ranges, n_rounds = [], [], [], [], []

        for fold_seed in FOLD_SEEDS:
            path = CV_DIR / f"fold_{fold_seed}_{model_key}.json"
            if not path.exists():
                continue
            with open(path) as f:
                curve = json.load(f)

            seed_accs.append(curve["accuracy"][0])
            final_accs.append(curve["accuracy"][-1])
            min_accs.append(min(curve["accuracy"]))
            auc_ranges.append(max(curve["auc"]) - min(curve["auc"]))
            n_rounds.append(len(curve["round"]) - 1)

        if not seed_accs:
            continue

        summary[model_key] = {
            "n_folds_completed": len(seed_accs),
            "seed_accuracy":  {"mean": float(np.mean(seed_accs)),  "std": float(np.std(seed_accs))},
            "final_accuracy": {"mean": float(np.mean(final_accs)), "std": float(np.std(final_accs))},
            "min_accuracy":   {"mean": float(np.mean(min_accs)),   "std": float(np.std(min_accs))},
            "auc_range":      {"mean": float(np.mean(auc_ranges)), "std": float(np.std(auc_ranges))},
            "mean_rounds":    float(np.mean(n_rounds)),
        }
    return summary


def write_summary(summary: dict) -> None:
    lines = ["=" * 90, "5-FOLD CROSS-VALIDATION SUMMARY — HITL v3 Configuration", "=" * 90,
             f"{'Model':<15} {'Folds':>6} {'Seed Acc':>16} {'Final Acc':>16} "
             f"{'Min Acc':>16} {'AUC Range':>16}", "-" * 90]
    for model_key, s in summary.items():
        lines.append(
            f"{model_key:<15} {s['n_folds_completed']:>6} "
            f"{s['seed_accuracy']['mean']:.3f}±{s['seed_accuracy']['std']:.3f}    "
            f"{s['final_accuracy']['mean']:.3f}±{s['final_accuracy']['std']:.3f}    "
            f"{s['min_accuracy']['mean']:.3f}±{s['min_accuracy']['std']:.3f}    "
            f"{s['auc_range']['mean']:.4f}±{s['auc_range']['std']:.4f}"
        )
    lines.append("=" * 90)
    text = "\n".join(lines)
    print("\n" + text)
    (RESULTS_DIR / "cv_summary.txt").write_text(text)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    train_df = load_split(DATA_DIR / "train_full_with_rationales.csv")
    val_df   = load_split(DATA_DIR / "validation_full_with_rationales.csv")
    test_df  = load_split(DATA_DIR / "test_full_with_rationales.csv")

    total_runs = len(FOLD_SEEDS) * len(HITL_MODELS)
    completed  = 0

    for fold_seed in FOLD_SEEDS:
        for model_key in HITL_MODELS:
            out_path = CV_DIR / f"fold_{fold_seed}_{model_key}.json"
            if out_path.exists():
                logger.info("Skipping fold=%d model=%s — already completed.",
                           fold_seed, model_key)
                completed += 1
                continue

            curve = run_fold(fold_seed, model_key, train_df, val_df, test_df)
            save_results(curve, out_path)
            completed += 1
            logger.info("Progress: %d / %d fold-model runs complete.",
                       completed, total_runs)

    # ── Aggregate and report ─────────────────────────────────────────────────
    summary = aggregate_cv_results()
    save_results(summary, RESULTS_DIR / "cv_summary.json")
    write_summary(summary)

    logger.info("5-fold cross-validation complete.")


if __name__ == "__main__":
    main()