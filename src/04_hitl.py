"""
04_hitl.py
==========
Implements the incremental Human-in-the-Loop (HITL) fine-tuning workflow for
T5-small, BioBERT, and ClinicalBERT.

The workflow per round t:
    1. Predict labels for the pool set with the current model θ_t
    2. Identify misclassified examples (simulated educator corrections)
    3. Sample min(CORRECTIONS_PER_ROUND, n_misclassified) corrected examples
    4. Fine-tune the model on corrected examples only:
           θ_{t+1} = θ_t − η ∇_{θ_t} L(D_corr)
    5. Evaluate on the held-out test set (never updated)
    6. Record accuracy, macro F1, and AUC

The HITL process is repeated for HITL_ROUNDS rounds.
Educator corrections are simulated: misclassified examples are relabelled
using their ground-truth labels, mimicking what an educator would do.

Usage
-----
    python src/04_hitl.py

Output
------
    results/hitl_results.json
    results/figures/learning_curve_<model>.png
"""

import copy
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import (AutoModelForSequenceClassification,
                          AutoTokenizer, Trainer, TrainingArguments)

from config import (BATCH_SIZE, CORRECTIONS_PER_ROUND, DATA_DIR, FP16,
                    HITL_FINETUNE_EPOCHS, HITL_LEARNING_RATE,
                    HITL_MODELS, HITL_ROUNDS, ID2LABEL, LABEL2ID,
                    MAX_INPUT_LENGTH, MODELS_DIR, NUM_LABELS,
                    RANDOM_SEED, RESULTS_DIR, SEED_FRACTION,
                    TRANSFORMER_MODELS)
from utils import (ClinicalDataset, build_input_text, compute_metrics,
                   load_split, make_hf_compute_metrics,
                   save_results, set_seed)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


# ── Data preparation ──────────────────────────────────────────────────────────

def prepare_seed_and_pool(
    train_df: pd.DataFrame,
    seed_fraction: float = SEED_FRACTION,
    random_state: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the training set into a seed set (for initial training) and a pool
    (from which corrections are drawn each HITL round).

    Parameters
    ----------
    train_df      : full training DataFrame
    seed_fraction : proportion used as seed (default 0.70)

    Returns
    -------
    seed_df, pool_df
    """
    seed_df = train_df.sample(
        frac=seed_fraction, random_state=random_state
    ).reset_index(drop=True)
    pool_df = train_df.drop(seed_df.index).reset_index(drop=True)
    logger.info("Seed: %d | Pool: %d", len(seed_df), len(pool_df))
    return seed_df, pool_df


# ── Tokenisation helper ───────────────────────────────────────────────────────

def encode(tokenizer, df: pd.DataFrame,
           max_length: int = MAX_INPUT_LENGTH):
    """Tokenise a DataFrame into a ClinicalDataset."""
    texts = [build_input_text(row) for _, row in df.iterrows()]
    enc   = tokenizer(
        texts, padding=True, truncation=True,
        max_length=max_length, return_tensors="pt",
    )
    return ClinicalDataset(enc, df["label"].tolist())


# ── HITL core ─────────────────────────────────────────────────────────────────

def baseline_train(
    model, tokenizer, seed_df: pd.DataFrame, val_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Train the model on the seed set to establish θ_0.
    This is the one-time baseline training stage (Zone 1 in the workflow diagram).
    """
    logger.info("  Baseline training on seed set (%d samples) …", len(seed_df))

    seed_dataset = encode(tokenizer, seed_df)
    val_dataset  = encode(tokenizer, val_df)

    args = TrainingArguments(
        output_dir              = str(output_dir / "seed_checkpoint"),
        num_train_epochs        = 3,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        learning_rate           = HITL_LEARNING_RATE,
        evaluation_strategy     = "epoch",
        save_strategy           = "no",
        fp16                    = FP16 and torch.cuda.is_available(),
        seed                    = RANDOM_SEED,
        logging_steps           = 50,
        report_to               = "none",
    )
    trainer = Trainer(
        model           = model,
        args            = args,
        train_dataset   = seed_dataset,
        eval_dataset    = val_dataset,
        compute_metrics = make_hf_compute_metrics(),
    )
    trainer.train()
    logger.info("  Baseline training complete.")


def predict_pool(
    model, tokenizer, pool_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference on the pool and return predictions and probabilities.

    Returns
    -------
    y_pred : (n,)   predicted integer labels
    y_prob : (n, 3) class probabilities
    """
    dataset = encode(tokenizer, pool_df)
    args    = TrainingArguments(
        output_dir                  = "/tmp/hitl_pred",
        per_device_eval_batch_size  = BATCH_SIZE,
        fp16                        = FP16 and torch.cuda.is_available(),
        report_to                   = "none",
        no_cuda                     = not torch.cuda.is_available(),
    )
    trainer = Trainer(model=model, args=args)
    raw     = trainer.predict(dataset)
    y_pred  = np.argmax(raw.predictions, axis=-1)
    y_prob  = torch.softmax(torch.tensor(raw.predictions), dim=-1).numpy()
    return y_pred, y_prob


def get_corrections(
    pool_df: pd.DataFrame,
    y_pred:  np.ndarray,
    n:       int = CORRECTIONS_PER_ROUND,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identify misclassified pool examples and select up to n of them as
    simulated educator corrections.

    The 'correction' is simply restoring the ground-truth label — in a real
    system, this is where an educator would review and relabel the examples.

    Returns
    -------
    corrections : DataFrame of up to n corrected examples
    remaining_pool : pool DataFrame with corrections removed
    """
    y_true     = pool_df["label"].to_numpy()
    wrong_mask = (y_pred != y_true)
    wrong_idx  = pool_df.index[wrong_mask].tolist()

    n_available = len(wrong_idx)
    n_selected  = min(n, n_available)

    if n_available == 0:
        logger.info("  No misclassifications found — pool exhausted or model converged.")
        return pd.DataFrame(), pool_df

    selected_idx  = wrong_idx[:n_selected]
    corrections   = pool_df.loc[selected_idx].copy()
    # Ground-truth labels are already correct; this line makes the intent explicit
    corrections["label"] = y_true[pool_df.index.get_indexer(selected_idx)]

    remaining_pool = pool_df.drop(selected_idx).reset_index(drop=True)
    logger.info("  Misclassified: %d | Selected for correction: %d | "
                "Pool remaining: %d",
                n_available, n_selected, len(remaining_pool))
    return corrections, remaining_pool


def incremental_finetune(
    model, tokenizer, corrections: pd.DataFrame, output_dir: Path,
) -> None:
    """
    Fine-tune the model on newly corrected samples only, keeping all
    previously learned parameters (θ_t → θ_{t+1}).

    This lightweight update avoids memory growth from accumulating the full
    expanded dataset, while preserving the cumulative learning effect.

    Update rule:  θ_{t+1} = θ_t − η ∇_{θ_t} L(D_corr)
    """
    if corrections.empty:
        return

    corr_dataset = encode(tokenizer, corrections)
    args = TrainingArguments(
        output_dir              = str(output_dir / "hitl_incremental"),
        num_train_epochs        = HITL_FINETUNE_EPOCHS,
        per_device_train_batch_size = min(BATCH_SIZE, len(corrections)),
        learning_rate           = HITL_LEARNING_RATE,
        fp16                    = FP16 and torch.cuda.is_available(),
        save_strategy           = "no",
        seed                    = RANDOM_SEED,
        logging_steps           = 10,
        report_to               = "none",
    )
    trainer = Trainer(
        model        = model,
        args         = args,
        train_dataset = corr_dataset,
    )
    trainer.train()


def evaluate_on_test(
    model, tokenizer, test_df: pd.DataFrame,
) -> dict:
    """Evaluate the current model on the held-out test set."""
    test_dataset = encode(tokenizer, test_df)
    args = TrainingArguments(
        output_dir                  = "/tmp/hitl_eval",
        per_device_eval_batch_size  = BATCH_SIZE,
        fp16                        = FP16 and torch.cuda.is_available(),
        report_to                   = "none",
        no_cuda                     = not torch.cuda.is_available(),
    )
    trainer = Trainer(model=model, args=args)
    raw     = trainer.predict(test_dataset)
    y_pred  = np.argmax(raw.predictions, axis=-1)
    y_prob  = torch.softmax(torch.tensor(raw.predictions), dim=-1).numpy()
    y_true  = test_df["label"].to_numpy()
    return compute_metrics(y_true, y_pred, y_prob)


# ── Full HITL loop for one model ──────────────────────────────────────────────

def run_hitl(
    model_key: str,
    model_name: str,
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
) -> Dict[str, List]:
    """
    Run the full HITL pipeline for a single model.

    Returns
    -------
    A dict of learning curves: {
        "round": [0, 1, ..., R],
        "accuracy": [...], "macro_f1": [...], "auc": [...]
    }
    Round 0 = pre-HITL (seed-trained baseline).
    """
    logger.info("▶ HITL — %s", model_key)
    output_dir = MODELS_DIR / f"{model_key}_hitl"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokeniser and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels  = NUM_LABELS,
        id2label    = ID2LABEL,
        label2id    = LABEL2ID,
        ignore_mismatched_sizes = True,
    )

    # Prepare seed and pool
    seed_df, pool_df = prepare_seed_and_pool(train_df)

    # Zone 1: baseline training on seed set
    baseline_train(model, tokenizer, seed_df, val_df, output_dir)

    # Round 0 evaluation (pre-HITL)
    pre_metrics = evaluate_on_test(model, tokenizer, test_df)
    logger.info("  Round 0 (pre-HITL): %s", pre_metrics)

    curves: Dict[str, List] = {
        "round":    [0],
        "accuracy": [pre_metrics["accuracy"]],
        "macro_f1": [pre_metrics["macro_f1"]],
        "auc":      [pre_metrics["auc"]],
    }

    # Zone 2: HITL loop
    for r in range(1, HITL_ROUNDS + 1):
        logger.info("  Round %d / %d", r, HITL_ROUNDS)

        if pool_df.empty:
            logger.info("  Pool exhausted at round %d.", r)
            break

        # Step 1: predict on pool
        y_pred, _ = predict_pool(model, tokenizer, pool_df)

        # Step 2: simulated educator corrections
        corrections, pool_df = get_corrections(pool_df, y_pred)

        if corrections.empty:
            logger.info("  No corrections available — stopping early.")
            break

        # Step 3: incremental fine-tuning
        incremental_finetune(model, tokenizer, corrections, output_dir)

        # Step 4: evaluate on test set
        metrics = evaluate_on_test(model, tokenizer, test_df)
        logger.info("  Round %d: accuracy=%.4f | F1=%.4f | AUC=%.4f",
                    r, metrics["accuracy"], metrics["macro_f1"], metrics["auc"])

        curves["round"].append(r)
        curves["accuracy"].append(metrics["accuracy"])
        curves["macro_f1"].append(metrics["macro_f1"])
        curves["auc"].append(metrics["auc"])

    # Save final HITL model
    model.save_pretrained(str(output_dir / "final_model"))
    tokenizer.save_pretrained(str(output_dir / "final_model"))
    logger.info("  Final HITL model saved → %s/final_model", output_dir)

    return curves


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_learning_curve(curves: dict, model_key: str, out_path: Path) -> None:
    """
    Learning curve: test accuracy across HITL rounds.
    Includes a dashed baseline line at round 0.
    """
    rounds    = curves["round"]
    accuracy  = curves["accuracy"]
    baseline  = accuracy[0]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rounds, accuracy, marker="o", color="steelblue",
            label="Test accuracy")
    ax.axhline(baseline, color="gray", linestyle="--", linewidth=1,
               label=f"Pre-HITL baseline ({baseline:.3f})")

    ax.set_xlabel("HITL Round")
    ax.set_ylabel("Test Accuracy")
    ax.set_title(f"HITL Learning Curve — {model_key}")
    ax.set_xticks([r for r in rounds if r % 2 == 0])   # integer ticks only
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Learning curve saved → %s", out_path)


def plot_all_curves(all_curves: dict, out_path: Path) -> None:
    """Overlay learning curves for all HITL models in a single figure."""
    fig, ax = plt.subplots(figsize=(9, 5))
    colors  = {"t5-small": "steelblue", "biobert": "darkorange",
                "clinicalbert": "forestgreen"}

    for model_key, curves in all_curves.items():
        ax.plot(curves["round"], curves["accuracy"],
                marker="o", label=model_key,
                color=colors.get(model_key))

    ax.set_xlabel("HITL Round")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("HITL Learning Curves — All Models")
    ax.set_xticks(range(0, HITL_ROUNDS + 1, 2))
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Combined learning curve saved → %s", out_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    set_seed(RANDOM_SEED)

    train_df = load_split(DATA_DIR / "train_full_with_rationales.csv")
    val_df   = load_split(DATA_DIR / "validation_full_with_rationales.csv")
    test_df  = load_split(DATA_DIR / "test_full_with_rationales.csv")

    all_curves = {}
    for model_key in HITL_MODELS:
        model_name = TRANSFORMER_MODELS[model_key]
        curves     = run_hitl(model_key, model_name,
                               train_df, val_df, test_df)
        all_curves[model_key] = curves

        plot_learning_curve(
            curves, model_key,
            RESULTS_DIR / "figures" / f"learning_curve_{model_key}.png"
        )

    plot_all_curves(all_curves, RESULTS_DIR / "figures" / "learning_curves_all.png")
    save_results(all_curves, RESULTS_DIR / "hitl_results.json")

    logger.info("HITL experiment complete.")


if __name__ == "__main__":
    main()