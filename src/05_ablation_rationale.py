"""
05_ablation_rationale.py
========================
Ablation study: evaluates whether BioGPT-generated rationales contribute
meaningful signal to model performance.

Condition A — WITHOUT rationale: input = scenario + judgment only
Condition B — WITH rationale:    input = scenario + judgment + rationale (Table 1)

Three models are evaluated: PubMedBERT, ClinicalBERT, RoBERTa.
Results are compared against the existing transformer_results.json (Condition B).

Usage
-----
    python src/05_ablation_rationale.py

Output
------
    results/ablation_results.json
    results/figures/ablation_comparison.png
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import (AutoModelForSequenceClassification,
                          AutoTokenizer, Trainer, TrainingArguments)

from config import (BATCH_SIZE, DATA_DIR, FP16, ID2LABEL, LABEL2ID,
                    LEARNING_RATE, MAX_INPUT_LENGTH, MODELS_DIR,
                    NUM_LABELS, NUM_TRAIN_EPOCHS, RANDOM_SEED,
                    RESULTS_DIR, TRANSFORMER_MODELS,
                    WARMUP_RATIO, WEIGHT_DECAY)
from utils import (ClinicalDataset, compute_metrics, load_split,
                   make_hf_compute_metrics, save_results, set_seed)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

# ── Ablation models — the three HITL candidates ───────────────────────────────
ABLATION_MODELS = ["pubmedbert", "clinicalbert", "roberta"]


# ── Input builder WITHOUT rationale ──────────────────────────────────────────

def build_input_no_rationale(row) -> str:
    """
    Concatenate scenario and judgment ONLY — no rationale.
    This is the ablation condition to test rationale contribution.
    """
    scenario = str(row.get("scenario", "")).strip()
    judgment = str(row.get("judgment", "")).strip()
    return f"{scenario} [SEP] {judgment}"


# ── Tokenisation ───────────────────────────────────────────────────────────────

def tokenise(tokenizer, texts, max_length: int = MAX_INPUT_LENGTH):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


# ── Single model training ──────────────────────────────────────────────────────

def run_ablation_model(
    model_key: str,
    model_name: str,
    train_df, val_df, test_df,
) -> dict:
    """
    Fine-tune one model WITHOUT rationale and evaluate on the held-out test set.
    Uses identical hyperparameters to 03_transformers.py for a fair comparison.
    """
    logger.info("=" * 60)
    logger.info("Ablation (no rationale): %s", model_key)
    logger.info("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels  = NUM_LABELS,
        id2label    = ID2LABEL,
        label2id    = LABEL2ID,
        ignore_mismatched_sizes = True,
    )

    # Build inputs WITHOUT rationale
    train_texts = [build_input_no_rationale(r) for _, r in train_df.iterrows()]
    val_texts   = [build_input_no_rationale(r) for _, r in val_df.iterrows()]
    test_texts  = [build_input_no_rationale(r) for _, r in test_df.iterrows()]

    train_enc = tokenise(tokenizer, train_texts)
    val_enc   = tokenise(tokenizer, val_texts)
    test_enc  = tokenise(tokenizer, test_texts)

    train_dataset = ClinicalDataset(train_enc, train_df["label"].tolist())
    val_dataset   = ClinicalDataset(val_enc,   val_df["label"].tolist())
    test_dataset  = ClinicalDataset(test_enc,  test_df["label"].tolist())

    output_dir = MODELS_DIR / f"{model_key}_ablation_no_rationale"
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir                   = str(output_dir),
        num_train_epochs             = NUM_TRAIN_EPOCHS,
        per_device_train_batch_size  = BATCH_SIZE,
        per_device_eval_batch_size   = BATCH_SIZE,
        learning_rate                = LEARNING_RATE,
        warmup_ratio                 = WARMUP_RATIO,
        weight_decay                 = WEIGHT_DECAY,
        evaluation_strategy          = "epoch",
        save_strategy                = "epoch",
        load_best_model_at_end       = True,
        metric_for_best_model        = "macro_f1",
        greater_is_better            = True,
        fp16                         = FP16 and torch.cuda.is_available(),
        seed                         = RANDOM_SEED,
        logging_steps                = 50,
        report_to                    = "none",
    )

    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_dataset,
        eval_dataset    = val_dataset,
        compute_metrics = make_hf_compute_metrics(),
    )

    trainer.train()

    # Evaluate on held-out test set
    raw_preds = trainer.predict(test_dataset)
    logits    = raw_preds.predictions
    if isinstance(logits, tuple):
        logits = logits[0]

    y_pred  = np.argmax(logits, axis=-1)
    y_prob  = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    y_true  = test_df["label"].to_numpy()

    metrics = compute_metrics(y_true, y_pred, y_prob, verbose=True)
    logger.info(
        "%s (no rationale) → accuracy: %.4f | F1: %.4f | AUC: %.4f | MCC: %.4f",
        model_key, metrics["accuracy"], metrics["macro_f1"],
        metrics["auc"], metrics["mcc"]
    )
    return metrics


# ── Visualisation ──────────────────────────────────────────────────────────────

def plot_ablation(ablation_results: dict, with_rationale: dict,
                  out_path: Path) -> None:
    """
    Grouped bar chart comparing with vs without rationale for each model.
    """
    models  = list(ablation_results.keys())
    metrics = ["accuracy", "macro_f1", "auc", "mcc"]
    labels  = ["Accuracy", "Macro F1", "AUC", "MCC"]
    x       = np.arange(len(models))
    width   = 0.18

    fig, ax = plt.subplots(figsize=(12, 6))
    colors_with    = ["steelblue", "darkorange", "forestgreen", "purple"]
    colors_without = ["lightsteelblue", "moccasin", "lightgreen", "plum"]

    for i, (metric, label) in enumerate(zip(metrics, labels)):
        vals_with    = [with_rationale.get(m, {}).get(metric, 0) for m in models]
        vals_without = [ablation_results[m][metric] for m in models]

        offset = (i - 1.5) * width
        bars_with    = ax.bar(x + offset - width/2, vals_with,
                              width, label=f"{label} (with rationale)",
                              color=colors_with[i], alpha=0.9)
        bars_without = ax.bar(x + offset + width/2, vals_without,
                              width, label=f"{label} (no rationale)",
                              color=colors_without[i], alpha=0.9,
                              hatch="//")

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Ablation Study: With vs Without BioGPT Rationale", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Ablation chart saved → %s", out_path)


def print_comparison_table(ablation: dict, with_rat: dict) -> None:
    """Print a clean comparison table to the console."""
    print("\n" + "=" * 80)
    print(f"{'Model':<15} {'Condition':<20} {'Accuracy':>10} {'F1':>10} "
          f"{'AUC':>10} {'MCC':>10}")
    print("=" * 80)
    for model in ablation:
        wr = with_rat.get(model, {})
        nr = ablation[model]
        print(f"{model:<15} {'with rationale':<20} "
              f"{wr.get('accuracy',0):>10.4f} {wr.get('macro_f1',0):>10.4f} "
              f"{wr.get('auc',0):>10.4f} {wr.get('mcc',0):>10.4f}")
        print(f"{'':<15} {'no rationale':<20} "
              f"{nr['accuracy']:>10.4f} {nr['macro_f1']:>10.4f} "
              f"{nr['auc']:>10.4f} {nr['mcc']:>10.4f}")

        # Delta
        delta_acc = wr.get('accuracy', 0) - nr['accuracy']
        delta_f1  = wr.get('macro_f1', 0) - nr['macro_f1']
        delta_auc = wr.get('auc', 0)      - nr['auc']
        delta_mcc = wr.get('mcc', 0)      - nr['mcc']
        sign = lambda v: f"+{v:.4f}" if v >= 0 else f"{v:.4f}"
        print(f"{'':<15} {'Δ (with − no rat.)':<20} "
              f"{sign(delta_acc):>10} {sign(delta_f1):>10} "
              f"{sign(delta_auc):>10} {sign(delta_mcc):>10}")
        print("-" * 80)
    print("=" * 80 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    set_seed(RANDOM_SEED)

    train_df = load_split(DATA_DIR / "train_full_with_rationales.csv")
    val_df   = load_split(DATA_DIR / "validation_full_with_rationales.csv")
    test_df  = load_split(DATA_DIR / "test_full_with_rationales.csv")

    # Load existing with-rationale results for comparison
    import json
    with_rat_path = RESULTS_DIR / "transformer_results.json"
    with_rationale = {}
    if with_rat_path.exists():
        with open(with_rat_path) as f:
            with_rationale = json.load(f)
        logger.info("Loaded existing transformer results for comparison.")
    else:
        logger.warning("transformer_results.json not found — "
                       "comparison will be incomplete.")

    # Run ablation for each model
    ablation_results = {}
    for model_key in ABLATION_MODELS:
        model_name = TRANSFORMER_MODELS[model_key]
        ablation_results[model_key] = run_ablation_model(
            model_key, model_name, train_df, val_df, test_df
        )
        # Save incrementally
        save_results(ablation_results,
                     RESULTS_DIR / "ablation_results.json")

    # Print comparison
    print_comparison_table(ablation_results, with_rationale)

    # Plot
    plot_ablation(
        ablation_results, with_rationale,
        RESULTS_DIR / "figures" / "ablation_comparison.png"
    )

    logger.info("Ablation study complete.")
    logger.info("Results saved → %s", RESULTS_DIR / "ablation_results.json")


if __name__ == "__main__":
    main()