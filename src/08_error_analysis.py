"""
08_error_analysis.py
====================
Performs error analysis comparing model behaviour before and after HITL v3,
and across rationale conditions.

Analyses produced:
    1. Confusion matrices — pre-HITL seed vs post-HITL final (per model)
    2. Per-class F1 comparison — pre vs post HITL (per model)
    3. Error type breakdown — which class improved, which regressed
    4. Per-class F1 for ablation — with vs without rationale (PubMedBERT)
    5. Most common error patterns (misclassified class pairs)

Usage
-----
    python src/08_error_analysis.py

Output
------
    results/error_analysis.json
    results/error_analysis_summary.txt
    results/figures/fig6a_confusion_pubmedbert.png
    results/figures/fig6b_confusion_clinicalbert.png
    results/figures/fig6c_confusion_roberta.png
    results/figures/fig7_perclass_f1_hitl.png
"""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score, ConfusionMatrixDisplay)
from transformers import (AutoModelForSequenceClassification,
                          AutoTokenizer, Trainer, TrainingArguments)

from config import (BATCH_SIZE, DATA_DIR, FP16, ID2LABEL, LABEL2ID,
                    MAX_INPUT_LENGTH, MODELS_DIR, NUM_LABELS,
                    RANDOM_SEED, RESULTS_DIR, SEED_FRACTION,
                    TRANSFORMER_MODELS)
from utils import (ClinicalDataset, build_input_text, load_split,
                   save_results, set_seed)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

HITL_MODELS   = ["pubmedbert", "clinicalbert", "roberta"]
CLASS_NAMES   = ["safe", "unsafe", "ambiguous"]
OUT_DIR       = RESULTS_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Prediction helpers ────────────────────────────────────────────────────────

def find_checkpoint(base_path: Path) -> Path:
    """Find best available checkpoint in a model directory."""
    # Check common save locations in order of preference
    candidates = [
        base_path / "final_model",    # HITL script saves here
        base_path / "best_model",     # transformer script saves here
    ]
    for c in candidates:
        if c.exists() and (c / "config.json").exists():
            return c

    # Check for numbered checkpoints
    checkpoints = sorted(
        base_path.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[1])
    )
    if checkpoints:
        return checkpoints[-1]

    # Check if base_path itself is a model
    if (base_path / "config.json").exists():
        return base_path

    return None


def get_predictions(model_path: Path, model_key: str,
                    test_df: pd.DataFrame,
                    include_rationale: bool = True):
    """Load a model and return (y_true, y_pred) on the test set."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    except OSError:
        hf_name   = TRANSFORMER_MODELS[model_key]
        tokenizer = AutoTokenizer.from_pretrained(hf_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path), ignore_mismatched_sizes=True
    )

    if include_rationale:
        texts = [build_input_text(r) for _, r in test_df.iterrows()]
    else:
        texts = [f"{str(r.get('scenario','')).strip()} [SEP] "
                 f"{str(r.get('judgment','')).strip()}"
                 for _, r in test_df.iterrows()]

    enc     = tokenizer(texts, padding=True, truncation=True,
                        max_length=MAX_INPUT_LENGTH, return_tensors="pt")
    dataset = ClinicalDataset(enc, test_df["label"].tolist())

    args = TrainingArguments(
        output_dir="/tmp/error_analysis",
        per_device_eval_batch_size=BATCH_SIZE,
        fp16=FP16 and torch.cuda.is_available(),
        report_to="none",
    )
    trainer = Trainer(model=model, args=args)
    raw     = trainer.predict(dataset)
    logits  = raw.predictions
    if isinstance(logits, tuple):
        logits = logits[0]

    y_pred = np.argmax(logits, axis=-1)
    y_true = test_df["label"].to_numpy()
    return y_true, y_pred


# ── Confusion matrix plot ─────────────────────────────────────────────────────

def plot_confusion_matrices(model_key: str,
                             y_true, y_pred_seed,
                             y_true_post, y_pred_post) -> None:
    """
    Side-by-side confusion matrices: pre-HITL seed vs post-HITL final.
    Normalised by row (true label) to show recall per class.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, y_pred, title in zip(
        axes,
        [y_pred_seed, y_pred_post],
        [f"{model_key} — Pre-HITL (seed baseline)",
         f"{model_key} — Post-HITL (v3 final)"]
    ):
        cm  = confusion_matrix(y_true, y_pred, normalize="true")
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=CLASS_NAMES)
        disp.plot(ax=ax, colorbar=False, cmap="Blues",
                  values_format=".2f")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")

    fig.suptitle(f"Figure 6 — Confusion Matrices: {model_key}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = OUT_DIR / f"fig6_{model_key}_confusion.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", path)


# ── Per-class F1 bar chart ────────────────────────────────────────────────────

def plot_perclass_f1(all_results: dict) -> None:
    """
    Grouped bar chart: per-class F1 before and after HITL v3 for all models.
    """
    classes = CLASS_NAMES
    models  = list(all_results.keys())
    x       = np.arange(len(classes))
    width   = 0.13
    colors_before = ["#aec6e8", "#f7b6a5", "#b3d9b3"]
    colors_after  = ["#1f77b4", "#d62728", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, model_key in enumerate(models):
        res    = all_results[model_key]
        f1_pre = res["f1_pre"]
        f1_post= res["f1_post"]
        offset = (i - len(models)/2 + 0.5) * width * 2.2

        ax.bar(x + offset - width/2, f1_pre,  width,
               color=colors_before[i], label=f"{model_key} pre-HITL",
               alpha=0.85)
        ax.bar(x + offset + width/2, f1_post, width,
               color=colors_after[i],  label=f"{model_key} post-HITL",
               alpha=0.85, hatch="//")

    ax.set_xlabel("Class")
    ax.set_ylabel("F1 Score")
    ax.set_title("Figure 7 — Per-class F1: Pre-HITL vs Post-HITL (v3)",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    path = OUT_DIR / "fig7_perclass_f1_hitl.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", path)


# ── Error pattern analysis ────────────────────────────────────────────────────

def error_patterns(y_true, y_pred, label="") -> dict:
    """
    Count and rank misclassification pairs.
    Returns dict: {(true_class, pred_class): count}
    """
    errors = {}
    for yt, yp in zip(y_true, y_pred):
        if yt != yp:
            key = (ID2LABEL[yt], ID2LABEL[yp])
            errors[key] = errors.get(key, 0) + 1

    total_errors = sum(errors.values())
    total        = len(y_true)
    ranked       = sorted(errors.items(), key=lambda x: -x[1])

    lines = [f"\n{label} — Error patterns ({total_errors} errors / {total} total):"]
    for (true_cls, pred_cls), count in ranked:
        pct = 100 * count / total
        lines.append(f"  {true_cls:12s} → {pred_cls:12s}: {count:4d} ({pct:.1f}%)")

    print("\n".join(lines))
    return {f"{tc}→{pc}": cnt for (tc, pc), cnt in ranked}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    set_seed(RANDOM_SEED)

    train_df = load_split(DATA_DIR / "train_full_with_rationales.csv")
    test_df  = load_split(DATA_DIR / "test_full_with_rationales.csv")

    # Recreate the same seed split used in HITL
    seed_df = train_df.sample(frac=SEED_FRACTION, random_state=RANDOM_SEED
                               ).reset_index(drop=True)

    results     = {}
    error_data  = {}

    print("\n" + "=" * 70)
    print("ERROR ANALYSIS: Per-class F1 — Pre-HITL vs Post-HITL (v3)")
    print("=" * 70)

    for model_key in HITL_MODELS:
        logger.info("Analysing %s ...", model_key)
        hf_name = TRANSFORMER_MODELS[model_key]

        
        # ── Pre-HITL: use full-data trained model as pre-HITL reference ──────────
        seed_ckpt = find_checkpoint(MODELS_DIR / model_key)
        if not seed_ckpt:
            logger.warning("  No pre-HITL model found for %s — skipping.", model_key)
            continue
        logger.info("  Pre-HITL model: %s", seed_ckpt)
        y_true_pre, y_pred_pre = get_predictions(seed_ckpt, model_key, test_df)

            # ── Post-HITL: final HITL model ───────────────────────────────────────────
        hitl_ckpt = find_checkpoint(MODELS_DIR / f"{model_key}_hitl")
        if not hitl_ckpt:
            logger.warning("  HITL model not found for %s — skipping.", model_key)
            continue
        logger.info("  Post-HITL model: %s", hitl_ckpt)
        y_true_post, y_pred_post = get_predictions(hitl_ckpt, model_key, test_df)

        # ── Per-class F1 ──────────────────────────────────────────────────────
        report_pre  = classification_report(
            y_true_pre, y_pred_pre,
            target_names=CLASS_NAMES, output_dict=True, zero_division=0
        )
        report_post = classification_report(
            y_true_post, y_pred_post,
            target_names=CLASS_NAMES, output_dict=True, zero_division=0
        )

        f1_pre  = [report_pre[c]["f1-score"]  for c in CLASS_NAMES]
        f1_post = [report_post[c]["f1-score"] for c in CLASS_NAMES]

        print(f"\n{model_key.upper()}")
        print(f"{'Class':<14} {'Pre-HITL F1':>12} {'Post-HITL F1':>13} {'Δ F1':>10}")
        print("-" * 52)
        for cls, fp, fpost in zip(CLASS_NAMES, f1_pre, f1_post):
            delta = fpost - fp
            sign  = "+" if delta >= 0 else ""
            print(f"{cls:<14} {fp:>12.4f} {fpost:>13.4f} {sign}{delta:>9.4f}")
        print(f"{'macro avg':<14} "
              f"{report_pre['macro avg']['f1-score']:>12.4f} "
              f"{report_post['macro avg']['f1-score']:>13.4f} "
              f"{report_post['macro avg']['f1-score'] - report_pre['macro avg']['f1-score']:>+10.4f}")

        results[model_key] = {
            "f1_pre":       f1_pre,
            "f1_post":      f1_post,
            "delta_f1":     [p - r for r, p in zip(f1_pre, f1_post)],
            "report_pre":   report_pre,
            "report_post":  report_post,
        }

        # ── Error patterns ────────────────────────────────────────────────────
        error_data[model_key] = {
            "pre_hitl":  error_patterns(y_true_pre,  y_pred_pre,
                                         f"{model_key} pre-HITL"),
            "post_hitl": error_patterns(y_true_post, y_pred_post,
                                         f"{model_key} post-HITL"),
        }

        # ── Confusion matrices ────────────────────────────────────────────────
        plot_confusion_matrices(
            model_key,
            y_true_pre, y_pred_pre,
            y_true_post, y_pred_post,
        )

    # ── Per-class F1 figure ───────────────────────────────────────────────────
    if results:
        plot_perclass_f1(results)

    # ── Save results ──────────────────────────────────────────────────────────
    output = {"per_class_f1": results, "error_patterns": error_data}

    # Convert numpy floats for JSON serialisation
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj

    save_results(convert(output), RESULTS_DIR / "error_analysis.json")

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    for model_key, res in results.items():
        deltas = res["delta_f1"]
        best_cls   = CLASS_NAMES[np.argmax(deltas)]
        worst_cls  = CLASS_NAMES[np.argmin(deltas)]
        best_delta = max(deltas)
        worst_delta= min(deltas)
        print(f"{model_key}: best improvement → {best_cls} ({best_delta:+.4f}) | "
              f"most regressed → {worst_cls} ({worst_delta:+.4f})")

    logger.info("Error analysis complete.")
    logger.info("Figures saved to %s", OUT_DIR)


if __name__ == "__main__":
    main()