"""
03_transformers.py
------------------
Fine-tunes five transformer models (T5-small, PubMedBERT, ClinicalBERT,
DistilBERT, RoBERTa) on the clinical reasoning classification task.
Establishes pre-HITL transformer baselines across accuracy, macro F1,
AUC, and MCC for comparison with the HITL-refined models in 04_hitl.py.

Usage:
    python src/03_transformers.py

Outputs:
    results/transformer_results.json
    results/figures/transformer_comparison.png
    models/<model_key>/   (fine-tuned weights and tokeniser)
"""

import logging
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, matthews_corrcoef)

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
from utils import (ClinicalDataset, build_input_text, compute_metrics,
                   load_split, make_hf_compute_metrics,
                   save_results, set_seed)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


#  Tokenisation ########################################

def tokenise_split(tokenizer, texts, max_length: int = MAX_INPUT_LENGTH):
   
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


#  Single model training and evaluation ###############################

def run_transformer(
    model_key: str,
    model_name: str,
    train_df, val_df, test_df,
) -> dict:
    """
     a dict with keys: accuracy, macro_f1, auc, mcc.
    """
    logger.info("=" * 60)
    logger.info("Fine-tuning: %s (%s)", model_key, model_name)
    logger.info("=" * 60)

    # Load tokeniser and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    # Build input texts
    train_texts = [build_input_text(row) for _, row in train_df.iterrows()]
    val_texts   = [build_input_text(row) for _, row in val_df.iterrows()]
    test_texts  = [build_input_text(row) for _, row in test_df.iterrows()]

    # Tokenise
    train_enc = tokenise_split(tokenizer, train_texts)
    val_enc   = tokenise_split(tokenizer, val_texts)
    test_enc  = tokenise_split(tokenizer, test_texts)

    # Build datasets
    train_dataset = ClinicalDataset(train_enc, train_df["label"].tolist())
    val_dataset   = ClinicalDataset(val_enc,   val_df["label"].tolist())
    test_dataset  = ClinicalDataset(test_enc,  test_df["label"].tolist())

    # Output directory for this model
    output_dir = MODELS_DIR / model_key
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training configuration
    training_args = TrainingArguments(
        output_dir              = str(output_dir),
        num_train_epochs        = NUM_TRAIN_EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        learning_rate           = LEARNING_RATE,
        warmup_ratio            = WARMUP_RATIO,
        weight_decay            = WEIGHT_DECAY,
        evaluation_strategy     = "epoch",
        save_strategy           = "epoch",
        load_best_model_at_end  = True,
        metric_for_best_model   = "macro_f1",
        greater_is_better       = True,
        fp16                    = FP16 and torch.cuda.is_available(),
        seed                    = RANDOM_SEED,
        logging_steps           = 50,
        report_to               = "none",
    )

    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_dataset,
        eval_dataset    = val_dataset,
        compute_metrics = make_hf_compute_metrics(),
    )

    trainer.train()

    # Save the best model
    trainer.save_model(str(output_dir / "best_model"))
    tokenizer.save_pretrained(str(output_dir / "best_model"))
    logger.info("Model saved → %s/best_model", output_dir)

    # Evaluate on held-out test set
    raw_preds = trainer.predict(test_dataset)
    logits    = raw_preds.predictions

    # T5 returns a tuple (logits, past_key_values),  extract logits only
    if isinstance(logits, tuple):
        logits = logits[0]

    y_pred = np.argmax(logits, axis=-1)
    y_prob = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    y_true = test_df["label"].to_numpy()

    metrics = compute_metrics(y_true, y_pred, y_prob, verbose=True)
    logger.info(
        "%s TEST → accuracy: %.4f | macro_F1: %.4f | AUC: %.4f",
        model_key, metrics["accuracy"], metrics["macro_f1"], metrics["auc"],
    )
    return metrics


#  Visualization (Bar chart comparing accuracy and AUC for all transformer models)

def plot_transformer_comparison(results: dict, out_path: Path) -> None:
   
    models  = list(results.keys())
    x       = np.arange(len(models))
    width   = 0.3
    acc     = [results[m]["accuracy"] for m in models]
    auc     = [results[m]["auc"]      for m in models]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, acc, width, label="Accuracy", color="steelblue")
    ax.bar(x + width / 2, auc, width, label="AUC",      color="darkorange")

    for i, (a, u) in enumerate(zip(acc, auc)):
        ax.text(i - width / 2, a + 0.005, f"{a:.3f}",
                ha="center", va="bottom", fontsize=8)
        ax.text(i + width / 2, u + 0.005, f"{u:.3f}",
                ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Transformer Baseline Comparison (No HITL)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0, 1.08)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Transformer comparison chart saved → %s", out_path)


# Main ##################################

def main() -> None:
    set_seed(RANDOM_SEED)

    train_df = load_split(DATA_DIR / "train_full_with_rationales.csv")
    val_df   = load_split(DATA_DIR / "validation_full_with_rationales.csv")
    test_df  = load_split(DATA_DIR / "test_full_with_rationales.csv")

    # Load existing results so completed models are not re-run
    import json
    results_path = RESULTS_DIR / "transformer_results.json"
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
        logger.info("Loaded existing results: %s", list(all_results.keys()))
    else:
        all_results = {}

    for model_key, model_name in TRANSFORMER_MODELS.items():
        all_results[model_key] = run_transformer(
            model_key, model_name, train_df, val_df, test_df
        )

    save_results(all_results, RESULTS_DIR / "transformer_results.json")
    plot_transformer_comparison(
        all_results,
        RESULTS_DIR / "figures" / "transformer_comparison.png",
    )

    logger.info("Transformer evaluation complete.")


if __name__ == "__main__":
    main()
    
