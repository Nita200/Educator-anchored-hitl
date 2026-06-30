"""
09_mediqa_transfer.py
=====================
Transfer test: evaluates MedNLI-trained models (no retraining) on MEDIQA NLI —
a different clinical NLI benchmark — to test generalisation beyond MedNLI.

Models evaluated (no-rationale ablation versions, since MEDIQA NLI has no
rationale field — this keeps the input format consistent):
    - PubMedBERT
    - ClinicalBERT
    - RoBERTa

Usage
-----
    python src/09_mediqa_transfer.py

Output
------
    results/mediqa_transfer_results.json
    results/mediqa_transfer_summary.txt
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import (AutoModelForSequenceClassification,
                          AutoTokenizer, Trainer, TrainingArguments)

from config import (BATCH_SIZE, FP16, LABEL2ID, MAX_INPUT_LENGTH,
                    MODELS_DIR, RANDOM_SEED, RESULTS_DIR, TRANSFORMER_MODELS)
from utils import ClinicalDataset, compute_metrics, save_results, set_seed

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

HITL_MODELS = ["pubmedbert", "clinicalbert", "roberta"]

# MEDIQA NLI uses the same three-way NLI label scheme as MedNLI
MEDIQA_LABEL_MAP = {
    "entailment":    "safe",
    "contradiction": "unsafe",
    "neutral":       "ambiguous",
    # Some MEDIQA versions use numeric labels — map defensively
    0: "safe",
    1: "unsafe",
    2: "ambiguous",
}


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_mediqa_nli() -> pd.DataFrame:
    """
    Load MEDIQA NLI test set. Tries multiple loading strategies since
    bigbio-hosted datasets sometimes use deprecated loading scripts.
    """
    logger.info("Loading MEDIQA NLI ...")

    attempts = [
        # Attempt 1: bigbio config without trust_remote_code (newer datasets lib)
        lambda: load_dataset("bigbio/mediqa_nli", name="mediqa_nli_bigbio_te"),
        # Attempt 2: bigbio config with trust_remote_code (older datasets lib)
        lambda: load_dataset("bigbio/mediqa_nli", name="mediqa_nli_bigbio_te",
                             trust_remote_code=True),
        # Attempt 3: source config
        lambda: load_dataset("bigbio/mediqa_nli", name="mediqa_nli_source"),
    ]

    dataset = None
    for i, attempt in enumerate(attempts, 1):
        try:
            dataset = attempt()
            logger.info("Loaded successfully (attempt %d).", i)
            break
        except Exception as exc:
            logger.warning("Attempt %d failed: %s", i, exc)

    if dataset is None:
        logger.error(
            "All automated loading attempts failed.\n"
            "MANUAL FALLBACK: Download MEDIQA NLI manually from\n"
            "https://huggingface.co/datasets/bigbio/mediqa_nli\n"
            "and place the test split as a CSV at data/mediqa_nli_test.csv\n"
            "with columns: premise, hypothesis, label"
        )
        manual_path = Path("data/mediqa_nli_test.csv")
        if manual_path.exists():
            logger.info("Found manual fallback file — loading from CSV.")
            return pd.read_csv(manual_path)
        raise RuntimeError("Could not load MEDIQA NLI. See log for manual fallback.")

    # Use test split if available, else the only available split
    split_name = "test" if "test" in dataset else list(dataset.keys())[0]
    logger.info("Using split: %s", split_name)

    records = []
    for row in dataset[split_name]:
        # Column names vary across bigbio configs — handle defensively
        premise    = row.get("premise") or row.get("sentence1") or row.get("text_1", "")
        hypothesis = row.get("hypothesis") or row.get("sentence2") or row.get("text_2", "")
        label_raw  = row.get("label", row.get("gold_label"))

        records.append({
            "scenario": premise,
            "judgment": hypothesis,
            "label_raw": label_raw,
        })

    df = pd.DataFrame(records)
    logger.info("Loaded %d MEDIQA NLI records.", len(df))
    return df


def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map MEDIQA NLI labels to the safe/unsafe/ambiguous scheme."""
    df = df.copy()
    df["label_str"] = df["label_raw"].map(MEDIQA_LABEL_MAP)

    missing = df["label_str"].isna().sum()
    if missing > 0:
        logger.warning("%d records had unmapped labels (unique raw values: %s) "
                       "— dropping.", missing, df["label_raw"].unique())
        df = df.dropna(subset=["label_str"])

    df["label"] = df["label_str"].map(LABEL2ID).astype(int)
    logger.info("MEDIQA label distribution: %s",
                df["label_str"].value_counts().to_dict())
    return df


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_on_mediqa(model_key: str, mediqa_df: pd.DataFrame) -> dict:
    """
    Load the no-rationale ablation model for model_key and evaluate on
    MEDIQA NLI without any retraining (pure transfer test).
    """
    model_path = MODELS_DIR / f"{model_key}_ablation_no_rationale"

    # Find best checkpoint
    candidates = [model_path / "best_model"]
    candidates += sorted(model_path.glob("checkpoint-*"),
                         key=lambda p: int(p.name.split("-")[1]), reverse=True)
    ckpt = next((c for c in candidates
                if c.exists() and (c / "config.json").exists()), None)

    if not ckpt:
        logger.warning("No ablation model found for %s — skipping.", model_key)
        return None

    logger.info("Evaluating %s on MEDIQA NLI (checkpoint: %s) ...",
               model_key, ckpt.name)

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODELS[model_key])

    model = AutoModelForSequenceClassification.from_pretrained(
        str(ckpt), ignore_mismatched_sizes=True
    )

    texts = [f"{str(r['scenario']).strip()} [SEP] {str(r['judgment']).strip()}"
             for _, r in mediqa_df.iterrows()]
    enc = tokenizer(texts, padding=True, truncation=True,
                    max_length=MAX_INPUT_LENGTH, return_tensors="pt")
    dataset = ClinicalDataset(enc, mediqa_df["label"].tolist())

    args = TrainingArguments(
        output_dir                 = "/tmp/mediqa_eval",
        per_device_eval_batch_size = BATCH_SIZE,
        fp16                       = FP16 and torch.cuda.is_available(),
        report_to                  = "none",
    )
    trainer = Trainer(model=model, args=args)
    raw     = trainer.predict(dataset)
    logits  = raw.predictions
    if isinstance(logits, tuple):
        logits = logits[0]

    y_pred = np.argmax(logits, axis=-1)
    y_prob = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    y_true = mediqa_df["label"].to_numpy()

    metrics = compute_metrics(y_true, y_pred, y_prob, verbose=True)
    logger.info("%s on MEDIQA → accuracy: %.4f | F1: %.4f | AUC: %.4f | MCC: %.4f",
               model_key, metrics["accuracy"], metrics["macro_f1"],
               metrics["auc"], metrics["mcc"])
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    set_seed(RANDOM_SEED)

    mediqa_df = load_mediqa_nli()
    mediqa_df = map_labels(mediqa_df)

    results = {}
    for model_key in HITL_MODELS:
        metrics = evaluate_on_mediqa(model_key, mediqa_df)
        if metrics:
            results[model_key] = metrics
            save_results(results, RESULTS_DIR / "mediqa_transfer_results.json")

    # ── Summary ────────────────────────────────────────────────────────────────
    lines = ["=" * 70, "MEDIQA NLI TRANSFER TEST RESULTS", "=" * 70,
             f"{'Model':<15} {'Accuracy':>10} {'F1':>10} {'AUC':>10} {'MCC':>10}",
             "-" * 70]
    for model_key, m in results.items():
        lines.append(f"{model_key:<15} {m['accuracy']:>10.4f} {m['macro_f1']:>10.4f} "
                     f"{m['auc']:>10.4f} {m['mcc']:>10.4f}")
    lines.append("=" * 70)
    summary = "\n".join(lines)
    print("\n" + summary)
    (RESULTS_DIR / "mediqa_transfer_summary.txt").write_text(summary)

    logger.info("MEDIQA transfer test complete.")


if __name__ == "__main__":
    main()