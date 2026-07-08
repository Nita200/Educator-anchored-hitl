"""
06_bootstrap_ci.py
++++++++++++++++++++++++++++++++++++++++++
Computes 95% bootstrap confidence intervals (n=1,000) for accuracy,
macro F1, AUC, and MCC across baseline ML models, transformer models
(with and without rationale), and HITL v3 final-round results.

Resamples test set predictions with replacement , no retraining required.
The 2.5th and 97.5th percentiles of the bootstrap distribution form each CI.

Usage:
    python src/06_bootstrap_ci.py

Outputs:
    results/bootstrap_ci.json
    results/bootstrap_ci_summary.txt
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, f1_score,
                              matthews_corrcoef, roc_auc_score)
from sklearn.preprocessing import LabelBinarizer
from transformers import (AutoModelForSequenceClassification,
                          AutoTokenizer, Trainer, TrainingArguments)

from config import (BATCH_SIZE, DATA_DIR, FP16, ID2LABEL, LABEL2ID,
                    MAX_INPUT_LENGTH, MODELS_DIR, NUM_LABELS,
                    RANDOM_SEED, RESULTS_DIR, TRANSFORMER_MODELS)
from utils import (ClinicalDataset, build_input_text, load_split,
                   save_results, set_seed)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

N_BOOTSTRAP  = 1000   # number of bootstrap samples
CONFIDENCE   = 0.95   # confidence level
ALPHA        = 1 - CONFIDENCE
HITL_MODELS  = ["pubmedbert", "clinicalbert", "roberta"]


#  Bootstrap engine 

def bootstrap_metrics(y_true: np.ndarray,
                      y_pred: np.ndarray,
                      y_prob: np.ndarray,
                      n_bootstrap: int = N_BOOTSTRAP,
                      seed: int = RANDOM_SEED) -> dict:
    """
    Compute 95% bootstrap confidence intervals for accuracy, macro F1,
    AUC, and MCC.

    Parameters
    ----------
    y_true      : true integer labels
    y_pred      : predicted integer labels
    y_prob      : class probability matrix (n_samples, n_classes)
    n_bootstrap : number of bootstrap iterations
    seed        : random seed for reproducibility

    Returns
    -------
    dict with keys: accuracy, macro_f1, auc, mcc
    Each value is {"mean": float, "ci_low": float, "ci_high": float}
    """
    rng  = np.random.default_rng(seed)
    n    = len(y_true)
    lb   = LabelBinarizer().fit(range(NUM_LABELS))

    acc_boot, f1_boot, auc_boot, mcc_boot = [], [], [], []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)   # sample with replacement
        yt  = y_true[idx]
        yp  = y_pred[idx]
        yb  = y_prob[idx]

        # Skip degenerate samples (only one class present)
        if len(np.unique(yt)) < 2:
            continue

        y_bin = lb.transform(yt)
        try:
            auc = roc_auc_score(y_bin, yb, multi_class="ovr", average="macro")
        except ValueError:
            continue

        acc_boot.append(accuracy_score(yt, yp))
        f1_boot.append(f1_score(yt, yp, average="macro", zero_division=0))
        auc_boot.append(auc)
        mcc_boot.append(matthews_corrcoef(yt, yp))

    def ci(samples):
        arr = np.array(samples)
        return {
            "mean":    round(float(arr.mean()), 4),
            "ci_low":  round(float(np.percentile(arr, 100 * ALPHA / 2)), 4),
            "ci_high": round(float(np.percentile(arr, 100 * (1 - ALPHA / 2))), 4),
        }

    return {
        "accuracy": ci(acc_boot),
        "macro_f1": ci(f1_boot),
        "auc":      ci(auc_boot),
        "mcc":      ci(mcc_boot),
    }


#  Prediction helpers 

def get_predictions_transformer(model_key: str,
                                 model_path: Path,
                                 test_df: pd.DataFrame,
                                 include_rationale: bool = True):
    # If model_path has no config.json, find best checkpoint inside it
    if not (model_path / "config.json").exists():
        checkpoints = sorted(model_path.glob("checkpoint-*"),
                             key=lambda p: int(p.name.split("-")[1]))
        if checkpoints:
            model_path = checkpoints[-1]
            logger.info("Using checkpoint: %s", model_path.name)
        else:
            raise FileNotFoundError(f"No config.json or checkpoints in {model_path}")

    # Load tokenizer fall back to HF if not saved in checkpoint
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    except OSError:
        hf_name   = TRANSFORMER_MODELS[model_key]
        logger.info("No tokenizer in checkpoint — loading from %s", hf_name)
        tokenizer = AutoTokenizer.from_pretrained(hf_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path), ignore_mismatched_sizes=True
    )

    if include_rationale:
        texts = [build_input_text(r) for _, r in test_df.iterrows()]
    else:
        texts = [f"{str(r.get('scenario', '')).strip()} [SEP] "
                 f"{str(r.get('judgment', '')).strip()}"
                 for _, r in test_df.iterrows()]

    enc     = tokenizer(texts, padding=True, truncation=True,
            max_length=MAX_INPUT_LENGTH, return_tensors="pt")
    dataset = ClinicalDataset(enc, test_df["label"].tolist())

    args = TrainingArguments(
        output_dir                  = "/tmp/bootstrap_pred",
        per_device_eval_batch_size  = BATCH_SIZE,
        fp16                        = FP16 and torch.cuda.is_available(),
        report_to                   = "none",
    )
    trainer  = Trainer(model=model, args=args)
    raw      = trainer.predict(dataset)
    logits   = raw.predictions
    if isinstance(logits, tuple):
        logits = logits[0]

    y_pred = np.argmax(logits, axis=-1)
    y_prob = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    y_true = test_df["label"].to_numpy()
    return y_true, y_pred, y_prob


def get_predictions_baseline(model_key: str,
                              train_df: pd.DataFrame,
                              test_df: pd.DataFrame):
    """
    Retrain a baseline ML model and return test predictions.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from xgboost import XGBClassifier
    from config import (TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, BASELINE_MODELS)

    train_texts = [build_input_text(r) for _, r in train_df.iterrows()]
    test_texts  = [build_input_text(r) for _, r in test_df.iterrows()]
    y_train     = train_df["label"].to_numpy()
    y_true      = test_df["label"].to_numpy()

    vec     = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES,
                              ngram_range=TFIDF_NGRAM_RANGE, sublinear_tf=True)
    X_train = vec.fit_transform(train_texts)
    X_test  = vec.transform(test_texts)

    params = BASELINE_MODELS[model_key]
    if model_key == "logistic_regression":
        clf = LogisticRegression(**params)
    elif model_key == "random_forest":
        clf = RandomForestClassifier(**params)
    elif model_key == "xgboost":
        clf = XGBClassifier(**params)
    elif model_key == "svm":
        clf = CalibratedClassifierCV(LinearSVC(**params))

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    return y_true, y_pred, y_prob


# Summary table 

def format_ci(ci_dict: dict, metric: str) -> str:
    """Format a metric with its CI as 'mean [low–high]'."""
    m  = ci_dict[metric]
    return f"{m['mean']:.3f} [{m['ci_low']:.3f}–{m['ci_high']:.3f}]"


def write_summary(all_ci: dict, path: Path) -> None:
    """Write a human-readable summary table."""
    lines = []
    lines.append("=" * 100)
    lines.append("BOOTSTRAP 95% CONFIDENCE INTERVALS (n=1000)")
    lines.append("Format: mean [ci_low–ci_high]")
    lines.append("=" * 100)
    lines.append(f"{'Model / Condition':<35} {'Accuracy':>24} {'Macro F1':>24} "
                 f"{'AUC':>24} {'MCC':>24}")
    lines.append("-" * 100)

    for model_key, conditions in all_ci.items():
        for condition, ci_dict in conditions.items():
            label = f"{model_key} ({condition})"
            lines.append(
                f"{label:<35} "
                f"{format_ci(ci_dict,'accuracy'):>24} "
                f"{format_ci(ci_dict,'macro_f1'):>24} "
                f"{format_ci(ci_dict,'auc'):>24} "
                f"{format_ci(ci_dict,'mcc'):>24}"
            )
        lines.append("-" * 100)

    lines.append("=" * 100)
    text = "\n".join(lines)
    path.write_text(text)
    print("\n" + text)
    logger.info("Summary saved → %s", path)


#  Main ******************

def main() -> None:
    set_seed(RANDOM_SEED)

    train_df = load_split(DATA_DIR / "train_full_with_rationales.csv")
    test_df  = load_split(DATA_DIR / "test_full_with_rationales.csv")

    all_ci = {}

    # 1. Baseline ML models
    logger.info("Computing CIs for baseline ML models ...")
    baseline_keys = ["logistic_regression", "random_forest", "xgboost", "svm"]
    for key in baseline_keys:
        logger.info("  %s", key)
        y_true, y_pred, y_prob = get_predictions_baseline(key, train_df, test_df)
        ci = bootstrap_metrics(y_true, y_pred, y_prob)
        all_ci[key] = {"baseline": ci}
        save_results(all_ci, RESULTS_DIR / "bootstrap_ci.json")

    #  2. Transformer models with rationale 
    logger.info("Computing CIs for transformers (with rationale) ...")
    for model_key in HITL_MODELS:
        model_path = MODELS_DIR / model_key / "best_model"
        if not model_path.exists():
            logger.warning("Model not found: %s — skipping.", model_path)
            continue
        y_true, y_pred, y_prob = get_predictions_transformer(
            model_key, model_path, test_df, include_rationale=True
        )
        ci = bootstrap_metrics(y_true, y_pred, y_prob)
        all_ci[model_key] = {"with_rationale": ci}
        save_results(all_ci, RESULTS_DIR / "bootstrap_ci.json")

    #  3. Transformer models — without rationale (ablation) 
    logger.info("Computing CIs for transformers (no rationale / ablation) ...")
    for model_key in HITL_MODELS:
        model_path = MODELS_DIR / f"{model_key}_ablation_no_rationale"
        # Find best checkpoint or final model
        candidates = [
            model_path / "best_model",
            model_path,
        ]
        found = next((p for p in candidates if p.exists()), None)
        if not found:
            logger.warning("Ablation model not found: %s — skipping.", model_path)
            continue
        y_true, y_pred, y_prob = get_predictions_transformer(
            model_key, found, test_df, include_rationale=False
        )
        ci = bootstrap_metrics(y_true, y_pred, y_prob)
        if model_key not in all_ci:
            all_ci[model_key] = {}
        all_ci[model_key]["no_rationale"] = ci
        save_results(all_ci, RESULTS_DIR / "bootstrap_ci.json")

    # ── 4. HITL C3 : seed baseline (round 0) and final round 
    logger.info("Computing CIs for HITL v3 models ...")
    for model_key in HITL_MODELS:
        # Final HITL model
        hitl_path = MODELS_DIR / f"{model_key}_hitl" / "final_model"
        if not hitl_path.exists():
            logger.warning("HITL model not found: %s — skipping.", hitl_path)
            continue
        y_true, y_pred, y_prob = get_predictions_transformer(
            model_key, hitl_path, test_df, include_rationale=True
        )
        ci = bootstrap_metrics(y_true, y_pred, y_prob)
        hitl_key = f"{model_key}_hitl_final"
        all_ci[hitl_key] = {"hitl_v3_final": ci}
        save_results(all_ci, RESULTS_DIR / "bootstrap_ci.json")

    #  Write summary *************************
    write_summary(all_ci, RESULTS_DIR / "bootstrap_ci_summary.txt")
    logger.info("Bootstrap CI computation complete.")
    logger.info("Results → %s", RESULTS_DIR / "bootstrap_ci.json")


if __name__ == "__main__":
    main()