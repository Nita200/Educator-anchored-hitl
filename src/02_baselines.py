"""
02_baselines.py
---------------
Classical ML baseline evaluation (Logistic Regression, Random Forest,
XGBoost, SVM) using TF-IDF features on the clinical reasoning
classification task. Establishes pre-HITL performance benchmarks
across accuracy, macro F1, AUC, and MCC (computed via utils.compute_metrics).
Bootstrap confidence intervals for these models are computed separately
in 06_bootstrap_ci.py.

Usage:
    python src/02_baselines.py

Outputs:
    results/baseline_results.json
    results/figures/baseline_comparison.png
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from config import (BASELINE_MODELS, DATA_DIR, RESULTS_DIR,
                    TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, RANDOM_SEED)
from utils import (build_input_text, compute_metrics,
                   load_split, save_results, set_seed)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


#  Feature extraction 

def build_tfidf_features(
    train_texts, val_texts, test_texts
):
    """
    Fit a TF-IDF vectoriser on the training set and transform all splits.
    """
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_val   = vectorizer.transform(val_texts)
    X_test  = vectorizer.transform(test_texts)
    logger.info("TF-IDF vocabulary size: %d", len(vectorizer.vocabulary_))
    return X_train, X_val, X_test, vectorizer


# Model factory 

def build_model(name: str):
   
    params = BASELINE_MODELS[name]
    if name == "logistic_regression":
        return LogisticRegression(**params)
    elif name == "random_forest":
        return RandomForestClassifier(**params)
    elif name == "xgboost":
        return XGBClassifier(**params)
    elif name == "svm":
        name = LinearSVC(**params)
        return CalibratedClassifierCV(name)
    else:
        raise ValueError(f"Unknown baseline model: {name}")


#  Training and evaluation 

def train_and_evaluate(
    model_name: str,
    X_train, y_train,
    X_test,  y_test,
) -> dict:
    """
    Train a single baseline model and evaluate it on the held-out test set.

    Returns a dictionnary with accuracy, macro_f1, mcc,and auc.
    """
    logger.info("Training %s …", model_name)
    clf = build_model(model_name)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    metrics = compute_metrics(
        np.array(y_test), np.array(y_pred), np.array(y_prob),
        verbose=True
    )
    logger.info("%s → accuracy: %.4f | macro_F1: %.4f | AUC: %.4f | MCC: %.4f",
                model_name, metrics["accuracy"],
                metrics["macro_f1"], metrics["auc"])
    return metrics


#  Visualisation 

def plot_baseline_comparison(results: dict, out_path: Path) -> None:
    """
    Bar chart comparing accuracy, macro F1, and AUC across all baselines.
    """
    models  = list(results.keys())
    metrics = ["accuracy", "macro_f1", "auc"]
    labels  = ["Accuracy", "Macro F1", "AUC"]
    x       = np.arange(len(models))
    width   = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (metric, label) in enumerate(zip(metrics, labels)):
        vals = [results[m][metric] for m in models]
        ax.bar(x + i * width, vals, width, label=label)

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Baseline Model Comparison (No HITL)")
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.replace("_", "\n") for m in models])
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Baseline comparison chart saved → %s", out_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    set_seed(RANDOM_SEED)

    # Load data
    train_df = load_split(DATA_DIR / "train_full_with_rationales.csv")
    test_df  = load_split(DATA_DIR / "test_full_with_rationales.csv")

    train_texts = [build_input_text(row) for _, row in train_df.iterrows()]
    test_texts  = [build_input_text(row) for _, row in test_df.iterrows()]
    y_train     = train_df["label"].tolist()
    y_test      = test_df["label"].tolist()

    # TF-IDF features
    X_train, _, X_test, _ = build_tfidf_features(
        train_texts, train_texts, test_texts
    )

    # Train and evaluate each baseline
    all_results = {}
    for model_name in BASELINE_MODELS:
        all_results[model_name] = train_and_evaluate(
            model_name, X_train, y_train, X_test, y_test
        )

    # Save results
    save_results(all_results, RESULTS_DIR / "baseline_results.json")
    plot_baseline_comparison(
        all_results,
        RESULTS_DIR / "figures" / "baseline_comparison.png"
    )

    logger.info("Baseline evaluation complete.")


if __name__ == "__main__":
    main()
