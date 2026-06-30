"""
utils.py — Shared utilities: data loading, input construction, and metrics.
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, classification_report,
                              f1_score, roc_auc_score)
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import matthews_corrcoef

from config import ID2LABEL, LABEL2ID

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

#------------------------------------------------------------------------
def save_results(results: dict, path: Path) -> None:
    """Serialise results dict to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved → %s", path)



# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Random seed set to %d", seed)


# ── Data I/O ──────────────────────────────────────────────────────────────────

def load_split(path: Path) -> pd.DataFrame:
    import csv
    import io

    rows = []
    with open(path, 'r', encoding='utf-8', errors='replace', newline='') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
            header = [h.strip() for h in header]
        except StopIteration:
            logger.error("Empty file: %s", path)
            return pd.DataFrame()

        n_cols = len(header)
        for row in reader:
            if len(row) == n_cols:
                # Perfect row — use as-is
                rows.append(row)
            elif len(row) > n_cols:
                # Extra commas in last field — rejoin overflow back into it
                fixed = row[:n_cols - 1] + [','.join(row[n_cols - 1:])]
                rows.append(fixed)
            # rows with fewer columns than header are truly malformed — skip

    df = pd.DataFrame(rows, columns=header)

    if "label" in df.columns:
        df["label"] = df["label"].astype(str).str.strip().str.lower()
        valid_labels = set(LABEL2ID.keys())
        before = len(df)
        df = df[df["label"].isin(valid_labels)].copy()
        dropped = before - len(df)
        if dropped > 0:
            logger.warning("Dropped %d rows with invalid labels.", dropped)
        df["label"] = df["label"].map(LABEL2ID).astype(int)

    logger.info("Loaded %d records from %s", len(df), path.name)
    return df

# ── Input construction ────────────────────────────────────────────────────────

def build_input_text(row: pd.Series,
                     include_rationale: bool = True) -> str:
    parts = [
        str(row.get("scenario", "")).strip(),
        str(row.get("judgment", "")).strip(),
    ]
    if include_rationale:
        rationale = row.get("generated_rationale", "")
        if pd.notna(rationale) and str(rationale).strip():
            parts.append(str(rationale).strip())
    return " [SEP] ".join(p for p in parts if p)


def prepare_texts(df: pd.DataFrame,
                  include_rationale: bool = True) -> List[str]:
    """Apply build_input_text across a DataFrame."""
    return [build_input_text(row, include_rationale)
            for _, row in df.iterrows()]


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    y_prob: np.ndarray,
                    verbose: bool = False) -> Dict[str, float]:
    """
    Compute accuracy, macro F1, and macro-OvR AUC.

    Parameters
    ----------
    y_true : 1-D array of integer labels
    y_pred : 1-D array of predicted integer labels
    y_prob : 2-D array of class probabilities, shape (n_samples, n_classes)
    verbose : if True, print the full classification report
    """
    lb      = LabelBinarizer().fit(range(len(LABEL2ID)))
    y_bin   = lb.transform(y_true)

    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro",
                                         zero_division=0)), 4),
        "auc":      round(float(roc_auc_score(y_bin, y_prob,
                                              multi_class="ovr",
                                              average="macro")), 4),
        "mcc":      round(float(matthews_corrcoef(y_true, y_pred)), 4),
    }

    if verbose:
        target_names = [ID2LABEL[i] for i in range(len(LABEL2ID))]
        print(classification_report(y_true, y_pred,
                                    target_names=target_names,
                                    zero_division=0))
    return metrics


# ── Transformer dataset helper ────────────────────────────────────────────────

class ClinicalDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for tokenised clinical reasoning inputs.

    Parameters
    ----------
    encodings : BatchEncoding from a HuggingFace tokeniser
    labels    : list or array of integer labels
    """

    def __init__(self, encodings, labels: List[int]) -> None:
        self.encodings = encodings
        self.labels    = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: v[idx].clone().detach() if isinstance(v[idx], torch.Tensor)
                else torch.tensor(v[idx])
                for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def make_hf_compute_metrics(id2label: Optional[Dict] = None):
    """
    Return a compute_metrics function compatible with HuggingFace Trainer.
    Trainer calls this with an EvalPrediction object.
    """
    from transformers import EvalPrediction

    def _compute(eval_pred):
        logits, labels = eval_pred
        # T5 returns a tuple (logits, past_key_values) — take first element
        if isinstance(logits, tuple):
            logits = logits[0]
        preds = np.argmax(logits, axis=-1)
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        return compute_metrics(labels, preds, probs)
    return _compute
