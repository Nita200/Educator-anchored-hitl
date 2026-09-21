"""
01_data_preparation.py
----------------------
Loads MedNLI (train/validation/test splits), maps NLI labels to clinical
reasoning safety categories (safe, unsafe, ambiguous), and generates
BioMistral-7B clinical rationales using a label-independent prompt to
prevent voice-label leakage. Records for which generation produces no
output or contains placeholder text are dropped, yielding a clean dataset
with consistent three-part inputs for downstream training and ablation.

Usage:
    python src/01_data_preparation.py

Outputs:
    data/train_full_with_rationales.csv        (~10,364 records)
    data/validation_full_with_rationales.csv   (~1,277 records)
    data/test_full_with_rationales.csv         (~1,301 records)
    data/train_full.csv
    data/validation_full.csv
    data/test_full.csv
"""

import csv
import logging
import time
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (RATIONALE_MODEL, DATA_DIR, LABEL_MAP,
                    MAX_RATIONALE_LEN, RANDOM_SEED)
from utils import set_seed

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINT_EVERY = 500
RATIONALE_PROMPT = (
    "Clinical scenario: {sentence1} "
    "Student judgment: {sentence2} "
    "Clinical explanation:"
)


# 1. Load full MedNLI

def load_mednli() -> dict:
    logger.info("Loading full MedNLI dataset ...")
    dataset = load_dataset("presencesw/mednli")
    splits = {}
    for split_name in ["train", "validation", "test"]:
        records = []
        for row in dataset[split_name]:
            records.append({
                "scenario": row["sentence1"],
                "judgment": row["sentence2"],
                "gold_label": row["gold_label"],
            })
        splits[split_name] = pd.DataFrame(records)
        logger.info("Loaded %s: %d records", split_name, len(splits[split_name]))
    logger.info("Total records: %d", sum(len(v) for v in splits.values()))
    return splits


# 2. Map labels

def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    entailment    → safe      (0)
    contradiction → unsafe    (1)
    neutral       → ambiguous (2)
    """
    df = df.copy()
    df["label"] = df["gold_label"].map(LABEL_MAP)
    missing = df["label"].isna().sum()
    if missing > 0:
        logger.warning("Dropping %d records with unrecognised labels.", missing)
        df = df.dropna(subset=["label"])
    logger.info("Label distribution: %s", df["label"].value_counts().to_dict())
    return df


# 3. Rationale generator

class RationaleGenerator:
    """Generates label-independent clinical rationales using BioMistral."""

    def __init__(self, model_name: str = RATIONALE_MODEL) -> None:
        logger.info("Loading BioMistral: %s ...", model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=self.dtype, device_map="auto"
        )
        self.model.eval()
        logger.info("BioMistral loaded on %s.", self.device)

    @torch.no_grad()
    def generate(self, scenario: str, judgment: str) -> str:
        prompt = RATIONALE_PROMPT.format(
            sentence1=scenario, sentence2=judgment
        )
        inputs = self.tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=1024
        ).to(self.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=MAX_RATIONALE_LEN,
            do_sample=False,
            repetition_penalty=1.3,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(
            new_tokens, skip_special_tokens=True
        ).strip()


# 4. Generate rationales with checkpointing

def generate_with_checkpoint(
    df: pd.DataFrame,
    generator: RationaleGenerator,
    checkpoint_path: Path,
    split_name: str,
) -> pd.DataFrame:
    df = df.copy()

    if checkpoint_path.exists():
        logger.info("Checkpoint found for %s — resuming ...", split_name)
        checkpoint_df = pd.read_csv(checkpoint_path)
        done_count = len(checkpoint_df)
        logger.info("Already processed: %d / %d", done_count, len(df))
        if done_count >= len(df):
            logger.info("All records already processed for %s.", split_name)
            return checkpoint_df
        completed_rows = checkpoint_df.to_dict("records")
        remaining_df = df.iloc[done_count:].reset_index(drop=True)
    else:
        completed_rows = []
        remaining_df = df
        done_count = 0

    total = len(df)
    start_time = time.time()

    for i, (_, row) in enumerate(remaining_df.iterrows()):
        try:
            rationale = generator.generate(row["scenario"], row["judgment"])
        except Exception as exc:
            logger.warning("Generation failed at index %d: %s", i, exc)
            rationale = ""

        completed_rows.append({
            "scenario":            row["scenario"],
            "judgment":            row["judgment"],
            "label":               row["label"],
            "generated_rationale": rationale,
        })

        current = done_count + i + 1

        if current % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (total - current) / rate if rate > 0 else 0
            logger.info(
                "  %s: %d / %d (%.1f rec/s — ~%.0f min remaining)",
                split_name, current, total, rate, remaining / 60
            )

        if current % CHECKPOINT_EVERY == 0:
            ckpt_df = pd.DataFrame(completed_rows)
            ckpt_df.to_csv(checkpoint_path, index=False, quoting=csv.QUOTE_ALL)
            logger.info("  Checkpoint saved: %d records", current)

    result_df = pd.DataFrame(completed_rows)

    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Checkpoint removed for %s.", split_name)

    return result_df


# 5. Save splits

def save_split(df: pd.DataFrame, name: str) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{name}_with_rationales.csv"
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL)
    logger.info("Saved %s — %d records → %s", name, len(df), path.name)
    base_path = DATA_DIR / f"{name}.csv"
    df[["scenario", "judgment", "label"]].to_csv(
        base_path, index=False, quoting=csv.QUOTE_ALL
    )
    logger.info("Saved base split → %s", base_path.name)


# 6. Filter to clean rationales

def filter_rationales(split_name: str) -> None:
    path = DATA_DIR / f"{split_name}_full_with_rationales.csv"
    df = pd.read_csv(path)
    before = len(df)
    has_rationale = (
        df["generated_rationale"].notna() &
        (df["generated_rationale"].astype(str).str.strip() != "") &
        (df["generated_rationale"].astype(str) != "nan") &
        (~df["generated_rationale"].astype(str).str.contains(
            r"\[|\]", regex=True, na=False))
    )
    df = df[has_rationale].reset_index(drop=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL)
    df[["scenario", "judgment", "label"]].to_csv(
        DATA_DIR / f"{split_name}_full.csv", index=False, quoting=csv.QUOTE_ALL
    )
    logger.info("%s: %d → %d records (%d dropped)",
                split_name, before, len(df), before - len(df))


# Main

def main() -> None:
    set_seed(RANDOM_SEED)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    splits = load_mednli()

    for name in splits:
        splits[name] = map_labels(splits[name])

    generator = RationaleGenerator()

    for split_name, df in splits.items():
        logger.info("=" * 60)
        logger.info("Generating rationales for %s split (%d records) ...",
                    split_name, len(df))
        logger.info("=" * 60)
        checkpoint_path = DATA_DIR / f".checkpoint_{split_name}.csv"
        result_df = generate_with_checkpoint(
            df, generator, checkpoint_path, split_name
        )
        save_split(result_df, f"{split_name}_full")

    logger.info("Filtering to clean rationales ...")
    for split_name in splits:
        filter_rationales(split_name)

    logger.info("Data preparation complete.")
    logger.info("Final sizes:")
    for split_name in splits:
        path = DATA_DIR / f"{split_name}_full_with_rationales.csv"
        if path.exists():
            n = sum(1 for _ in open(path)) - 1
            logger.info("  %s: %d records", split_name, n)


if __name__ == "__main__":
    main()