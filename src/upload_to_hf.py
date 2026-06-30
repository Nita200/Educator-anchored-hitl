"""
upload_to_hf.py
===============
Uploads your final HITL-trained models to HuggingFace Hub.
Run this AFTER 04_hitl.py has completed successfully.

Prerequisites
-------------
1. Create a free account at https://huggingface.co
2. Create an access token with WRITE permission:
       https://huggingface.co/settings/tokens
3. Install the HuggingFace CLI:
       pip install huggingface_hub
4. Log in from the terminal:
       huggingface-cli login
   Paste your token when prompted.

Usage
-----
    python src/upload_to_hf.py --username your_hf_username

What it does
------------
For each HITL-trained model (T5-small, BioBERT, ClinicalBERT) the script:
    1. Creates a model repository on HuggingFace Hub (if it does not exist)
    2. Uploads the model weights, tokeniser, and config
    3. Writes a model card (README.md) describing the model
    4. Prints the permanent URL for each uploaded model

After running, add the returned URLs to the README.md in this project.
"""

import argparse
import logging
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import HITL_MODELS, MODELS_DIR, TRANSFORMER_MODELS

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


# ── Model card template ───────────────────────────────────────────────────────

MODEL_CARD_TEMPLATE = """---
language:
  - en
license: mit
tags:
  - clinical-nlp
  - nursing-education
  - human-in-the-loop
  - text-classification
  - transformer
datasets:
  - bigbio/mednli
metrics:
  - accuracy
  - f1
  - roc_auc
---

# {model_name} — HITL Clinical Reasoning Classifier

This model is a fine-tuned version of [{base_model}](https://huggingface.co/{base_model})
trained on the [MedNLI](https://physionet.org/content/mednli/1.0.0/) dataset for
clinical reasoning classification in nursing education.

It was refined using a simulated Human-in-the-Loop (HITL) workflow across
{hitl_rounds} rounds of incremental educator-guided corrections, as described in:

> *Hybrid Intelligence for Nursing Education: A Simulation Study of an
> Educator-in-the-Loop Approach Using Transformer and Machine Learning Models
> for Clinical Reasoning Assessment* — [citation pending]

## Labels

| ID | Label     | Description                          |
|----|-----------|--------------------------------------|
| 0  | safe      | Clinically appropriate reasoning     |
| 1  | unsafe    | Clinically unsafe or incorrect       |
| 2  | ambiguous | Requires further clinical evaluation |

## Usage

```python
from transformers import pipeline

clf = pipeline(
    "text-classification",
    model="{repo_id}",
)
result = clf("Patient has chest pain. Student assessment: possible GERD. "
             "[SEP] Rationale: The patient's history is consistent with GERD "
             "given the absence of cardiac risk factors.")
print(result)
# [{'label': 'ambiguous', 'score': 0.81}]
```

## Training

- **Base model**: {base_model}
- **Dataset**: MedNLI (PhysioNet credentialed access required)
- **Split**: 70 / 10 / 20 (train / val / test)
- **HITL rounds**: {hitl_rounds}
- **Corrections per round**: 150
- **Learning rate**: 2e-5
- **Epochs per round**: 2
"""


# ── Upload logic ──────────────────────────────────────────────────────────────

def upload_model(
    model_key: str,
    username:  str,
    api:       HfApi,
    hitl_rounds: int = 20,
) -> str:
    """
    Upload one HITL-trained model to HuggingFace Hub.

    Returns the model URL.
    """
    model_dir  = MODELS_DIR / f"{model_key}_hitl" / "final_model"
    base_model = TRANSFORMER_MODELS[model_key]
    repo_name  = f"hitl-nursing-{model_key}"
    repo_id    = f"{username}/{repo_name}"

    if not model_dir.exists():
        logger.warning(
            "Model directory not found: %s — skipping %s. "
            "Run 04_hitl.py first.", model_dir, model_key
        )
        return ""

    # Create repo (public, so anyone can use the model)
    logger.info("Creating repository: %s", repo_id)
    create_repo(repo_id=repo_id, exist_ok=True, private=False)

    # Write model card
    card_path = model_dir / "README.md"
    card_path.write_text(MODEL_CARD_TEMPLATE.format(
        model_name  = f"{model_key.upper()} HITL Clinical Reasoning",
        base_model  = base_model,
        hitl_rounds = hitl_rounds,
        repo_id     = repo_id,
    ))

    # Upload all files in the model directory
    logger.info("Uploading %s to %s …", model_key, repo_id)
    upload_folder(
        folder_path = str(model_dir),
        repo_id     = repo_id,
        repo_type   = "model",
        commit_message = f"Upload HITL-trained {model_key} ({hitl_rounds} rounds)",
    )

    url = f"https://huggingface.co/{repo_id}"
    logger.info("✓ %s uploaded → %s", model_key, url)
    return url


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload HITL models to HuggingFace Hub."
    )
    parser.add_argument(
        "--username", required=True,
        help="Your HuggingFace username (e.g. jsmith)"
    )
    parser.add_argument(
        "--rounds", type=int, default=20,
        help="Number of HITL rounds used in training (for model card)"
    )
    args = parser.parse_args()

    api  = HfApi()
    urls = {}

    for model_key in HITL_MODELS:
        url = upload_model(model_key, args.username, api, args.rounds)
        if url:
            urls[model_key] = url

    print("\n" + "=" * 60)
    print("Upload complete. Add these URLs to your README.md:")
    print("=" * 60)
    for key, url in urls.items():
        print(f"  {key:15s}  →  {url}")
    print("=" * 60)


if __name__ == "__main__":
    main()
