"""
upload_to_hf.py

Uploads trained HITL v3 model checkpoints to HuggingFace Hub with
auto-generated model cards including the simulation study disclaimer
and correct v3 hyperparameters.

Models are already publicly available at:
    https://huggingface.co/Nita200/educator-anchored-hitl-pubmedbert
    https://huggingface.co/Nita200/educator-anchored-hitl-clinicalbert
    https://huggingface.co/Nita200/educator-anchored-hitl-roberta

Only needed if re-uploading after retraining. Requires HuggingFace
write token , see PUBLISHING_GUIDE.md.

Usage:
    python src/upload_to_hf.py --username Nita200
"""

import argparse
import json
import logging
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import HITL_MODELS, MODELS_DIR, RESULTS_DIR, TRANSFORMER_MODELS

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


# Model card template , reflects v3 canonical configuration.
# Update hyperparameters if uploading a v1 or v2 model.

MODEL_CARD_TEMPLATE = """---
language:
  - en
license: mit
tags:
  - clinical-nlp
  - healthcare-education
  - human-in-the-loop
  - educator-anchored
  - text-classification
  - transformer
datasets:
  - bigbio/mednli
metrics:
  - accuracy
  - f1
  - roc_auc
  - matthews_correlation
---

# {model_name} : Educator-Anchored HITL Clinical Reasoning Classifier

This model is a fine-tuned version of [{base_model}](https://huggingface.co/{base_model})
trained on the [MedNLI](https://physionet.org/content/mednli/1.0.0/) dataset,
used as a proxy for clinical reasoning scenarios in healthcare education.

It was refined using an **educator-anchored Human-in-the-Loop (HITL)** workflow
across {hitl_rounds} rounds of incremental, simulated educator-guided corrections,
as described in:

> *Educator-Anchored Human-in-the-Loop Learning: A Simulation Study of
> Transformer Models for Clinical Reasoning Assessment in Healthcare
> Education* — [citation pending]

## ⚠️ Important: Simulation Study Disclaimer

This model is a **research artefact from a simulation study**, not a
clinically validated or deployment-ready tool. Specifically:

- Training data (MedNLI) is a proxy for clinical reasoning, sourced from
  MIMIC-III clinical notes — it is **not** authentic learner submissions
  from a healthcare education context.
- HITL "educator corrections" during refinement were **simulated**
  (automated ground-truth relabelling of misclassified examples), not
  provided by real human educators.
- Do **not** use this model for actual clinical decision-making, patient
  safety assessment, or student grading without further validation by
  qualified healthcare educators.

This model is intended for **reproducibility and further research only**.

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
rresult = clf(
    "She was evaluated by neurosurgery, deemed to be intact neurologically. "
    "[SEP] Her neurological exam was normal and no deficits were identified."
)
print(result)
```

## Training

- **Base model**: {base_model}
- **Dataset**: MedNLI (PhysioNet credentialed access required), original
  80/10/10 train/validation/test split preserved
- **HITL configuration**: v3 (catastrophic-forgetting-mitigated)
- **HITL rounds completed**: {hitl_rounds} (stops early if the correction
  pool is exhausted before reaching the maximum of 20 rounds)
- **Corrections per round**: 50
- **Replay buffer size**: 100 (seed examples resampled each round to
  anchor prior representations and prevent catastrophic forgetting)
- **Learning rate**: 5e-6
- **Epochs per round**: 1
- **Seed/pool split**: 70% seed / 30% pool

This configuration was selected after a systematic three-version
comparison (see paper Section 4.3 and 5.2) showing that a naive
incremental fine-tuning configuration (higher learning rate, larger
correction batches, no replay buffer) produces catastrophic forgetting.
Five-fold cross-validation (paper Section 5.7) confirms that the AUC
stability achieved under this configuration generalises across
independent data splits, while the magnitude of accuracy improvement in
any single run is split-dependent.
"""


#  Round-count auto-detection

def get_actual_rounds_completed(model_key: str) -> int:
    """
    Read rounds completed from the v3 results JSON rather than assuming
    a fixed value , models exhaust their correction pool at different points.
    """
    results_path = RESULTS_DIR / "hitl_results_v3_seed70.json"
    if not results_path.exists():
        # Fall back to default path if the explicitly named file isn't present
        results_path = RESULTS_DIR / "hitl_results.json"

    if not results_path.exists():
        logger.warning(
            "Could not find HITL results JSON to auto-detect round count "
            "for %s — defaulting to 20.", model_key
        )
        return 20

    with open(results_path) as f:
        data = json.load(f)

    if model_key not in data:
        logger.warning(
            "%s not found in %s — defaulting to 20.", model_key, results_path
        )
        return 20

    # Final round number = max value in the "round" list (round 0 is the
    # seed baseline, not an actual HITL round)
    rounds = data[model_key]["round"]
    return max(rounds)


#  Upload logic 

def upload_model(
    model_key: str,
    username:  str,
    api:       HfApi,
) -> str:
    """
    Upload one HITL-trained model to HuggingFace Hub.

    Returns the model URL.
    """
    model_dir    = MODELS_DIR / f"{model_key}_hitl" / "final_model"
    base_model   = TRANSFORMER_MODELS[model_key]
    repo_name    = f"educator-anchored-hitl-{model_key}"
    repo_id      = f"{username}/{repo_name}"
    hitl_rounds  = get_actual_rounds_completed(model_key)

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
        model_name  = f"{model_key.upper()} Educator-Anchored HITL",
        base_model  = base_model,
        hitl_rounds = hitl_rounds,
        repo_id     = repo_id,
    ))

    # Upload all files in the model directory
    logger.info(
        "Uploading %s (%d rounds completed) to %s …",
        model_key, hitl_rounds, repo_id
    )
    upload_folder(
        folder_path = str(model_dir),
        repo_id     = repo_id,
        repo_type   = "model",
        commit_message = f"Upload HITL v3 {model_key} ({hitl_rounds} rounds)",
    )

    url = f"https://huggingface.co/{repo_id}"
    logger.info("✓ %s uploaded → %s", model_key, url)
    return url


#  Main 

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload HITL v3 models to HuggingFace Hub."
    )
    parser.add_argument(
        "--username", required=True,
        help="Your HuggingFace username (e.g. Nita200)"
    )
    args = parser.parse_args()

    api  = HfApi()
    urls = {}

    for model_key in HITL_MODELS:
        url = upload_model(model_key, args.username, api)
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



    