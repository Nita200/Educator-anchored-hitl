# Hybrid Intelligence for Nursing Education
## A Simulation Study of an Educator-in-the-Loop HITL Approach

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
<!-- Add Zenodo DOI badge here after uploading -->

Code for the paper:
> *Hybrid Intelligence for Nursing Education: A Simulation Study of an Educator-in-the-Loop Approach Using Transformer and Machine Learning Models for Clinical Reasoning Assessment*
> [Citation pending]

---

## Overview

This study simulates a Human-in-the-Loop (HITL) workflow for clinical reasoning
assessment in nursing education. Transformer models (T5-small, BioBERT,
ClinicalBERT) are incrementally refined through simulated educator corrections
over 20 rounds and compared against classical ML baselines (Logistic Regression,
Random Forest, XGBoost).

The MedNLI dataset is used as a proxy for nursing judgment scenarios.
BioGPT generates clinical rationales in three student voices (novice, clinical,
confident), which are concatenated with scenarios and judgments as model input.

---

## Repository Structure

```
HITL_NursingAI/
│
├── src/
│   ├── config.py               ← All hyperparameters and paths (edit here)
│   ├── utils.py                ← Shared utilities and dataset class
│   ├── 01_data_preparation.py  ← Load MedNLI, generate rationales, save splits
│   ├── 02_baselines.py         ← ML baselines (LR, RF, XGBoost) with TF-IDF
│   ├── 03_transformers.py      ← Transformer fine-tuning (no HITL)
│   ├── 04_hitl.py              ← HITL incremental fine-tuning loop
│   └── upload_to_hf.py         ← Upload trained models to HuggingFace Hub
│
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_baselines.ipynb
│   ├── 03_transformers.ipynb
│   └── 04_hitl.ipynb
│
├── data/
│   ├── train_full_with_rationales.csv
│   ├── validation_full_with_rationales.csv
│   ├── test_full_with_rationales.csv
│   ├── train_full.csv
│   ├── validation_full.csv
│   └── test_full.csv
│
├── results/                    ← Created automatically on first run
│   ├── baseline_results.json
│   ├── transformer_results.json
│   ├── hitl_results.json
│   └── figures/
│
├── models/                     ← Saved model checkpoints (gitignored)
├── requirements.txt
├── PUBLISHING_GUIDE.md         ← Step-by-step GitHub and HuggingFace upload guide
└── README.md
```
'''
## Experiment Versions

- `results/hitl_results_v1_catastrophic_forgetting.json` — Initial HITL run 
  (lr=2e-5, 150 corrections/round, no replay buffer). Exhibits catastrophic 
  forgetting as documented in Section 4.3.1.
- `results/hitl_results.json` — Corrected HITL run (lr=5e-6, 50 corrections/round, 
  replay buffer=50). Stable learning curves reported in Section 5.
'''
---

## Reproducing the Experiments

### 1. Setup

```bash
git clone https://github.com/your-username/hitl-nursing-education.git
cd hitl-nursing-education
pip install -r requirements.txt
```

A GPU is strongly recommended for scripts 3 and 4.

### 2. Run in order

```bash
# Step 1 — Data preparation and rationale generation
python src/01_data_preparation.py

# Step 2 — Classical ML baselines (CPU, ~5 min)
python src/02_baselines.py

# Step 3 — Transformer baselines without HITL (GPU, ~2 hrs)
python src/03_transformers.py

# Step 4 — HITL incremental fine-tuning (GPU, ~3–4 hrs)
python src/04_hitl.py
```

All hyperparameters (learning rate, HITL rounds, corrections per round, etc.)
are centralised in `src/config.py`.

### 3. Publish trained models

After step 4 completes, upload models to HuggingFace Hub:

```bash
python src/upload_to_hf.py --username your_hf_username
```

See `PUBLISHING_GUIDE.md` for full instructions including GitHub releases
and Zenodo DOI registration.

---

### 4. ## Reproducing the Catastrophic Forgetting Results (Section 4.3.1)

To reproduce the initial unstable HITL results documented in Section 4.3.1,
change the following values in `src/config.py` before running `04_hitl.py`:

    CORRECTIONS_PER_ROUND = 150   # original value
    HITL_FINETUNE_EPOCHS  = 2     # original value
    HITL_LEARNING_RATE    = 2e-5  # original value
    REPLAY_BUFFER_SIZE    = 0     # disable replay buffer

The output of this run is preserved in:
    results/hitl_results_v1_catastrophic_forgetting.json

## Data Split

| Split      | File                                | Records |
|------------|-------------------------------------|---------|
| Train      | train_full_with_rationales.csv      | ~9,800  |
| Validation | validation_full_with_rationales.csv | ~1,395  |
| Test       | test_full_with_rationales.csv       | ~2,805  |

Split ratio: 70 / 10 / 20. The test set is never updated during HITL.

---

## Key Results

| Model             | Accuracy | Macro F1 | AUC   |
|-------------------|----------|----------|-------|
| Logistic Regression | 0.712  | 0.71     | 0.868 |
| Random Forest     | 0.660    | 0.66     | 0.830 |
| XGBoost           | 0.710    | 0.71     | 0.875 |
| T5-small          | 0.910    | 0.91     | 0.929 |
| ClinicalBERT      | 0.890    | 0.89     | 0.977 |
| BioBERT           | 0.870    | 0.87     | 0.968 |

After 20 HITL rounds all three transformer models exceed their pre-HITL
baselines, with T5-small reaching the highest accuracy.

---

## Pre-trained Model Weights

> [To be updated after HuggingFace upload]

---

## Dataset

[MedNLI](https://physionet.org/content/mednli/1.0.0/) is available on PhysioNet
and requires credentialed access. The CSV files in `data/` contain the
preprocessed splits with BioGPT-generated rationales used in this study.

---

## Citation

```bibtex
@article{nana2025hitlnursing,
  title   = {Hybrid Intelligence for Nursing Education: A Simulation Study
             of an Educator-in-the-Loop Approach},
  author  = {Nana, V.K. and others},
  journal = {[pending]},
  year    = {2025}
}
```

---

## License

Code: MIT License. Data derived from MedNLI is subject to the
[PhysioNet Credentialed Health Data License](https://physionet.org/about/licenses/physionet-credentialed-health-data-license-150/).
