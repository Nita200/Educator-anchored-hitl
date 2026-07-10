# Educator-Anchored Human-in-the-Loop Learning

A simulation study of an educator-anchored Human-in-the-Loop (HITL) framework
for clinical reasoning assessment in healthcare education, using transformer
and classical machine learning models.

> *Educator-Anchored Human-in-the-Loop Learning: A Simulation Study of
> Transformer Models for Clinical Reasoning Assessment in Healthcare
> Education*

## Overview

This repository contains the full experimental pipeline, results, and
analysis for a study investigating whether AI-generated formative feedback
on clinical reasoning can remain reliably anchored to educator-guided
corrections over repeated incremental updates, without catastrophic
forgetting.

Three transformer models (PubMedBERT, ClinicalBERT, RoBERTa) and four
classical machine learning baselines (Logistic Regression, Random Forest,
XGBoost, SVM) are evaluated on a clinical reasoning classification task
built from the MedNLI dataset, used as a proxy for clinical reasoning
scenarios in the absence of a public dataset of authentic learner
submissions with educator markings.

## Key Findings

* Transformer models substantially outperform classical ML baselines across
  accuracy, macro F1, AUC, and Matthews Correlation Coefficient (MCC), with
  non-overlapping 95% bootstrap confidence intervals confirming the gap.
* A systematic three-configuration comparison (v1: no replay buffer →
  catastrophic forgetting; v2: replay buffer alone → partial improvement;
  v3: all fixes combined → stable) demonstrates that low learning rate,
  small correction batches, single fine-tuning epochs, and a replay buffer
  must be applied together to eliminate catastrophic forgetting.
* Five-fold cross-validation across independent seed/pool splits confirms
  that AUC stability (range 0.0022–0.0030 across folds) is a robust,
  generalisable property of the optimised configuration  while the
  magnitude of accuracy improvement observed in any single run is
  split-dependent and does not reliably generalise.
* An ablation study shows that BioGPT-generated rationales do not reliably
  improve classification performance; PubMedBERT achieves its highest
  point estimate without rationale augmentation.

See the full paper (linked once published) for complete methodology,
results, and discussion.

## Repository Structure

```
educator-anchored-hitl/
├── src/
│   ├── config.py                    # Central configuration
│   ├── utils.py                     # Shared utilities
│   ├── 01_data_preparation.py       # MedNLI loading + BioGPT rationale generation
│   ├── 02_baselines.py              # Classical ML models
│   ├── 03_transformers.py           # Transformer fine-tuning (5 models)
│   ├── 04_hitl.py                   # HITL loop with replay buffer
│   ├── 05_ablation_rationale.py     # Rationale contribution ablation
│   ├── 06_bootstrap_ci.py           # Bootstrap 95% confidence intervals
│   ├── 07_plot_learning_curves.py   # Learning curve figures
│   ├── 08_error_analysis.py         # Confusion matrices, per-class F1
│   ├── 09_mediqa_transfer.py        # MEDIQA NLI transfer test (blocked, see below)
│   ├── 10_cross_validation.py       # 5-fold cross-validation
│   
├── results/
│   ├── hitl_results_v1_catastrophic_forgetting.json
│   ├── hitl_results_v2_replay50.json
│   ├── hitl_results_v3_seed70.json  # canonical reported configuration
│   ├── bootstrap_ci.json
│   ├── ablation_results.json
│   ├── cv_summary.json
│   ├── cv/                          # per-fold cross-validation results
│   ├── error_analysis.json
│   └── figures/                     # all generated figures
├── requirements.txt
└── LICENSE                          # MIT
```

## Setup

```bash
git clone https://github.com/Nita200/Educator-anchored-hitl.git
cd Educator-anchored-hitl
pip install -r requirements.txt
```

GPU strongly recommended (experiments were run on an NVIDIA A100 80GB).

## Reproducing the Results

Run scripts in order:

```bash
python src/01_data_preparation.py     
python src/02_baselines.py             
python src/03_transformers.py          
python src/04_hitl.py                  
python src/05_ablation_rationale.py   
python src/06_bootstrap_ci.py         
python src/07_plot_learning_curves.py  
python src/08_error_analysis.py        
python src/10_cross_validation.py      
```

### Reproducing Specific HITL Configurations

The canonical reported configuration (v3) is set by default in
`src/config.py`. To reproduce the other two configurations documented in
the paper, edit `src/config.py` before running `04_hitl.py`:

**v1: catastrophic forgetting baseline:**
```python
CORRECTIONS_PER_ROUND = 150
HITL_FINETUNE_EPOCHS  = 2
HITL_LEARNING_RATE    = 2e-5
REPLAY_BUFFER_SIZE     = 0
```

**v2: replay buffer only:**
```python
CORRECTIONS_PER_ROUND = 150
HITL_FINETUNE_EPOCHS  = 2
HITL_LEARNING_RATE    = 2e-5
REPLAY_BUFFER_SIZE     = 50
```

**v3:  all fixes (canonical, default in config.py):**
```python
CORRECTIONS_PER_ROUND = 50
HITL_FINETUNE_EPOCHS  = 1
HITL_LEARNING_RATE    = 5e-6
REPLAY_BUFFER_SIZE     = 100
SEED_FRACTION          = 0.70
```

A secondary robustness check with `SEED_FRACTION = 0.85` is also
documented in the paper (Section 5.7) and noted as a comment in
`config.py`.

## Known Limitation — MEDIQA NLI Transfer Test

`09_mediqa_transfer.py` is included for completeness but the evaluation
could not be completed in this study. The MEDIQA NLI test set is hosted on
PhysioNet and requires credentialed access and a signed Data Use
Agreement, which was not obtained within the study timeframe. See the
paper's limitations section for details.

## Trained Models

Final v3 HITL model checkpoints are available on HuggingFace:

| Model | Link |
|---|---|
| PubMedBERT (HITL v3) | https://huggingface.co/Nita200/educator-anchored-hitl-pubmedbert |
| ClinicalBERT (HITL v3) | https://huggingface.co/Nita200/educator-anchored-hitl-clinicalbert |
| RoBERTa (HITL v3) | https://huggingface.co/Nita200/educator-anchored-hitl-roberta |

## Citation

```bibtex
@article{nana2026educatoranchoredhitl,
  title   = {Educator-Anchored Human-in-the-Loop Learning: A Simulation
             Study of Transformer Models for Clinical Reasoning Assessment
             in Healthcare Education},
  author  = {Nana, Vanita Kouomogne, Mark T. Marshall},
  year    = {2026},
  note    = {Manuscript under review}
}
```

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

This work uses the MedNLI dataset, originally derived from MIMIC-III
clinical notes, and the BioGPT model for rationale generation. See the
paper's References section for full citations.
