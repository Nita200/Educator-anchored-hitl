# Publishing Guide

This guide covers publishing this repository to GitHub and uploading
trained model checkpoints to HuggingFace.

## 1. GitHub

Repository already created and pushed:

- **Repository name**: `Educator-anchored-hitl`
- **URL**: https://github.com/Nita200/Educator-anchored-hitl
- **Description**: *Simulation study of an educator-anchored Human-in-the-Loop
  framework for clinical reasoning assessment in healthcare education*

### Pushing further updates

```bash
cd path/to/Educator-anchored-hitl
git add .
git commit -m "Description of changes"
git push origin main
```

### Adding a DOI (optional, for citability)

Once the repository is stable, archive a release via Zenodo for a
permanent DOI:

1. Connect the repo at https://zenodo.org/account/settings/github/
2. Create a GitHub release (tag e.g. `v1.0`)
3. Zenodo automatically archives it and issues a DOI
4. Add the DOI badge to README.md

## 2. HuggingFace — Trained Models

Three final v3 HITL model checkpoints should be uploaded:

```bash
python src/upload_to_hf.py
```

Verify `src/upload_to_hf.py` points at the correct local checkpoint paths
before running — these should be:

```
models/pubmedbert_hitl/final_model
models/clinicalbert_hitl/final_model
models/roberta_hitl/final_model
```

### Expected HuggingFace repos after upload

```
pubmedbert       →  https://huggingface.co/Nita200/educator-anchored-hitl-pubmedbert
clinicalbert     →  https://huggingface.co/Nita200/educator-anchored-hitl-clinicalbert
roberta          →  https://huggingface.co/Nita200/educator-anchored-hitl-roberta
```

### Model card checklist

Each HuggingFace model repo should include a model card noting:

- This is a **research artefact from a simulation study**, not a
  clinically validated or deployed tool
- Trained on MedNLI (proxy data), not authentic learner submissions
- Educator corrections during HITL refinement were **simulated**, not
  provided by real educators
https://huggingface.co/Nita200/educator-anchored-hitl-pubmedbert https://huggingface.co/Nita200/educator-anchored-hitl-clinicalbert https://huggingface.co/Nita200/educator-anchored-hitl-roberta - Intended for reproducibility and further research only

| Model | HuggingFace Link |
|---|---|
| PubMedBERT (HITL v3) | https://huggingface.co/Nita200/educator-anchored-hitl-pubmedbert |
| ClinicalBERT (HITL v3) | https://huggingface.co/Nita200/educator-anchored-hitl-clinicalbert |
| RoBERTa (HITL v3) | https://huggingface.co/Nita200/educator-anchored-hitl-roberta |

## 3. Linking From the Paper

Add to the Methods section or as a footnote on the first page:

```latex
Code and experimental results are publicly available at
\url{https://github.com/Nita200/Educator-anchored-hitl}. Trained model
checkpoints are available on HuggingFace (links in the repository README).
```

## 4. Pre-Publication Checklist

- [ ] `src/config.py` has `SEED_FRACTION = 0.70` active (canonical v3),
      not `0.85` (robustness-check variant)
- [ ] All three HITL version result files present and correctly named:
      `hitl_results_v1_catastrophic_forgetting.json`,
      `hitl_results_v2_replay50.json`, `hitl_results_v3_seed70.json`
- [ ] README reflects current paper title and framing (healthcare
      education, not nursing-specific)
- [ ] No leftover references to BioBERT (superseded by PubMedBERT)
- [ ] HuggingFace model cards include the simulation-study disclaimer
- [ ] GitHub and HuggingFace links added to the paper
