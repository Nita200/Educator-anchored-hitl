# Publishing Your Work: GitHub & HuggingFace Hub

This guide walks you through publishing the code on GitHub and the trained
models on HuggingFace Hub. Do this **after** you are happy with the paper
and code — ideally upon journal acceptance, or as a preprint alongside
arXiv submission.

---

## Part 1 — GitHub (Code Repository)

### Step 1: Create a GitHub repository

1. Go to https://github.com and sign in (or create a free account)
2. Click **New repository** (top-right `+` icon)
3. Fill in:
   - **Repository name**: `hitl-nursing-education` (or similar)
   - **Description**: *Simulation study of an educator-in-the-loop HITL approach for clinical reasoning assessment in nursing education*
   - **Visibility**: Public (required for open-science compliance with most journals)
   - **Initialise**: leave unchecked (you will push existing files)
4. Click **Create repository**

### Step 2: Initialise Git locally

Open a terminal in the `HITL_NursingAI/` folder:

```bash
cd path/to/HITL_NursingAI

git init
git add .
git commit -m "Initial submission: HITL nursing education simulation study"
```

### Step 3: Connect and push

Replace `your-username` with your GitHub username:

```bash
git remote add origin https://github.com/your-username/hitl-nursing-education.git
git branch -M main
git push -u origin main
```

### Step 4: Create a release (for journal citation)

Journals require a permanent, citable version of the code.

1. On GitHub, click **Releases** → **Draft a new release**
2. Tag: `v1.0.0`
3. Title: `Submission v1.0.0`
4. Description: summarise what the release contains
5. Click **Publish release**

### Step 5: Get a DOI via Zenodo (recommended for Q1 journals)

Many Q1 journals (including IEEE TLT) now require or strongly recommend a
DOI for the code repository.

1. Go to https://zenodo.org and sign in with your GitHub account
2. Go to **GitHub** tab → enable the toggle for your repository
3. On GitHub, create a new release (Step 4 above)
4. Zenodo automatically creates a DOI — copy it
5. Add the DOI badge to your README.md:
   ```
   [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
   ```

---

## Part 2 — HuggingFace Hub (Model Weights)

### Step 1: Create a HuggingFace account

Go to https://huggingface.co and create a free account.

### Step 2: Create an access token

1. Go to https://huggingface.co/settings/tokens
2. Click **New token**
3. Name: `upload-hitl-models`
4. Permission: **Write**
5. Copy the token — you will need it once

### Step 3: Install and log in

```bash
pip install huggingface_hub
huggingface-cli login
# Paste your token when prompted
```

### Step 4: Run the upload script

Make sure `04_hitl.py` has already completed (models are in `models/`), then:

```bash
python src/upload_to_hf.py --username your_hf_username --rounds 20
```

This will:
- Create three model repositories on HuggingFace Hub
- Upload weights, tokeniser, config, and a model card for each
- Print the permanent URLs

Example output:
```
============================================================
Upload complete. Add these URLs to your README.md:
============================================================
  t5-small         →  https://huggingface.co/your-username/hitl-nursing-t5-small
  biobert          →  https://huggingface.co/your-username/hitl-nursing-biobert
  clinicalbert     →  https://huggingface.co/your-username/hitl-nursing-clinicalbert
============================================================
```

### Step 5: Update README.md

Open `README.md` and fill in the placeholder:

```markdown
## Pre-trained Model Weights

Fine-tuned HITL model checkpoints are hosted on HuggingFace Hub:

| Model        | Link |
|--------------|------|
| T5-small     | https://huggingface.co/your-username/hitl-nursing-t5-small |
| BioBERT      | https://huggingface.co/your-username/hitl-nursing-biobert |
| ClinicalBERT | https://huggingface.co/your-username/hitl-nursing-clinicalbert |
```

---

## Part 3 — What to include in the paper

Once both are live, add the following to your paper's **Data Availability**
or **Code Availability** statement (IEEE TLT requires one):

> The code and data for this study are publicly available at
> [https://github.com/your-username/hitl-nursing-education](https://github.com/your-username/hitl-nursing-education)
> (DOI: 10.5281/zenodo.XXXXXXX).
> Pre-trained model weights are available on HuggingFace Hub at
> [https://huggingface.co/your-username](https://huggingface.co/your-username).

---

## Summary checklist

- [ ] GitHub repository created and code pushed
- [ ] GitHub release tagged (`v1.0.0`)
- [ ] Zenodo DOI obtained and badge added to README
- [ ] HuggingFace account created
- [ ] Models uploaded via `upload_to_hf.py`
- [ ] HuggingFace URLs added to README
- [ ] Data availability statement written in paper
