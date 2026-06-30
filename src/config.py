"""
config.py — Central configuration for all hyperparameters and paths.
Modify this file to adjust experimental settings without touching any other script.

HITL version history:
    v1: CORRECTIONS=150, EPOCHS=2, LR=2e-5, REPLAY=0    → catastrophic forgetting
    v2: CORRECTIONS=150, EPOCHS=2, LR=2e-5, REPLAY=50   → improved AUC, still oscillates
    v3: CORRECTIONS=50,  EPOCHS=1, LR=5e-6, REPLAY=100  → current (all fixes applied)
"""

from pathlib import Path

# ── Directory layout ──────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MODELS_DIR  = ROOT / "models"

# ── Label mapping ─────────────────────────────────────────────────────────────
LABEL_MAP = {
    "entailment":    "safe",
    "contradiction": "unsafe",
    "neutral":       "ambiguous",
}
LABEL2ID   = {"safe": 0, "unsafe": 1, "ambiguous": 2}
ID2LABEL   = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)

# ── Dataset splits ────────────────────────────────────────────────────────────
TRAIN_FILE = DATA_DIR / "train_full_with_rationales.csv"
VAL_FILE   = DATA_DIR / "validation_full_with_rationales.csv"
TEST_FILE  = DATA_DIR / "test_full_with_rationales.csv"

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10
RANDOM_SEED = 42

# ── Rationale generation (01_data_preparation.py) ─────────────────────────────
BIOGPT_MODEL      = "microsoft/biogpt"
MAX_RATIONALE_LEN = 128
STUDENT_VOICES    = ["novice", "clinical", "confident"]

VOICE_PROMPTS = {
    "novice": (
        "The patient has {sentence1}. The student thinks: {sentence2}. "
        "Explain in simple terms why this is correct or incorrect clinically."
    ),
    "clinical": (
        "Given the clinical scenario: {sentence1}. Assessment: {sentence2}. "
        "Provide a structured clinical rationale."
    ),
    "confident": (
        "Clinical scenario: {sentence1}. Judgement: {sentence2}. "
        "Give a concise, authoritative clinical justification."
    ),
}

# ── Baseline ML (02_baselines.py) ─────────────────────────────────────────────
TFIDF_MAX_FEATURES = 10_000
TFIDF_NGRAM_RANGE  = (1, 2)

BASELINE_MODELS = {
    "logistic_regression": {"C": 1.0, "max_iter": 1000,
                            "random_state": RANDOM_SEED},
    "random_forest":       {"n_estimators": 200,
                            "random_state": RANDOM_SEED},
    "xgboost":             {"n_estimators": 200,
                            "random_state": RANDOM_SEED,
                            "use_label_encoder": False,
                            "eval_metric": "mlogloss"},
    "svm":                 {"C": 1.0, "max_iter": 2000,
                            "random_state": RANDOM_SEED},
}

# ── Transformer fine-tuning (03_transformers.py) ──────────────────────────────
TRANSFORMER_MODELS = {
    "t5-small":     "t5-small",
    "pubmedbert":   "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "clinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
    "distilbert":   "distilbert-base-uncased",
    "roberta":      "roberta-base",
}

MAX_INPUT_LENGTH = 256
BATCH_SIZE       = 16
LEARNING_RATE    = 2e-5
WARMUP_RATIO     = 0.1
WEIGHT_DECAY     = 0.01
NUM_TRAIN_EPOCHS = 3
FP16             = True   # Set False if running on CPU

# ── HITL workflow (04_hitl.py) — v3 configuration ─────────────────────────────
#
# All four changes applied together for the first time in v3:
#   1. Reduced corrections per round   (150 → 50)  : smaller, more stable updates
#   2. Reduced fine-tuning epochs      (2 → 1)     : limits update magnitude
#   3. Reduced learning rate           (2e-5 → 5e-6): prevents overwriting prior knowledge
#   4. Increased replay buffer         (50 → 100)  : stronger anchor to seed representations
#
HITL_MODELS           = ["pubmedbert", "clinicalbert", "roberta"]
HITL_ROUNDS           = 20
CORRECTIONS_PER_ROUND = 50       # v1: 150 | v2: 150 | v3: 50
HITL_FINETUNE_EPOCHS  = 1        # v1: 2   | v2: 2   | v3: 1
HITL_LEARNING_RATE    = 5e-6     # v1: 2e-5| v2: 2e-5| v3: 5e-6
REPLAY_BUFFER_SIZE    = 100      # v1: 0   | v2: 50  | v3: 100
SEED_FRACTION         = 0.70     # proportion of train set used as initial seed
#SEED_FRACTION = 0.85   # was 0.70
