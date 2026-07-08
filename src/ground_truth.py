"""
ground_truth.py

Reads all results JSON files and generates a single reference document
containing every number reported in the paper.  this serves to cross-check
table cells and prose claims against the actual experimental output.

Usage:
    python src/ground_truth.py

Output:
    results/ground_truth.txt
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import RESULTS_DIR

OUT_PATH = RESULTS_DIR / "ground_truth.txt"
lines = []


def section(title):
    lines.append("")
    lines.append("=" * 70)
    lines.append(f"  {title}")
    lines.append("=" * 70)


def subsection(title):
    lines.append("")
    lines.append(f"── {title} " + "─" * (66 - len(title)))


def load(filename):
    path = RESULTS_DIR / filename
    if not path.exists():
        lines.append(f"  ⚠️  NOT FOUND: {filename}")
        return None
    with open(path) as f:
        return json.load(f)


def fmt(v, decimals=4):
    return f"{v:.{decimals}f}"


def fmt_ci(d, metric, decimals=3):
    m = d[metric]
    return (f"{m['mean']:.{decimals}f} "
            f"[{m['ci_low']:.{decimals}f}–{m['ci_high']:.{decimals}f}]")


# ═════════════════════════════════════════════════════════════════════════════
#  TABLE 1 — Baseline models (from baseline_results.json + ablation)
# ═════════════════════════════════════════════════════════════════════════════

section("TABLE 1 — Baseline model comparison")

baseline = load("baseline_results.json")
ablation = load("ablation_results.json")
transformer = load("transformer_results.json")

lines.append("")
lines.append(f"  {'Model':<30} {'Accuracy':>10} {'Macro F1':>10} "
             f"{'AUC':>10} {'MCC':>10}")
lines.append("  " + "-" * 64)

if baseline:
    lines.append("  Classical ML Baselines")
    for model, metrics in baseline.items():
        lines.append(
            f"  {model:<30} "
            f"{fmt(metrics.get('accuracy', 0)):>10} "
            f"{fmt(metrics.get('macro_f1', 0)):>10} "
            f"{fmt(metrics.get('auc', 0)):>10} "
            f"{fmt(metrics.get('mcc', 0)):>10}"
        )

if transformer:
    lines.append("  Transformer models — with rationale")
    for model, metrics in transformer.items():
        lines.append(
            f"  {model:<30} "
            f"{fmt(metrics.get('accuracy', 0)):>10} "
            f"{fmt(metrics.get('macro_f1', 0)):>10} "
            f"{fmt(metrics.get('auc', 0)):>10} "
            f"{fmt(metrics.get('mcc', 0)):>10}"
        )

if ablation:
    lines.append("  Transformer models — without rationale (ablation)")
    for model, metrics in ablation.items():
        lines.append(
            f"  {model+'†':<30} "
            f"{fmt(metrics.get('accuracy', 0)):>10} "
            f"{fmt(metrics.get('macro_f1', 0)):>10} "
            f"{fmt(metrics.get('auc', 0)):>10} "
            f"{fmt(metrics.get('mcc', 0)):>10}"
        )


# ═════════════════════════════════════════════════════════════════════════════
#  TABLE 3 — Three-version HITL comparison
# ═════════════════════════════════════════════════════════════════════════════

section("TABLE 3 — Three-version HITL comparison")

v1 = load("hitl_results_v1_catastrophic_forgetting.json")
v2 = load("hitl_results_v2_replay50.json")
v3 = load("hitl_results_v3_seed70.json")

lines.append("")
lines.append(f"  {'Version/Model':<22} {'Seed acc':>10} {'Min acc':>10} "
             f"{'Final acc':>10} {'Δ acc':>8} "
             f"{'AUC min':>9} {'AUC max':>9} {'Rounds':>7}")
lines.append("  " + "-" * 90)

for label, data in [("v1", v1), ("v2", v2), ("v3", v3)]:
    if not data:
        lines.append(f"  {label}: FILE NOT FOUND")
        continue
    lines.append(f"  {label}")
    for model, curves in data.items():
        acc   = curves["accuracy"]
        auc   = curves["auc"]
        seed  = acc[0]
        final = acc[-1]
        mn    = min(acc)
        delta = final - seed
        auc_min = min(auc)
        auc_max = max(auc)
        rounds  = len(acc) - 1
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"  {'  '+model:<22} "
            f"{fmt(seed):>10} "
            f"{fmt(mn):>10} "
            f"{fmt(final):>10} "
            f"{sign}{fmt(delta):>7} "
            f"{fmt(auc_min):>9} "
            f"{fmt(auc_max):>9} "
            f"{rounds:>7}"
        )
    lines.append("")


# ═════════════════════════════════════════════════════════════════════════════
#  TABLE 4 — HITL v3 bootstrap CIs (final round models)
# ═════════════════════════════════════════════════════════════════════════════

section("TABLE 4 — Bootstrap 95% CIs for HITL v3 final models")

ci = load("bootstrap_ci.json")
if ci:
    lines.append("")
    lines.append(f"  {'Model / Condition':<38} "
                 f"{'Accuracy [95% CI]':>26} "
                 f"{'AUC [95% CI]':>26} "
                 f"{'MCC [95% CI]':>26}")
    lines.append("  " + "-" * 120)

    hitl_keys = {
        "pubmedbert_hitl_final":  "PubMedBERT HITL v3 final",
        "clinicalbert_hitl_final": "ClinicalBERT HITL v3 final",
        "roberta_hitl_final":      "RoBERTa HITL v3 final",
    }
    for key, label in hitl_keys.items():
        if key not in ci:
            lines.append(f"  {label}: NOT FOUND in bootstrap_ci.json")
            continue
        d = ci[key].get("hitl_v3_final", {})
        lines.append(
            f"  {label:<38} "
            f"{fmt_ci(d,'accuracy'):>26} "
            f"{fmt_ci(d,'auc'):>26} "
            f"{fmt_ci(d,'mcc'):>26}"
        )


# ═════════════════════════════════════════════════════════════════════════════
#  TABLE 5 — Ablation study bootstrap CIs
# ═════════════════════════════════════════════════════════════════════════════

section("TABLE 5 — Ablation study bootstrap CIs")

if ci:
    lines.append("")
    lines.append(f"  {'Model / Condition':<38} "
                 f"{'Accuracy [95% CI]':>26} "
                 f"{'MCC [95% CI]':>26}")
    lines.append("  " + "-" * 94)

    for model in ["pubmedbert", "clinicalbert", "roberta"]:
        if model not in ci:
            lines.append(f"  {model}: NOT FOUND")
            continue
        for condition in ["with_rationale", "no_rationale"]:
            if condition not in ci[model]:
                lines.append(f"  {model} ({condition}): NOT FOUND")
                continue
            d     = ci[model][condition]
            label = f"{model} ({condition})"
            lines.append(
                f"  {label:<38} "
                f"{fmt_ci(d,'accuracy'):>26} "
                f"{fmt_ci(d,'mcc'):>26}"
            )
        lines.append("")


# ═════════════════════════════════════════════════════════════════════════════
#  TABLE 6 — 5-fold cross-validation summary
# ═════════════════════════════════════════════════════════════════════════════

section("TABLE 6 — 5-fold cross-validation summary")

cv = load("cv_summary.json")
if cv:
    lines.append("")
    lines.append(f"  {'Model':<15} {'Folds':>6} "
                 f"{'Seed Acc mean±std':>22} "
                 f"{'Final Acc mean±std':>22} "
                 f"{'Min Acc mean±std':>22} "
                 f"{'AUC Range mean±std':>22}")
    lines.append("  " + "-" * 115)

    for model, stats in cv.items():
        lines.append(
            f"  {model:<15} {stats['n_folds_completed']:>6} "
            f"{stats['seed_accuracy']['mean']:.3f}±{stats['seed_accuracy']['std']:.3f}{'':>12} "
            f"{stats['final_accuracy']['mean']:.3f}±{stats['final_accuracy']['std']:.3f}{'':>12} "
            f"{stats['min_accuracy']['mean']:.3f}±{stats['min_accuracy']['std']:.3f}{'':>12} "
            f"{stats['auc_range']['mean']:.4f}±{stats['auc_range']['std']:.4f}{'':>8}"
        )


# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE CROSS-CHECK — per-figure values to verify against PNGs
# ═════════════════════════════════════════════════════════════════════════════

section("FIGURE CROSS-CHECK — values to verify against each figure")

subsection("Figure 2 — v1 learning curves (catastrophic forgetting)")
if v1:
    for model, curves in v1.items():
        acc = curves["accuracy"]
        auc = curves["auc"]
        lines.append(
            f"  {model}: seed={fmt(acc[0])} "
            f"min={fmt(min(acc))} (round {acc.index(min(acc))}) "
            f"final={fmt(acc[-1])} "
            f"AUC range={fmt(min(auc))}–{fmt(max(auc))}"
        )

subsection("Figure 3 — v3 learning curves (optimised)")
if v3:
    for model, curves in v3.items():
        acc = curves["accuracy"]
        auc = curves["auc"]
        lines.append(
            f"  {model}: seed={fmt(acc[0])} "
            f"min={fmt(min(acc))} (round {acc.index(min(acc))}) "
            f"final={fmt(acc[-1])} "
            f"AUC range={fmt(min(auc))}–{fmt(max(auc))} "
            f"(spread={fmt(max(auc)-min(auc))})"
        )

subsection("Figure 4 — Three-version comparison: PubMedBERT")
lines.append("  Check figure shows THREE lines (v1 red, v2 orange, v3 green)")
lines.append("  Check shaded 'catastrophic forgetting zone' appears below 0.65")
for label, data in [("v1", v1), ("v2", v2), ("v3", v3)]:
    if data and "pubmedbert" in data:
        acc = data["pubmedbert"]["accuracy"]
        auc = data["pubmedbert"]["auc"]
        lines.append(
            f"  PubMedBERT {label}: "
            f"seed={fmt(acc[0])} min={fmt(min(acc))} final={fmt(acc[-1])} "
            f"AUC {fmt(min(auc))}–{fmt(max(auc))}"
        )

subsection("Figure 5 — AUC stability: v1 vs v3 all models")
lines.append("  Check THREE panels (one per model), each showing v1 and v3 lines")
for model in ["pubmedbert", "clinicalbert", "roberta"]:
    for label, data in [("v1", v1), ("v3", v3)]:
        if data and model in data:
            auc = data[model]["auc"]
            spread = max(auc) - min(auc)
            lines.append(
                f"  {model} {label}: "
                f"AUC {fmt(min(auc))}–{fmt(max(auc))} "
                f"(spread={fmt(spread)} — annotated in figure)"
            )

subsection("Figure 6 — Confusion matrices (in limitations / appendix)")
lines.append("  Check: pre-HITL = full-data model (Table 1 numbers)")
lines.append("  Check: post-HITL = v3 final model (Table 4 numbers)")
lines.append("  Known finding: ClinicalBERT unsafe→safe errors 37→87 (+135%)")
lines.append("  Known finding: RoBERTa ambiguous→safe errors 75→142 (+89%)")

subsection("Figure 7 — Per-class F1 before and after HITL")
lines.append("  Check solid bars = pre-HITL (Table 1 values)")
lines.append("  Check hatched bars = post-HITL v3 final (Table 4 values)")
lines.append("  Expected: macro-average F1 decreases for all three models post-HITL")
lines.append("  (comparison is full-data pre vs 70%-seed post — inherently unfair)")


# ═════════════════════════════════════════════════════════════════════════════
#  PROSE CLAIMS — every hardcoded number that appears in the paper body
# ═════════════════════════════════════════════════════════════════════════════

section("PROSE CLAIMS — numbers to verify in paper body text")

lines.append("")
lines.append("  These are the key numbers that appear as claims in the")
lines.append("  Introduction, Results, Discussion, and Conclusion.")
lines.append("  Each is computed directly from source JSON below.")
lines.append("")

if v1 and "pubmedbert" in v1:
    acc = v1["pubmedbert"]["accuracy"]
    auc = v1["pubmedbert"]["auc"]
    min_idx = acc.index(min(acc))
    lines.append(f"  v1 PubMedBERT: dropped from {fmt(acc[0])} at round 0 "
                 f"to {fmt(min(acc))} by round {min_idx}")
    lines.append(f"  v1 AUC range: {fmt(min(auc))}–{fmt(max(auc))}")

if v3:
    for model, curves in v3.items():
        acc = curves["accuracy"]
        auc = curves["auc"]
        seed, final = acc[0], acc[-1]
        delta = final - seed
        sign = "+" if delta >= 0 else ""
        auc_range = max(auc) - min(auc)
        rounds = len(acc) - 1
        lines.append(
            f"  v3 {model}: seed={fmt(seed)} → final={fmt(final)} "
            f"({sign}{fmt(delta)}) | "
            f"AUC {fmt(min(auc))}–{fmt(max(auc))} "
            f"(spread={fmt(auc_range)}) | {rounds} rounds"
        )

if ci:
    lines.append("")
    lines.append("  Bootstrap CIs (classical baselines — non-overlap claim):")
    for model in ["logistic_regression", "random_forest", "xgboost", "svm"]:
        if model in ci:
            d = ci[model].get("baseline", {})
            lines.append(
                f"    {model}: MCC {fmt_ci(d,'mcc')}"
            )
    lines.append("  Bootstrap CIs (top transformer with rationale):")
    for model in ["pubmedbert", "clinicalbert", "roberta"]:
        if model in ci and "with_rationale" in ci[model]:
            d = ci[model]["with_rationale"]
            lines.append(
                f"    {model}: accuracy {fmt_ci(d,'accuracy')} "
                f"MCC {fmt_ci(d,'mcc')}"
            )

if cv:
    lines.append("")
    lines.append("  Cross-validation AUC ranges (key robust finding):")
    for model, stats in cv.items():
        lines.append(
            f"    {model}: AUC range "
            f"{stats['auc_range']['mean']:.4f}±{stats['auc_range']['std']:.4f}"
        )
    lines.append("  Cross-validation accuracy changes (seed → final):")
    for model, stats in cv.items():
        delta = (stats['final_accuracy']['mean']
                 - stats['seed_accuracy']['mean'])
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"    {model}: {sign}{delta:.3f} "
            f"(seed {stats['seed_accuracy']['mean']:.3f} → "
            f"final {stats['final_accuracy']['mean']:.3f})"
        )


# ═════════════════════════════════════════════════════════════════════════════
#  WRITE OUTPUT
# ═════════════════════════════════════════════════════════════════════════════

output = "\n".join(lines)
OUT_PATH.write_text(output)
print(output)
print(f"\n{'='*70}")
print(f"Saved to: {OUT_PATH}")
print("Cross-check every table cell and prose claim in the paper against")
print("this file. If any number differs, the paper is wrong, not this file.")
print(f"{'='*70}")