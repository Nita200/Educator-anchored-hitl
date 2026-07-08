"""
07_plot_learning_curves.py
###################################
Generates learning curve figures for all three HITL configurations
(v1, v2, v3). Reads directly from the three results JSON files 
NB: to be rerun after any change to those files to regenerate all figures.

Figures produced:
    fig2a — v1 accuracy curves (catastrophic forgetting baseline)
    fig2b — v1 AUC curves
    fig3a — v3 accuracy curves (optimised configuration)
    fig3b — v3 AUC curves
    fig4  — three-version accuracy comparison for PubMedBERT
    fig5  — three-version AUC comparison for all models

Usage:
    python src/07_plot_learning_curves.py

Outputs:
    results/figures/fig2a_v1_accuracy.png
    results/figures/fig2b_v1_auc.png
    results/figures/fig3a_v3_accuracy.png
    results/figures/fig3b_v3_auc.png
    results/figures/fig4_version_comparison_pubmedbert.png
    results/figures/fig5_auc_version_comparison.png
"""
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from config import RESULTS_DIR

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

# Style 
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "legend.fontsize":   10,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "figure.dpi":        150,
    "lines.linewidth":   1.8,
    "lines.markersize":  5,
    "axes.grid":         True,
    "grid.linestyle":    "--",
    "grid.alpha":        0.4,
})

# Colours (greyscale-friendly) 
COLORS = {
    "pubmedbert":   "#1f77b4",   # steelblue
    "clinicalbert": "#d62728",   # brick red
    "roberta":      "#2ca02c",   # forest green
}
MARKERS = {
    "pubmedbert":   "o",
    "clinicalbert": "s",
    "roberta":      "^",
}
VERSION_COLORS = {
    "v1": "#d62728",   # red  — danger / catastrophic forgetting
    "v2": "#ff7f0e",   # orange — partial improvement
    "v3": "#2ca02c",   # green — success
}
VERSION_LABELS = {
    "v1": "v1 — no replay (lr=2e-5, 150 corr.)",
    "v2": "v2 — replay=50 (lr=2e-5, 150 corr.)",
    "v3": "v3 — all fixes (lr=5e-6, 50 corr., replay=100)",
}

OUT_DIR = RESULTS_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


#  Data loading 

def load_results(filename: str) -> dict:
    path = RESULTS_DIR / filename
    if not path.exists():
        logger.warning("File not found: %s", path)
        return {}
    with open(path) as f:
        return json.load(f)


#  Plot helpers 

def integer_xticks(ax, rounds):
    """Ensure x-axis uses only integer tick marks."""
    max_round = max(rounds)
    step = 2 if max_round > 10 else 1
    ax.set_xticks(range(0, max_round + 1, step))


def add_baseline_line(ax, value, color, label=None):
    """Add a dashed horizontal baseline reference line."""
    ax.axhline(value, color=color, linestyle="--",
               linewidth=1.0, alpha=0.5, label=label)


#  Figure 2: v1 learning curves (catastrophic forgetting) 

def plot_v1_curves(v1: dict) -> None:
    if not v1:
        logger.warning("v1 results not found — skipping Figure 2.")
        return

    # 2a  Accuracy
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model_key, curves in v1.items():
        rounds   = curves["round"]
        accuracy = curves["accuracy"]
        ax.plot(rounds, accuracy,
                color=COLORS.get(model_key, "gray"),
                marker=MARKERS.get(model_key, "o"),
                label=model_key)
        add_baseline_line(ax, accuracy[0], COLORS.get(model_key, "gray"))

    ax.set_xlabel("HITL Round")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Figure 2a — v1: Accuracy Under Catastrophic Forgetting\n"
                 "(lr=2e−5, 150 corrections/round, 2 epochs, no replay buffer)")
    integer_xticks(ax, rounds)
    ax.set_ylim(0.45, 0.92)
    ax.legend(loc="lower right")

    # Annotate the collapse
    ax.annotate("Catastrophic\nforgetting",
                xy=(1, min(v1["pubmedbert"]["accuracy"][1:3])),
                xytext=(4, 0.58),
                arrowprops=dict(arrowstyle="->", color="black", lw=1),
                fontsize=9, color="black")

    fig.tight_layout()
    path = OUT_DIR / "fig2a_v1_accuracy.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", path)

    # 2b — AUC
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model_key, curves in v1.items():
        rounds = curves["round"]
        auc    = curves["auc"]
        ax.plot(rounds, auc,
                color=COLORS.get(model_key, "gray"),
                marker=MARKERS.get(model_key, "o"),
                label=model_key)
        add_baseline_line(ax, auc[0], COLORS.get(model_key, "gray"))

    ax.set_xlabel("HITL Round")
    ax.set_ylabel("AUC")
    ax.set_title("Figure 2b — v1: AUC Under Catastrophic Forgetting")
    integer_xticks(ax, rounds)
    ax.legend(loc="lower right")

    fig.tight_layout()
    path = OUT_DIR / "fig2b_v1_auc.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", path)


# Figure 3: v3 learning curves (optimized) 

def plot_v3_curves(v3: dict) -> None:
    if not v3:
        logger.warning("v3 results not found — skipping Figure 3.")
        return

    # 3a — Accuracy
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model_key, curves in v3.items():
        rounds   = curves["round"]
        accuracy = curves["accuracy"]
        ax.plot(rounds, accuracy,
                color=COLORS.get(model_key, "gray"),
                marker=MARKERS.get(model_key, "o"),
                label=model_key)
        add_baseline_line(ax, accuracy[0], COLORS.get(model_key, "gray"))

    ax.set_xlabel("HITL Round")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Figure 3a — v3: Accuracy Under Optimised HITL\n"
                 "(lr=5e−6, 50 corrections/round, 1 epoch, replay=100)")
    integer_xticks(ax, rounds)
    ax.legend(loc="lower right")

    # 
    cb = v3.get("clinicalbert")
    if cb:
        final_round = cb["round"][-1]
        final_acc   = cb["accuracy"][-1]
        ax.annotate(f"ClinicalBERT +0.015\n(R0→R{final_round})",
                    xy=(final_round, final_acc),
                    xytext=(final_round - 6, final_acc + 0.015),
                    arrowprops=dict(arrowstyle="->", color=COLORS["clinicalbert"], lw=1),
                    fontsize=9, color=COLORS["clinicalbert"])

    fig.tight_layout()
    path = OUT_DIR / "fig3a_v3_accuracy.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", path)

    # 3b — AUC
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model_key, curves in v3.items():
        rounds = curves["round"]
        auc    = curves["auc"]
        ax.plot(rounds, auc,
                color=COLORS.get(model_key, "gray"),
                marker=MARKERS.get(model_key, "o"),
                label=model_key)
        add_baseline_line(ax, auc[0], COLORS.get(model_key, "gray"))

    # Add shaded band for PubMedBERT AUC range to show stability
    pb = v3.get("pubmedbert")
    if pb:
        pb_auc = pb["auc"]
        ax.fill_between(pb["round"],
                        [min(pb_auc)] * len(pb_auc),
                        [max(pb_auc)] * len(pb_auc),
                        alpha=0.08, color=COLORS["pubmedbert"],
                        label=f"PubMedBERT AUC range: "
                              f"{min(pb_auc):.3f}–{max(pb_auc):.3f}")

    ax.set_xlabel("HITL Round")
    ax.set_ylabel("AUC")
    ax.set_title("Figure 3b — v3: AUC Stability Under Optimised HITL")
    integer_xticks(ax, rounds)
    ax.legend(loc="lower right")

    fig.tight_layout()
    path = OUT_DIR / "fig3b_v3_auc.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", path)


#  Figure 4: Three-version comparison  PubMedBERT accuracy 

def plot_version_comparison(v1: dict, v2: dict, v3: dict) -> None:
    model_key = "pubmedbert"
    data = {"v1": v1, "v2": v2, "v3": v3}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: Accuracy
    ax = axes[0]
    for version, results in data.items():
        if not results or model_key not in results:
            continue
        curves = results[model_key]
        rounds = curves["round"]
        acc    = curves["accuracy"]
        ax.plot(rounds, acc,
                color=VERSION_COLORS[version],
                marker="o", markersize=4,
                label=VERSION_LABELS[version])
        add_baseline_line(ax, acc[0], VERSION_COLORS[version])

    ax.set_xlabel("HITL Round")
    ax.set_ylabel("Test Accuracy")
    ax.set_title(f"PubMedBERT — Accuracy Across Configurations")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0.50, 0.90)

    # Add shaded "danger zone" for v1 collapse
    if v1 and model_key in v1:
        ax.axhspan(0.50, 0.65, alpha=0.05, color="red")
        ax.text(0.5, 0.56, "Catastrophic\nforgetting zone",
                fontsize=8, color="red", alpha=0.6)

    # Right: AUC
    ax = axes[1]
    for version, results in data.items():
        if not results or model_key not in results:
            continue
        curves = results[model_key]
        rounds = curves["round"]
        auc    = curves["auc"]
        ax.plot(rounds, auc,
                color=VERSION_COLORS[version],
                marker="o", markersize=4,
                label=VERSION_LABELS[version])
        add_baseline_line(ax, auc[0], VERSION_COLORS[version])

    ax.set_xlabel("HITL Round")
    ax.set_ylabel("AUC")
    ax.set_title(f"PubMedBERT:  AUC Across Configurations")
    ax.legend(loc="lower right", fontsize=8)

    fig.suptitle("Figure 4 : Three-Version HITL Comparison: PubMedBERT",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = OUT_DIR / "fig4_version_comparison_pubmedbert.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", path)


# Figure 5: AUC stability across all models  v1 vs v3 

def plot_auc_version_all(v1: dict, v3: dict) -> None:
    models = ["pubmedbert", "clinicalbert", "roberta"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)

    for ax, model_key in zip(axes, models):
        if v1 and model_key in v1:
            v1_auc = v1[model_key]["auc"]
            ax.plot(v1[model_key]["round"], v1_auc,
                    color=VERSION_COLORS["v1"], marker="o", markersize=3,
                    label="v1 (no replay)", linewidth=1.5)
            ax.fill_between(v1[model_key]["round"],
                            [min(v1_auc)] * len(v1_auc),
                            [max(v1_auc)] * len(v1_auc),
                            alpha=0.08, color=VERSION_COLORS["v1"])

        if v3 and model_key in v3:
            v3_auc = v3[model_key]["auc"]
            ax.plot(v3[model_key]["round"], v3_auc,
                    color=VERSION_COLORS["v3"], marker="s", markersize=3,
                    label="v3 (all fixes)", linewidth=1.5)
            ax.fill_between(v3[model_key]["round"],
                            [min(v3_auc)] * len(v3_auc),
                            [max(v3_auc)] * len(v3_auc),
                            alpha=0.08, color=VERSION_COLORS["v3"])

        ax.set_title(model_key)
        ax.set_xlabel("HITL Round")
        ax.set_ylabel("AUC")
        ax.legend(fontsize=8)

        # Annotate AUC range for v3
        if v3 and model_key in v3:
            v3_auc = v3[model_key]["auc"]
            rng    = max(v3_auc) - min(v3_auc)
            ax.text(0.05, 0.05,
                    f"v3 range: {rng:.4f}",
                    transform=ax.transAxes,
                    fontsize=8, color=VERSION_COLORS["v3"],
                    verticalalignment="bottom")

    fig.suptitle("Figure 5 — AUC Stability: v1 vs v3 Across All HITL Models",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = OUT_DIR / "fig5_auc_version_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", path)


# Main 

def main() -> None:
    logger.info("Loading HITL results ...")

    v1 = load_results("hitl_results_v1_catastrophic_forgetting.json")
    v2 = load_results("hitl_results_v2_replay50.json")
    v3 = load_results("hitl_results.json")

    if not any([v1, v2, v3]):
        logger.error("No HITL result files found in %s", RESULTS_DIR)
        return

    logger.info("Generating Figure 2 — v1 learning curves ...")
    plot_v1_curves(v1)

    logger.info("Generating Figure 3 — v3 learning curves ...")
    plot_v3_curves(v3)

    logger.info("Generating Figure 4 — three-version comparison (PubMedBERT) ...")
    plot_version_comparison(v1, v2, v3)

    logger.info("Generating Figure 5 — AUC stability comparison ...")
    plot_auc_version_all(v1, v3)

    logger.info("All figures saved to %s", OUT_DIR)
    logger.info("Files:")
    for f in sorted(OUT_DIR.glob("fig*.png")):
        logger.info("  %s", f.name)


if __name__ == "__main__":
    main()