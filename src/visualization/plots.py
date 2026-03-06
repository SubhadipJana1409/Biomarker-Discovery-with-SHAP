"""
All visualisation functions for the SHAP biomarker pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

logger = logging.getLogger(__name__)

# ── Colour palette ────────────────────────────────────────────────────────────
BG = "#0f0f1a"
AX = "#1a1a2e"
TEXT = "#e8e8f0"
ACC = "#7c6af7"
RED = "#e74c3c"
GRN = "#2ecc71"
GRID = "#2e2e4e"


def set_style() -> None:
    """Apply global dark publication style."""
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": AX,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "legend.facecolor": AX,
            "legend.edgecolor": GRID,
            "grid.color": GRID,
            "grid.alpha": 0.5,
            "font.size": 11,
            "savefig.facecolor": BG,
            "savefig.dpi": 150,
        }
    )


def _save(fig: plt.Figure, path: Path, **kwargs) -> None:
    fig.savefig(path, bbox_inches="tight", facecolor=BG, **kwargs)
    plt.close(fig)
    logger.info("Saved: %s", path)


def plot_roc_confusion(fpr, tpr, test_auc: float, cm: np.ndarray, out_dir: Path) -> None:
    """Fig 1 — ROC curve + confusion matrix."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(BG)

    axes[0].plot(fpr, tpr, color=ACC, lw=2.5, label=f"RF (AUC = {test_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="#555577", lw=1.2)
    axes[0].fill_between(fpr, tpr, alpha=0.12, color=ACC)
    axes[0].set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="ROC Curve — IBD vs Healthy",
    )
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Purples",
        cbar=False,
        xticklabels=["Healthy", "IBD"],
        yticklabels=["Healthy", "IBD"],
        ax=axes[1],
        annot_kws={"size": 14, "weight": "bold"},
    )
    axes[1].set(xlabel="Predicted", ylabel="Actual", title="Confusion Matrix")

    plt.tight_layout()
    _save(fig, out_dir / "fig1_roc_confusion.png")


def plot_biomarker_bar(top_df: pd.DataFrame, out_dir: Path, n: int = 20) -> None:
    """Fig 2 — Top-N SHAP bar chart vs published effect sizes."""
    top = top_df.head(n)
    has_pub = "Published_Effect" in top.columns

    ncols = 2 if has_pub else 1
    fig, axes = plt.subplots(1, ncols, figsize=(10 * ncols, 9))
    fig.patch.set_facecolor(BG)
    if not has_pub:
        axes = [axes]
    fig.suptitle(
        "Top Microbial Biomarkers — SHAP Feature Importance",
        fontsize=14,
        fontweight="bold",
    )

    colors = [RED if d == "IBD↑" else GRN for d in top["SHAP_Direction"]]
    axes[0].barh(
        top["Taxon"][::-1],
        top["Mean_SHAP"][::-1],
        color=colors[::-1],
        edgecolor="#333355",
        linewidth=0.5,
        height=0.7,
    )
    axes[0].set_xlabel("Mean |SHAP Value|")
    axes[0].set_title("SHAP Importance")
    axes[0].grid(axis="x", alpha=0.3)
    axes[0].tick_params(axis="y", labelsize=8)

    if has_pub:
        pub_colors = [RED if v > 0 else GRN for v in top["Published_Effect"].fillna(0)]
        axes[1].barh(
            top["Taxon"][::-1],
            top["Published_Effect"].fillna(0)[::-1],
            color=pub_colors[::-1],
            edgecolor="#333355",
            linewidth=0.5,
            height=0.7,
        )
        axes[1].axvline(0, color=TEXT, lw=1, alpha=0.5)
        axes[1].set_xlabel("Published Effect Size (Duvallet 2017)")
        axes[1].set_title("Published Meta-Analysis Effect")
        axes[1].grid(axis="x", alpha=0.3)
        axes[1].tick_params(axis="y", labelsize=8)

    legend_handles = [
        mpatches.Patch(color=RED, label="Enriched in IBD ↑"),
        mpatches.Patch(color=GRN, label="Depleted in IBD ↓"),
    ]
    axes[0].legend(handles=legend_handles, loc="lower right")
    plt.tight_layout()
    _save(fig, out_dir / "fig2_biomarker_bar.png")


def plot_shap_beeswarm(
    shap_ibd: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
    out_dir: Path,
    max_display: int = 20,
) -> None:
    """Fig 3 — SHAP beeswarm / summary plot."""
    fig, _ = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(BG)
    shap.summary_plot(
        shap_ibd,
        X_test,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
        plot_size=None,
    )
    plt.title(
        "SHAP Beeswarm — Microbial Biomarkers for IBD",
        fontsize=13,
        fontweight="bold",
        color="white",
        pad=10,
    )
    plt.tight_layout()
    _save(fig, out_dir / "fig3_shap_beeswarm.png")


def plot_waterfall(
    shap_exp: shap.Explanation,
    sample_idx: int,
    out_dir: Path,
    max_display: int = 15,
) -> None:
    """Fig 4 — SHAP waterfall for a single sample."""
    fig, _ = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(BG)
    shap.waterfall_plot(shap_exp[sample_idx], max_display=max_display, show=False)
    plt.title(
        f"SHAP Waterfall — Sample #{sample_idx}",
        fontsize=13,
        fontweight="bold",
        color="white",
        pad=10,
    )
    plt.tight_layout()
    _save(fig, out_dir / "fig4_shap_waterfall.png")


def plot_dependency(
    shap_ibd: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
    top_taxa: list[str],
    out_dir: Path,
) -> None:
    """Fig 5 — SHAP dependency plots for top-4 taxa."""
    n = min(4, len(top_taxa))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor(BG)
    fig.suptitle("SHAP Dependency Plots — Top Biomarkers", fontsize=14, fontweight="bold")

    for ax, taxon in zip(axes.flat, top_taxa[:n]):
        idx = feature_names.index(taxon)
        x_ = X_test[:, idx]
        s_ = shap_ibd[:, idx]
        sc = ax.scatter(x_, s_, c=x_, cmap="RdYlGn_r", alpha=0.75, edgecolors="none", s=40)
        ax.axhline(0, color="#555577", linestyle="--", lw=1)
        ax.set_xlabel("CLR Abundance")
        ax.set_ylabel("SHAP Value")
        ax.set_title(taxon.replace("_", " "), style="italic", fontsize=10)
        ax.grid(True, alpha=0.25)
        plt.colorbar(sc, ax=ax, label="Abundance")

    plt.tight_layout()
    _save(fig, out_dir / "fig5_dependency_plots.png")


def plot_shap_heatmap(
    shap_ibd: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    top_taxa: list[str],
    out_dir: Path,
) -> None:
    """Fig 6 — SHAP heatmap: samples × top-15 biomarkers."""
    idx_list = [feature_names.index(t) for t in top_taxa[:15]]
    shap_mat = shap_ibd[:, idx_list]
    short_names = [t.replace("_", " ") for t in top_taxa[:15]]
    sort_order = np.argsort(y_test)
    col_colors = [RED if y == 1 else GRN for y in y_test]

    fig, ax = plt.subplots(figsize=(16, 6))
    fig.patch.set_facecolor(BG)
    sns.heatmap(
        shap_mat[sort_order].T,
        cmap="RdBu_r",
        center=0,
        xticklabels=False,
        yticklabels=short_names,
        ax=ax,
        cbar_kws={"label": "SHAP Value (IBD)"},
        linewidths=0,
    )
    for i, idx in enumerate(sort_order):
        ax.add_patch(plt.Rectangle((i, -1.2), 1, 0.8, color=col_colors[idx], clip_on=False))
    ax.set_title(
        "SHAP Heatmap — Top 15 Biomarkers × All Test Samples",
        fontsize=13,
        fontweight="bold",
        pad=20,
    )
    ax.tick_params(axis="y", labelsize=9)
    ax.legend(
        handles=[
            mpatches.Patch(color=RED, label="IBD"),
            mpatches.Patch(color=GRN, label="Healthy"),
        ],
        loc="upper right",
        bbox_to_anchor=(1.14, 1.0),
    )
    plt.tight_layout()
    _save(fig, out_dir / "fig6_shap_heatmap.png")


def plot_shap_vs_published(biomarker_df: pd.DataFrame, out_dir: Path) -> None:
    """Fig 7 — Scatter: SHAP importance vs published meta-analysis effect size."""
    if "Published_Effect" not in biomarker_df.columns:
        logger.warning("No Published_Effect column — skipping fig7")
        return

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(BG)
    colors = [RED if (pd.notna(v) and v > 0) else GRN for v in biomarker_df["Published_Effect"]]
    ax.scatter(
        biomarker_df["Published_Effect"],
        biomarker_df["Mean_SHAP"],
        c=colors,
        s=60,
        alpha=0.8,
        edgecolors="#555577",
        linewidth=0.5,
    )
    for _, row in biomarker_df.head(10).iterrows():
        ax.annotate(
            row["Taxon"].replace("_", " "),
            (row["Published_Effect"], row["Mean_SHAP"]),
            fontsize=7,
            xytext=(5, 5),
            textcoords="offset points",
            color=TEXT,
            alpha=0.9,
        )
    ax.axvline(0, color="#555577", lw=1, alpha=0.5, linestyle="--")
    ax.set_xlabel("Published Effect Size (Duvallet et al. 2017)", fontsize=11)
    ax.set_ylabel("Mean |SHAP Value|", fontsize=11)
    ax.set_title("SHAP Discovery vs Published Meta-Analysis", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(
        handles=[
            mpatches.Patch(color=RED, label="Enriched in IBD"),
            mpatches.Patch(color=GRN, label="Depleted in IBD"),
        ]
    )
    plt.tight_layout()
    _save(fig, out_dir / "fig7_shap_vs_published.png")
