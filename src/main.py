"""
main.py — SHAP Biomarker Discovery Pipeline
============================================
Entry point for the full pipeline.

Usage
-----
# Default (built-in Duvallet et al. 2017 data):
    python -m src.main

# Custom OTU table:
    python -m src.main --config configs/config.yaml \\
                       --otu-table data/otu_table.csv \\
                       --metadata data/metadata.tsv  \\
                       --label-column diagnosis       \\
                       --positive-class IBD
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from src.data.loader    import load_duvallet_effects, load_custom_data, simulate_from_effects
from src.data.preprocessor import clr_transform, prevalence_filter, prepare_dataset
from src.models.trainer     import train_models, evaluate_models
from src.models.shap_analysis import compute_shap, build_biomarker_table
from src.visualization.plots import (
    set_style, plot_roc_confusion, plot_biomarker_bar,
    plot_shap_beeswarm, plot_waterfall, plot_dependency,
    plot_shap_heatmap, plot_shap_vs_published,
)
from src.utils.config import load_config
from src.utils.logger import setup_logging

logger = logging.getLogger(__name__)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SHAP Biomarker Discovery Pipeline")
    p.add_argument("--config",         default="configs/config.yaml")
    p.add_argument("--otu-table",      default=None,
                   help="Path to OTU table CSV (rows=samples, cols=taxa)")
    p.add_argument("--metadata",       default=None,
                   help="Path to sample metadata CSV/TSV")
    p.add_argument("--label-column",   default="diagnosis")
    p.add_argument("--positive-class", default="IBD")
    p.add_argument("--out-dir",        default=None,
                   help="Override output directory from config")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg  = load_config(args.config)
    setup_logging(**cfg.get("logging", {}))
    set_style()

    out_dir = Path(args.out_dir or cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", out_dir)

    # ── 1. DATA ──────────────────────────────────────────────────────────────
    published_effects = None

    if args.otu_table and args.metadata:
        logger.info("Loading custom data: %s + %s", args.otu_table, args.metadata)
        otu_df, y_raw = load_custom_data(
            args.otu_table, args.metadata,
            label_column   = args.label_column,
            positive_class = args.positive_class,
        )
        X_raw    = otu_df.values.astype(float)
        y        = y_raw.values
        microbes = otu_df.columns.tolist()
    else:
        logger.info("Using built-in Duvallet et al. 2017 IBD effect sizes")
        eff_df = load_duvallet_effects()
        published_effects = eff_df["mean_ibd_effect"]
        X_raw, y, microbes = simulate_from_effects(
            eff_df,
            n_samples   = cfg["data"]["n_samples"],
            random_seed = cfg["data"]["random_seed"],
        )

    # ── 2. PREPROCESS ────────────────────────────────────────────────────────
    data = prepare_dataset(
        X_raw, y, microbes,
        test_size      = cfg["models"]["test_size"],
        pseudo_count   = cfg["data"]["pseudo_count"],
        min_prevalence = cfg["data"]["prevalence_filter"],
        random_seed    = cfg["data"]["random_seed"],
    )
    feature_names = data["feature_names"]

    # ── 3. TRAIN ─────────────────────────────────────────────────────────────
    trained = train_models(
        data["X_train"], data["y_train"],
        data["X_clr"],   data["y_enc"],
        model_cfg = cfg["models"],
        cv_cfg    = cfg["models"]["cross_validation"],
    )
    rf = trained["rf"]

    # ── 4. EVALUATE ──────────────────────────────────────────────────────────
    results = evaluate_models(rf, data["X_test"], data["y_test"])

    # ── 5. SHAP ──────────────────────────────────────────────────────────────
    shap_ibd, shap_exp, _ = compute_shap(rf, data["X_test"], feature_names)
    biomarker_df = build_biomarker_table(
        shap_ibd, feature_names, rf, published_effects
    )

    logger.info("\n🏆 Top 10 Biomarkers:")
    cols = ["Rank", "Taxon", "Mean_SHAP", "SHAP_Direction"]
    if "Published_Effect" in biomarker_df.columns:
        cols += ["Published_Effect", "Agreement"]
    print(biomarker_df[cols].head(10).to_string(index=False))

    # ── 6. PLOTS ─────────────────────────────────────────────────────────────
    top_taxa = biomarker_df["Taxon"].head(20).tolist()
    ibd_test_idx = int(np.where(data["y_test"] == 1)[0][cfg["shap"]["sample_idx"]])

    plot_roc_confusion(results["fpr"], results["tpr"], results["test_auc"],
                       results["confusion_matrix"], out_dir)
    plot_biomarker_bar(biomarker_df, out_dir)
    plot_shap_beeswarm(shap_ibd, data["X_test"], feature_names, out_dir,
                       cfg["shap"]["max_display"])
    plot_waterfall(shap_exp, ibd_test_idx, out_dir)
    plot_dependency(shap_ibd, data["X_test"], feature_names, top_taxa[:4], out_dir)
    plot_shap_heatmap(shap_ibd, data["y_test"], feature_names, top_taxa, out_dir)
    plot_shap_vs_published(biomarker_df, out_dir)

    # ── 7. EXPORT ────────────────────────────────────────────────────────────
    if cfg["output"]["save_biomarker_table"]:
        csv_path = out_dir / "biomarker_table.csv"
        biomarker_df.to_csv(csv_path, index=False)
        logger.info("Biomarker table saved: %s", csv_path)

    # ── SUMMARY ──────────────────────────────────────────────────────────────
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║  Day 19 — SHAP Biomarker Discovery  ✅  COMPLETE             ║
╠═══════════════════════════════════════════════════════════════╣
║  Genera       : {len(feature_names):<47} ║
║  Samples      : {len(y):<47} ║
║  RF CV AUC    : {trained['rf_cv_auc']:<47.3f} ║
║  XGB CV AUC   : {trained['xgb_cv_auc']:<47.3f} ║
║  Test AUC     : {results['test_auc']:<47.3f} ║
╠═══════════════════════════════════════════════════════════════╣
║  Outputs saved to: {str(out_dir):<43} ║
╚═══════════════════════════════════════════════════════════════╝
""")
    return biomarker_df


if __name__ == "__main__":
    main()
