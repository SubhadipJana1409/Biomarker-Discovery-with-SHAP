"""
SHAP-based biomarker discovery.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


def compute_shap(
    rf,
    X_test: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, shap.Explanation, float]:
    """
    Compute SHAP values for the IBD (positive) class using TreeExplainer.

    Parameters
    ----------
    rf            : Trained RandomForestClassifier.
    X_test        : Test feature matrix.
    feature_names : Taxon names.

    Returns
    -------
    shap_ibd   : (n_test, n_taxa) SHAP values for IBD class.
    shap_exp   : shap.Explanation object (for waterfall / beeswarm).
    base_value : Baseline prediction (expected value).
    """
    logger.info("Computing SHAP values via TreeExplainer …")
    explainer = shap.TreeExplainer(rf)
    shap_vals = explainer.shap_values(X_test)

    # Handle both old list format and new 3-D tensor format
    if isinstance(shap_vals, list):
        shap_ibd = shap_vals[1]
    elif shap_vals.ndim == 3:
        shap_ibd = shap_vals[:, :, 1]
    else:
        shap_ibd = shap_vals

    base_value = (
        explainer.expected_value[1]
        if hasattr(explainer.expected_value, "__len__")
        else explainer.expected_value
    )

    shap_exp = shap.Explanation(
        values=shap_ibd,
        base_values=np.full(len(X_test), base_value),
        data=X_test,
        feature_names=feature_names,
    )

    logger.info("SHAP computation complete. Base value: %.4f", base_value)
    return shap_ibd, shap_exp, base_value


def build_biomarker_table(
    shap_ibd: np.ndarray,
    feature_names: list[str],
    rf,
    published_effects: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Build a ranked biomarker table from SHAP values.

    Parameters
    ----------
    shap_ibd          : (n_test, n_taxa) SHAP values for IBD class.
    feature_names     : Taxon names.
    rf                : Trained RF (for Gini feature importances).
    published_effects : Optional pd.Series indexed by taxon name with
                        published effect sizes for comparison.

    Returns
    -------
    pd.DataFrame sorted by Mean_SHAP (descending) with columns:
        Rank, Taxon, Mean_SHAP, Median_SHAP, SHAP_Direction,
        RF_Importance, [Published_Effect, Agreement if provided]
    """
    mean_shap = np.abs(shap_ibd).mean(axis=0)
    df = pd.DataFrame(
        {
            "Taxon": feature_names,
            "Mean_SHAP": mean_shap,
            "Median_SHAP": np.median(np.abs(shap_ibd), axis=0),
            "SHAP_Direction": [
                "IBD↑" if shap_ibd[:, i].mean() > 0 else "IBD↓" for i in range(len(feature_names))
            ],
            "RF_Importance": rf.feature_importances_,
            "Pct_Samples_Active": (np.abs(shap_ibd) > 0.001).mean(axis=0) * 100,
        }
    )

    if published_effects is not None:
        df["Published_Effect"] = df["Taxon"].map(published_effects)
        df["Pub_Direction"] = df["Published_Effect"].apply(
            lambda v: "IBD↑" if pd.notna(v) and v > 0 else ("IBD↓" if pd.notna(v) else None)
        )
        df["Agreement"] = df["SHAP_Direction"] == df["Pub_Direction"]
        agree_pct = df["Agreement"].mean() * 100
        logger.info("SHAP ↔ Published direction agreement: %.0f%%", agree_pct)

    df = df.sort_values("Mean_SHAP", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", df.index + 1)
    return df
