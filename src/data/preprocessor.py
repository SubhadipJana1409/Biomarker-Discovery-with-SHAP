"""
Preprocessing utilities for compositional microbiome data.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def clr_transform(X: np.ndarray, pseudo_count: float = 0.5) -> np.ndarray:
    """
    Centered Log-Ratio (CLR) transformation for compositional microbiome data.

    CLR maps raw counts / relative abundances from the simplex to Euclidean
    space where standard distance metrics and ML algorithms are valid.

    Formula: CLR(x_i) = log(x_i / geometric_mean(x))

    Parameters
    ----------
    X            : (n_samples, n_taxa) raw abundance matrix.
    pseudo_count : Added before log to handle zeros (default 0.5).

    Returns
    -------
    np.ndarray of same shape, CLR-transformed.

    References
    ----------
    Aitchison J. (1986). The Statistical Analysis of Compositional Data.
    """
    X_pc  = X + pseudo_count
    log_X = np.log(X_pc)
    return log_X - log_X.mean(axis=1, keepdims=True)


def prevalence_filter(
    X: np.ndarray,
    feature_names: list[str],
    min_prevalence: float = 0.05,
) -> tuple[np.ndarray, list[str]]:
    """
    Remove taxa present in fewer than `min_prevalence` fraction of samples.

    Parameters
    ----------
    X               : (n_samples, n_taxa) abundance matrix.
    feature_names   : List of taxon names (length n_taxa).
    min_prevalence  : Minimum fraction of samples a taxon must appear in.

    Returns
    -------
    X_filtered     : Filtered abundance matrix.
    names_filtered : Corresponding taxon names.
    """
    prevalence = (X > 0).mean(axis=0)
    mask = prevalence >= min_prevalence
    kept = int(mask.sum())
    logger.info(
        "Prevalence filter (>= %.0f%%): %d → %d taxa retained",
        min_prevalence * 100, len(feature_names), kept,
    )
    return X[:, mask], [n for n, m in zip(feature_names, mask) if m]


def prepare_dataset(
    X_raw: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    test_size: float = 0.25,
    pseudo_count: float = 0.5,
    min_prevalence: float = 0.05,
    random_seed: int = 42,
) -> dict:
    """
    Full preprocessing pipeline: prevalence filter → CLR → train/test split.

    Parameters
    ----------
    X_raw         : Raw abundance matrix (n_samples, n_taxa).
    y             : Label array (strings).
    feature_names : Taxon names.
    test_size     : Fraction held out for testing.
    pseudo_count  : CLR pseudo-count.
    min_prevalence: Prevalence filter threshold.
    random_seed   : Random seed.

    Returns
    -------
    dict with keys:
        X_train, X_test, y_train, y_test (numpy arrays),
        X_clr (full CLR matrix),
        y_enc (full encoded labels),
        feature_names (filtered),
        label_encoder.
    """
    # Prevalence filter
    X_filt, feat_filt = prevalence_filter(X_raw, feature_names, min_prevalence)

    # CLR transform
    X_clr = clr_transform(X_filt, pseudo_count)
    logger.info("CLR transform applied (pseudo_count=%.2f)", pseudo_count)

    # Encode labels
    le    = LabelEncoder()
    y_enc = le.fit_transform(y)
    logger.info("Classes: %s → %s", le.classes_.tolist(), [0, 1])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_clr, y_enc, test_size=test_size, stratify=y_enc, random_state=random_seed
    )
    logger.info("Train: %d | Test: %d", len(y_train), len(y_test))

    return {
        "X_train":       X_train,
        "X_test":        X_test,
        "y_train":       y_train,
        "y_test":        y_test,
        "X_clr":         X_clr,
        "y_enc":         y_enc,
        "feature_names": feat_filt,
        "label_encoder": le,
    }
