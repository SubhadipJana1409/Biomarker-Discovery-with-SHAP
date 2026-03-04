"""
Model training and evaluation.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
import xgboost as xgb

logger = logging.getLogger(__name__)


def build_random_forest(cfg: dict) -> RandomForestClassifier:
    """Build a Random Forest from config dict."""
    return RandomForestClassifier(
        n_estimators    = cfg.get("n_estimators", 400),
        max_depth       = cfg.get("max_depth", 8),
        min_samples_leaf= cfg.get("min_samples_leaf", 3),
        n_jobs          = cfg.get("n_jobs", -1),
        random_state    = cfg.get("random_state", 42),
    )


def build_xgboost(cfg: dict) -> xgb.XGBClassifier:
    """Build an XGBoost classifier from config dict."""
    return xgb.XGBClassifier(
        n_estimators     = cfg.get("n_estimators", 200),
        max_depth        = cfg.get("max_depth", 5),
        learning_rate    = cfg.get("learning_rate", 0.05),
        subsample        = cfg.get("subsample", 0.8),
        colsample_bytree = cfg.get("colsample_bytree", 0.8),
        eval_metric      = cfg.get("eval_metric", "logloss"),
        random_state     = cfg.get("random_state", 42),
        verbosity        = 0,
    )


def train_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_clr: np.ndarray,
    y_enc: np.ndarray,
    model_cfg: dict,
    cv_cfg: dict,
) -> dict:
    """
    Train Random Forest and XGBoost, compute cross-validation AUC.

    Returns
    -------
    dict with keys: rf, xgb, rf_cv_auc, xgb_cv_auc
    """
    rf_cfg  = {**model_cfg.get("random_forest", {}), "random_state": 42}
    xgb_cfg = {**model_cfg.get("xgboost", {}),       "random_state": 42}

    rf      = build_random_forest(rf_cfg)
    xgb_clf = build_xgboost(xgb_cfg)

    logger.info("Training Random Forest …")
    rf.fit(X_train, y_train)

    logger.info("Training XGBoost …")
    xgb_clf.fit(X_train, y_train)

    cv = StratifiedKFold(
        n_splits = cv_cfg.get("n_splits", 5),
        shuffle  = cv_cfg.get("shuffle", True),
        random_state = 42,
    )
    rf_cv_auc  = cross_val_score(rf,      X_clr, y_enc, cv=cv, scoring="roc_auc").mean()
    xgb_cv_auc = cross_val_score(xgb_clf, X_clr, y_enc, cv=cv, scoring="roc_auc").mean()

    logger.info("5-Fold CV AUC → RF: %.3f  |  XGBoost: %.3f", rf_cv_auc, xgb_cv_auc)

    return {
        "rf":          rf,
        "xgb":         xgb_clf,
        "rf_cv_auc":   rf_cv_auc,
        "xgb_cv_auc":  xgb_cv_auc,
    }


def evaluate_models(
    rf,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Full evaluation on test set.

    Returns
    -------
    dict with: y_pred, y_prob, test_auc, fpr, tpr, confusion_matrix, report
    """
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    cm     = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Healthy", "IBD"])

    logger.info("Test AUC (RF): %.3f", auc)
    print(report)

    return {
        "y_pred":           y_pred,
        "y_prob":           y_prob,
        "test_auc":         auc,
        "fpr":              fpr,
        "tpr":              tpr,
        "confusion_matrix": cm,
        "report":           report,
    }
