"""Unit tests for model training and SHAP."""

import pytest
from sklearn.datasets import make_classification

from src.models.shap_analysis import build_biomarker_table, compute_shap
from src.models.trainer import build_random_forest, build_xgboost, evaluate_models


@pytest.fixture
def toy_data():
    X, y = make_classification(n_samples=120, n_features=20, n_informative=8, random_state=42)
    split = 90
    return X[:split], X[split:], y[:split], y[split:]


@pytest.fixture
def trained_rf(toy_data):
    X_tr, _, y_tr, _ = toy_data
    rf = build_random_forest({"n_estimators": 50, "random_state": 42})
    rf.fit(X_tr, y_tr)
    return rf


class TestModelBuilders:
    def test_rf_default(self):
        rf = build_random_forest({})
        assert rf.n_estimators == 400

    def test_xgb_default(self):
        xg = build_xgboost({})
        assert xg.n_estimators == 200


class TestEvaluate:
    def test_auc_range(self, trained_rf, toy_data):
        _, X_te, _, y_te = toy_data
        res = evaluate_models(trained_rf, X_te, y_te)
        assert 0.5 <= res["test_auc"] <= 1.0

    def test_keys(self, trained_rf, toy_data):
        _, X_te, _, y_te = toy_data
        res = evaluate_models(trained_rf, X_te, y_te)
        for k in ("y_pred", "y_prob", "test_auc", "confusion_matrix"):
            assert k in res


class TestSHAP:
    def test_shap_shape(self, trained_rf, toy_data):
        _, X_te, _, _ = toy_data
        names = [f"f{i}" for i in range(X_te.shape[1])]
        shap_ibd, shap_exp, _ = compute_shap(trained_rf, X_te, names)
        assert shap_ibd.shape == X_te.shape

    def test_biomarker_table(self, trained_rf, toy_data):
        _, X_te, _, _ = toy_data
        names = [f"f{i}" for i in range(X_te.shape[1])]
        shap_ibd, _, _ = compute_shap(trained_rf, X_te, names)
        df = build_biomarker_table(shap_ibd, names, trained_rf)
        assert "Rank" in df.columns
        assert "Mean_SHAP" in df.columns
        assert df.iloc[0]["Rank"] == 1
        assert df["Mean_SHAP"].is_monotonic_decreasing
