"""Unit tests for data preprocessing."""

import numpy as np
import pytest

from src.data.preprocessor import clr_transform, prevalence_filter, prepare_dataset


class TestCLR:
    def test_shape_preserved(self):
        X = np.random.lognormal(size=(20, 10)) + 0.01
        assert clr_transform(X).shape == X.shape

    def test_row_sums_zero(self):
        X = np.random.lognormal(size=(20, 10)) + 0.01
        result = clr_transform(X)
        np.testing.assert_allclose(result.sum(axis=1), 0, atol=1e-10)

    def test_handles_zeros(self):
        X = np.zeros((5, 8))
        result = clr_transform(X, pseudo_count=0.5)
        assert np.all(np.isfinite(result))


class TestPrevalenceFilter:
    def test_removes_rare_taxa(self):
        X = np.zeros((100, 5))
        X[:, :3] = 1.0          # first 3 taxa present in all samples
        X[:4, 3] = 1.0          # taxon 3 present in only 4%
        names = [f"t{i}" for i in range(5)]
        X_filt, names_filt = prevalence_filter(X, names, min_prevalence=0.05)
        assert X_filt.shape[1] == 3
        assert "t3" not in names_filt
        assert "t4" not in names_filt

    def test_keeps_all_above_threshold(self):
        X = np.ones((50, 4))
        names = ["a", "b", "c", "d"]
        X_filt, names_filt = prevalence_filter(X, names, min_prevalence=0.10)
        assert X_filt.shape[1] == 4


class TestPrepareDataset:
    def test_output_keys(self):
        X = np.random.lognormal(size=(80, 15)) + 0.1
        y = np.array(["Healthy"] * 40 + ["IBD"] * 40)
        names = [f"taxon_{i}" for i in range(15)]
        result = prepare_dataset(X, y, names, test_size=0.25)
        for key in ("X_train", "X_test", "y_train", "y_test",
                    "X_clr", "y_enc", "feature_names", "label_encoder"):
            assert key in result

    def test_split_sizes(self):
        X = np.random.lognormal(size=(100, 10)) + 0.1
        y = np.array(["Healthy"] * 50 + ["IBD"] * 50)
        names = [f"g{i}" for i in range(10)]
        result = prepare_dataset(X, y, names, test_size=0.25)
        assert result["X_train"].shape[0] == 75
        assert result["X_test"].shape[0] == 25

    def test_labels_binary(self):
        X = np.random.lognormal(size=(60, 8)) + 0.1
        y = np.array(["Healthy"] * 30 + ["IBD"] * 30)
        names = [f"g{i}" for i in range(8)]
        result = prepare_dataset(X, y, names)
        assert set(result["y_enc"]) == {0, 1}
