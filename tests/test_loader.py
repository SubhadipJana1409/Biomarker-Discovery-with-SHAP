"""Unit tests for data loading."""

import numpy as np
import pandas as pd

from src.data.loader import load_duvallet_effects, simulate_from_effects


class TestDuvalletLoader:
    def test_returns_dataframe(self):
        df = load_duvallet_effects()
        assert isinstance(df, pd.DataFrame)

    def test_has_ibd_columns(self):
        df = load_duvallet_effects()
        for col in [
            "ibd_papa",
            "ibd_gevers",
            "ibd_morgan",
            "ibd_willing",
            "mean_ibd_effect",
            "n_studies",
        ]:
            assert col in df.columns

    def test_minimum_genera(self):
        df = load_duvallet_effects()
        assert len(df) >= 30

    def test_faecalibacterium_present(self):
        df = load_duvallet_effects()
        assert "Faecalibacterium" in df.index

    def test_faecalibacterium_depleted_in_ibd(self):
        """F. prausnitzii is consistently depleted in IBD — known biology."""
        df = load_duvallet_effects()
        assert df.loc["Faecalibacterium", "mean_ibd_effect"] < 0


class TestSimulateFromEffects:
    def test_output_shapes(self):
        df = load_duvallet_effects()
        X, y, names = simulate_from_effects(df, n_samples=100)
        assert X.shape == (100, len(names))
        assert len(y) == 100

    def test_balanced_classes(self):
        df = load_duvallet_effects()
        _, y, _ = simulate_from_effects(df, n_samples=100)
        counts = pd.Series(y).value_counts()
        assert counts["IBD"] == 50
        assert counts["Healthy"] == 50

    def test_all_positive(self):
        df = load_duvallet_effects()
        X, _, _ = simulate_from_effects(df, n_samples=40)
        assert np.all(X >= 0)
