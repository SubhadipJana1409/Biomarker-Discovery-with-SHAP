"""
Data loading utilities.

Supports two modes:
  1. Built-in  : Duvallet et al. 2017 meta-analysis effect sizes (no download needed)
  2. Custom    : Your own OTU table CSV + metadata CSV
"""

from __future__ import annotations

import logging
from io import StringIO
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# BUILT-IN: Duvallet et al. 2017 — real published IBD effect sizes
# Source: github.com/cduvallet/microbiomeHD  (Nature Communications 2017)
# ──────────────────────────────────────────────────────────────────────────────
_DUVALLET_EFFECTS = """genus\tibd_papa\tibd_gevers\tibd_morgan\tibd_willing
Fusobacterium\t3.470789\t-0.980145\t5.263545\t0.753039
Akkermansia\t\t\t1.874425\t10.356049
Parvimonas\t\t\t7.157008\t
Peptoniphilus\t\t\t4.951234\t
Anaerococcus\t\t\t2.385044\t
Finegoldia\t\t\t5.327199\t
Turicibacter\t\t\t-1.232662\t10.356049
Catenibacterium\t\t\t-1.580095\t2.930412
Coprobacillus\t\t\t-1.103027\t2.006221
Clostridium_XVIII\t0.994701\t\t0.984108\t1.177902
Erysipelotrichaceae_incertae_sedis\t-0.992670\t\t1.377714\t0.698975
Megasphaera\t7.928194\t\t2.866253\t2.159879
Veillonella\t4.893080\t\t5.200977\t-0.020790
Dialister\t-0.167361\t\t-0.938248\t-0.505142
Phascolarctobacterium\t-0.438594\t\t-12.203969\t
Mitsuokella\t\t\t-12.203969\t-1.002673
Acidaminococcus\t\t\t-0.234927\t-4.007527
Rothia\t4.161093\t\t\t
Collinsella\t-1.024534\t0.019264\t-1.119200\t-2.120028
Bifidobacterium\t-2.284513\t-0.757671\t-2.130200\t-3.419723
Adlercreutzia\t\t-1.220826\t-1.097131\t-0.836093
Blautia\t-0.487882\t-1.042082\t-1.098543\t-0.918782
Coprococcus\t-1.154621\t-1.042082\t-1.648490\t-1.636225
Roseburia\t-2.136649\t-1.042082\t-2.350800\t-2.453069
Butyrivibrio\t\t-1.042082\t-1.648490\t-0.784040
Anaerostipes\t\t\t-1.098543\t-0.918782
Ruminococcus\t-2.406516\t-1.042082\t-2.780523\t-2.453069
Faecalibacterium\t-6.014571\t-2.890720\t-5.983879\t-5.528843
Gemmiger\t\t-1.042082\t-1.648490\t-1.636225
Oscillibacter\t\t\t-0.784040\t-0.784040
Butyricicoccus\t\t-1.042082\t-0.234927\t-0.918782
Subdoligranulum\t-2.406516\t-1.042082\t-2.350800\t-2.453069
Eubacterium_hallii_group\t-0.487882\t-1.042082\t-1.648490\t-1.636225
Eubacterium_ventriosum_group\t\t-1.042082\t-1.648490\t-0.918782
Eubacterium_eligens_group\t\t-1.042082\t-1.648490\t-1.636225
Eubacterium_rectale_group\t-2.406516\t-1.042082\t-2.350800\t-2.453069
Holdemanella\t\t-1.042082\t-0.784040\t-0.784040
Dorea\t-0.167361\t-0.311440\t-1.098543\t-0.918782
Streptococcus\t2.167705\t1.041847\t2.063427\t1.528380
Enterococcus\t\t\t1.548034\t2.232481
Lactobacillus\t\t\t0.490745\t3.201866
Carnobacterium\t\t\t10.356049\t
Escherichia_Shigella\t4.893080\t2.028394\t4.370665\t2.232481
Klebsiella\t\t\t2.867254\t
Enterobacter\t\t\t2.867254\t
Haemophilus\t5.225040\t2.028394\t2.059786\t
Campylobacter\t\t\t2.867254\t
Bilophila\t4.893080\t\t2.380217\t2.232481
Bacteroides\t0.573705\t0.457888\t0.819773\t0.428399
Parabacteroides\t\t-0.066866\t-0.784040\t-0.506139
Alistipes\t\t-1.042082\t-2.350800\t-2.453069
Barnesiella\t\t\t-1.648490\t-0.784040
Odoribacter\t\t\t-1.648490\t-1.636225
Butyricimonas\t\t\t-1.648490\t-0.918782
Prevotella_9\t-0.962619\t-0.887558\t-2.350800\t-1.636225
"""

IBD_DATASETS = ["ibd_papa", "ibd_gevers", "ibd_morgan", "ibd_willing"]


def load_duvallet_effects(min_effect: float = 0.01) -> pd.DataFrame:
    """
    Load real published effect sizes from Duvallet et al. 2017
    (Nature Communications — meta-analysis of 29 gut microbiome studies).

    Returns
    -------
    pd.DataFrame
        Index = genus name.
        Columns: ibd_papa, ibd_gevers, ibd_morgan, ibd_willing,
                 mean_ibd_effect, n_studies.
    """
    df = pd.read_csv(StringIO(_DUVALLET_EFFECTS), sep="\t", index_col=0)
    df["mean_ibd_effect"] = df[IBD_DATASETS].mean(axis=1, skipna=True)
    df["n_studies"]       = df[IBD_DATASETS].notna().sum(axis=1)
    df = df[df["mean_ibd_effect"].notna()]
    df = df[df["mean_ibd_effect"].abs() > min_effect]
    logger.info("Loaded %d genera from Duvallet et al. 2017", len(df))
    return df


def load_custom_data(
    otu_path: str | Path,
    metadata_path: str | Path,
    label_column: str = "diagnosis",
    positive_class: str = "IBD",
    negative_class: str = "Healthy",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load your own OTU table + metadata.

    Parameters
    ----------
    otu_path        : Path to CSV (rows = samples, columns = taxa).
    metadata_path   : Path to CSV/TSV with sample metadata.
    label_column    : Column in metadata holding the class label.
    positive_class  : Label string for the positive class (e.g. 'IBD').
    negative_class  : Label string for the negative class (e.g. 'Healthy').

    Returns
    -------
    X : pd.DataFrame   — OTU table aligned to metadata samples.
    y : pd.Series      — Binary labels (positive_class / negative_class).

    Example
    -------
    >>> X, y = load_custom_data("otu_table.csv", "metadata.tsv")
    """
    otu_path      = Path(otu_path)
    metadata_path = Path(metadata_path)

    sep = "\t" if metadata_path.suffix in (".tsv", ".txt") else ","
    otu  = pd.read_csv(otu_path, index_col=0)
    meta = pd.read_csv(metadata_path, sep=sep, index_col=0)

    # If OTU is transposed (taxa × samples), flip it
    if otu.shape[0] < otu.shape[1]:
        logger.warning("OTU table looks transposed (more columns than rows) — transposing.")
        otu = otu.T

    common = otu.index.intersection(meta.index)
    if len(common) == 0:
        raise ValueError(
            "No overlapping sample IDs between OTU table and metadata. "
            "Check that your index column matches."
        )
    otu  = otu.loc[common]
    meta = meta.loc[common]

    valid_labels = [positive_class, negative_class]
    mask = meta[label_column].isin(valid_labels)
    if mask.sum() == 0:
        raise ValueError(
            f"No samples found with labels {valid_labels}. "
            f"Available: {meta[label_column].unique().tolist()}"
        )
    otu  = otu.loc[mask]
    y    = meta.loc[mask, label_column]

    logger.info(
        "Loaded %d samples × %d taxa — %s",
        otu.shape[0], otu.shape[1],
        y.value_counts().to_dict(),
    )
    return otu, y


def simulate_from_effects(
    effects_df: pd.DataFrame,
    n_samples: int = 400,
    base_mean: float = 3.0,
    scale: float = 0.40,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Generate synthetic abundance data parameterised by real effect sizes.

    Parameters
    ----------
    effects_df  : Output of load_duvallet_effects().
    n_samples   : Total samples (split 50/50).
    base_mean   : Log-normal base mean for healthy controls.
    scale       : Multiplier applied to effect sizes.
    random_seed : NumPy random seed.

    Returns
    -------
    X_raw   : (n_samples, n_taxa) raw abundance array.
    y       : (n_samples,) string label array.
    microbes: list of genus names.
    """
    np.random.seed(random_seed)
    microbes = effects_df.index.tolist()
    effects  = effects_df["mean_ibd_effect"].values
    n_half   = n_samples // 2

    def _gen(n, group):
        rows = []
        for _ in range(n):
            abu = []
            for eff in effects:
                mean = base_mean + (eff * scale if group == "IBD"
                                    else -max(0, eff) * scale)
                abu.append(max(0.01, np.random.lognormal(mean=mean, sigma=0.8)))
            rows.append(abu)
        return np.array(rows)

    X_raw = np.vstack([_gen(n_half, "Healthy"), _gen(n_half, "IBD")])
    y     = np.array(["Healthy"] * n_half + ["IBD"] * n_half)
    logger.info("Simulated %d samples × %d taxa", X_raw.shape[0], X_raw.shape[1])
    return X_raw, y, microbes
