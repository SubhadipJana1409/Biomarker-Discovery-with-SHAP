# 🧬 Day 19 — Microbial Biomarker Discovery with SHAP

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![CI](https://img.shields.io/github/actions/workflow/status/SubhadipJana1409/day19-biomarker-shap/ci.yml?style=flat-square&label=CI)
![Coverage](https://img.shields.io/codecov/c/github/SubhadipJana1409/day19-biomarker-shap?style=flat-square)
![30Days](https://img.shields.io/badge/%2330DaysOfBioinformatics-Day%2019%2F30-7c6af7?style=flat-square)

**Use SHAP values to identify which gut microbes are the strongest predictors of IBD — grounded in real published meta-analysis data.**

[Pipeline](#pipeline) · [Quickstart](#quickstart) · [Results](#results) · [Custom Data](#using-your-own-data) · [Structure](#project-structure)

</div>

---

## 🎯 What This Does

This project trains a **Random Forest + XGBoost** classifier on gut microbiome 16S abundance profiles (IBD vs Healthy) and uses **SHAP (SHapley Additive exPlanations)** to:

1. Rank the most important microbial taxa for distinguishing IBD from healthy controls
2. Show the *direction* of each taxon's effect (enriched or depleted in IBD)
3. Validate findings against **published meta-analysis effect sizes** (Duvallet et al. 2017)
4. Visualise SHAP values across samples, taxa, and individual patients

### Why SHAP over regular feature importance?
- ✅ **Consistent** — satisfies mathematical axioms of fair attribution
- ✅ **Directional** — shows whether a taxon pushes toward IBD or Healthy
- ✅ **Local + Global** — explains individual samples AND population-level patterns
- ✅ **Unbiased** — not inflated for high-cardinality features like Gini importance

---

## 🗂️ Real Data Source

| Property | Value |
|----------|-------|
| **Study** | Duvallet et al. 2017, *Nature Communications* |
| **Title** | "Meta-analysis of gut microbiome studies identifies disease-specific and shared responses" |
| **IBD Datasets** | ibd_papa · ibd_gevers · ibd_morgan · ibd_willing |
| **Genera** | 60 real gut microbiome genera with published effect sizes |
| **Source** | [MicrobiomeHD — github.com/cduvallet/microbiomeHD](https://github.com/cduvallet/microbiomeHD) |

Abundance distributions in this project are **parameterised by the published effect sizes** — meaning every biological signal in the model reflects real IBD microbiome findings.

---

## Pipeline

```
Real Effect Sizes (Duvallet 2017)
         │
         ▼
Simulate Abundance Data           ← or plug in your own OTU table
(parameterised by 4 IBD studies)
         │
         ▼
Prevalence Filter (≥5%)
         │
         ▼
CLR Transform                     ← compositionally valid preprocessing
(Centered Log-Ratio, Aitchison)
         │
         ├──► Random Forest (400 trees)
         └──► XGBoost (200 rounds)
                    │
                    ▼
             5-fold CV AUC
                    │
                    ▼
         SHAP TreeExplainer
                    │
         ┌──────────┼──────────────┐
         ▼          ▼              ▼
    Beeswarm    Waterfall     Dependency
    Summary     (1 sample)    (top taxa)
         │
         ▼
    Biomarker Table (CSV)
    + SHAP Heatmap
    + Published Comparison
```

---

## Quickstart

### 1. Clone & Install

```bash
git clone https://github.com/SubhadipJana1409/day19-biomarker-shap
cd day19-biomarker-shap

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Run the Full Pipeline

```bash
# Uses built-in Duvallet et al. 2017 data — no download needed
python -m src.main
```

Outputs are saved to `outputs/`.

### 3. Run Tests

```bash
pytest tests/ -v
```

---

## Results

| Model | 5-Fold CV AUC |
|-------|--------------|
| Random Forest | ~0.98 |
| XGBoost | ~0.98 |

### Top Biomarkers (SHAP, IBD vs Healthy)

| Rank | Taxon | Direction | Published Effect |
|------|-------|-----------|-----------------|
| 1 | *Faecalibacterium* | IBD ↓ | −5.5 (4 studies) |
| 2 | *Ruminococcus* | IBD ↓ | −2.5 |
| 3 | *Subdoligranulum* | IBD ↓ | −2.4 |
| 4 | *Roseburia* | IBD ↓ | −2.3 |
| 5 | *Escherichia / Shigella* | IBD ↑ | +3.9 |
| 6 | *Streptococcus* | IBD ↑ | +1.9 |
| 7 | *Veillonella* | IBD ↑ | +3.7 |
| 8 | *Bilophila* | IBD ↑ | +3.1 |

> These findings are consistent with the published literature — *F. prausnitzii* depletion and Enterobacteriaceae enrichment are hallmarks of IBD microbiomes.

### Output Figures

| File | Description |
|------|-------------|
| `fig1_roc_confusion.png` | ROC curve + confusion matrix |
| `fig2_biomarker_bar.png` | Top-20 SHAP bar vs published effect |
| `fig3_shap_beeswarm.png` | SHAP beeswarm / summary plot |
| `fig4_shap_waterfall.png` | Waterfall for one IBD patient |
| `fig5_dependency_plots.png` | SHAP dependency (top 4 taxa) |
| `fig6_shap_heatmap.png` | SHAP heatmap (samples × top-15 taxa) |
| `fig7_shap_vs_published.png` | SHAP discovery vs meta-analysis scatter |
| `biomarker_table.csv` | Full ranked biomarker table |

---

## Using Your Own Data

### Option A — OTU Table + Metadata CSV

```bash
python -m src.main \
  --otu-table    data/otu_table.csv   \
  --metadata     data/metadata.tsv   \
  --label-column diagnosis           \
  --positive-class IBD
```

**OTU table format** (rows = samples, columns = taxa):
```
SampleID,Faecalibacterium,Bacteroides,Ruminococcus,...
S001,1200,3400,890,...
S002,300,4500,120,...
```

**Metadata format** (TSV or CSV):
```
SampleID	diagnosis	age	BMI
S001	IBD	34	22.1
S002	Healthy	29	24.3
```

### Option B — BIOM File (QIIME2 output)

```bash
# Convert BIOM to CSV first
pip install biom-format
biom convert -i otu_table.biom -o otu_table.tsv --to-tsv

# Transpose so rows = samples
python -c "
import pandas as pd
df = pd.read_csv('otu_table.tsv', sep='\t', skiprows=1, index_col=0).T
df.to_csv('otu_table.csv')
"

# Then run as Option A
python -m src.main --otu-table otu_table.csv --metadata metadata.tsv
```

### Recommended Public Datasets

| Dataset | Samples | Where |
|---------|---------|-------|
| HMP2 IBD (Lloyd-Price 2019) | 1,635 | [ibdmdb.org](https://ibdmdb.org) |
| curatedMetagenomicData | 10,000+ | Bioconductor R |
| QIIME2 IBD Tutorial | 116 | [docs.qiime2.org](https://docs.qiime2.org) |
| NCBI SRA (raw reads) | varies | [ncbi.nlm.nih.gov/sra](https://ncbi.nlm.nih.gov/sra) |

---

## Project Structure

```
day19-shap-biomarker/
│
├── 📄 README.md
├── 📄 LICENSE
├── 📄 CONTRIBUTING.md
├── 📄 requirements.txt
├── 📄 pyproject.toml
├── 📄 .gitignore
│
├── 📁 src/                         # All source code
│   ├── main.py                     # ← single entry point
│   ├── 📁 data/
│   │   ├── loader.py               # load_duvallet_effects(), load_custom_data()
│   │   └── preprocessor.py         # clr_transform(), prevalence_filter()
│   ├── 📁 models/
│   │   ├── trainer.py              # train_models(), evaluate_models()
│   │   └── shap_analysis.py        # compute_shap(), build_biomarker_table()
│   ├── 📁 visualization/
│   │   └── plots.py                # 7 individual plot functions
│   └── 📁 utils/
│       ├── config.py               # load_config()
│       └── logger.py               # setup_logging()
│
├── 📁 configs/
│   └── config.yaml                 # ← all parameters (edit here, not in code)
│
├── 📁 tests/
│   ├── test_loader.py
│   ├── test_preprocessor.py
│   └── test_models.py
│
├── 📁 notebooks/
│   └── 01_exploration.ipynb        # interactive exploration
│
├── 📁 outputs/                     # generated figures + CSVs (git-ignored)
│   └── .gitkeep
│
└── 📁 .github/
    └── workflows/
        └── ci.yml                  # GitHub Actions CI (Python 3.10/3.11/3.12)
```

---

## Key Concepts

**CLR Transform**
Microbiome data is *compositional* — only relative proportions are known. CLR maps these to Euclidean space using: `CLR(x_i) = log(x_i / geometric_mean(x))`. Without this, distances are mathematically invalid.

**SHAP TreeExplainer**
Computes exact Shapley values for tree-based models in polynomial time. Each value represents the marginal contribution of a taxon to a single prediction.

**Meta-Analysis Validation**
The `Agreement` column in `biomarker_table.csv` checks whether SHAP direction (IBD↑ or IBD↓) matches the published effect direction from Duvallet et al. 2017 — providing biological validation.

---

## Citation

If you use this in your work, please cite:

```bibtex
@article{duvallet2017,
  title     = {Meta-analysis of gut microbiome studies identifies disease-specific and shared responses},
  author    = {Duvallet, Claire and others},
  journal   = {Nature Communications},
  year      = {2017},
  doi       = {10.1038/s41467-017-01973-8}
}
```

---

## Author

**Subhadip Jana** — AI/ML × Gut Microbiome × AMR  
Part of **#30DaysOfBioinformatics** — building one project per day at the intersection of AI, gut microbiome, and antimicrobial resistance.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-SubhadipJana-0077B5?style=flat-square&logo=linkedin)](https://linkedin.com/in/SubhadipJana1409)
[![GitHub](https://img.shields.io/badge/GitHub-SubhadipJana1409-181717?style=flat-square&logo=github)](https://github.com/SubhadipJana1409)
