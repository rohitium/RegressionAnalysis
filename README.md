# HIV Drug Resistance Mutation Regression Analysis

Cross-validated OLS/LASSO regression analysis for HIV drug resistance mutations using the Stanford HIVDB GenoPheno datasets.

## Overview

The toolkit quantifies associations between drug resistance mutations (DRMs) and phenotypic resistance across the major HIV drug classes (NRTI, NNRTI, INSTI, PI). Datasets are filtered to samples with Method="PhenoSense" and position columns are detected dynamically (P1, P2, …). It loads curated DRM lists, constructs mutation features, fits regression models with repeated k-fold cross-validation, and produces coefficient visualisations and summary reports. A benchmark mode compares default DRM sets against scored lists over multiple mutation-frequency thresholds.

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install pandas numpy statsmodels scikit-learn matplotlib
```

## Command-line Usage

The primary entry point is `src.main`:

```bash
python -m src.main [OPTIONS]
```

Common options:

| Argument | Description | Default |
|----------|-------------|---------|
| `--min_muts` | Minimum number of sequences required to keep a mutation | `5` |
| `--nfold` | Number of CV folds | `5` |
| `--nrep` | Number of CV repetitions | `10` |
| `--no-lars` | Disable LASSO (OLS always runs) | Enabled by default |
| `--drm_lists PATH...` | Custom DRM CSV files or a directory containing `DRM_*.csv` | `inputs/default/` |
| `--drugs DRUG...` | Restrict to selected drugs (e.g., `3TC ABC LPV`) | All drugs |
| `--no-legend` | Omit legends from coefficient plots | Legends shown by default |
| `--benchmark` | Run the benchmark sweep across DRM sets and `min_muts` values | Off |
| `--paper` | Generate publication-ready grids over preset `min_muts`/mixture values | Off |
| `--mixtures N` | Allow mixtures up to `N` residues (0 excludes mixtures; unset allows all) | All mixtures |

### Examples

Analyze all drugs with default settings:

```bash
python -m src.main
```

Analyze a subset of drugs and disable LASSO:

```bash
python -m src.main --drugs 3TC ABC --no-lars
```

Use custom DRM lists and tighten the mutation filter:

```bash
python -m src.main --drm_lists inputs/custom/ --min_muts 10 --mixtures 3
```

Run the benchmark suite (ignores `--drm_lists`):

```bash
python -m src.main --benchmark
```

Generate publication-ready grids (ignores `--drm_lists`, `--mixtures`, and `--min_muts`):

```bash
python -m src.main --paper
```

## Custom DRM Lists

DRM CSV files must include two columns: `Pos` (integer position) and `Mut` (mutation string). Filenames should follow `DRM_{drug_or_class}.csv` (e.g., `DRM_3TC.csv`, `DRM_NRTI.csv`). When both a drug-specific and a class-level list exist, the drug-specific list takes precedence.

Example (`inputs/custom/DRM_PI.csv`):

```csv
Pos,Mut
10,F
24,I
32,I
```

## Outputs

Results are written into `results/` and plots into `plots/`:

- `results/Regression_results_{CLASS}.csv` &mdash; Combined OLS/LASSO coefficients per drug.
- `results/Regression_results_performance.csv` &mdash; Cross-validation summary for each drug.
- `plots/coefficients_{DRUG}.png` &mdash; Coefficient bar plots (with optional legends).

When `--benchmark` is used, additional files appear:

- `results/Regression_results_benchmarks.csv` &mdash; Long-format MSE comparisons across DRM sets, models, and `min_muts`.
- `plots/benchmarks_mse.png` &mdash; Heatmap visualising mean CV MSE for each scenario.
- `plots/benchmarks_coefficients.png` &mdash; Coefficients for the best-performing setting.

## Repository Structure

- `DRMcv.py` &mdash; Core analysis workflows (`DRMcv` and `run_benchmark`).
- `main.py` &mdash; Command-line interface.
- `data_loading.py`, `data_processing.py`, `models.py`, `results.py`, `plots.py` &mdash; Modular helpers.
- `inputs/default/`, `inputs/drm_with_scores/` &mdash; Reference DRM lists.
- `Agents.md` &mdash; Extended guide to modules and workflows.

## Supported Drugs

- **NRTI**: 3TC, ABC, AZT, TFV
- **NNRTI**: EFV, RPV, DOR
- **INSTI**: DTG, BIC, CAB
- **PI**: LPV, ATV, DRV

## Attribution

Python refactor: Robert Shafer, Rohit Satija  

Original R implementation: Haley Hedlin (2014)
