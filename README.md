# HIV Drug Resistance Mutation Regression Analysis

Cross-validated OLS regression analysis for HIV drug resistance mutations using Stanford HIVDB GenoPhenoDatasets.

## Overview

This pipeline analyzes the association between drug resistance mutations (DRMs) and phenotypic drug resistance across HIV drug classes (NRTI, NNRTI, INSTI, PI). It automatically downloads genotype-phenotype data from Stanford HIVDB, fits linear regression models for each drug, and generates coefficient plots with error bars.

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install pandas numpy statsmodels scikit-learn matplotlib
```

## Usage

### Basic usage (analyze all drugs with default DRM lists):
```bash
python RegressionAnalysis.py
```

### Analyze specific drugs:
```bash
python RegressionAnalysis.py --drugs 3TC ABC LPV DRV
```

### Use custom DRM lists:
```bash
python RegressionAnalysis.py --drm_lists inputs/DRM_PI.csv inputs/DRM_NRTI.csv
```

### Adjust parameters:
```bash
python RegressionAnalysis.py --min_muts 5 --nfold 10 --nrep 20 --lars
```

## Command-line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--min_muts` | Maximum minimum mutations threshold | 10 |
| `--nfold` | Number of cross-validation folds | 5 |
| `--nrep` | Number of CV repetitions | 10 |
| `--lars` | Run LASSO regression | False |
| `--drm_lists` | Paths to custom DRM CSV files | Default lists |
| `--drugs` | Specific drugs to analyze | All drugs |

## Custom DRM Lists

Create CSV files with two columns: `Pos` and `Mut`

Example (`DRM_PI.csv`):
```csv
Pos,Mut
10,F
24,I
32,I
```

Filename must follow pattern: `DRM_{drug_class}.csv` (e.g., `DRM_NRTI.csv`, `DRM_PI.csv`)

## Output

### Results folder (`results/`):
- `OLScoefs_{drug_class}.txt`: Coefficients and standard errors
- `CVmse_{drug_class}.txt`: Cross-validation MSE scores
- `LARScoef_{drug_class}.txt`: LASSO coefficients (if `--lars` enabled)

### Plots folder (`plots/`):
- `coefficients_{drug_class}.png`: Coefficient bar plots with error bars

## Supported Drugs

- **NRTI**: 3TC, ABC, AZT, TFV
- **NNRTI**: EFV, RPV
- **INSTI**: DTG, BIC, CAB
- **PI**: LPV, ATV, DRV

## Authors

Python implementation: ChatGPT for Robert Shafer, Claude for Rohit Satija

Original R version: Haley Hedlin (2014)
