"""
Helper functions for HIVDB regression analysis.

Provides utilities for:
- Data loading and preprocessing
- Mutation parsing and validation
- OLS and LASSO regression modeling
- Cross-validation
- Results plotting and saving
"""

import os
import re
from typing import List, Tuple, Dict, Set

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, LassoCV, Lasso
from sklearn.model_selection import RepeatedKFold, cross_val_score

from constants import (
    DATA_URLS,
    POSITION_SLICES,
    INSERTION_KEYWORDS,
    DELETION_KEYWORDS,
)


def _normalize_mutation_symbol(symbol: str) -> str:
    """Normalize mutation suffix while respecting uppercase insertion/deletion."""
    raw = symbol.strip()
    if not raw or raw == '.':
        return ''

    lower = raw.lower()
    if lower in INSERTION_KEYWORDS and not raw.isupper():
        return '#'
    if lower in DELETION_KEYWORDS and not raw.isupper():
        return '~'

    return re.sub(r'\d', '', raw).upper()


def _expand_mutation_targets(mutation: str) -> Set[str]:
    """Return the set of residue codes considered a match for a DRM."""
    if not mutation:
        return set()

    code = mutation.strip()
    lower = code.lower()

    if code == '#':
        return {'#'}
    if code == '~':
        return {'~'}
    if lower in INSERTION_KEYWORDS and not code.isupper():
        return {'#'}
    if lower in DELETION_KEYWORDS and not code.isupper():
        return {'~'}

    return {ch for ch in code.upper() if ch.isalpha() or ch in {'#', '~'}}


def _tokenize_sequence_value(value: str) -> Set[str]:
    """Convert a sequence string into normalized residue tokens."""
    if value is None:
        return set()
    raw = str(value).strip()
    if not raw or raw == '.':
        return set()

    lower = raw.lower()
    if lower in INSERTION_KEYWORDS and not raw.isupper():
        return {'#'}
    if lower in DELETION_KEYWORDS and not raw.isupper():
        return {'~'}

    return {ch for ch in raw.upper() if ch.isalpha() or ch in {'#', '~'}}


# ============================================================================
# Mutation parsing and validation
# ============================================================================

def convert_muts(muts_in: List[str]) -> List[str]:
    """Convert insertion/deletion suffixes to standard markers.

    Converts: 'ins'/'i' → '#', 'del'/'d' → '~'
    Examples: '69ins' → '69#', '70del' → '70~'
    """
    converted = []
    for mut in muts_in:
        mut = mut.strip()
        if not mut:
            continue
        match = re.match(r'^(\d{2,3})(.+)$', mut)
        if not match:
            raise ValueError(f"Mutation '{mut}' must start with 2-3 digit position.")
        pos, symbol = match.groups()
        canonical = _normalize_mutation_symbol(symbol)
        if not canonical:
            raise ValueError(f"Mutation '{mut}' does not specify a valid residue.")
        converted.append(f"{int(pos)}{canonical}")
    return converted


def check_muts(muts_in: List[str]) -> None:
    """Validate mutation format: 2-3 digits + amino acid letter or #/~.

    Raises ValueError if any mutation doesn't match pattern.
    """
    pattern = re.compile(r'^\d{2,3}[A-Z#~]{1,8}$', re.IGNORECASE)
    if not all(pattern.match(m.strip()) for m in muts_in):
        raise ValueError('Each mutation must be 2-3 digits followed by 1-8 letters, #, or ~.')


def parse_mut_tokens(muts_in: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Extract position and amino acid from mutation strings.

    Args:
        muts_in: List of mutation strings (e.g., ['98G', '184V', '69#'])

    Returns:
        Tuple of (positions array, amino acids array)
    """
    pattern = re.compile(r'^(\d{2,3})([A-Z#~]{1,8})$', re.IGNORECASE)
    parsed = [pattern.match(m.strip()) for m in muts_in]
    pos = np.array([int(m.group(1)) for m in parsed], dtype=int)
    mut = np.array([m.group(2).upper() for m in parsed], dtype=object)
    return pos, mut


# ============================================================================
# Data loading
# ============================================================================

def load_dataset(drug_class: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load dataset from Stanford HIVDB and extract position columns.

    Args:
        drug_class: One of 'NRTI', 'NNRTI', 'INSTI', 'PI'

    Returns:
        Tuple of (full_dataframe, positions_only_dataframe)
        - full_dataframe: Complete dataset with metadata and drug phenotypes
        - positions_only_dataframe: Sequence positions only, columns renamed to 1, 2, 3, ...
    """
    # Load dataset from URL
    df = pd.read_csv(DATA_URLS[drug_class], sep="\t", dtype=str,
                     comment='@', engine='python')

    # Extract position columns and renumber them 1..N for easy position lookup
    start, end = POSITION_SLICES[drug_class]
    positions_df = df.iloc[:, start:end]
    positions_df.columns = np.arange(1, positions_df.shape[1] + 1)

    return df, positions_df


def load_drm_lists_from_csv(csv_paths: List[str]) -> Dict[str, List[str]]:
    """Load DRM lists from CSV files or directory.

    Args:
        csv_paths: List of CSV file paths OR single directory path
                  Files must have 'Pos' and 'Mut' columns
                  Filenames should follow pattern: DRM_{drug_or_class}.csv

    Returns:
        Dictionary mapping drug/class name to list of mutations
        Example: {'3TC': ['41L', '184V'], 'NRTI': ['41L', '65R', ...]}
    """
    # Handle directory input - find all DRM_*.csv files
    if len(csv_paths) == 1 and os.path.isdir(csv_paths[0]):
        directory = csv_paths[0]
        csv_files = [
            os.path.join(directory, f) for f in os.listdir(directory)
            if f.startswith('DRM_') and f.endswith('.csv')
        ]
        if not csv_files:
            raise ValueError(f"No DRM_*.csv files found in directory {directory}")
    else:
        csv_files = csv_paths

    # Load each CSV file
    drm_dict = {}
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if 'Pos' not in df.columns or 'Mut' not in df.columns:
            raise ValueError(f"CSV file {csv_file} must have 'Pos' and 'Mut' columns")

        # Extract drug or class name from filename (DRM_3TC.csv → 3TC)
        basename = os.path.basename(csv_file)
        if not (basename.startswith('DRM_') and basename.endswith('.csv')):
            raise ValueError(f"CSV filename {csv_file} must follow pattern DRM_{{name}}.csv")

        drug_or_class = basename[4:-4]  # Remove 'DRM_' prefix and '.csv' suffix

        # Combine Pos and Mut into mutation strings (e.g., "41L", "184V")
        mutations = [f"{row['Pos']}{row['Mut']}" for _, row in df.iterrows()]
        drm_dict[drug_or_class] = mutations

    return drm_dict


# ============================================================================
# Feature matrix construction
# ============================================================================

def build_X(positions_df: pd.DataFrame, mut_positions: np.ndarray,
            mut_aas: np.ndarray) -> pd.DataFrame:
    """Build binary feature matrix indicating presence of mutations.

    For each sequence and mutation, set 1 if any of the target residues (or
    insertion/deletion markers) are present at that position, else 0.

    Args:
        positions_df: DataFrame with position columns (1, 2, 3, ...)
        mut_positions: Array of mutation positions to check
        mut_aas: Array of amino acids corresponding to each position

    Returns:
        Binary feature matrix with columns named like "41L", "184V", etc.
    """
    n_samples = positions_df.shape[0]
    n_mutations = len(mut_positions)
    X = np.zeros((n_samples, n_mutations), dtype=float)

    max_pos = positions_df.shape[1]  # Maximum position available in dataset

    # Pre-tokenize observed mutations for each unique position (for efficiency)
    mutation_cache = {}
    for pos in np.unique(mut_positions):
        # Skip positions beyond dataset range
        if pos > max_pos:
            continue
        col = positions_df.iloc[:, pos - 1]
        tokenized = col.fillna('').astype(str).apply(_tokenize_sequence_value)
        mutation_cache[pos] = tokenized.tolist()

    # Build feature matrix
    for j, (pos, aa) in enumerate(zip(mut_positions, mut_aas)):
        if pos in mutation_cache:  # Skip if position out of bounds
            observed_tokens = mutation_cache[pos]
            targets = _expand_mutation_targets(aa)
            X[:, j] = np.array([
                1.0 if tokens and (tokens & targets) else 0.0
                for tokens in observed_tokens
            ], dtype=float)

    # Create column names (e.g., "41L", "184V")
    colnames = [f"{pos}{aa}" for pos, aa in zip(mut_positions, mut_aas)]
    return pd.DataFrame(X, columns=colnames, index=positions_df.index)


# ============================================================================
# Modeling functions
# ============================================================================

def fit_ols_model(df_drug: pd.DataFrame) -> pd.DataFrame:
    """Fit OLS regression model.

    Args:
        df_drug: DataFrame with 'Y' column (outcome) and mutation columns (features)

    Returns:
        DataFrame with columns ['coef', 'se'] and index ['Intercept', mutation names]
    """
    y = df_drug["Y"].values
    X = sm.add_constant(df_drug.drop(columns=["Y"]).values)
    model = sm.OLS(y, X).fit()

    coef_names = ["Intercept"] + df_drug.columns.drop("Y").tolist()
    return pd.DataFrame({"coef": model.params, "se": model.bse}, index=coef_names)


def cross_validate_model(df_drug: pd.DataFrame, nfold: int = 5,
                         nrep: int = 10) -> np.ndarray:
    """Perform repeated k-fold cross-validation.

    Args:
        df_drug: DataFrame with 'Y' column (outcome) and mutation columns (features)
        nfold: Number of cross-validation folds
        nrep: Number of repetitions

    Returns:
        Array of MSE scores from each fold
    """
    y = df_drug["Y"].values
    X = df_drug.drop(columns=["Y"]).values

    cv = RepeatedKFold(n_splits=nfold, n_repeats=nrep, random_state=123)
    mse_scores = -cross_val_score(
        LinearRegression(), X, y,
        scoring='neg_mean_squared_error', cv=cv
    )
    return mse_scores


def fit_lasso_model(df_drug: pd.DataFrame, nfold: int = 5, nrep: int = 10) -> Tuple[pd.DataFrame, np.ndarray]:
    """Fit LASSO regression with cross-validated regularization and compute MSE.

    Args:
        df_drug: DataFrame with 'Y' column (outcome) and mutation columns (features)
        nfold: Number of cross-validation folds
        nrep: Number of CV repetitions

    Returns:
        Tuple of (coefficients DataFrame, MSE scores array)
    """
    y = df_drug["Y"].values
    X = df_drug.drop(columns=["Y"]).values

    # Fit LASSO with cross-validation to find best alpha
    model = LassoCV(cv=5, random_state=123, alphas=100, max_iter=10000)
    model.fit(X, y)

    # Combine intercept and coefficients
    coef = np.r_[model.intercept_, model.coef_]
    names = ["Intercept"] + df_drug.columns.drop("Y").tolist()
    coef_df = pd.DataFrame({"mutation": names, "coef": coef})

    # Refit on repeated CV splits to estimate coefficient variability and MSE
    cv = RepeatedKFold(n_splits=nfold, n_repeats=nrep, random_state=123)
    coef_samples = []
    mse_scores = []

    for train_idx, test_idx in cv.split(X):
        lasso = Lasso(alpha=model.alpha_, max_iter=10000)
        lasso.fit(X[train_idx], y[train_idx])
        coef_samples.append(np.r_[lasso.intercept_, lasso.coef_])

        preds = lasso.predict(X[test_idx])
        mse_scores.append(np.mean((y[test_idx] - preds) ** 2))

    coef_samples = np.array(coef_samples)
    if coef_samples.shape[0] > 1:
        coef_se = coef_samples.std(axis=0, ddof=1)
    else:
        coef_se = np.zeros_like(coef)

    coef_df['se'] = coef_se

    return coef_df, np.array(mse_scores)


# ============================================================================
# Plotting and saving results
# ============================================================================

def plot_drug_coefficients(drug: str, coef_data: pd.DataFrame,
                           ols_mse: float = None, lars_mse: float = None,
                           output_dir: str = "plots") -> str:
    """Create bar plot comparing OLS and LASSO coefficients side-by-side.

    Args:
        drug: Drug name (used for title and filename)
        coef_data: DataFrame with columns ['mutation', 'OLS_coef', 'OLS_se', 'LARS_coef'] (no Intercept)
        ols_mse: Mean OLS CV MSE (for legend)
        lars_mse: Mean LASSO CV MSE (for legend)
        output_dir: Directory to save plot

    Returns:
        Path to saved plot file
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract plot data
    mutations = coef_data['mutation'].tolist()
    ols_coef = coef_data['OLS_coef'].values
    ols_se = coef_data['OLS_se'].values

    # Check if LASSO coefficients exist
    has_lars = 'LARS_coef' in coef_data.columns
    lars_err = None
    if has_lars:
        lars_coef = coef_data['LARS_coef'].values
        if 'LARS_se' in coef_data.columns:
            lars_err = coef_data['LARS_se'].values

    # Calculate y-axis limits with 10% padding
    upper_bounds = ols_coef + ols_se
    lower_bounds = ols_coef - ols_se
    if has_lars:
        if lars_err is not None:
            upper_bounds = np.maximum(upper_bounds, lars_coef + lars_err)
            lower_bounds = np.minimum(lower_bounds, lars_coef - lars_err)
        else:
            upper_bounds = np.maximum(upper_bounds, lars_coef)
            lower_bounds = np.minimum(lower_bounds, lars_coef)

    y_max = np.max(upper_bounds)
    y_min = np.min(lower_bounds)
    y_range = y_max - y_min
    y_limits = (y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 4))

    # Set up bar positions
    x_pos = np.arange(len(mutations))
    if has_lars:
        width = 0.35
        x_ols = x_pos - width/2
        x_lars = x_pos + width/2
    else:
        width = 0.7
        x_ols = x_pos

    # Plot OLS bars
    ax.bar(x_ols, ols_coef, width, color='steelblue', edgecolor='black',
           linewidth=0.5, yerr=ols_se, capsize=3,
           error_kw={'linewidth': 1, 'ecolor': 'black'},
           label='OLS')

    # Plot LASSO bars if available
    if has_lars:
        ax.bar(
            x_lars, lars_coef, width, color='firebrick', edgecolor='black',
            linewidth=0.5, yerr=lars_err, capsize=3 if lars_err is not None else 0,
            error_kw={'linewidth': 1, 'ecolor': 'black'} if lars_err is not None else None,
            label='LASSO'
        )

    # Styling
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_ylabel('Coefficient', fontsize=10)
    ax.set_title(drug, fontsize=12, fontweight='bold')
    ax.set_ylim(y_limits)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(mutations, rotation=90)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.tick_params(axis='both', labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Add legend with MSE values
    legend_labels = []
    if ols_mse is not None:
        legend_labels.append(f'OLS (CV MSE: {ols_mse:.4f})')
    else:
        legend_labels.append('OLS')

    if has_lars:
        if lars_mse is not None:
            legend_labels.append(f'LASSO (CV MSE: {lars_mse:.4f})')
        else:
            legend_labels.append('LASSO')

    handles, _ = ax.get_legend_handles_labels()
    ax.legend(
        handles, legend_labels,
        loc='upper left', bbox_to_anchor=(1.01, 1),
        fontsize=8, framealpha=0.9, borderaxespad=0
    )

    # Save
    fig.tight_layout(rect=[0, 0, 0.88, 1])
    plot_path = os.path.join(output_dir, f"coefficients_{drug}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    return plot_path


def prepare_drug_class_results(
    drug_class: str,
    ols_coefs: List[pd.DataFrame],
    lars_coefs: List[pd.DataFrame] = None
) -> pd.DataFrame:
    """Assemble regression coefficients into a single DataFrame per drug class."""
    # Combine and process OLS coefficients (exclude Intercept)
    combined_ols = pd.concat(ols_coefs, ignore_index=True)
    combined_ols = combined_ols[combined_ols['mutation'] != 'Intercept'].copy()

    # Extract position and mutation from strings like "98G", "69#"
    combined_ols['Pos'] = combined_ols['mutation'].str.extract(r'^(\d+)')[0].astype(int)
    combined_ols['Mut'] = combined_ols['mutation'].str.extract(r'(\D+)$')[0]
    combined_ols['Mutation'] = combined_ols['mutation']
    combined_ols = combined_ols.rename(columns={'coef': 'OLS_coef', 'se': 'OLS_se', 'drug': 'Drug'})

    # Start with core columns
    result = combined_ols[['Pos', 'Mut', 'Mutation', 'Drug', 'OLS_coef', 'OLS_se']].copy()

    # Add LARS coefficients if provided
    if lars_coefs:
        combined_lars = pd.concat(lars_coefs, ignore_index=True)
        combined_lars = combined_lars[combined_lars['mutation'] != 'Intercept'].copy()
        combined_lars = combined_lars.rename(
            columns={'coef': 'LARS_coef', 'se': 'LARS_se', 'drug': 'Drug'}
        )
        combined_lars = combined_lars.assign(Mutation=combined_lars['mutation'])
        result = result.merge(
            combined_lars[['Drug', 'Mutation', 'LARS_coef', 'LARS_se']],
            on=['Drug', 'Mutation'],
            how='left'
        )

    # Reorder columns for readability
    ordered_cols = ['Pos', 'Mut', 'Mutation', 'Drug', 'OLS_coef', 'OLS_se']
    if 'LARS_coef' in result.columns:
        ordered_cols.append('LARS_coef')
    if 'LARS_se' in result.columns:
        ordered_cols.append('LARS_se')
    result = result[ordered_cols]

    # Sort and save
    result = result.sort_values(['Drug', 'Pos'])
    result['Drug_class'] = drug_class
    return result


def save_drug_class_results(
    drug_class: str,
    ols_coefs: List[pd.DataFrame],
    lars_coefs: List[pd.DataFrame] = None,
    output_dir: str = "results",
    precomputed: pd.DataFrame = None
) -> Tuple[str, pd.DataFrame]:
    """Save regression coefficients to CSV and return the assembled DataFrame."""
    os.makedirs(output_dir, exist_ok=True)

    result = precomputed.copy() if precomputed is not None else prepare_drug_class_results(
        drug_class, ols_coefs, lars_coefs
    )
    filepath = os.path.join(output_dir, f"Regression_results_{drug_class}.csv")
    result.to_csv(filepath, index=False)

    return filepath, result


def save_performance_metrics(
    performance_data: List[Dict],
    output_dir: str = "results"
) -> str:
    """Save model performance metrics to a single CSV file.

    Creates Regression_results_performance.csv with columns:
    Drug_class, Drug, N_samples, N_features, CV_folds, CV_repeats,
    OLS_Mean_CV_MSE, LARS_Mean_CV_MSE (if available)

    Args:
        performance_data: List of dicts with performance metrics per drug
        output_dir: Directory to save results

    Returns:
        Path to saved CSV file
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create DataFrame from performance data
    df = pd.DataFrame(performance_data)

    # Sort by drug class and drug name
    df = df.sort_values(['Drug_class', 'Drug'])

    # Save to CSV
    filepath = os.path.join(output_dir, "Regression_results_performance.csv")
    df.to_csv(filepath, index=False)

    return filepath
