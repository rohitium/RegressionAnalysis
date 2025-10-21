"""
DRMcv: Cross-validated OLS regression for HIVDB GenoPheno datasets

Performs OLS regression (and optional LASSO) to predict drug resistance
phenotypes from genotypic mutations.

Original R author: Haley Hedlin (2014)
Python port: ChatGPT for R. Shafer

Usage examples:
  # Use default DRM lists
  python RegressionAnalysis.py

  # Use custom DRM lists from directory
  python RegressionAnalysis.py --drm_lists inputs/drm_with_scores/

  # Use specific CSV files
  python RegressionAnalysis.py --drm_lists inputs/DRM_NRTI.csv inputs/DRM_PI.csv

  # Run LASSO in addition to OLS
  python RegressionAnalysis.py --lars

  # Analyze specific drugs only
  python RegressionAnalysis.py --drugs 3TC ABC LPV DRV
"""

import argparse
import math
import os
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from constants import DRUGS_BY_DRUG_CLASS, DRUG_ALIASES
from helpers import (
    load_dataset, convert_muts, check_muts, parse_mut_tokens, build_X,
    load_drm_lists_from_csv, fit_ols_model, cross_validate_model,
    fit_lasso_model, plot_drug_coefficients, save_drug_class_results,
    save_performance_metrics, prepare_drug_class_results
)


def DRMcv(min_muts: int = 5, nfold: int = 5, nrep: int = 10,
          lars: bool = True, drm_lists: Dict[str, List[str]] = None,
          drugs: List[str] = None, generate_outputs: bool = True,
          collect_data: bool = False, run_metadata: Dict = None):
    """Run cross-validated OLS and LASSO regression for HIVDB datasets.

    Args:
        min_muts: Maximum threshold for rare mutation filtering (default: 5)
                 Actual threshold per drug is min(1% of samples, min_muts), at least 1
        nfold: Number of cross-validation folds (default: 5)
        nrep: Number of CV repetitions (default: 10)
        lars: Whether to run LASSO regression (default: True)
        drm_lists: Custom DRM lists - dict mapping drug/class name to mutations.
                  Drug-specific lists take precedence over class-level lists.
                  If None, loads CSVs from inputs/default.
        drugs: Specific drugs to analyze (if None, analyze all)
        generate_outputs: When False, skip writing per-run results and plots
        collect_data: When True, return performance and coefficient data instead of None
        run_metadata: Optional dict merged into performance/coefficient records

    Outputs:
        Creates results/ and plots/ directories with:
        - Regression_results_{drug_class}.csv: OLS and LASSO coefficients
        - Regression_results_performance.csv: Model performance metrics
        - coefficients_{drug}.png: Coefficient plots per drug
    """
    run_metadata = run_metadata or {}

    # Use default DRM lists from inputs/default if none provided
    if drm_lists is None:
        default_dir = os.path.join(os.path.dirname(__file__), "inputs", "default")
        if not os.path.isdir(default_dir):
            raise FileNotFoundError(
                f"Default DRM directory not found at '{default_dir}'. "
                "Provide custom lists with --drm_lists."
            )
        drm_lists = load_drm_lists_from_csv([default_dir])

    if generate_outputs:
        os.makedirs("results", exist_ok=True)
        os.makedirs("plots", exist_ok=True)

    # Track performance metrics across all drugs
    all_performance_data = []
    collected_coeffs = [] if collect_data else None

    # Process each drug class
    for drug_class, all_drugs in DRUGS_BY_DRUG_CLASS.items():
        # Filter to specific drugs if requested
        drugs_to_process = [d for d in all_drugs if d in drugs] if drugs else all_drugs
        if not drugs_to_process:
            continue

        # Skip if no DRM lists available for this class
        has_drm_list = drug_class in drm_lists or any(d in drm_lists for d in drugs_to_process)
        if not has_drm_list:
            continue

        # Load dataset from Stanford HIVDB
        complete_df, positions_df = load_dataset(drug_class)

        # Accumulate results for all drugs in this class
        all_ols_coefs = []
        all_cvmse = []
        all_lars_coefs = []

        # Process each drug
        for drug in drugs_to_process:
            # ================================================================
            # 1. Load and validate DRM list
            # ================================================================

            # Determine which DRM list to use (drug-specific or class-level)
            if drug in drm_lists:
                drm_list_raw = drm_lists[drug]
                print(f"Using drug-specific DRM list for {drug}")
            elif drug_class in drm_lists:
                drm_list_raw = drm_lists[drug_class]
                print(f"Using drug-class DRM list for {drug}")
            else:
                warnings.warn(f"No DRM list found for '{drug}' or '{drug_class}'. Skipping.")
                continue

            # Convert insertions/deletions and validate format
            drm_list = convert_muts(drm_list_raw)
            check_muts(drm_list)
            positions, aas = parse_mut_tokens(drm_list)

            # ================================================================
            # 2. Build feature matrix
            # ================================================================

            X = build_X(positions_df, positions, aas)

            # ================================================================
            # 3. Load phenotypic data
            # ================================================================

            # Find drug column (check aliases if needed)
            drug_col = drug
            if drug not in complete_df.columns:
                for alias, canonical in DRUG_ALIASES.items():
                    if canonical == drug and alias in complete_df.columns:
                        drug_col = alias
                        print(f"Using alias '{alias}' for drug '{drug}'")
                        break
                else:
                    warnings.warn(f"Drug '{drug}' not found in {drug_class} dataset. Skipping.")
                    continue

            # Convert phenotype to log10 (fold resistance)
            fold_resistant = pd.to_numeric(complete_df[drug_col], errors='coerce')
            fold_resistant_log10 = np.log10(fold_resistant.where(fold_resistant > 0, np.nan))

            # Check for valid samples
            n_valid = fold_resistant_log10.notna().sum()
            if n_valid == 0:
                print(f"WARNING: Drug '{drug}': No valid samples available for analysis. Skipping.")
                continue

            n_missing = len(fold_resistant) - n_valid
            if n_missing > 0:
                print(f"NOTE: Drug '{drug}': {n_missing} of {len(fold_resistant)} samples have "
                      f"missing/invalid phenotypic data in column '{drug_col}'. "
                      f"{n_valid} valid samples available.")

            # ================================================================
            # 4. Prepare regression dataframe
            # ================================================================

            # Combine features and outcome, drop rows with missing phenotype
            df_drug = pd.concat([pd.Series(fold_resistant_log10, name="Y"), X], axis=1).dropna()

            # Filter rare mutations (dynamic threshold based on sample size)
            sample_size = len(df_drug)
            drug_min_muts = max(1, min(int(0.01 * sample_size), min_muts))

            rare_cols = [c for c in df_drug.columns if c != 'Y' and df_drug[c].sum() < drug_min_muts]
            if rare_cols:
                print(f"Excluding rare mutations (<{drug_min_muts} seqs, {sample_size} total): "
                      f"{', '.join(rare_cols)}")
                df_drug = df_drug.drop(columns=rare_cols)

            # Check if enough features remain
            if df_drug.shape[1] <= 1:  # Only Y column
                print(f"WARNING: Drug '{drug}' has no mutations after filtering. Skipping.")
                continue

            # Check if enough samples for reliable regression
            n_features = df_drug.shape[1] - 1
            if sample_size < n_features + 2:
                print(f"WARNING: Drug '{drug}' has too few samples ({sample_size}) "
                      f"for {n_features} mutations. Skipping.")
                continue

            # ================================================================
            # 5. Fit models
            # ================================================================

            # OLS regression
            ols_out = fit_ols_model(df_drug)
            ols_out['drug'] = drug
            ols_out['mutation'] = ols_out.index
            all_ols_coefs.append(ols_out)

            # Cross-validation
            ols_mse_scores = cross_validate_model(df_drug, nfold, nrep)
            mse_df = pd.DataFrame({
                'drug': drug,
                'fold': range(1, len(ols_mse_scores) + 1),
                'mse': ols_mse_scores
            })
            all_cvmse.append(mse_df)

            # LASSO regression
            lars_mse_scores = None
            if lars:
                lasso_out, lars_mse_scores = fit_lasso_model(df_drug, nfold, nrep)
                lasso_out['drug'] = drug
                all_lars_coefs.append(lasso_out)

            # Track performance metrics for this drug
            perf_entry = {
                'Drug_class': drug_class,
                'Drug': drug,
                'N_samples': sample_size,
                'N_features': n_features,
                'CV_folds': nfold,
                'CV_repeats': nrep,
                'OLS_Mean_CV_MSE': ols_mse_scores.mean()
            }
            if lars and lars_mse_scores is not None:
                perf_entry['LARS_Mean_CV_MSE'] = lars_mse_scores.mean()
            perf_entry.update(run_metadata)
            all_performance_data.append(perf_entry)

        # ================================================================
        # 6. Save results and create plots
        # ================================================================

        if all_ols_coefs:
            class_results_df = prepare_drug_class_results(
                drug_class, all_ols_coefs, all_lars_coefs if lars else None
            )

            if collect_data:
                coeff_df = class_results_df.copy()
                for key, value in run_metadata.items():
                    coeff_df[key] = value
                collected_coeffs.append(coeff_df)

            # Save coefficient results
            if generate_outputs:
                result_file, saved_df = save_drug_class_results(
                    drug_class, all_ols_coefs,
                    all_lars_coefs if lars else None,
                    precomputed=class_results_df
                )
                print(f"  - {result_file}")
                results_df = saved_df.copy()
            else:
                results_df = class_results_df.copy()

            # Create MSE lookup from performance data
            mse_lookup = {}
            for perf in all_performance_data:
                if perf['Drug_class'] == drug_class:
                    mse_lookup[perf['Drug']] = {
                        'ols': perf.get('OLS_Mean_CV_MSE'),
                        'lars': perf.get('LARS_Mean_CV_MSE')
                    }

            if generate_outputs:
                # Plot each drug
                for drug in results_df['Drug'].unique():
                    drug_data = results_df[results_df['Drug'] == drug].copy()
                    if 'Mutation' in drug_data.columns:
                        drug_data = drug_data.rename(columns={'Mutation': 'mutation'})
                    else:
                        drug_data = drug_data.rename(columns={'Mut': 'mutation'})

                    ols_mse = mse_lookup.get(drug, {}).get('ols')
                    lars_mse = mse_lookup.get(drug, {}).get('lars')

                    plot_filename = plot_drug_coefficients(drug, drug_data, ols_mse, lars_mse)
                    print(f"  - {plot_filename}")

    # ================================================================
    # 7. Save performance metrics
    # ================================================================

    if generate_outputs and all_performance_data:
        perf_file = save_performance_metrics(all_performance_data)
        print(f"  - {perf_file}")

    if collect_data:
        return all_performance_data, collected_coeffs


def _plot_benchmark_mse(perf_long: pd.DataFrame, output_path: str) -> None:
    """Create heatmaps of mean CV MSE across drugs, DRM sources, and min_muts."""
    if perf_long.empty:
        warnings.warn("No benchmark performance data available for MSE heatmap.")
        return

    drugs = sorted(perf_long['Drug'].unique())
    combo_keys = (
        perf_long[['DRM_source', 'Model']]
        .drop_duplicates()
        .sort_values(['DRM_source', 'Model'])
        .itertuples(index=False, name=None)
    )
    combo_keys = list(combo_keys)
    if not combo_keys:
        warnings.warn("No benchmark combinations found for heatmap.")
        return

    min_muts_values = sorted(perf_long['Min_muts'].unique())
    n_drugs = len(drugs)
    n_cols = 2 if n_drugs > 1 else 1
    n_rows = math.ceil(n_drugs / n_cols)

    fig_width = max(6, len(min_muts_values) * 2.2 * n_cols)
    fig_height = max(3.5, n_rows * 3)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
    axes_flat = axes.flatten()
    vmin = perf_long['Mean_CV_MSE'].min()
    vmax = perf_long['Mean_CV_MSE'].max()
    color_map = plt.cm.viridis
    last_im = None

    row_labels = [f"{src}\n{model}" for src, model in combo_keys]

    for ax, drug in zip(axes_flat, drugs):
        subset = perf_long[perf_long['Drug'] == drug]
        grid = np.full((len(combo_keys), len(min_muts_values)), np.nan)
        for i, (src, model) in enumerate(combo_keys):
            for j, mm in enumerate(min_muts_values):
                match = subset[
                    (subset['DRM_source'] == src) &
                    (subset['Model'] == model) &
                    (subset['Min_muts'] == mm)
                ]['Mean_CV_MSE']
                if not match.empty:
                    grid[i, j] = match.mean()
        masked = np.ma.masked_invalid(grid)
        last_im = ax.imshow(masked, aspect='auto', cmap=color_map, vmin=vmin, vmax=vmax)
        ax.set_title(drug, fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(min_muts_values)))
        ax.set_xticklabels(min_muts_values)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels)
        for (i, j), val in np.ndenumerate(grid):
            if not np.isnan(val):
                text_color = 'white' if not np.isnan(vmin) and val < (vmin + vmax) / 2 else 'black'
                ax.text(
                    j, i, f"{val:.3f}",
                    ha='center', va='center',
                    color=text_color,
                    fontsize=7
                )
    # Remove unused axes
    for ax in axes_flat[len(drugs):]:
        fig.delaxes(ax)

    if last_im is not None:
        cbar_ax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
        cbar = fig.colorbar(last_im, cax=cbar_ax)
        cbar.set_label('Mean CV MSE', fontsize=10)

    fig.suptitle('Benchmark MSE Comparison', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 0.9, 0.96])
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def _plot_benchmark_coefficients(coeff_df: pd.DataFrame, model: str,
                                 run_info: Dict, output_path: str) -> None:
    """Visualize coefficient values for the best-performing benchmark run."""
    if coeff_df.empty:
        warnings.warn("No coefficient data available for benchmark plotting.")
        return

    coef_col = 'OLS_coef' if model == 'OLS' else 'LARS_coef'
    se_col = 'OLS_se' if model == 'OLS' else 'LARS_se'
    available = coeff_df.dropna(subset=[coef_col])
    if available.empty:
        warnings.warn(f"No coefficients found for model '{model}' in benchmark results.")
        return

    drugs = sorted(available['Drug'].unique())
    n_drugs = len(drugs)
    fig_height = max(3, n_drugs * 2.5)
    fig, axes = plt.subplots(n_drugs, 1, figsize=(14, fig_height), squeeze=False)
    axes = axes.flatten()

    for ax, drug in zip(axes, drugs):
        sub = available[available['Drug'] == drug].copy()
        sub = sub.sort_values(['Pos', 'Mutation'])
        x_pos = np.arange(len(sub))
        errs = sub[se_col].fillna(0.0).values
        ax.bar(
            x_pos, sub[coef_col].values,
            yerr=errs if np.any(errs) else None,
            capsize=3 if np.any(errs) else 0,
            color='slateblue', edgecolor='black', linewidth=0.5
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(sub['Mutation'], rotation=90)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_ylabel('Coefficient', fontsize=10)
        ax.set_title(drug, fontsize=12, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.3)

    for ax in axes[len(drugs):]:
        ax.axis('off')

    title = (f"Benchmark Coefficients - Best Model: {model} | "
             f"DRM source: {run_info.get('DRM_source')} | "
             f"min_muts: {run_info.get('Min_muts')}")
    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def run_benchmark(lars: bool, nfold: int, nrep: int,
                  drugs: List[str] = None, min_muts_grid: List[int] = None) -> None:
    """Execute benchmark runs across DRM list sets and min_muts grid."""
    base_dir = os.path.dirname(__file__)
    min_muts_values = min_muts_grid or [1, 5, 10, 20]
    benchmark_sets = [
        ('default', os.path.join(base_dir, 'inputs', 'default')),
        ('drm_with_scores', os.path.join(base_dir, 'inputs', 'drm_with_scores'))
    ]

    all_perf_records = []
    coeff_records = []

    for source_label, path in benchmark_sets:
        if not os.path.isdir(path):
            warnings.warn(f"Benchmark DRM directory '{path}' not found. Skipping.")
            continue
        drm_lists = load_drm_lists_from_csv([path])
        for mm in min_muts_values:
            print(f"Benchmark run: source={source_label}, min_muts={mm}")
            run_meta = {'DRM_source': source_label, 'Min_muts': mm}
            result = DRMcv(
                min_muts=mm,
                nfold=nfold,
                nrep=nrep,
                lars=lars,
                drm_lists=drm_lists,
                drugs=drugs,
                generate_outputs=False,
                collect_data=True,
                run_metadata=run_meta
            )
            if result is None:
                continue
            perf_data, coeff_data = result
            all_perf_records.extend(perf_data)
            if coeff_data:
                coeff_records.extend(coeff_data)

    if not all_perf_records:
        warnings.warn("No benchmark results were generated.")
        return

    results_dir = os.path.join(base_dir, 'results')
    plots_dir = os.path.join(base_dir, 'plots')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    perf_df = pd.DataFrame(all_perf_records)
    value_cols = [col for col in ['OLS_Mean_CV_MSE', 'LARS_Mean_CV_MSE'] if col in perf_df.columns]
    id_cols = [col for col in perf_df.columns if col not in value_cols]

    perf_long = perf_df.melt(
        id_vars=id_cols,
        value_vars=value_cols,
        var_name='Model',
        value_name='Mean_CV_MSE'
    ).dropna(subset=['Mean_CV_MSE'])
    perf_long['Model'] = perf_long['Model'].str.replace('_Mean_CV_MSE', '', regex=False)

    benchmark_csv = os.path.join(results_dir, "Regression_results_benchmarks.csv")
    perf_long.to_csv(benchmark_csv, index=False)
    print(f"  - {benchmark_csv}")

    heatmap_path = os.path.join(plots_dir, "benchmarks_mse.png")
    _plot_benchmark_mse(perf_long, heatmap_path)
    print(f"  - {heatmap_path}")

    coeff_df = pd.concat(coeff_records, ignore_index=True) if coeff_records else pd.DataFrame()

    # Identify best-performing combination by mean MSE across drugs
    mean_scores = (
        perf_long.groupby(['DRM_source', 'Min_muts', 'Model'])['Mean_CV_MSE']
        .mean()
        .reset_index()
    )
    if mean_scores.empty:
        warnings.warn("No benchmark MSE scores available for coefficient plot.")
        return

    best_row = mean_scores.loc[mean_scores['Mean_CV_MSE'].idxmin()]
    best_filter = (
        (coeff_df['DRM_source'] == best_row['DRM_source']) &
        (coeff_df['Min_muts'] == best_row['Min_muts'])
    )
    coeff_best = coeff_df[best_filter].copy() if not coeff_df.empty else pd.DataFrame()

    coeff_plot_path = os.path.join(plots_dir, "benchmarks_coefficients.png")
    _plot_benchmark_coefficients(
        coeff_best,
        best_row['Model'],
        {'DRM_source': best_row['DRM_source'], 'Min_muts': best_row['Min_muts']},
        coeff_plot_path
    )
    print(f"  - {coeff_plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='DRMcv: Cross-validated OLS for HIVDB GenoPheno datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--min_muts', type=int, default=5,
                        help='Maximum threshold for rare mutation filtering (default: 5). '
                             'Actual threshold is min(1%% of samples, min_muts), at least 1.')
    parser.add_argument('--nfold', type=int, default=5,
                        help='Number of cross-validation folds (default: 5)')
    parser.add_argument('--nrep', type=int, default=10,
                        help='Number of CV repetitions (default: 10)')
    parser.add_argument('--no-lars', dest='lars', action='store_false',
                        help='Disable LASSO regression (enabled by default)')
    parser.set_defaults(lars=True)
    parser.add_argument('--drm_lists', nargs='+', default=None,
                        help='Directory path or CSV files with DRM lists (Pos and Mut columns). '
                             'Files must follow pattern: DRM_{drug_or_class}.csv '
                             '(e.g., DRM_3TC.csv, DRM_NRTI.csv). '
                             'Drug-specific lists take precedence over class-wide lists. '
                             'Defaults to inputs/default when omitted.')
    parser.add_argument('--drugs', nargs='+', default=None,
                        help='Specific drugs to analyze (e.g., 3TC ABC LPV DRV)')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run benchmarking across default and drm_with_scores DRM sets '
                             'for min_muts values {1,5,10,20} and generate aggregate outputs.')

    args = parser.parse_args()

    if args.benchmark:
        if args.drm_lists:
            warnings.warn("--drm_lists is ignored when --benchmark is specified.")
        run_benchmark(
            lars=args.lars,
            nfold=args.nfold,
            nrep=args.nrep,
            drugs=args.drugs,
            min_muts_grid=[1, 5, 10, 20]
        )
    else:
        custom_drm_lists = None
        if args.drm_lists:
            custom_drm_lists = load_drm_lists_from_csv(args.drm_lists)

        DRMcv(
            min_muts=args.min_muts,
            nfold=args.nfold,
            nrep=args.nrep,
            lars=args.lars,
            drm_lists=custom_drm_lists,
            drugs=args.drugs
        )
