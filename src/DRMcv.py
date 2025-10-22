"""
Core analysis workflows for HIVDB regression modelling and benchmarking.
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .constants import DRUGS_BY_DRUG_CLASS, DRUG_ALIASES
from .data_loading import load_dataset, load_drm_lists_from_csv
from .data_processing import convert_muts, check_muts, parse_mut_tokens, build_X
from .models import fit_ols_model, cross_validate_model, fit_lasso_model
from .plots import (
    plot_drug_coefficients,
    plot_benchmark_mse,
    plot_benchmark_coefficients
)
from .results import (
    prepare_drug_class_results,
    save_drug_class_results,
    save_performance_metrics
)


def _default_drm_lists_path() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "inputs", "default")


def _ensure_default_lists() -> Dict[str, List[str]]:
    default_dir = _default_drm_lists_path()
    if not os.path.isdir(default_dir):
        raise FileNotFoundError(
            f"Default DRM directory not found at '{default_dir}'. "
            "Provide custom lists with --drm_lists."
        )
    return load_drm_lists_from_csv([default_dir])


def DRMcv(min_muts: int = 5, nfold: int = 5, nrep: int = 10,
          lars: bool = True, drm_lists: Optional[Dict[str, List[str]]] = None,
          drugs: Optional[List[str]] = None, generate_outputs: bool = True,
          collect_data: bool = False, run_metadata: Optional[Dict] = None,
          show_legend: bool = True, mixture_limit: Optional[int] = None,
          results_dir: str = "results", plots_dir: str = "plots",
          paper_mode: bool = False) -> Optional[Tuple[List[Dict], List[pd.DataFrame]]]:
    """Run cross-validated OLS and LASSO regression for HIVDB datasets.

    Args:
        min_muts: Minimum occurrence threshold for retaining mutations.
        nfold: Number of cross-validation folds.
        nrep: Number of cross-validation repetitions.
        lars: Whether to include the LASSO model.
        drm_lists: Optional mapping of drug/class to DRM definitions.
        drugs: Optional list of drugs to analyse.
        generate_outputs: When False, suppress file/plot creation.
        collect_data: When True, return performance and coefficient data.
        run_metadata: Extra metadata merged into output records.
        show_legend: Whether to display legends on coefficient plots.
        mixture_limit: Maximum number of mixture residues to accept (None for unlimited).
        results_dir: Base directory for result CSVs.
        plots_dir: Base directory for coefficient plots.
        paper_mode: When True, harmonise mutation coverage and axis limits across drugs.
    """
    run_metadata = run_metadata or {}

    if drm_lists is None:
        drm_lists = _ensure_default_lists()

    if generate_outputs:
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

    all_performance_data: List[Dict] = []
    collected_coeffs: Optional[List[pd.DataFrame]] = [] if collect_data else None

    for drug_class, all_drugs in DRUGS_BY_DRUG_CLASS.items():
        drugs_to_process = [d for d in all_drugs if d in drugs] if drugs else all_drugs
        if not drugs_to_process:
            continue

        has_drm_list = drug_class in drm_lists or any(d in drm_lists for d in drugs_to_process)
        if not has_drm_list:
            continue

        complete_df, positions_df = load_dataset(drug_class)
        all_ols_coefs = []
        all_lars_coefs = []

        for drug in drugs_to_process:
            if drug in drm_lists:
                drm_list_raw = drm_lists[drug]
                print(f"Using drug-specific DRM list for {drug}")
            elif drug_class in drm_lists:
                drm_list_raw = drm_lists[drug_class]
                print(f"Using drug-class DRM list for {drug}")
            else:
                warnings.warn(f"No DRM list found for '{drug}' or '{drug_class}'. Skipping.")
                continue

            drm_list = convert_muts(drm_list_raw)
            check_muts(drm_list)
            positions, aas = parse_mut_tokens(drm_list)

            X = build_X(positions_df, positions, aas, mixture_limit=mixture_limit)

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

            fold_resistant = pd.to_numeric(complete_df[drug_col], errors='coerce')
            fold_resistant_log10 = np.log10(fold_resistant.where(fold_resistant > 0, np.nan))

            n_valid = fold_resistant_log10.notna().sum()
            if n_valid == 0:
                print(f"WARNING: Drug '{drug}': No valid samples available for analysis. Skipping.")
                continue

            n_missing = len(fold_resistant) - n_valid
            if n_missing > 0:
                print(f"NOTE: Drug '{drug}': {n_missing} of {len(fold_resistant)} samples have "
                      f"missing/invalid phenotypic data in column '{drug_col}'. "
                      f"{n_valid} valid samples available.")

            df_drug = pd.concat([pd.Series(fold_resistant_log10, name="Y"), X], axis=1).dropna()

            sample_size = len(df_drug)
            rare_cols = [c for c in df_drug.columns if c != 'Y' and df_drug[c].sum() <= min_muts]
            if rare_cols:
                print(f"Excluding rare mutations (≤{min_muts} seqs, {sample_size} total): "
                      f"{', '.join(rare_cols)}")
                df_drug = df_drug.drop(columns=rare_cols)

            if df_drug.shape[1] <= 1:
                print(f"WARNING: Drug '{drug}' has no mutations after filtering. Skipping.")
                continue

            n_features = df_drug.shape[1] - 1
            if sample_size < n_features + 2:
                print(f"WARNING: Drug '{drug}' has too few samples ({sample_size}) "
                      f"for {n_features} mutations. Skipping.")
                continue

            ols_out = fit_ols_model(df_drug)
            ols_out['drug'] = drug
            ols_out['mutation'] = ols_out.index
            all_ols_coefs.append(ols_out)

            ols_mse_scores = cross_validate_model(df_drug, nfold, nrep)

            lars_mse_scores = None
            if lars:
                lasso_out, lars_mse_scores = fit_lasso_model(df_drug, nfold, nrep)
                lasso_out['drug'] = drug
                all_lars_coefs.append(lasso_out)

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
            perf_entry['Mixture_limit'] = 'all' if mixture_limit is None else str(mixture_limit)
            perf_entry.update(run_metadata)
            all_performance_data.append(perf_entry)

        if all_ols_coefs:
            class_results_df = prepare_drug_class_results(
                drug_class, all_ols_coefs, all_lars_coefs if lars else None
            )

            mutation_order = None
            class_y_limits = None

            if paper_mode:
                mutation_meta = (
                    class_results_df[['Mutation', 'Pos', 'Mut']]
                    .drop_duplicates()
                    .set_index('Mutation')
                )
                mutation_order = mutation_meta.sort_values(['Pos', 'Mut']).index.tolist()

                expanded_frames = []
                for drug_name in class_results_df['Drug'].unique():
                    drug_df = class_results_df[class_results_df['Drug'] == drug_name].set_index('Mutation')
                    expanded = drug_df.reindex(mutation_order)
                    for col in ['OLS_coef', 'OLS_se']:
                        expanded[col] = expanded[col].fillna(0.0)
                    if lars and 'LARS_coef' in expanded.columns:
                        expanded['LARS_coef'] = expanded['LARS_coef'].fillna(0.0)
                    if lars and 'LARS_se' in expanded.columns:
                        expanded['LARS_se'] = expanded['LARS_se'].fillna(0.0)
                    expanded['Drug'] = drug_name
                    expanded['Drug_class'] = drug_class
                    expanded['Pos'] = mutation_meta.loc[mutation_order, 'Pos'].values
                    expanded['Mut'] = mutation_meta.loc[mutation_order, 'Mut'].values
                    expanded['Mutation'] = mutation_order
                    expanded_frames.append(expanded.reset_index(drop=True))
                class_results_df = pd.concat(expanded_frames, ignore_index=True)
                class_results_df = class_results_df.sort_values(['Drug', 'Pos', 'Mut']).reset_index(drop=True)

                lower = (class_results_df['OLS_coef'] - class_results_df['OLS_se']).min()
                upper = (class_results_df['OLS_coef'] + class_results_df['OLS_se']).max()
                if lars and 'LARS_coef' in class_results_df.columns:
                    lars_coef = class_results_df['LARS_coef']
                    lars_se = class_results_df['LARS_se'] if 'LARS_se' in class_results_df.columns else 0
                    upper = max(upper, (lars_coef + lars_se).max())
                    lower = min(lower, (lars_coef - lars_se).min())
                span = upper - lower
                padding = 0.1 * span if span else 0.1
                class_y_limits = (lower - padding, upper + padding)

            if collect_data:
                coeff_df = class_results_df.copy()
                coeff_df['Mixture_limit'] = 'all' if mixture_limit is None else str(mixture_limit)
                for key, value in run_metadata.items():
                    coeff_df[key] = value
                collected_coeffs.append(coeff_df)

            if generate_outputs:
                result_file, saved_df = save_drug_class_results(
                    drug_class,
                    all_ols_coefs,
                    all_lars_coefs if lars else None,
                    output_dir=results_dir,
                    precomputed=class_results_df
                )
                print(f"  - {result_file}")
                results_df = saved_df.copy()
            else:
                results_df = class_results_df.copy()

            if generate_outputs:
                mse_lookup = {}
                for perf in all_performance_data:
                    if perf['Drug_class'] == drug_class:
                        mse_lookup[perf['Drug']] = {
                            'ols': perf.get('OLS_Mean_CV_MSE'),
                            'lars': perf.get('LARS_Mean_CV_MSE')
                        }

                for drug in results_df['Drug'].unique():
                    drug_data = results_df[results_df['Drug'] == drug].copy()
                    if 'Mutation' in drug_data.columns:
                        drug_data = drug_data.rename(columns={'Mutation': 'mutation'})
                    else:
                        drug_data = drug_data.rename(columns={'Mut': 'mutation'})

                    ols_mse = mse_lookup.get(drug, {}).get('ols')
                    lars_mse = mse_lookup.get(drug, {}).get('lars')

                    plot_path = plot_drug_coefficients(
                        drug,
                        drug_data,
                        ols_mse,
                        lars_mse,
                        output_dir=plots_dir,
                        show_legend=show_legend,
                        mutations_order=mutation_order,
                        y_limits=class_y_limits
                    )
                    print(f"  - {plot_path}")

    if generate_outputs and all_performance_data:
        perf_file = save_performance_metrics(all_performance_data, output_dir=results_dir)
        print(f"  - {perf_file}")

    if collect_data:
        return all_performance_data, collected_coeffs or []
    return None


def run_benchmark(lars: bool, nfold: int, nrep: int,
                  drugs: Optional[List[str]] = None, min_muts_grid: Optional[List[int]] = None,
                  show_legend: bool = True, mixture_limit: Optional[int] = None) -> None:
    """Execute benchmark runs across DRM list sets and min_muts grid."""
    base_dir = os.path.dirname(__file__)
    min_muts_values = min_muts_grid or [1, 5, 10, 20]
    benchmark_sets = [
        ('default', os.path.join(base_dir, 'inputs', 'default')),
        ('drm_with_scores', os.path.join(base_dir, 'inputs', 'drm_with_scores'))
    ]

    all_perf_records: List[Dict] = []
    coeff_records: List[pd.DataFrame] = []

    for source_label, path in benchmark_sets:
        if not os.path.isdir(path):
            warnings.warn(f"Benchmark DRM directory '{path}' not found. Skipping.")
            continue
        drm_lists = load_drm_lists_from_csv([path])
        for mm in min_muts_values:
            print(f"Benchmark run: source={source_label}, min_muts={mm}")
            run_meta = {'DRM_source': source_label, 'Min_muts': mm, 'Mixture_limit': mixture_limit if mixture_limit is not None else 'all'}
            result = DRMcv(
                min_muts=mm,
                nfold=nfold,
                nrep=nrep,
                lars=lars,
                drm_lists=drm_lists,
                drugs=drugs,
                generate_outputs=False,
                collect_data=True,
                run_metadata=run_meta,
                show_legend=show_legend,
                mixture_limit=mixture_limit
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
    value_cols = [c for c in ['OLS_Mean_CV_MSE', 'LARS_Mean_CV_MSE'] if c in perf_df.columns]
    id_cols = [c for c in perf_df.columns if c not in value_cols]

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
    plot_benchmark_mse(perf_long, heatmap_path)
    print(f"  - {heatmap_path}")

    coeff_df = pd.concat(coeff_records, ignore_index=True) if coeff_records else pd.DataFrame()
    mean_scores = (
        perf_long.groupby(['DRM_source', 'Min_muts', 'Model'])['Mean_CV_MSE']
        .mean()
        .reset_index()
    )
    if mean_scores.empty:
        warnings.warn("No benchmark MSE scores available for coefficient plot.")
        return

    best_row = mean_scores.loc[mean_scores['Mean_CV_MSE'].idxmin()]
    if coeff_df.empty:
        warnings.warn("No coefficient records available for benchmark coefficient plot.")
        return

    best_filter = (
        (coeff_df['DRM_source'] == best_row['DRM_source']) &
        (coeff_df['Min_muts'] == best_row['Min_muts'])
    )
    coeff_best = coeff_df[best_filter].copy()

    coeff_plot_path = os.path.join(plots_dir, "benchmarks_coefficients.png")
    plot_benchmark_coefficients(
        coeff_best,
        best_row['Model'],
        {'DRM_source': best_row['DRM_source'], 'Min_muts': best_row['Min_muts']},
        coeff_plot_path
    )
    print(f"  - {coeff_plot_path}")
