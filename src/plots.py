"""
Plotting utilities for regression analysis outputs and benchmarks.
"""

import math
import os
import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_drug_coefficients(drug: str, coef_data: pd.DataFrame,
                           ols_mse: float = None, lars_mse: float = None,
                           output_dir: str = "plots",
                           show_legend: bool = True,
                           mutations_order: Optional[List[str]] = None,
                           y_limits: Optional[Tuple[float, float]] = None) -> str:
    os.makedirs(output_dir, exist_ok=True)

    data = coef_data.copy()
    if mutations_order is not None:
        mutation_info = coef_data.set_index('mutation') if 'mutation' in coef_data.columns else None
        data = data.set_index('mutation').reindex(mutations_order)
        for col in ['OLS_coef', 'OLS_se']:
            if col in data.columns:
                data[col] = data[col].fillna(0.0)
        if 'LARS_coef' in data.columns:
            data['LARS_coef'] = data['LARS_coef'].fillna(0.0)
        if 'LARS_se' in data.columns:
            data['LARS_se'] = data['LARS_se'].fillna(0.0)
        data['mutation'] = mutations_order
        if mutation_info is not None:
            if 'Pos' in data.columns:
                data['Pos'] = mutation_info['Pos'].reindex(mutations_order).values
            if 'Mut' in data.columns:
                data['Mut'] = mutation_info['Mut'].reindex(mutations_order).values
        data['Drug'] = drug
        data = data.reset_index(drop=True)

    mutations = data['mutation'].tolist()
    ols_coef = data['OLS_coef'].values
    ols_se = data['OLS_se'].values

    has_lars = 'LARS_coef' in coef_data.columns
    lars_err = None
    if has_lars:
        lars_coef = data['LARS_coef'].values
        if 'LARS_se' in data.columns:
            lars_err = data['LARS_se'].values

    if y_limits is None:
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
        y_limits = (y_min - 0.1 * y_range, y_max + 0.1 * y_range) if y_range else (-0.1, 0.1)

    fig, ax = plt.subplots(figsize=(12, 4))

    x_pos = np.arange(len(mutations))
    if has_lars:
        width = 0.35
        x_ols = x_pos - width / 2
        x_lars = x_pos + width / 2
    else:
        width = 0.7
        x_ols = x_pos

    ax.bar(
        x_ols, ols_coef, width, color='steelblue', edgecolor='black',
        linewidth=0.5, yerr=ols_se, capsize=3,
        error_kw={'linewidth': 1, 'ecolor': 'black'},
        label='OLS'
    )

    if has_lars:
        ax.bar(
            x_lars, lars_coef, width, color='firebrick', edgecolor='black',
            linewidth=0.5, yerr=lars_err, capsize=3 if lars_err is not None else 0,
            error_kw={'linewidth': 1, 'ecolor': 'black'} if lars_err is not None else None,
            label='LASSO'
        )

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

    if show_legend:
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

    fig.tight_layout(rect=[0, 0, 0.88 if show_legend else 1, 1])
    plot_path = os.path.join(output_dir, f"coefficients_{drug}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return plot_path


def plot_benchmark_mse(perf_long: pd.DataFrame, output_path: str) -> None:
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


def plot_benchmark_coefficients(coeff_df: pd.DataFrame, model: str,
                                 run_info: Dict, output_path: str) -> None:
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
