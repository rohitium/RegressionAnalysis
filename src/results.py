"""
Helpers for assembling and saving regression results.
"""

import os
from typing import Dict, List, Tuple

import pandas as pd


def prepare_drug_class_results(
    drug_class: str,
    ols_coefs: List[pd.DataFrame],
    lars_coefs: List[pd.DataFrame] = None
) -> pd.DataFrame:
    combined_ols = pd.concat(ols_coefs, ignore_index=True)
    combined_ols = combined_ols[combined_ols['mutation'] != 'Intercept'].copy()

    combined_ols['Pos'] = combined_ols['mutation'].str.extract(r'^(\d+)')[0].astype(int)
    combined_ols['Mut'] = combined_ols['mutation'].str.extract(r'(\D+)$')[0]
    combined_ols['Mutation'] = combined_ols['mutation']
    combined_ols = combined_ols.rename(columns={'coef': 'OLS_coef', 'se': 'OLS_se', 'drug': 'Drug'})

    result = combined_ols[['Pos', 'Mut', 'Mutation', 'Drug', 'OLS_coef', 'OLS_se']].copy()

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

    ordered_cols = ['Pos', 'Mut', 'Mutation', 'Drug', 'OLS_coef', 'OLS_se']
    if 'LARS_coef' in result.columns:
        ordered_cols.append('LARS_coef')
    if 'LARS_se' in result.columns:
        ordered_cols.append('LARS_se')

    result = result[ordered_cols]
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
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(performance_data)
    df = df.sort_values(['Drug_class', 'Drug'])

    filepath = os.path.join(output_dir, "Regression_results_performance.csv")
    df.to_csv(filepath, index=False)
    return filepath

