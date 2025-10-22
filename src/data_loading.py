"""
Utilities for loading HIVDB datasets and DRM lists.
"""

import os
import re
import warnings
from typing import Dict, List, Tuple

import pandas as pd

from .constants import DATA_URLS


def _detect_position_columns(columns: List[str]) -> List[str]:
    """Return ordered list of position columns strictly matching P1, P2, …"""
    pattern = re.compile(r'^P(\d{1,3})$', re.IGNORECASE)
    detected = []
    for col in columns:
        stripped = col.strip()
        if stripped.lower() == 'ptid':
            continue
        match = pattern.match(stripped)
        if match:
            detected.append((int(match.group(1)), col))
    detected.sort(key=lambda item: item[0])
    return [col for _, col in detected]


def load_dataset(drug_class: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load dataset from Stanford HIVDB, filter Method, and extract position columns."""
    df = pd.read_csv(
        DATA_URLS[drug_class],
        sep="\t",
        dtype=str,
        comment='@',
        engine='python'
    )

    if 'Method' in df.columns:
        method_mask = df['Method'].astype(str).str.contains('PhenoSense', case=False, na=False)
        if method_mask.any():
            df = df[method_mask].copy()
        else:
            warnings.warn(
                f"No rows with Method containing 'PhenoSense' found in {drug_class} dataset; using all rows."
            )
    else:
        warnings.warn(f"'Method' column not found in {drug_class} dataset; using all rows.")

    position_cols = _detect_position_columns(df.columns.tolist())
    if not position_cols:
        raise ValueError(f"Could not detect position columns in {drug_class} dataset.")

    positions_df = df[position_cols].copy().reset_index(drop=True)
    positions_df.columns = range(1, positions_df.shape[1] + 1)

    return df.reset_index(drop=True), positions_df


def load_drm_lists_from_csv(csv_paths: List[str]) -> Dict[str, List[str]]:
    """Load DRM lists from CSV files or directory."""
    if len(csv_paths) == 1 and os.path.isdir(csv_paths[0]):
        directory = csv_paths[0]
        csv_files = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.startswith('DRM_') and f.endswith('.csv')
        ]
        if not csv_files:
            raise ValueError(f"No DRM_*.csv files found in directory {directory}")
    else:
        csv_files = csv_paths

    drm_dict = {}
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if 'Pos' not in df.columns or 'Mut' not in df.columns:
            raise ValueError(f"CSV file {csv_file} must have 'Pos' and 'Mut' columns")

        basename = os.path.basename(csv_file)
        if not (basename.startswith('DRM_') and basename.endswith('.csv')):
            raise ValueError(f"CSV filename {csv_file} must follow pattern DRM_{{name}}.csv")

        drug_or_class = basename[4:-4]
        mutations = [f"{row['Pos']}{row['Mut']}" for _, row in df.iterrows()]
        drm_dict[drug_or_class] = mutations

    return drm_dict
