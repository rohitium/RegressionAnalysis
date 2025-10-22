"""
Mutation parsing and feature construction utilities.
"""

import re
from typing import List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .constants import INSERTION_KEYWORDS, DELETION_KEYWORDS


def _normalize_mutation_symbol(symbol: str) -> str:
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


def _tokenize_sequence_value(value: str, mixture_limit: Optional[int] = None) -> Set[str]:
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

    tokens = {ch for ch in raw.upper() if ch.isalpha() or ch in {'#', '~'}}

    mixture_size = len(tokens) if len(tokens) > 1 else 0
    if mixture_limit is not None and mixture_size > mixture_limit:
        return set()

    return tokens


def convert_muts(muts_in: List[str]) -> List[str]:
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
    pattern = re.compile(r'^\d{2,3}[A-Z#~]{1,8}$', re.IGNORECASE)
    if not all(pattern.match(m.strip()) for m in muts_in):
        raise ValueError('Each mutation must be 2-3 digits followed by 1-8 letters, #, or ~.')


def parse_mut_tokens(muts_in: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    pattern = re.compile(r'^(\d{2,3})([A-Z#~]{1,8})$', re.IGNORECASE)
    parsed = [pattern.match(m.strip()) for m in muts_in]
    pos = np.array([int(m.group(1)) for m in parsed], dtype=int)
    mut = np.array([m.group(2).upper() for m in parsed], dtype=object)
    return pos, mut


def build_X(positions_df: pd.DataFrame, mut_positions: np.ndarray,
            mut_aas: np.ndarray, mixture_limit: Optional[int] = None) -> pd.DataFrame:
    n_samples = positions_df.shape[0]
    n_mutations = len(mut_positions)
    X = np.zeros((n_samples, n_mutations), dtype=float)

    max_pos = positions_df.shape[1]
    mutation_cache = {}
    for pos in np.unique(mut_positions):
        if pos > max_pos:
            continue
        col = positions_df.iloc[:, pos - 1]
        tokenized = col.fillna('').astype(str).apply(
            lambda v: _tokenize_sequence_value(v, mixture_limit)
        )
        mutation_cache[pos] = tokenized.tolist()

    for j, (pos, aa) in enumerate(zip(mut_positions, mut_aas)):
        if pos in mutation_cache:
            observed = mutation_cache[pos]
            targets = _expand_mutation_targets(aa)
            X[:, j] = np.array(
                [1.0 if tokens and (tokens & targets) else 0.0 for tokens in observed],
                dtype=float
            )

    colnames = [f"{pos}{aa}" for pos, aa in zip(mut_positions, mut_aas)]
    return pd.DataFrame(X, columns=colnames, index=positions_df.index)
