"""Out-of-fold (OOF) embedding and score extraction to prevent fusion-training optimism."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
import torch

from src.data.splits import discover_strata

logger = logging.getLogger(__name__)


def generate_identity_folds(
    train_df: pd.DataFrame,
    n_folds: int = 5,
    group_column: str = "identity_id",
    stratify_columns: Optional[List[str]] = None,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Partition training identities into K folds with zero identity overlap across folds.

    Returns:
        List of (train_indices, held_out_indices) for each fold k in [0, n_folds-1].
    """
    if stratify_columns is None:
        stratify_columns = ["ethnicity", "gender"]

    rng = np.random.RandomState(seed)
    ident_to_stratum, _ = discover_strata(train_df, stratify_columns=stratify_columns, group_column=group_column)

    # Group identities by stratum
    stratum_to_idents: Dict[str, List[str]] = defaultdict(list)
    for ident, stratum in ident_to_stratum.items():
        stratum_to_idents[stratum].append(ident)

    # Assign identities to folds round-robin per stratum
    fold_to_idents: Dict[int, Set[str]] = defaultdict(set)
    for stratum, idents in stratum_to_idents.items():
        shuffled = list(idents)
        rng.shuffle(shuffled)
        for i, ident in enumerate(shuffled):
            fold_idx = i % n_folds
            fold_to_idents[fold_idx].add(ident)

    # Generate row index masks for each fold
    all_identities = set(ident_to_stratum.keys())
    folds_indices: List[Tuple[np.ndarray, np.ndarray]] = []
    df_idents = train_df[group_column].astype(str)

    for k in range(n_folds):
        held_out_ids = fold_to_idents[k]
        train_ids = all_identities - held_out_ids

        # Strict identity disjointness assertion per fold
        assert train_ids.isdisjoint(held_out_ids), f"Identity overlap in fold {k}!"

        held_out_mask = df_idents.isin(held_out_ids).to_numpy()
        train_mask = df_idents.isin(train_ids).to_numpy()

        train_indices = np.where(train_mask)[0]
        held_out_indices = np.where(held_out_mask)[0]

        folds_indices.append((train_indices, held_out_indices))
        logger.info(
            f"Fold {k}: {len(train_ids)} train identities ({len(train_indices)} rows), "
            f"{len(held_out_ids)} held-out identities ({len(held_out_indices)} rows)"
        )

    return folds_indices


def assemble_oof_dataset(
    extracted_folds: List[Dict[str, np.ndarray]],
    output_path: Optional[str | Path] = None,
) -> Dict[str, np.ndarray]:
    """Assemble out-of-fold feature arrays from all K folds and save to NPZ."""
    combined: Dict[str, List[np.ndarray]] = defaultdict(list)
    for fold_dict in extracted_folds:
        for key, arr in fold_dict.items():
            combined[key].append(arr)

    assembled = {key: np.concatenate(arr_list, axis=0) for key, arr_list in combined.items()}

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_p, **assembled)
        logger.info(f"OOF feature dataset saved to {out_p}")

    return assembled
