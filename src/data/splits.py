"""Identity-aware dataset splitting with dynamic stratification discovery."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def discover_strata(
    df: pd.DataFrame,
    stratify_columns: List[str],
    group_column: str = "identity_id",
    min_samples_warning: int = 3,
) -> Tuple[Dict[str, str], pd.DataFrame]:
    """Discover composite stratification keys dynamically from manifest metadata.

    Returns:
        group_to_stratum: Mapping from group ID (e.g. identity_id) to stratum key.
        summary_df: DataFrame summarizing strata counts and warning flags.
    """
    valid_cols = [c for c in stratify_columns if c in df.columns]
    if not valid_cols:
        logger.warning(
            f"None of stratify columns {stratify_columns} found in DataFrame. Using 'all' as single stratum."
        )
        unique_groups = df[group_column].dropna().unique()
        group_to_stratum = {g: "default_stratum" for g in unique_groups}
        summary_df = pd.DataFrame([{"stratum": "default_stratum", "identities_count": len(unique_groups), "underrepresented": False}])
        return group_to_stratum, summary_df

    # Extract dominant stratum per group
    group_to_stratum: Dict[str, str] = {}
    for group_id, group_df in df.groupby(group_column):
        # Build composite key using mode/first
        key_parts = []
        for col in valid_cols:
            val = group_df[col].dropna().mode()
            key_parts.append(str(val.iloc[0]) if not val.empty else "unknown")
        composite_key = "_".join(key_parts)
        group_to_stratum[str(group_id)] = composite_key

    # Calculate summary
    stratum_counts: Dict[str, int] = defaultdict(int)
    for stratum in group_to_stratum.values():
        stratum_counts[stratum] += 1

    summary_records = []
    for stratum, count in sorted(stratum_counts.items()):
        is_underrep = count < min_samples_warning
        summary_records.append(
            {
                "stratum": stratum,
                "identities_count": count,
                "underrepresented": is_underrep,
            }
        )
        if is_underrep:
            logger.warning(
                f"Stratum '{stratum}' is underrepresented with only {count} identities (threshold: {min_samples_warning})"
            )

    summary_df = pd.DataFrame(summary_records)
    logger.info(f"Discovered {len(stratum_counts)} distinct strata across {len(group_to_stratum)} identities.")
    return group_to_stratum, summary_df


def build_connected_identity_groups(
    df: pd.DataFrame,
    group_column: str = "identity_id",
    linkage_columns: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Union-find grouping to link source identities and any cross-identity derivatives."""
    parent: Dict[str, str] = {}

    def find(i: str) -> str:
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i: str, j: str):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_j] = root_i

    # Initialize all identities
    all_identities = set(df[group_column].astype(str).unique())
    for ident in all_identities:
        parent[ident] = ident

    # If linkage columns exist (e.g. source_id, target_id)
    if linkage_columns:
        for col in linkage_columns:
            if col in df.columns:
                for _, row in df[[group_column, col]].dropna().iterrows():
                    id_a = str(row[group_column])
                    id_b = str(row[col])
                    if id_b in parent:
                        union(id_a, id_b)

    # Return mapping from identity -> cluster root
    return {ident: find(ident) for ident in all_identities}


def create_identity_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    group_column: str = "identity_id",
    stratify_columns: Optional[List[str]] = None,
    seed: int = 42,
    output_dir: Optional[str | Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create identity-stratified train, validation, and test splits with zero identity leakage."""
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"
    if stratify_columns is None:
        stratify_columns = ["ethnicity", "gender"]

    rng = np.random.RandomState(seed)

    # 1. Connect identities if related derivatives are linked
    ident_to_cluster = build_connected_identity_groups(df, group_column=group_column)

    # 2. Discover dynamic strata
    ident_to_stratum, summary_df = discover_strata(df, stratify_columns=stratify_columns, group_column=group_column)

    # Cluster to stratum mapping
    cluster_to_stratum: Dict[str, str] = {}
    for ident, cluster in ident_to_cluster.items():
        if cluster not in cluster_to_stratum:
            cluster_to_stratum[cluster] = ident_to_stratum.get(ident, "default")

    # Group clusters by stratum
    stratum_to_clusters: Dict[str, List[str]] = defaultdict(list)
    for cluster, stratum in cluster_to_stratum.items():
        stratum_to_clusters[stratum].append(cluster)

    train_clusters: Set[str] = set()
    val_clusters: Set[str] = set()
    test_clusters: Set[str] = set()

    # Stratified allocation per stratum
    for stratum, clusters in stratum_to_clusters.items():
        shuffled = list(clusters)
        rng.shuffle(shuffled)
        n = len(shuffled)

        if n == 1:
            # Single cluster: assign to train
            train_clusters.add(shuffled[0])
        elif n == 2:
            train_clusters.add(shuffled[0])
            test_clusters.add(shuffled[1])
        else:
            n_train = max(1, int(round(n * train_ratio)))
            n_val = max(1, int(round(n * val_ratio)))
            # If train + val would exceed total, adjust
            if n_train + n_val >= n:
                n_val = max(1, n - n_train)

            train_clusters.update(shuffled[:n_train])
            val_clusters.update(shuffled[n_train : n_train + n_val])
            test_clusters.update(shuffled[n_train + n_val :])

    # Map back from clusters to individual identities
    train_identities = {ident for ident, cluster in ident_to_cluster.items() if cluster in train_clusters}
    val_identities = {ident for ident, cluster in ident_to_cluster.items() if cluster in val_clusters}
    test_identities = {ident for ident, cluster in ident_to_cluster.items() if cluster in test_clusters}

    # Strict leakage verification
    assert train_identities.isdisjoint(val_identities), "CRITICAL: Train and Val identities overlap!"
    assert train_identities.isdisjoint(test_identities), "CRITICAL: Train and Test identities overlap!"
    assert val_identities.isdisjoint(test_identities), "CRITICAL: Val and Test identities overlap!"

    # Split rows
    df_str_group = df[group_column].astype(str)
    train_df = df[df_str_group.isin(train_identities)].copy().reset_index(drop=True)
    val_df = df[df_str_group.isin(val_identities)].copy().reset_index(drop=True)
    test_df = df[df_str_group.isin(test_identities)].copy().reset_index(drop=True)

    # Attach split label
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    logger.info(
        f"Splits created: Train={len(train_df)} rows ({len(train_identities)} ids), "
        f"Val={len(val_df)} rows ({len(val_identities)} ids), "
        f"Test={len(test_df)} rows ({len(test_identities)} ids)"
    )

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(out_path / "train.csv", index=False)
        val_df.to_csv(out_path / "val.csv", index=False)
        test_df.to_csv(out_path / "test.csv", index=False)
        summary_df.to_csv(out_path / "strata_summary.csv", index=False)
        logger.info(f"Split CSVs saved to {out_path}")

    return train_df, val_df, test_df
