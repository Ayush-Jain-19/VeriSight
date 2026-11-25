"""Unit tests for out-of-fold data integrity and leakage prevention in fusion training."""

import numpy as np
import pandas as pd
import pytest
from src.training.oof_embeddings import generate_identity_folds


@pytest.fixture
def mock_train_df():
    """Create a sample training set with 15 identities across 3 strata."""
    records = []
    strata_map = {
        f"id_{i:02d}": ("Caucasian" if i < 5 else ("African" if i < 10 else "Asian"), "man" if i % 2 == 0 else "woman")
        for i in range(15)
    }

    for ident, (eth, gen) in strata_map.items():
        for sample_i in range(3):
            records.append(
                {
                    "identity_id": ident,
                    "ethnicity": eth,
                    "gender": gen,
                    "video_path": f"/path/{ident}_{sample_i}.mp4",
                    "overall_label": "REAL" if sample_i == 0 else "FAKE",
                }
            )

    return pd.DataFrame(records)


def test_oof_folds_zero_identity_leakage(mock_train_df):
    """Verify that every fold strictly segregates training identities from held-out identities."""
    n_folds = 3
    folds = generate_identity_folds(mock_train_df, n_folds=n_folds, seed=42)

    assert len(folds) == n_folds

    all_held_out_indices = []
    for k, (train_idx, held_out_idx) in enumerate(folds):
        train_ids = set(mock_train_df.iloc[train_idx]["identity_id"].unique())
        held_out_ids = set(mock_train_df.iloc[held_out_idx]["identity_id"].unique())

        # Crucial anti-leakage test
        assert train_ids.isdisjoint(held_out_ids), f"Identity overlap detected in fold {k}!"
        all_held_out_indices.extend(held_out_idx.tolist())

    # Verify that the union of all held-out folds covers every sample in train_df exactly once
    assert len(all_held_out_indices) == len(mock_train_df)
    assert len(set(all_held_out_indices)) == len(mock_train_df)
