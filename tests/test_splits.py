"""Unit tests for identity-aware dataset splitting and dynamic stratification."""

import pandas as pd
import pytest
from src.data.splits import create_identity_splits, discover_strata, build_connected_identity_groups


@pytest.fixture
def sample_manifest_df():
    """Create a realistic synthetic manifest DataFrame across multiple identities and strata."""
    records = []
    # 10 identities across 3 ethnicities and 2 genders
    data_spec = [
        ("id01", "Caucasian", "man", 4),
        ("id02", "Caucasian", "man", 3),
        ("id03", "Caucasian", "woman", 4),
        ("id04", "Caucasian", "woman", 2),
        ("id05", "African", "man", 5),
        ("id06", "African", "man", 3),
        ("id07", "African", "woman", 4),
        ("id08", "Asian", "man", 4),
        ("id09", "Asian", "woman", 3),
        ("id10", "Asian", "woman", 2),
    ]

    for ident, eth, gen, count in data_spec:
        for i in range(count):
            cat = "RealVideo-RealAudio" if i == 0 else "FakeVideo-FakeAudio"
            records.append(
                {
                    "video_path": f"/fake/path/{cat}/{eth}/{gen}/{ident}/{i:04d}.mp4",
                    "category": cat,
                    "ethnicity": eth,
                    "gender": gen,
                    "identity_id": ident,
                    "method": "none" if i == 0 else "wav2lip",
                    "video_label": "REAL" if i == 0 else "FAKE",
                    "audio_label": "REAL" if i == 0 else "FAKE",
                    "overall_label": "REAL" if i == 0 else "FAKE",
                    "filename": f"{i:04d}.mp4",
                }
            )

    return pd.DataFrame(records)


def test_discover_strata(sample_manifest_df):
    """Test dynamic strata discovery without hardcoded values."""
    group_to_stratum, summary_df = discover_strata(
        sample_manifest_df,
        stratify_columns=["ethnicity", "gender"],
        group_column="identity_id",
        min_samples_warning=3,
    )

    assert len(group_to_stratum) == 10
    # Should discover combinations dynamically: Caucasian_man, Caucasian_woman, etc.
    unique_strata = summary_df["stratum"].unique()
    assert "Caucasian_man" in unique_strata
    assert "African_woman" in unique_strata
    assert "Asian_man" in unique_strata


def test_zero_identity_leakage(sample_manifest_df):
    """Test strict identity separation between train, val, and test splits."""
    train_df, val_df, test_df = create_identity_splits(
        sample_manifest_df,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=123,
    )

    train_ids = set(train_df["identity_id"].unique())
    val_ids = set(val_df["identity_id"].unique())
    test_ids = set(test_df["identity_id"].unique())

    # Crucial assertions: zero leakage!
    assert train_ids.isdisjoint(val_ids), f"Leakage detected between Train and Val: {train_ids & val_ids}"
    assert train_ids.isdisjoint(test_ids), f"Leakage detected between Train and Test: {train_ids & test_ids}"
    assert val_ids.isdisjoint(test_ids), f"Leakage detected between Val and Test: {val_ids & test_ids}"

    # Verify all samples accounted for
    total_split_rows = len(train_df) + len(val_df) + len(test_df)
    assert total_split_rows == len(sample_manifest_df)


def test_connected_identity_groups():
    """Test that source identities and cross-identity derivatives are clustered together."""
    df = pd.DataFrame(
        [
            {"identity_id": "id01", "source_id": "id02"},
            {"identity_id": "id02", "source_id": None},
            {"identity_id": "id03", "source_id": None},
        ]
    )

    clusters = build_connected_identity_groups(df, group_column="identity_id", linkage_columns=["source_id"])
    assert clusters["id01"] == clusters["id02"]
    assert clusters["id01"] != clusters["id03"]
