"""Regression tests for the Tiki preprocessing pipeline."""

import numpy as np
import pandas as pd

from config import DATA_DIR, PREPROCESSED_DATA_DIR
from data_preprocessing.features import (
    filter_valid_year_joined_rows,
    wilson_review_share_score,
)
from data_preprocessing.pipeline import preprocess_tiki_data


def test_filter_valid_year_joined_rows_preserves_source_indices() -> None:
    """Invalid years are removed without changing valid source indices."""

    raw_df = pd.DataFrame(
        {"Year Joined": [2020, 0, 2027]},
        index=[10, 20, 30],
    )

    cleaned_df = filter_valid_year_joined_rows(raw_df, reference_year=2026)

    assert cleaned_df.index.tolist() == [10]


def test_invalid_review_share_returns_nan() -> None:
    """Inconsistent review totals are marked missing before row filtering."""

    score = wilson_review_share_score(positive_count=50, total_count=20)

    assert np.isnan(score)


def test_raw_tiki_data_reproduces_canonical_preprocessed_csv(tmp_path) -> None:
    """The recovered raw workbook reproduces the canonical feature matrix."""

    expected_df = pd.read_csv(PREPROCESSED_DATA_DIR / "tiki_preprocessed.csv")
    mapping_df = pd.read_csv(PREPROCESSED_DATA_DIR / "tiki_row_mapping.csv")
    output_path = tmp_path / "tiki_preprocessed.csv"

    result = preprocess_tiki_data(
        input_path=DATA_DIR / "raw_data" / "tiki_raw_data.xlsx",
        output_path=output_path,
    )
    actual_df = pd.read_csv(output_path)

    assert result.raw_df.shape == (1823, 16)
    assert result.selected_features_df.shape == (1822, 7)
    assert result.dbscan.clusters_info == {-1: 22, 0: 1799}
    assert result.dbscan.non_noise_df.index.is_unique
    assert mapping_df["merged_row_index"].is_unique
    assert set(result.dbscan.non_noise_df.index) == set(mapping_df["merged_row_index"])
    assert actual_df.columns.tolist() == expected_df.columns.tolist()
    np.testing.assert_allclose(actual_df, expected_df, rtol=0, atol=1e-9)
