"""Tests for the Air Pollution preprocessing forensic audit."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.airpollution_forensic import (
    PAPER_COLUMNS,
    PAPER_TABLE4,
    SOURCE_COLUMNS,
    Candidate,
    build_candidates,
    evaluate_candidate,
    load_airpollution_csv,
    match_table4,
)


def _valid_frame(rows: int = 32) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    matrix = rng.uniform(0.01, 1.0, size=(rows, len(SOURCE_COLUMNS)))
    matrix[:, SOURCE_COLUMNS.index("highway_primary")] *= 50_000
    matrix[:, SOURCE_COLUMNS.index("neighbours_no2_variance")] *= 1e-10
    return pd.DataFrame(matrix, columns=SOURCE_COLUMNS)


def test_loader_rejects_wrong_schema_and_non_finite_values(tmp_path: Path) -> None:
    wrong_schema = _valid_frame().rename(columns={SOURCE_COLUMNS[0]: "wrong"})
    wrong_path = tmp_path / "wrong.csv"
    wrong_schema.to_csv(wrong_path, index=False)

    with pytest.raises(ValueError, match="Unexpected Air Pollution schema"):
        load_airpollution_csv(wrong_path)

    non_finite = _valid_frame()
    non_finite.iloc[0, 0] = np.nan
    non_finite_path = tmp_path / "non_finite.csv"
    non_finite.to_csv(non_finite_path, index=False)

    with pytest.raises(ValueError, match="NaN or infinite"):
        load_airpollution_csv(non_finite_path)


def test_primary_transforms_preserve_shape_and_finite_values() -> None:
    frame = _valid_frame()

    candidates = build_candidates(frame)

    assert len(candidates) >= 10
    assert all(candidate.matrix.shape == frame.shape for candidate in candidates)
    assert all(np.isfinite(candidate.matrix).all() for candidate in candidates)
    assert all(not candidate.diagnostic_only for candidate in candidates)


def test_table4_match_recovers_exact_rows_in_source_column_order() -> None:
    source_order = [PAPER_COLUMNS.index(column) for column in SOURCE_COLUMNS]
    exact_source_matrix = PAPER_TABLE4[:, source_order]

    match = match_table4(exact_source_matrix)

    assert match.exact_rounded_rows == len(PAPER_TABLE4)
    assert match.mean_rmse == pytest.approx(0.0)
    assert match.max_rmse == pytest.approx(0.0)
    assert sorted(match.source_row_indices) == list(range(len(PAPER_TABLE4)))


def test_candidate_evaluation_is_deterministic_and_finite() -> None:
    candidate = Candidate(
        name="synthetic",
        description="Synthetic deterministic control.",
        matrix=_valid_frame(rows=40).to_numpy(),
    )

    first = evaluate_candidate(
        candidate,
        seeds=(7,),
        n_init=2,
        compute_elbow=False,
    )
    second = evaluate_candidate(
        candidate,
        seeds=(7,),
        n_init=2,
        compute_elbow=False,
    )

    assert first.silhouette_mean == pytest.approx(second.silhouette_mean)
    assert np.isfinite(first.silhouette_mean)
    assert np.isfinite(first.scale_factor_to_paper_k4_distortion)
    assert first.cluster_sizes_by_seed == second.cluster_sizes_by_seed
    assert first.elbow_inertia == {}


def test_weight_grid_is_explicitly_diagnostic_only() -> None:
    candidates = build_candidates(
        _valid_frame(),
        include_weight_grid=True,
        weight_values=(0.5, 2.0),
    )
    diagnostics = [candidate for candidate in candidates if candidate.diagnostic_only]

    assert len(diagnostics) == 4
    assert all(
        candidate.name.startswith("diagnostic_weighted_") for candidate in diagnostics
    )
