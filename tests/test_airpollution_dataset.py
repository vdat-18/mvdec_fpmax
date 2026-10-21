from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.airpollution_dataset import (
    PAPER_FEATURES,
    PAPER_ROWS,
    add_no2_neighbor_candidates,
    audit_reproduction_readiness,
    feature_contracts,
)


def test_contract_has_exactly_the_paper_13_features() -> None:
    contracts = feature_contracts()
    assert tuple(contract.name for contract in contracts) == PAPER_FEATURES
    assert len(contracts) == 13
    assert {
        contract.name for contract in contracts if contract.status == "blocked"
    } == {
        "satellite_no2",
        "neighbours_no2_variance",
        "neighbours_no2_distance_mean",
        "neighbours_no2_distance_max",
    }


def test_no2_neighbor_candidates_exclude_the_center_cell() -> None:
    grid = pd.DataFrame(
        {
            "x_luref": [0.0, 1.0, 2.0],
            "y_luref": [0.0, 0.0, 0.0],
            "no2": [1.0, 2.0, 4.0],
        }
    )
    output = add_no2_neighbor_candidates(grid, no2_column="no2", radius_meters=10.0)
    assert output["neighbours_no2_count"].tolist() == [2, 2, 2]
    assert output.loc[0, "neighbours_no2_variance"] == pytest.approx(1.0)
    assert output.loc[0, "neighbours_no2_distance_mean"] == pytest.approx(-2.0)
    assert output.loc[0, "neighbours_no2_distance_max"] == pytest.approx(-3.0)
    assert output.loc[1, "neighbours_no2_variance"] == pytest.approx(2.25)


def test_no2_neighbor_candidates_preserve_missing_centers() -> None:
    grid = pd.DataFrame(
        {
            "x_luref": [0.0, 1.0],
            "y_luref": [0.0, 0.0],
            "no2": [np.nan, 2.0],
        }
    )
    output = add_no2_neighbor_candidates(grid, no2_column="no2", radius_meters=10.0)
    assert np.isnan(output.loc[0, "neighbours_no2_distance_mean"])
    assert output.loc[0, "neighbours_no2_count"] == 1
    assert output.loc[1, "neighbours_no2_count"] == 0


def test_readiness_report_never_claims_exact_reproduction(tmp_path: Path) -> None:
    grid_path = tmp_path / "grid.csv"
    pd.DataFrame(
        {
            "grid_index": range(PAPER_ROWS + 1),
            "x_luref": np.arange(PAPER_ROWS + 1),
            "y_luref": np.arange(PAPER_ROWS + 1),
        }
    ).to_csv(grid_path, index=False)
    report = audit_reproduction_readiness(grid_path=grid_path)
    assert report["claim"] == "cleanroom_candidate_not_exact_reproduction"
    assert report["exact_reproduction_ready"] is False
    assert len(report["paper_ambiguities"]) == 4
    assert report["grid"]["paper_row_gap"] == 1
