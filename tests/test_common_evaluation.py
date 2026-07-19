"""Tests for shared-space clustering evaluation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.common_evaluation import (
    COMMON_DISTANCE_CONTRACTS,
    AssignmentRun,
    aggregate_common_evaluation,
    evaluate_common_assignments,
    load_assignment_source,
)


def test_row_level_assignment_source_restores_original_order(tmp_path) -> None:
    """A row-level cluster CSV must be aligned by its explicit row index."""

    source_path = tmp_path / "mvdec_clusters.csv"
    pd.DataFrame(
        {
            "orig_index": [2, 0, 1, 5, 3, 4],
            "cluster": [0, 0, 0, 1, 1, 1],
        }
    ).to_csv(source_path, index=False)

    runs = load_assignment_source("MvDEC", source_path, expected_rows=6)

    assert len(runs) == 1
    assert runs[0].labels.tolist() == [0, 0, 0, 1, 1, 1]


def test_serialized_source_deduplicates_distance_ablation_rows(tmp_path) -> None:
    """The same seed assignment may appear once per native distance contract."""

    source_path = tmp_path / "seed_runs.csv"
    assignments = json.dumps([0, 0, 0, 1, 1, 1])
    pd.DataFrame(
        [
            {
                "random_seed": 40,
                "distance_contract": "asymmetric",
                "cluster_assignments": assignments,
                "status": "ok",
            },
            {
                "random_seed": 40,
                "distance_contract": "symmetric",
                "cluster_assignments": assignments,
                "status": "ok",
            },
        ]
    ).to_csv(source_path, index=False)

    runs = load_assignment_source("MiMvDEC", source_path, expected_rows=6)

    assert len(runs) == 1
    assert runs[0].random_seed == 40
    assert runs[0].labels.tolist() == [0, 0, 0, 1, 1, 1]


def test_common_evaluation_uses_identical_distances_for_every_method() -> None:
    """Only labels may differ when methods are compared in the common space."""

    common_df = pd.DataFrame(
        {
            "x": [0.0, 0.1, 0.2, 4.8, 4.9, 5.0],
            "y": [0.0, 0.1, 0.0, 5.0, 4.9, 5.0],
        }
    )
    assignments = [
        AssignmentRun(
            method="good",
            source_path=Path(__file__),
            source_run_index=0,
            random_seed=42,
            labels=np.array([0, 0, 0, 1, 1, 1]),
        ),
        AssignmentRun(
            method="bad",
            source_path=Path(__file__),
            source_run_index=0,
            random_seed=42,
            labels=np.array([0, 1, 0, 1, 0, 1]),
        ),
    ]

    runs = evaluate_common_assignments(common_df, assignments, n_clusters=2)

    assert len(runs) == 4
    assert set(runs["distance_contract"]) == set(COMMON_DISTANCE_CONTRACTS)
    assert set(runs["status"]) == {"ok"}
    scores = runs.pivot(
        index="method",
        columns="distance_contract",
        values="silhouette_score",
    )
    assert (scores.loc["good"] > scores.loc["bad"]).all()

    summary = aggregate_common_evaluation(runs)
    assert set(summary["method"]) == {"good", "bad"}
    assert set(summary["successful_runs"]) == {1}
    assert set(summary["failed_runs"]) == {0}
