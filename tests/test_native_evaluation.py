"""Tests for native continuous-representation evaluation."""

import pickle

import numpy as np
import pandas as pd
import pytest

from pipeline.native_evaluation import (
    NATIVE_NUMERIC_GOWER_CONTRACT,
    aggregate_native_evaluation,
    evaluate_native_runs,
    load_native_source,
    parse_source_spec,
)


def _representation() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [4.8, 5.0],
            [4.9, 4.9],
            [5.0, 5.0],
        ]
    )


def test_pickle_native_source_loads_representation_labels_and_seed(tmp_path) -> None:
    """A trusted artifact must provide a numeric representation and labels."""

    source_path = tmp_path / "dekm_seed_11.pkl"
    with source_path.open("wb") as file:
        pickle.dump(
            {
                "h_fused": _representation(),
                "labels": np.array([0, 0, 0, 1, 1, 1]),
                "n_clusters": 2,
                "config": {"seed": 11, "n_clusters": 2},
            },
            file,
        )

    method, path, key = parse_source_spec(f"DEKM={source_path}")
    run = load_native_source(method, path, key, n_clusters=2)

    assert run.method == "DEKM"
    assert run.random_seed == 11
    assert run.representation.shape == (6, 2)
    assert run.labels.tolist() == [0, 0, 0, 1, 1, 1]


def test_csv_native_source_restores_row_order_and_excludes_metadata(tmp_path) -> None:
    """CSV metadata columns must not leak into the native representation."""

    source_path = tmp_path / "mvdec_clusters.csv"
    pd.DataFrame(
        {
            "orig_index": [2, 0, 1, 5, 3, 4],
            "h_0": [0.2, 0.0, 0.1, 5.0, 4.8, 4.9],
            "h_1": [0.0, 0.0, 0.1, 5.0, 5.0, 4.9],
            "cluster": [0, 0, 0, 1, 1, 1],
            "random_seed": [42] * 6,
        }
    ).to_csv(source_path, index=False)

    run = load_native_source("MvDEC", source_path, "h_fused", n_clusters=2)

    assert run.random_seed == 42
    assert run.representation.columns.tolist() == ["h_0", "h_1"]
    assert run.labels.tolist() == [0, 0, 0, 1, 1, 1]


def test_native_evaluation_aggregates_repeated_method_runs(tmp_path) -> None:
    """Repeated artifacts with one method name must produce mean and seed std."""

    runs = []
    for seed, labels in (
        (40, [0, 0, 0, 1, 1, 1]),
        (41, [0, 1, 0, 1, 0, 1]),
    ):
        source_path = tmp_path / f"dekm_seed_{seed}.pkl"
        with source_path.open("wb") as file:
            pickle.dump(
                {
                    "h_fused": _representation(),
                    "labels": np.asarray(labels),
                    "config": {"seed": seed, "n_clusters": 2},
                },
                file,
            )
        runs.append(load_native_source("DEKM", source_path, "h_fused", n_clusters=2))

    evaluated = evaluate_native_runs(runs, n_clusters=2)
    summary = aggregate_native_evaluation(evaluated)

    assert len(evaluated) == 2
    assert set(evaluated["distance_contract"]) == {NATIVE_NUMERIC_GOWER_CONTRACT}
    assert evaluated.loc[0, "silhouette_score"] > evaluated.loc[1, "silhouette_score"]
    assert summary.loc[0, "requested_runs"] == 2
    assert summary.loc[0, "successful_runs"] == 2
    assert summary.loc[0, "silhouette_std_across_seeds"] > 0


def test_native_artifact_rejects_another_cluster_count(tmp_path) -> None:
    """A source generated for another K must not enter the comparison table."""

    source_path = tmp_path / "wrong_k.pkl"
    with source_path.open("wb") as file:
        pickle.dump(
            {
                "h_fused": _representation(),
                "labels": np.array([0, 0, 0, 1, 1, 1]),
                "n_clusters": 3,
            },
            file,
        )

    with pytest.raises(ValueError, match="cluster count"):
        load_native_source("DEKM", source_path, "h_fused", n_clusters=2)
