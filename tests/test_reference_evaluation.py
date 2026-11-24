"""Tests for fixed reference-space scoring."""

import numpy as np
import pandas as pd
import pytest

from pipeline.clustering import compute_mixed_gower_distance
from pipeline.reference_evaluation import (
    build_reference_distance,
    gower_distance_matrix,
    pairwise_ari_stability,
    score_internal,
)


def _overlapping_blobs(seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.repeat([0, 1], 60)
    values = rng.normal(0.0, 1.0, size=(120, 3)) + labels[:, None] * 0.4
    return pd.DataFrame(values, columns=["a", "b", "c"]), labels


def test_label_copy_feature_is_not_rewarded_in_reference_space() -> None:
    """A feature equal to the labels inflates mixed-space silhouette only."""

    X, y = _overlapping_blobs()
    leak = pd.DataFrame({"leak": y})
    mixed = compute_mixed_gower_distance(X, leak)
    from sklearn.metrics import silhouette_score

    mixed_silhouette = silhouette_score(mixed, y, metric="precomputed")
    reference = score_internal(build_reference_distance(X), y).silhouette

    assert mixed_silhouette > reference + 0.2


def test_reference_space_is_independent_of_method_and_deterministic() -> None:
    X, _ = _overlapping_blobs()

    first = build_reference_distance(X)
    second = build_reference_distance(X)

    assert np.array_equal(first.distance, second.distance)
    assert np.array_equal(first.row_index, second.row_index)


def test_invalid_partition_scores_nan() -> None:
    X, _ = _overlapping_blobs()
    scores = score_internal(build_reference_distance(X), np.zeros(len(X), dtype=int))

    assert np.isnan(scores.silhouette)
    assert np.isnan(scores.davies_bouldin)
    assert np.isnan(scores.calinski_harabasz)


def test_singleton_cluster_scores_nan() -> None:
    X, y = _overlapping_blobs()
    labels = y.copy()
    labels[:] = 0
    labels[0] = 1

    scores = score_internal(build_reference_distance(X), labels)

    assert np.isnan(scores.silhouette)


def test_subsample_is_deterministic_and_bounded() -> None:
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(300, 2)), columns=["a", "b"])

    first = build_reference_distance(X, max_rows=100, random_state=0)
    second = build_reference_distance(X, max_rows=100, random_state=0)

    assert first.distance.shape == (100, 100)
    assert np.array_equal(first.row_index, second.row_index)
    assert len(set(first.row_index.tolist())) == 100


def test_scores_use_subsampled_labels() -> None:
    rng = np.random.default_rng(2)
    labels = np.repeat([0, 1], 150)
    X = pd.DataFrame(
        rng.normal(size=(300, 2)) + labels[:, None] * 6.0, columns=["a", "b"]
    )
    space = build_reference_distance(X, max_rows=100, random_state=0)

    scores = score_internal(space, labels)

    assert scores.silhouette > 0.5


def test_mixed_numeric_and_categorical_columns_are_supported() -> None:
    X = pd.DataFrame(
        {
            "num": [0.0, 0.1, 5.0, 5.1],
            "cat": ["u", "u", "v", "v"],
        }
    )

    space = build_reference_distance(X)

    assert space.distance[0, 1] < space.distance[0, 2]
    assert space.distance[0, 0] == 0.0


def test_pairwise_ari_stability_identical_and_relabelled() -> None:
    a = np.array([0, 0, 1, 1, 2, 2])
    relabelled = np.array([5, 5, 3, 3, 9, 9])

    assert pairwise_ari_stability([a, a]) == 1.0
    assert pairwise_ari_stability([a, relabelled]) == 1.0


def test_cluster_lost_by_subsampling_scores_nan() -> None:
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(500, 2)), columns=["a", "b"])
    space = build_reference_distance(X, max_rows=50, random_state=0)
    outside = np.setdiff1d(np.arange(500), space.row_index)
    labels = np.zeros(500, dtype=int)
    labels[250:] = 1
    labels[outside[:2]] = 2  # third cluster exists only outside the sample

    scores = score_internal(space, labels)

    assert 2 not in set(labels[space.row_index].tolist())
    assert np.isnan(scores.silhouette)


def test_cosine_reference_space_ignores_row_scale() -> None:
    X = pd.DataFrame({"a": [1.0, 5.0, 0.0], "b": [0.0, 0.0, 1.0]})

    space = build_reference_distance(X, metric="cosine")

    assert space.distance[0, 1] == pytest.approx(0.0)
    assert space.distance[0, 2] == pytest.approx(1.0)


def test_cosine_space_subsamples_and_scores_full_labelling() -> None:
    rng = np.random.default_rng(4)
    labels = np.repeat([0, 1], 200)
    directions = np.where(labels[:, None] == 0, [1.0, 0.1], [0.1, 1.0])
    X = pd.DataFrame(
        np.abs(directions + rng.normal(0, 0.05, size=(400, 2))), columns=["a", "b"]
    )
    space = build_reference_distance(X, max_rows=100, metric="cosine")

    scores = score_internal(space, labels)

    assert space.distance.shape == (100, 100)
    assert scores.silhouette > 0.5


def test_unknown_metric_is_rejected() -> None:
    with pytest.raises(ValueError, match="metric"):
        build_reference_distance(pd.DataFrame({"a": [1.0, 2.0]}), metric="l2")


def test_gower_distance_matrix_accepts_string_columns() -> None:
    df = pd.DataFrame({"num": [0.0, 1.0], "cat": pd.array(["u", "v"], dtype="string")})

    distance = gower_distance_matrix(df)

    assert distance.shape == (2, 2)
    assert distance[0, 1] > 0
