"""Tests for the DEKM 2021 released-code protocol helpers."""

import numpy as np
import pytest

from representation_learning.dekm_protocol import (
    DekmConfig,
    assignment_change_count,
    build_frozen_greedy_target,
    config_hash,
    sequential_batch_indices,
    sorted_eigendecomposition,
    within_cluster_scatter,
)


def test_config_validation_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        DekmConfig(pretrain_epochs=0).validate()

    with pytest.raises(ValueError, match="range"):
        DekmConfig(assignment_change_tolerance=1.0).validate()


def test_config_hash_is_stable_and_sensitive_to_runtime_values() -> None:
    config = DekmConfig()

    assert config_hash(config) == config_hash(DekmConfig())
    assert config_hash(config) != config_hash(DekmConfig(pretrain_epochs=1))
    assert len(config_hash(config)) == 64


def test_sorted_eigendecomposition_orders_values_ascending() -> None:
    matrix = np.array([[3.0, 0.0], [0.0, 1.0]])

    eigenvalues, eigenvectors = sorted_eigendecomposition(matrix)

    np.testing.assert_allclose(eigenvalues, [1.0, 3.0])
    np.testing.assert_allclose(
        eigenvectors.T @ matrix @ eigenvectors, np.diag(eigenvalues)
    )


def test_within_cluster_scatter_matches_manual_result() -> None:
    embeddings = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 0, 1])
    centroids = np.array([[1.0, 0.0], [0.0, 1.0]])

    scatter = within_cluster_scatter(embeddings, labels, centroids)

    np.testing.assert_allclose(scatter, [[2.0, 0.0], [0.0, 0.0]])


def test_greedy_target_changes_only_last_dimension() -> None:
    embeddings = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    centroids = np.array([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]])

    target = build_frozen_greedy_target(embeddings, centroids, np.array([1, 0]))

    np.testing.assert_allclose(target, [[1.0, 2.0, 22.0], [4.0, 5.0, 12.0]])
    np.testing.assert_array_equal(embeddings, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def test_assignment_change_count_ignores_cluster_id_permutations() -> None:
    previous = np.array([0, 0, 1, 1])
    current = np.array([1, 1, 0, 0])

    assert assignment_change_count(previous, current, n_clusters=2) == 0


def test_assignment_change_count_treats_initial_assignments_as_changed() -> None:
    previous = np.full(4, -1)
    current = np.array([0, 0, 1, 1])

    assert assignment_change_count(previous, current, n_clusters=2) == 4


def test_sequential_batches_wrap_after_the_final_partial_batch() -> None:
    first, next_index = sequential_batch_indices(5, 2, 0)
    second, next_index = sequential_batch_indices(5, 2, next_index)
    third, next_index = sequential_batch_indices(5, 2, next_index)
    wrapped, final_index = sequential_batch_indices(5, 2, next_index)

    np.testing.assert_array_equal(first, [0, 1])
    np.testing.assert_array_equal(second, [2, 3])
    np.testing.assert_array_equal(third, [4])
    np.testing.assert_array_equal(wrapped, [0, 1])
    assert final_index == 1
