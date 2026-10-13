"""Tests for the DEKM 2021 released-code protocol helpers."""

from types import SimpleNamespace

import numpy as np
import pytest

from pipeline.public_data import PAPER_RMS_NORMALIZATION, RELEASE_L2_NORMALIZATION
from representation_learning.DEKM_dense import (
    DETERMINISTIC_RUNTIME_POLICY,
    LATEST_AUDITED_DENSE_SHA256,
    LATEST_AUDITED_SOURCE_REVISION,
    PAPER_ERA_DENSE_SHA256,
    PAPER_ERA_SOURCE_REVISION,
    build_config,
    enable_tensorflow_determinism,
    method_name,
    runtime_metadata,
)
from representation_learning.dekm_protocol import (
    DEKM_20NEWS_SKLEARN_2021_PROTOCOL_ID,
    DEKM_FULL_BATCH_2021_PROTOCOL_ID,
    DEKM_MODERN_PROTOCOL_ID,
    DEKM_PAPER_TEXT_FULL_BATCH_2021_PROTOCOL_ID,
    DEKM_RELEASE_2021_PROTOCOL_ID,
    DEKM_RELEASE_MINIBATCH_2021_PROTOCOL_ID,
    DekmConfig,
    assignment_change_count,
    build_frozen_greedy_target,
    config_hash,
    sequential_batch_indices,
    sorted_eigendecomposition,
    within_cluster_scatter,
)


def config_args(**overrides: object) -> SimpleNamespace:
    """Return CLI-like values for protocol configuration tests."""

    values = {
        "dataset": "REUTERS",
        "protocol": "modern-deterministic",
        "pretrain_epochs": 200,
        "pretrain_batch_size": 256,
        "batch_size": None,
        "update_interval": 10,
        "max_refinement_steps": 14_000,
        "assignment_tolerance": None,
        "initial_kmeans_n_init": 100,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_source_provenance_distinguishes_paper_era_from_latest_audit() -> None:
    assert PAPER_ERA_SOURCE_REVISION == "129b7d08b545f02d920ccff0948ec44b654b3158"
    assert PAPER_ERA_DENSE_SHA256 == (
        "1298eff9d4de23f75dccc8a341945f98e908521cf100c440b43a6521d78e5185"
    )
    assert LATEST_AUDITED_SOURCE_REVISION == (
        "e58f506922137abe2905aa0e4eb04a6bf952e80d"
    )
    assert LATEST_AUDITED_DENSE_SHA256 == (
        "8536630e35b05e5ad8cd0f9235c2012537f31e5b3b2a5b3313ee1c246cb3dec9"
    )
    assert PAPER_ERA_SOURCE_REVISION != LATEST_AUDITED_SOURCE_REVISION


def test_modern_protocol_defaults_are_explicit() -> None:
    config = build_config(config_args())

    assert config.protocol_id == DEKM_MODERN_PROTOCOL_ID
    assert config.kernel_initializer == "glorot_uniform"
    assert config.eigen_solver == "symmetric_eigh"
    assert config.kmeans_random_state == "seeded"
    assert config.deterministic_runtime is True
    assert config.data_shuffle == "direct_seed"
    assert config.input_normalization == RELEASE_L2_NORMALIZATION
    assert config.refinement_batch_size == 256
    assert config.assignment_change_tolerance == 0.001


def test_release_2021_reuters_defaults_match_released_code() -> None:
    config = build_config(config_args(protocol="release-2021"))
    contract = config.contract()

    assert config.protocol_id == DEKM_RELEASE_2021_PROTOCOL_ID
    assert config.kernel_initializer == "uniform"
    assert config.eigen_solver == "general_eig"
    assert config.kmeans_random_state == "global"
    assert config.deterministic_runtime is False
    assert config.data_shuffle == "release_random_0_99"
    assert config.input_normalization == RELEASE_L2_NORMALIZATION
    assert config.refinement_batch_size == 10_000
    assert config.assignment_change_tolerance == 0.005
    assert contract["architecture"]["kernel_initializer"] == "uniform"
    assert contract["refinement"]["eigen_solver"] == "general_eig"


def test_release_2021_non_reuters_and_cli_overrides() -> None:
    config = build_config(
        config_args(
            dataset="20NEWS",
            protocol="release-2021",
            batch_size=512,
            assignment_tolerance=0.01,
        )
    )

    assert config.refinement_batch_size == 512
    assert config.assignment_change_tolerance == 0.01


def test_20news_sklearn_protocol_is_source_faithful() -> None:
    config = build_config(
        config_args(dataset="20NEWS", protocol="release-2021-sklearn-20news")
    )

    assert config.protocol_id == DEKM_20NEWS_SKLEARN_2021_PROTOCOL_ID
    assert config.refinement_batch_size == 256
    assert config.input_normalization == RELEASE_L2_NORMALIZATION
    assert config.kernel_initializer == "uniform"
    assert config.eigen_solver == "general_eig"


def test_20news_sklearn_protocol_rejects_other_datasets() -> None:
    with pytest.raises(ValueError, match="valid only for 20NEWS"):
        build_config(
            config_args(dataset="RCV1", protocol="release-2021-sklearn-20news")
        )


@pytest.mark.parametrize(
    ("dataset", "expected_samples"),
    [("REUTERS", 10_000), ("20NEWS", 18_846), ("RCV1", 10_000)],
)
def test_full_batch_protocol_uses_the_complete_dataset(
    dataset: str,
    expected_samples: int,
) -> None:
    config = build_config(config_args(dataset=dataset, protocol="full-batch-2021"))

    assert config.protocol_id == DEKM_FULL_BATCH_2021_PROTOCOL_ID
    assert config.refinement_batch_size == expected_samples
    assert config.kernel_initializer == "uniform"
    assert config.eigen_solver == "general_eig"
    assert config.assignment_change_tolerance == 0.005
    assert config.contract()["refinement"]["batching"] == ("full_dataset_each_update")
    assert method_name(config) == "DEKM_F"


def test_official_source_minibatch_protocol_uses_256_rows() -> None:
    config = build_config(
        config_args(dataset="REUTERS", protocol="release-minibatch-2021")
    )

    assert config.protocol_id == DEKM_RELEASE_MINIBATCH_2021_PROTOCOL_ID
    assert config.refinement_batch_size == 256
    assert config.contract()["refinement"]["batching"] == "sequential_wraparound"
    assert method_name(config) == "DEKM"


def test_paper_text_full_batch_protocol_is_isolated_from_source_contract() -> None:
    config = build_config(
        config_args(dataset="RCV1", protocol="paper-text-full-batch-2021")
    )

    assert config.protocol_id == DEKM_PAPER_TEXT_FULL_BATCH_2021_PROTOCOL_ID
    assert config.kernel_initializer == "glorot_uniform"
    assert config.eigen_solver == "symmetric_eigh"
    assert config.kmeans_random_state == "seeded"
    assert config.deterministic_runtime is True
    assert config.data_shuffle == "direct_seed"
    assert config.input_normalization == PAPER_RMS_NORMALIZATION
    assert config.refinement_batch_size == 10_000
    assert config.assignment_change_tolerance == 0.001
    assert config.contract()["refinement"]["batching"] == "full_dataset_each_update"
    assert config.contract()["preprocessing"]["input_normalization"] == (
        PAPER_RMS_NORMALIZATION
    )
    assert method_name(config) == "DEKM_F_PAPER_TEXT"


def test_config_validation_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        DekmConfig(pretrain_epochs=0).validate()

    with pytest.raises(ValueError, match="range"):
        DekmConfig(assignment_change_tolerance=1.0).validate()

    with pytest.raises(ValueError, match="input-normalization"):
        DekmConfig(input_normalization="unknown").validate()


def test_config_hash_is_stable_and_sensitive_to_runtime_values() -> None:
    config = DekmConfig()

    assert config_hash(config) == config_hash(DekmConfig())
    assert config_hash(config) != config_hash(DekmConfig(pretrain_epochs=1))
    release_config = build_config(config_args(protocol="release-2021"))
    assert config_hash(config) != config_hash(release_config)
    assert len(config_hash(config)) == 64


def test_sorted_eigendecomposition_orders_values_ascending() -> None:
    matrix = np.array([[3.0, 0.0], [0.0, 1.0]])

    eigenvalues, eigenvectors = sorted_eigendecomposition(matrix)

    np.testing.assert_allclose(eigenvalues, [1.0, 3.0])
    np.testing.assert_allclose(
        eigenvectors.T @ matrix @ eigenvectors, np.diag(eigenvalues)
    )


def test_general_eigendecomposition_matches_symmetric_eigenpairs() -> None:
    matrix = np.array([[3.0, 1.0], [1.0, 2.0]])

    expected, _ = sorted_eigendecomposition(matrix, solver="symmetric_eigh")
    actual, eigenvectors = sorted_eigendecomposition(matrix, solver="general_eig")

    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(
        matrix @ eigenvectors, eigenvectors * actual[np.newaxis, :]
    )


def test_sorted_eigendecomposition_rejects_unknown_solver() -> None:
    with pytest.raises(ValueError, match="Unsupported DEKM eigen solver"):
        sorted_eigendecomposition(np.eye(2), solver="unknown")


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


def test_tensorflow_determinism_is_enabled_and_reported() -> None:
    calls: list[str] = []
    fake_tf = SimpleNamespace(
        __version__="test",
        config=SimpleNamespace(
            experimental=SimpleNamespace(
                enable_op_determinism=lambda: calls.append("enabled")
            ),
            list_physical_devices=lambda kind: [],
        ),
    )

    enable_tensorflow_determinism(fake_tf)
    metadata = runtime_metadata(fake_tf)

    assert calls == ["enabled"]
    assert metadata["deterministic_runtime_policy"] == DETERMINISTIC_RUNTIME_POLICY
    assert metadata["tensorflow_op_determinism"] is True
    assert metadata["tf_data_deterministic"] is True


def test_runtime_metadata_reports_nondeterministic_release_policy() -> None:
    fake_tf = SimpleNamespace(
        __version__="test",
        config=SimpleNamespace(list_physical_devices=lambda kind: []),
    )

    metadata = runtime_metadata(fake_tf, deterministic_runtime=False)

    assert metadata["deterministic_runtime_policy"] == "disabled"
    assert metadata["tensorflow_op_determinism"] is False
    assert metadata["tf_data_deterministic"] is False
