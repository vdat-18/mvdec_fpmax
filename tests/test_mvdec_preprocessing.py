import importlib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.data import file_sha256, load_mvdec_result
from pipeline.external_metrics import compute_external_metrics
from pipeline.mvdec_runs import (
    build_run_id,
    canonical_config_hash,
    write_run_manifest,
)


def _load_mvdec(monkeypatch):
    pytest.importorskip("tensorflow")
    representation_dir = (
        Path(__file__).resolve().parents[1] / "src" / "representation_learning"
    )
    monkeypatch.syspath_prepend(str(representation_dir))
    return importlib.import_module("MVDEC_dense")


def _restore_original_order(values, shuffled_indices):
    restored = np.empty_like(values)
    restored[shuffled_indices] = values
    return restored


def test_airpollution_minmax_scales_each_column_before_shuffle(tmp_path, monkeypatch):
    mvdec = _load_mvdec(monkeypatch)
    data_path = tmp_path / "airpollution.csv"
    source = pd.DataFrame(
        {
            "large_distance": [10.0, 20.0, 30.0],
            "small_pollution": [0.00001, 0.00002, 0.00003],
            "constant": [7.0, 7.0, 7.0],
        }
    )
    source.to_csv(data_path, index=False)

    scaled, indices, columns, raw, metadata = mvdec.get_x_unlabeled_csv(
        data_path,
        log_print=False,
        shuffle_seed=42,
        scaling_method="minmax",
        include_preprocessing=True,
    )

    np.testing.assert_allclose(
        _restore_original_order(scaled, indices),
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [1.0, 1.0, 0.0]],
    )
    np.testing.assert_allclose(
        _restore_original_order(raw, indices),
        source.to_numpy(dtype=np.float32),
    )
    assert columns == list(source.columns)
    assert metadata["method"] == "minmax"
    assert metadata["feature_range"] == [0.0, 1.0]
    assert metadata["constant_features"] == ["constant"]
    assert len(metadata["source_sha256"]) == 64


def test_none_preprocessing_preserves_input_values(tmp_path, monkeypatch):
    mvdec = _load_mvdec(monkeypatch)
    data_path = tmp_path / "tiki.csv"
    source = pd.DataFrame({"feature_a": [-2.0, 5.0], "feature_b": [10.0, 20.0]})
    source.to_csv(data_path, index=False)

    values, indices, _, raw, metadata = mvdec.get_x_unlabeled_csv(
        data_path,
        log_print=False,
        shuffle_seed=42,
        scaling_method="none",
        include_preprocessing=True,
    )

    np.testing.assert_allclose(values, raw)
    np.testing.assert_allclose(
        _restore_original_order(values, indices),
        source.to_numpy(dtype=np.float32),
    )
    assert metadata["method"] == "none"
    assert metadata["feature_range"] is None


def test_standard_preprocessing_scales_columns_before_shuffle(tmp_path, monkeypatch):
    mvdec = _load_mvdec(monkeypatch)
    data_path = tmp_path / "airpollution.csv"
    source = pd.DataFrame(
        {
            "distance": [10.0, 20.0, 30.0],
            "pollution": [0.1, 0.2, 0.3],
            "constant": [7.0, 7.0, 7.0],
        }
    )
    source.to_csv(data_path, index=False)

    scaled, indices, _, _, metadata = mvdec.get_x_unlabeled_csv(
        data_path,
        log_print=False,
        shuffle_seed=42,
        scaling_method="standard",
        include_preprocessing=True,
    )
    restored = _restore_original_order(scaled, indices)

    np.testing.assert_allclose(restored[:, :2].mean(axis=0), 0.0, atol=1e-6)
    np.testing.assert_allclose(restored[:, :2].std(axis=0), 1.0, atol=1e-6)
    np.testing.assert_allclose(restored[:, 2], 0.0)
    np.testing.assert_allclose(metadata["data_mean"], [20.0, 0.2, 7.0])
    np.testing.assert_allclose(
        metadata["data_std"],
        source.to_numpy(dtype=float).std(axis=0),
    )
    assert metadata["method"] == "standard"
    assert metadata["feature_range"] is None


def test_dataset_registry_scales_only_airpollution(monkeypatch):
    mvdec = _load_mvdec(monkeypatch)

    assert mvdec.UNLABELED_DATASETS["AIRPOLLUTION"]["scaling_method"] == "minmax"
    assert mvdec.UNLABELED_DATASETS["TIKI"]["scaling_method"] == "none"
    assert mvdec.resolve_unlabeled_scaling_method("AIRPOLLUTION") == "minmax"
    assert mvdec.resolve_unlabeled_scaling_method("AIRPOLLUTION", "none") == "none"
    assert (
        mvdec.resolve_unlabeled_scaling_method("AIRPOLLUTION", "standard") == "standard"
    )
    assert mvdec._preprocessing_input_space(None) == "x"
    assert mvdec._preprocessing_input_space({"method": "minmax"}) == "x_minmax"
    assert mvdec._preprocessing_input_space({"method": "standard"}) == "x_standard"
    assert mvdec._preprocessing_input_space({"method": "none"}) == "x_raw"
    with pytest.raises(ValueError, match="Unsupported preprocessing metadata"):
        mvdec._preprocessing_input_space({"method": "unknown"})
    assert mvdec.KMEANS_N_INIT == 100
    assert mvdec.assignment_change_tolerance == 0.01
    assert mvdec.resolve_assignment_change_tolerance("AIRPOLLUTION") == 0.01
    assert mvdec.resolve_assignment_change_tolerance("TIKI") == 0.01
    assert mvdec.resolve_assignment_change_tolerance("REUTERS") == 0.001
    assert mvdec.resolve_assignment_change_tolerance("20NEWS") == 0.001
    assert mvdec.resolve_assignment_change_tolerance("RCV1") == 0.001
    assert mvdec.resolve_assignment_change_tolerance("REUTERS", 0.005) == 0.005
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        mvdec.resolve_assignment_change_tolerance("REUTERS", 0.0)

    kmeans = mvdec.make_kmeans(random_seed=42)
    assert kmeans.n_init == 100
    assert kmeans.random_state == 42
    assert mvdec.KMEANS_REFRESH_POLICY == "one_epoch"
    assert mvdec.REFINEMENT_BATCHING_POLICY == "balanced_shuffled_each_epoch"
    run_config = mvdec.resolved_run_config(
        protocol=mvdec.PRIMARY_MVDEC_PROTOCOL,
        dataset_name="TIKI",
        tolerance=0.01,
        max_refinement_epochs=1400,
        source_sha256="source-hash",
        feature_columns=["feature"],
        preprocessing_metadata=None,
    )
    assert (
        run_config["deterministic_runtime_policy"] == mvdec.DETERMINISTIC_RUNTIME_POLICY
    )
    assert run_config["kmeans_refresh_policy"] == "one_epoch"
    assert run_config["refinement_batching_policy"] == (
        "balanced_shuffled_each_epoch"
    )
    assert "update_interval" not in run_config
    assert "max_training_steps" not in run_config
    assert "kmeans_n_init_policy" not in run_config
    assert mvdec.DEFAULT_GREEDY_EIGEN_DIRECTION == "largest"
    assert mvdec.DEFAULT_GREEDY_TARGET_MODE == "frozen_snapshot"
    assert mvdec.validate_max_refinement_epochs(1400) == 1400
    with pytest.raises(ValueError, match="max_refinement_epochs"):
        mvdec.validate_max_refinement_epochs(0)
    assert mvdec.validate_progress_interval(25) == 25
    with pytest.raises(ValueError, match="progress_interval"):
        mvdec.validate_progress_interval(0)
    assert mvdec.should_log_progress(1, 200, 25)
    assert mvdec.should_log_progress(25, 200, 25)
    assert mvdec.should_log_progress(200, 200, 25)
    assert not mvdec.should_log_progress(26, 200, 25)
    assert mvdec.number_of_batches(3765, 256) == 15
    assert mvdec.number_of_batches(1799, 256) == 7
    assert mvdec.number_of_batches(512, 256) == 2

    air_bounds = mvdec.epoch_batch_bounds(3765, 256)
    assert len(air_bounds) == 15
    assert air_bounds[0] == (0, 251)
    assert air_bounds[-1] == (3514, 3765)
    assert sum(end - start for start, end in air_bounds) == 3765

    tiki_bounds = mvdec.epoch_batch_bounds(1799, 256)
    assert tiki_bounds == [
        (0, 257),
        (257, 514),
        (514, 771),
        (771, 1028),
        (1028, 1285),
        (1285, 1542),
        (1542, 1799),
    ]

    exact_bounds = mvdec.epoch_batch_bounds(512, 256)
    assert exact_bounds == [(0, 256), (256, 512)]
    assert [step for step in range(14) if mvdec.is_refinement_epoch_end(step, 7)] == [
        6,
        13,
    ]


def test_epoch_batches_are_balanced_complete_and_reproducible(monkeypatch):
    mvdec = _load_mvdec(monkeypatch)
    first_rng = np.random.default_rng(42)
    first_epoch = mvdec.epoch_batch_indices(1799, 256, first_rng)
    second_epoch = mvdec.epoch_batch_indices(1799, 256, first_rng)
    repeated_first_epoch = mvdec.epoch_batch_indices(
        1799,
        256,
        np.random.default_rng(42),
    )

    assert [len(batch) for batch in first_epoch] == [257] * 7
    np.testing.assert_array_equal(
        np.sort(np.concatenate(first_epoch)),
        np.arange(1799),
    )
    np.testing.assert_array_equal(
        np.concatenate(first_epoch),
        np.concatenate(repeated_first_epoch),
    )
    assert not np.array_equal(
        np.concatenate(first_epoch),
        np.concatenate(second_epoch),
    )


def test_deterministic_runtime_repeats_keras_initialization(monkeypatch):
    """The same run seed must recreate identical Keras model weights."""

    mvdec = _load_mvdec(monkeypatch)
    monkeypatch.setattr(mvdec, "input_shape", 2)
    monkeypatch.setattr(mvdec, "hidden_units", 1)
    monkeypatch.setattr(mvdec, "view1_filters", [3])

    first_metadata = mvdec.configure_deterministic_runtime(42)
    first_weights = mvdec.model_view1(load_weights=False).get_weights()
    second_metadata = mvdec.configure_deterministic_runtime(42)
    second_weights = mvdec.model_view1(load_weights=False).get_weights()

    assert first_metadata == second_metadata
    assert first_metadata["policy"] == mvdec.DETERMINISTIC_RUNTIME_POLICY
    for first, second in zip(first_weights, second_weights, strict=True):
        np.testing.assert_array_equal(first, second)
    with pytest.raises(ValueError, match="explicit seed"):
        mvdec.configure_deterministic_runtime(None)


def test_pretraining_dataset_repeats_seeded_epoch_order(monkeypatch):
    """Seeded tf.data shuffling must repeat across isolated run setup."""

    mvdec = _load_mvdec(monkeypatch)
    monkeypatch.setattr(mvdec, "pretrain_batch_size", 2, raising=False)
    values = np.arange(12, dtype=np.float32).reshape(6, 2)

    mvdec.configure_deterministic_runtime(42)
    first_dataset = mvdec.make_pretraining_dataset(values, 42)
    first_order = np.concatenate([batch_x.numpy() for batch_x, _ in first_dataset])

    mvdec.configure_deterministic_runtime(42)
    second_dataset = mvdec.make_pretraining_dataset(values, 42)
    second_order = np.concatenate([batch_x.numpy() for batch_x, _ in second_dataset])

    np.testing.assert_array_equal(first_order, second_order)


def test_assignment_change_count_ignores_cluster_id_permutations(monkeypatch):
    mvdec = _load_mvdec(monkeypatch)
    monkeypatch.setattr(mvdec, "n_clusters", 3)
    previous = np.array([0, 0, 1, 1, 2, 2])

    assert mvdec.count_aligned_assignment_changes(
        np.full_like(previous, -1),
        previous,
    ) == len(previous)
    assert (
        mvdec.count_aligned_assignment_changes(
            previous,
            np.array([2, 2, 0, 0, 1, 1]),
        )
        == 0
    )
    assert (
        mvdec.count_aligned_assignment_changes(
            previous,
            np.array([2, 2, 0, 1, 1, 1]),
        )
        == 1
    )


def test_final_clustering_is_recomputed_from_current_model_outputs(monkeypatch):
    mvdec = _load_mvdec(monkeypatch)
    monkeypatch.setattr(mvdec, "hidden_units", 2)
    monkeypatch.setattr(mvdec, "n_clusters", 2)

    class FakeModel:
        def __init__(self, output):
            self.output = mvdec.tf.constant(output, dtype=mvdec.tf.float32)

        def __call__(self, _):
            return self.output

    view1 = FakeModel([[0.0, 0.0], [0.0, 0.2], [5.0, 5.0], [5.0, 5.2]])
    view2 = FakeModel([[0.0, 0.2], [0.0, 0.0], [5.0, 5.2], [5.0, 5.0]])

    h1, h2, h_fused, labels = mvdec.recompute_final_clustering(
        view1,
        view2,
        np.zeros((4, 1), dtype=np.float32),
        random_seed=42,
    )

    np.testing.assert_allclose(h_fused, (h1 + h2) / 2)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_epoch_losses_are_averaged_across_all_batches(monkeypatch):
    mvdec = _load_mvdec(monkeypatch)

    means = mvdec.mean_epoch_losses(
        [
            {
                "sample_count": 3,
                "total": 8.0,
                "reconstruction": 6.0,
                "greedy": 2.0,
            },
            {
                "sample_count": 1,
                "total": 4.0,
                "reconstruction": 2.0,
                "greedy": 1.0,
            },
        ]
    )

    assert means == {"total": 7.0, "reconstruction": 5.0, "greedy": 1.75}
    with pytest.raises(ValueError, match="At least one batch loss"):
        mvdec.mean_epoch_losses([])


def test_greedy_eigen_modes_select_opposite_ends(monkeypatch):
    mvdec = _load_mvdec(monkeypatch)

    assert mvdec.greedy_eigen_index("smallest") == 0
    assert mvdec.greedy_eigen_index("largest") == -1
    with pytest.raises(ValueError, match="Unsupported greedy eigen direction"):
        mvdec.greedy_eigen_index("middle")


@pytest.mark.parametrize(
    ("direction", "expected"),
    [
        (
            "smallest",
            [[10.0, 2.0, 3.0], [40.0, 5.0, 6.0]],
        ),
        (
            "largest",
            [[1.0, 2.0, 30.0], [4.0, 5.0, 60.0]],
        ),
    ],
)
def test_build_greedy_target_changes_only_selected_direction(
    monkeypatch,
    direction,
    expected,
):
    mvdec = _load_mvdec(monkeypatch)
    transformed = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    centroids = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])

    target = mvdec.build_greedy_target(
        transformed,
        centroids,
        np.array([0, 1]),
        mvdec.greedy_eigen_index(direction),
    )

    np.testing.assert_allclose(target, expected)
    np.testing.assert_allclose(
        transformed,
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    )


def test_greedy_target_modes_are_explicit(monkeypatch):
    mvdec = _load_mvdec(monkeypatch)

    assert mvdec.validate_greedy_target_mode("selected_dimension_only") == (
        "selected_dimension_only"
    )
    assert mvdec.validate_greedy_target_mode("frozen_snapshot") == ("frozen_snapshot")
    with pytest.raises(ValueError, match="Unsupported greedy target mode"):
        mvdec.validate_greedy_target_mode("moving_snapshot")


def test_primary_protocol_is_explicit_and_immutable(monkeypatch):
    mvdec = _load_mvdec(monkeypatch)

    protocol = mvdec.resolve_mvdec_protocol(
        mvdec.PRIMARY_PROTOCOL_ID,
        kmeans_weight=0.0,
        greedy_weight=1.0,
        eigen_direction="largest",
        target_mode="frozen_snapshot",
    )

    assert protocol == mvdec.PRIMARY_MVDEC_PROTOCOL
    assert (
        mvdec.final_training_objective(protocol)
        == "mvdec_dekm_consistent_l1_reconstruction_plus_l4_greedy"
    )
    contract = protocol.manifest_contract(0.001)
    assert contract["objective"]["loss_terms"]["L2_kmeans"] == {
        "weight": 0.0,
        "optimized": False,
    }
    assert contract["objective"]["loss_terms"]["L3_scatter_trace"]["optimized"] is False
    assert contract["eigen"]["direction"] == "largest"
    assert contract["greedy_target"]["mode"] == "frozen_snapshot"
    assert "schedule" not in contract
    primary_schedule = mvdec.resolve_refinement_schedule(
        protocol,
        n_samples=10_000,
        current_batch_size=256,
        max_refinement_epochs=1400,
    )
    assert primary_schedule.batches_per_epoch == 39
    assert primary_schedule.kmeans_refresh_interval == 39
    assert primary_schedule.max_training_steps == 54_600
    assert (
        mvdec.validate_protocol_assignment_change_tolerance(
            protocol,
            "REUTERS",
            0.001,
        )
        == 0.001
    )
    with pytest.raises(ValueError, match="--protocol custom"):
        mvdec.validate_protocol_assignment_change_tolerance(
            protocol,
            "REUTERS",
            0.005,
        )

    with pytest.raises(ValueError, match="immutable"):
        mvdec.resolve_mvdec_protocol(
            mvdec.PRIMARY_PROTOCOL_ID,
            kmeans_weight=1.0,
            greedy_weight=1.0,
            eigen_direction="largest",
            target_mode="frozen_snapshot",
        )


def test_public_reproduction_protocol_locks_release_schedule(monkeypatch):
    mvdec = _load_mvdec(monkeypatch)

    protocol = mvdec.resolve_mvdec_protocol(
        mvdec.PUBLIC_REPRODUCTION_PROTOCOL_ID,
        kmeans_weight=0.0,
        greedy_weight=1.0,
        eigen_direction="largest",
        target_mode="frozen_snapshot",
    )
    schedule = mvdec.resolve_refinement_schedule(
        protocol,
        n_samples=10_000,
        current_batch_size=256,
        max_refinement_epochs=1400,
    )

    assert protocol == mvdec.PUBLIC_REPRODUCTION_PROTOCOL
    assert protocol.reconstruction_weight == 1.0
    assert protocol.kmeans_weight == 0.0
    assert protocol.scatter_trace_weight == 0.0
    assert protocol.greedy_weight == 1.0
    assert mvdec.final_training_objective(protocol) == (
        "mvdec_2025_public_reproduction_release_"
        "l1_reconstruction_plus_l4_greedy_mse"
    )
    assert schedule.batches_per_epoch == 40
    assert schedule.kmeans_refresh_interval == 10
    assert schedule.max_training_steps == 14_000
    assert protocol.manifest_contract(0.001)["architecture"] == {
        "view2": mvdec.VIEW2_ARCHITECTURE_ID,
    }
    assert protocol.manifest_contract(0.001)["schedule"]["refinement"] == {
        "objective": "release_reconstruction_plus_greedy_mse",
        "batch_size": 256,
        "batching_policy": "sequential_release_order",
        "kmeans_refresh_policy": "fixed_10_updates",
        "update_interval": 10,
        "max_training_steps": 14_000,
        "kmeans_n_init_policy": "initial_100_then_twice_previous_n_iter",
    }
    mvdec.validate_protocol_dataset_scope(protocol, "REUTERS")
    with pytest.raises(ValueError, match="supports only"):
        mvdec.validate_protocol_dataset_scope(protocol, "TIKI")
    with pytest.raises(ValueError, match="fixes refinement"):
        mvdec.validate_protocol_max_refinement_epochs(protocol, 100)


def test_encoder_bottleneck_protocol_changes_only_view2_architecture(monkeypatch):
    mvdec = _load_mvdec(monkeypatch)

    protocol = mvdec.resolve_mvdec_protocol(
        mvdec.PUBLIC_ENCODER_BOTTLENECK_PROTOCOL_ID,
        kmeans_weight=0.0,
        greedy_weight=1.0,
        eigen_direction="largest",
        target_mode="frozen_snapshot",
    )

    assert protocol == mvdec.PUBLIC_ENCODER_BOTTLENECK_PROTOCOL
    assert protocol.view2_architecture_id == (
        mvdec.VIEW2_ENCODER_BOTTLENECK_ARCHITECTURE_ID
    )
    assert mvdec.protocol_method_name(protocol) == (
        "MvDEC-2025-View2-encoder-bottleneck-ablation"
    )
    assert mvdec.final_training_objective(protocol) == mvdec.final_training_objective(
        mvdec.PUBLIC_REPRODUCTION_PROTOCOL
    )
    assert protocol.manifest_contract(0.001)["schedule"] == (
        mvdec.PUBLIC_REPRODUCTION_PROTOCOL.manifest_contract(0.001)["schedule"]
    )
    assert protocol.manifest_contract(0.001)["architecture"] != (
        mvdec.PUBLIC_REPRODUCTION_PROTOCOL.manifest_contract(0.001)["architecture"]
    )
    mvdec.validate_protocol_dataset_scope(protocol, "REUTERS")


def test_direct_joint_head_protocol_changes_only_view2_architecture(monkeypatch):
    mvdec = _load_mvdec(monkeypatch)

    protocol = mvdec.resolve_mvdec_protocol(
        mvdec.PUBLIC_DIRECT_JOINT_HEAD_PROTOCOL_ID,
        kmeans_weight=0.0,
        greedy_weight=1.0,
        eigen_direction="largest",
        target_mode="frozen_snapshot",
    )

    assert protocol == mvdec.PUBLIC_DIRECT_JOINT_HEAD_PROTOCOL
    assert protocol.view2_architecture_id == (
        mvdec.VIEW2_DIRECT_JOINT_HEAD_ARCHITECTURE_ID
    )
    assert mvdec.protocol_method_name(protocol) == (
        "MvDEC-2025-View2-direct-joint-head-ablation"
    )
    assert mvdec.final_training_objective(protocol) == mvdec.final_training_objective(
        mvdec.PUBLIC_REPRODUCTION_PROTOCOL
    )
    assert protocol.manifest_contract(0.001)["schedule"] == (
        mvdec.PUBLIC_REPRODUCTION_PROTOCOL.manifest_contract(0.001)["schedule"]
    )
    assert protocol.manifest_contract(0.001)["architecture"] != (
        mvdec.PUBLIC_REPRODUCTION_PROTOCOL.manifest_contract(0.001)["architecture"]
    )
    mvdec.validate_protocol_dataset_scope(protocol, "REUTERS")


def test_release_batch_order_and_dynamic_kmeans_restarts(monkeypatch):
    mvdec = _load_mvdec(monkeypatch)

    np.testing.assert_array_equal(
        mvdec.sequential_release_batch_indices(1000, 256, 0),
        np.arange(0, 256),
    )
    np.testing.assert_array_equal(
        mvdec.sequential_release_batch_indices(1000, 256, 3),
        np.arange(768, 1000),
    )
    np.testing.assert_array_equal(
        mvdec.sequential_release_batch_indices(1000, 256, 4),
        np.arange(0, 256),
    )

    class FittedKMeans:
        n_iter_ = 7

    assert (
        mvdec.next_kmeans_n_init(
            mvdec.PUBLIC_REPRODUCTION_PROTOCOL,
            FittedKMeans(),
        )
        == 14
    )
    assert (
        mvdec.next_kmeans_n_init(
            mvdec.PRIMARY_MVDEC_PROTOCOL,
            FittedKMeans(),
        )
        == 100
    )


def test_public_representation_metrics_use_independent_kmeans(monkeypatch):
    mvdec = _load_mvdec(monkeypatch)
    monkeypatch.setattr(mvdec, "n_clusters", 2)
    truth = np.array([0, 0, 1, 1])
    h_view1 = np.array([[0.0], [0.1], [5.0], [5.1]])
    h_view2 = np.array([[10.0], [10.1], [-5.0], [-5.1]])
    h_fused = np.column_stack((h_view1[:, 0], h_view2[:, 0]))

    evaluations = mvdec.evaluate_public_representations(
        h_view1,
        h_view2,
        h_fused,
        truth,
        random_seed=42,
    )

    assert set(evaluations) == {"view1", "view2", "fused"}
    assert all(evaluation.metrics.acc == 1.0 for evaluation in evaluations.values())
    assert all(evaluation.metrics.nmi == 1.0 for evaluation in evaluations.values())


def test_public_reproduction_defers_ground_truth_until_final(monkeypatch):
    mvdec = _load_mvdec(monkeypatch)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Ground truth was used during refinement")

    monkeypatch.setattr(mvdec, "compute_external_metrics", fail_if_called)
    metric_text, metric_value = mvdec._training_metric_for_labels(
        np.zeros((4, 2)),
        np.array([0, 0, 1, 1]),
        np.array([0, 0, 1, 1]),
        mvdec.PUBLIC_REPRODUCTION_PROTOCOL,
    )

    assert metric_text == "external_metrics = deferred_to_final"
    assert metric_value is None
    assert (
        mvdec.PUBLIC_REPRODUCTION_PROTOCOL.manifest_contract(0.001)["schedule"][
            "final_evaluation"
        ]["ground_truth_usage"]
        == "final_evaluation_only"
    )


def test_custom_protocol_cannot_claim_primary_objective(monkeypatch):
    mvdec = _load_mvdec(monkeypatch)

    protocol = mvdec.resolve_mvdec_protocol(
        mvdec.CUSTOM_PROTOCOL_ID,
        kmeans_weight=1.0,
        greedy_weight=1.0,
        eigen_direction="smallest",
        target_mode="selected_dimension_only",
    )

    assert protocol.protocol_id == "custom"
    assert mvdec.final_training_objective(protocol).startswith("mvdec_custom_")
    assert "Eq. 11" not in mvdec.final_training_objective(protocol)
    assert (
        mvdec.validate_protocol_assignment_change_tolerance(
            protocol,
            "REUTERS",
            0.005,
        )
        == 0.005
    )


def test_airpollution_artifact_requires_preprocessing_metadata(tmp_path, monkeypatch):
    mvdec = _load_mvdec(monkeypatch)
    monkeypatch.setattr(mvdec, "ds_name", "AIRPOLLUTION")
    h_view1 = np.zeros((2, 2), dtype=np.float32)
    h_view2 = np.ones((2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="require preprocessing metadata"):
        mvdec.save_airpollution_mvdec_artifact(
            artifact_path=tmp_path / "artifact.pkl",
            h_view1=h_view1,
            h_view2=h_view2,
            h_fused=(h_view1 + h_view2) / 2,
            labels=np.array([0, 1]),
            score=0.1,
            iteration=1,
            orig_idx=np.array([0, 1]),
            feature_columns=["a", "b"],
            random_seed=42,
            greedy_eigen_direction="largest",
            greedy_target_mode="frozen_snapshot",
            batches_per_epoch=1,
            max_refinement_epochs=5,
            stop_reason="converged_assignment",
            refinement_epochs_completed=3,
            preprocessing_metadata=None,
        )


def test_public_mvdec_artifact_restores_release_row_order(tmp_path, monkeypatch):
    """Labeled MvDEC artifacts must be canonical and carry external metrics."""

    mvdec = _load_mvdec(monkeypatch)
    monkeypatch.setattr(mvdec, "ds_name", "REUTERS")
    monkeypatch.setattr(mvdec, "input_shape", 2)
    monkeypatch.setattr(mvdec, "hidden_units", 2)
    monkeypatch.setattr(mvdec, "n_clusters", 2)
    monkeypatch.setattr(mvdec, "assignment_change_tolerance", 0.001)
    shuffled_indices = np.array([2, 0, 3, 1])
    h_view1 = np.arange(8, dtype=np.float32).reshape(4, 2)
    h_view2 = h_view1 + 1.0
    labels = np.array([1, 0, 1, 0])
    truth = np.array([1, 0, 1, 0])
    metrics = compute_external_metrics(truth, labels, n_clusters=2)
    artifact_path = tmp_path / "reuters.pkl"
    data_path = tmp_path / "reuters.csv"
    pd.DataFrame({"feature": range(4)}).to_csv(data_path, index=False)
    run_config = {
        "dataset": "REUTERS",
        "protocol_contract": mvdec.PRIMARY_MVDEC_PROTOCOL.manifest_contract(0.001),
    }
    config_hash = canonical_config_hash(run_config)
    run_id = build_run_id(
        "REUTERS",
        mvdec.PRIMARY_PROTOCOL_ID,
        config_hash,
        seed=42,
        run_index=1,
    )

    mvdec.save_airpollution_mvdec_artifact(
        artifact_path=artifact_path,
        h_view1=h_view1,
        h_view2=h_view2,
        h_fused=(h_view1 + h_view2) / 2,
        labels=labels,
        score=0.5,
        iteration=3,
        orig_idx=shuffled_indices,
        feature_columns=["a", "b"],
        random_seed=42,
        greedy_eigen_direction="largest",
        greedy_target_mode="frozen_snapshot",
        batches_per_epoch=1,
        max_refinement_epochs=5,
        stop_reason="converged_assignment",
        refinement_epochs_completed=3,
        preprocessing_metadata=None,
        true_labels=truth,
        source_sha256="c" * 64,
        external_metrics=metrics,
        run_id=run_id,
        config_hash=config_hash,
    )

    write_run_manifest(
        tmp_path / "manifest.json",
        {
            "run_id": run_id,
            "status": "complete",
            "dataset": "REUTERS",
            "protocol_id": mvdec.PRIMARY_PROTOCOL_ID,
            "config_hash": config_hash,
            "config": run_config,
            "seed": 42,
            "paths": {"artifact": str(artifact_path)},
            "output_sha256": {"artifact": file_sha256(artifact_path)},
        },
    )

    with artifact_path.open("rb") as file:
        artifact = pickle.load(file)
    expected_labels = _restore_original_order(labels, shuffled_indices)
    assert artifact["labels"].tolist() == expected_labels.tolist()
    assert artifact["true_labels"].tolist() == expected_labels.tolist()
    assert artifact["row_indices"].tolist() == [0, 1, 2, 3]
    assert artifact["source_sha256"] == "c" * 64
    assert artifact["acc"] == 1.0
    assert artifact["nmi"] == 1.0
    assert artifact["protocol_id"] == mvdec.PRIMARY_PROTOCOL_ID
    assert artifact["algorithm_family"] == "MvDEC"
    assert artifact["algorithm"] == "MvDEC-DEKM-consistent"
    assert artifact["method_name"] == "MvDEC-DEKM-consistent"
    assert artifact["claim_scope"] == mvdec.PRIMARY_MVDEC_PROTOCOL.claim_scope
    assert artifact["view2_architecture_id"] == mvdec.VIEW2_ARCHITECTURE_ID
    assert artifact["config"]["view2_architecture_id"] == (
        mvdec.VIEW2_ARCHITECTURE_ID
    )
    assert artifact["protocol_contract"]["eigen"]["direction"] == "largest"
    assert artifact["protocol_contract"]["greedy_target"]["mode"] == ("frozen_snapshot")
    assert len(artifact["protocol_contract_sha256"]) == 64

    loaded = load_mvdec_result(result_path=artifact_path, data_path=data_path)
    assert loaded.raw["method_name"] == "MvDEC-DEKM-consistent"

    artifact["view2_architecture_id"] = "legacy_pre_decoder_bottleneck"
    artifact["config"]["view2_architecture_id"] = "legacy_pre_decoder_bottleneck"
    artifact["protocol_contract"]["architecture"]["view2"] = (
        "legacy_pre_decoder_bottleneck"
    )
    tampered_hash = mvdec.protocol_contract_sha256(artifact["protocol_contract"])
    artifact["protocol_contract_sha256"] = tampered_hash
    artifact["config"]["protocol_contract"] = artifact["protocol_contract"]
    artifact["config"]["protocol_contract_sha256"] = tampered_hash
    with artifact_path.open("wb") as file:
        pickle.dump(artifact, file)
    with pytest.raises(ValueError, match="architecture identity"):
        load_mvdec_result(result_path=artifact_path, data_path=data_path)

    artifact["view2_architecture_id"] = mvdec.VIEW2_ARCHITECTURE_ID
    artifact["config"]["view2_architecture_id"] = mvdec.VIEW2_ARCHITECTURE_ID
    artifact["protocol_contract"]["architecture"]["view2"] = (
        mvdec.VIEW2_ARCHITECTURE_ID
    )
    artifact["protocol_contract"]["eigen"]["direction"] = "smallest"
    with artifact_path.open("wb") as file:
        pickle.dump(artifact, file)
    with pytest.raises(ValueError, match="contract SHA-256"):
        load_mvdec_result(result_path=artifact_path, data_path=data_path)

    tampered_hash = mvdec.protocol_contract_sha256(artifact["protocol_contract"])
    artifact["protocol_contract_sha256"] = tampered_hash
    artifact["config"]["protocol_contract"] = artifact["protocol_contract"]
    artifact["config"]["protocol_contract_sha256"] = tampered_hash
    with artifact_path.open("wb") as file:
        pickle.dump(artifact, file)
    with pytest.raises(ValueError, match="immutable protocol"):
        load_mvdec_result(result_path=artifact_path, data_path=data_path)


@pytest.mark.parametrize(
    ("protocol_attribute", "protocol_id", "expected_architecture", "expected_method"),
    (
        (
            "PUBLIC_REPRODUCTION_PROTOCOL",
            "mvdec_2025_public_reproduction_v1",
            "mvdec2025_post_skip_latent_bottleneck_v2",
            "MvDEC-2025-public-reproduction",
        ),
        (
            "PUBLIC_ENCODER_BOTTLENECK_PROTOCOL",
            "mvdec_2025_view2_encoder_bottleneck_v1",
            "mvdec2025_encoder_bottleneck_skip_decoder_v1",
            "MvDEC-2025-View2-encoder-bottleneck-ablation",
        ),
        (
            "PUBLIC_DIRECT_JOINT_HEAD_PROTOCOL",
            "mvdec_2025_view2_direct_23_split_v1",
            "mvdec2025_post_skip_direct_latent_reconstruction_head_v1",
            "MvDEC-2025-View2-direct-joint-head-ablation",
        ),
    ),
)
def test_public_reproduction_artifact_persists_view_metrics(
    tmp_path,
    monkeypatch,
    protocol_attribute,
    protocol_id,
    expected_architecture,
    expected_method,
):
    mvdec = _load_mvdec(monkeypatch)
    protocol = getattr(mvdec, protocol_attribute)
    monkeypatch.setattr(mvdec, "ds_name", "REUTERS")
    monkeypatch.setattr(mvdec, "input_shape", 2)
    monkeypatch.setattr(mvdec, "hidden_units", 2)
    monkeypatch.setattr(mvdec, "n_clusters", 2)
    monkeypatch.setattr(mvdec, "assignment_change_tolerance", 0.001)
    shuffled_indices = np.array([2, 0, 3, 1])
    h_view1 = np.array([[5.0, 5.0], [0.0, 0.0], [5.1, 5.0], [0.1, 0.0]])
    h_view2 = h_view1 + 0.2
    labels = np.array([1, 0, 1, 0])
    truth = labels.copy()
    metrics = compute_external_metrics(truth, labels, n_clusters=2)
    representation_metrics = {
        "view1": metrics,
        "view2": metrics,
        "fused": metrics,
    }
    artifact_path = tmp_path / "reuters_reproduction.pkl"
    data_path = tmp_path / "reuters.csv"
    pd.DataFrame({"feature": range(4)}).to_csv(data_path, index=False)
    run_config = {
        "dataset": "REUTERS",
        "protocol_contract": protocol.manifest_contract(0.001),
    }
    config_hash = canonical_config_hash(run_config)
    run_id = build_run_id(
        "REUTERS",
        protocol_id,
        config_hash,
        seed=42,
        run_index=1,
    )

    mvdec.save_airpollution_mvdec_artifact(
        artifact_path=artifact_path,
        h_view1=h_view1,
        h_view2=h_view2,
        h_fused=(h_view1 + h_view2) / 2,
        labels=labels,
        score=0.5,
        iteration=30,
        orig_idx=shuffled_indices,
        feature_columns=["a", "b"],
        random_seed=42,
        greedy_eigen_direction="largest",
        greedy_target_mode="frozen_snapshot",
        batches_per_epoch=1,
        max_refinement_epochs=1400,
        stop_reason="converged_assignment",
        refinement_epochs_completed=30,
        preprocessing_metadata=None,
        true_labels=truth,
        source_sha256="e" * 64,
        external_metrics=metrics,
        representation_external_metrics=representation_metrics,
        training_steps_completed=30,
        protocol=protocol,
        run_id=run_id,
        config_hash=config_hash,
    )
    manifest_payload = {
        "run_id": run_id,
        "status": "complete",
        "dataset": "REUTERS",
        "protocol_id": protocol_id,
        "config_hash": config_hash,
        "config": run_config,
        "seed": 42,
        "paths": {"artifact": str(artifact_path)},
        "output_sha256": {"artifact": file_sha256(artifact_path)},
    }
    write_run_manifest(tmp_path / "manifest.json", manifest_payload)

    loaded = load_mvdec_result(result_path=artifact_path, data_path=data_path)

    assert loaded.raw["view1_acc"] == 1.0
    assert loaded.raw["view2_nmi"] == 1.0
    assert loaded.raw["fused_acc"] == loaded.raw["acc"] == 1.0
    assert loaded.raw["training_steps_completed"] == 30
    assert loaded.raw["view2_architecture_id"] == expected_architecture
    assert loaded.raw["method_name"] == expected_method

    with artifact_path.open("rb") as file:
        artifact = pickle.load(file)
    artifact["config"]["update_interval"] = 9
    with artifact_path.open("wb") as file:
        pickle.dump(artifact, file)
    manifest_payload["output_sha256"]["artifact"] = file_sha256(artifact_path)
    write_run_manifest(tmp_path / "manifest.json", manifest_payload)
    with pytest.raises(ValueError, match="public-reproduction artifact"):
        load_mvdec_result(result_path=artifact_path, data_path=data_path)


def test_unlabeled_preprocessing_rejects_unknown_policy(tmp_path, monkeypatch):
    mvdec = _load_mvdec(monkeypatch)
    data_path = tmp_path / "data.csv"
    pd.DataFrame({"feature": [1.0, 2.0]}).to_csv(data_path, index=False)

    with pytest.raises(ValueError, match="Unsupported scaling method"):
        mvdec.get_x_unlabeled_csv(
            data_path,
            log_print=False,
            scaling_method="robust",
        )
