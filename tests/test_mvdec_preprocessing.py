import importlib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.external_metrics import compute_external_metrics


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


def test_tiki_none_policy_preserves_input_values(tmp_path, monkeypatch):
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


def test_dataset_registry_scales_only_airpollution(monkeypatch):
    mvdec = _load_mvdec(monkeypatch)

    assert mvdec.UNLABELED_DATASETS["AIRPOLLUTION"]["scaling_method"] == "minmax"
    assert mvdec.UNLABELED_DATASETS["TIKI"]["scaling_method"] == "none"
    assert mvdec.KMEANS_N_INIT == 100
    assert mvdec.assignment_change_tolerance == 0.01

    kmeans = mvdec.make_kmeans(random_seed=42)
    assert kmeans.n_init == 100
    assert kmeans.random_state == 42
    assert mvdec.KMEANS_REFRESH_POLICY == "one_epoch"
    assert mvdec.REFINEMENT_BATCHING_POLICY == "balanced_shuffled_each_epoch"
    assert mvdec.validate_max_refinement_epochs(1400) == 1400
    with pytest.raises(ValueError, match="max_refinement_epochs"):
        mvdec.validate_max_refinement_epochs(0)
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


def test_largest_mode_uses_default_artifact_path(monkeypatch):
    mvdec = _load_mvdec(monkeypatch)

    assert (
        mvdec.artifact_path_for_greedy_mode(
            "air.pkl",
            "largest",
            "selected_dimension_only",
        )
        == "air.pkl"
    )
    assert (
        mvdec.artifact_path_for_greedy_mode(
            "air.pkl",
            "smallest",
            "selected_dimension_only",
        )
        == "air_smallest_eigen_selected_dimension_only.pkl"
    )


def test_greedy_target_modes_are_explicit(monkeypatch):
    mvdec = _load_mvdec(monkeypatch)

    assert mvdec.validate_greedy_target_mode("selected_dimension_only") == (
        "selected_dimension_only"
    )
    assert mvdec.validate_greedy_target_mode("frozen_snapshot") == ("frozen_snapshot")
    with pytest.raises(ValueError, match="Unsupported greedy target mode"):
        mvdec.validate_greedy_target_mode("moving_snapshot")


def test_airpollution_artifact_requires_scaler_metadata(tmp_path, monkeypatch):
    mvdec = _load_mvdec(monkeypatch)
    monkeypatch.setattr(mvdec, "ds_name", "AIRPOLLUTION")
    h_view1 = np.zeros((2, 2), dtype=np.float32)
    h_view2 = np.ones((2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="require Min-Max preprocessing metadata"):
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
    shuffled_indices = np.array([2, 0, 3, 1])
    h_view1 = np.arange(8, dtype=np.float32).reshape(4, 2)
    h_view2 = h_view1 + 1.0
    labels = np.array([1, 0, 1, 0])
    truth = np.array([1, 0, 1, 0])
    metrics = compute_external_metrics(truth, labels, n_clusters=2)
    artifact_path = tmp_path / "reuters.pkl"

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
        greedy_target_mode="selected_dimension_only",
        batches_per_epoch=1,
        max_refinement_epochs=5,
        stop_reason="converged_assignment",
        refinement_epochs_completed=3,
        preprocessing_metadata=None,
        true_labels=truth,
        source_sha256="c" * 64,
        external_metrics=metrics,
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
