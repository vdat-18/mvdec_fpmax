import hashlib
import pickle

import numpy as np
import pandas as pd
import pytest

from config import H_FUSED_COLUMNS
from pipeline.clustering import compute_gower_distance, compute_silhouette_diagnostics
from pipeline.data import load_mvdec_result


def _normalized_file_sha256(path) -> str:
    """Return the source hash contract used by MvDEC CSV artifacts."""

    content = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def _write_mvdec_inputs(tmp_path, n_rows=4):
    data_path = tmp_path / "tiki_preprocessed.csv"
    result_path = tmp_path / "fused_representation.pkl"

    pd.DataFrame({"feature": range(n_rows)}).to_csv(data_path, index=False)
    h_fused = np.arange(n_rows * len(H_FUSED_COLUMNS), dtype=np.float32).reshape(
        n_rows, len(H_FUSED_COLUMNS)
    )
    labels = np.array([0, 0, 1, 1][:n_rows])
    with result_path.open("wb") as file:
        pickle.dump(
            {
                "h_fused": h_fused,
                "labels": labels,
                "true_labels": np.array([0, 0, 1, 1][:n_rows]),
                "row_indices": np.arange(n_rows),
                "acc": 1.0,
                "nmi": 1.0,
                "source_sha256": "b" * 64,
                "init": "k-means++",
                "score": 0.1,
                "iteration": 1,
            },
            file,
        )

    return result_path, data_path


def test_load_mvdec_result_uses_fused_embedding_columns(tmp_path):
    result_path, data_path = _write_mvdec_inputs(tmp_path)

    result = load_mvdec_result(result_path=result_path, data_path=data_path)

    assert list(result.h_fused_df.columns) == [
        f"fused_{index}" for index in range(1, 12)
    ]
    expected_score, expected_std, expected_negative_fraction = (
        compute_silhouette_diagnostics(
            compute_gower_distance(result.h_fused_df),
            result.labels,
        )
    )
    assert result.evaluation_score == expected_score
    assert result.evaluation_sample_std == expected_std
    assert result.evaluation_negative_fraction == expected_negative_fraction
    assert result.true_labels.tolist() == [0, 0, 1, 1]
    assert result.row_indices.tolist() == [0, 1, 2, 3]
    assert result.acc == 1.0
    assert result.nmi == 1.0
    assert result.source_sha256 == "b" * 64


def test_load_mvdec_result_rejects_preprocessed_row_mismatch(tmp_path):
    result_path, data_path = _write_mvdec_inputs(tmp_path)
    pd.DataFrame({"feature": [1, 2, 3]}).to_csv(data_path, index=False)

    with pytest.raises(ValueError, match="row count does not match"):
        load_mvdec_result(result_path=result_path, data_path=data_path)


def test_load_mvdec_result_rejects_protocol_less_encoder_average(tmp_path):
    data_path = tmp_path / "airpollution.csv"
    result_path = tmp_path / "mvdec2025.pkl"
    pd.DataFrame({"feature": range(4)}).to_csv(data_path, index=False)
    h_view1 = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [5.0, 5.0],
            [5.1, 5.0],
        ],
        dtype=np.float32,
    )
    h_view2 = h_view1 + 0.2
    h_fused = (h_view1 + h_view2) / 2
    labels = np.array([0, 0, 1, 1])
    with result_path.open("wb") as file:
        pickle.dump(
            {
                "fusion_contract": "mvdec2025_encoder_average",
                "view_output_layout": "eq5_compatible_10_plus_13",
                "final_training_objective": (
                    "mvdec2025_latent_joint_reconstruction_kmeans_greedy_l3_trace_logged"
                ),
                "h_view1": h_view1,
                "h_view2": h_view2,
                "h_fused": h_fused,
                "labels": labels,
                "fusion_dim": 2,
                "view1_latent_dim": 2,
                "kmeans_n_init": 100,
                "init": "k-means",
                "score": 0.5,
                "iteration": 3,
            },
            file,
        )

    with pytest.raises(ValueError, match="Legacy MvDEC artifact"):
        load_mvdec_result(result_path=result_path, data_path=data_path)


def test_load_public_mvdec_result_rejects_protocol_less_artifact(tmp_path):

    data_path = tmp_path / "mvdec_assignments.csv"
    result_path = tmp_path / "mvdec_public.pkl"
    pd.DataFrame(
        {
            "dataset": ["REUTERS"] * 4,
            "method": ["MvDEC"] * 4,
            "sample_index": [0, 1, 2, 3],
            "cluster": [0, 0, 1, 1],
            "true_label": [0, 0, 1, 1],
        }
    ).to_csv(data_path, index=False)
    h_view1 = np.array(
        [[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]],
        dtype=np.float32,
    )
    h_view2 = h_view1 + 0.2
    with result_path.open("wb") as file:
        pickle.dump(
            {
                "dataset": "REUTERS",
                "fusion_contract": "mvdec2025_encoder_average",
                "view_output_layout": "eq5_compatible_2_plus_2000",
                "final_training_objective": (
                    "mvdec2025_latent_joint_reconstruction_greedy_l3_trace_logged"
                ),
                "h_view1": h_view1,
                "h_view2": h_view2,
                "h_fused": (h_view1 + h_view2) / 2,
                "labels": np.array([0, 0, 1, 1]),
                "true_labels": np.array([0, 0, 1, 1]),
                "row_indices": np.arange(4),
                "fusion_dim": 2,
                "view1_latent_dim": 2,
                "n_clusters": 2,
                "init": "k-means",
                "score": 0.9,
                "iteration": 3,
                "acc": 1.0,
                "nmi": 1.0,
                "source_sha256": "d" * 64,
            },
            file,
        )

    with pytest.raises(ValueError, match="Legacy MvDEC artifact"):
        load_mvdec_result(result_path=result_path, data_path=data_path)


def test_load_mvdec_result_rejects_unexpected_mvdec2025_layout(tmp_path):
    data_path = tmp_path / "airpollution.csv"
    result_path = tmp_path / "mvdec2025.pkl"
    pd.DataFrame({"feature": range(4)}).to_csv(data_path, index=False)
    h_view1 = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [5.0, 5.0],
            [5.1, 5.0],
        ],
        dtype=np.float32,
    )
    h_view2 = h_view1 + 0.2
    labels = np.array([0, 0, 1, 1])
    with result_path.open("wb") as file:
        pickle.dump(
            {
                "fusion_contract": "mvdec2025_encoder_average",
                "view_output_layout": "figure_dense_23",
                "h_view1": h_view1,
                "h_view2": h_view2,
                "h_fused": (h_view1 + h_view2) / 2,
                "labels": labels,
                "fusion_dim": 2,
                "init": "k-means",
                "score": 0.5,
                "iteration": 3,
            },
            file,
        )

    with pytest.raises(ValueError, match="view_output_layout"):
        load_mvdec_result(result_path=result_path, data_path=data_path)


def test_load_mvdec_result_validates_tiki_source_order_across_line_endings(
    tmp_path,
):
    """Tiki artifacts reject reordered rows but accept CRLF checkout changes."""

    data_path = tmp_path / "tiki.csv"
    result_path = tmp_path / "tiki.pkl"
    source_df = pd.DataFrame(
        {
            "followers": [0.0, 0.1, 5.0, 5.1],
            "revenue": [0.0, 0.2, 5.0, 5.2],
        }
    )
    source_df.to_csv(data_path, index=False, lineterminator="\n")
    source_sha256 = _normalized_file_sha256(data_path)
    data_path.write_bytes(data_path.read_bytes().replace(b"\n", b"\r\n"))
    h_view1 = source_df.to_numpy(dtype=np.float32)
    h_view2 = h_view1 + 0.2
    with result_path.open("wb") as file:
        pickle.dump(
            {
                "h_fused": (h_view1 + h_view2) / 2,
                "labels": np.array([0, 0, 1, 1]),
                "preprocessing": {
                    "method": "none",
                    "feature_columns": list(source_df.columns),
                    "source_sha256": source_sha256,
                },
                "init": "k-means",
                "score": 0.9,
                "iteration": 3,
            },
            file,
        )

    result = load_mvdec_result(result_path=result_path, data_path=data_path)

    assert result.h_fused.shape == (4, 2)

    source_df.iloc[::-1].to_csv(data_path, index=False, lineterminator="\n")
    with pytest.raises(ValueError, match="source CSV hash"):
        load_mvdec_result(result_path=result_path, data_path=data_path)


def test_load_mvdec_result_rejects_legacy_full_view_average(tmp_path):
    data_path = tmp_path / "airpollution.csv"
    result_path = tmp_path / "legacy_mvdec2025.pkl"
    pd.DataFrame({"feature": range(4)}).to_csv(data_path, index=False)
    h_view1 = np.array(
        [[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]],
        dtype=np.float32,
    )
    h_view2 = h_view1 + 0.2
    with result_path.open("wb") as file:
        pickle.dump(
            {
                "fusion_contract": "mvdec2025_figure_output_average",
                "view_output_layout": "eq5_compatible_10_plus_13",
                "h_view1": h_view1,
                "h_view2": h_view2,
                "h_fused": (h_view1 + h_view2) / 2,
                "labels": np.array([0, 0, 1, 1]),
                "fusion_dim": 2,
                "init": "k-means",
                "score": 0.5,
                "iteration": 3,
            },
            file,
        )

    with pytest.raises(ValueError, match="legacy full-view-output average"):
        load_mvdec_result(result_path=result_path, data_path=data_path)


@pytest.mark.parametrize(
    (
        "greedy_eigen_direction",
        "greedy_eigen_index",
        "greedy_target_mode",
    ),
    [
        ("largest", 1, "frozen_snapshot"),
        ("largest", 1, "selected_dimension_only"),
        ("smallest", 0, "frozen_snapshot"),
        ("smallest", 0, "selected_dimension_only"),
    ],
)
def test_load_mvdec_result_rejects_legacy_airpollution_modes(
    tmp_path,
    greedy_eigen_direction,
    greedy_eigen_index,
    greedy_target_mode,
):
    data_path = tmp_path / "airpollution.csv"
    result_path = tmp_path / "mvdec2025.pkl"
    source_df = pd.DataFrame(
        {
            "distance": [0.0, 1.0, 2.0, 3.0],
            "pollution": [10.0, 20.0, 30.0, 40.0],
        }
    )
    source_df.to_csv(data_path, index=False)
    source_sha256 = _normalized_file_sha256(data_path)
    h_view1 = np.array(
        [[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]],
        dtype=np.float32,
    )
    h_view2 = h_view1 + 0.2
    with result_path.open("wb") as file:
        pickle.dump(
            {
                "dataset": "AIRPOLLUTION",
                "fusion_contract": "mvdec2025_encoder_average",
                "view_output_layout": "eq5_compatible_2_plus_2",
                "h_view1": h_view1,
                "h_view2": h_view2,
                "h_fused": (h_view1 + h_view2) / 2,
                "labels": np.array([0, 0, 1, 1]),
                "fusion_dim": 2,
                "view1_latent_dim": 2,
                "eigenvalue_order": "ascending",
                "greedy_eigen_direction": greedy_eigen_direction,
                "greedy_eigen_index": greedy_eigen_index,
                "greedy_target_mode": greedy_target_mode,
                "kmeans_refresh_policy": "one_epoch",
                "batches_per_epoch": 2,
                "stop_reason": "converged_assignment",
                "refinement_epochs_completed": 3,
                "preprocessing": {
                    "method": "minmax",
                    "feature_range": [0.0, 1.0],
                    "feature_columns": list(source_df.columns),
                    "data_min": [0.0, 10.0],
                    "data_max": [3.0, 40.0],
                    "source_sha256": source_sha256,
                },
                "config": {
                    "kmeans_n_init": 100,
                    "batch_size": 2,
                    "kmeans_refresh_policy": "one_epoch",
                    "batches_per_epoch": 2,
                    "update_interval": 2,
                    "max_refinement_epochs": 5,
                    "max_training_steps": 10,
                    "stop_reason": "converged_assignment",
                    "refinement_epochs_completed": 3,
                    "eigenvalue_order": "ascending",
                    "greedy_eigen_direction": greedy_eigen_direction,
                    "greedy_eigen_index": greedy_eigen_index,
                    "greedy_target_mode": greedy_target_mode,
                },
                "init": "k-means",
                "score": 0.5,
                "iteration": 3,
            },
            file,
        )

    with pytest.raises(ValueError, match="Legacy MvDEC artifact"):
        load_mvdec_result(result_path=result_path, data_path=data_path)


def test_load_mvdec_result_rejects_missing_airpollution_eigen_mode(tmp_path):
    data_path = tmp_path / "airpollution.csv"
    result_path = tmp_path / "mvdec2025.pkl"
    source_df = pd.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0]})
    source_df.to_csv(data_path, index=False)
    h_view1 = np.array(
        [[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]],
        dtype=np.float32,
    )
    h_view2 = h_view1 + 0.2
    with result_path.open("wb") as file:
        pickle.dump(
            {
                "dataset": "AIRPOLLUTION",
                "fusion_contract": "mvdec2025_encoder_average",
                "view_output_layout": "eq5_compatible_2_plus_1",
                "h_view1": h_view1,
                "h_view2": h_view2,
                "h_fused": (h_view1 + h_view2) / 2,
                "labels": np.array([0, 0, 1, 1]),
                "fusion_dim": 2,
                "view1_latent_dim": 2,
                "preprocessing": {
                    "method": "minmax",
                    "feature_range": [0.0, 1.0],
                    "feature_columns": ["feature"],
                    "data_min": [0.0],
                    "data_max": [3.0],
                    "source_sha256": _normalized_file_sha256(data_path),
                },
                "init": "k-means",
                "score": 0.5,
                "iteration": 3,
            },
            file,
        )

    with pytest.raises(ValueError, match="Legacy MvDEC artifact"):
        load_mvdec_result(result_path=result_path, data_path=data_path)


def test_load_mvdec_result_rejects_encoder_average_latent_dim_mismatch(tmp_path):
    data_path = tmp_path / "airpollution.csv"
    result_path = tmp_path / "mvdec2025.pkl"
    pd.DataFrame({"feature": range(4)}).to_csv(data_path, index=False)
    h_view1 = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [5.0, 5.0],
            [5.1, 5.0],
        ],
        dtype=np.float32,
    )
    h_view2 = h_view1 + 0.2
    labels = np.array([0, 0, 1, 1])
    with result_path.open("wb") as file:
        pickle.dump(
            {
                "fusion_contract": "mvdec2025_encoder_average",
                "view_output_layout": "eq5_compatible_10_plus_13",
                "h_view1": h_view1,
                "h_view2": h_view2,
                "h_fused": (h_view1 + h_view2) / 2,
                "labels": labels,
                "fusion_dim": 2,
                "view1_latent_dim": 3,
                "init": "k-means",
                "score": 0.5,
                "iteration": 3,
            },
            file,
        )

    with pytest.raises(ValueError, match="Legacy MvDEC artifact"):
        load_mvdec_result(result_path=result_path, data_path=data_path)


def test_load_mvdec_result_rejects_legacy_concat_named_h_fused(tmp_path):
    data_path = tmp_path / "airpollution.csv"
    result_path = tmp_path / "legacy.pkl"
    pd.DataFrame({"feature": range(4)}).to_csv(data_path, index=False)
    h_fused = np.arange(4 * 17, dtype=np.float32).reshape(4, 17)
    labels = np.array([0, 0, 1, 1])
    with result_path.open("wb") as file:
        pickle.dump(
            {
                "h_fused": h_fused,
                "labels": labels,
                "input_dim": 13,
                "config": {"view1_latent_dim": 4},
                "init": "k-means++",
                "score": 0.1,
                "iteration": 1,
            },
            file,
        )

    with pytest.raises(ValueError, match="legacy concat representation"):
        load_mvdec_result(result_path=result_path, data_path=data_path)
