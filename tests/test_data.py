import hashlib
import pickle

import numpy as np
import pandas as pd
import pytest

from config import H_FUSED_COLUMNS
from pipeline import data as data_module
from pipeline.clustering import compute_gower_distance, compute_silhouette_diagnostics
from pipeline.data import (
    _validate_airpollution_preprocessing,
    _validate_mvdec2025_contract,
    _validate_mvdec_run_manifest,
    _validate_view1_refinement_transition_metadata,
    load_mvdec_result,
)
from pipeline.dekm_checkpoint import (
    CHECKPOINT_PROVENANCE_SHA_FIELDS,
    DEKM_F_CONFIG_HASH,
    DEKM_F_MANIFEST_SHA256,
    DEKM_F_PROTOCOL_ID,
    DEKM_F_REUTERS_SOURCE_SHA256,
    DEKM_F_WEIGHT_SHA256,
)
from pipeline.mvdec_contract import FUSION_ENCODER_L2_NORMALIZED_AVERAGE
from pipeline.mvdec_runs import canonical_config_hash, file_sha256, write_run_manifest


@pytest.fixture(autouse=True)
def _register_imported_joint_protocol(request, monkeypatch):
    """Treat the imported-View1 protocol ID as jointly refined for joint tests.

    The registered protocol freezes View1, so the joint-transition validators are
    exercised only by tests that opt in through their name.
    """

    if "imported_joint" not in request.node.name:
        return
    joint_id = (
        "mvdec_2025_frozen_dekm_f_view1_eq11_true_bottleneck_release_l2_seed42_v1"
    )
    monkeypatch.setattr(
        data_module, "IMPORTED_JOINT_VIEW1_PROTOCOL_IDS", frozenset({joint_id})
    )
    monkeypatch.setattr(
        data_module,
        "FROZEN_VIEW1_PROTOCOL_IDS",
        data_module.FROZEN_VIEW1_PROTOCOL_IDS - {joint_id},
    )


def _external_view1_provenance():
    return {
        "method": "DEKM_F",
        "protocol_id": DEKM_F_PROTOCOL_ID,
        "config_hash": DEKM_F_CONFIG_HASH,
        "dataset": "REUTERS",
        "seed": 42,
        "source_sha256": DEKM_F_REUTERS_SOURCE_SHA256,
        "manifest_sha256": DEKM_F_MANIFEST_SHA256,
        "weight_sha256": DEKM_F_WEIGHT_SHA256,
        "input_normalization": "release_l2_unit_norm",
        "refinement_batching": "full_dataset_each_update",
        "claim_scope": "controlled_diagnostic_not_exact_paper_truth",
    }


def _write_frozen_run_manifest(tmp_path, external_checkpoint):
    artifact_path = tmp_path / "artifact.pkl"
    artifact_path.write_bytes(b"artifact")
    config = {
        "training_commit": "f" * 40,
        "source_id": "dekm_google_drive_reuters_10k",
        "source_sha256": DEKM_F_REUTERS_SOURCE_SHA256,
        "n_samples": 10_000,
        "n_features": 2_000,
        "n_classes": 4,
        "view_input_contract": "same_full_feature_matrix_both_views_v1",
        "representation_metric_semantics": (
            "frozen_dekm_f_view1_refined_view2_independent_kmeans_v1"
        ),
        "view1_initialization_policy": "imported_dekm_f_view1",
        "view1_refinement_policy": "frozen_imported_dekm_f_view1",
        "external_view1_checkpoint": dict(external_checkpoint),
    }
    config_hash = canonical_config_hash(config)
    artifact = {
        **config,
        "run_id": "reuters_frozen_seed_42",
        "config_hash": config_hash,
        "dataset": "REUTERS",
        "protocol_id": "mvdec_2025_frozen_dekm_f_view1_release_l2_v1",
        "config": {
            **config,
            "run_id": "reuters_frozen_seed_42",
            "config_hash": config_hash,
            "seed": 42,
        },
    }
    manifest = {
        **config,
        "run_id": artifact["run_id"],
        "status": "complete",
        "dataset": artifact["dataset"],
        "protocol_id": artifact["protocol_id"],
        "config_hash": config_hash,
        "config": config,
        "seed": 42,
        "paths": {"artifact": artifact_path.name},
        "output_sha256": {"artifact": file_sha256(artifact_path)},
    }
    write_run_manifest(tmp_path / "manifest.json", manifest)
    return artifact, artifact_path, manifest




def _imported_joint_transition(h_view1):
    from pipeline.mvdec_integrity import array_sha256

    final_latent_hash = array_sha256(h_view1)
    return {
        "schema_version": 1,
        "weight_count": 2,
        "initial_weights_sha256": "1" * 64,
        "final_weights_sha256": "2" * 64,
        "initial_latent_sha256": "3" * 64,
        "final_latent_sha256": final_latent_hash,
        "initial_view1_acc": 0.7557,
        "initial_view1_nmi": 0.593243233623544,
        "final_view1_acc": 0.76,
        "final_view1_nmi": 0.60,
        "view1_nonzero_gradient_observed": True,
        "view1_max_abs_gradient": 0.25,
        "weights_changed": True,
        "latents_changed": True,
        "h_view1_sha256": final_latent_hash,
    }


def _write_imported_joint_run_manifest(tmp_path):
    artifact_path = tmp_path / "artifact.pkl"
    artifact_path.write_bytes(b"artifact")
    h_view1 = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.2, 5.1]])
    transition = _imported_joint_transition(h_view1)
    config = {
        "training_commit": "f" * 40,
        "source_id": "dekm_google_drive_reuters_10k",
        "source_sha256": DEKM_F_REUTERS_SOURCE_SHA256,
        "n_samples": 10_000,
        "n_features": 2_000,
        "n_classes": 4,
        "view_input_contract": "same_full_feature_matrix_both_views_v1",
        "representation_metric_semantics": (
            "imported_dekm_f_view1_jointly_refined_views_independent_kmeans_v1"
        ),
        "view1_initialization_policy": "imported_dekm_f_view1",
        "view1_refinement_policy": "jointly_refined_views",
        "external_view1_checkpoint": _external_view1_provenance(),
        "view1_refinement_transition": transition,
    }
    config_hash = canonical_config_hash(config)
    artifact = {
        **config,
        "h_view1": h_view1,
        "run_id": "reuters_imported_joint_seed_42",
        "config_hash": config_hash,
        "dataset": "REUTERS",
        "protocol_id": (
            "mvdec_2025_frozen_dekm_f_view1_eq11_true_bottleneck_release_l2_seed42_v1"
        ),
        "config": {
            **config,
            "run_id": "reuters_imported_joint_seed_42",
            "config_hash": config_hash,
            "seed": 42,
        },
    }
    manifest = {
        **config,
        "run_id": artifact["run_id"],
        "status": "complete",
        "dataset": artifact["dataset"],
        "protocol_id": artifact["protocol_id"],
        "config_hash": config_hash,
        "config": dict(config),
        "seed": 42,
        "paths": {"artifact": artifact_path.name},
        "output_sha256": {"artifact": file_sha256(artifact_path)},
    }
    write_run_manifest(tmp_path / "manifest.json", manifest)
    return artifact, artifact_path, manifest


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


def test_l2_normalized_average_contract_is_validated_without_tensorflow(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        data_module,
        "_validate_mvdec_protocol_contract",
        lambda _result: None,
    )
    monkeypatch.setattr(
        data_module,
        "_validate_mvdec_run_manifest",
        lambda _result, _path: None,
    )
    h_view1 = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    h_view2 = np.array([[0.0, 2.0], [0.0, 0.0]], dtype=np.float32)
    expected = np.array([[0.3, 0.9], [0.0, 0.0]], dtype=np.float32)
    artifact = {
        "protocol_id": "mvdec_2025_view2_direct_23_split_l2norm_average_v1",
        "view_output_layout": "eq5_compatible_2_plus_2000",
        "h_view1": h_view1,
        "h_view2": h_view2,
        "fusion_dim": 2,
        "fusion_contract": FUSION_ENCODER_L2_NORMALIZED_AVERAGE,
    }

    _validate_mvdec2025_contract(artifact, expected, tmp_path / "artifact.pkl")

    with pytest.raises(ValueError, match="does not match"):
        _validate_mvdec2025_contract(
            artifact,
            expected + 0.01,
            tmp_path / "artifact.pkl",
        )


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


@pytest.mark.parametrize(
    ("method", "fitted_metadata"),
    [
        ("none", {}),
        (
            "standard",
            {
                "data_mean": [1.5, 25.0],
                "data_std": [1.118033988749895, 11.180339887498949],
            },
        ),
    ],
)
def test_airpollution_preprocessing_accepts_supported_scaling(
    tmp_path,
    method,
    fitted_metadata,
):
    """Air Pollution sensitivity artifacts retain auditable scaler metadata."""

    data_path = tmp_path / "airpollution.csv"
    source_df = pd.DataFrame(
        {
            "distance": [0.0, 1.0, 2.0, 3.0],
            "pollution": [10.0, 20.0, 30.0, 40.0],
        }
    )
    source_df.to_csv(data_path, index=False)
    artifact = {
        "dataset": "AIRPOLLUTION",
        "eigenvalue_order": "ascending",
        "greedy_eigen_direction": "largest",
        "greedy_eigen_index": 1,
        "greedy_target_mode": "frozen_snapshot",
        "fusion_dim": 2,
        "kmeans_refresh_policy": "one_epoch",
        "batches_per_epoch": 2,
        "stop_reason": "converged_assignment",
        "refinement_epochs_completed": 3,
        "preprocessing": {
            "method": method,
            "feature_range": None,
            "feature_columns": list(source_df.columns),
            "data_min": [0.0, 10.0],
            "data_max": [3.0, 40.0],
            "source_sha256": _normalized_file_sha256(data_path),
            **fitted_metadata,
        },
        "config": {
            "eigenvalue_order": "ascending",
            "greedy_eigen_direction": "largest",
            "greedy_eigen_index": 1,
            "greedy_target_mode": "frozen_snapshot",
            "kmeans_n_init": 100,
            "batch_size": 2,
            "kmeans_refresh_policy": "one_epoch",
            "batches_per_epoch": 2,
            "update_interval": 2,
            "max_refinement_epochs": 5,
            "max_training_steps": 10,
            "stop_reason": "converged_assignment",
            "refinement_epochs_completed": 3,
        },
    }

    _validate_airpollution_preprocessing(artifact, data_path, source_df)
    if method == "standard":
        artifact["preprocessing"]["data_mean"][0] = 999.0
        with pytest.raises(ValueError, match="mean/std metadata"):
            _validate_airpollution_preprocessing(artifact, data_path, source_df)


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


def test_frozen_run_manifest_accepts_consistent_validated_provenance(tmp_path) -> None:
    artifact, artifact_path, _ = _write_frozen_run_manifest(
        tmp_path,
        _external_view1_provenance(),
    )

    _validate_mvdec_run_manifest(artifact, artifact_path)


def test_frozen_true_bottleneck_run_manifest_accepts_frozen_provenance(
    tmp_path,
) -> None:
    artifact, artifact_path, manifest = _write_frozen_run_manifest(
        tmp_path,
        _external_view1_provenance(),
    )
    protocol_id = (
        "mvdec_2025_frozen_dekm_f_view1_eq11_true_bottleneck_release_l2_seed42_v1"
    )
    artifact["protocol_id"] = protocol_id
    manifest["protocol_id"] = protocol_id
    write_run_manifest(tmp_path / "manifest.json", manifest)

    _validate_mvdec_run_manifest(artifact, artifact_path)


@pytest.mark.parametrize("sha_field", CHECKPOINT_PROVENANCE_SHA_FIELDS)
def test_frozen_run_manifest_validates_every_checkpoint_sha(
    tmp_path,
    sha_field,
) -> None:
    artifact, artifact_path, _ = _write_frozen_run_manifest(
        tmp_path,
        _external_view1_provenance(),
    )
    artifact["external_view1_checkpoint"][sha_field] = "not-a-sha"
    artifact["config"]["external_view1_checkpoint"][sha_field] = "not-a-sha"

    with pytest.raises(ValueError, match="run manifest"):
        _validate_mvdec_run_manifest(artifact, artifact_path)


@pytest.mark.parametrize("location", ["artifact_config", "manifest", "manifest_config"])
def test_frozen_run_manifest_rejects_checkpoint_provenance_location_mismatch(
    tmp_path,
    location,
) -> None:
    provenance = _external_view1_provenance()
    artifact, artifact_path, manifest = _write_frozen_run_manifest(
        tmp_path,
        provenance,
    )
    mismatched = {**provenance, "manifest_sha256": "b" * 64}
    if location == "artifact_config":
        artifact["config"]["external_view1_checkpoint"] = mismatched
    elif location == "manifest":
        manifest["external_view1_checkpoint"] = mismatched
        write_run_manifest(tmp_path / "manifest.json", manifest)
    else:
        manifest["config"]["external_view1_checkpoint"] = mismatched
        manifest["config_hash"] = canonical_config_hash(manifest["config"])
        write_run_manifest(tmp_path / "manifest.json", manifest)

    with pytest.raises(ValueError, match="run manifest"):
        _validate_mvdec_run_manifest(artifact, artifact_path)


def test_non_frozen_run_manifest_rejects_external_checkpoint_provenance(
    tmp_path,
) -> None:
    provenance = _external_view1_provenance()
    artifact, artifact_path, manifest = _write_frozen_run_manifest(
        tmp_path,
        provenance,
    )
    artifact["view1_refinement_policy"] = "jointly_refined_views"
    artifact["config"]["view1_refinement_policy"] = "jointly_refined_views"
    manifest["view1_refinement_policy"] = "jointly_refined_views"
    manifest["config"]["view1_refinement_policy"] = "jointly_refined_views"
    manifest["config_hash"] = canonical_config_hash(manifest["config"])
    artifact["config_hash"] = manifest["config_hash"]
    artifact["config"]["config_hash"] = manifest["config_hash"]
    write_run_manifest(tmp_path / "manifest.json", manifest)

    with pytest.raises(ValueError, match="run manifest"):
        _validate_mvdec_run_manifest(artifact, artifact_path)


def test_imported_joint_run_manifest_requires_consistent_transition(tmp_path) -> None:
    artifact, artifact_path, _ = _write_imported_joint_run_manifest(tmp_path)

    _validate_mvdec_run_manifest(artifact, artifact_path)


@pytest.mark.parametrize(
    ("policy_field", "invalid_value"),
    [
        ("view1_initialization_policy", "pretrained_mvdec_view1"),
        ("view1_refinement_policy", "frozen_imported_dekm_f_view1"),
    ],
)
@pytest.mark.parametrize(
    "location",
    ["artifact", "artifact_config", "manifest", "manifest_config"],
)
def test_imported_joint_run_manifest_rejects_policy_location_mismatch(
    tmp_path,
    policy_field,
    invalid_value,
    location,
) -> None:
    artifact, artifact_path, manifest = _write_imported_joint_run_manifest(tmp_path)
    locations = {
        "artifact": artifact,
        "artifact_config": artifact["config"],
        "manifest": manifest,
        "manifest_config": manifest["config"],
    }
    locations[location][policy_field] = invalid_value
    if location in {"manifest", "manifest_config"}:
        write_run_manifest(tmp_path / "manifest.json", manifest)

    with pytest.raises(ValueError, match="run manifest"):
        _validate_mvdec_run_manifest(artifact, artifact_path)


def test_ordinary_run_manifest_allows_historical_absent_view1_policies(
    tmp_path,
) -> None:
    artifact, artifact_path, manifest = _write_imported_joint_run_manifest(tmp_path)
    protocol_id = "mvdec_2025_paper_resolved_release_l2_full_batch_v1"
    external_fields = (
        "view1_initialization_policy",
        "view1_refinement_policy",
        "external_view1_checkpoint",
        "view1_refinement_transition",
    )
    for location in (artifact, artifact["config"], manifest, manifest["config"]):
        for field in external_fields:
            location.pop(field, None)
    artifact["protocol_id"] = protocol_id
    manifest["protocol_id"] = protocol_id
    config_hash = canonical_config_hash(manifest["config"])
    artifact["config_hash"] = config_hash
    artifact["config"]["config_hash"] = config_hash
    manifest["config_hash"] = config_hash
    write_run_manifest(tmp_path / "manifest.json", manifest)

    _validate_mvdec_run_manifest(artifact, artifact_path)


@pytest.mark.parametrize(
    "location",
    ["artifact", "artifact_config", "manifest", "manifest_config"],
)
def test_imported_joint_run_manifest_rejects_missing_transition(
    tmp_path,
    location,
) -> None:
    artifact, artifact_path, manifest = _write_imported_joint_run_manifest(tmp_path)
    if location == "artifact":
        artifact.pop("view1_refinement_transition")
    elif location == "artifact_config":
        artifact["config"].pop("view1_refinement_transition")
    elif location == "manifest":
        manifest.pop("view1_refinement_transition")
        write_run_manifest(tmp_path / "manifest.json", manifest)
    else:
        manifest["config"].pop("view1_refinement_transition")
        write_run_manifest(tmp_path / "manifest.json", manifest)

    with pytest.raises(ValueError, match="run manifest"):
        _validate_mvdec_run_manifest(artifact, artifact_path)


def test_imported_joint_run_manifest_rejects_tampered_final_latent(tmp_path) -> None:
    artifact, artifact_path, manifest = _write_imported_joint_run_manifest(tmp_path)
    tampered = dict(artifact["view1_refinement_transition"])
    tampered["h_view1_sha256"] = "0" * 64
    for location in (artifact, artifact["config"], manifest, manifest["config"]):
        location["view1_refinement_transition"] = dict(tampered)
    write_run_manifest(tmp_path / "manifest.json", manifest)

    with pytest.raises(ValueError, match="run manifest"):
        _validate_mvdec_run_manifest(artifact, artifact_path)


def test_imported_joint_run_manifest_rejects_frozen_proof(tmp_path) -> None:
    artifact, artifact_path, manifest = _write_imported_joint_run_manifest(tmp_path)
    for location in (artifact, artifact["config"], manifest, manifest["config"]):
        location["frozen_view1_proof"] = {"invalid": True}
    write_run_manifest(tmp_path / "manifest.json", manifest)

    with pytest.raises(ValueError, match="run manifest"):
        _validate_mvdec_run_manifest(artifact, artifact_path)


def test_loader_requires_and_validates_imported_joint_transition() -> None:
    h_view1 = np.array([[0.0, 0.0], [1.0, 1.0]])
    transition = _imported_joint_transition(h_view1)
    artifact = {
        "protocol_id": (
            "mvdec_2025_frozen_dekm_f_view1_eq11_true_bottleneck_release_l2_seed42_v1"
        ),
        "view1_refinement_transition": transition,
        "config": {"view1_refinement_transition": dict(transition)},
    }

    assert (
        _validate_view1_refinement_transition_metadata(artifact, h_view1)
        == transition
    )

    missing = {**artifact, "view1_refinement_transition": None}
    with pytest.raises(ValueError, match="transition fields"):
        _validate_view1_refinement_transition_metadata(missing, h_view1)

    tampered = {
        **artifact,
        "view1_refinement_transition": {
            **transition,
            "h_view1_sha256": "0" * 64,
        },
    }
    tampered["config"] = {
        "view1_refinement_transition": dict(
            tampered["view1_refinement_transition"]
        )
    }
    with pytest.raises(ValueError, match="latent hash"):
        _validate_view1_refinement_transition_metadata(tampered, h_view1)


def test_loader_rejects_transition_for_non_imported_joint_protocol() -> None:
    h_view1 = np.array([[0.0, 0.0], [1.0, 1.0]])
    transition = _imported_joint_transition(h_view1)
    artifact = {
        "protocol_id": "mvdec_2025_frozen_dekm_f_view1_release_l2_v1",
        "view1_refinement_transition": transition,
        "config": {"view1_refinement_transition": dict(transition)},
    }

    with pytest.raises(ValueError, match="Only imported-joint"):
        _validate_view1_refinement_transition_metadata(artifact, h_view1)
