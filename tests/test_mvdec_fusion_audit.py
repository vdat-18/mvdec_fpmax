import pickle

import numpy as np
import pytest

from pipeline.mvdec_fusion_audit import (
    alignment_diagnostics,
    audit_artifact,
    audit_root,
    diagnostic_representations,
    fused_representation,
)


def test_alignment_diagnostics_detects_opposite_latents() -> None:
    view1 = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    diagnostics = alignment_diagnostics(view1, -view1)

    assert diagnostics.paired_cosine_mean == pytest.approx(-1.0)
    assert diagnostics.centered_paired_cosine_mean == pytest.approx(-1.0)
    assert diagnostics.fusion_retained_norm_mean == pytest.approx(0.0)
    assert diagnostics.centered_fusion_retained_norm_mean == pytest.approx(0.0)
    assert diagnostics.same_coordinate_correlation_mean == pytest.approx(-1.0)


def test_alignment_diagnostics_separates_shared_offset_from_structure() -> None:
    centered = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    shared_offset = np.array([10.0, 10.0])

    diagnostics = alignment_diagnostics(
        centered + shared_offset,
        -centered + shared_offset,
    )

    assert diagnostics.mean_vector_cosine == pytest.approx(1.0)
    assert diagnostics.paired_cosine_mean > 0.98
    assert diagnostics.centered_paired_cosine_mean == pytest.approx(-1.0)
    assert diagnostics.centered_fusion_retained_norm_mean == pytest.approx(0.0)


def test_l2_normalized_fusion_matches_declared_contract() -> None:
    view1 = np.array([[3.0, 0.0], [0.0, 0.0]])
    view2 = np.array([[0.0, 4.0], [0.0, 2.0]])

    fused = fused_representation(
        view1,
        view2,
        "mvdec2025_encoder_l2_normalized_average",
    )

    np.testing.assert_allclose(fused, [[0.5, 0.5], [0.0, 0.5]])


def test_procrustes_control_recovers_rotated_latent() -> None:
    view1 = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    view2 = view1 @ rotation

    representations = diagnostic_representations(view1, view2)

    np.testing.assert_allclose(
        representations["procrustes_aligned_average"],
        view1,
        atol=1e-12,
    )


def test_audit_artifact_exports_independent_assignments(tmp_path) -> None:
    view1 = np.array([[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    view2 = view1.copy()
    artifact = {
        "dataset": "TEST",
        "protocol_id": "protocol",
        "run_id": "run",
        "training_commit": "a" * 40,
        "source_id": "source",
        "source_sha256": "b" * 64,
        "fusion_contract": "mvdec2025_encoder_average",
        "config": {"seed": 42},
        "n_clusters": 2,
        "kmeans_n_init": 100,
        "h_view1": view1,
        "h_view2": view2,
        "h_fused": (view1 + view2) / 2,
        "true_labels": np.array([0, 0, 1, 1]),
    }
    artifact_path = tmp_path / "artifact.pkl"
    with artifact_path.open("wb") as file:
        pickle.dump(artifact, file)

    metrics, diagnostics = audit_artifact(artifact_path, tmp_path / "audit")

    assert set(metrics["representation"]) >= {"view1", "view2", "fused"}
    assert metrics.set_index("representation").loc["fused", "acc"] == 1.0
    assert diagnostics.paired_cosine_mean == pytest.approx(1.0)
    assert (tmp_path / "audit/test/view1_assignments.csv").is_file()
    assert (tmp_path / "audit/test/view2_assignments.csv").is_file()
    assert (tmp_path / "audit/test/fused_assignments.csv").is_file()


def test_audit_artifact_rejects_noncanonical_kmeans_restart_count(tmp_path) -> None:
    view = np.array([[-1.0, 0.0], [1.0, 0.0]])
    artifact = {
        "dataset": "TEST",
        "fusion_contract": "mvdec2025_encoder_average",
        "config": {"seed": 42},
        "n_clusters": 2,
        "kmeans_n_init": 10,
        "h_view1": view,
        "h_view2": view,
        "h_fused": view,
        "true_labels": np.array([0, 1]),
    }
    artifact_path = tmp_path / "artifact.pkl"
    with artifact_path.open("wb") as file:
        pickle.dump(artifact, file)

    with pytest.raises(ValueError, match="n_init=100"):
        audit_artifact(artifact_path, tmp_path / "audit")


def test_audit_artifact_honors_l2_normalized_fusion(tmp_path) -> None:
    view1 = np.array([[-3.0, 0.0], [-2.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    view2 = np.array([[0.0, -4.0], [0.0, -1.0], [0.0, 1.0], [0.0, 4.0]])
    fusion_contract = "mvdec2025_encoder_l2_normalized_average"
    artifact = {
        "dataset": "TEST",
        "protocol_id": "protocol",
        "run_id": "run",
        "training_commit": "a" * 40,
        "source_id": "source",
        "source_sha256": "b" * 64,
        "fusion_contract": fusion_contract,
        "config": {"seed": 42},
        "n_clusters": 2,
        "kmeans_n_init": 100,
        "h_view1": view1,
        "h_view2": view2,
        "h_fused": fused_representation(view1, view2, fusion_contract),
        "true_labels": np.array([0, 0, 1, 1]),
    }
    artifact_path = tmp_path / "artifact.pkl"
    with artifact_path.open("wb") as file:
        pickle.dump(artifact, file)

    metrics, _ = audit_artifact(artifact_path, tmp_path / "audit")

    fused_metrics = metrics.set_index("representation").loc["fused"]
    assert fused_metrics["fusion_contract"] == fusion_contract
    assert fused_metrics["run_id"] == "run"


def test_audit_artifact_rejects_zero_norm_rows(tmp_path) -> None:
    view1 = np.array([[0.0, 0.0], [1.0, 0.0]])
    artifact = {
        "dataset": "TEST",
        "fusion_contract": "mvdec2025_encoder_average",
        "config": {"seed": 42},
        "n_clusters": 2,
        "kmeans_n_init": 100,
        "h_view1": view1,
        "h_view2": np.array([[1.0, 0.0], [2.0, 0.0]]),
        "h_fused": np.array([[0.5, 0.0], [1.5, 0.0]]),
        "true_labels": np.array([0, 1]),
    }
    artifact_path = tmp_path / "artifact.pkl"
    with artifact_path.open("wb") as file:
        pickle.dump(artifact, file)

    with pytest.raises(ValueError, match="zero-norm latent rows"):
        audit_artifact(artifact_path, tmp_path / "audit")


def test_audit_root_rejects_unmanifested_artifacts(tmp_path) -> None:
    (tmp_path / "artifact.pkl").write_bytes(b"not trusted")

    with pytest.raises(ValueError, match="No MvDEC run manifests"):
        audit_root(tmp_path, tmp_path / "audit")
