import numpy as np
import pandas as pd
import pytest

import pipeline.mvdec_audit as mvdec_audit
from pipeline.mvdec_audit import (
    artifact_data_path,
    audit_run_root,
    validate_public_assignments,
    validate_public_source,
)
from pipeline.mvdec_runs import (
    RUN_OUTPUT_NAMES,
    file_sha256,
    resolve_run_paths,
    write_run_manifest,
)


def test_artifact_data_path_uses_structured_preprocessing_metadata(tmp_path):
    source_path = tmp_path / "source.csv"
    manifest = {
        "config": {"preprocessing": {"source_path": str(source_path)}},
        "paths": {"assignments": "unused.csv"},
    }

    assert artifact_data_path(manifest) == source_path


def test_audit_retains_failed_manifest_without_loading_artifact(tmp_path):
    config = {"batch_size": 256}
    paths = resolve_run_paths(
        tmp_path,
        "TIKI",
        "mvdec_dekm_consistent_v1",
        config,
        seed=42,
        run_index=1,
    )
    paths.run_dir.mkdir(parents=True)
    write_run_manifest(
        paths.manifest,
        {
            "run_id": paths.run_id,
            "status": "failed",
            "config": config,
            "config_hash": paths.config_hash,
        },
    )

    assert audit_run_root(tmp_path) == (0, 1)


def test_audit_validates_and_reloads_complete_artifact(tmp_path, monkeypatch):
    source_path = tmp_path / "source.csv"
    source_path.write_text("feature\n1\n", encoding="utf-8")
    config = {"preprocessing": {"source_path": str(source_path)}}
    paths = resolve_run_paths(
        tmp_path,
        "TIKI",
        "mvdec_dekm_consistent_v1",
        config,
        seed=42,
        run_index=1,
    )
    paths.run_dir.mkdir(parents=True)
    paths.artifact.write_bytes(b"artifact")
    paths.assignments.write_text("cluster\n0\n", encoding="utf-8")
    for name in RUN_OUTPUT_NAMES - {"artifact", "assignments"}:
        getattr(paths, name).write_bytes(name.encode())
    output_paths = {name: getattr(paths, name).name for name in RUN_OUTPUT_NAMES}
    write_run_manifest(
        paths.manifest,
        {
            "run_id": paths.run_id,
            "status": "complete",
            "config": config,
            "config_hash": paths.config_hash,
            "paths": output_paths,
            "output_sha256": {
                name: file_sha256(getattr(paths, name))
                for name in RUN_OUTPUT_NAMES
            },
        },
    )
    loaded = []

    def fake_load_mvdec_result(*, result_path, data_path):
        loaded.append((result_path, data_path))

    monkeypatch.setattr(mvdec_audit, "load_mvdec_result", fake_load_mvdec_result)

    assert audit_run_root(tmp_path) == (1, 0)
    assert loaded == [(paths.artifact, source_path)]


def test_validate_public_source_rejects_noncanonical_20news(tmp_path) -> None:
    manifest = {
        "dataset": "20NEWS",
        "source_id": "dekm_google_drive_npz_18057",
        "config": {
            "protocol_contract": {
                "preprocessing": {"input_normalization": "release_l2_unit_norm"}
            }
        },
    }

    with pytest.raises(ValueError, match="sklearn source"):
        validate_public_source(manifest, tmp_path)


def test_validate_public_source_rejects_runtime_mismatch(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(mvdec_audit.np, "__version__", "2.4.6")
    monkeypatch.setattr(mvdec_audit, "sklearn_version", "1.6.1")
    manifest = {
        "dataset": "RCV1",
        "source_id": "dekm_google_drive_indices_sklearn_rcv1",
        "runtime": {
            "numpy_version": "2.1.3",
            "scikit_learn_version": "1.6.1",
        },
        "config": {
            "protocol_contract": {
                "preprocessing": {
                    "input_normalization": "paper_mean_squared_norm_one"
                }
            }
        },
    }

    with pytest.raises(ValueError, match="NumPy expected '2.1.3', found '2.4.6'"):
        validate_public_source(manifest, tmp_path)


def test_validate_public_assignments_recomputes_metrics() -> None:
    assignments = pd.DataFrame(
        {
            "sample_index": np.arange(4),
            "cluster": [1, 1, 0, 0],
            "true_label": [0, 0, 1, 1],
        }
    )
    manifest = {
        "n_samples": 4,
        "n_classes": 2,
        "metrics": {"acc": 1.0, "nmi": 1.0},
    }
    artifact = {
        "labels": np.array([1, 1, 0, 0]),
        "true_labels": np.array([0, 0, 1, 1]),
        "acc": 1.0,
        "nmi": 1.0,
    }

    validate_public_assignments(manifest, artifact, assignments)


def test_validate_public_assignments_rejects_metric_mismatch() -> None:
    assignments = pd.DataFrame(
        {
            "sample_index": np.arange(4),
            "cluster": [1, 1, 0, 0],
            "true_label": [0, 0, 1, 1],
        }
    )
    manifest = {
        "n_samples": 4,
        "n_classes": 2,
        "metrics": {"acc": 0.5, "nmi": 1.0},
    }
    artifact = {
        "labels": np.array([1, 1, 0, 0]),
        "true_labels": np.array([0, 0, 1, 1]),
        "acc": 1.0,
        "nmi": 1.0,
    }

    with pytest.raises(ValueError, match="ACC/NMI"):
        validate_public_assignments(manifest, artifact, assignments)
