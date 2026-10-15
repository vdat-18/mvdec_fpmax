import numpy as np
import pandas as pd
import pytest

import pipeline.mvdec_audit as mvdec_audit
from pipeline.mvdec_audit import (
    artifact_data_path,
    audit_run_root,
    validate_expected_run_set,
    validate_public_assignments,
    validate_public_source,
    validate_training_log,
)
from pipeline.mvdec_runs import (
    HDF5_SIGNATURE,
    RUN_OUTPUT_NAMES,
    WEIGHT_OUTPUT_NAMES,
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
            "dataset": "TIKI",
            "protocol_id": "mvdec_dekm_consistent_v1",
            "seed": 42,
            "run_index": 1,
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
        content = HDF5_SIGNATURE if name in WEIGHT_OUTPUT_NAMES else name.encode()
        getattr(paths, name).write_bytes(content)
    output_paths = {name: getattr(paths, name).name for name in RUN_OUTPUT_NAMES}
    write_run_manifest(
        paths.manifest,
        {
            "run_id": paths.run_id,
            "status": "complete",
            "dataset": "TIKI",
            "protocol_id": "mvdec_dekm_consistent_v1",
            "seed": 42,
            "run_index": 1,
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

def _write_training_log(
    path,
    include_l5: bool = True,
    windows: tuple[int, ...] = (1, 2),
    l5_value: str = "0.5",
) -> None:
    loss = "loss:L1_reconstruction:1.0"
    if include_l5:
        loss += f", L5_alignment:{l5_value}"
    step_rows = [
        (
            f"phase:paper_step9_after_release_update_window_{window},{loss},"
            f"refinement_epoch:{window}"
        )
        for window in windows
    ]
    path.write_text(
        "\n".join(
            [
                *step_rows,
                (
                    "phase:training_stop,stop_reason:converged_assignment,"
                    "training_steps_completed:20,refinement_epochs_completed:2,"
                    "refinement_progress_unit:kmeans_refresh_windows"
                ),
                "phase:final_recomputed_state",
                "run 1/1,seed:42,run_id:run,protocol_id:protocol",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_validate_training_log_checks_complete_l5_trajectory(tmp_path) -> None:
    log_path = tmp_path / "training_log.csv"
    _write_training_log(log_path)
    manifest = {
        "seed": 42,
        "run_id": "run",
        "protocol_id": "protocol",
        "paths": {"training_log": str(log_path)},
    }
    artifact = {
        "training_steps_completed": 20,
        "refinement_epochs_completed": 2,
        "refinement_progress_unit": "kmeans_refresh_windows",
        "stop_reason": "converged_assignment",
        "config": {"loss_weights": {"cross_view_alignment": 100.0}},
    }

    validate_training_log(manifest, artifact)


def test_validate_training_log_rejects_missing_l5_trajectory(tmp_path) -> None:
    log_path = tmp_path / "training_log.csv"
    _write_training_log(log_path, include_l5=False)
    manifest = {
        "seed": 42,
        "run_id": "run",
        "protocol_id": "protocol",
        "paths": {"training_log": str(log_path)},
    }
    artifact = {
        "training_steps_completed": 20,
        "refinement_epochs_completed": 2,
        "refinement_progress_unit": "kmeans_refresh_windows",
        "stop_reason": "converged_assignment",
        "config": {"loss_weights": {"cross_view_alignment": 100.0}},
    }

    with pytest.raises(ValueError, match="L5 trajectory"):
        validate_training_log(manifest, artifact)

def test_validate_training_log_rejects_prefix_only_provenance_match(tmp_path) -> None:
    log_path = tmp_path / "training_log.csv"
    _write_training_log(log_path)
    manifest = {
        "seed": 4,
        "run_id": "run",
        "protocol_id": "protocol",
        "paths": {"training_log": str(log_path)},
    }
    artifact = {
        "training_steps_completed": 20,
        "refinement_epochs_completed": 2,
        "refinement_progress_unit": "kmeans_refresh_windows",
        "stop_reason": "converged_assignment",
        "config": {"loss_weights": {"cross_view_alignment": 100.0}},
    }

    with pytest.raises(ValueError, match="summary provenance"):
        validate_training_log(manifest, artifact)

def test_validate_training_log_rejects_noncontiguous_windows(tmp_path) -> None:
    log_path = tmp_path / "training_log.csv"
    _write_training_log(log_path, windows=(1, 1))
    manifest = {
        "seed": 42,
        "run_id": "run",
        "protocol_id": "protocol",
        "paths": {"training_log": str(log_path)},
    }
    artifact = {
        "training_steps_completed": 20,
        "refinement_epochs_completed": 2,
        "refinement_progress_unit": "kmeans_refresh_windows",
        "stop_reason": "converged_assignment",
        "config": {"loss_weights": {"cross_view_alignment": 100.0}},
    }

    with pytest.raises(ValueError, match="not contiguous"):
        validate_training_log(manifest, artifact)


def test_validate_training_log_rejects_nonfinite_l5(tmp_path) -> None:
    log_path = tmp_path / "training_log.csv"
    _write_training_log(log_path, l5_value="nan")
    manifest = {
        "seed": 42,
        "run_id": "run",
        "protocol_id": "protocol",
        "paths": {"training_log": str(log_path)},
    }
    artifact = {
        "training_steps_completed": 20,
        "refinement_epochs_completed": 2,
        "refinement_progress_unit": "kmeans_refresh_windows",
        "stop_reason": "converged_assignment",
        "config": {"loss_weights": {"cross_view_alignment": 100.0}},
    }

    with pytest.raises(ValueError, match="non-finite L5"):
        validate_training_log(manifest, artifact)

def test_validate_expected_run_set_checks_order_and_provenance() -> None:
    manifests = [
        {
            "dataset": "REUTERS",
            "started_at_utc": "2026-10-15T00:00:00+00:00",
            "protocol_id": "protocol",
            "training_commit": "a" * 40,
            "seed": 42,
        },
        {
            "dataset": "RCV1",
            "started_at_utc": "2026-10-15T01:00:00+00:00",
            "protocol_id": "protocol",
            "training_commit": "a" * 40,
            "seed": 42,
        },
        {
            "dataset": "20NEWS",
            "started_at_utc": "2026-10-15T02:00:00+00:00",
            "protocol_id": "protocol",
            "training_commit": "a" * 40,
            "seed": 42,
        },
    ]

    validate_expected_run_set(
        manifests,
        expected_datasets=("REUTERS", "RCV1", "20NEWS"),
        expected_protocol_id="protocol",
        expected_training_commit="a" * 40,
        expected_seed=42,
    )


def test_validate_expected_run_set_rejects_wrong_dataset_order() -> None:
    manifests = [
        {
            "dataset": "RCV1",
            "started_at_utc": "2026-10-15T00:00:00+00:00",
        },
        {
            "dataset": "REUTERS",
            "started_at_utc": "2026-10-15T01:00:00+00:00",
        },
    ]

    with pytest.raises(ValueError, match="execution order"):
        validate_expected_run_set(
            manifests,
            expected_datasets=("REUTERS", "RCV1"),
        )
