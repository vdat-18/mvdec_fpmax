from pathlib import Path

import numpy as np
import pytest
from sklearn import __version__ as sklearn_version

from pipeline.mvdec_contract import (
    FUSION_ENCODER_AVERAGE,
    VIEW2_DIRECT_JOINT_HEAD_ARCHITECTURE_ID,
    VIEW_INPUT_CONTRACT,
)
from pipeline.mvdec_paper_figure_audit import (
    EQ4_REPRESENTATION,
    FIGURE1_REPRESENTATION,
    FIGURE2_REPRESENTATION,
    LATENT_CONCATENATION_REPRESENTATION,
    MAX_INFERENCE_BATCH_SIZE,
    RAW_INPUT_REPRESENTATION,
    RECONSTRUCTION_AVERAGE_REPRESENTATION,
    run_paper_figure_diagnostic,
)
from pipeline.mvdec_runs import (
    RUN_OUTPUT_NAMES,
    build_run_id,
    canonical_config_hash,
    file_sha256,
)
from pipeline.public_data import PublicDataset, PublicDatasetSpec


def _write_manifest(
    root: Path,
    *,
    status: str = "complete",
    architecture_mismatch: bool = False,
) -> Path:
    dataset = "REUTERS"
    protocol_id = "mvdec_test_protocol"
    source_sha256 = "b" * 64
    architecture_id = VIEW2_DIRECT_JOINT_HEAD_ARCHITECTURE_ID
    config = {
        "dataset": dataset,
        "protocol_contract": {
            "architecture": {
                "view2": (
                    "wrong_architecture" if architecture_mismatch else architecture_id
                ),
                "view_input_contract": VIEW_INPUT_CONTRACT,
            },
            "preprocessing": {"input_normalization": "release_l2_unit_norm"},
        },
        "source_sha256": source_sha256,
        "source_id": "synthetic_reuters",
        "training_commit": "a" * 40,
        "n_samples": 6,
        "n_features": 2,
        "n_classes": 2,
        "view_input_contract": VIEW_INPUT_CONTRACT,
        "n_clusters": 2,
        "input_shape": 2,
        "hidden_units": 1,
        "view1_filters": [4, 4],
        "view2_base_units": 4,
        "view2_architecture_id": architecture_id,
        "kmeans_n_init": 100,
        "fusion_contract": FUSION_ENCODER_AVERAGE,
    }
    config_hash = canonical_config_hash(config)
    run_dir = root / dataset.lower() / protocol_id / config_hash[:12] / "seed_42"
    run_dir.mkdir(parents=True)
    paths = {
        name: (
            f"{name}.weights.h5"
            if name
            in {"pretrain_view1", "pretrain_view2", "final_view1", "final_view2"}
            else f"{name}.csv"
            if name != "artifact"
            else "artifact.pkl"
        )
        for name in RUN_OUTPUT_NAMES
    }
    for name, relative_path in paths.items():
        path = run_dir / relative_path
        if path.suffix == ".h5":
            path.write_bytes(b"\x89HDF\r\n\x1a\nweights")
        else:
            path.write_bytes(f"{name}\n".encode())
    output_sha256 = {
        name: file_sha256(run_dir / relative_path)
        for name, relative_path in paths.items()
    }
    manifest = {
        "schema_version": 1,
        "config": config,
        "config_hash": config_hash,
        "dataset": dataset,
        "method": "MvDEC test",
        "protocol_id": protocol_id,
        "run_id": build_run_id(dataset, protocol_id, config_hash, 42, 1),
        "run_index": 1,
        "seed": 42,
        "source_id": "synthetic_reuters",
        "source_sha256": source_sha256,
        "training_commit": "a" * 40,
        "view_input_contract": VIEW_INPUT_CONTRACT,
        "n_samples": 6,
        "n_features": 2,
        "n_classes": 2,
        "status": status,
        "paths": paths,
        "output_sha256": output_sha256,
        "runtime": {
            "numpy_version": np.__version__,
            "scikit_learn_version": sklearn_version,
        },
    }
    manifest_path = run_dir / "manifest.json"
    import json

    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _dataset(source_sha256: str = "b" * 64) -> PublicDataset:
    X = np.array(
        [
            [-3.0, -1.0],
            [-2.0, -1.0],
            [-1.0, -1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [3.0, 1.0],
        ],
        dtype=np.float32,
    )
    return PublicDataset(
        spec=PublicDatasetSpec("REUTERS", "Synthetic", 2, 2),
        X=X,
        y=np.array([0, 0, 0, 1, 1, 1], dtype=np.int32),
        source_sha256=source_sha256,
        source_id="synthetic_reuters",
    )


def _model_loader(config, view1_weights, view2_weights):
    assert view1_weights.is_file()
    assert view2_weights.is_file()

    def view1(batch):
        latent = batch[:, :1]
        return np.concatenate((latent, batch), axis=1)

    def view2(batch):
        latent = 2 * batch[:, :1]
        return np.concatenate((latent, 2 * batch), axis=1)

    return view1, view2


def test_paper_figure_runner_exports_metrics_assignments_and_checksums(
    tmp_path,
) -> None:
    manifest_path = _write_manifest(tmp_path / "runs")
    output_dir = tmp_path / "audit"

    metrics = run_paper_figure_diagnostic(
        manifest_path,
        tmp_path / "datasets",
        output_dir,
        inference_batch_size=2,
        model_loader=_model_loader,
        dataset_loader=lambda manifest, root: _dataset(),
    )

    assert tuple(metrics["representation"]) == (
        EQ4_REPRESENTATION,
        FIGURE1_REPRESENTATION,
        FIGURE2_REPRESENTATION,
        RAW_INPUT_REPRESENTATION,
        LATENT_CONCATENATION_REPRESENTATION,
        RECONSTRUCTION_AVERAGE_REPRESENTATION,
    )
    assert metrics["acc"].tolist() == [1.0] * 6
    assert metrics["representation_width"].tolist() == [1, 3, 6, 2, 2, 2]
    assert (output_dir / "paper_figure_diagnostic_metrics.csv").is_file()
    assert (output_dir / "paper_figure_diagnostic_metadata.json").is_file()
    assert (output_dir / "paper_figure_diagnostic_manifest.json").is_file()
    for name in metrics["representation"]:
        assert (output_dir / f"{name}_assignments.csv").is_file()

    import json

    audit_manifest = json.loads(
        (output_dir / "paper_figure_diagnostic_manifest.json").read_text()
    )
    assert audit_manifest["claim_scope"].endswith("not_exact_paper_truth")
    assert audit_manifest["source_run"]["run_id"] == metrics.iloc[0]["run_id"]
    assert set(audit_manifest["outputs"]) == {
        "eq4_latent_average_assignments",
        "figure1_full_output_average_assignments",
        "figure2_full_output_concatenation_assignments",
        "latent_concatenation_control_assignments",
        "raw_input_baseline_assignments",
        "reconstruction_average_control_assignments",
        "metadata",
        "metrics",
    }


def test_paper_figure_runner_rejects_missing_final_weight(tmp_path) -> None:
    manifest_path = _write_manifest(tmp_path / "runs")
    (manifest_path.parent / "final_view1.weights.h5").unlink()

    with pytest.raises(ValueError, match="output hash does not match"):
        run_paper_figure_diagnostic(
            manifest_path,
            tmp_path,
            tmp_path / "audit",
            model_loader=_model_loader,
            dataset_loader=lambda manifest, root: _dataset(),
        )


def test_paper_figure_runner_rejects_architecture_mismatch(tmp_path) -> None:
    manifest_path = _write_manifest(tmp_path / "runs", architecture_mismatch=True)

    with pytest.raises(ValueError, match="architecture identity"):
        run_paper_figure_diagnostic(
            manifest_path,
            tmp_path,
            tmp_path / "audit",
            model_loader=_model_loader,
            dataset_loader=lambda manifest, root: _dataset(),
        )


def test_paper_figure_runner_rejects_source_hash_mismatch(tmp_path) -> None:
    manifest_path = _write_manifest(tmp_path / "runs")

    with pytest.raises(ValueError, match="source_sha256"):
        run_paper_figure_diagnostic(
            manifest_path,
            tmp_path,
            tmp_path / "audit",
            model_loader=_model_loader,
            dataset_loader=lambda manifest, root: _dataset("c" * 64),
        )


def test_paper_figure_runner_rejects_incomplete_manifest(tmp_path) -> None:
    manifest_path = _write_manifest(tmp_path / "runs", status="failed")

    with pytest.raises(ValueError, match="Only complete"):
        run_paper_figure_diagnostic(
            manifest_path,
            tmp_path,
            tmp_path / "audit",
            model_loader=_model_loader,
            dataset_loader=lambda manifest, root: _dataset(),
        )


def test_paper_figure_runner_rejects_excessive_batch_size(tmp_path) -> None:
    manifest_path = _write_manifest(tmp_path / "runs")

    with pytest.raises(ValueError, match="inference_batch_size"):
        run_paper_figure_diagnostic(
            manifest_path,
            tmp_path,
            tmp_path / "audit",
            inference_batch_size=MAX_INFERENCE_BATCH_SIZE + 1,
            model_loader=_model_loader,
            dataset_loader=lambda manifest, root: _dataset(),
        )
