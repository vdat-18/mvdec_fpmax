"""Audit completed MvDEC run manifests and reload every saved artifact."""

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn import __version__ as sklearn_version

from config import DEKM_DATASET_DIR
from pipeline.data import load_mvdec_result, validate_mvdec_artifact
from pipeline.external_metrics import compute_external_metrics
from pipeline.mvdec_runs import (
    RUN_MANIFEST_FILENAME,
    load_run_manifest,
    resolve_manifest_output_path,
    validate_manifest_outputs,
)
from pipeline.public_data import (
    TWENTY_NEWS_RELEASE_NPZ,
    TWENTY_NEWS_SKLEARN_SOURCE,
    load_dekm_public_dataset,
)


def artifact_data_path(manifest: dict[str, object]) -> Path:
    """Resolve the source rows used by the loader from structured metadata."""

    config = manifest.get("config")
    paths = manifest.get("paths")
    if not isinstance(config, dict) or not isinstance(paths, dict):
        raise ValueError("MvDEC manifest is missing config or output paths.")
    preprocessing = config.get("preprocessing")
    if isinstance(preprocessing, dict) and preprocessing.get("source_path"):
        return Path(preprocessing["source_path"])
    return resolve_manifest_output_path(manifest, "assignments")


def _validate_public_source_runtime(manifest: dict[str, object]) -> None:
    """Require the numerical runtime that produced exact source fingerprints."""

    runtime = manifest.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("Public MvDEC manifest is missing runtime provenance.")
    expected = {
        "NumPy": runtime.get("numpy_version"),
        "scikit-learn": runtime.get("scikit_learn_version"),
    }
    actual = {"NumPy": np.__version__, "scikit-learn": sklearn_version}
    mismatches = [
        f"{name} expected {expected[name]!r}, found {actual[name]!r}"
        for name in expected
        if expected[name] != actual[name]
    ]
    if mismatches:
        details = "; ".join(mismatches)
        raise ValueError(
            "Exact public-source audit requires the recorded numerical runtime: "
            f"{details}."
        )


def validate_public_source(
    manifest: dict[str, object],
    dataset_root: Path,
) -> pd.DataFrame | None:
    """Rebuild one canonical public input and verify all provenance fields."""

    dataset_name = manifest.get("dataset")
    if dataset_name not in {"REUTERS", "20NEWS", "RCV1"}:
        return None
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError("Public MvDEC manifest is missing its resolved config.")
    source_id = manifest.get("source_id")
    expected_20news_source = (
        TWENTY_NEWS_SKLEARN_SOURCE
        if dataset_name == "20NEWS"
        else TWENTY_NEWS_RELEASE_NPZ
    )
    if dataset_name == "20NEWS" and source_id != TWENTY_NEWS_SKLEARN_SOURCE:
        raise ValueError("Canonical MvDEC 20NEWS runs must use the sklearn source.")
    _validate_public_source_runtime(manifest)
    contract = config.get("protocol_contract")
    normalization = (
        contract.get("preprocessing", {}).get("input_normalization")
        if isinstance(contract, dict)
        else None
    )
    if not isinstance(normalization, str):
        raise ValueError("Public MvDEC config is missing input normalization.")
    dataset = load_dekm_public_dataset(
        str(dataset_name),
        dataset_root,
        normalization=normalization,
        twenty_news_source=expected_20news_source,
    )
    expected = {
        "source_id": dataset.source_id,
        "source_sha256": dataset.source_sha256,
        "n_samples": len(dataset.X),
        "n_features": dataset.X.shape[1],
        "n_classes": dataset.spec.n_clusters,
    }
    if any(
        manifest.get(field) != value or config.get(field) != value
        for field, value in expected.items()
    ):
        raise ValueError(
            "Public MvDEC source provenance does not match canonical input."
        )

    assignments = pd.read_csv(resolve_manifest_output_path(manifest, "assignments"))
    for field in ("source_id", "source_sha256", "training_commit"):
        if field not in assignments or assignments[field].nunique(dropna=False) != 1:
            raise ValueError(f"MvDEC assignments are missing stable {field} metadata.")
        if str(assignments[field].iloc[0]) != str(manifest.get(field)):
            raise ValueError(f"MvDEC assignments {field} does not match the manifest.")
    return assignments


def validate_public_assignments(
    manifest: dict[str, object],
    artifact: dict[str, object],
    assignments: pd.DataFrame,
) -> None:
    """Verify row alignment and independently recompute public ACC/NMI."""

    required_columns = {"sample_index", "cluster", "true_label"}
    missing = required_columns.difference(assignments.columns)
    if missing:
        raise ValueError(f"MvDEC assignments are missing columns: {sorted(missing)}")
    expected_rows = int(manifest["n_samples"])
    if len(assignments) != expected_rows:
        raise ValueError("MvDEC assignments row count does not match the manifest.")
    if not np.array_equal(assignments["sample_index"], np.arange(expected_rows)):
        raise ValueError("MvDEC assignments are not in canonical sample order.")

    labels = np.asarray(artifact["labels"])
    true_labels = np.asarray(artifact["true_labels"])
    if not np.array_equal(assignments["cluster"], labels):
        raise ValueError("MvDEC assignments do not match artifact cluster labels.")
    if not np.array_equal(assignments["true_label"], true_labels):
        raise ValueError("MvDEC assignments do not match artifact true labels.")

    metrics = compute_external_metrics(
        assignments["true_label"],
        assignments["cluster"],
        n_clusters=int(manifest["n_classes"]),
    )
    manifest_metrics = manifest.get("metrics")
    if not isinstance(manifest_metrics, dict):
        raise ValueError("Public MvDEC manifest is missing metrics.")
    recorded = (
        artifact.get("acc"),
        artifact.get("nmi"),
        manifest_metrics.get("acc"),
        manifest_metrics.get("nmi"),
    )
    expected_metrics = (metrics.acc, metrics.nmi, metrics.acc, metrics.nmi)
    if not all(
        value is not None and np.isclose(float(value), expected)
        for value, expected in zip(recorded, expected_metrics, strict=True)
    ):
        raise ValueError("MvDEC recorded ACC/NMI do not match assignments.")


def validate_training_log(
    manifest: dict[str, object],
    artifact: dict[str, object],
) -> None:
    """Verify terminal provenance and the complete refinement-loss trajectory."""

    log_path = resolve_manifest_output_path(manifest, "training_log")
    with log_path.open(newline="", encoding="utf-8") as file:
        rows = [[field.strip() for field in row] for row in csv.reader(file)]
    stop_rows = [row for row in rows if "phase:training_stop" in row]
    final_rows = [
        row
        for row in rows
        if any(field.startswith("phase:final_recomputed_") for field in row)
    ]
    step9_rows = [
        row
        for row in rows
        if any(field.startswith("phase:paper_step9_after_") for field in row)
    ]
    expected_windows = int(artifact.get("refinement_epochs_completed", -1))
    if len(stop_rows) != 1 or len(final_rows) != 1:
        raise ValueError("MvDEC training log is missing unique terminal phases.")
    if expected_windows < 1 or len(step9_rows) != expected_windows:
        raise ValueError("MvDEC training log has an incomplete refinement trajectory.")

    summary_rows = [row for row in rows if row and row[0].startswith("run ")]
    if len(summary_rows) != 1:
        raise ValueError("MvDEC training log is missing its unique run summary.")
    summary = ";".join(summary_rows[0])
    expected_summary_fields = (
        f"seed:{manifest.get('seed')}",
        f"run_id:{manifest.get('run_id')}",
        f"protocol_id:{manifest.get('protocol_id')}",
    )
    if any(field not in summary for field in expected_summary_fields):
        raise ValueError("MvDEC training-log summary provenance does not match.")

    stop = ";".join(stop_rows[0])
    stop_fields = (
        f"training_steps_completed:{artifact.get('training_steps_completed')}",
        f"refinement_epochs_completed:{expected_windows}",
        f"refinement_progress_unit:{artifact.get('refinement_progress_unit')}",
        f"stop_reason:{artifact.get('stop_reason')}",
    )
    if any(field not in stop for field in stop_fields):
        raise ValueError("MvDEC training-log stopping metadata does not match.")

    loss_weights = artifact.get("config", {}).get("loss_weights", {})
    if loss_weights.get("cross_view_alignment", 0) > 0 and any(
        not any("L5_alignment:" in field for field in row) for row in step9_rows
    ):
        raise ValueError("MvDEC training log is missing its L5 trajectory.")


def audit_run_root(
    root: Path,
    dataset_root: Path = DEKM_DATASET_DIR,
) -> tuple[int, int]:
    """Validate all complete runs and return completed/failed manifest counts."""

    completed = 0
    failed = 0
    manifest_paths = sorted(root.rglob(RUN_MANIFEST_FILENAME))
    if not manifest_paths:
        raise ValueError(f"No MvDEC run manifests found below: {root}")
    for manifest_path in manifest_paths:
        manifest = load_run_manifest(manifest_path)
        status = manifest.get("status")
        if status == "failed":
            failed += 1
            logger.warning("Failed run retained for audit: {}", manifest["run_id"])
            continue
        if status != "complete":
            raise ValueError(
                f"MvDEC run is neither complete nor failed: {manifest_path}"
            )
        validate_manifest_outputs(manifest)
        assignments = validate_public_source(manifest, dataset_root)
        artifact_path = resolve_manifest_output_path(manifest, "artifact")
        if assignments is None:
            load_mvdec_result(
                result_path=artifact_path,
                data_path=artifact_data_path(manifest),
            )
        else:
            artifact = validate_mvdec_artifact(artifact_path)
            validate_public_assignments(manifest, artifact, assignments)
            validate_training_log(manifest, artifact)
        completed += 1
        logger.info("Validated MvDEC run: {}", manifest["run_id"])
    return completed, failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("output/mvdec_runs"),
        help="Root containing dataset/protocol/config/seed run directories.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEKM_DATASET_DIR,
        help="Root containing canonical DEKM public dataset inputs.",
    )
    args = parser.parse_args()
    completed, failed = audit_run_root(args.root, args.dataset_root)
    logger.info("Audit complete: {} complete, {} failed", completed, failed)


if __name__ == "__main__":
    main()
