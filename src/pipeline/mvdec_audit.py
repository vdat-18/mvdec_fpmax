"""Audit completed MvDEC run manifests and reload every saved artifact."""

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn import __version__ as sklearn_version

from config import DEKM_DATASET_DIR
from pipeline.data import load_mvdec_result, validate_mvdec_artifact
from pipeline.external_metrics import compute_external_metrics
from pipeline.mvdec_bundle_audit import validate_archive_bundle
from pipeline.mvdec_runs import (
    RUN_MANIFEST_FILENAME,
    load_run_manifest,
    resolve_manifest_output_path,
    validate_manifest_identity,
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


def validate_numerical_runtime(manifest: dict[str, object]) -> None:
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
    validate_numerical_runtime(manifest)
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
    window_numbers = []
    for row in step9_rows:
        phase = next(
            field for field in row if field.startswith("phase:paper_step9_after_")
        )
        match = re.fullmatch(
            r"phase:paper_step9_after_(?:release_update_window|full_epoch)_(\d+)",
            phase,
        )
        if match is None:
            raise ValueError("MvDEC training log has an invalid refinement phase.")
        window = int(match.group(1))
        if f"refinement_epoch:{window}" not in row:
            raise ValueError("MvDEC training log refinement metadata is inconsistent.")
        window_numbers.append(window)
    if window_numbers != list(range(1, expected_windows + 1)):
        raise ValueError("MvDEC training log refinement windows are not contiguous.")

    required_loss_names = (
        "total",
        "L1_reconstruction",
        "L2_kmeans",
        "L3_trace_logged",
        "L4_greedy",
    )
    config = artifact.get("config")
    loss_weights = config.get("loss_weights") if isinstance(config, dict) else None
    if not isinstance(loss_weights, dict):
        raise ValueError("MvDEC training log is missing objective weights.")
    objective_weights = {
        "L1_reconstruction": loss_weights.get("reconstruction"),
        "L2_kmeans": loss_weights.get("kmeans"),
        "L3_trace_logged": loss_weights.get("scatter_trace_diagnostic"),
        "L4_greedy": loss_weights.get("greedy"),
        "L5_alignment": loss_weights.get("cross_view_alignment", 0.0),
    }
    if any(
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not np.isfinite(weight)
        or weight < 0
        for weight in objective_weights.values()
    ):
        raise ValueError("MvDEC training log has invalid objective weights.")
    if objective_weights["L5_alignment"] > 0:
        required_loss_names += ("L5_alignment",)

    for row in step9_rows:
        encoded_row = ",".join(row)
        parsed_losses = {}
        for loss_name in required_loss_names:
            pattern = rf"(?:^|[,\s])(?:loss:)?{loss_name}:([^,\s]+)"
            matches = re.findall(pattern, encoded_row)
            if len(matches) != 1:
                raise ValueError(
                    f"MvDEC training log is missing unique {loss_name} values."
                )
            if not np.isfinite(float(matches[0])):
                raise ValueError(
                    f"MvDEC training log has a non-finite {loss_name} value."
                )
            parsed_losses[loss_name] = float(matches[0])

        expected_total = sum(
            objective_weights[name] * parsed_losses.get(name, 0.0)
            for name in objective_weights
        )
        # Values come from float32 TensorFlow losses and are then rounded to
        # five decimals. Bound both sources of representational error.
        rounding_tolerance = 5e-6 * (
            1.0 + sum(abs(weight) for weight in objective_weights.values())
        )
        weighted_magnitude = sum(
            abs(objective_weights[name] * parsed_losses.get(name, 0.0))
            for name in objective_weights
        )
        float32_tolerance = (
            8.0
            * np.finfo(np.float32).eps
            * max(weighted_magnitude, abs(parsed_losses["total"]), 1.0)
        )
        if not np.isclose(
            parsed_losses["total"],
            expected_total,
            rtol=0.0,
            atol=rounding_tolerance + float32_tolerance,
        ):
            raise ValueError(
                "MvDEC logged total loss does not match its weighted components."
            )

    if config.get("refinement_batch_size_policy") == "full_dataset":
        update_interval = int(config.get("update_interval", 0))
        n_samples = int(artifact.get("n_samples", 0))
        if update_interval < 1 or n_samples < 1:
            raise ValueError("MvDEC full-batch audit metadata is incomplete.")
        expected_batch_field = f"step9_loss_batches:{update_interval}"
        expected_sample_field = f"step9_loss_samples:{update_interval * n_samples}"
        if any(
            expected_batch_field not in row or expected_sample_field not in row
            for row in step9_rows
        ):
            raise ValueError(
                "MvDEC training log does not prove full-dataset refinement batches."
            )

    summary_rows = [row for row in rows if row and row[0].startswith("run ")]
    if len(summary_rows) != 1:
        raise ValueError("MvDEC training log is missing its unique run summary.")
    summary_fields = set(summary_rows[0])
    expected_summary_fields = (
        f"seed:{manifest.get('seed')}",
        f"run_id:{manifest.get('run_id')}",
        f"protocol_id:{manifest.get('protocol_id')}",
    )
    if any(field not in summary_fields for field in expected_summary_fields):
        raise ValueError("MvDEC training-log summary provenance does not match.")

    stop_fields_found = set(stop_rows[0])
    stop_fields = (
        f"training_steps_completed:{artifact.get('training_steps_completed')}",
        f"refinement_epochs_completed:{expected_windows}",
        f"refinement_progress_unit:{artifact.get('refinement_progress_unit')}",
        f"stop_reason:{artifact.get('stop_reason')}",
    )
    if any(field not in stop_fields_found for field in stop_fields):
        raise ValueError("MvDEC training-log stopping metadata does not match.")

    if loss_weights.get("cross_view_alignment", 0) > 0:
        l5_values = []
        for row in step9_rows:
            matches = re.findall(r"L5_alignment:([^,\s]+)", ",".join(row))
            if len(matches) != 1:
                raise ValueError("MvDEC training log is missing its L5 trajectory.")
            value = float(matches[0])
            if not np.isfinite(value):
                raise ValueError("MvDEC training log has a non-finite L5 value.")
            l5_values.append(value)
        if len(l5_values) != expected_windows:
            raise ValueError("MvDEC training log has an incomplete L5 trajectory.")


def validate_expected_run_set(
    manifests: list[dict[str, object]],
    expected_datasets: tuple[str, ...] | None = None,
    expected_protocol_id: str | None = None,
    expected_training_commit: str | None = None,
    expected_seed: int | None = None,
    require_all_complete: bool = False,
) -> None:
    """Fail closed when a downloaded run set differs from the requested gate."""

    if expected_datasets is not None:
        started = []
        for manifest in manifests:
            started_at = manifest.get("started_at_utc")
            dataset = manifest.get("dataset")
            if not isinstance(started_at, str) or not isinstance(dataset, str):
                raise ValueError("Expected MvDEC run set requires start timestamps.")
            started.append((started_at, dataset))
        discovered = tuple(dataset for _, dataset in sorted(started))
        if discovered != expected_datasets:
            raise ValueError(
                "MvDEC dataset execution order does not match the expected gate."
            )
    if require_all_complete and any(
        manifest.get("status") != "complete" for manifest in manifests
    ):
        raise ValueError("Expected MvDEC run set contains an incomplete run.")
    checks = (
        ("protocol_id", expected_protocol_id),
        ("training_commit", expected_training_commit),
        ("seed", expected_seed),
    )
    for field, expected in checks:
        if expected is not None and any(
            manifest.get(field) != expected for manifest in manifests
        ):
            raise ValueError(f"MvDEC run set has unexpected {field}.")


def audit_run_root(
    root: Path,
    dataset_root: Path = DEKM_DATASET_DIR,
    expected_datasets: tuple[str, ...] | None = None,
    expected_protocol_id: str | None = None,
    expected_training_commit: str | None = None,
    expected_seed: int | None = None,
    require_all_complete: bool = False,
) -> tuple[int, int]:
    """Validate all complete runs and return completed/failed manifest counts."""

    completed = 0
    failed = 0
    manifest_paths = sorted(root.rglob(RUN_MANIFEST_FILENAME))
    if not manifest_paths:
        raise ValueError(f"No MvDEC run manifests found below: {root}")
    manifests = [load_run_manifest(path) for path in manifest_paths]
    for manifest in manifests:
        validate_manifest_identity(manifest)
    validate_expected_run_set(
        manifests,
        expected_datasets=expected_datasets,
        expected_protocol_id=expected_protocol_id,
        expected_training_commit=expected_training_commit,
        expected_seed=expected_seed,
        require_all_complete=require_all_complete,
    )
    for manifest in manifests:
        status = manifest.get("status")
        if status == "failed":
            failed += 1
            logger.warning("Failed run retained for audit: {}", manifest["run_id"])
            continue
        if status != "complete":
            manifest_path = manifest.get("_manifest_path")
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
    parser.add_argument("--expected-datasets", nargs="+")
    parser.add_argument("--expected-protocol-id")
    parser.add_argument("--expected-training-commit")
    parser.add_argument("--expected-seed", type=int)
    parser.add_argument(
        "--require-all-complete",
        action="store_true",
        help="Fail when any discovered manifest is not complete.",
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        help="Optional directory containing matrix summaries and run archives.",
    )
    parser.add_argument(
        "--combined-zip",
        type=Path,
        help="Optional combined ZIP to verify against --archive-root.",
    )
    args = parser.parse_args()
    completed, failed = audit_run_root(
        args.root,
        args.dataset_root,
        expected_datasets=(
            tuple(args.expected_datasets) if args.expected_datasets else None
        ),
        expected_protocol_id=args.expected_protocol_id,
        expected_training_commit=args.expected_training_commit,
        expected_seed=args.expected_seed,
        require_all_complete=args.require_all_complete,
    )
    if args.combined_zip is not None and args.archive_root is None:
        parser.error("--combined-zip requires --archive-root")
    if args.archive_root is not None:
        manifest_paths = sorted(args.root.rglob(RUN_MANIFEST_FILENAME))
        manifests = [load_run_manifest(path) for path in manifest_paths]
        validate_archive_bundle(args.archive_root, manifests, args.combined_zip)
        logger.info("Archive bundle audit complete: {}", args.archive_root)
    logger.info("Audit complete: {} complete, {} failed", completed, failed)


if __name__ == "__main__":
    main()
