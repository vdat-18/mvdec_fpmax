"""Audit completed MvDEC run manifests and reload every saved artifact."""

import argparse
from pathlib import Path

import pandas as pd
from loguru import logger

from config import DEKM_DATASET_DIR
from pipeline.data import load_mvdec_result
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


def validate_public_source(
    manifest: dict[str, object],
    dataset_root: Path,
) -> None:
    """Rebuild one canonical public input and verify all provenance fields."""

    dataset_name = manifest.get("dataset")
    if dataset_name not in {"REUTERS", "20NEWS", "RCV1"}:
        return
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
        validate_public_source(manifest, dataset_root)
        load_mvdec_result(
            result_path=resolve_manifest_output_path(manifest, "artifact"),
            data_path=artifact_data_path(manifest),
        )
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
