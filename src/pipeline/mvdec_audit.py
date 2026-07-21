"""Audit completed MvDEC run manifests and reload every saved artifact."""

import argparse
from pathlib import Path

from loguru import logger

from pipeline.data import load_mvdec_result
from pipeline.mvdec_runs import (
    RUN_MANIFEST_FILENAME,
    load_run_manifest,
    resolve_manifest_output_path,
    validate_manifest_outputs,
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


def audit_run_root(root: Path) -> tuple[int, int]:
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
    args = parser.parse_args()
    completed, failed = audit_run_root(args.root)
    logger.info("Audit complete: {} complete, {} failed", completed, failed)


if __name__ == "__main__":
    main()
