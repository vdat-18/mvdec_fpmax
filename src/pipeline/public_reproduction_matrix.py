"""Run and package a reproducible public-dataset MvDEC experiment matrix."""

import argparse
import csv
import hashlib
import os
import subprocess
import sys
import tarfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from config import DEKM_DATASET_DIR, PROJECT_DIR, PUBLIC_BENCHMARK_OUTPUT_DIR
from pipeline.io import replace_with_retry
from pipeline.mvdec_runs import (
    RUN_MANIFEST_FILENAME,
    load_run_manifest,
    validate_manifest_outputs,
)

SUMMARY_FILENAME = "public_mvdec_matrix_runs.csv"
SUMMARY_FIELDS = (
    "dataset",
    "protocol_id",
    "seed",
    "run_id",
    "config_hash",
    "acc",
    "nmi",
    "runtime_seconds",
    "archive_path",
    "archive_sha256",
)


@dataclass(frozen=True)
class MatrixRunSpec:
    """Identify one public dataset, protocol, and seed combination."""

    dataset: str
    protocol_id: str
    seed: int


@dataclass(frozen=True)
class CompletedMatrixRun:
    """Describe one audited and packaged matrix run."""

    spec: MatrixRunSpec
    archive_path: Path
    checksum_path: Path
    skipped: bool


def build_run_specs(
    datasets: Sequence[str],
    protocols: Sequence[str],
    seeds: Sequence[int],
) -> tuple[MatrixRunSpec, ...]:
    """Build the ordered Cartesian product requested by the caller."""

    if not datasets or not protocols or not seeds:
        raise ValueError("Datasets, protocols, and seeds must all be non-empty.")
    return tuple(
        MatrixRunSpec(dataset.upper(), protocol, int(seed))
        for dataset in datasets
        for protocol in protocols
        for seed in seeds
    )


def file_sha256(path: Path) -> str:
    """Return the raw SHA-256 digest for one archive."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_paths(archive_root: Path, spec: MatrixRunSpec) -> tuple[Path, Path]:
    """Resolve stable archive and checksum paths for one matrix run."""

    stem = f"{spec.dataset.lower()}__{spec.protocol_id.lower()}__seed_{spec.seed}"
    archive_path = archive_root / f"{stem}.tar.gz"
    return archive_path, archive_path.with_suffix(f"{archive_path.suffix}.sha256")


def checksum_matches(archive_path: Path, checksum_path: Path) -> bool:
    """Return whether an existing archive matches its checksum file."""

    if not archive_path.is_file() or not checksum_path.is_file():
        return False
    fields = checksum_path.read_text(encoding="utf-8").split()
    if len(fields) != 2 or fields[1] != archive_path.name:
        return False
    return fields[0].lower() == file_sha256(archive_path)


def matching_complete_manifests(
    output_root: Path,
    spec: MatrixRunSpec,
) -> list[dict[str, object]]:
    """Find complete manifests matching one matrix run identity."""

    protocol_root = output_root / spec.dataset.lower() / spec.protocol_id.lower()
    manifests: list[dict[str, object]] = []
    for path in sorted(protocol_root.rglob(RUN_MANIFEST_FILENAME)):
        manifest = load_run_manifest(path)
        if (
            manifest.get("dataset") == spec.dataset
            and manifest.get("protocol_id") == spec.protocol_id
            and manifest.get("seed") == spec.seed
            and manifest.get("status") == "complete"
        ):
            manifests.append(manifest)
    return manifests


def require_single_complete_manifest(
    output_root: Path,
    spec: MatrixRunSpec,
) -> dict[str, object]:
    """Return the unique complete manifest produced for one specification."""

    manifests = matching_complete_manifests(output_root, spec)
    if len(manifests) != 1:
        raise ValueError(
            "Expected exactly one complete manifest for "
            f"{spec.dataset}/{spec.protocol_id}/seed_{spec.seed}; "
            f"found {len(manifests)}."
        )
    return manifests[0]


def run_training(
    spec: MatrixRunSpec,
    *,
    dataset_root: Path,
    output_root: Path,
    progress_interval: int,
) -> None:
    """Execute one public MvDEC training run with deterministic seed metadata."""

    training_script = PROJECT_DIR / "src" / "representation_learning" / "MVDEC_dense.py"
    command = [
        sys.executable,
        "-u",
        str(training_script),
        spec.dataset,
        "--runs",
        "1",
        "--seed",
        str(spec.seed),
        "--protocol",
        spec.protocol_id,
        "--progress-interval",
        str(progress_interval),
        "--dataset-root",
        str(dataset_root),
        "--public-output-dir",
        str(output_root),
        "--force",
    ]
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = str(spec.seed)
    logger.info(
        "Starting public MvDEC run: dataset={}, protocol={}, seed={}",
        spec.dataset,
        spec.protocol_id,
        spec.seed,
    )
    subprocess.run(
        command,
        cwd=PROJECT_DIR,
        env=environment,
        check=True,
    )


def audit_run(run_dir: Path) -> None:
    """Run the repository audit command against one completed run directory."""

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pipeline.mvdec_audit",
            "--root",
            str(run_dir),
        ],
        cwd=PROJECT_DIR,
        check=True,
    )


def package_run(
    manifest: dict[str, object],
    archive_path: Path,
    checksum_path: Path,
) -> str:
    """Create an atomic tar archive and checksum for one audited run."""

    manifest_path_value = manifest.get("_manifest_path")
    if not isinstance(manifest_path_value, str):
        raise ValueError("Loaded manifest is missing its source path.")
    run_dir = Path(manifest_path_value).parent.resolve()
    try:
        archive_name = run_dir.relative_to(PROJECT_DIR.resolve())
    except ValueError as error:
        raise ValueError(
            f"MvDEC run directory must be below the project root: {run_dir}"
        ) from error

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_archive = archive_path.with_name(f"{archive_path.name}.tmp")
    if temporary_archive.exists():
        temporary_archive.unlink()
    with tarfile.open(temporary_archive, "w:gz") as archive:
        archive.add(run_dir, arcname=archive_name)
    replace_with_retry(temporary_archive, archive_path)

    digest = file_sha256(archive_path)
    temporary_checksum = checksum_path.with_name(f"{checksum_path.name}.tmp")
    temporary_checksum.write_text(
        f"{digest}  {archive_path.name}\n",
        encoding="utf-8",
    )
    replace_with_retry(temporary_checksum, checksum_path)
    return digest


def summary_record(
    manifest: dict[str, object],
    archive_path: Path,
    archive_sha256: str,
) -> dict[str, object]:
    """Build the final-metric summary row for one packaged run."""

    metrics = manifest.get("metrics")
    if not isinstance(metrics, dict) or "acc" not in metrics or "nmi" not in metrics:
        raise ValueError("Public MvDEC manifest must contain final ACC and NMI.")
    return {
        "dataset": manifest["dataset"],
        "protocol_id": manifest["protocol_id"],
        "seed": manifest["seed"],
        "run_id": manifest["run_id"],
        "config_hash": manifest["config_hash"],
        "acc": metrics["acc"],
        "nmi": metrics["nmi"],
        "runtime_seconds": manifest["runtime_seconds"],
        "archive_path": str(archive_path.resolve()),
        "archive_sha256": archive_sha256,
    }


def update_summary(summary_path: Path, record: dict[str, object]) -> None:
    """Atomically upsert one dataset/protocol/seed row in the matrix summary."""

    records: list[dict[str, object]] = []
    if summary_path.is_file():
        with summary_path.open(newline="", encoding="utf-8") as file:
            records.extend(csv.DictReader(file))
    identity = (
        str(record["dataset"]),
        str(record["protocol_id"]),
        str(record["seed"]),
    )
    records = [
        existing
        for existing in records
        if (
            str(existing["dataset"]),
            str(existing["protocol_id"]),
            str(existing["seed"]),
        )
        != identity
    ]
    records.append(record)
    records.sort(
        key=lambda item: (
            str(item["dataset"]),
            str(item["protocol_id"]),
            int(item["seed"]),
        )
    )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = summary_path.with_name(f"{summary_path.name}.tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    replace_with_retry(temporary_path, summary_path)


def run_public_matrix(
    *,
    datasets: Sequence[str],
    protocols: Sequence[str],
    seeds: Sequence[int],
    dataset_root: Path = DEKM_DATASET_DIR,
    output_root: Path = PUBLIC_BENCHMARK_OUTPUT_DIR / "mvdec",
    archive_root: Path,
    progress_interval: int = 100,
    resume: bool = False,
) -> Iterator[CompletedMatrixRun]:
    """Run, audit, package, and yield each requested matrix combination."""

    if progress_interval <= 0:
        raise ValueError("progress_interval must be positive.")
    specs = build_run_specs(datasets, protocols, seeds)
    summary_path = archive_root / SUMMARY_FILENAME
    for index, spec in enumerate(specs, start=1):
        archive_path, checksum_path = archive_paths(archive_root, spec)
        if resume and checksum_matches(archive_path, checksum_path):
            logger.info(
                "Skipping packaged run {}/{}: dataset={}, protocol={}, seed={}",
                index,
                len(specs),
                spec.dataset,
                spec.protocol_id,
                spec.seed,
            )
            yield CompletedMatrixRun(spec, archive_path, checksum_path, True)
            continue

        manifests = matching_complete_manifests(output_root, spec)
        if len(manifests) > 1:
            require_single_complete_manifest(output_root, spec)
        if manifests:
            manifest = manifests[0]
        else:
            run_training(
                spec,
                dataset_root=dataset_root,
                output_root=output_root,
                progress_interval=progress_interval,
            )
            manifest = require_single_complete_manifest(output_root, spec)

        validate_manifest_outputs(manifest)
        manifest_path = Path(str(manifest["_manifest_path"]))
        audit_run(manifest_path.parent)
        archive_sha256 = package_run(manifest, archive_path, checksum_path)
        update_summary(
            summary_path,
            summary_record(manifest, archive_path, archive_sha256),
        )
        logger.info(
            "Packaged run {}/{}: {}",
            index,
            len(specs),
            archive_path,
        )
        yield CompletedMatrixRun(spec, archive_path, checksum_path, False)


def main() -> None:
    """Run the public reproduction matrix from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--protocols", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEKM_DATASET_DIR)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PUBLIC_BENCHMARK_OUTPUT_DIR / "mvdec",
    )
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    for completed in run_public_matrix(
        datasets=args.datasets,
        protocols=args.protocols,
        seeds=args.seeds,
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        archive_root=args.archive_root,
        progress_interval=args.progress_interval,
        resume=args.resume,
    ):
        logger.info("Archive ready: {}", completed.archive_path)
        logger.info("Checksum ready: {}", completed.checksum_path)


if __name__ == "__main__":
    main()
