"""Create immutable MvDEC run directories and authoritative manifests."""

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pipeline.io import replace_with_retry

RUN_MANIFEST_SCHEMA_VERSION = 1
RUN_MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True)
class MvdecRunPaths:
    """Resolved output paths owned by one dataset/config/seed run."""

    run_id: str
    config_hash: str
    config_dir: Path
    run_dir: Path
    manifest: Path
    artifact: Path
    assignments: Path
    pretrain_view1: Path
    pretrain_view2: Path
    final_view1: Path
    final_view2: Path
    training_log: Path


def file_sha256(path: Path) -> str:
    """Hash one run output using the repository's text normalization contract."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        if path.suffix.lower() in {".csv", ".tsv", ".txt"}:
            for line in file:
                digest.update(line.replace(b"\r\n", b"\n"))
        else:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def canonical_config_hash(config: dict[str, object]) -> str:
    """Return a stable SHA-256 for one resolved seed-independent config."""

    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_run_id(
    dataset: str,
    protocol_id: str,
    config_hash: str,
    seed: int | None,
    run_index: int,
) -> str:
    """Return a human-readable identity containing protocol, config, and seed."""

    seed_part = f"seed_{seed}" if seed is not None else f"run_{run_index:02d}"
    return f"{dataset.lower()}_{protocol_id.lower()}_{config_hash[:12]}_{seed_part}"


def resolve_run_paths(
    output_root: Path,
    dataset: str,
    protocol_id: str,
    config: dict[str, object],
    seed: int | None,
    run_index: int,
) -> MvdecRunPaths:
    """Resolve the isolated directory and files for one MvDEC run."""

    config_hash = canonical_config_hash(config)
    config_dir = output_root / dataset.lower() / protocol_id.lower() / config_hash[:12]
    run_id = build_run_id(dataset, protocol_id, config_hash, seed, run_index)
    seed_dir = f"seed_{seed}" if seed is not None else f"run_{run_index:02d}"
    run_dir = config_dir / seed_dir
    return MvdecRunPaths(
        run_id=run_id,
        config_hash=config_hash,
        config_dir=config_dir,
        run_dir=run_dir,
        manifest=run_dir / RUN_MANIFEST_FILENAME,
        artifact=run_dir / "artifact.pkl",
        assignments=run_dir / "assignments.csv",
        pretrain_view1=run_dir / "pretrain_view1.weights.h5",
        pretrain_view2=run_dir / "pretrain_view2.weights.h5",
        final_view1=run_dir / "final_view1.weights.h5",
        final_view2=run_dir / "final_view2.weights.h5",
        training_log=run_dir / "training_log.csv",
    )


def prepare_run_directory(paths: MvdecRunPaths, force: bool = False) -> None:
    """Create a run directory and reject accidental reuse unless forced."""

    if paths.run_dir.exists() and any(paths.run_dir.iterdir()) and not force:
        raise FileExistsError(
            f"MvDEC run already exists: {paths.run_dir}. Use --force to overwrite "
            "the files owned by this exact run_id."
        )
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    if force:
        for owned_path in (
            paths.manifest,
            paths.artifact,
            paths.assignments,
            paths.pretrain_view1,
            paths.pretrain_view2,
            paths.final_view1,
            paths.final_view2,
            paths.training_log,
        ):
            if owned_path.exists():
                owned_path.unlink()


def write_run_manifest(path: Path, payload: dict[str, object]) -> None:
    """Atomically persist one versioned run manifest."""

    document = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        **payload,
    }
    temporary_path = path.with_name(f"{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(document, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    replace_with_retry(temporary_path, path)


def append_run_log(path: Path, fields: list[object]) -> None:
    """Append one structured CSV row to a run-owned training log."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(fields)


def load_run_manifest(path: Path) -> dict[str, object]:
    """Load and validate one versioned run manifest and its config hash."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid MvDEC run manifest: {path}") from error
    if payload.get("schema_version") != RUN_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"Unsupported MvDEC run manifest schema: {path}")
    config = payload.get("config")
    if not isinstance(config, dict) or payload.get(
        "config_hash"
    ) != canonical_config_hash(config):
        raise ValueError(f"MvDEC run manifest config hash does not match: {path}")
    payload["_manifest_path"] = str(path.resolve())
    return payload


def resolve_manifest_output_path(
    manifest: dict[str, object],
    name: str,
) -> Path:
    """Resolve one absolute or manifest-relative output path."""

    paths = manifest.get("paths")
    output_path = paths.get(name) if isinstance(paths, dict) else None
    if not isinstance(output_path, str):
        raise ValueError(f"MvDEC manifest is missing output path {name!r}.")
    path = Path(output_path)
    if path.is_absolute():
        return path
    manifest_path = manifest.get("_manifest_path")
    if not isinstance(manifest_path, str):
        raise ValueError("MvDEC manifest-relative path requires manifest location.")
    return Path(manifest_path).parent / path


def validate_manifest_outputs(manifest: dict[str, object]) -> None:
    """Verify every output path and SHA-256 declared by a complete manifest."""

    if manifest.get("status") != "complete":
        raise ValueError("Only complete MvDEC manifests have auditable outputs.")
    paths = manifest.get("paths")
    output_hashes = manifest.get("output_sha256")
    if not isinstance(paths, dict) or not isinstance(output_hashes, dict):
        raise ValueError("Complete MvDEC manifest is missing output paths or hashes.")
    if not paths or paths.keys() != output_hashes.keys():
        raise ValueError("Complete MvDEC manifest must hash every declared output.")
    for name, expected_hash in output_hashes.items():
        if not isinstance(expected_hash, str):
            raise ValueError(f"Invalid MvDEC output metadata for {name!r}.")
        path = resolve_manifest_output_path(manifest, name)
        if not path.is_file() or file_sha256(path) != expected_hash:
            raise ValueError(f"MvDEC output hash does not match for {name!r}: {path}")


def load_run_manifests(config_dir: Path) -> list[dict[str, object]]:
    """Load every valid manifest below one immutable config directory."""

    manifests = []
    for path in sorted(config_dir.glob(f"*/{RUN_MANIFEST_FILENAME}")):
        manifests.append(load_run_manifest(path))
    return manifests


def complete_run_manifests(config_dir: Path) -> list[dict[str, object]]:
    """Return only completed manifests for authoritative metric summaries."""

    return [
        manifest
        for manifest in load_run_manifests(config_dir)
        if manifest.get("status") == "complete"
    ]


def build_summary_frames(
    manifests: list[dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build run and aggregate frames using completed manifests only."""

    complete = [
        manifest for manifest in manifests if manifest.get("status") == "complete"
    ]
    if not complete:
        raise ValueError("At least one complete MvDEC manifest is required.")
    config_hashes = {manifest.get("config_hash") for manifest in manifests}
    if len(config_hashes) != 1:
        raise ValueError("MvDEC summary manifests must share one config_hash.")

    run_records = [
        {
            "run_id": manifest["run_id"],
            "dataset": manifest["dataset"],
            "method": manifest["method"],
            "protocol_id": manifest["protocol_id"],
            "config_hash": manifest["config_hash"],
            "random_seed": manifest["seed"],
            "run": manifest["run_index"],
            "runtime_seconds": manifest["runtime_seconds"],
            "artifact_path": str(resolve_manifest_output_path(manifest, "artifact")),
            "assignments_path": str(
                resolve_manifest_output_path(manifest, "assignments")
            ),
            **manifest["metrics"],
        }
        for manifest in complete
    ]
    runs_frame = pd.DataFrame.from_records(run_records)
    summary_record = {
        "dataset": complete[0]["dataset"],
        "method": complete[0]["method"],
        "protocol_id": complete[0]["protocol_id"],
        "config_hash": complete[0]["config_hash"],
        "discovered_runs": len(manifests),
        "completed_runs": len(complete),
        "failed_runs": sum(
            manifest.get("status") == "failed" for manifest in manifests
        ),
    }
    for metric_name in ("silhouette", "acc", "nmi"):
        if metric_name in runs_frame:
            summary_record[f"{metric_name}_mean"] = runs_frame[metric_name].mean()
            summary_record[f"{metric_name}_std"] = (
                runs_frame[metric_name].std(ddof=1) if len(runs_frame) > 1 else 0.0
            )
    return runs_frame, pd.DataFrame([summary_record])
