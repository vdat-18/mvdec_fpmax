"""Validate the single verified DEKM_F checkpoint used by the MvDEC diagnostic."""

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

from pipeline.mvdec_runs import HDF5_SIGNATURE, SHA256_PATTERN, file_sha256

DEKM_F_METHOD = "DEKM_F"
DEKM_F_PROTOCOL_ID = "dekm_f_2021_official_source_full_batch_v1"
DEKM_F_CONFIG_HASH = "16ec44629a68de5aa993527037fa357bb2802cad3ef1e087a9e7ee7b26f57091"
DEKM_F_REUTERS_SOURCE_SHA256 = (
    "427993db6b0544dfce1b8e3985e63e248d4d006de122cb202c504489d951706b"
)
DEKM_F_REUTERS_ACC = 0.7557
DEKM_F_REUTERS_NMI = 0.593243233623544
DEKM_F_WEIGHT_SHA256 = (
    "ca55a1e995697c346d99540ddd9cc8bd8e6182a7d21d0b09410d7e76b6050702"
)
DEKM_F_MANIFEST_SHA256 = (
    "71e3c2d423d70bbde1b7423c91c176c1d33a4fb473e7cb6febec187fff8f0607"
)
RELEASE_L2_NORMALIZATION = "release_l2_unit_norm"
MAX_MANIFEST_BYTES = 1024 * 1024
CHECKPOINT_PROVENANCE_SHA_FIELDS = (
    "config_hash",
    "source_sha256",
    "manifest_sha256",
    "weight_sha256",
)
_EXPECTED_MANIFEST_KEYS = {
    "config",
    "config_hash",
    "dataset",
    "fit_time_seconds",
    "method",
    "metrics",
    "n_clusters",
    "n_features",
    "n_samples",
    "official_dense_sha256",
    "official_repository",
    "official_source_revision",
    "output_sha256",
    "paths",
    "protocol_id",
    "refreshes_completed",
    "runtime",
    "schema_version",
    "seed",
    "source_sha256",
    "status",
    "steps_completed",
    "stop_reason",
}
_EXPECTED_OUTPUT_KEYS = {
    "artifact",
    "assignments",
    "final_weights",
    "pretrain_weights",
    "training_log",
}

_EXPECTED_CONFIG = {
    "assignment_change_tolerance": 0.005,
    "data_shuffle": "release_random_0_99",
    "deterministic_runtime": False,
    "eigen_solver": "general_eig",
    "initial_kmeans_n_init": 100,
    "kernel_initializer": "uniform",
    "kmeans_random_state": "global",
    "latent_dim": 10,
    "max_refinement_steps": 14_000,
    "pretrain_batch_size": 256,
    "pretrain_epochs": 200,
    "protocol_id": DEKM_F_PROTOCOL_ID,
    "refinement_batch_size": 10_000,
    "update_interval": 10,
}


@dataclass(frozen=True)
class ValidatedDekmCheckpoint:
    """Validated local checkpoint plus machine-independent run provenance."""

    manifest_path: Path
    weights_path: Path
    manifest_sha256: str
    weights_sha256: str
    config_hash: str
    dataset: str
    seed: int
    source_sha256: str

    def provenance(self) -> dict[str, object]:
        """Return stable checkpoint identity without retaining a local path."""

        return {
            "method": DEKM_F_METHOD,
            "protocol_id": DEKM_F_PROTOCOL_ID,
            "config_hash": self.config_hash,
            "dataset": self.dataset,
            "seed": self.seed,
            "source_sha256": self.source_sha256,
            "manifest_sha256": self.manifest_sha256,
            "weight_sha256": self.weights_sha256,
            "input_normalization": RELEASE_L2_NORMALIZATION,
            "refinement_batching": "full_dataset_each_update",
            "claim_scope": "controlled_diagnostic_not_exact_paper_truth",
        }


def validate_dekm_f_checkpoint_provenance(
    provenance: object,
    *,
    expected_source_sha256: str,
) -> dict[str, object]:
    """Validate path-free provenance for the authorized checkpoint."""

    if expected_source_sha256 != DEKM_F_REUTERS_SOURCE_SHA256:
        raise ValueError("DEKM_F checkpoint provenance has an invalid source.")
    expected = {
        "method": DEKM_F_METHOD,
        "protocol_id": DEKM_F_PROTOCOL_ID,
        "config_hash": DEKM_F_CONFIG_HASH,
        "dataset": "REUTERS",
        "seed": 42,
        "source_sha256": DEKM_F_REUTERS_SOURCE_SHA256,
        "manifest_sha256": DEKM_F_MANIFEST_SHA256,
        "weight_sha256": DEKM_F_WEIGHT_SHA256,
        "input_normalization": RELEASE_L2_NORMALIZATION,
        "refinement_batching": "full_dataset_each_update",
        "claim_scope": "controlled_diagnostic_not_exact_paper_truth",
    }
    expected_keys = set(expected)
    if not isinstance(provenance, dict) or set(provenance) != expected_keys:
        raise ValueError("DEKM_F checkpoint provenance has an invalid schema.")
    if any(provenance.get(field) != value for field, value in expected.items()):
        raise ValueError("DEKM_F checkpoint provenance has an invalid identity.")
    if any(
        SHA256_PATTERN.fullmatch(str(provenance.get(field, ""))) is None
        for field in CHECKPOINT_PROVENANCE_SHA_FIELDS
    ):
        raise ValueError("DEKM_F checkpoint provenance has an invalid SHA-256.")
    return dict(provenance)


def _load_manifest(path: Path) -> tuple[dict[str, object], Path]:
    manifest_path = Path(path)
    if manifest_path.is_symlink():
        raise ValueError("DEKM checkpoint manifest must not be a symlink.")
    try:
        if manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
            raise ValueError("DEKM checkpoint manifest is unexpectedly large.")
        resolved = manifest_path.resolve(strict=True)
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Invalid DEKM checkpoint manifest: {manifest_path}"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError("DEKM checkpoint manifest must contain a JSON object.")
    if set(payload) != _EXPECTED_MANIFEST_KEYS:
        raise ValueError("DEKM checkpoint manifest has an invalid schema.")
    for field in ("paths", "output_sha256"):
        value = payload.get(field)
        if not isinstance(value, dict) or set(value) != _EXPECTED_OUTPUT_KEYS:
            raise ValueError(f"DEKM checkpoint manifest has invalid {field}.")
    return payload, resolved


def _require_exact_config(manifest: dict[str, object]) -> None:
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError("DEKM checkpoint manifest is missing its config.")
    if config != _EXPECTED_CONFIG:
        raise ValueError(
            "DEKM checkpoint config is not the verified release-L2 full-batch contract."
        )
    if manifest.get("config_hash") != DEKM_F_CONFIG_HASH:
        raise ValueError("DEKM checkpoint config hash does not match the verified run.")


def _resolve_weights(
    manifest: dict[str, object], manifest_path: Path
) -> tuple[Path, str]:
    paths = manifest.get("paths")
    hashes = manifest.get("output_sha256")
    relative = paths.get("final_weights") if isinstance(paths, dict) else None
    expected_hash = hashes.get("final_weights") if isinstance(hashes, dict) else None
    if not isinstance(relative, str) or not relative:
        raise ValueError("DEKM checkpoint manifest is missing final_weights.")
    posix_path = PurePosixPath(relative)
    windows_path = PureWindowsPath(relative)
    if (
        "\\" in relative
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or any(part in {"", ".", ".."} for part in posix_path.parts)
    ):
        raise ValueError("DEKM final_weights must be a safe manifest-relative path.")
    relative_path = Path(*posix_path.parts)
    if (
        not isinstance(expected_hash, str)
        or SHA256_PATTERN.fullmatch(expected_hash) is None
    ):
        raise ValueError("DEKM final_weights SHA-256 is invalid.")
    if expected_hash != DEKM_F_WEIGHT_SHA256:
        raise ValueError("DEKM final_weights is not the authorized checkpoint.")

    run_dir = manifest_path.parent.resolve(strict=True)
    candidate = run_dir / relative_path
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ValueError("DEKM final_weights file is missing.") from error
    if not resolved.is_relative_to(run_dir):
        raise ValueError("DEKM final_weights escapes its manifest run directory.")

    current = run_dir
    for part in relative_path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("DEKM final_weights must not traverse a symlink.")
    if not resolved.is_file():
        raise ValueError("DEKM final_weights is not a regular file.")
    if file_sha256(resolved) != expected_hash:
        raise ValueError("DEKM final_weights SHA-256 does not match the manifest.")
    with resolved.open("rb") as file:
        if file.read(len(HDF5_SIGNATURE)) != HDF5_SIGNATURE:
            raise ValueError("DEKM final_weights is not an HDF5 container.")
    return resolved, expected_hash


def validate_dekm_f_view1_checkpoint(
    manifest_path: Path,
    *,
    expected_dataset: str,
    expected_seed: int,
    expected_source_sha256: str,
) -> ValidatedDekmCheckpoint:
    """Fail closed unless a manifest is the verified REUTERS DEKM_F seed-42 run."""

    manifest, resolved_manifest = _load_manifest(manifest_path)
    expected_identity = {
        "schema_version": 1,
        "status": "complete",
        "method": DEKM_F_METHOD,
        "protocol_id": DEKM_F_PROTOCOL_ID,
        "dataset": expected_dataset,
        "seed": expected_seed,
        "source_sha256": expected_source_sha256,
        "n_samples": 10_000,
        "n_features": 2_000,
        "n_clusters": 4,
    }
    for field, expected in expected_identity.items():
        if manifest.get(field) != expected:
            raise ValueError(
                f"DEKM checkpoint {field} does not match the requested run."
            )
    if any(
        (
            expected_dataset != "REUTERS",
            expected_seed != 42,
            expected_source_sha256 != DEKM_F_REUTERS_SOURCE_SHA256,
        )
    ):
        raise ValueError(
            "Only the verified REUTERS seed-42 DEKM_F checkpoint is supported."
        )

    metrics = manifest.get("metrics")
    if (
        not isinstance(metrics, dict)
        or metrics.get("acc") != DEKM_F_REUTERS_ACC
        or metrics.get("nmi") != DEKM_F_REUTERS_NMI
    ):
        raise ValueError("DEKM checkpoint metrics do not match the verified run.")
    _require_exact_config(manifest)
    manifest_sha256 = file_sha256(resolved_manifest)
    if manifest_sha256 != DEKM_F_MANIFEST_SHA256:
        raise ValueError("DEKM checkpoint manifest is not the authorized manifest.")
    weights_path, weights_sha256 = _resolve_weights(manifest, resolved_manifest)
    return ValidatedDekmCheckpoint(
        manifest_path=resolved_manifest,
        weights_path=weights_path,
        manifest_sha256=manifest_sha256,
        weights_sha256=weights_sha256,
        config_hash=DEKM_F_CONFIG_HASH,
        dataset=expected_dataset,
        seed=expected_seed,
        source_sha256=expected_source_sha256,
    )


def copy_validated_checkpoint(
    checkpoint: ValidatedDekmCheckpoint,
    destination: Path,
) -> None:
    """Atomically copy validated weights into the MvDEC-owned run directory."""

    destination = Path(destination)
    if destination.is_symlink():
        raise ValueError("MvDEC checkpoint destination must not be a symlink.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(checkpoint.weights_path, temporary)
        if file_sha256(temporary) != checkpoint.weights_sha256:
            raise ValueError(
                "Copied DEKM checkpoint hash does not match its provenance."
            )
        with temporary.open("rb") as file:
            if file.read(len(HDF5_SIGNATURE)) != HDF5_SIGNATURE:
                raise ValueError("Copied DEKM checkpoint is not an HDF5 container.")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
