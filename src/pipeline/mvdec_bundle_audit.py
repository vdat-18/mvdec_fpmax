"""Verify packaged MvDEC run archives and matrix summaries."""

import json
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from pipeline.mvdec_runs import (
    RUN_MANIFEST_FILENAME,
    RUN_OUTPUT_NAMES,
    file_sha256,
    resolve_manifest_output_path,
)


def _validate_checksum_sidecar(archive_path: Path) -> str:
    """Verify one archive against its adjacent GNU-style checksum file."""

    checksum_path = archive_path.with_suffix(f"{archive_path.suffix}.sha256")
    if not checksum_path.is_file():
        raise ValueError(f"MvDEC archive checksum is missing: {checksum_path}")
    fields = checksum_path.read_text(encoding="utf-8").split()
    if len(fields) != 2 or fields[1] != archive_path.name:
        raise ValueError(f"MvDEC archive checksum metadata is invalid: {checksum_path}")
    actual = file_sha256(archive_path)
    if fields[0].lower() != actual:
        raise ValueError(f"MvDEC archive checksum does not match: {archive_path}")
    return actual


def _safe_archive_member(name: str) -> PurePosixPath:
    """Reject absolute and parent-traversing archive members."""

    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"MvDEC archive contains an unsafe member: {name}")
    return path


def _validate_run_archive(
    archive_path: Path,
    manifest: dict[str, object],
) -> None:
    """Verify that a run archive exactly preserves its audited run files."""

    manifest_paths = manifest.get("paths")
    if not isinstance(manifest_paths, dict):
        raise ValueError("MvDEC run archive manifest is missing output paths.")
    expected_names = {
        RUN_MANIFEST_FILENAME,
        *(Path(str(manifest_paths[name])).name for name in RUN_OUTPUT_NAMES),
    }
    local_manifest_path = Path(str(manifest["_manifest_path"]))
    with tarfile.open(archive_path, "r:gz") as archive:
        file_members = [member for member in archive.getmembers() if member.isfile()]
        if any(
            not (member.isfile() or member.isdir()) for member in archive.getmembers()
        ):
            raise ValueError(
                f"MvDEC run archive contains a non-regular member: {archive_path}"
            )
        paths = [_safe_archive_member(member.name) for member in file_members]
        manifest_members = [
            (member, member_path)
            for member, member_path in zip(file_members, paths, strict=True)
            if member_path.name == RUN_MANIFEST_FILENAME
        ]
        if len(manifest_members) != 1:
            raise ValueError(
                f"MvDEC run archive must contain one manifest: {archive_path}"
            )
        run_root = manifest_members[0][1].parent
        archived_names = {
            member_path.name for member_path in paths if member_path.parent == run_root
        }
        if archived_names != expected_names or len(file_members) != len(expected_names):
            raise ValueError(
                f"MvDEC run archive has unexpected contents: {archive_path}"
            )
        for member, member_path in zip(file_members, paths, strict=True):
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(
                    f"MvDEC run archive member is unreadable: {member.name}"
                )
            if member_path.name == RUN_MANIFEST_FILENAME:
                local_path = local_manifest_path
            else:
                output_name = next(
                    name
                    for name in RUN_OUTPUT_NAMES
                    if Path(str(manifest["paths"][name])).name == member_path.name
                )
                local_path = resolve_manifest_output_path(manifest, output_name)
            if extracted.read() != local_path.read_bytes():
                raise ValueError(
                    "MvDEC run archive member differs from audited output: "
                    f"{member.name}"
                )


def _validate_combined_zip(zip_path: Path, archive_root: Path) -> None:
    """Verify the combined ZIP checksum, paths, and every archived byte."""

    _validate_checksum_sidecar(zip_path)
    expected_files = {
        path.relative_to(archive_root).as_posix(): path
        for path in archive_root.rglob("*")
        if path.is_file()
    }
    with zipfile.ZipFile(zip_path) as archive:
        members = [info for info in archive.infolist() if not info.is_dir()]
        member_names = [
            _safe_archive_member(info.filename).as_posix() for info in members
        ]
        if len(member_names) != len(set(member_names)):
            raise ValueError("MvDEC combined ZIP contains duplicate file members.")
        if set(member_names) != set(expected_files):
            raise ValueError(
                "MvDEC combined ZIP does not mirror the archive directory."
            )
        for info, name in zip(members, member_names, strict=True):
            if archive.read(info) != expected_files[name].read_bytes():
                raise ValueError(f"MvDEC combined ZIP member differs locally: {name}")


def _validate_aggregate_summary(
    archive_root: Path,
    manifests: list[dict[str, object]],
) -> None:
    """Recompute aggregate rows from complete manifests."""

    aggregate_path = archive_root / "public_mvdec_matrix_aggregate.csv"
    try:
        aggregate = pd.read_csv(aggregate_path)
    except (OSError, pd.errors.ParserError) as error:
        raise ValueError(
            "MvDEC archive bundle has an invalid aggregate summary."
        ) from error
    grouped: dict[tuple[object, object, object], list[dict[str, object]]] = {}
    for manifest in manifests:
        key = (
            manifest.get("dataset"),
            manifest.get("protocol_id"),
            manifest.get("config_hash"),
        )
        grouped.setdefault(key, []).append(manifest)
    aggregate_identities = {
        (row.dataset, row.protocol_id, row.config_hash)
        for row in aggregate.itertuples(index=False)
    }
    if len(aggregate) != len(grouped) or len(aggregate_identities) != len(aggregate):
        raise ValueError("MvDEC aggregate summary has duplicate or missing rows.")
    if aggregate_identities != set(grouped):
        raise ValueError("MvDEC aggregate summary identities do not match manifests.")

    for row in aggregate.itertuples(index=False):
        runs = sorted(
            grouped[(row.dataset, row.protocol_id, row.config_hash)],
            key=lambda manifest: int(manifest["seed"]),
        )
        acc = np.array([float(manifest["metrics"]["acc"]) for manifest in runs])
        nmi = np.array([float(manifest["metrics"]["nmi"]) for manifest in runs])
        best_acc = int(np.argmax(acc))
        best_nmi = int(np.argmax(nmi))
        expected_std_acc = float(acc.std(ddof=1)) if len(acc) > 1 else np.nan
        expected_std_nmi = float(nmi.std(ddof=1)) if len(nmi) > 1 else np.nan
        if any(
            (
                int(row.completed_runs) != len(runs),
                str(row.seeds) != ",".join(str(run["seed"]) for run in runs),
                not np.isclose(float(row.acc_mean), float(acc.mean())),
                not np.isclose(float(row.nmi_mean), float(nmi.mean())),
                not (
                    np.isnan(expected_std_acc)
                    and pd.isna(row.acc_std)
                    or np.isclose(float(row.acc_std), expected_std_acc)
                ),
                not (
                    np.isnan(expected_std_nmi)
                    and pd.isna(row.nmi_std)
                    or np.isclose(float(row.nmi_std), expected_std_nmi)
                ),
                not np.isclose(float(row.best_acc), float(acc[best_acc])),
                int(row.best_acc_seed) != int(runs[best_acc]["seed"]),
                not np.isclose(float(row.best_nmi), float(nmi[best_nmi])),
                int(row.best_nmi_seed) != int(runs[best_nmi]["seed"]),
                row.primary_reporting != "multi_seed_mean_sample_std",
                row.best_reporting != "sensitivity_only_not_primary",
            )
        ):
            raise ValueError("MvDEC aggregate summary metrics do not match manifests.")


def validate_archive_bundle(
    archive_root: Path,
    manifests: list[dict[str, object]],
    combined_zip: Path | None = None,
) -> None:
    """Verify failure sentinel, summaries, per-run archives, and combined ZIP."""

    failures_path = archive_root / "failures.json"
    try:
        failures = json.loads(failures_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            "MvDEC archive bundle has an invalid failures.json."
        ) from error
    if failures != []:
        raise ValueError("MvDEC archive bundle records failed runs.")
    if any(manifest.get("status") != "complete" for manifest in manifests):
        raise ValueError(
            "MvDEC archive bundle cannot declare success with incomplete manifests."
        )

    summary_path = archive_root / "public_mvdec_matrix_runs.csv"
    try:
        summary = pd.read_csv(summary_path)
    except (OSError, pd.errors.ParserError) as error:
        raise ValueError("MvDEC archive bundle has an invalid run summary.") from error
    complete = [
        manifest for manifest in manifests if manifest.get("status") == "complete"
    ]
    manifest_by_identity = {
        (manifest["dataset"], manifest["protocol_id"], int(manifest["seed"])): manifest
        for manifest in complete
    }
    summary_identities = {
        (row.dataset, row.protocol_id, int(row.seed))
        for row in summary.itertuples(index=False)
    }
    if (
        len(manifest_by_identity) != len(complete)
        or len(summary_identities) != len(summary)
        or len(summary) != len(complete)
        or summary_identities != set(manifest_by_identity)
    ):
        raise ValueError("MvDEC archive summary does not match complete manifests.")

    stable_fields = (
        "run_id",
        "config_hash",
        "training_commit",
        "source_id",
        "source_sha256",
    )
    for row in summary.itertuples(index=False):
        identity = (row.dataset, row.protocol_id, int(row.seed))
        manifest = manifest_by_identity[identity]
        if any(
            str(getattr(row, field)) != str(manifest.get(field))
            for field in stable_fields
        ):
            raise ValueError(
                "MvDEC archive summary provenance does not match manifests."
            )
        integer_fields = ("n_samples", "n_features", "n_classes")
        if any(
            int(getattr(row, field)) != int(manifest.get(field, -1))
            for field in integer_fields
        ):
            raise ValueError("MvDEC archive summary dimensions do not match manifests.")
        metrics = manifest.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError("MvDEC archive manifest is missing summary metrics.")
        metric_fields = (
            "acc",
            "nmi",
            "view1_acc",
            "view1_nmi",
            "view2_acc",
            "view2_nmi",
            "fused_acc",
            "fused_nmi",
        )
        if any(
            not np.isclose(float(getattr(row, field)), float(metrics.get(field)))
            for field in metric_fields
        ):
            raise ValueError("MvDEC archive summary metrics do not match manifests.")
        archive_path = archive_root / Path(str(row.archive_path)).name
        if not archive_path.is_file():
            raise ValueError(f"MvDEC run archive is missing: {archive_path}")
        digest = _validate_checksum_sidecar(archive_path)
        if str(row.archive_sha256).lower() != digest:
            raise ValueError("MvDEC run summary archive hash does not match.")
        _validate_run_archive(archive_path, manifest)

    _validate_aggregate_summary(archive_root, complete)
    if combined_zip is not None:
        _validate_combined_zip(combined_zip, archive_root)
