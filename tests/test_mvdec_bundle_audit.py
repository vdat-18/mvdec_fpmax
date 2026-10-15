import tarfile
import zipfile

import pandas as pd
import pytest

from pipeline.mvdec_bundle_audit import validate_archive_bundle
from pipeline.mvdec_runs import (
    RUN_OUTPUT_NAMES,
    file_sha256,
    load_run_manifest,
    resolve_run_paths,
    write_run_manifest,
)


def test_validate_archive_bundle_checks_archives_and_combined_zip(tmp_path) -> None:
    config = {"batch_size": 4}
    paths = resolve_run_paths(
        tmp_path / "output",
        "REUTERS",
        "protocol",
        config,
        seed=42,
        run_index=1,
    )
    paths.run_dir.mkdir(parents=True)
    for name in RUN_OUTPUT_NAMES:
        getattr(paths, name).write_bytes(f"{name}\n".encode())
    output_paths = {name: getattr(paths, name).name for name in RUN_OUTPUT_NAMES}
    write_run_manifest(
        paths.manifest,
        {
            "run_id": paths.run_id,
            "status": "complete",
            "dataset": "REUTERS",
            "protocol_id": "protocol",
            "seed": 42,
            "run_index": 1,
            "config": config,
            "config_hash": paths.config_hash,
            "training_commit": "a" * 40,
            "source_id": "source",
            "source_sha256": "b" * 64,
            "n_samples": 4,
            "n_features": 2,
            "n_classes": 2,
            "metrics": {
                "acc": 0.75,
                "nmi": 0.5,
                "view1_acc": 0.75,
                "view1_nmi": 0.5,
                "view2_acc": 0.5,
                "view2_nmi": 0.25,
                "fused_acc": 0.75,
                "fused_nmi": 0.5,
            },
            "paths": output_paths,
        },
    )
    manifest = load_run_manifest(paths.manifest)

    archive_root = tmp_path / "archives"
    archive_root.mkdir()
    archive_path = archive_root / "run.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(paths.run_dir, arcname=paths.run_dir.name)
    digest = file_sha256(archive_path)
    archive_path.with_suffix(".gz.sha256").write_text(
        f"{digest}  {archive_path.name}\n",
        encoding="utf-8",
    )
    (archive_root / "failures.json").write_text("[]\n", encoding="utf-8")
    pd.DataFrame(
        [
            {
                "dataset": "REUTERS",
                "protocol_id": "protocol",
                "seed": 42,
                "run_id": paths.run_id,
                "config_hash": paths.config_hash,
                "training_commit": "a" * 40,
                "source_id": "source",
                "source_sha256": "b" * 64,
                "n_samples": 4,
                "n_features": 2,
                "n_classes": 2,
                "acc": 0.75,
                "nmi": 0.5,
                "view1_acc": 0.75,
                "view1_nmi": 0.5,
                "view2_acc": 0.5,
                "view2_nmi": 0.25,
                "fused_acc": 0.75,
                "fused_nmi": 0.5,
                "archive_path": f"/kaggle/working/{archive_path.name}",
                "archive_sha256": digest,
            }
        ]
    ).to_csv(archive_root / "public_mvdec_matrix_runs.csv", index=False)
    pd.DataFrame(
        [
            {
                "dataset": "REUTERS",
                "protocol_id": "protocol",
                "config_hash": paths.config_hash,
                "completed_runs": 1,
                "seeds": "42",
                "acc_mean": 0.75,
                "acc_std": None,
                "nmi_mean": 0.5,
                "nmi_std": None,
                "best_acc": 0.75,
                "best_acc_seed": 42,
                "best_nmi": 0.5,
                "best_nmi_seed": 42,
                "primary_reporting": "multi_seed_mean_sample_std",
                "best_reporting": "sensitivity_only_not_primary",
            }
        ]
    ).to_csv(archive_root / "public_mvdec_matrix_aggregate.csv", index=False)

    combined_zip = tmp_path / "combined.zip"
    with zipfile.ZipFile(combined_zip, "w") as archive:
        for local_path in archive_root.rglob("*"):
            if local_path.is_file():
                archive.write(local_path, local_path.relative_to(archive_root))
    combined_digest = file_sha256(combined_zip)
    combined_zip.with_suffix(".zip.sha256").write_text(
        f"{combined_digest}  {combined_zip.name}\n",
        encoding="utf-8",
    )

    validate_archive_bundle(archive_root, [manifest], combined_zip)


def test_validate_archive_bundle_rejects_recorded_failures(tmp_path) -> None:
    archive_root = tmp_path / "archives"
    archive_root.mkdir()
    (archive_root / "failures.json").write_text(
        '[{"dataset": "RCV1"}]\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="records failed runs"):
        validate_archive_bundle(archive_root, [])
