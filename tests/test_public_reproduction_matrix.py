"""Tests for the public MvDEC reproduction matrix runner."""

import csv
import tarfile
from pathlib import Path

import pytest

import pipeline.public_reproduction_matrix as matrix


def test_build_run_specs_preserves_matrix_order() -> None:
    """Dataset, protocol, and seed order must remain caller-controlled."""

    specs = matrix.build_run_specs(
        ["reuters", "20NEWS"],
        ["protocol_a", "protocol_b"],
        [42, 43],
    )

    assert specs == (
        matrix.MatrixRunSpec("REUTERS", "protocol_a", 42),
        matrix.MatrixRunSpec("REUTERS", "protocol_a", 43),
        matrix.MatrixRunSpec("REUTERS", "protocol_b", 42),
        matrix.MatrixRunSpec("REUTERS", "protocol_b", 43),
        matrix.MatrixRunSpec("20NEWS", "protocol_a", 42),
        matrix.MatrixRunSpec("20NEWS", "protocol_a", 43),
        matrix.MatrixRunSpec("20NEWS", "protocol_b", 42),
        matrix.MatrixRunSpec("20NEWS", "protocol_b", 43),
    )


def test_package_run_writes_portable_archive_and_checksum(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Each completed run must be independently downloadable and verifiable."""

    project_dir = tmp_path / "repo"
    run_dir = project_dir / "output" / "public" / "seed_42"
    run_dir.mkdir(parents=True)
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    (run_dir / "artifact.pkl").write_bytes(b"artifact")
    monkeypatch.setattr(matrix, "PROJECT_DIR", project_dir)

    archive_path = tmp_path / "archives" / "run.tar.gz"
    checksum_path = archive_path.with_suffix(".gz.sha256")
    digest = matrix.package_run(
        {"_manifest_path": str(manifest_path)},
        archive_path,
        checksum_path,
    )

    assert digest == matrix.file_sha256(archive_path)
    assert matrix.checksum_matches(archive_path, checksum_path)
    with tarfile.open(archive_path, "r:gz") as archive:
        names = archive.getnames()
    assert "output/public/seed_42/manifest.json" in names
    assert "output/public/seed_42/artifact.pkl" in names


def test_package_run_supports_output_outside_project_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Cloud output directories may be siblings of the cloned repository."""

    project_dir = tmp_path / "repo"
    project_dir.mkdir()
    run_dir = tmp_path / "cloud-output" / "seed_42"
    run_dir.mkdir(parents=True)
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    (run_dir / "artifact.pkl").write_bytes(b"artifact")
    monkeypatch.setattr(matrix, "PROJECT_DIR", project_dir)

    archive_path = tmp_path / "archives" / "run.tar.gz"
    checksum_path = archive_path.with_suffix(".gz.sha256")
    matrix.package_run(
        {"_manifest_path": str(manifest_path)},
        archive_path,
        checksum_path,
    )

    assert matrix.checksum_matches(archive_path, checksum_path)
    with tarfile.open(archive_path, "r:gz") as archive:
        names = archive.getnames()
    assert "seed_42/manifest.json" in names
    assert "seed_42/artifact.pkl" in names


def test_run_training_redirects_verbose_child_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Training noise must be retained in a file instead of flooding stdout."""

    project_dir = tmp_path / "repo"
    training_script = project_dir / "src" / "representation_learning" / "MVDEC_dense.py"
    training_script.parent.mkdir(parents=True)
    training_script.write_text("print('verbose child output')\n", encoding="utf-8")
    monkeypatch.setattr(matrix, "PROJECT_DIR", project_dir)
    stdout_path = tmp_path / "training.log"

    matrix.run_training(
        matrix.MatrixRunSpec("REUTERS", "protocol_a", 42),
        dataset_root=tmp_path / "datasets",
        output_root=tmp_path / "output",
        progress_interval=100,
        stdout_path=stdout_path,
        heartbeat_seconds=300,
    )

    assert stdout_path.read_text(encoding="utf-8") == "verbose child output\n"


def test_reusable_complete_manifest_rejects_corrupt_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Resume must rebuild a run whose manifest hashes no longer validate."""

    manifest = {
        "status": "complete",
        "training_commit": "a" * 40,
        "_manifest_path": str(tmp_path / "run" / "manifest.json"),
    }
    monkeypatch.setattr(
        matrix,
        "matching_complete_manifests",
        lambda _output_root, _spec: [manifest],
    )

    def reject_run(*args, **kwargs):
        raise ValueError("hash mismatch")

    monkeypatch.setattr(matrix, "audit_run_root", reject_run)
    monkeypatch.setattr(matrix, "resolve_training_commit", lambda _root: "a" * 40)

    reusable = matrix.reusable_complete_manifest(
        tmp_path,
        matrix.MatrixRunSpec("REUTERS", "protocol_a", 42),
        tmp_path / "datasets",
    )

    assert reusable is None


def test_summary_record_includes_all_representation_metrics(tmp_path: Path) -> None:
    metrics = {
        "acc": 0.8,
        "nmi": 0.7,
        "view1_acc": 0.6,
        "view1_nmi": 0.5,
        "view2_acc": 0.65,
        "view2_nmi": 0.55,
        "fused_acc": 0.8,
        "fused_nmi": 0.7,
    }
    manifest = {
        "dataset": "REUTERS",
        "protocol_id": "protocol",
        "seed": 42,
        "run_id": "run",
        "config_hash": "hash",
        "training_commit": "a" * 40,
        "source_id": "source",
        "source_sha256": "b" * 64,
        "n_samples": 10,
        "n_features": 2,
        "n_classes": 2,
        "runtime_seconds": 1.0,
        "metrics": metrics,
    }

    record = matrix.summary_record(manifest, tmp_path / "run.tar.gz", "digest")

    assert {name: record[name] for name in metrics} == metrics


def test_update_summary_upserts_one_run_identity(tmp_path: Path) -> None:
    """Resumed packaging must replace, rather than duplicate, a summary row."""

    summary_path = tmp_path / matrix.SUMMARY_FILENAME
    base_record = {
        "dataset": "REUTERS",
        "protocol_id": "protocol_a",
        "seed": 42,
        "run_id": "run_old",
        "config_hash": "hash_old",
        "acc": 0.7,
        "nmi": 0.6,
        "runtime_seconds": 10.0,
        "archive_path": "old.tar.gz",
        "archive_sha256": "old",
    }
    matrix.update_summary(summary_path, base_record)
    matrix.update_summary(
        summary_path,
        {
            **base_record,
            "run_id": "run_new",
            "acc": 0.8,
            "archive_sha256": "new",
        },
    )

    with summary_path.open(newline="", encoding="utf-8") as file:
        records = list(csv.DictReader(file))
    assert len(records) == 1
    assert records[0]["run_id"] == "run_new"
    assert records[0]["acc"] == "0.8"
    assert records[0]["archive_sha256"] == "new"


def test_aggregate_summary_separates_mean_std_from_best_sensitivity() -> None:
    records = [
        {
            "dataset": "REUTERS",
            "protocol_id": "protocol_a",
            "config_hash": "same",
            "seed": seed,
            "acc": acc,
            "nmi": nmi,
        }
        for seed, acc, nmi in (
            (42, 0.70, 0.50),
            (43, 0.80, 0.40),
            (44, 0.60, 0.60),
        )
    ]

    aggregate = matrix.aggregate_summary_records(records)[0]

    assert aggregate["completed_runs"] == 3
    assert aggregate["seeds"] == "42,43,44"
    assert aggregate["acc_mean"] == 0.7
    assert aggregate["acc_std"] == pytest.approx(0.1)
    assert aggregate["nmi_mean"] == 0.5
    assert aggregate["nmi_std"] == pytest.approx(0.1)
    assert aggregate["best_acc"] == 0.8
    assert aggregate["best_acc_seed"] == 43
    assert aggregate["best_nmi"] == 0.6
    assert aggregate["best_nmi_seed"] == 44
    assert aggregate["primary_reporting"] == "multi_seed_mean_sample_std"
    assert aggregate["best_reporting"] == "sensitivity_only_not_primary"


def test_write_aggregate_summary_groups_by_config_hash(tmp_path: Path) -> None:
    summary_path = tmp_path / matrix.SUMMARY_FILENAME
    records = [
        {
            "dataset": "REUTERS",
            "protocol_id": "protocol_a",
            "seed": 42,
            "run_id": "run_42",
            "config_hash": "config_a",
            "acc": 0.7,
            "nmi": 0.5,
            "runtime_seconds": 1.0,
            "archive_path": "a.tar.gz",
            "archive_sha256": "a",
        },
        {
            "dataset": "REUTERS",
            "protocol_id": "protocol_a",
            "seed": 43,
            "run_id": "run_43",
            "config_hash": "config_b",
            "acc": 0.8,
            "nmi": 0.6,
            "runtime_seconds": 1.0,
            "archive_path": "b.tar.gz",
            "archive_sha256": "b",
        },
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=matrix.SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(records)

    aggregate_path = tmp_path / matrix.AGGREGATE_SUMMARY_FILENAME
    matrix.write_aggregate_summary(summary_path, aggregate_path)

    with aggregate_path.open(newline="", encoding="utf-8") as file:
        aggregates = list(csv.DictReader(file))
    assert len(aggregates) == 2
    assert {row["config_hash"] for row in aggregates} == {"config_a", "config_b"}
    assert all(row["acc_std"] == "" for row in aggregates)
    assert all(row["nmi_std"] == "" for row in aggregates)


def test_reusable_manifest_rejects_stale_training_commit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest = {"status": "complete", "training_commit": "a" * 40}
    monkeypatch.setattr(
        matrix,
        "matching_complete_manifests",
        lambda _output_root, _spec: [manifest],
    )
    monkeypatch.setattr(matrix, "resolve_training_commit", lambda _root: "b" * 40)

    assert (
        matrix.reusable_complete_manifest(
            tmp_path,
            matrix.MatrixRunSpec("REUTERS", "protocol_a", 42),
            tmp_path / "datasets",
        )
        is None
    )


def test_reusable_manifest_ignores_stale_duplicate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    current = {
        "status": "complete",
        "training_commit": "b" * 40,
        "_manifest_path": str(tmp_path / "current" / "manifest.json"),
    }
    stale = {"status": "complete", "training_commit": "a" * 40}
    monkeypatch.setattr(
        matrix,
        "matching_complete_manifests",
        lambda _output_root, _spec: [stale, current],
    )
    monkeypatch.setattr(matrix, "resolve_training_commit", lambda _root: "b" * 40)
    monkeypatch.setattr(matrix, "audit_run_root", lambda *args, **kwargs: (1, 0))

    reusable = matrix.reusable_complete_manifest(
        tmp_path,
        matrix.MatrixRunSpec("REUTERS", "protocol_a", 42),
        tmp_path / "datasets",
    )

    assert reusable is current


def test_auditable_manifest_filter_ignores_invalid_same_commit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    valid = {
        "status": "complete",
        "training_commit": "b" * 40,
        "_manifest_path": str(tmp_path / "valid" / "manifest.json"),
    }
    invalid = {
        "status": "complete",
        "training_commit": "b" * 40,
        "_manifest_path": str(tmp_path / "invalid" / "manifest.json"),
    }
    monkeypatch.setattr(
        matrix,
        "matching_complete_manifests",
        lambda _output_root, _spec: [invalid, valid],
    )
    def audit_candidate(root, *args, **kwargs):
        if root.name == "invalid":
            raise ValueError("corrupt artifact")
        return 1, 0

    monkeypatch.setattr(matrix, "audit_run_root", audit_candidate)

    selected = matrix.require_single_auditable_manifest(
        tmp_path,
        matrix.MatrixRunSpec("REUTERS", "protocol_a", 42),
        tmp_path / "datasets",
        "b" * 40,
    )

    assert selected is valid


def test_archive_path_is_bound_to_config_hash(tmp_path: Path) -> None:
    spec = matrix.MatrixRunSpec("REUTERS", "protocol_a", 42)

    first, _ = matrix.archive_paths(tmp_path, spec, "a" * 64)
    second, _ = matrix.archive_paths(tmp_path, spec, "b" * 64)

    assert first != second
    assert "aaaaaaaaaaaa" in first.name
