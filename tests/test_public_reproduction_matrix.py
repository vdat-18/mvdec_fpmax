"""Tests for the public MvDEC reproduction matrix runner."""

import csv
import tarfile
from pathlib import Path

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
