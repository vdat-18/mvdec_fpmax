import json
from pathlib import Path

import pytest

from pipeline.mvdec_runs import (
    RUN_OUTPUT_NAMES,
    build_summary_frames,
    canonical_config_hash,
    complete_run_manifests,
    file_sha256,
    load_run_manifest,
    load_run_manifests,
    prepare_run_directory,
    resolve_run_paths,
    resolve_training_commit,
    validate_manifest_outputs,
    write_run_manifest,
)


def _complete_outputs(root: Path) -> tuple[dict[str, str], dict[str, str]]:
    paths = {}
    hashes = {}
    for name in RUN_OUTPUT_NAMES:
        path = root / f"{name}.bin"
        path.write_bytes(name.encode())
        paths[name] = str(path)
        hashes[name] = file_sha256(path)
    return paths, hashes


def test_run_paths_isolate_seed_and_config(tmp_path):
    config = {"protocol": "primary", "batch_size": 256}
    seed_42 = resolve_run_paths(
        tmp_path,
        "TIKI",
        "mvdec_dekm_consistent_v1",
        config,
        seed=42,
        run_index=1,
    )
    seed_43 = resolve_run_paths(
        tmp_path,
        "TIKI",
        "mvdec_dekm_consistent_v1",
        config,
        seed=43,
        run_index=2,
    )
    different_config = resolve_run_paths(
        tmp_path,
        "TIKI",
        "mvdec_dekm_consistent_v1",
        {**config, "batch_size": 128},
        seed=42,
        run_index=1,
    )

    assert seed_42.config_dir == seed_43.config_dir
    assert seed_42.run_dir != seed_43.run_dir
    assert seed_42.config_dir != different_config.config_dir
    assert "seed_42" in seed_42.run_id
    assert seed_42.config_hash[:12] in seed_42.run_id


def test_prepare_run_directory_rejects_reuse_without_force(tmp_path):
    paths = resolve_run_paths(
        tmp_path,
        "REUTERS",
        "mvdec_dekm_consistent_v1",
        {"batch_size": 256},
        seed=42,
        run_index=1,
    )
    prepare_run_directory(paths)
    paths.artifact.write_bytes(b"artifact")
    unrelated = paths.run_dir / "research_notes.txt"
    unrelated.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--force"):
        prepare_run_directory(paths)

    prepare_run_directory(paths, force=True)
    assert not paths.artifact.exists()
    assert unrelated.read_text(encoding="utf-8") == "keep"


def test_manifest_loader_keeps_failed_runs_for_audit(tmp_path):
    config = {"batch_size": 256}
    complete = resolve_run_paths(
        tmp_path,
        "TIKI",
        "mvdec_dekm_consistent_v1",
        config,
        seed=42,
        run_index=1,
    )
    failed = resolve_run_paths(
        tmp_path,
        "TIKI",
        "mvdec_dekm_consistent_v1",
        config,
        seed=43,
        run_index=2,
    )
    prepare_run_directory(complete)
    prepare_run_directory(failed)
    write_run_manifest(
        complete.manifest,
        {
            "run_id": complete.run_id,
            "status": "complete",
            "config": config,
            "config_hash": complete.config_hash,
        },
    )
    write_run_manifest(
        failed.manifest,
        {
            "run_id": failed.run_id,
            "status": "failed",
            "config": config,
            "config_hash": failed.config_hash,
        },
    )

    manifests = load_run_manifests(complete.config_dir)
    completed = complete_run_manifests(complete.config_dir)

    assert [manifest["status"] for manifest in manifests] == ["complete", "failed"]
    assert [manifest["run_id"] for manifest in completed] == [complete.run_id]
    assert (
        json.loads(complete.manifest.read_text(encoding="utf-8"))["schema_version"] == 1
    )


def test_summary_frames_exclude_failed_manifests(tmp_path):
    shared = {
        "dataset": "REUTERS",
        "method": "MvDEC-DEKM-consistent",
        "protocol_id": "mvdec_dekm_consistent_v1",
        "config_hash": "a" * 64,
    }
    manifests = [
        {
            **shared,
            "run_id": "seed_42",
            "status": "complete",
            "seed": 42,
            "run_index": 1,
            "runtime_seconds": 10.0,
            "paths": {
                "artifact": str(tmp_path / "42.pkl"),
                "assignments": str(tmp_path / "42.csv"),
            },
            "metrics": {
                "acc": 0.8,
                "nmi": 0.7,
                "view1_acc": 0.75,
                "view1_nmi": 0.65,
                "view2_acc": 0.7,
                "view2_nmi": 0.6,
                "fused_acc": 0.8,
                "fused_nmi": 0.7,
            },
        },
        {
            **shared,
            "run_id": "seed_43",
            "status": "failed",
            "seed": 43,
            "run_index": 2,
        },
    ]

    runs, summary = build_summary_frames(manifests)

    assert runs["run_id"].tolist() == ["seed_42"]
    assert summary.loc[0, "discovered_runs"] == 2
    assert summary.loc[0, "completed_runs"] == 1
    assert summary.loc[0, "failed_runs"] == 1
    assert summary.loc[0, "acc_mean"] == pytest.approx(0.8)
    assert summary.loc[0, "view1_acc_mean"] == pytest.approx(0.75)
    assert summary.loc[0, "view2_nmi_mean"] == pytest.approx(0.6)
    assert summary.loc[0, "fused_acc_mean"] == pytest.approx(0.8)


def test_manifest_loader_rejects_tampered_config(tmp_path):
    config = {"batch_size": 256}
    paths = resolve_run_paths(
        tmp_path,
        "TIKI",
        "mvdec_dekm_consistent_v1",
        config,
        seed=42,
        run_index=1,
    )
    prepare_run_directory(paths)
    write_run_manifest(
        paths.manifest,
        {
            "run_id": paths.run_id,
            "status": "complete",
            "config": config,
            "config_hash": paths.config_hash,
        },
    )
    payload = json.loads(paths.manifest.read_text(encoding="utf-8"))
    payload["config"]["batch_size"] = 128
    paths.manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="config hash"):
        load_run_manifests(paths.config_dir)


def test_manifest_output_path_cannot_escape_run_directory(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    paths, hashes = _complete_outputs(run_dir)
    outside = tmp_path / "outside.pkl"
    outside.write_bytes(b"artifact")
    paths["artifact"] = "../outside.pkl"
    hashes["artifact"] = file_sha256(outside)
    manifest_path = run_dir / "manifest.json"
    write_run_manifest(
        manifest_path,
        {
            "status": "complete",
            "config": {},
            "config_hash": canonical_config_hash({}),
            "paths": paths,
            "output_sha256": hashes,
        },
    )
    manifest = load_run_manifest(manifest_path)

    with pytest.raises(ValueError, match="escapes its run directory"):
        validate_manifest_outputs(manifest)


def test_manifest_absolute_output_path_cannot_escape_run_directory(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    paths, hashes = _complete_outputs(run_dir)
    outside = tmp_path / "outside.pkl"
    outside.write_bytes(b"artifact")
    paths["artifact"] = str(outside.resolve())
    hashes["artifact"] = file_sha256(outside)
    manifest_path = run_dir / "manifest.json"
    write_run_manifest(
        manifest_path,
        {
            "status": "complete",
            "config": {},
            "config_hash": canonical_config_hash({}),
            "paths": paths,
            "output_sha256": hashes,
        },
    )
    manifest = load_run_manifest(manifest_path)

    with pytest.raises(ValueError, match="escapes its run directory"):
        validate_manifest_outputs(manifest)


def test_complete_manifest_output_hashes_are_verified(tmp_path):
    paths, hashes = _complete_outputs(tmp_path)
    manifest = {
        "status": "complete",
        "paths": paths,
        "output_sha256": hashes,
    }

    validate_manifest_outputs(manifest)
    artifact = Path(paths["artifact"])
    artifact.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="output hash"):
        validate_manifest_outputs(manifest)


def test_complete_manifest_requires_every_run_output(tmp_path):
    paths, hashes = _complete_outputs(tmp_path)
    paths.pop("final_view2")
    hashes.pop("final_view2")
    manifest = {
        "status": "complete",
        "paths": paths,
        "output_sha256": hashes,
    }

    with pytest.raises(ValueError, match="every required run output"):
        validate_manifest_outputs(manifest)


def test_complete_manifest_requires_hash_for_every_output(tmp_path):
    paths, hashes = _complete_outputs(tmp_path)
    hashes.pop("assignments")
    manifest = {
        "status": "complete",
        "paths": paths,
        "output_sha256": hashes,
    }

    with pytest.raises(ValueError, match="hash every declared output"):
        validate_manifest_outputs(manifest)


def test_resolve_training_commit_returns_full_head() -> None:
    commit = resolve_training_commit(Path(__file__).resolve().parents[1])

    assert len(commit) == 40
    assert set(commit) <= set("0123456789abcdef")
