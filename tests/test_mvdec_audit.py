import pipeline.mvdec_audit as mvdec_audit
from pipeline.mvdec_audit import artifact_data_path, audit_run_root
from pipeline.mvdec_runs import file_sha256, resolve_run_paths, write_run_manifest


def test_artifact_data_path_uses_structured_preprocessing_metadata(tmp_path):
    source_path = tmp_path / "source.csv"
    manifest = {
        "config": {"preprocessing": {"source_path": str(source_path)}},
        "paths": {"assignments": "unused.csv"},
    }

    assert artifact_data_path(manifest) == source_path


def test_audit_retains_failed_manifest_without_loading_artifact(tmp_path):
    config = {"batch_size": 256}
    paths = resolve_run_paths(
        tmp_path,
        "TIKI",
        "mvdec_dekm_consistent_v1",
        config,
        seed=42,
        run_index=1,
    )
    paths.run_dir.mkdir(parents=True)
    write_run_manifest(
        paths.manifest,
        {
            "run_id": paths.run_id,
            "status": "failed",
            "config": config,
            "config_hash": paths.config_hash,
        },
    )

    assert audit_run_root(tmp_path) == (0, 1)


def test_audit_validates_and_reloads_complete_artifact(tmp_path, monkeypatch):
    source_path = tmp_path / "source.csv"
    source_path.write_text("feature\n1\n", encoding="utf-8")
    config = {"preprocessing": {"source_path": str(source_path)}}
    paths = resolve_run_paths(
        tmp_path,
        "TIKI",
        "mvdec_dekm_consistent_v1",
        config,
        seed=42,
        run_index=1,
    )
    paths.run_dir.mkdir(parents=True)
    paths.artifact.write_bytes(b"artifact")
    paths.assignments.write_text("cluster\n0\n", encoding="utf-8")
    write_run_manifest(
        paths.manifest,
        {
            "run_id": paths.run_id,
            "status": "complete",
            "config": config,
            "config_hash": paths.config_hash,
            "paths": {
                "artifact": paths.artifact.name,
                "assignments": paths.assignments.name,
            },
            "output_sha256": {
                "artifact": file_sha256(paths.artifact),
                "assignments": file_sha256(paths.assignments),
            },
        },
    )
    loaded = []

    def fake_load_mvdec_result(*, result_path, data_path):
        loaded.append((result_path, data_path))

    monkeypatch.setattr(mvdec_audit, "load_mvdec_result", fake_load_mvdec_result)

    assert audit_run_root(tmp_path) == (1, 0)
    assert loaded == [(paths.artifact, source_path)]
