"""Tests for dataset-scoped experiment outputs."""

import json
from types import SimpleNamespace

import pandas as pd
import pytest

import cli
import pipeline.experiment_context as experiment_context
from pipeline.experiment_context import (
    MANIFEST_FILENAME,
    ExperimentContext,
    artifact_n_clusters,
    build_experiment_context,
)


def _artifact(dataset: str = "TIKI", n_clusters: int = 5) -> dict:
    return {
        "dataset": dataset,
        "n_clusters": n_clusters,
        "config": {"n_clusters": n_clusters},
    }


def test_experiment_context_writes_and_reuses_matching_manifest(tmp_path) -> None:
    """The same data and artifact may safely resume in one output directory."""

    data_path = tmp_path / "data.csv"
    representation_path = tmp_path / "artifact.pkl"
    output_dir = tmp_path / "results"
    data_path.write_text("x\n1\n", encoding="utf-8")
    representation_path.write_bytes(b"trusted artifact")

    first = build_experiment_context(
        _artifact(),
        data_path,
        representation_path,
        output_dir,
    )
    second = build_experiment_context(
        _artifact(),
        data_path,
        representation_path,
        output_dir,
    )

    manifest = json.loads((output_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert first == second
    assert first.n_clusters == 5
    assert manifest["schema_version"] == 4
    assert manifest["dataset"] == "TIKI"
    assert manifest["n_clusters"] == 5
    assert manifest["kprototypes_n_init"] == 10
    assert manifest["distance_contract"] == "gower_numeric_asymmetric_binary_v1"


def test_experiment_context_records_requested_downstream_random_state(
    tmp_path,
) -> None:
    """The manifest must record the seed used by downstream clustering."""

    data_path = tmp_path / "data.csv"
    representation_path = tmp_path / "artifact.pkl"
    output_dir = tmp_path / "results"
    data_path.write_text("x\n1\n", encoding="utf-8")
    representation_path.write_bytes(b"trusted artifact")

    build_experiment_context(
        _artifact(),
        data_path,
        representation_path,
        output_dir,
        random_state=44,
    )

    manifest = json.loads((output_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))

    assert manifest["random_state"] == 44


def test_default_output_directories_are_dataset_scoped(
    tmp_path,
    monkeypatch,
) -> None:
    """Tiki and Air artifacts must not share the default result directory."""

    output_root = tmp_path / "output"
    data_path = tmp_path / "data.csv"
    representation_path = tmp_path / "artifact.pkl"
    data_path.write_text("x\n1\n", encoding="utf-8")
    representation_path.write_bytes(b"artifact")
    monkeypatch.setattr(experiment_context, "OUTPUT_DIR", output_root)

    tiki = build_experiment_context(
        _artifact(dataset="TIKI"),
        data_path,
        representation_path,
    )
    air = build_experiment_context(
        _artifact(dataset="AIRPOLLUTION", n_clusters=4),
        data_path,
        representation_path,
    )

    assert tiki.output_dir == output_root / "tiki"
    assert air.output_dir == output_root / "airpollution"


def test_experiment_context_rejects_another_artifact_in_same_directory(
    tmp_path,
) -> None:
    """Resume cannot silently mix results from different artifacts."""

    data_path = tmp_path / "data.csv"
    representation_path = tmp_path / "artifact.pkl"
    output_dir = tmp_path / "results"
    data_path.write_text("x\n1\n", encoding="utf-8")
    representation_path.write_bytes(b"artifact one")
    build_experiment_context(
        _artifact(),
        data_path,
        representation_path,
        output_dir,
    )

    representation_path.write_bytes(b"artifact two")
    with pytest.raises(ValueError, match="another dataset or artifact"):
        build_experiment_context(
            _artifact(),
            data_path,
            representation_path,
            output_dir,
        )


def test_experiment_context_rejects_legacy_csv_directory_without_manifest(
    tmp_path,
) -> None:
    """Existing unscoped result files require a new output directory."""

    data_path = tmp_path / "data.csv"
    representation_path = tmp_path / "artifact.pkl"
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    data_path.write_text("x\n1\n", encoding="utf-8")
    representation_path.write_bytes(b"artifact")
    (output_dir / "ffs_results.csv").write_text("job_index\n0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="without a manifest"):
        build_experiment_context(
            _artifact(),
            data_path,
            representation_path,
            output_dir,
        )


def test_experiment_context_rejects_invalid_manifest(tmp_path) -> None:
    """A corrupt manifest must not allow an unsafe resume."""

    data_path = tmp_path / "data.csv"
    representation_path = tmp_path / "artifact.pkl"
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    data_path.write_text("x\n1\n", encoding="utf-8")
    representation_path.write_bytes(b"artifact")
    (output_dir / MANIFEST_FILENAME).write_text("{invalid", encoding="utf-8")

    with pytest.raises(ValueError, match="manifest is invalid"):
        build_experiment_context(
            _artifact(),
            data_path,
            representation_path,
            output_dir,
        )


def test_artifact_n_clusters_rejects_inconsistent_metadata() -> None:
    """Cluster count must agree between top-level and nested metadata."""

    artifact = _artifact(n_clusters=4)
    artifact["config"]["n_clusters"] = 5

    with pytest.raises(ValueError, match="does not match"):
        artifact_n_clusters(artifact)


def test_cli_passes_artifact_cluster_count_and_scoped_output(
    tmp_path,
    monkeypatch,
) -> None:
    """Without-FFS uses artifact metadata instead of the old five-cluster default."""

    args = SimpleNamespace(
        mode="without-ffs-kprototypes",
        workers=1,
        param_workers=1,
        candidate_workers=1,
        limit=1,
        no_resume=True,
        backend="kprototypes",
        data_path=tmp_path / "air.csv",
        representation_path=tmp_path / "air.pkl",
        output_dir=tmp_path / "air-results",
    )
    mvdec_result = SimpleNamespace(
        h_fused_df=pd.DataFrame({"fused_1": [0.0, 1.0]}),
        score=0.5,
        evaluation_score=0.4,
    )
    experiment = ExperimentContext(
        dataset="AIRPOLLUTION",
        n_clusters=4,
        output_dir=args.output_dir,
    )
    captured = {}
    monkeypatch.setattr(cli, "parse_args", lambda: args)
    monkeypatch.setattr(
        cli,
        "load_experiment_context",
        lambda _args: (mvdec_result, experiment),
    )
    monkeypatch.setattr(
        cli,
        "run_without_ffs",
        lambda **kwargs: captured.update(kwargs),
    )

    cli.main()

    assert captured["n_clusters"] == 4
    assert captured["baseline_score"] == 0.4
    assert captured["save_path"] == args.output_dir / "without_ffs_results.csv"
