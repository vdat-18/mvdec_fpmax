"""Tests for configured private post-Intuitive orchestration."""

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import pipeline.private_intuitive as private_intuitive
from pipeline.experiment_context import ExperimentContext
from pipeline.private_without_ffs import PrivateExperimentSpec


def _fixture_spec(tmp_path: Path) -> tuple[PrivateExperimentSpec, Path]:
    """Create one seed-isolated private experiment fixture."""

    data_path = tmp_path / "tiki.csv"
    data_path.write_text("x\n1\n", encoding="utf-8")
    representation_root = tmp_path / "mvdec"
    artifact_path = representation_root / "config" / "seed_44" / "artifact.pkl"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(b"artifact")
    spec = PrivateExperimentSpec(
        dataset="TIKI",
        data_path=data_path,
        representation_root=representation_root,
        output_root=tmp_path / "fpmax",
        seeds=(44,),
        workers=4,
    )
    return spec, artifact_path


def test_configured_intuitive_uses_seed_and_persists_provenance(
    tmp_path,
    monkeypatch,
) -> None:
    """The runner must propagate seed 44 and write an auditable manifest."""

    spec, _artifact_path = _fixture_spec(tmp_path)
    output_dir = spec.output_root / "seed_44"
    output_dir.mkdir(parents=True)
    source_path = output_dir / "ffs_results.csv"
    source_path.write_text("job_index,status\n0,ok\n", encoding="utf-8")
    mvdec_result = SimpleNamespace(
        raw={"dataset": "TIKI", "n_clusters": 5},
        h_fused_df=pd.DataFrame({"fused_1": [0.0, 1.0]}),
    )
    captured = {}

    monkeypatch.setattr(
        private_intuitive,
        "load_experiment_spec",
        lambda *_args: spec,
    )
    monkeypatch.setattr(
        private_intuitive,
        "load_mvdec_result",
        lambda _representation, _data: mvdec_result,
    )

    def fake_context(**kwargs):
        captured["context"] = kwargs
        return ExperimentContext(
            dataset="TIKI",
            n_clusters=5,
            output_dir=kwargs["requested_output_dir"],
        )

    def fake_run(**kwargs):
        captured["run"] = kwargs
        results = pd.DataFrame(
            [{"job_index": 0, "random_state": kwargs["random_state"]}]
        )
        results.to_csv(kwargs["save_path"], index=False)
        return results

    monkeypatch.setattr(private_intuitive, "build_experiment_context", fake_context)
    monkeypatch.setattr(private_intuitive, "run_post_ffs_intuitive", fake_run)

    results = private_intuitive.run_configured_intuitive(
        dataset="TIKI",
        seed=44,
        source="ffs",
        param_workers=3,
    )

    assert len(results) == 1
    assert captured["context"]["random_state"] == 44
    assert captured["run"]["random_state"] == 44
    assert captured["run"]["param_workers"] == 3
    assert captured["run"]["ffs_path"] == source_path
    assert captured["run"]["resume"] is False

    manifest_path = output_dir / "post_ffs_intuitive_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["contract"]["seed"] == 44
    assert manifest["contract"]["source"] == "ffs"
    assert manifest["contract"]["initialization"] == {
        "strategy": "farthest_first",
        "strict": False,
    }
    assert len(manifest["contract"]["parameter_grid"]) > 1
    assert manifest["output"]["row_count"] == 1
    assert manifest["output"]["results_sha256"]


def test_configured_intuitive_resume_rejects_changed_source(
    tmp_path,
    monkeypatch,
) -> None:
    """Resume must fail when the source grid has changed."""

    spec, _artifact_path = _fixture_spec(tmp_path)
    output_dir = spec.output_root / "seed_44"
    output_dir.mkdir(parents=True)
    source_path = output_dir / "ffs_results.csv"
    source_path.write_text("job_index,status\n0,ok\n", encoding="utf-8")
    mvdec_result = SimpleNamespace(
        raw={"dataset": "TIKI", "n_clusters": 5},
        h_fused_df=pd.DataFrame({"fused_1": [0.0, 1.0]}),
    )

    monkeypatch.setattr(
        private_intuitive,
        "load_experiment_spec",
        lambda *_args: spec,
    )
    monkeypatch.setattr(
        private_intuitive,
        "load_mvdec_result",
        lambda _representation, _data: mvdec_result,
    )
    monkeypatch.setattr(
        private_intuitive,
        "build_experiment_context",
        lambda **kwargs: ExperimentContext(
            dataset="TIKI",
            n_clusters=5,
            output_dir=kwargs["requested_output_dir"],
        ),
    )

    def fake_run(**kwargs):
        results = pd.DataFrame([{"job_index": 0}])
        results.to_csv(kwargs["save_path"], index=False)
        return results

    monkeypatch.setattr(private_intuitive, "run_post_ffs_intuitive", fake_run)
    private_intuitive.run_configured_intuitive("TIKI", 44, "ffs")

    source_path.write_text("job_index,status\n0,ok\n1,ok\n", encoding="utf-8")
    with pytest.raises(ValueError, match="contract does not match"):
        private_intuitive.run_configured_intuitive(
            "TIKI",
            44,
            "ffs",
            resume=True,
        )
