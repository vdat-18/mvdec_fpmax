"""Tests for configured private FFS orchestration."""

from types import SimpleNamespace

import pandas as pd

import pipeline.private_ffs as private_ffs
from pipeline.experiment_context import ExperimentContext
from pipeline.private_without_ffs import PrivateExperimentSpec


def test_configured_ffs_uses_selected_seed_and_worker_limits(
    tmp_path,
    monkeypatch,
) -> None:
    """The selected MvDEC seed must control FFS clustering and its manifest."""

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
    mvdec_result = SimpleNamespace(
        raw={"dataset": "TIKI", "n_clusters": 5},
        h_fused_df=pd.DataFrame({"fused_1": [0.0, 1.0]}),
        evaluation_score=0.4,
    )
    captured = {}

    monkeypatch.setattr(private_ffs, "load_experiment_spec", lambda *_args: spec)
    monkeypatch.setattr(
        private_ffs,
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

    monkeypatch.setattr(private_ffs, "build_experiment_context", fake_context)
    monkeypatch.setattr(
        private_ffs,
        "run_ffs",
        lambda **kwargs: captured.update({"run": kwargs}),
    )

    private_ffs.run_configured_ffs(
        dataset="TIKI",
        seed=44,
        workers=3,
        candidate_workers=2,
    )

    assert captured["context"]["random_state"] == 44
    assert captured["run"]["random_state"] == 44
    assert captured["run"]["workers"] == 3
    assert captured["run"]["candidate_workers"] == 2
    assert captured["run"]["resume"] is False
