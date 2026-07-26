"""Tests for configured private without-FFS orchestration."""

import json
from types import SimpleNamespace

import pandas as pd
import pytest

import pipeline.private_without_ffs as private_without_ffs
from pipeline.experiment_context import ExperimentContext
from pipeline.private_without_ffs import (
    PrivateExperimentSpec,
    load_experiment_spec,
    resolve_seed_artifact,
    run_configured_experiment,
)


def _write_config(config_path, seeds: list[int]) -> None:
    """Write one minimal private experiment registry for tests."""

    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "datasets": {
                    "TIKI": {
                        "data_path": "data/tiki.csv",
                        "representation_root": "output/mvdec/tiki",
                        "output_root": "output/tiki/fpmax",
                        "seeds": seeds,
                        "workers": 4,
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_private_experiment_config_resolves_paths_and_seeds(tmp_path) -> None:
    """Repository-relative settings must produce one validated experiment spec."""

    config_path = tmp_path / "private.json"
    _write_config(config_path, [42, 43, 44])

    spec = load_experiment_spec(config_path, "TIKI", project_dir=tmp_path)

    assert spec.data_path == tmp_path / "data/tiki.csv"
    assert spec.representation_root == tmp_path / "output/mvdec/tiki"
    assert spec.output_root == tmp_path / "output/tiki/fpmax"
    assert spec.seeds == (42, 43, 44)
    assert spec.workers == 4


def test_repository_config_defines_airpollution_contract() -> None:
    """The checked-in private registry must expose the full Air experiment."""

    spec = load_experiment_spec(
        private_without_ffs.DEFAULT_CONFIG_PATH,
        "AIRPOLLUTION",
    )

    assert spec.data_path == (
        private_without_ffs.PROJECT_DIR / "data/preprocessed_data/data_demvk.csv"
    )
    assert spec.representation_root == (
        private_without_ffs.PROJECT_DIR
        / "output/mvdec_runs/airpollution/mvdec_dekm_consistent_v1"
    )
    assert spec.output_root == (
        private_without_ffs.PROJECT_DIR / "output/airpollution/fpmax_without_ffs"
    )
    assert spec.seeds == (42, 43, 44)
    assert spec.workers == 4


def test_seed_artifact_requires_exactly_one_match(tmp_path) -> None:
    """Artifact discovery must fail instead of mixing seed outputs."""

    config_path = tmp_path / "private.json"
    _write_config(config_path, [42])
    spec = load_experiment_spec(config_path, "TIKI", project_dir=tmp_path)

    with pytest.raises(FileNotFoundError, match="found 0"):
        resolve_seed_artifact(spec, 42)

    artifact_path = spec.representation_root / "config" / "seed_42" / "artifact.pkl"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(b"artifact")

    assert resolve_seed_artifact(spec, 42) == artifact_path


def test_configured_runner_uses_each_seed_for_downstream_clustering(
    tmp_path,
    monkeypatch,
) -> None:
    """Each MvDEC seed must also control its downstream K-Prototypes run."""

    data_path = tmp_path / "tiki.csv"
    data_path.write_text("x\n1\n", encoding="utf-8")
    representation_root = tmp_path / "mvdec"
    output_root = tmp_path / "fpmax"
    for seed in (42, 43):
        artifact_path = representation_root / "config" / f"seed_{seed}" / "artifact.pkl"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_bytes(b"artifact")
    spec = PrivateExperimentSpec(
        dataset="TIKI",
        data_path=data_path,
        representation_root=representation_root,
        output_root=output_root,
        seeds=(42, 43),
        workers=2,
    )
    calls = []
    mvdec_result = SimpleNamespace(
        raw={"dataset": "TIKI", "n_clusters": 5},
        h_fused_df=pd.DataFrame({"fused_1": [0.0, 1.0]}),
        evaluation_score=0.4,
    )

    monkeypatch.setattr(
        private_without_ffs,
        "load_mvdec_result",
        lambda _representation, _data: mvdec_result,
    )

    def fake_context(**kwargs):
        calls.append(("context", kwargs))
        return ExperimentContext(
            dataset="TIKI",
            n_clusters=5,
            output_dir=kwargs["requested_output_dir"],
        )

    monkeypatch.setattr(private_without_ffs, "build_experiment_context", fake_context)
    monkeypatch.setattr(
        private_without_ffs,
        "run_without_ffs",
        lambda **kwargs: calls.append(("run", kwargs)),
    )

    run_configured_experiment(spec)

    context_calls = [kwargs for kind, kwargs in calls if kind == "context"]
    run_calls = [kwargs for kind, kwargs in calls if kind == "run"]
    assert [call["random_state"] for call in context_calls] == [42, 43]
    assert [call["random_state"] for call in run_calls] == [42, 43]
    assert [call["resume"] for call in run_calls] == [False, False]
    assert [call["workers"] for call in run_calls] == [2, 2]
