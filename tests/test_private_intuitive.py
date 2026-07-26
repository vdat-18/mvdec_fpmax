"""Tests for configured private native-Intuitive orchestration."""

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import pipeline.private_intuitive as private_intuitive
from pipeline.intuitive_protocol import (
    DEFAULT_PROTOCOL_CONFIG_PATH,
    PRIMARY_PROTOCOL_ID,
    VIEW_WEIGHTED_PROTOCOL_ID,
    load_intuitive_protocol,
)
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


def test_mimvdec_intuitive_protocol_config_is_versioned_and_valid() -> None:
    """The configured primary protocol must match the implemented behavior."""

    protocol = load_intuitive_protocol(
        DEFAULT_PROTOCOL_CONFIG_PATH,
        PRIMARY_PROTOCOL_ID,
    )

    assert protocol.protocol_id == "mimvdec_intuitive_v1"
    assert protocol.numeric_input == "h_fused"
    assert protocol.numeric_preprocessing == "none"
    assert protocol.selection_metric == (
        "silhouette_gower_numeric_asymmetric_binary_v1"
    )
    assert protocol.zero_phi_policy == "paper_literal_zero_weight"
    assert protocol.without_ffs_mode == "intuitive_native_all_fpmax_features"
    assert protocol.ffs_mode == "intuitive_native_forward_selection"
    assert protocol.parameter_search == "two_stage"
    assert protocol.ffs_min_improvement == 0.0
    assert protocol.fpmax_strategies == ("uniform", "quantile", "kmeans")
    assert protocol.fpmax_n_bins == (3, 5, 7)
    assert len(protocol.fpmax_min_supports) == 19
    assert len(protocol.mu_params) == 9
    assert len(protocol.gammas) == 9
    assert protocol.betas == (2.0, 3.0, 4.0, 5.0)
    assert protocol.view_weight_alphas == ()


def test_view_weighted_protocol_is_isolated_and_versioned() -> None:
    """The comparison protocol must declare alpha without changing the metric."""

    protocol = load_intuitive_protocol(
        DEFAULT_PROTOCOL_CONFIG_PATH,
        VIEW_WEIGHTED_PROTOCOL_ID,
    )
    parameter_grid = private_intuitive._parameter_grid(protocol)

    assert protocol.distance_mode == "view_weighted_attribute_weights_v1"
    assert protocol.selection_metric == (
        "silhouette_gower_numeric_asymmetric_binary_v1"
    )
    assert protocol.without_ffs_mode == (
        "intuitive_view_weighted_native_all_fpmax_features"
    )
    assert protocol.ffs_mode == "intuitive_view_weighted_native_forward_selection"
    assert protocol.parameter_search == "two_stage"
    assert len(protocol.view_weight_alphas) == 9
    assert len(parameter_grid) == 2916
    assert all(params.view_weight_alpha is not None for params in parameter_grid)


def test_protocol_config_rejects_behavior_not_supported_by_runner(tmp_path) -> None:
    """A manifest contract must not claim behavior the runner does not execute."""

    payload = json.loads(DEFAULT_PROTOCOL_CONFIG_PATH.read_text(encoding="utf-8"))
    payload["protocols"][PRIMARY_PROTOCOL_ID]["numeric_preprocessing"] = "minmax"
    config_path = tmp_path / "intuitive_protocols.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported settings"):
        load_intuitive_protocol(config_path, PRIMARY_PROTOCOL_ID)


def test_configured_intuitive_uses_seed_and_persists_provenance(
    tmp_path,
    monkeypatch,
) -> None:
    """The runner must use the MiMvDEC file set and preserve provenance."""

    spec, _artifact_path = _fixture_spec(tmp_path)
    seed_output_dir = spec.output_root / "seed_44"
    mvdec_result = SimpleNamespace(
        raw={"dataset": "TIKI", "n_clusters": 5},
        h_fused_df=pd.DataFrame({"fused_1": [0.0, 1.0]}),
        evaluation_score=0.25,
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

    def fake_run(**kwargs):
        captured["run"] = kwargs
        results = pd.DataFrame(
            [{"job_index": 0, "random_state": kwargs["random_state"]}]
        )
        results.to_csv(kwargs["save_path"], index=False)
        return results

    monkeypatch.setattr(private_intuitive, "run_ffs_intuitive_native", fake_run)

    results = private_intuitive.run_configured_intuitive(
        dataset="TIKI",
        seed=44,
        source="ffs",
        param_workers=3,
    )

    assert len(results) == 1
    assert captured["run"]["random_state"] == 44
    assert captured["run"]["param_workers"] == 3
    assert captured["run"]["resume"] is False
    assert captured["run"]["baseline_score"] == 0.25
    assert captured["run"]["two_stage"] is True
    assert captured["run"]["candidate_workers"] == 1
    assert captured["run"]["persist_trials"] is False
    assert len(captured["run"]["supports"]) == 19
    assert captured["run"]["init_strategy"] == "farthest_first"
    assert captured["run"]["non_membership"] == "paper"
    assert captured["run"]["max_iter"] == 100
    assert len(captured["run"]["intuitive_param_grid"]) == 324

    output_dir = seed_output_dir
    assert {path.name for path in output_dir.iterdir()} == {
        "mimvdec_intuitive_with_ffs_results.csv",
        "mvdec_experiment_manifest.json",
    }
    manifest_path = output_dir / "mvdec_experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    workflow = manifest["workflows"]["mimvdec_intuitive_v1"]
    contract = workflow["contract"]
    assert manifest["schema_version"] == 4
    assert manifest["random_state"] == 44
    assert contract["seed"] == 44
    assert contract["protocol_id"] == "mimvdec_intuitive_v1"
    assert contract["protocol"]["numeric_preprocessing"] == "none"
    assert contract["protocol"]["initialization_strategy"] == ("farthest_first")
    assert contract["protocol"]["strict_initialization"] is False
    assert contract["protocol"]["ffs_mode"] == ("intuitive_native_forward_selection")
    assert contract["baseline_numeric_gower_silhouette"] == 0.25
    assert contract["model_audit_columns"]
    assert len(contract["parameter_grid"]) == 324
    assert workflow["contract_hash"]


def test_configured_intuitive_resume_rejects_changed_artifact(
    tmp_path,
    monkeypatch,
) -> None:
    """Resume must fail when the MvDEC representation has changed."""

    spec, _artifact_path = _fixture_spec(tmp_path)
    mvdec_result = SimpleNamespace(
        raw={"dataset": "TIKI", "n_clusters": 5},
        h_fused_df=pd.DataFrame({"fused_1": [0.0, 1.0]}),
        evaluation_score=0.25,
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

    def fake_run(**kwargs):
        results = pd.DataFrame([{"job_index": 0}])
        results.to_csv(kwargs["save_path"], index=False)
        return results

    monkeypatch.setattr(private_intuitive, "run_ffs_intuitive_native", fake_run)
    private_intuitive.run_configured_intuitive("TIKI", 44, "ffs")

    _artifact_path.write_bytes(b"changed artifact")
    with pytest.raises(ValueError, match="another dataset or artifact"):
        private_intuitive.run_configured_intuitive(
            "TIKI",
            44,
            "ffs",
            resume=True,
        )


def test_without_ffs_runs_native_intuitive_without_kprototypes_source(
    tmp_path,
    monkeypatch,
) -> None:
    """Without-FFS must build FP-Max directly instead of reading K-Prototypes CSV."""

    spec, _artifact_path = _fixture_spec(tmp_path)
    mvdec_result = SimpleNamespace(
        raw={"dataset": "TIKI", "n_clusters": 5},
        h_fused_df=pd.DataFrame({"fused_1": [0.0, 1.0]}),
        evaluation_score=0.25,
    )
    captured = {}

    monkeypatch.setattr(private_intuitive, "load_experiment_spec", lambda *_: spec)
    monkeypatch.setattr(
        private_intuitive,
        "load_mvdec_result",
        lambda *_: mvdec_result,
    )

    def fake_run(**kwargs):
        captured.update(kwargs)
        results = pd.DataFrame([{"job_index": 0, "status": "ok"}])
        results.to_csv(kwargs["save_path"], index=False)
        return results

    monkeypatch.setattr(
        private_intuitive,
        "run_without_ffs_intuitive_native",
        fake_run,
    )

    private_intuitive.run_configured_intuitive("TIKI", 44, "without-ffs")

    assert captured["save_path"].name == ("mimvdec_intuitive_without_ffs_results.csv")
    assert captured["persist_trials"] is False
    assert captured["two_stage"] is True
    assert len(captured["intuitive_param_grid"]) == 324


def test_view_weighted_protocol_dispatches_to_isolated_runner(
    tmp_path,
    monkeypatch,
) -> None:
    """The comparison protocol must use view-weighted outputs and alpha trials."""

    spec, _artifact_path = _fixture_spec(tmp_path)
    mvdec_result = SimpleNamespace(
        raw={"dataset": "TIKI", "n_clusters": 5},
        h_fused_df=pd.DataFrame({"fused_1": [0.0, 1.0]}),
        evaluation_score=0.25,
    )
    captured = {}

    monkeypatch.setattr(private_intuitive, "load_experiment_spec", lambda *_: spec)
    monkeypatch.setattr(private_intuitive, "load_mvdec_result", lambda *_: mvdec_result)

    def fake_run(**kwargs):
        captured.update(kwargs)
        results = pd.DataFrame([{"job_index": 0, "status": "ok"}])
        results.to_csv(kwargs["save_path"], index=False)
        return results

    monkeypatch.setattr(
        private_intuitive,
        "run_without_ffs_intuitive_view_weighted_native",
        fake_run,
    )

    private_intuitive.run_configured_intuitive(
        "TIKI",
        44,
        "without-ffs",
        protocol_id=VIEW_WEIGHTED_PROTOCOL_ID,
    )

    assert captured["save_path"].name == (
        "mimvdec_intuitive_view_weighted_without_ffs_results.csv"
    )
    assert captured["persist_trials"] is False
    assert captured["two_stage"] is True
    assert len(captured["intuitive_param_grid"]) == 2916
    assert {
        params.view_weight_alpha for params in captured["intuitive_param_grid"]
    } == {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}
    manifest_path = spec.output_root / "seed_44" / "mvdec_experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = manifest["workflows"][VIEW_WEIGHTED_PROTOCOL_ID]["contract"]
    assert contract["protocol_id"] == VIEW_WEIGHTED_PROTOCOL_ID
    assert contract["protocol"]["view_weight_alphas"]


def test_configured_intuitive_rejects_multiple_parallel_axes() -> None:
    """One process hierarchy at a time prevents nested worker oversubscription."""

    with pytest.raises(ValueError, match="only one parallel axis"):
        private_intuitive.run_configured_intuitive(
            "TIKI",
            44,
            "ffs",
            workers=2,
            param_workers=2,
        )
