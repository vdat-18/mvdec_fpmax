from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pipeline.experiments import (
    ExperimentJob,
    IntuitiveParams,
    NativeIntuitiveContext,
    NativeIntuitiveTrial,
    PostFfsIntuitiveContext,
    init_native_intuitive_worker,
    init_post_ffs_intuitive_worker,
    run_native_intuitive_forward_selection,
    run_native_intuitive_trial,
    run_post_ffs_intuitive_job,
    trial_selection_score,
)
from pipeline.forward_selection import run_forward_selection


def test_trial_selection_score_uses_common_mixed_gower_score():
    trial = NativeIntuitiveTrial(
        clustering=SimpleNamespace(score=0.9),
        mu_param=0.1,
        gamma=0.2,
        beta=2.0,
        status="ok",
        error_message=None,
        view_weight_alpha=0.5,
        view_weighted_score=0.4,
    )

    assert trial_selection_score(trial) == 0.9


def test_trial_selection_score_falls_back_to_standard_score():
    trial = NativeIntuitiveTrial(
        clustering=SimpleNamespace(score=0.9),
        mu_param=0.1,
        gamma=0.2,
        beta=2.0,
        status="ok",
        error_message=None,
    )

    assert trial_selection_score(trial) == 0.9


def test_trial_selection_score_failed_trial_is_negative_infinity():
    trial = NativeIntuitiveTrial(
        clustering=None,
        mu_param=None,
        gamma=None,
        beta=None,
        status="failed_all_params",
        error_message="failed",
    )

    assert trial_selection_score(trial) == float("-inf")


def test_native_intuitive_trial_converts_value_error_to_failed_trial(monkeypatch):
    captured = {}

    def raise_value_error(**kwargs):
        captured.update(kwargs)
        raise ValueError("Algorithm 1 could not find enough prototypes.")

    monkeypatch.setattr(
        "pipeline.experiments.run_intuitive_kprototypes",
        raise_value_error,
    )
    init_native_intuitive_worker(
        NativeIntuitiveContext(
            h_fused_df=pd.DataFrame({"x": [0.0, 1.0]}),
            binary_df=pd.DataFrame({"b": [0, 1]}),
            distance_matrix=np.zeros((2, 2)),
            n_clusters=2,
            random_state=44,
            init_strategy="farthest_first",
            strict_init=False,
            non_membership="paper",
            max_iter=100,
            empty_cluster_policy="raise",
            min_cluster_size=1,
        )
    )

    trial = run_native_intuitive_trial(
        ExperimentJob(
            job_index=0,
            min_support=0.2,
            intuitive_params=IntuitiveParams(mu_param=0.1, gamma=0.2, beta=2.0),
        )
    )

    assert trial.clustering is None
    assert trial.status == "failed_value_error"
    assert "could not find enough prototypes" in trial.error_message
    assert captured["random_state"] == 44
    assert captured["non_membership"] == "paper"
    assert captured["max_iter"] == 100
    assert captured["empty_cluster_policy"] == "raise"


def test_post_ffs_job_propagates_versioned_protocol_settings(monkeypatch) -> None:
    """The configured protocol must control the underlying model call."""

    captured = {}

    def raise_empty_cluster(**kwargs):
        captured.update(kwargs)
        raise ValueError("Empty clusters are not defined by the paper: [1].")

    monkeypatch.setattr(
        "pipeline.experiments.run_intuitive_kprototypes",
        raise_empty_cluster,
    )
    init_post_ffs_intuitive_worker(
        PostFfsIntuitiveContext(
            h_fused_df=pd.DataFrame({"x": [0.0, 1.0]}),
            binary_df=pd.DataFrame({"b": [0, 1]}),
            distance_matrix=np.zeros((2, 2)),
            view_weighted_distance_matrices=None,
            ffs_job_index=3,
            strategy="quantile",
            n_bins=3,
            min_support=0.1,
            ffs_score=0.2,
            selected_features="b",
            n_selected_features=1,
            n_clusters=2,
            random_state=44,
            protocol_id="mimvdec_intuitive_v1",
            init_strategy="farthest_first",
            strict_init=False,
            non_membership="paper",
            max_iter=100,
            empty_cluster_policy="raise",
            min_cluster_size=1,
        )
    )

    record = run_post_ffs_intuitive_job(
        ExperimentJob(
            job_index=0,
            min_support=0.1,
            intuitive_params=IntuitiveParams(mu_param=0.5, gamma=0.5, beta=2.0),
        )
    )

    assert record.protocol_id == "mimvdec_intuitive_v1"
    assert captured["random_state"] == 44
    assert captured["init_strategy"] == "farthest_first"
    assert captured["non_membership"] == "paper"
    assert captured["max_iter"] == 100
    assert captured["empty_cluster_policy"] == "raise"


def test_native_intuitive_ffs_rejects_nested_inner_parallelism():
    with pytest.raises(ValueError, match="candidate_workers or param_workers"):
        run_native_intuitive_forward_selection(
            continuous_df=pd.DataFrame({"x": [0.0, 1.0]}),
            binary_df=pd.DataFrame({"b": [0, 1]}),
            n_clusters=2,
            random_state=42,
            baseline_score=0.0,
            param_workers=2,
            candidate_workers=2,
        )


def test_kprototypes_ffs_rejects_invalid_candidate_workers():
    with pytest.raises(ValueError, match="candidate_workers"):
        run_forward_selection(
            continuous_df=pd.DataFrame({"x": [0.0, 1.0]}),
            binary_df=pd.DataFrame({"b": [0, 1]}),
            n_clusters=2,
            baseline_score=0.0,
            candidate_workers=0,
        )
