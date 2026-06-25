from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pipeline.experiments import (
    ExperimentJob,
    IntuitiveParams,
    NativeIntuitiveContext,
    NativeIntuitiveTrial,
    init_native_intuitive_worker,
    run_native_intuitive_forward_selection,
    run_native_intuitive_trial,
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
    def raise_value_error(**kwargs):
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
            random_state=42,
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
            candidate_workers=0,
        )

