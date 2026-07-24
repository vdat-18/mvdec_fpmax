from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pipeline.experiments import (
    ExperimentJob,
    IntuitiveParams,
    NativeIntuitiveContext,
    NativeIntuitiveTrial,
    build_two_stage_intuitive_param_grids,
    default_intuitive_param_grid,
    default_view_weighted_intuitive_param_grid,
    init_native_intuitive_worker,
    run_native_intuitive_forward_selection,
    run_native_intuitive_trial,
    select_native_intuitive_trials,
    trial_selection_score,
)
from pipeline.forward_selection import improves_score, run_forward_selection


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


def test_two_stage_intuitive_search_uses_coarse_then_local_refinement() -> None:
    """Two-stage tuning evaluates a small subset of the full parameter grid."""

    full_grid = default_intuitive_param_grid()
    best_coarse_trial = NativeIntuitiveTrial(
        clustering=SimpleNamespace(score=0.5),
        mu_param=0.5,
        gamma=0.5,
        beta=2.0,
        status="ok",
        error_message=None,
    )

    broad_grid, refined_grid = build_two_stage_intuitive_param_grids(
        full_grid,
        [best_coarse_trial],
    )

    assert len(full_grid) == 324
    assert len(broad_grid) == 36
    assert len(refined_grid) == 8
    assert len(broad_grid) + len(refined_grid) < len(full_grid)


def test_two_stage_intuitive_search_does_not_fallback_to_exhaustive(
    monkeypatch,
) -> None:
    """An invalid coarse stage stops instead of evaluating the remaining grid."""

    batch_sizes = []

    def fake_trials(*, intuitive_param_grid, **_kwargs):
        batch_sizes.append(len(intuitive_param_grid))
        return [
            NativeIntuitiveTrial(
                clustering=None,
                mu_param=None,
                gamma=None,
                beta=None,
                status="failed_all_params",
                error_message="invalid coarse trial",
            )
        ]

    monkeypatch.setattr(
        "pipeline.experiments.run_native_intuitive_trials",
        fake_trials,
    )

    result = select_native_intuitive_trials(
        h_fused_df=pd.DataFrame({"x": [0.0, 1.0]}),
        binary_df=pd.DataFrame({"b": [0, 1]}),
        n_clusters=2,
        random_state=44,
        param_workers=1,
        two_stage=True,
    )

    assert batch_sizes == [36]
    assert result.best_trial.status == "failed_all_params"


def test_view_weighted_two_stage_search_refines_alpha_locally() -> None:
    """View weighting adds alpha to coarse/refine without exhaustive search."""

    full_grid = default_view_weighted_intuitive_param_grid()
    best_coarse_trial = NativeIntuitiveTrial(
        clustering=SimpleNamespace(score=0.5),
        mu_param=0.5,
        gamma=0.5,
        beta=2.0,
        status="ok",
        error_message=None,
        view_weight_alpha=0.5,
    )

    broad_grid, refined_grid = build_two_stage_intuitive_param_grids(
        full_grid,
        [best_coarse_trial],
    )

    assert len(full_grid) == 2916
    assert len(broad_grid) == 108
    assert len(refined_grid) == 26
    assert len(broad_grid) + len(refined_grid) == 134


def test_kprototypes_ffs_rejects_invalid_candidate_workers():
    with pytest.raises(ValueError, match="candidate_workers"):
        run_forward_selection(
            continuous_df=pd.DataFrame({"x": [0.0, 1.0]}),
            binary_df=pd.DataFrame({"b": [0, 1]}),
            n_clusters=2,
            baseline_score=0.0,
            candidate_workers=0,
        )


def test_ffs_requires_strict_score_improvement() -> None:
    """Equal scores stop FFS while any positive improvement remains eligible."""

    assert not improves_score(0.5, 0.5)
    assert improves_score(0.500000000001, 0.5)
