"""Experiment runners for MiMvDEC with and without forward feature selection."""

from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, replace
from functools import partial
from itertools import product
from pathlib import Path
from queue import Empty
from time import sleep
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from config import (
    FFS_INTUITIVE_EXHAUSTIVE_NATIVE_RESULTS_PATH,
    FFS_INTUITIVE_NATIVE_RESULTS_PATH,
    FFS_INTUITIVE_PAPER_EXHAUSTIVE_NATIVE_RESULTS_PATH,
    FFS_INTUITIVE_PAPER_NATIVE_RESULTS_PATH,
    FFS_INTUITIVE_VIEW_WEIGHTED_EXHAUSTIVE_NATIVE_RESULTS_PATH,
    FFS_INTUITIVE_VIEW_WEIGHTED_NATIVE_RESULTS_PATH,
    FFS_INTUITIVE_VIEW_WEIGHTED_PAPER_EXHAUSTIVE_NATIVE_RESULTS_PATH,
    FFS_INTUITIVE_VIEW_WEIGHTED_PAPER_NATIVE_RESULTS_PATH,
    FFS_RESULTS_PATH,
    POST_FFS_INTUITIVE_BEST_RESULTS_PATH,
    POST_FFS_INTUITIVE_EXHAUSTIVE_BEST_RESULTS_PATH,
    POST_FFS_INTUITIVE_RESULTS_PATH,
    POST_FFS_INTUITIVE_VIEW_WEIGHTED_BEST_RESULTS_PATH,
    POST_FFS_INTUITIVE_VIEW_WEIGHTED_EXHAUSTIVE_BEST_RESULTS_PATH,
    POST_FFS_INTUITIVE_VIEW_WEIGHTED_RESULTS_PATH,
    POST_WITHOUT_FFS_INTUITIVE_BEST_RESULTS_PATH,
    POST_WITHOUT_FFS_INTUITIVE_EXHAUSTIVE_BEST_RESULTS_PATH,
    POST_WITHOUT_FFS_INTUITIVE_RESULTS_PATH,
    POST_WITHOUT_FFS_INTUITIVE_VIEW_WEIGHTED_BEST_RESULTS_PATH,
    POST_WITHOUT_FFS_INTUITIVE_VIEW_WEIGHTED_EXHAUSTIVE_BEST_RESULTS_PATH,
    POST_WITHOUT_FFS_INTUITIVE_VIEW_WEIGHTED_RESULTS_PATH,
    RANDOM_STATE,
    WITHOUT_FFS_INTUITIVE_EXHAUSTIVE_NATIVE_RESULTS_PATH,
    WITHOUT_FFS_INTUITIVE_NATIVE_RESULTS_PATH,
    WITHOUT_FFS_INTUITIVE_NATIVE_TRIALS_PATH,
    WITHOUT_FFS_INTUITIVE_PAPER_EXHAUSTIVE_NATIVE_RESULTS_PATH,
    WITHOUT_FFS_INTUITIVE_PAPER_NATIVE_RESULTS_PATH,
    WITHOUT_FFS_INTUITIVE_VIEW_WEIGHTED_EXHAUSTIVE_NATIVE_RESULTS_PATH,
    WITHOUT_FFS_INTUITIVE_VIEW_WEIGHTED_NATIVE_RESULTS_PATH,
    WITHOUT_FFS_INTUITIVE_VIEW_WEIGHTED_PAPER_EXHAUSTIVE_NATIVE_RESULTS_PATH,
    WITHOUT_FFS_INTUITIVE_VIEW_WEIGHTED_PAPER_NATIVE_RESULTS_PATH,
    WITHOUT_FFS_KPROTOTYPES_RESULTS_PATH,
)
from pipeline.clustering import (
    MixedClusteringResult,
    cluster_sizes,
    compute_gower_distance,
    compute_view_weighted_gower_distance,
    make_mixed_features,
    run_intuitive_kprototypes,
    run_kprototypes,
)
from pipeline.forward_selection import run_forward_selection
from pipeline.fpmax import (
    BIN_LABELS_BY_SIZE,
    DEFAULT_STRATEGIES,
    DEFAULT_SUPPORTS,
    DiscretizeStrategy,
    FpmaxFeatures,
    extract_fpmax_features,
)
from pipeline.io import (
    empty_results,
    is_job_done,
    load_results,
    result_csv_path,
    save_results,
)

# Two-stage Intuitive grid: coarse values cover low/balanced/high settings;
# local refinement uses +/-0.1 around the best coarse alpha/mu/gamma value.
INTUITIVE_COARSE_MU_PARAMS = (0.2, 0.5, 0.8)
INTUITIVE_COARSE_GAMMAS = (0.2, 0.5, 0.8)
INTUITIVE_MU_PARAMS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
INTUITIVE_GAMMAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
INTUITIVE_BETAS = (2.0, 3.0, 4.0, 5.0)
INTUITIVE_VIEW_WEIGHT_ALPHAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
INTUITIVE_BROAD_MU_PARAMS = INTUITIVE_COARSE_MU_PARAMS
INTUITIVE_BROAD_GAMMAS = INTUITIVE_COARSE_GAMMAS
INTUITIVE_BROAD_BETAS = INTUITIVE_BETAS
INTUITIVE_BROAD_VIEW_WEIGHT_ALPHAS = (0.2, 0.5, 0.8)
INTUITIVE_TWO_STAGE_TOP_K = 1

WITHOUT_FFS_COLUMNS = [
    "job_index",
    "strategy",
    "n_bins",
    "min_support",
    "silhouette_score",
    "n_selected_features",
    "selected_features",
    "init",
    "cluster_sizes",
    "status",
    "error_message",
]

FFS_COLUMNS = [
    "job_index",
    "strategy",
    "n_bins",
    "min_support",
    "final_score",
    "n_selected_features",
    "selected_features",
    "init",
    "cluster_sizes",
    "status",
    "error_message",
]

WITHOUT_FFS_INTUITIVE_NATIVE_COLUMNS = [
    "job_index",
    "strategy",
    "n_bins",
    "min_support",
    "silhouette_score",
    "n_selected_features",
    "selected_features",
    "cluster_sizes",
    "mu_param",
    "gamma",
    "beta",
    "status",
    "error_message",
]

WITHOUT_FFS_INTUITIVE_NATIVE_TRIAL_COLUMNS = [
    "job_index",
    "strategy",
    "n_bins",
    "min_support",
    "selection_step",
    "candidate_feature",
    "candidate_features",
    "trial_index",
    "stage",
    "silhouette_score",
    "n_selected_features",
    "selected_features",
    "cluster_sizes",
    "alpha",
    "mu_param",
    "gamma",
    "beta",
    "status",
    "error_message",
    "is_best",
]

NATIVE_INTUITIVE_TRIAL_COLUMNS = WITHOUT_FFS_INTUITIVE_NATIVE_TRIAL_COLUMNS

WITHOUT_FFS_INTUITIVE_VIEW_WEIGHTED_NATIVE_COLUMNS = [
    "job_index",
    "strategy",
    "n_bins",
    "min_support",
    "final_score",
    "score_delta",
    "n_selected_features",
    "selected_features",
    "cluster_sizes",
    "alpha",
    "mu_param",
    "gamma",
    "beta",
    "status",
    "error_message",
]

FFS_INTUITIVE_NATIVE_COLUMNS = [
    "job_index",
    "strategy",
    "n_bins",
    "min_support",
    "final_score",
    "score_delta",
    "n_selected_features",
    "selected_features",
    "cluster_sizes",
    "mu_param",
    "gamma",
    "beta",
    "status",
    "error_message",
]

FFS_INTUITIVE_VIEW_WEIGHTED_NATIVE_COLUMNS = [
    "job_index",
    "strategy",
    "n_bins",
    "min_support",
    "final_score",
    "score_delta",
    "n_selected_features",
    "selected_features",
    "cluster_sizes",
    "alpha",
    "mu_param",
    "gamma",
    "beta",
    "status",
    "error_message",
]

POST_FFS_INTUITIVE_COLUMNS = [
    "job_index",
    "ffs_job_index",
    "strategy",
    "n_bins",
    "min_support",
    "ffs_score",
    "intuitive_score",
    "score_delta",
    "n_selected_features",
    "selected_features",
    "cluster_sizes",
    "mu_param",
    "gamma",
    "beta",
    "status",
    "error_message",
]

POST_FFS_INTUITIVE_VIEW_WEIGHTED_COLUMNS = [
    "job_index",
    "ffs_job_index",
    "strategy",
    "n_bins",
    "min_support",
    "ffs_score",
    "intuitive_score",
    "score_delta",
    "n_selected_features",
    "selected_features",
    "cluster_sizes",
    "alpha",
    "mu_param",
    "gamma",
    "beta",
    "status",
    "error_message",
]

POST_WITHOUT_FFS_INTUITIVE_COLUMNS = [
    "job_index",
    "without_ffs_job_index",
    "strategy",
    "n_bins",
    "min_support",
    "kprototypes_score",
    "intuitive_score",
    "score_delta",
    "n_selected_features",
    "selected_features",
    "cluster_sizes",
    "mu_param",
    "gamma",
    "beta",
    "status",
    "error_message",
]

POST_WITHOUT_FFS_INTUITIVE_VIEW_WEIGHTED_COLUMNS = [
    "job_index",
    "without_ffs_job_index",
    "strategy",
    "n_bins",
    "min_support",
    "kprototypes_score",
    "intuitive_score",
    "score_delta",
    "n_selected_features",
    "selected_features",
    "cluster_sizes",
    "alpha",
    "mu_param",
    "gamma",
    "beta",
    "status",
    "error_message",
]


POST_FFS_INTUITIVE_BEST_COLUMNS = [
    "job_index",
    "ffs_job_index",
    "strategy",
    "n_bins",
    "min_support",
    "ffs_score",
    "intuitive_score",
    "score_delta",
    "n_selected_features",
    "selected_features",
    "cluster_sizes",
    "mu_param",
    "gamma",
    "beta",
    "status",
    "error_message",
]

POST_FFS_INTUITIVE_VIEW_WEIGHTED_BEST_COLUMNS = [
    "job_index",
    "ffs_job_index",
    "strategy",
    "n_bins",
    "min_support",
    "ffs_score",
    "intuitive_score",
    "score_delta",
    "n_selected_features",
    "selected_features",
    "cluster_sizes",
    "alpha",
    "mu_param",
    "gamma",
    "beta",
    "status",
    "error_message",
]

POST_WITHOUT_FFS_INTUITIVE_BEST_COLUMNS = [
    "job_index",
    "without_ffs_job_index",
    "strategy",
    "n_bins",
    "min_support",
    "kprototypes_score",
    "intuitive_score",
    "score_delta",
    "n_selected_features",
    "selected_features",
    "cluster_sizes",
    "mu_param",
    "gamma",
    "beta",
    "status",
    "error_message",
]

POST_WITHOUT_FFS_INTUITIVE_VIEW_WEIGHTED_BEST_COLUMNS = [
    "job_index",
    "without_ffs_job_index",
    "strategy",
    "n_bins",
    "min_support",
    "kprototypes_score",
    "intuitive_score",
    "score_delta",
    "n_selected_features",
    "selected_features",
    "cluster_sizes",
    "alpha",
    "mu_param",
    "gamma",
    "beta",
    "status",
    "error_message",
]
@dataclass(frozen=True)
class IntuitiveParams:
    """One Intuitive-K-prototypes parameter combination."""

    mu_param: float
    gamma: float
    beta: float
    view_weight_alpha: float | None = None


@dataclass(frozen=True)
class ExperimentJob:
    """One support value within a strategy and bin group."""

    job_index: int
    min_support: float
    intuitive_params: IntuitiveParams | None = None


@dataclass(frozen=True)
class ExperimentGroup:
    """One worker group sharing a strategy and bin count."""

    group_index: int
    strategy: DiscretizeStrategy
    n_bins: int
    jobs: tuple[ExperimentJob, ...]


@dataclass(frozen=True)
class WithoutFfsRecord:
    """One without-FFS K-Prototypes result row."""

    job_index: int
    strategy: str
    n_bins: int
    min_support: float
    silhouette_score: float
    n_selected_features: int
    selected_features: str
    init: str | None
    cluster_sizes: list[int]
    status: str
    error_message: str | None



@dataclass(frozen=True)
class FfsRecord:
    """One FFS K-Prototypes result row."""

    job_index: int
    strategy: str
    n_bins: int
    min_support: float
    final_score: float
    n_selected_features: int
    selected_features: str
    init: str | None
    cluster_sizes: list[int]
    status: str
    error_message: str | None


@dataclass(frozen=True)
class NativeIntuitiveTrialRecord:
    """One persisted Intuitive parameter trial for a native experiment job."""

    job_index: int
    strategy: str
    n_bins: int
    min_support: float
    selection_step: int | None
    candidate_feature: str | None
    candidate_features: str | None
    trial_index: int
    stage: str
    silhouette_score: float
    view_weighted_score: float | None
    n_selected_features: int
    selected_features: str
    cluster_sizes: list[int]
    alpha: float | None
    mu_param: float | None
    gamma: float | None
    beta: float | None
    status: str
    error_message: str | None
    is_best: bool

@dataclass(frozen=True)
class WithoutFfsIntuitiveNativeRecord:
    """One without-FFS FP-Max support configuration clustered by Intuitive."""

    job_index: int
    strategy: str
    n_bins: int
    min_support: float
    silhouette_score: float
    n_selected_features: int
    selected_features: str
    cluster_sizes: list[int]
    mu_param: float | None
    gamma: float | None
    beta: float | None
    status: str
    error_message: str | None
    trial_records: tuple[NativeIntuitiveTrialRecord, ...] = ()

@dataclass(frozen=True)
class WithoutFfsIntuitiveViewWeightedNativeRecord:
    """One view-weighted Intuitive result for a without-FFS config."""

    job_index: int
    strategy: str
    n_bins: int
    min_support: float
    final_score: float
    view_weighted_score: float
    score_delta: float
    n_selected_features: int
    selected_features: str
    cluster_sizes: list[int]
    alpha: float | None
    mu_param: float | None
    gamma: float | None
    beta: float | None
    status: str
    error_message: str | None
    trial_records: tuple[NativeIntuitiveTrialRecord, ...] = ()

@dataclass(frozen=True)
class FfsIntuitiveNativeRecord:
    """One Intuitive-native FFS result row."""

    job_index: int
    strategy: str
    n_bins: int
    min_support: float
    final_score: float
    score_delta: float
    n_selected_features: int
    selected_features: str
    cluster_sizes: list[int]
    mu_param: float | None
    gamma: float | None
    beta: float | None
    status: str
    error_message: str | None
    trial_records: tuple[NativeIntuitiveTrialRecord, ...] = ()

@dataclass(frozen=True)
class FfsIntuitiveViewWeightedNativeRecord:
    """One view-weighted Intuitive-native FFS result row."""

    job_index: int
    strategy: str
    n_bins: int
    min_support: float
    final_score: float
    view_weighted_score: float
    score_delta: float
    n_selected_features: int
    selected_features: str
    cluster_sizes: list[int]
    alpha: float | None
    mu_param: float | None
    gamma: float | None
    beta: float | None
    status: str
    error_message: str | None
    trial_records: tuple[NativeIntuitiveTrialRecord, ...] = ()


@dataclass(frozen=True)
class PostFfsIntuitiveRecord:
    """One Intuitive run on a feature set selected by FFS."""

    job_index: int
    ffs_job_index: int
    strategy: str
    n_bins: int
    min_support: float
    ffs_score: float
    intuitive_score: float
    score_delta: float
    n_selected_features: int
    selected_features: str
    cluster_sizes: list[int]
    mu_param: float
    gamma: float
    beta: float
    status: str
    error_message: str | None
    alpha: float | None = None
    view_weighted_score: float | None = None


@dataclass(frozen=True)
class PostFfsIntuitiveContext:
    """Shared inputs for Intuitive runs on one selected FFS feature set."""

    h_fused_df: pd.DataFrame
    binary_df: pd.DataFrame
    distance_matrix: np.ndarray
    view_weighted_distance_matrices: dict[float, np.ndarray] | None
    ffs_job_index: int
    strategy: str
    n_bins: int
    min_support: float
    ffs_score: float
    selected_features: str
    n_selected_features: int
    n_clusters: int
    random_state: int

@dataclass(frozen=True)
class PostWithoutFfsIntuitiveRecord:
    """One Intuitive run on one K-Prototypes without-FFS feature set."""

    job_index: int
    without_ffs_job_index: int
    strategy: str
    n_bins: int
    min_support: float
    kprototypes_score: float
    intuitive_score: float
    score_delta: float
    n_selected_features: int
    selected_features: str
    cluster_sizes: list[int]
    mu_param: float
    gamma: float
    beta: float
    status: str
    error_message: str | None
    alpha: float | None = None
    view_weighted_score: float | None = None


@dataclass(frozen=True)
class PostWithoutFfsIntuitiveContext:
    """Shared inputs for Intuitive runs on one without-FFS feature set."""

    h_fused_df: pd.DataFrame
    binary_df: pd.DataFrame
    distance_matrix: np.ndarray
    view_weighted_distance_matrices: dict[float, np.ndarray] | None
    without_ffs_job_index: int
    strategy: str
    n_bins: int
    min_support: float
    kprototypes_score: float
    selected_features: str
    n_selected_features: int
    n_clusters: int
    random_state: int


@dataclass(frozen=True)
class PostFfsIntuitiveBestRecord:
    """Best Intuitive result on one selected FFS feature set."""

    job_index: int
    ffs_job_index: int
    strategy: str
    n_bins: int
    min_support: float
    ffs_score: float
    intuitive_score: float
    score_delta: float
    n_selected_features: int
    selected_features: str
    cluster_sizes: list[int]
    mu_param: float | None
    gamma: float | None
    beta: float | None
    status: str
    error_message: str | None
    alpha: float | None = None
    view_weighted_score: float | None = None

@dataclass(frozen=True)
class PostWithoutFfsIntuitiveBestRecord:
    """Best Intuitive result on one without-FFS K-Prototypes feature set."""

    job_index: int
    without_ffs_job_index: int
    strategy: str
    n_bins: int
    min_support: float
    kprototypes_score: float
    intuitive_score: float
    score_delta: float
    n_selected_features: int
    selected_features: str
    cluster_sizes: list[int]
    mu_param: float | None
    gamma: float | None
    beta: float | None
    status: str
    error_message: str | None
    alpha: float | None = None
    view_weighted_score: float | None = None
@dataclass(frozen=True)
class NativeIntuitiveContext:
    """Shared inputs for one native Intuitive parameter sweep."""

    h_fused_df: pd.DataFrame
    binary_df: pd.DataFrame
    distance_matrix: np.ndarray
    n_clusters: int
    random_state: int
    init_strategy: str = "farthest_first"
    strict_init: bool = False
    view_weighted_distance_matrices: dict[float, np.ndarray] | None = None


@dataclass(frozen=True)
class NativeIntuitiveTrial:
    """One Intuitive parameter trial on a fixed mixed feature set."""

    clustering: MixedClusteringResult | None
    mu_param: float | None
    gamma: float | None
    beta: float | None
    status: str
    error_message: str | None
    view_weight_alpha: float | None = None
    view_weighted_score: float | None = None
    stage: str = ""

@dataclass(frozen=True)
class NativeIntuitiveSelectionResult:
    """Best native Intuitive trial plus all parameter trials evaluated."""

    best_trial: NativeIntuitiveTrial
    trials: tuple[NativeIntuitiveTrial, ...]

@dataclass(frozen=True)
class NativeIntuitiveCandidateTrace:
    """All parameter trials for one FFS candidate feature set."""

    selection_step: int
    candidate_feature: str
    candidate_feature_names: tuple[str, ...]
    trials: tuple[NativeIntuitiveTrial, ...]
    best_trial: NativeIntuitiveTrial

@dataclass(frozen=True)
class NativeIntuitiveCandidateJob:
    """One FFS candidate feature set to evaluate."""

    selection_step: int
    candidate_feature: str
    candidate_feature_names: tuple[str, ...]

@dataclass(frozen=True)
class NativeIntuitiveCandidateResult:
    """Evaluation result for one FFS candidate feature set."""

    candidate_feature: str
    score: float
    trial: NativeIntuitiveTrial
    trace: NativeIntuitiveCandidateTrace

@dataclass(frozen=True)
class NativeIntuitiveCandidateContext:
    """Shared inputs for candidate-level FFS evaluation."""

    continuous_df: pd.DataFrame
    binary_df: pd.DataFrame
    n_clusters: int
    random_state: int
    param_workers: int
    intuitive_param_grid: tuple[IntuitiveParams, ...]
    two_stage: bool
    init_strategy: str
    strict_init: bool

@dataclass(frozen=True)
class NativeIntuitiveForwardSelectionResult:
    """Greedy feature selection result using Intuitive as the evaluator."""

    selected_feature_names: list[str]
    score: float
    trial: NativeIntuitiveTrial | None
    status: str
    error_message: str | None
    candidate_traces: tuple[NativeIntuitiveCandidateTrace, ...] = ()

def trial_selection_score(trial: NativeIntuitiveTrial) -> float:
    """Return the common mixed-Gower score optimized across Intuitive trials."""

    if trial.clustering is None:
        return -np.inf
    return float(trial.clustering.score)


ExperimentRecord = (
    WithoutFfsRecord
    | FfsRecord
    | WithoutFfsIntuitiveNativeRecord
    | WithoutFfsIntuitiveViewWeightedNativeRecord
    | FfsIntuitiveNativeRecord
    | FfsIntuitiveViewWeightedNativeRecord
)

JobRunner = Callable[
    [pd.DataFrame, ExperimentGroup, ExperimentJob, int, int, float, int],
    ExperimentRecord,
]
GroupRunner = Callable[
    [pd.DataFrame, ExperimentGroup, int, int, float, JobRunner, int, Any | None],
    list[ExperimentRecord],
]


def default_intuitive_param_grid() -> tuple[IntuitiveParams, ...]:
    """Return the configured Intuitive-K-prototypes parameter grid."""

    return tuple(
        IntuitiveParams(
            mu_param=mu_param,
            gamma=gamma,
            beta=beta,
        )
        for mu_param, gamma, beta in product(
            INTUITIVE_MU_PARAMS,
            INTUITIVE_GAMMAS,
            INTUITIVE_BETAS,
        )
    )

def default_view_weighted_intuitive_param_grid() -> tuple[IntuitiveParams, ...]:
    """Return the Intuitive grid extended with MVDEC/FP-Max view weights."""

    return tuple(
        IntuitiveParams(
            mu_param=mu_param,
            gamma=gamma,
            beta=beta,
            view_weight_alpha=alpha,
        )
        for alpha, mu_param, gamma, beta in product(
            INTUITIVE_VIEW_WEIGHT_ALPHAS,
            INTUITIVE_MU_PARAMS,
            INTUITIVE_GAMMAS,
            INTUITIVE_BETAS,
        )
    )

def dedupe_intuitive_param_grid(
    params: tuple[IntuitiveParams, ...],
) -> tuple[IntuitiveParams, ...]:
    """Return params in first-seen order without duplicate combinations."""

    return tuple(dict.fromkeys(params))

def default_broad_intuitive_param_grid() -> tuple[IntuitiveParams, ...]:
    """Return the coarse first-stage Intuitive grid."""

    return tuple(
        IntuitiveParams(
            mu_param=mu_param,
            gamma=gamma,
            beta=beta,
        )
        for mu_param, gamma, beta in product(
            INTUITIVE_BROAD_MU_PARAMS,
            INTUITIVE_BROAD_GAMMAS,
            INTUITIVE_BROAD_BETAS,
        )
    )

def default_broad_view_weighted_intuitive_param_grid() -> tuple[IntuitiveParams, ...]:
    """Return the coarse first-stage view-weighted Intuitive grid."""

    return tuple(
        IntuitiveParams(
            mu_param=mu_param,
            gamma=gamma,
            beta=beta,
            view_weight_alpha=alpha,
        )
        for alpha, mu_param, gamma, beta in product(
            INTUITIVE_BROAD_VIEW_WEIGHT_ALPHAS,
            INTUITIVE_BROAD_MU_PARAMS,
            INTUITIVE_BROAD_GAMMAS,
            INTUITIVE_BROAD_BETAS,
        )
    )

def neighbor_values(values: tuple[float, ...], center: float) -> tuple[float, ...]:
    """Return center and immediate grid neighbors from a sorted value list."""

    sorted_values = tuple(sorted(values))
    center_index = sorted_values.index(center)
    start = max(0, center_index - 1)
    stop = min(len(sorted_values), center_index + 2)
    return sorted_values[start:stop]

def build_refined_intuitive_param_grid(
    center: IntuitiveParams,
    full_grid: tuple[IntuitiveParams, ...],
) -> tuple[IntuitiveParams, ...]:
    """Build a local refinement grid around one promising parameter combo."""

    mu_values = tuple({params.mu_param for params in full_grid})
    gamma_values = tuple({params.gamma for params in full_grid})
    mu_neighbors = neighbor_values(mu_values, center.mu_param)
    gamma_neighbors = neighbor_values(gamma_values, center.gamma)

    if center.view_weight_alpha is None:
        return tuple(
            IntuitiveParams(mu_param=mu_param, gamma=gamma, beta=beta)
            for mu_param, gamma, beta in product(
                mu_neighbors,
                gamma_neighbors,
                (center.beta,),
            )
        )

    return tuple(
        IntuitiveParams(
            mu_param=mu_param,
            gamma=gamma,
            beta=beta,
            view_weight_alpha=alpha,
        )
        for alpha, mu_param, gamma, beta in product(
            neighbor_values(
                tuple({params.view_weight_alpha for params in full_grid}),
                center.view_weight_alpha,
            ),
            mu_neighbors,
            gamma_neighbors,
            (center.beta,),
        )
    )

def build_two_stage_intuitive_param_grids(
    full_grid: tuple[IntuitiveParams, ...],
    top_trials: list[NativeIntuitiveTrial],
) -> tuple[tuple[IntuitiveParams, ...], tuple[IntuitiveParams, ...]]:
    """Return broad and refined grids for native Intuitive tuning."""

    uses_view_weights = any(
        params.view_weight_alpha is not None for params in full_grid
    )
    broad_candidates = (
        default_broad_view_weighted_intuitive_param_grid()
        if uses_view_weights
        else default_broad_intuitive_param_grid()
    )
    full_set = set(full_grid)
    broad_grid = tuple(params for params in broad_candidates if params in full_set)
    if not broad_grid or len(broad_grid) >= len(full_grid):
        return full_grid, ()

    refine_params: list[IntuitiveParams] = []
    for trial in top_trials[:INTUITIVE_TWO_STAGE_TOP_K]:
        if trial.mu_param is None or trial.gamma is None or trial.beta is None:
            continue

        refine_params.extend(
            build_refined_intuitive_param_grid(
                IntuitiveParams(
                    mu_param=trial.mu_param,
                    gamma=trial.gamma,
                    beta=trial.beta,
                    view_weight_alpha=trial.view_weight_alpha,
                ),
                full_grid,
            )
        )

    refined_grid = tuple(
        params for params in dedupe_intuitive_param_grid(tuple(refine_params))
        if params in full_set and params not in set(broad_grid)
    )
    return broad_grid, refined_grid

def build_view_weighted_distance_matrices(
    h_fused_df: pd.DataFrame,
    binary_df: pd.DataFrame,
    intuitive_param_grid: tuple[IntuitiveParams, ...],
) -> dict[float, np.ndarray] | None:
    """Precompute one validation distance matrix per configured view weight."""

    alphas = sorted(
        {
            params.view_weight_alpha
            for params in intuitive_param_grid
            if params.view_weight_alpha is not None
        }
    )
    if not alphas:
        return None

    return {
        alpha: compute_view_weighted_gower_distance(h_fused_df, binary_df, alpha)
        for alpha in alphas
    }

def parse_selected_features(value: Any) -> list[str]:
    """Parse the persisted comma-separated selected feature names."""

    if not isinstance(value, str) or not value.strip():
        return []
    return [feature.strip() for feature in value.split(",") if feature.strip()]

def canonical_feature_name(feature_name: str) -> str:
    """Return an order-stable key for one FP-Max itemset feature name."""

    return "+".join(sorted(item.strip() for item in feature_name.split("+")))

def resolve_rebuilt_feature_names(
    selected_feature_names: list[str],
    rebuilt_feature_names: list[str],
) -> list[str]:
    """Map persisted FFS feature names to rebuilt FP-Max column names."""

    rebuilt_by_key = {
        canonical_feature_name(feature_name): feature_name
        for feature_name in rebuilt_feature_names
    }
    missing_features = [
        feature_name
        for feature_name in selected_feature_names
        if canonical_feature_name(feature_name) not in rebuilt_by_key
    ]
    if missing_features:
        msg = (
            "Selected FFS features are missing from rebuilt FP-Max output: "
            f"{missing_features}."
        )
        raise ValueError(msg)

    return [
        rebuilt_by_key[canonical_feature_name(feature_name)]
        for feature_name in selected_feature_names
    ]

def load_usable_ffs_rows(ffs_path: Path = FFS_RESULTS_PATH) -> pd.DataFrame:
    """Load FFS rows that can be followed by Intuitive evaluation."""

    csv_path = result_csv_path(ffs_path)
    if not csv_path.exists():
        msg = (
            f"FFS results not found: {csv_path}. "
            "Run `mvdec-fpmax ffs-kprototypes` first."
        )
        raise FileNotFoundError(msg)

    ffs_df = pd.read_csv(csv_path)
    if ffs_df.empty:
        msg = f"FFS results are empty: {csv_path}."
        raise ValueError(msg)

    required_columns = {"cluster_sizes", "status", "error_message"}
    if not required_columns.issubset(ffs_df.columns):
        missing_columns = sorted(required_columns - set(ffs_df.columns))
        msg = (
            "FFS results were created with an older schema and are missing "
            f"{missing_columns}. Rerun `mvdec-fpmax ffs-kprototypes --no-resume`."
        )
        raise ValueError(msg)

    score_column = (
        "final_score" if "final_score" in ffs_df.columns else "silhouette_score"
    )
    usable = ffs_df.dropna(subset=[score_column]).copy()
    usable = usable[np.isfinite(usable[score_column])]
    usable = usable[usable["status"] == "ok"]
    usable["_n_selected_features"] = usable["selected_features"].map(
        lambda value: len(parse_selected_features(value))
    )
    skipped_rows = int((usable["_n_selected_features"] == 0).sum())
    if skipped_rows:
        logger.info(
            "Skipping {} FFS rows with no selected features before Intuitive follow-up",
            skipped_rows,
        )
    usable = usable[usable["_n_selected_features"] > 0]
    if usable.empty:
        msg = "No FFS row with selected features was found."
        raise ValueError(msg)

    return usable.sort_values("job_index").reset_index(drop=True)

def load_usable_without_ffs_rows(
    without_ffs_path: Path = WITHOUT_FFS_KPROTOTYPES_RESULTS_PATH,
) -> pd.DataFrame:
    """Load without-FFS rows that can be followed by Intuitive evaluation."""

    csv_path = result_csv_path(without_ffs_path)
    if not csv_path.exists():
        msg = (
            f"Without-FFS results not found: {csv_path}. "
            "Run `mvdec-fpmax without-ffs-kprototypes` first."
        )
        raise FileNotFoundError(msg)

    results_df = pd.read_csv(csv_path)
    if results_df.empty:
        msg = f"Without-FFS results are empty: {csv_path}."
        raise ValueError(msg)

    required_columns = {
        "job_index",
        "strategy",
        "n_bins",
        "min_support",
        "silhouette_score",
        "n_selected_features",
        "selected_features",
        "cluster_sizes",
        "status",
        "error_message",
    }
    if not required_columns.issubset(results_df.columns):
        missing_columns = sorted(required_columns - set(results_df.columns))
        msg = (
            "Without-FFS results were created with an older schema and are "
            f"missing {missing_columns}. Rerun "
            "mvdec-fpmax without-ffs-kprototypes --no-resume."
        )
        raise ValueError(msg)

    usable = results_df.dropna(subset=["silhouette_score"]).copy()
    usable = usable[np.isfinite(usable["silhouette_score"])]
    usable = usable[usable["status"] == "ok"]
    usable["_n_selected_features"] = usable["selected_features"].map(
        lambda value: len(parse_selected_features(value))
    )
    skipped_rows = int((usable["_n_selected_features"] == 0).sum())
    if skipped_rows:
        logger.info(
            "Skipping {} without-FFS rows with no FP-Max features before "
            "Intuitive follow-up",
            skipped_rows,
        )
    usable = usable[usable["_n_selected_features"] > 0]
    if usable.empty:
        msg = "No K-Prototypes without-FFS row with FP-Max features was found."
        raise ValueError(msg)

    return usable.sort_values("job_index").reset_index(drop=True)

def validate_post_ffs_resume_source(
    existing_df: pd.DataFrame,
    base_rows: pd.DataFrame,
    intuitive_param_grid: tuple[IntuitiveParams, ...],
) -> None:
    """Ensure existing post-FFS output matches the all-base-row job mapping."""

    if existing_df.empty:
        return

    for row in existing_df.itertuples(index=False):
        job_index = int(row.job_index)
        source_position, param_index = divmod(job_index, len(intuitive_param_grid))
        if source_position >= len(base_rows):
            msg = "Existing post-FFS Intuitive output does not match base rows."
            raise ValueError(f"{msg} Rerun with --no-resume.")

        base_row = base_rows.iloc[source_position]
        params = intuitive_param_grid[param_index]
        same_alpha = (
            True
            if params.view_weight_alpha is None
            else hasattr(row, "alpha")
            and np.isclose(float(row.alpha), params.view_weight_alpha)
        )
        same_source = (
            int(row.ffs_job_index) == int(base_row["job_index"])
            and row.strategy == base_row["strategy"]
            and int(row.n_bins) == int(base_row["n_bins"])
            and np.isclose(float(row.min_support), float(base_row["min_support"]))
            and np.isclose(float(row.mu_param), params.mu_param)
            and np.isclose(float(row.gamma), params.gamma)
            and np.isclose(float(row.beta), params.beta)
            and same_alpha
        )
        if not same_source:
            msg = (
                "Existing post-FFS Intuitive output was created from a different "
                "job mapping. Rerun with --no-resume."
            )
            raise ValueError(msg)

def validate_post_without_ffs_resume_source(
    existing_df: pd.DataFrame,
    base_rows: pd.DataFrame,
    intuitive_param_grid: tuple[IntuitiveParams, ...],
) -> None:
    """Ensure existing post-without-FFS output matches the all-base-row mapping."""

    if existing_df.empty:
        return

    for row in existing_df.itertuples(index=False):
        job_index = int(row.job_index)
        source_position, param_index = divmod(job_index, len(intuitive_param_grid))
        if source_position >= len(base_rows):
            msg = (
                "Existing post-without-FFS Intuitive output does not match "
                "base rows. Rerun with --no-resume."
            )
            raise ValueError(msg)

        base_row = base_rows.iloc[source_position]
        params = intuitive_param_grid[param_index]
        same_alpha = (
            True
            if params.view_weight_alpha is None
            else hasattr(row, "alpha")
            and np.isclose(float(row.alpha), params.view_weight_alpha)
        )
        same_source = (
            int(row.without_ffs_job_index) == int(base_row["job_index"])
            and row.strategy == base_row["strategy"]
            and int(row.n_bins) == int(base_row["n_bins"])
            and np.isclose(float(row.min_support), float(base_row["min_support"]))
            and np.isclose(float(row.mu_param), params.mu_param)
            and np.isclose(float(row.gamma), params.gamma)
            and np.isclose(float(row.beta), params.beta)
            and same_alpha
        )
        if not same_source:
            msg = (
                "Existing post-without-FFS Intuitive output was created from a "
                "different job mapping. Rerun with --no-resume."
            )
            raise ValueError(msg)

def make_post_ffs_intuitive_best_record(
    *,
    job_index: int,
    base_row: pd.Series,
    score_column: str,
    selected_feature_names: list[str],
    best_trial: NativeIntuitiveTrial,
) -> PostFfsIntuitiveBestRecord:
    """Convert the best post-FFS Intuitive trial into one persisted row."""

    clustering = best_trial.clustering
    intuitive_score = clustering.score if clustering else np.nan
    view_weighted_score = (
        best_trial.view_weighted_score
        if best_trial.view_weighted_score is not None
        else np.nan
    )
    ffs_score = float(base_row[score_column])
    objective_score = trial_selection_score(best_trial) if clustering else np.nan
    return PostFfsIntuitiveBestRecord(
        job_index=job_index,
        ffs_job_index=int(base_row["job_index"]),
        strategy=base_row["strategy"],
        n_bins=int(base_row["n_bins"]),
        min_support=float(base_row["min_support"]),
        ffs_score=ffs_score,
        intuitive_score=intuitive_score,
        view_weighted_score=view_weighted_score,
        score_delta=objective_score - ffs_score,
        n_selected_features=len(selected_feature_names),
        selected_features=", ".join(selected_feature_names),
        cluster_sizes=(clustering.cluster_sizes if clustering else []),
        alpha=best_trial.view_weight_alpha,
        mu_param=best_trial.mu_param,
        gamma=best_trial.gamma,
        beta=best_trial.beta,
        status=best_trial.status,
        error_message=best_trial.error_message,
    )

def make_post_without_ffs_intuitive_best_record(
    *,
    job_index: int,
    base_row: pd.Series,
    selected_feature_names: list[str],
    best_trial: NativeIntuitiveTrial,
) -> PostWithoutFfsIntuitiveBestRecord:
    """Convert the best post-without-FFS Intuitive trial into one row."""

    clustering = best_trial.clustering
    intuitive_score = clustering.score if clustering else np.nan
    view_weighted_score = (
        best_trial.view_weighted_score
        if best_trial.view_weighted_score is not None
        else np.nan
    )
    kprototypes_score = float(base_row["silhouette_score"])
    objective_score = trial_selection_score(best_trial) if clustering else np.nan
    return PostWithoutFfsIntuitiveBestRecord(
        job_index=job_index,
        without_ffs_job_index=int(base_row["job_index"]),
        strategy=base_row["strategy"],
        n_bins=int(base_row["n_bins"]),
        min_support=float(base_row["min_support"]),
        kprototypes_score=kprototypes_score,
        intuitive_score=intuitive_score,
        view_weighted_score=view_weighted_score,
        score_delta=objective_score - kprototypes_score,
        n_selected_features=len(selected_feature_names),
        selected_features=", ".join(selected_feature_names),
        cluster_sizes=(clustering.cluster_sizes if clustering else []),
        alpha=best_trial.view_weight_alpha,
        mu_param=best_trial.mu_param,
        gamma=best_trial.gamma,
        beta=best_trial.beta,
        status=best_trial.status,
        error_message=best_trial.error_message,
    )

def validate_post_ffs_best_resume_source(
    existing_df: pd.DataFrame,
    base_rows: pd.DataFrame,
) -> None:
    """Ensure existing best-post-FFS output matches the base-row mapping."""

    if existing_df.empty:
        return

    for row in existing_df.itertuples(index=False):
        job_index = int(row.job_index)
        if job_index >= len(base_rows):
            msg = "Existing best post-FFS output does not match base rows."
            raise ValueError(f"{msg} Rerun with --no-resume.")

        base_row = base_rows.iloc[job_index]
        same_source = (
            int(row.ffs_job_index) == int(base_row["job_index"])
            and row.strategy == base_row["strategy"]
            and int(row.n_bins) == int(base_row["n_bins"])
            and np.isclose(float(row.min_support), float(base_row["min_support"]))
        )
        if not same_source:
            msg = "Existing best post-FFS output has a different job mapping."
            raise ValueError(f"{msg} Rerun with --no-resume.")

def validate_post_without_ffs_best_resume_source(
    existing_df: pd.DataFrame,
    base_rows: pd.DataFrame,
) -> None:
    """Ensure existing best-post-without-FFS output matches base rows."""

    if existing_df.empty:
        return

    for row in existing_df.itertuples(index=False):
        job_index = int(row.job_index)
        if job_index >= len(base_rows):
            msg = "Existing best post-without-FFS output does not match base rows."
            raise ValueError(f"{msg} Rerun with --no-resume.")

        base_row = base_rows.iloc[job_index]
        same_source = (
            int(row.without_ffs_job_index) == int(base_row["job_index"])
            and row.strategy == base_row["strategy"]
            and int(row.n_bins) == int(base_row["n_bins"])
            and np.isclose(float(row.min_support), float(base_row["min_support"]))
        )
        if not same_source:
            msg = (
                "Existing best post-without-FFS output has a different "
                "job mapping."
            )
            raise ValueError(f"{msg} Rerun with --no-resume.")

def run_post_ffs_intuitive_best(
    h_fused_df: pd.DataFrame,
    save_path: Path = POST_FFS_INTUITIVE_BEST_RESULTS_PATH,
    ffs_path: Path = FFS_RESULTS_PATH,
    n_clusters: int = 5,
    random_state: int = RANDOM_STATE,
    resume: bool = True,
    param_workers: int = 1,
    columns: list[str] | None = None,
    intuitive_param_grid: tuple[IntuitiveParams, ...] | None = None,
    label: str = "post-FFS/intuitive-best",
    two_stage: bool = True,
) -> pd.DataFrame:
    """Run two-stage Intuitive and keep one best row per usable FFS row."""

    if param_workers < 1:
        msg = "param_workers must be at least 1."
        raise ValueError(msg)

    columns = columns or POST_FFS_INTUITIVE_BEST_COLUMNS
    existing_df = load_results(save_path, columns) if resume else empty_results(columns)
    intuitive_param_grid = intuitive_param_grid or default_intuitive_param_grid()
    base_rows = load_usable_ffs_rows(ffs_path)
    validate_post_ffs_best_resume_source(existing_df, base_rows)
    score_column = (
        "final_score" if "final_score" in base_rows.columns else "silhouette_score"
    )
    records: dict[int, PostFfsIntuitiveBestRecord] = {}
    logger.info(
        "Running {} on {} FFS rows with two-stage tuning",
        label,
        len(base_rows),
    )

    for source_position, base_row in base_rows.iterrows():
        if is_job_done(existing_df, source_position):
            continue

        try:
            selected_feature_names = parse_selected_features(
                base_row["selected_features"]
            )
            fpmax_features = extract_fpmax_features(
                df=h_fused_df,
                n_bins=int(base_row["n_bins"]),
                bin_labels=BIN_LABELS_BY_SIZE[int(base_row["n_bins"])],
                strategy=base_row["strategy"],
                min_support=float(base_row["min_support"]),
                drop_original_numeric=True,
            )
            rebuilt_selected_feature_names = resolve_rebuilt_feature_names(
                selected_feature_names=selected_feature_names,
                rebuilt_feature_names=fpmax_features.features.columns.tolist(),
            )
            selected_binary_df = fpmax_features.features[
                rebuilt_selected_feature_names
            ]
            best_trial = select_best_native_intuitive_trial(
                h_fused_df=h_fused_df,
                binary_df=selected_binary_df,
                n_clusters=n_clusters,
                random_state=random_state,
                param_workers=param_workers,
                intuitive_param_grid=intuitive_param_grid,
                two_stage=two_stage,
            )
            records[source_position] = make_post_ffs_intuitive_best_record(
                job_index=source_position,
                base_row=base_row,
                score_column=score_column,
                selected_feature_names=selected_feature_names,
                best_trial=best_trial,
            )
            save_results(existing_df, records, save_path, label, columns)
        except Exception:
            logger.exception(
                "Skipping failed {} source_position={} ffs_job_index={}. It "
                "will be retried on the next resume run.",
                label,
                source_position,
                int(base_row["job_index"]),
            )
            continue

    return save_results(existing_df, records, save_path, label, columns)

def run_post_without_ffs_intuitive_best(
    h_fused_df: pd.DataFrame,
    save_path: Path = POST_WITHOUT_FFS_INTUITIVE_BEST_RESULTS_PATH,
    without_ffs_path: Path = WITHOUT_FFS_KPROTOTYPES_RESULTS_PATH,
    n_clusters: int = 5,
    random_state: int = RANDOM_STATE,
    resume: bool = True,
    param_workers: int = 1,
    columns: list[str] | None = None,
    intuitive_param_grid: tuple[IntuitiveParams, ...] | None = None,
    label: str = "post-without-FFS/intuitive-best",
    two_stage: bool = True,
) -> pd.DataFrame:
    """Run two-stage Intuitive and keep one best row per without-FFS row."""

    if param_workers < 1:
        msg = "param_workers must be at least 1."
        raise ValueError(msg)

    columns = columns or POST_WITHOUT_FFS_INTUITIVE_BEST_COLUMNS
    existing_df = load_results(save_path, columns) if resume else empty_results(columns)
    intuitive_param_grid = intuitive_param_grid or default_intuitive_param_grid()
    base_rows = load_usable_without_ffs_rows(without_ffs_path)
    validate_post_without_ffs_best_resume_source(existing_df, base_rows)
    records: dict[int, PostWithoutFfsIntuitiveBestRecord] = {}
    logger.info(
        "Running {} on {} without-FFS rows with two-stage tuning",
        label,
        len(base_rows),
    )

    for source_position, base_row in base_rows.iterrows():
        if is_job_done(existing_df, source_position):
            continue

        try:
            selected_feature_names = parse_selected_features(
                base_row["selected_features"]
            )
            fpmax_features = extract_fpmax_features(
                df=h_fused_df,
                n_bins=int(base_row["n_bins"]),
                bin_labels=BIN_LABELS_BY_SIZE[int(base_row["n_bins"])],
                strategy=base_row["strategy"],
                min_support=float(base_row["min_support"]),
                drop_original_numeric=True,
            )
            rebuilt_selected_feature_names = resolve_rebuilt_feature_names(
                selected_feature_names=selected_feature_names,
                rebuilt_feature_names=fpmax_features.features.columns.tolist(),
            )
            selected_binary_df = fpmax_features.features[
                rebuilt_selected_feature_names
            ]
            best_trial = select_best_native_intuitive_trial(
                h_fused_df=h_fused_df,
                binary_df=selected_binary_df,
                n_clusters=n_clusters,
                random_state=random_state,
                param_workers=param_workers,
                intuitive_param_grid=intuitive_param_grid,
                two_stage=two_stage,
            )
            records[source_position] = make_post_without_ffs_intuitive_best_record(
                job_index=source_position,
                base_row=base_row,
                selected_feature_names=selected_feature_names,
                best_trial=best_trial,
            )
            save_results(existing_df, records, save_path, label, columns)
        except Exception:
            logger.exception(
                "Skipping failed {} source_position={} without_ffs_job_index={}. "
                "It will be retried on the next resume run.",
                label,
                source_position,
                int(base_row["job_index"]),
            )
            continue

    return save_results(existing_df, records, save_path, label, columns)

def run_post_ffs_intuitive_view_weighted_best(
    h_fused_df: pd.DataFrame,
    save_path: Path = POST_FFS_INTUITIVE_VIEW_WEIGHTED_BEST_RESULTS_PATH,
    ffs_path: Path = FFS_RESULTS_PATH,
    n_clusters: int = 5,
    random_state: int = RANDOM_STATE,
    resume: bool = True,
    param_workers: int = 1,
) -> pd.DataFrame:
    """Run two-stage view-weighted Intuitive after usable FFS rows."""

    return run_post_ffs_intuitive_best(
        h_fused_df=h_fused_df,
        save_path=save_path,
        ffs_path=ffs_path,
        n_clusters=n_clusters,
        random_state=random_state,
        resume=resume,
        param_workers=param_workers,
        columns=POST_FFS_INTUITIVE_VIEW_WEIGHTED_BEST_COLUMNS,
        intuitive_param_grid=default_view_weighted_intuitive_param_grid(),
        label="post-FFS/intuitive-view-weighted-best",
    )

def run_post_without_ffs_intuitive_view_weighted_best(
    h_fused_df: pd.DataFrame,
    save_path: Path = POST_WITHOUT_FFS_INTUITIVE_VIEW_WEIGHTED_BEST_RESULTS_PATH,
    without_ffs_path: Path = WITHOUT_FFS_KPROTOTYPES_RESULTS_PATH,
    n_clusters: int = 5,
    random_state: int = RANDOM_STATE,
    resume: bool = True,
    param_workers: int = 1,
) -> pd.DataFrame:
    """Run two-stage view-weighted Intuitive after without-FFS rows."""

    return run_post_without_ffs_intuitive_best(
        h_fused_df=h_fused_df,
        save_path=save_path,
        without_ffs_path=without_ffs_path,
        n_clusters=n_clusters,
        random_state=random_state,
        resume=resume,
        param_workers=param_workers,
        columns=POST_WITHOUT_FFS_INTUITIVE_VIEW_WEIGHTED_BEST_COLUMNS,
        intuitive_param_grid=default_view_weighted_intuitive_param_grid(),
        label="post-without-FFS/intuitive-view-weighted-best",
    )
def run_post_ffs_intuitive_exhaustive_best(
    h_fused_df: pd.DataFrame,
    save_path: Path = POST_FFS_INTUITIVE_EXHAUSTIVE_BEST_RESULTS_PATH,
    ffs_path: Path = FFS_RESULTS_PATH,
    n_clusters: int = 5,
    random_state: int = RANDOM_STATE,
    resume: bool = True,
    param_workers: int = 1,
) -> pd.DataFrame:
    """Run exhaustive Intuitive tuning after usable FFS rows."""

    return run_post_ffs_intuitive_best(
        h_fused_df=h_fused_df,
        save_path=save_path,
        ffs_path=ffs_path,
        n_clusters=n_clusters,
        random_state=random_state,
        resume=resume,
        param_workers=param_workers,
        label="post-FFS/intuitive-exhaustive-best",
        two_stage=False,
    )

def run_post_without_ffs_intuitive_exhaustive_best(
    h_fused_df: pd.DataFrame,
    save_path: Path = POST_WITHOUT_FFS_INTUITIVE_EXHAUSTIVE_BEST_RESULTS_PATH,
    without_ffs_path: Path = WITHOUT_FFS_KPROTOTYPES_RESULTS_PATH,
    n_clusters: int = 5,
    random_state: int = RANDOM_STATE,
    resume: bool = True,
    param_workers: int = 1,
) -> pd.DataFrame:
    """Run exhaustive Intuitive tuning after usable without-FFS rows."""

    return run_post_without_ffs_intuitive_best(
        h_fused_df=h_fused_df,
        save_path=save_path,
        without_ffs_path=without_ffs_path,
        n_clusters=n_clusters,
        random_state=random_state,
        resume=resume,
        param_workers=param_workers,
        label="post-without-FFS/intuitive-exhaustive-best",
        two_stage=False,
    )

def run_post_ffs_intuitive_view_weighted_exhaustive_best(
    h_fused_df: pd.DataFrame,
    save_path: Path = POST_FFS_INTUITIVE_VIEW_WEIGHTED_EXHAUSTIVE_BEST_RESULTS_PATH,
    ffs_path: Path = FFS_RESULTS_PATH,
    n_clusters: int = 5,
    random_state: int = RANDOM_STATE,
    resume: bool = True,
    param_workers: int = 1,
) -> pd.DataFrame:
    """Run exhaustive view-weighted Intuitive after usable FFS rows."""

    return run_post_ffs_intuitive_best(
        h_fused_df=h_fused_df,
        save_path=save_path,
        ffs_path=ffs_path,
        n_clusters=n_clusters,
        random_state=random_state,
        resume=resume,
        param_workers=param_workers,
        columns=POST_FFS_INTUITIVE_VIEW_WEIGHTED_BEST_COLUMNS,
        intuitive_param_grid=default_view_weighted_intuitive_param_grid(),
        label="post-FFS/intuitive-view-weighted-exhaustive-best",
        two_stage=False,
    )

def run_post_without_ffs_intuitive_view_weighted_exhaustive_best(
    h_fused_df: pd.DataFrame,
    save_path: Path = (
        POST_WITHOUT_FFS_INTUITIVE_VIEW_WEIGHTED_EXHAUSTIVE_BEST_RESULTS_PATH
    ),
    without_ffs_path: Path = WITHOUT_FFS_KPROTOTYPES_RESULTS_PATH,
    n_clusters: int = 5,
    random_state: int = RANDOM_STATE,
    resume: bool = True,
    param_workers: int = 1,
) -> pd.DataFrame:
    """Run exhaustive view-weighted Intuitive after without-FFS rows."""

    return run_post_without_ffs_intuitive_best(
        h_fused_df=h_fused_df,
        save_path=save_path,
        without_ffs_path=without_ffs_path,
        n_clusters=n_clusters,
        random_state=random_state,
        resume=resume,
        param_workers=param_workers,
        columns=POST_WITHOUT_FFS_INTUITIVE_VIEW_WEIGHTED_BEST_COLUMNS,
        intuitive_param_grid=default_view_weighted_intuitive_param_grid(),
        label="post-without-FFS/intuitive-view-weighted-exhaustive-best",
        two_stage=False,
    )
def run_post_ffs_intuitive(
    h_fused_df: pd.DataFrame,
    save_path: Path = POST_FFS_INTUITIVE_RESULTS_PATH,
    ffs_path: Path = FFS_RESULTS_PATH,
    n_clusters: int = 5,
    random_state: int = RANDOM_STATE,
    resume: bool = True,
    param_workers: int = 1,
    columns: list[str] | None = None,
    intuitive_param_grid: tuple[IntuitiveParams, ...] | None = None,
    label: str = "post-FFS/intuitive",
) -> pd.DataFrame:
    """Run Intuitive-K-prototypes after every usable FFS result row."""

    if param_workers < 1:
        msg = "param_workers must be at least 1."
        raise ValueError(msg)

    columns = columns or POST_FFS_INTUITIVE_COLUMNS
    existing_df = (
        load_results(save_path, columns) if resume else empty_results(columns)
    )
    intuitive_param_grid = intuitive_param_grid or default_intuitive_param_grid()
    base_rows = load_usable_ffs_rows(ffs_path)
    validate_post_ffs_resume_source(existing_df, base_rows, intuitive_param_grid)
    score_column = (
        "final_score" if "final_score" in base_rows.columns else "silhouette_score"
    )
    logger.info(
        "Running {} on {} FFS rows with {} params each",
        label,
        len(base_rows),
        len(intuitive_param_grid),
    )

    records: dict[int, PostFfsIntuitiveRecord] = {}
    for source_position, base_row in base_rows.iterrows():
        try:
            selected_feature_names = parse_selected_features(
                base_row["selected_features"]
            )
            if not selected_feature_names:
                logger.info(
                    "Skipping post-FFS/intuitive row {} because it has no "
                    "selected features",
                    int(base_row["job_index"]),
                )
                continue

            fpmax_features = extract_fpmax_features(
                df=h_fused_df,
                n_bins=int(base_row["n_bins"]),
                bin_labels=BIN_LABELS_BY_SIZE[int(base_row["n_bins"])],
                strategy=base_row["strategy"],
                min_support=float(base_row["min_support"]),
                drop_original_numeric=True,
            )
            rebuilt_selected_feature_names = resolve_rebuilt_feature_names(
                selected_feature_names=selected_feature_names,
                rebuilt_feature_names=fpmax_features.features.columns.tolist(),
            )
            selected_binary_df = fpmax_features.features[
                rebuilt_selected_feature_names
            ]
            combined_df, _, _ = make_mixed_features(
                continuous_df=h_fused_df,
                binary_df=selected_binary_df,
            )
            distance_matrix = compute_gower_distance(combined_df)
            view_weighted_distance_matrices = build_view_weighted_distance_matrices(
                h_fused_df,
                selected_binary_df,
                intuitive_param_grid,
            )
            ffs_score = float(base_row[score_column])
            jobs = [
                ExperimentJob(
                    job_index=(source_position * len(intuitive_param_grid))
                    + param_index,
                    min_support=float(base_row["min_support"]),
                    intuitive_params=intuitive_params,
                )
                for param_index, intuitive_params in enumerate(intuitive_param_grid)
                if not is_job_done(
                    existing_df,
                    (source_position * len(intuitive_param_grid)) + param_index,
                )
            ]
            if not jobs:
                continue

            logger.info(
                "Running post-FFS/intuitive source: ffs_job_index={} strategy={} "
                "n_bins={} min_support={} ffs_score={:.4f} features={} params={}",
                int(base_row["job_index"]),
                base_row["strategy"],
                int(base_row["n_bins"]),
                float(base_row["min_support"]),
                ffs_score,
                len(selected_feature_names),
                len(jobs),
            )
            context = PostFfsIntuitiveContext(
                h_fused_df=h_fused_df,
                binary_df=selected_binary_df,
                distance_matrix=distance_matrix,
                view_weighted_distance_matrices=view_weighted_distance_matrices,
                ffs_job_index=int(base_row["job_index"]),
                strategy=base_row["strategy"],
                n_bins=int(base_row["n_bins"]),
                min_support=float(base_row["min_support"]),
                ffs_score=ffs_score,
                selected_features=", ".join(selected_feature_names),
                n_selected_features=len(selected_feature_names),
                n_clusters=n_clusters,
                random_state=random_state,
            )
        except Exception:
            logger.exception(
                "Skipping failed {} source_position={} ffs_job_index={}. It "
                "will be retried on the next resume run.",
                label,
                source_position,
                int(base_row["job_index"]),
            )
            continue
        if param_workers > 1:
            max_workers = min(param_workers, len(jobs))
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=init_post_ffs_intuitive_worker,
                initargs=(context,),
            ) as executor:
                for record in executor.map(run_post_ffs_intuitive_job_safely, jobs):
                    if record is None:
                        continue
                    records[record.job_index] = record
                    save_results(
                        existing_df,
                        records,
                        save_path,
                        label,
                        columns,
                    )
            continue

        init_post_ffs_intuitive_worker(context)
        for job in jobs:
            record = run_post_ffs_intuitive_job_safely(job)
            if record is None:
                continue
            records[record.job_index] = record
            save_results(existing_df, records, save_path, label, columns)

    return save_results(existing_df, records, save_path, label, columns)

def run_post_without_ffs_intuitive(
    h_fused_df: pd.DataFrame,
    save_path: Path = POST_WITHOUT_FFS_INTUITIVE_RESULTS_PATH,
    without_ffs_path: Path = WITHOUT_FFS_KPROTOTYPES_RESULTS_PATH,
    n_clusters: int = 5,
    random_state: int = RANDOM_STATE,
    resume: bool = True,
    param_workers: int = 1,
    columns: list[str] | None = None,
    intuitive_param_grid: tuple[IntuitiveParams, ...] | None = None,
    label: str = "post-without-FFS/intuitive",
) -> pd.DataFrame:
    """Run Intuitive after every usable K-Prototypes without-FFS result row."""

    if param_workers < 1:
        msg = "param_workers must be at least 1."
        raise ValueError(msg)

    columns = columns or POST_WITHOUT_FFS_INTUITIVE_COLUMNS
    existing_df = (
        load_results(save_path, columns) if resume else empty_results(columns)
    )
    intuitive_param_grid = intuitive_param_grid or default_intuitive_param_grid()
    base_rows = load_usable_without_ffs_rows(without_ffs_path)
    validate_post_without_ffs_resume_source(
        existing_df,
        base_rows,
        intuitive_param_grid,
    )
    logger.info(
        "Running {} on {} K-Prototypes rows with {} "
        "params each",
        label,
        len(base_rows),
        len(intuitive_param_grid),
    )

    records: dict[int, PostWithoutFfsIntuitiveRecord] = {}
    for source_position, base_row in base_rows.iterrows():
        try:
            selected_feature_names = parse_selected_features(
                base_row["selected_features"]
            )
            if not selected_feature_names:
                logger.info(
                    "Skipping post-without-FFS/intuitive row {} because it has no "
                    "FP-Max features",
                    int(base_row["job_index"]),
                )
                continue

            fpmax_features = extract_fpmax_features(
                df=h_fused_df,
                n_bins=int(base_row["n_bins"]),
                bin_labels=BIN_LABELS_BY_SIZE[int(base_row["n_bins"])],
                strategy=base_row["strategy"],
                min_support=float(base_row["min_support"]),
                drop_original_numeric=True,
            )
            rebuilt_selected_feature_names = resolve_rebuilt_feature_names(
                selected_feature_names=selected_feature_names,
                rebuilt_feature_names=fpmax_features.features.columns.tolist(),
            )
            selected_binary_df = fpmax_features.features[
                rebuilt_selected_feature_names
            ]
            combined_df, _, _ = make_mixed_features(
                continuous_df=h_fused_df,
                binary_df=selected_binary_df,
            )
            distance_matrix = compute_gower_distance(combined_df)
            view_weighted_distance_matrices = build_view_weighted_distance_matrices(
                h_fused_df,
                selected_binary_df,
                intuitive_param_grid,
            )
            kprototypes_score = float(base_row["silhouette_score"])
            jobs = [
                ExperimentJob(
                    job_index=(source_position * len(intuitive_param_grid))
                    + param_index,
                    min_support=float(base_row["min_support"]),
                    intuitive_params=intuitive_params,
                )
                for param_index, intuitive_params in enumerate(intuitive_param_grid)
                if not is_job_done(
                    existing_df,
                    (source_position * len(intuitive_param_grid)) + param_index,
                )
            ]
            if not jobs:
                continue

            logger.info(
                "Running post-without-FFS/intuitive source: "
                "without_ffs_job_index={} strategy={} n_bins={} min_support={} "
                "kprototypes_score={:.4f} features={} params={}",
                int(base_row["job_index"]),
                base_row["strategy"],
                int(base_row["n_bins"]),
                float(base_row["min_support"]),
                kprototypes_score,
                len(selected_feature_names),
                len(jobs),
            )
            context = PostWithoutFfsIntuitiveContext(
                h_fused_df=h_fused_df,
                binary_df=selected_binary_df,
                distance_matrix=distance_matrix,
                view_weighted_distance_matrices=view_weighted_distance_matrices,
                without_ffs_job_index=int(base_row["job_index"]),
                strategy=base_row["strategy"],
                n_bins=int(base_row["n_bins"]),
                min_support=float(base_row["min_support"]),
                kprototypes_score=kprototypes_score,
                selected_features=", ".join(selected_feature_names),
                n_selected_features=len(selected_feature_names),
                n_clusters=n_clusters,
                random_state=random_state,
            )
        except Exception:
            logger.exception(
                "Skipping failed {} source_position={} without_ffs_job_index={}. "
                "It will be retried on the next resume run.",
                label,
                source_position,
                int(base_row["job_index"]),
            )
            continue
        if param_workers > 1:
            max_workers = min(param_workers, len(jobs))
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=init_post_without_ffs_intuitive_worker,
                initargs=(context,),
            ) as executor:
                for record in executor.map(
                    run_post_without_ffs_intuitive_job_safely,
                    jobs,
                ):
                    if record is None:
                        continue
                    records[record.job_index] = record
                    save_results(
                        existing_df,
                        records,
                        save_path,
                        label,
                        columns,
                    )
            continue

        init_post_without_ffs_intuitive_worker(context)
        for job in jobs:
            record = run_post_without_ffs_intuitive_job_safely(job)
            if record is None:
                continue
            records[record.job_index] = record
            save_results(
                existing_df,
                records,
                save_path,
                label,
                columns,
            )

    return save_results(
        existing_df,
        records,
        save_path,
        label,
        columns,
    )


def run_post_ffs_intuitive_view_weighted(
    h_fused_df: pd.DataFrame,
    save_path: Path = POST_FFS_INTUITIVE_VIEW_WEIGHTED_RESULTS_PATH,
    ffs_path: Path = FFS_RESULTS_PATH,
    n_clusters: int = 5,
    random_state: int = RANDOM_STATE,
    resume: bool = True,
    param_workers: int = 1,
) -> pd.DataFrame:
    """Run view-weighted Intuitive after every usable FFS result row."""

    return run_post_ffs_intuitive(
        h_fused_df=h_fused_df,
        save_path=save_path,
        ffs_path=ffs_path,
        n_clusters=n_clusters,
        random_state=random_state,
        resume=resume,
        param_workers=param_workers,
        columns=POST_FFS_INTUITIVE_VIEW_WEIGHTED_COLUMNS,
        intuitive_param_grid=default_view_weighted_intuitive_param_grid(),
        label="post-FFS/intuitive-view-weighted",
    )

def run_post_without_ffs_intuitive_view_weighted(
    h_fused_df: pd.DataFrame,
    save_path: Path = POST_WITHOUT_FFS_INTUITIVE_VIEW_WEIGHTED_RESULTS_PATH,
    without_ffs_path: Path = WITHOUT_FFS_KPROTOTYPES_RESULTS_PATH,
    n_clusters: int = 5,
    random_state: int = RANDOM_STATE,
    resume: bool = True,
    param_workers: int = 1,
) -> pd.DataFrame:
    """Run view-weighted Intuitive after usable without-FFS K-Prototypes rows."""

    return run_post_without_ffs_intuitive(
        h_fused_df=h_fused_df,
        save_path=save_path,
        without_ffs_path=without_ffs_path,
        n_clusters=n_clusters,
        random_state=random_state,
        resume=resume,
        param_workers=param_workers,
        columns=POST_WITHOUT_FFS_INTUITIVE_VIEW_WEIGHTED_COLUMNS,
        intuitive_param_grid=default_view_weighted_intuitive_param_grid(),
        label="post-without-FFS/intuitive-view-weighted",
    )

def run_without_ffs(
    h_fused_df: pd.DataFrame,
    save_path: Path = WITHOUT_FFS_KPROTOTYPES_RESULTS_PATH,
    strategies: tuple[DiscretizeStrategy, ...] = DEFAULT_STRATEGIES,
    n_bins_options: tuple[int, ...] = (3, 5, 7),
    supports: np.ndarray = DEFAULT_SUPPORTS,
    n_clusters: int = 5,
    random_state: int = RANDOM_STATE,
    baseline_score: float = 0.4069,
    resume: bool = True,
    workers: int = 3,
    limit: int | None = None,
) -> pd.DataFrame:
    """Run without-FFS K-Prototypes over the FP-Max grid."""

    return run_experiment_grid(
        h_fused_df=h_fused_df,
        save_path=save_path,
        columns=WITHOUT_FFS_COLUMNS,
        job_runner=run_without_ffs_job,
        strategies=strategies,
        n_bins_options=n_bins_options,
        supports=supports,
        n_clusters=n_clusters,
        random_state=random_state,
        baseline_score=baseline_score,
        resume=resume,
        workers=workers,
        param_workers=1,
        limit=limit,
        label="without-FFS/kprototypes",
    )


def run_without_ffs_intuitive_native(
    h_fused_df: pd.DataFrame,
    save_path: Path = WITHOUT_FFS_INTUITIVE_NATIVE_RESULTS_PATH,
    strategies: tuple[DiscretizeStrategy, ...] = DEFAULT_STRATEGIES,
    n_bins_options: tuple[int, ...] = (3, 5, 7),
    supports: np.ndarray = DEFAULT_SUPPORTS,
    n_clusters: int = 5,
    random_state: int = RANDOM_STATE,
    baseline_score: float = 0.4069,
    resume: bool = True,
    workers: int = 3,
    param_workers: int = 1,
    limit: int | None = None,
) -> pd.DataFrame:
    """Run Intuitive directly on every without-FFS FP-Max configuration."""

    return run_experiment_grid(
        h_fused_df=h_fused_df,
        save_path=save_path,
        columns=WITHOUT_FFS_INTUITIVE_NATIVE_COLUMNS,
        job_runner=run_without_ffs_intuitive_native_job,
        strategies=strategies,
        n_bins_options=n_bins_options,
        supports=supports,
        n_clusters=n_clusters,
        random_state=random_state,
        baseline_score=baseline_score,
        resume=resume,
        workers=workers,
        param_workers=param_workers,
        limit=limit,
        label="without-FFS/intuitive-native",
    )



def run_without_ffs_intuitive_view_weighted_native(
    h_fused_df: pd.DataFrame,
    save_path: Path = WITHOUT_FFS_INTUITIVE_VIEW_WEIGHTED_NATIVE_RESULTS_PATH,
    strategies: tuple[DiscretizeStrategy, ...] = DEFAULT_STRATEGIES,
    n_bins_options: tuple[int, ...] = (3, 5, 7),
    supports: np.ndarray = DEFAULT_SUPPORTS,
    n_clusters: int = 5,
    random_state: int = RANDOM_STATE,
    baseline_score: float = 0.4069,
    resume: bool = True,
    workers: int = 3,
    param_workers: int = 1,
    limit: int | None = None,
) -> pd.DataFrame:
    """Run native Intuitive with explicit MVDEC/FP-Max view weighting."""

    return run_experiment_grid(
        h_fused_df=h_fused_df,
        save_path=save_path,
        columns=WITHOUT_FFS_INTUITIVE_VIEW_WEIGHTED_NATIVE_COLUMNS,
        job_runner=run_without_ffs_intuitive_view_weighted_native_job,
        strategies=strategies,
        n_bins_options=n_bins_options,
        supports=supports,
        n_clusters=n_clusters,
        random_state=random_state,
        baseline_score=baseline_score,
        resume=resume,
        workers=workers,
        param_workers=param_workers,
        limit=limit,
        label="without-FFS/intuitive-view-weighted-native",
    )

def run_without_ffs_intuitive_exhaustive_native(
    h_fused_df: pd.DataFrame,
    save_path: Path = WITHOUT_FFS_INTUITIVE_EXHAUSTIVE_NATIVE_RESULTS_PATH,
    strategies: tuple[DiscretizeStrategy, ...] = DEFAULT_STRATEGIES,
    n_bins_options: tuple[int, ...] = (3, 5, 7),
    supports: np.ndarray = DEFAULT_SUPPORTS,
    n_clusters: int = 5,
    random_state: int = RANDOM_STATE,
    baseline_score: float = 0.4069,
    resume: bool = True,
    workers: int = 3,
    param_workers: int = 1,
    limit: int | None = None,
) -> pd.DataFrame:
    """Run exhaustive native Intuitive on every without-FFS config."""

    return run_experiment_grid(
        h_fused_df=h_fused_df,
        save_path=save_path,
        columns=WITHOUT_FFS_INTUITIVE_NATIVE_COLUMNS,
        job_runner=run_without_ffs_intuitive_exhaustive_native_job,
        strategies=strategies,
        n_bins_options=n_bins_options,
        supports=supports,
        n_clusters=n_clusters,
        random_state=random_state,
        baseline_score=baseline_score,
        resume=resume,
        workers=workers,
        param_workers=param_workers,
        limit=limit,
        label="without-FFS/intuitive-exhaustive-native",
    )

def run_without_ffs_intuitive_view_weighted_exhaustive_native(
    h_fused_df: pd.DataFrame,
    save_path: Path = (
        WITHOUT_FFS_INTUITIVE_VIEW_WEIGHTED_EXHAUSTIVE_NATIVE_RESULTS_PATH
    ),
    strategies: tuple[DiscretizeStrategy, ...] = DEFAULT_STRATEGIES,
    n_bins_options: tuple[int, ...] = (3, 5, 7),
    supports: np.ndarray = DEFAULT_SUPPORTS,
    n_clusters: int = 5,
    random_state: int = RANDOM_STATE,
    baseline_score: float = 0.4069,
    resume: bool = True,
    workers: int = 3,
    param_workers: int = 1,
    limit: int | None = None,
) -> pd.DataFrame:
    """Run exhaustive view-weighted native Intuitive over the grid."""

    return run_experiment_grid(
        h_fused_df=h_fused_df,
        save_path=save_path,
        columns=WITHOUT_FFS_INTUITIVE_VIEW_WEIGHTED_NATIVE_COLUMNS,
        job_runner=run_without_ffs_intuitive_view_weighted_exhaustive_native_job,
        strategies=strategies,
        n_bins_options=n_bins_options,
        supports=supports,
        n_clusters=n_clusters,
        random_state=random_state,
        baseline_score=baseline_score,
        resume=resume,
        workers=workers,
        param_workers=param_workers,
        limit=limit,
        label="without-FFS/intuitive-view-weighted-exhaustive-native",
    )
def run_ffs(
    h_fused_df: pd.DataFrame,
    save_path: Path = FFS_RESULTS_PATH,
    strategies: tuple[DiscretizeStrategy, ...] = DEFAULT_STRATEGIES,
    n_bins_options: tuple[int, ...] = (3, 5, 7),
    supports: np.ndarray = DEFAULT_SUPPORTS,
    n_clusters: int = 5,
    random_state: int = RANDOM_STATE,
    baseline_score: float = 0.4069,
    resume: bool = True,
    workers: int = 3,
    candidate_workers: int = 1,
    limit: int | None = None,
) -> pd.DataFrame:
    """Run MiMvDEC with forward feature selection."""

    return run_experiment_grid(
        h_fused_df=h_fused_df,
        save_path=save_path,
        columns=FFS_COLUMNS,
        job_runner=partial(
            run_ffs_job,
            candidate_workers=candidate_workers,
        ),
        strategies=strategies,
        n_bins_options=n_bins_options,
        supports=supports,
        n_clusters=n_clusters,
        random_state=random_state,
        baseline_score=baseline_score,
        resume=resume,
        workers=workers,
        param_workers=1,
        limit=limit,
        label="FFS",
    )


def run_ffs_intuitive_native(
    h_fused_df: pd.DataFrame,
    save_path: Path = FFS_INTUITIVE_NATIVE_RESULTS_PATH,
    strategies: tuple[DiscretizeStrategy, ...] = DEFAULT_STRATEGIES,
    n_bins_options: tuple[int, ...] = (3, 5, 7),
    supports: np.ndarray = DEFAULT_SUPPORTS,
    n_clusters: int = 5,
    random_state: int = RANDOM_STATE,
    baseline_score: float = 0.4069,
    resume: bool = True,
    workers: int = 3,
    param_workers: int = 1,
    candidate_workers: int = 1,
    limit: int | None = None,
) -> pd.DataFrame:
    """Run forward feature selection with Intuitive as the native evaluator."""

    return run_experiment_grid(
        h_fused_df=h_fused_df,
        save_path=save_path,
        columns=FFS_INTUITIVE_NATIVE_COLUMNS,
        job_runner=partial(
            run_ffs_intuitive_native_job,
            candidate_workers=candidate_workers,
        ),
        strategies=strategies,
        n_bins_options=n_bins_options,
        supports=supports,
        n_clusters=n_clusters,
        random_state=random_state,
        baseline_score=baseline_score,
        resume=resume,
        workers=workers,
        param_workers=param_workers,
        limit=limit,
        label="FFS/intuitive-native",
    )

def run_ffs_intuitive_view_weighted_native(
    h_fused_df: pd.DataFrame,
    save_path: Path = FFS_INTUITIVE_VIEW_WEIGHTED_NATIVE_RESULTS_PATH,
    strategies: tuple[DiscretizeStrategy, ...] = DEFAULT_STRATEGIES,
    n_bins_options: tuple[int, ...] = (3, 5, 7),
    supports: np.ndarray = DEFAULT_SUPPORTS,
    n_clusters: int = 5,
    random_state: int = RANDOM_STATE,
    baseline_score: float = 0.4069,
    resume: bool = True,
    workers: int = 3,
    param_workers: int = 1,
    candidate_workers: int = 1,
    limit: int | None = None,
) -> pd.DataFrame:
    """Run FFS with view-weighted Intuitive as the native evaluator."""

    return run_experiment_grid(
        h_fused_df=h_fused_df,
        save_path=save_path,
        columns=FFS_INTUITIVE_VIEW_WEIGHTED_NATIVE_COLUMNS,
        job_runner=partial(
            run_ffs_intuitive_view_weighted_native_job,
            candidate_workers=candidate_workers,
        ),
        strategies=strategies,
        n_bins_options=n_bins_options,
        supports=supports,
        n_clusters=n_clusters,
        random_state=random_state,
        baseline_score=baseline_score,
        resume=resume,
        workers=workers,
        param_workers=param_workers,
        limit=limit,
        label="FFS/intuitive-view-weighted-native",
    )

def run_ffs_intuitive_exhaustive_native(
    h_fused_df: pd.DataFrame,
    save_path: Path = FFS_INTUITIVE_EXHAUSTIVE_NATIVE_RESULTS_PATH,
    strategies: tuple[DiscretizeStrategy, ...] = DEFAULT_STRATEGIES,
    n_bins_options: tuple[int, ...] = (3, 5, 7),
    supports: np.ndarray = DEFAULT_SUPPORTS,
    n_clusters: int = 5,
    random_state: int = RANDOM_STATE,
    baseline_score: float = 0.4069,
    resume: bool = True,
    workers: int = 3,
    param_workers: int = 1,
    candidate_workers: int = 1,
    limit: int | None = None,
) -> pd.DataFrame:
    """Run FFS with exhaustive native Intuitive as the evaluator."""

    return run_experiment_grid(
        h_fused_df=h_fused_df,
        save_path=save_path,
        columns=FFS_INTUITIVE_NATIVE_COLUMNS,
        job_runner=partial(
            run_ffs_intuitive_exhaustive_native_job,
            candidate_workers=candidate_workers,
        ),
        strategies=strategies,
        n_bins_options=n_bins_options,
        supports=supports,
        n_clusters=n_clusters,
        random_state=random_state,
        baseline_score=baseline_score,
        resume=resume,
        workers=workers,
        param_workers=param_workers,
        limit=limit,
        label="FFS/intuitive-exhaustive-native",
    )

def run_ffs_intuitive_view_weighted_exhaustive_native(
    h_fused_df: pd.DataFrame,
    save_path: Path = FFS_INTUITIVE_VIEW_WEIGHTED_EXHAUSTIVE_NATIVE_RESULTS_PATH,
    strategies: tuple[DiscretizeStrategy, ...] = DEFAULT_STRATEGIES,
    n_bins_options: tuple[int, ...] = (3, 5, 7),
    supports: np.ndarray = DEFAULT_SUPPORTS,
    n_clusters: int = 5,
    random_state: int = RANDOM_STATE,
    baseline_score: float = 0.4069,
    resume: bool = True,
    workers: int = 3,
    param_workers: int = 1,
    candidate_workers: int = 1,
    limit: int | None = None,
) -> pd.DataFrame:
    """Run FFS with exhaustive view-weighted native Intuitive."""

    return run_experiment_grid(
        h_fused_df=h_fused_df,
        save_path=save_path,
        columns=FFS_INTUITIVE_VIEW_WEIGHTED_NATIVE_COLUMNS,
        job_runner=partial(
            run_ffs_intuitive_view_weighted_exhaustive_native_job,
            candidate_workers=candidate_workers,
        ),
        strategies=strategies,
        n_bins_options=n_bins_options,
        supports=supports,
        n_clusters=n_clusters,
        random_state=random_state,
        baseline_score=baseline_score,
        resume=resume,
        workers=workers,
        param_workers=param_workers,
        limit=limit,
        label="FFS/intuitive-view-weighted-exhaustive-native",
    )

def run_without_ffs_intuitive_paper_native(
    h_fused_df: pd.DataFrame,
    save_path: Path = WITHOUT_FFS_INTUITIVE_PAPER_NATIVE_RESULTS_PATH,
    strategies: tuple[DiscretizeStrategy, ...] = DEFAULT_STRATEGIES,
    n_bins_options: tuple[int, ...] = (3, 5, 7),
    supports: np.ndarray = DEFAULT_SUPPORTS,
    n_clusters: int = 5,
    random_state: int = RANDOM_STATE,
    baseline_score: float = 0.4069,
    resume: bool = True,
    workers: int = 3,
    param_workers: int = 1,
    limit: int | None = None,
) -> pd.DataFrame:
    """Run native Intuitive with Algorithm-1 paper initialization."""

    return run_experiment_grid(
        h_fused_df=h_fused_df,
        save_path=save_path,
        columns=WITHOUT_FFS_INTUITIVE_NATIVE_COLUMNS,
        job_runner=run_without_ffs_intuitive_paper_native_job,
        strategies=strategies,
        n_bins_options=n_bins_options,
        supports=supports,
        n_clusters=n_clusters,
        random_state=random_state,
        baseline_score=baseline_score,
        resume=resume,
        workers=workers,
        param_workers=param_workers,
        limit=limit,
        label="without-FFS/intuitive-paper-native",
    )

def run_without_ffs_intuitive_paper_exhaustive_native(
    h_fused_df: pd.DataFrame,
    save_path: Path = WITHOUT_FFS_INTUITIVE_PAPER_EXHAUSTIVE_NATIVE_RESULTS_PATH,
    strategies: tuple[DiscretizeStrategy, ...] = DEFAULT_STRATEGIES,
    n_bins_options: tuple[int, ...] = (3, 5, 7),
    supports: np.ndarray = DEFAULT_SUPPORTS,
    n_clusters: int = 5,
    random_state: int = RANDOM_STATE,
    baseline_score: float = 0.4069,
    resume: bool = True,
    workers: int = 3,
    param_workers: int = 1,
    limit: int | None = None,
) -> pd.DataFrame:
    """Run exhaustive native Intuitive with Algorithm-1 paper initialization."""

    return run_experiment_grid(
        h_fused_df=h_fused_df,
        save_path=save_path,
        columns=WITHOUT_FFS_INTUITIVE_NATIVE_COLUMNS,
        job_runner=run_without_ffs_intuitive_paper_exhaustive_native_job,
        strategies=strategies,
        n_bins_options=n_bins_options,
        supports=supports,
        n_clusters=n_clusters,
        random_state=random_state,
        baseline_score=baseline_score,
        resume=resume,
        workers=workers,
        param_workers=param_workers,
        limit=limit,
        label="without-FFS/intuitive-paper-exhaustive-native",
    )

def run_without_ffs_intuitive_view_weighted_paper_native(
    h_fused_df: pd.DataFrame,
    save_path: Path = WITHOUT_FFS_INTUITIVE_VIEW_WEIGHTED_PAPER_NATIVE_RESULTS_PATH,
    strategies: tuple[DiscretizeStrategy, ...] = DEFAULT_STRATEGIES,
    n_bins_options: tuple[int, ...] = (3, 5, 7),
    supports: np.ndarray = DEFAULT_SUPPORTS,
    n_clusters: int = 5,
    random_state: int = RANDOM_STATE,
    baseline_score: float = 0.4069,
    resume: bool = True,
    workers: int = 3,
    param_workers: int = 1,
    limit: int | None = None,
) -> pd.DataFrame:
    """Run view-weighted native Intuitive with paper initialization."""

    return run_experiment_grid(
        h_fused_df=h_fused_df,
        save_path=save_path,
        columns=WITHOUT_FFS_INTUITIVE_VIEW_WEIGHTED_NATIVE_COLUMNS,
        job_runner=run_without_ffs_intuitive_view_weighted_paper_native_job,
        strategies=strategies,
        n_bins_options=n_bins_options,
        supports=supports,
        n_clusters=n_clusters,
        random_state=random_state,
        baseline_score=baseline_score,
        resume=resume,
        workers=workers,
        param_workers=param_workers,
        limit=limit,
        label="without-FFS/intuitive-view-weighted-paper-native",
    )

def run_without_ffs_intuitive_view_weighted_paper_exhaustive_native(
    h_fused_df: pd.DataFrame,
    save_path: Path = (
        WITHOUT_FFS_INTUITIVE_VIEW_WEIGHTED_PAPER_EXHAUSTIVE_NATIVE_RESULTS_PATH
    ),
    strategies: tuple[DiscretizeStrategy, ...] = DEFAULT_STRATEGIES,
    n_bins_options: tuple[int, ...] = (3, 5, 7),
    supports: np.ndarray = DEFAULT_SUPPORTS,
    n_clusters: int = 5,
    random_state: int = RANDOM_STATE,
    baseline_score: float = 0.4069,
    resume: bool = True,
    workers: int = 3,
    param_workers: int = 1,
    limit: int | None = None,
) -> pd.DataFrame:
    """Run exhaustive view-weighted native Intuitive with paper init."""

    return run_experiment_grid(
        h_fused_df=h_fused_df,
        save_path=save_path,
        columns=WITHOUT_FFS_INTUITIVE_VIEW_WEIGHTED_NATIVE_COLUMNS,
        job_runner=run_without_ffs_intuitive_view_weighted_paper_exhaustive_native_job,
        strategies=strategies,
        n_bins_options=n_bins_options,
        supports=supports,
        n_clusters=n_clusters,
        random_state=random_state,
        baseline_score=baseline_score,
        resume=resume,
        workers=workers,
        param_workers=param_workers,
        limit=limit,
        label="without-FFS/intuitive-view-weighted-paper-exhaustive-native",
    )

def run_ffs_intuitive_paper_native(
    h_fused_df: pd.DataFrame,
    save_path: Path = FFS_INTUITIVE_PAPER_NATIVE_RESULTS_PATH,
    strategies: tuple[DiscretizeStrategy, ...] = DEFAULT_STRATEGIES,
    n_bins_options: tuple[int, ...] = (3, 5, 7),
    supports: np.ndarray = DEFAULT_SUPPORTS,
    n_clusters: int = 5,
    random_state: int = RANDOM_STATE,
    baseline_score: float = 0.4069,
    resume: bool = True,
    workers: int = 3,
    param_workers: int = 1,
    candidate_workers: int = 1,
    limit: int | None = None,
) -> pd.DataFrame:
    """Run Intuitive-native FFS with Algorithm-1 paper initialization."""

    return run_experiment_grid(
        h_fused_df=h_fused_df,
        save_path=save_path,
        columns=FFS_INTUITIVE_NATIVE_COLUMNS,
        job_runner=partial(
            run_ffs_intuitive_paper_native_job,
            candidate_workers=candidate_workers,
        ),
        strategies=strategies,
        n_bins_options=n_bins_options,
        supports=supports,
        n_clusters=n_clusters,
        random_state=random_state,
        baseline_score=baseline_score,
        resume=resume,
        workers=workers,
        param_workers=param_workers,
        limit=limit,
        label="FFS/intuitive-paper-native",
    )

def run_ffs_intuitive_paper_exhaustive_native(
    h_fused_df: pd.DataFrame,
    save_path: Path = FFS_INTUITIVE_PAPER_EXHAUSTIVE_NATIVE_RESULTS_PATH,
    strategies: tuple[DiscretizeStrategy, ...] = DEFAULT_STRATEGIES,
    n_bins_options: tuple[int, ...] = (3, 5, 7),
    supports: np.ndarray = DEFAULT_SUPPORTS,
    n_clusters: int = 5,
    random_state: int = RANDOM_STATE,
    baseline_score: float = 0.4069,
    resume: bool = True,
    workers: int = 3,
    param_workers: int = 1,
    candidate_workers: int = 1,
    limit: int | None = None,
) -> pd.DataFrame:
    """Run exhaustive Intuitive-native FFS with paper initialization."""

    return run_experiment_grid(
        h_fused_df=h_fused_df,
        save_path=save_path,
        columns=FFS_INTUITIVE_NATIVE_COLUMNS,
        job_runner=partial(
            run_ffs_intuitive_paper_exhaustive_native_job,
            candidate_workers=candidate_workers,
        ),
        strategies=strategies,
        n_bins_options=n_bins_options,
        supports=supports,
        n_clusters=n_clusters,
        random_state=random_state,
        baseline_score=baseline_score,
        resume=resume,
        workers=workers,
        param_workers=param_workers,
        limit=limit,
        label="FFS/intuitive-paper-exhaustive-native",
    )

def run_ffs_intuitive_view_weighted_paper_native(
    h_fused_df: pd.DataFrame,
    save_path: Path = FFS_INTUITIVE_VIEW_WEIGHTED_PAPER_NATIVE_RESULTS_PATH,
    strategies: tuple[DiscretizeStrategy, ...] = DEFAULT_STRATEGIES,
    n_bins_options: tuple[int, ...] = (3, 5, 7),
    supports: np.ndarray = DEFAULT_SUPPORTS,
    n_clusters: int = 5,
    random_state: int = RANDOM_STATE,
    baseline_score: float = 0.4069,
    resume: bool = True,
    workers: int = 3,
    param_workers: int = 1,
    candidate_workers: int = 1,
    limit: int | None = None,
) -> pd.DataFrame:
    """Run view-weighted Intuitive-native FFS with paper initialization."""

    return run_experiment_grid(
        h_fused_df=h_fused_df,
        save_path=save_path,
        columns=FFS_INTUITIVE_VIEW_WEIGHTED_NATIVE_COLUMNS,
        job_runner=partial(
            run_ffs_intuitive_view_weighted_paper_native_job,
            candidate_workers=candidate_workers,
        ),
        strategies=strategies,
        n_bins_options=n_bins_options,
        supports=supports,
        n_clusters=n_clusters,
        random_state=random_state,
        baseline_score=baseline_score,
        resume=resume,
        workers=workers,
        param_workers=param_workers,
        limit=limit,
        label="FFS/intuitive-view-weighted-paper-native",
    )

def run_ffs_intuitive_view_weighted_paper_exhaustive_native(
    h_fused_df: pd.DataFrame,
    save_path: Path = FFS_INTUITIVE_VIEW_WEIGHTED_PAPER_EXHAUSTIVE_NATIVE_RESULTS_PATH,
    strategies: tuple[DiscretizeStrategy, ...] = DEFAULT_STRATEGIES,
    n_bins_options: tuple[int, ...] = (3, 5, 7),
    supports: np.ndarray = DEFAULT_SUPPORTS,
    n_clusters: int = 5,
    random_state: int = RANDOM_STATE,
    baseline_score: float = 0.4069,
    resume: bool = True,
    workers: int = 3,
    param_workers: int = 1,
    candidate_workers: int = 1,
    limit: int | None = None,
) -> pd.DataFrame:
    """Run exhaustive view-weighted Intuitive-native FFS with paper init."""

    return run_experiment_grid(
        h_fused_df=h_fused_df,
        save_path=save_path,
        columns=FFS_INTUITIVE_VIEW_WEIGHTED_NATIVE_COLUMNS,
        job_runner=partial(
            run_ffs_intuitive_view_weighted_paper_exhaustive_native_job,
            candidate_workers=candidate_workers,
        ),
        strategies=strategies,
        n_bins_options=n_bins_options,
        supports=supports,
        n_clusters=n_clusters,
        random_state=random_state,
        baseline_score=baseline_score,
        resume=resume,
        workers=workers,
        param_workers=param_workers,
        limit=limit,
        label="FFS/intuitive-view-weighted-paper-exhaustive-native",
    )
def make_without_ffs_intuitive_native_trial_records(
    *,
    group: ExperimentGroup,
    job: ExperimentJob,
    fpmax_features: FpmaxFeatures,
    trials: tuple[NativeIntuitiveTrial, ...],
    best_trial: NativeIntuitiveTrial,
) -> tuple[NativeIntuitiveTrialRecord, ...]:
    """Convert all native Intuitive parameter trials into persisted rows."""

    selected_features = ", ".join(fpmax_features.features.columns)
    rows = []
    for trial_index, trial in enumerate(trials):
        clustering = trial.clustering
        rows.append(
            NativeIntuitiveTrialRecord(
                job_index=job.job_index,
                strategy=group.strategy,
                n_bins=group.n_bins,
                min_support=job.min_support,
                selection_step=None,
                candidate_feature=None,
                candidate_features=None,
                trial_index=trial_index,
                stage=trial.stage,
                silhouette_score=(clustering.score if clustering else np.nan),
                view_weighted_score=(
                    trial.view_weighted_score
                    if trial.view_weighted_score is not None
                    else np.nan
                ),
                n_selected_features=len(fpmax_features.features.columns),
                selected_features=selected_features,
                cluster_sizes=(clustering.cluster_sizes if clustering else []),
                alpha=trial.view_weight_alpha,
                mu_param=trial.mu_param,
                gamma=trial.gamma,
                beta=trial.beta,
                status=trial.status,
                error_message=trial.error_message,
                is_best=(trial is best_trial),
            )
        )
    return tuple(rows)

def make_ffs_intuitive_native_trial_records(
    *,
    group: ExperimentGroup,
    job: ExperimentJob,
    candidate_traces: tuple[NativeIntuitiveCandidateTrace, ...],
    final_best_trial: NativeIntuitiveTrial | None,
) -> tuple[NativeIntuitiveTrialRecord, ...]:
    """Convert native Intuitive FFS candidate trials into persisted rows."""

    rows = []
    for trace in candidate_traces:
        selected_features = ", ".join(trace.candidate_feature_names)
        for trial_index, trial in enumerate(trace.trials):
            clustering = trial.clustering
            rows.append(
                NativeIntuitiveTrialRecord(
                    job_index=job.job_index,
                    strategy=group.strategy,
                    n_bins=group.n_bins,
                    min_support=job.min_support,
                    selection_step=trace.selection_step,
                    candidate_feature=trace.candidate_feature,
                    candidate_features=selected_features,
                    trial_index=trial_index,
                    stage=trial.stage,
                    silhouette_score=(clustering.score if clustering else np.nan),
                    view_weighted_score=(
                        trial.view_weighted_score
                        if trial.view_weighted_score is not None
                        else np.nan
                    ),
                    n_selected_features=len(trace.candidate_feature_names),
                    selected_features=selected_features,
                    cluster_sizes=(clustering.cluster_sizes if clustering else []),
                    alpha=trial.view_weight_alpha,
                    mu_param=trial.mu_param,
                    gamma=trial.gamma,
                    beta=trial.beta,
                    status=trial.status,
                    error_message=trial.error_message,
                    is_best=(trial is final_best_trial),
                )
            )
    return tuple(rows)

def native_trial_sidecar_for(
    save_path: Path,
    columns: list[str],
) -> tuple[Path, list[str]] | tuple[None, None]:
    """Return the trial-level sidecar path for native Intuitive outputs."""

    native_column_schemas = {
        tuple(WITHOUT_FFS_INTUITIVE_NATIVE_COLUMNS),
        tuple(WITHOUT_FFS_INTUITIVE_VIEW_WEIGHTED_NATIVE_COLUMNS),
        tuple(FFS_INTUITIVE_NATIVE_COLUMNS),
        tuple(FFS_INTUITIVE_VIEW_WEIGHTED_NATIVE_COLUMNS),
    }
    csv_path = result_csv_path(save_path)
    if tuple(columns) in native_column_schemas and "intuitive" in csv_path.stem:
        if csv_path.stem.endswith("_results"):
            trial_path = csv_path.with_name(
                f"{csv_path.stem.removesuffix('_results')}_trials.csv"
            )
        else:
            trial_path = csv_path.with_name(f"{csv_path.stem}_trials.csv")
        return trial_path, NATIVE_INTUITIVE_TRIAL_COLUMNS
    return None, None

def save_trial_records_if_any(
    record: ExperimentRecord,
    existing_trial_df: pd.DataFrame,
    trial_records: dict[tuple[int, int | None, str | None, int], NativeIntuitiveTrialRecord],
    trial_save_path: Path | None,
    trial_columns: list[str] | None,
    label: str,
) -> None:
    """Save trial-level rows carried by a summary record, when configured."""

    if trial_save_path is None or trial_columns is None:
        return

    rows = getattr(record, "trial_records", ())
    if not rows:
        return

    for row in rows:
        trial_records[
            (
                row.job_index,
                row.selection_step,
                row.candidate_feature,
                row.trial_index,
            )
        ] = row

    save_results(
        existing_trial_df,
        trial_records,
        trial_save_path,
        f"{label}/trials",
        trial_columns,
        dedupe_columns=[
            "job_index",
            "selection_step",
            "candidate_feature",
            "trial_index",
        ],
    )

def run_experiment_grid(
    h_fused_df: pd.DataFrame,
    save_path: Path,
    columns: list[str],
    job_runner: JobRunner,
    strategies: tuple[DiscretizeStrategy, ...],
    n_bins_options: tuple[int, ...],
    supports: np.ndarray,
    n_clusters: int,
    random_state: int,
    baseline_score: float,
    resume: bool,
    workers: int,
    param_workers: int,
    limit: int | None,
    label: str,
    group_runner: GroupRunner | None = None,
) -> pd.DataFrame:
    """Run grouped sensitivity search and save CSV outputs."""

    if param_workers < 1:
        msg = "param_workers must be at least 1."
        raise ValueError(msg)
    group_runner = group_runner or run_group_sequentially
    existing_df = load_results(save_path, columns) if resume else empty_results(columns)
    trial_save_path, trial_columns = native_trial_sidecar_for(save_path, columns)
    existing_trial_df = (
        load_results(trial_save_path, trial_columns)
        if resume and trial_save_path is not None and trial_columns is not None
        else empty_results(trial_columns or [])
    )
    trial_records: dict[
        tuple[int, int | None, str | None, int], NativeIntuitiveTrialRecord
    ] = {}
    if resume and not existing_df.empty:
        missing_columns = [column for column in columns if column not in existing_df]
        if missing_columns:
            msg = (
                f"Existing {label} output is missing columns {missing_columns}. "
                "Rerun with --no-resume."
            )
            raise ValueError(msg)
    groups = build_experiment_groups(
        strategies,
        n_bins_options,
        supports,
        limit,
    )
    if resume and not existing_df.empty:
        validate_experiment_grid_resume_source(existing_df, groups, label)
    records: dict[int, ExperimentRecord] = {}
    pending_groups = filter_pending_groups(groups, existing_df)

    logger.info(
        "Running {} {} groups with workers={}", len(pending_groups), label, workers
    )
    if workers <= 1:
        for group in pending_groups:
            for record in group_runner(
                h_fused_df,
                group,
                n_clusters,
                random_state,
                baseline_score,
                job_runner,
                param_workers,
            ):
                records[record.job_index] = record
                save_results(existing_df, records, save_path, label, columns)
                save_trial_records_if_any(
                    record,
                    existing_trial_df,
                    trial_records,
                    trial_save_path,
                    trial_columns,
                    label,
                )
    else:
        run_groups_parallel(
            h_fused_df=h_fused_df,
            groups=pending_groups,
            n_clusters=n_clusters,
            random_state=random_state,
            baseline_score=baseline_score,
            workers=workers,
            existing_df=existing_df,
            records=records,
            save_path=save_path,
            label=label,
            columns=columns,
            job_runner=job_runner,
            group_runner=group_runner,
            param_workers=param_workers,
            existing_trial_df=existing_trial_df,
            trial_records=trial_records,
            trial_save_path=trial_save_path,
            trial_columns=trial_columns,
        )

    return save_results(existing_df, records, save_path, label, columns)


def build_experiment_groups(
    strategies: tuple[DiscretizeStrategy, ...],
    n_bins_options: tuple[int, ...],
    supports: np.ndarray,
    limit: int | None = None,
) -> list[ExperimentGroup]:
    """Build worker groups ordered by strategy, n_bins, then support."""

    groups = []
    job_index = 0
    for strategy in strategies:
        for n_bins in n_bins_options:
            support_jobs = []
            for min_support in supports:
                if limit is not None and job_index >= limit:
                    break

                support_jobs.append(
                    ExperimentJob(
                        job_index=job_index,
                        min_support=float(min_support),
                    )
                )
                job_index += 1

                if limit is not None and job_index >= limit:
                    break

            if support_jobs:
                groups.append(
                    ExperimentGroup(
                        group_index=len(groups),
                        strategy=strategy,
                        n_bins=n_bins,
                        jobs=tuple(support_jobs),
                    )
                )

            if limit is not None and job_index >= limit:
                break

        if limit is not None and job_index >= limit:
            break

    return groups


def validate_experiment_grid_resume_source(
    existing_df: pd.DataFrame,
    groups: list[ExperimentGroup],
    label: str,
) -> None:
    """Ensure existing base-grid rows match the current job mapping."""

    expected_by_job = {
        job.job_index: (group.strategy, group.n_bins, job.min_support)
        for group in groups
        for job in group.jobs
    }
    for row in existing_df.itertuples(index=False):
        job_index = int(row.job_index)
        if job_index not in expected_by_job:
            msg = (
                f"Existing {label} output contains job_index={job_index}, "
                "which is outside the current grid. Rerun with --no-resume "
                "or use a separate output file."
            )
            raise ValueError(msg)

        strategy, n_bins, min_support = expected_by_job[job_index]
        same_source = (
            row.strategy == strategy
            and int(row.n_bins) == int(n_bins)
            and np.isclose(float(row.min_support), float(min_support))
        )
        if not same_source:
            msg = (
                f"Existing {label} output has a different grid mapping for "
                f"job_index={job_index}. Rerun with --no-resume or use a "
                "separate output file."
            )
            raise ValueError(msg)


def filter_pending_groups(
    groups: list[ExperimentGroup],
    existing_df: pd.DataFrame,
) -> list[ExperimentGroup]:
    """Remove completed support jobs while keeping group execution order."""

    pending_groups = []
    for group in groups:
        pending_jobs = tuple(
            job for job in group.jobs if not is_job_done(existing_df, job.job_index)
        )
        if pending_jobs:
            pending_groups.append(
                ExperimentGroup(
                    group_index=group.group_index,
                    strategy=group.strategy,
                    n_bins=group.n_bins,
                    jobs=pending_jobs,
                )
            )
    return pending_groups


def run_groups_parallel(
    h_fused_df: pd.DataFrame,
    groups: list[ExperimentGroup],
    n_clusters: int,
    random_state: int,
    baseline_score: float,
    workers: int,
    existing_df: pd.DataFrame,
    records: dict[int, ExperimentRecord],
    save_path: Path,
    label: str,
    columns: list[str],
    job_runner: JobRunner,
    group_runner: GroupRunner,
    param_workers: int,
    existing_trial_df: pd.DataFrame,
    trial_records: dict[
        tuple[int, int | None, str | None, int], NativeIntuitiveTrialRecord
    ],
    trial_save_path: Path | None,
    trial_columns: list[str] | None,
) -> None:
    """Run groups in parallel and save each support result from the main process."""

    import multiprocessing as mp

    with mp.Manager() as manager:
        queue = manager.Queue()
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    group_runner,
                    h_fused_df,
                    group,
                    n_clusters,
                    random_state,
                    baseline_score,
                    job_runner,
                    param_workers,
                    queue,
                ): group
                for group in groups
            }

            while futures:
                done, _ = wait(
                    futures,
                    timeout=0.2,
                    return_when=FIRST_COMPLETED,
                )
                drain_queue(
                    queue,
                    existing_df,
                    records,
                    save_path,
                    label,
                    columns,
                    existing_trial_df,
                    trial_records,
                    trial_save_path,
                    trial_columns,
                )

                for future in done:
                    futures.pop(future)
                    future.result()

                sleep(0.05)

            drain_queue(
                queue,
                existing_df,
                records,
                save_path,
                label,
                columns,
                existing_trial_df,
                trial_records,
                trial_save_path,
                trial_columns,
            )


def run_group_sequentially(
    h_fused_df: pd.DataFrame,
    group: ExperimentGroup,
    n_clusters: int,
    random_state: int,
    baseline_score: float,
    job_runner: JobRunner,
    param_workers: int,
    queue: Any | None = None,
) -> list[ExperimentRecord]:
    """Run support values sequentially for one strategy and bin group."""

    records = []
    for job in group.jobs:
        try:
            record = job_runner(
                h_fused_df,
                group,
                job,
                n_clusters,
                random_state,
                baseline_score,
                param_workers,
            )
        except Exception:
            logger.exception(
                "Skipping failed {} job_index={} strategy={} n_bins={} "
                "min_support={}. It will be retried on the next resume run.",
                getattr(job_runner, "__name__", "experiment"),
                job.job_index,
                group.strategy,
                group.n_bins,
                job.min_support,
            )
            continue
        records.append(record)

        if queue is not None:
            queue.put(record)

    return records


def make_empty_without_ffs_record(
    *,
    group: ExperimentGroup,
    job: ExperimentJob,
    baseline_score: float,
) -> WithoutFfsRecord:
    """Return the baseline row for a support config with no binary features."""

    return WithoutFfsRecord(
        job_index=job.job_index,
        strategy=group.strategy,
        n_bins=group.n_bins,
        min_support=job.min_support,
        silhouette_score=baseline_score,
        n_selected_features=0,
        selected_features="",
        init=None,
        cluster_sizes=[],
        status="baseline_no_features",
        error_message=None,
    )

def make_empty_without_ffs_intuitive_native_record(
    *,
    group: ExperimentGroup,
    job: ExperimentJob,
    baseline_score: float,
) -> WithoutFfsIntuitiveNativeRecord:
    """Return the baseline native-Intuitive row for a config with no features."""

    return WithoutFfsIntuitiveNativeRecord(
        job_index=job.job_index,
        strategy=group.strategy,
        n_bins=group.n_bins,
        min_support=job.min_support,
        silhouette_score=baseline_score,
        n_selected_features=0,
        selected_features="",
        cluster_sizes=[],
        mu_param=None,
        gamma=None,
        beta=None,
        status="baseline_no_features",
        error_message=None,
    )

def make_empty_without_ffs_intuitive_view_weighted_native_record(
    *,
    group: ExperimentGroup,
    job: ExperimentJob,
    baseline_score: float,
) -> WithoutFfsIntuitiveViewWeightedNativeRecord:
    """Return the baseline row for a view-weighted native config."""

    return WithoutFfsIntuitiveViewWeightedNativeRecord(
        job_index=job.job_index,
        strategy=group.strategy,
        n_bins=group.n_bins,
        min_support=job.min_support,
        final_score=baseline_score,
        view_weighted_score=np.nan,
        score_delta=0.0,
        n_selected_features=0,
        selected_features="",
        cluster_sizes=[],
        alpha=None,
        mu_param=None,
        gamma=None,
        beta=None,
        status="baseline_no_features",
        error_message=None,
    )

def make_empty_ffs_intuitive_native_record(
    *,
    group: ExperimentGroup,
    job: ExperimentJob,
    baseline_score: float,
    status: str = "baseline_no_features",
    error_message: str | None = None,
) -> FfsIntuitiveNativeRecord:
    """Return the baseline native-Intuitive row for an FFS config with no features."""

    return FfsIntuitiveNativeRecord(
        job_index=job.job_index,
        strategy=group.strategy,
        n_bins=group.n_bins,
        min_support=job.min_support,
        final_score=baseline_score,
        score_delta=0.0,
        n_selected_features=0,
        selected_features="",
        cluster_sizes=[],
        mu_param=None,
        gamma=None,
        beta=None,
        status=status,
        error_message=error_message,
    )


def make_empty_ffs_intuitive_view_weighted_native_record(
    *,
    group: ExperimentGroup,
    job: ExperimentJob,
    baseline_score: float,
    status: str = "baseline_no_features",
    error_message: str | None = None,
) -> FfsIntuitiveViewWeightedNativeRecord:
    """Return the baseline row for view-weighted native FFS with no features."""

    return FfsIntuitiveViewWeightedNativeRecord(
        job_index=job.job_index,
        strategy=group.strategy,
        n_bins=group.n_bins,
        min_support=job.min_support,
        final_score=baseline_score,
        view_weighted_score=np.nan,
        score_delta=0.0,
        n_selected_features=0,
        selected_features="",
        cluster_sizes=[],
        alpha=None,
        mu_param=None,
        gamma=None,
        beta=None,
        status=status,
        error_message=error_message,
    )

_NATIVE_INTUITIVE_CANDIDATE_CONTEXT: NativeIntuitiveCandidateContext | None = None


def init_native_intuitive_candidate_worker(
    context: NativeIntuitiveCandidateContext,
) -> None:
    """Initialize per-process inputs for native Intuitive FFS candidates."""

    global _NATIVE_INTUITIVE_CANDIDATE_CONTEXT
    _NATIVE_INTUITIVE_CANDIDATE_CONTEXT = context


def run_native_intuitive_candidate(
    candidate: NativeIntuitiveCandidateJob,
) -> NativeIntuitiveCandidateResult:
    """Evaluate one native Intuitive FFS candidate feature set."""

    if _NATIVE_INTUITIVE_CANDIDATE_CONTEXT is None:
        msg = "Native Intuitive candidate worker was not initialized."
        raise RuntimeError(msg)

    context = _NATIVE_INTUITIVE_CANDIDATE_CONTEXT
    candidate_binary_df = context.binary_df[list(candidate.candidate_feature_names)]
    selection = select_native_intuitive_trials(
        h_fused_df=context.continuous_df,
        binary_df=candidate_binary_df,
        n_clusters=context.n_clusters,
        random_state=context.random_state,
        param_workers=context.param_workers,
        intuitive_param_grid=context.intuitive_param_grid,
        two_stage=context.two_stage,
        init_strategy=context.init_strategy,
        strict_init=context.strict_init,
    )
    trace = NativeIntuitiveCandidateTrace(
        selection_step=candidate.selection_step,
        candidate_feature=candidate.candidate_feature,
        candidate_feature_names=candidate.candidate_feature_names,
        trials=selection.trials,
        best_trial=selection.best_trial,
    )
    return NativeIntuitiveCandidateResult(
        candidate_feature=candidate.candidate_feature,
        score=trial_selection_score(selection.best_trial),
        trial=selection.best_trial,
        trace=trace,
    )

def run_native_intuitive_forward_selection(
    *,
    continuous_df: pd.DataFrame,
    binary_df: pd.DataFrame,
    n_clusters: int,
    random_state: int,
    baseline_score: float,
    param_workers: int,
    intuitive_param_grid: tuple[IntuitiveParams, ...] | None = None,
    min_improvement: float = 1e-6,
    two_stage: bool = True,
    init_strategy: str = "farthest_first",
    strict_init: bool = False,
    candidate_workers: int = 1,
) -> NativeIntuitiveForwardSelectionResult:
    """Greedily keep features that improve Intuitive silhouette."""

    if candidate_workers < 1:
        msg = "candidate_workers must be at least 1."
        raise ValueError(msg)
    if candidate_workers > 1 and param_workers > 1:
        msg = "Use either candidate_workers or param_workers above 1, not both."
        raise ValueError(msg)

    remaining = list(binary_df.columns)
    selected_feature_names: list[str] = []
    best_score = float(baseline_score)
    best_trial: NativeIntuitiveTrial | None = None
    intuitive_param_grid = intuitive_param_grid or default_intuitive_param_grid()
    candidate_traces: list[NativeIntuitiveCandidateTrace] = []
    saw_valid_candidate = False
    candidate_context = NativeIntuitiveCandidateContext(
        continuous_df=continuous_df,
        binary_df=binary_df,
        n_clusters=n_clusters,
        random_state=random_state,
        param_workers=param_workers,
        intuitive_param_grid=intuitive_param_grid,
        two_stage=two_stage,
        init_strategy=init_strategy,
        strict_init=strict_init,
    )

    while remaining:
        local_best_feature: str | None = None
        local_best_score = -np.inf
        local_best_trial: NativeIntuitiveTrial | None = None
        selection_step = len(selected_feature_names)
        candidate_jobs = [
            NativeIntuitiveCandidateJob(
                selection_step=selection_step,
                candidate_feature=feature_name,
                candidate_feature_names=tuple(selected_feature_names + [feature_name]),
            )
            for feature_name in remaining
        ]

        if candidate_workers > 1:
            max_workers = min(candidate_workers, len(candidate_jobs))
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=init_native_intuitive_candidate_worker,
                initargs=(candidate_context,),
            ) as executor:
                candidate_results = list(
                    executor.map(run_native_intuitive_candidate, candidate_jobs)
                )
        else:
            init_native_intuitive_candidate_worker(candidate_context)
            candidate_results = [
                run_native_intuitive_candidate(candidate) for candidate in candidate_jobs
            ]

        for result in candidate_results:
            candidate_traces.append(result.trace)
            trial = result.trial
            if trial.status != "ok" or trial.clustering is None:
                continue

            saw_valid_candidate = True
            if result.score > local_best_score:
                local_best_feature = result.candidate_feature
                local_best_score = result.score
                local_best_trial = trial

        if (
            local_best_feature is None
            or local_best_trial is None
            or local_best_score < best_score + min_improvement
        ):
            break

        selected_feature_names.append(local_best_feature)
        remaining.remove(local_best_feature)
        best_score = local_best_score
        best_trial = local_best_trial

    return NativeIntuitiveForwardSelectionResult(
        selected_feature_names=selected_feature_names,
        score=best_score,
        trial=best_trial,
        status=(
            "baseline_no_improvement"
            if saw_valid_candidate or selected_feature_names
            else "baseline_no_valid_candidate"
        ),
        error_message=(
            None
            if saw_valid_candidate or selected_feature_names
            else "No valid Intuitive candidate was available during FFS."
        ),
        candidate_traces=tuple(candidate_traces),
    )

def make_without_ffs_record(
    *,
    group: ExperimentGroup,
    job: ExperimentJob,
    fpmax_features: FpmaxFeatures,
    clustering: MixedClusteringResult,
) -> WithoutFfsRecord:
    """Convert one clustering result into the persisted without-FFS row."""

    has_singleton_cluster = any(size == 1 for size in clustering.cluster_sizes)
    status = "failed_singleton_cluster" if has_singleton_cluster else "ok"
    error_message = (
        "Singleton clusters are not considered valid segmentation results."
        if has_singleton_cluster
        else None
    )

    return WithoutFfsRecord(
        job_index=job.job_index,
        strategy=group.strategy,
        n_bins=group.n_bins,
        min_support=job.min_support,
        silhouette_score=np.nan if has_singleton_cluster else clustering.score,
        n_selected_features=len(clustering.binary_feature_names),
        selected_features=", ".join(clustering.binary_feature_names),
        init=clustering.init,
        cluster_sizes=clustering.cluster_sizes,
        status=status,
        error_message=error_message,
    )


def is_empty_cluster_error(error: Exception) -> bool:
    """Return whether an Intuitive run failed on a paper-undefined empty cluster."""

    return "Empty clusters are not defined by the paper" in str(error)


def native_intuitive_value_error_status(error: Exception) -> str:
    """Return a persisted failed-trial status for expected Intuitive ValueErrors."""

    if is_empty_cluster_error(error):
        return "failed_empty_cluster"
    return "failed_value_error"


_NATIVE_INTUITIVE_CONTEXT: NativeIntuitiveContext | None = None


def init_native_intuitive_worker(context: NativeIntuitiveContext) -> None:
    """Initialize per-process inputs for native Intuitive parameter jobs."""

    global _NATIVE_INTUITIVE_CONTEXT
    _NATIVE_INTUITIVE_CONTEXT = context


def run_native_intuitive_trial(job: ExperimentJob) -> NativeIntuitiveTrial:
    """Run one Intuitive parameter trial on a fixed mixed feature set."""

    if _NATIVE_INTUITIVE_CONTEXT is None:
        msg = "Native Intuitive worker was not initialized."
        raise RuntimeError(msg)
    if job.intuitive_params is None:
        msg = "intuitive_params is required for native Intuitive jobs."
        raise ValueError(msg)

    context = _NATIVE_INTUITIVE_CONTEXT
    alpha = job.intuitive_params.view_weight_alpha
    view_weighted_distance_matrix = (
        context.view_weighted_distance_matrices.get(alpha)
        if alpha is not None and context.view_weighted_distance_matrices is not None
        else None
    )
    try:
        clustering = run_intuitive_kprototypes(
            continuous_df=context.h_fused_df,
            binary_df=context.binary_df,
            n_clusters=context.n_clusters,
            random_state=context.random_state,
            mu_param=job.intuitive_params.mu_param,
            gamma=job.intuitive_params.gamma,
            beta=job.intuitive_params.beta,
            distance_matrix=context.distance_matrix,
            view_weight_alpha=alpha,
            view_weighted_distance_matrix=view_weighted_distance_matrix,
            init_strategy=context.init_strategy,
            strict_init=context.strict_init,
            verbose=False,
        )
    except ValueError as error:
        return NativeIntuitiveTrial(
            clustering=None,
            mu_param=job.intuitive_params.mu_param,
            gamma=job.intuitive_params.gamma,
            beta=job.intuitive_params.beta,
            status=native_intuitive_value_error_status(error),
            error_message=str(error),
            view_weight_alpha=alpha,
        )

    if any(size == 1 for size in clustering.cluster_sizes):
        return NativeIntuitiveTrial(
            clustering=clustering,
            mu_param=job.intuitive_params.mu_param,
            gamma=job.intuitive_params.gamma,
            beta=job.intuitive_params.beta,
            status="failed_singleton_cluster",
            error_message=(
                "Singleton clusters are not considered valid segmentation results."
            ),
            view_weight_alpha=alpha,
            view_weighted_score=clustering.view_weighted_score,
        )

    return NativeIntuitiveTrial(
        clustering=clustering,
        mu_param=job.intuitive_params.mu_param,
        gamma=job.intuitive_params.gamma,
        beta=job.intuitive_params.beta,
        status="ok",
        error_message=None,
        view_weight_alpha=alpha,
        view_weighted_score=clustering.view_weighted_score,
    )


def run_native_intuitive_trials(
    *,
    context: NativeIntuitiveContext,
    intuitive_param_grid: tuple[IntuitiveParams, ...],
    param_workers: int,
    stage: str = "",
) -> list[NativeIntuitiveTrial]:
    """Run a native Intuitive parameter batch on a fixed feature set."""

    jobs = [
        ExperimentJob(
            job_index=param_index,
            min_support=0.0,
            intuitive_params=intuitive_params,
        )
        for param_index, intuitive_params in enumerate(intuitive_param_grid)
    ]
    if not jobs:
        return []

    if param_workers > 1:
        max_workers = min(param_workers, len(jobs))
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=init_native_intuitive_worker,
            initargs=(context,),
        ) as executor:
            return [
                replace(trial, stage=stage)
                for trial in executor.map(run_native_intuitive_trial, jobs)
            ]

    init_native_intuitive_worker(context)
    return [replace(run_native_intuitive_trial(job), stage=stage) for job in jobs]

def select_native_intuitive_trials(
    *,
    h_fused_df: pd.DataFrame,
    binary_df: pd.DataFrame,
    n_clusters: int,
    random_state: int,
    param_workers: int,
    intuitive_param_grid: tuple[IntuitiveParams, ...] | None = None,
    two_stage: bool = True,
    init_strategy: str = "farthest_first",
    strict_init: bool = False,
) -> NativeIntuitiveSelectionResult:
    """Evaluate Intuitive params and return the best valid native trial."""

    intuitive_param_grid = intuitive_param_grid or default_intuitive_param_grid()
    intuitive_param_grid = dedupe_intuitive_param_grid(intuitive_param_grid)
    combined_df, _, _ = make_mixed_features(
        continuous_df=h_fused_df,
        binary_df=binary_df,
    )
    distance_matrix = compute_gower_distance(combined_df)
    view_alphas = sorted(
        {
            params.view_weight_alpha
            for params in intuitive_param_grid
            if params.view_weight_alpha is not None
        }
    )
    view_weighted_distance_matrices = (
        {
            alpha: compute_view_weighted_gower_distance(
                h_fused_df,
                binary_df,
                alpha,
            )
            for alpha in view_alphas
        }
        if view_alphas
        else None
    )
    context = NativeIntuitiveContext(
        h_fused_df=h_fused_df,
        binary_df=binary_df,
        distance_matrix=distance_matrix,
        n_clusters=n_clusters,
        random_state=random_state,
        init_strategy=init_strategy,
        strict_init=strict_init,
        view_weighted_distance_matrices=view_weighted_distance_matrices,
    )

    if two_stage:
        broad_grid, _ = build_two_stage_intuitive_param_grids(
            intuitive_param_grid,
            [],
        )
    else:
        broad_grid = intuitive_param_grid

    trials = run_native_intuitive_trials(
        context=context,
        intuitive_param_grid=broad_grid,
        param_workers=param_workers,
        stage=("coarse" if two_stage else "exhaustive"),
    )
    ok_trials = [
        trial
        for trial in trials
        if trial.status == "ok" and trial.clustering is not None
    ]

    if two_stage:
        if ok_trials:
            top_trials = sorted(
                ok_trials,
                key=trial_selection_score,
                reverse=True,
            )
            _, refined_grid = build_two_stage_intuitive_param_grids(
                intuitive_param_grid,
                top_trials,
            )
        else:
            refined_grid = tuple(
                params for params in intuitive_param_grid if params not in broad_grid
            )

        trials.extend(
            run_native_intuitive_trials(
                context=context,
                intuitive_param_grid=refined_grid,
                param_workers=param_workers,
                stage="refine",
            )
        )
        ok_trials = [
            trial
            for trial in trials
            if trial.status == "ok" and trial.clustering is not None
        ]

    if ok_trials:
        return NativeIntuitiveSelectionResult(
            best_trial=max(ok_trials, key=trial_selection_score),
            trials=tuple(trials),
        )

    failed_trials = [trial for trial in trials if trial.error_message]
    error_message = "No Intuitive parameter combination produced a valid result."
    if failed_trials:
        first_failure = failed_trials[0]
        error_message = (
            f"{error_message} First failure: {first_failure.status}: "
            f"{first_failure.error_message}"
        )

    failed_best = NativeIntuitiveTrial(
        clustering=None,
        mu_param=None,
        gamma=None,
        beta=None,
        status="failed_all_params",
        error_message=error_message,
    )
    return NativeIntuitiveSelectionResult(
        best_trial=failed_best,
        trials=tuple(trials),
    )

def select_best_native_intuitive_trial(
    *,
    h_fused_df: pd.DataFrame,
    binary_df: pd.DataFrame,
    n_clusters: int,
    random_state: int,
    param_workers: int,
    intuitive_param_grid: tuple[IntuitiveParams, ...] | None = None,
    two_stage: bool = True,
    init_strategy: str = "farthest_first",
    strict_init: bool = False,
) -> NativeIntuitiveTrial:
    """Evaluate Intuitive params and return the best valid native trial."""

    return select_native_intuitive_trials(
        h_fused_df=h_fused_df,
        binary_df=binary_df,
        n_clusters=n_clusters,
        random_state=random_state,
        param_workers=param_workers,
        intuitive_param_grid=intuitive_param_grid,
        two_stage=two_stage,
        init_strategy=init_strategy,
        strict_init=strict_init,
    ).best_trial

_POST_FFS_INTUITIVE_CONTEXT: PostFfsIntuitiveContext | None = None

def init_post_ffs_intuitive_worker(context: PostFfsIntuitiveContext) -> None:
    """Initialize per-process inputs for post-FFS Intuitive parameter jobs."""

    global _POST_FFS_INTUITIVE_CONTEXT
    _POST_FFS_INTUITIVE_CONTEXT = context

def make_post_ffs_intuitive_record(
    *,
    job: ExperimentJob,
    context: PostFfsIntuitiveContext,
    clustering: MixedClusteringResult | None,
    status: str,
    error_message: str | None,
) -> PostFfsIntuitiveRecord:
    """Convert one post-FFS Intuitive result into a persisted row."""

    if job.intuitive_params is None:
        msg = "intuitive_params is required for post-FFS Intuitive jobs."
        raise ValueError(msg)

    has_valid_score = status == "ok" and clustering is not None
    intuitive_score = clustering.score if has_valid_score else np.nan
    view_weighted_score = (
        clustering.view_weighted_score
        if has_valid_score and clustering.view_weighted_score is not None
        else np.nan
    )
    return PostFfsIntuitiveRecord(
        job_index=job.job_index,
        ffs_job_index=context.ffs_job_index,
        strategy=context.strategy,
        n_bins=context.n_bins,
        min_support=context.min_support,
        ffs_score=context.ffs_score,
        intuitive_score=intuitive_score,
        score_delta=intuitive_score - context.ffs_score,
        n_selected_features=context.n_selected_features,
        selected_features=context.selected_features,
        cluster_sizes=(clustering.cluster_sizes if clustering else []),
        mu_param=job.intuitive_params.mu_param,
        gamma=job.intuitive_params.gamma,
        beta=job.intuitive_params.beta,
        status=status,
        error_message=error_message,
        alpha=job.intuitive_params.view_weight_alpha,
        view_weighted_score=view_weighted_score,
    )

def run_post_ffs_intuitive_job(job: ExperimentJob) -> PostFfsIntuitiveRecord:
    """Run one Intuitive parameter config on the selected FFS feature set."""

    if _POST_FFS_INTUITIVE_CONTEXT is None:
        msg = "Post-FFS Intuitive worker was not initialized."
        raise RuntimeError(msg)
    if job.intuitive_params is None:
        msg = "intuitive_params is required for post-FFS Intuitive jobs."
        raise ValueError(msg)

    context = _POST_FFS_INTUITIVE_CONTEXT
    alpha = job.intuitive_params.view_weight_alpha
    view_weighted_distance_matrix = (
        context.view_weighted_distance_matrices.get(alpha)
        if alpha is not None and context.view_weighted_distance_matrices is not None
        else None
    )
    try:
        clustering = run_intuitive_kprototypes(
            continuous_df=context.h_fused_df,
            binary_df=context.binary_df,
            n_clusters=context.n_clusters,
            random_state=context.random_state,
            mu_param=job.intuitive_params.mu_param,
            gamma=job.intuitive_params.gamma,
            beta=job.intuitive_params.beta,
            distance_matrix=context.distance_matrix,
            view_weight_alpha=alpha,
            view_weighted_distance_matrix=view_weighted_distance_matrix,
            verbose=False,
        )
    except ValueError as error:
        if not is_empty_cluster_error(error):
            raise
        return make_post_ffs_intuitive_record(
            job=job,
            context=context,
            clustering=None,
            status="failed_empty_cluster",
            error_message=str(error),
        )

    if any(size == 1 for size in clustering.cluster_sizes):
        return make_post_ffs_intuitive_record(
            job=job,
            context=context,
            clustering=clustering,
            status="failed_singleton_cluster",
            error_message=(
                "Singleton clusters are not considered valid segmentation results."
            ),
        )

    return make_post_ffs_intuitive_record(
        job=job,
        context=context,
        clustering=clustering,
        status="ok",
        error_message=None,
    )

def run_post_ffs_intuitive_job_safely(
    job: ExperimentJob,
) -> PostFfsIntuitiveRecord | None:
    """Run one post-FFS job, leaving unexpected failures for resume retry."""

    try:
        return run_post_ffs_intuitive_job(job)
    except Exception:
        logger.exception(
            "Skipping failed post-FFS Intuitive job_index={}. It will be "
            "retried on the next resume run.",
            job.job_index,
        )
        return None

_POST_WITHOUT_FFS_INTUITIVE_CONTEXT: PostWithoutFfsIntuitiveContext | None = None

def init_post_without_ffs_intuitive_worker(
    context: PostWithoutFfsIntuitiveContext,
) -> None:
    """Initialize per-process inputs for post-without-FFS Intuitive jobs."""

    global _POST_WITHOUT_FFS_INTUITIVE_CONTEXT
    _POST_WITHOUT_FFS_INTUITIVE_CONTEXT = context

def make_post_without_ffs_intuitive_record(
    *,
    job: ExperimentJob,
    context: PostWithoutFfsIntuitiveContext,
    clustering: MixedClusteringResult | None,
    status: str,
    error_message: str | None,
) -> PostWithoutFfsIntuitiveRecord:
    """Convert one post-without-FFS Intuitive result into a persisted row."""

    if job.intuitive_params is None:
        msg = "intuitive_params is required for post-without-FFS Intuitive jobs."
        raise ValueError(msg)

    has_valid_score = status == "ok" and clustering is not None
    intuitive_score = clustering.score if has_valid_score else np.nan
    view_weighted_score = (
        clustering.view_weighted_score
        if has_valid_score and clustering.view_weighted_score is not None
        else np.nan
    )
    return PostWithoutFfsIntuitiveRecord(
        job_index=job.job_index,
        without_ffs_job_index=context.without_ffs_job_index,
        strategy=context.strategy,
        n_bins=context.n_bins,
        min_support=context.min_support,
        kprototypes_score=context.kprototypes_score,
        intuitive_score=intuitive_score,
        score_delta=intuitive_score - context.kprototypes_score,
        n_selected_features=context.n_selected_features,
        selected_features=context.selected_features,
        cluster_sizes=(clustering.cluster_sizes if clustering else []),
        mu_param=job.intuitive_params.mu_param,
        gamma=job.intuitive_params.gamma,
        beta=job.intuitive_params.beta,
        status=status,
        error_message=error_message,
        alpha=job.intuitive_params.view_weight_alpha,
        view_weighted_score=view_weighted_score,
    )

def run_post_without_ffs_intuitive_job(
    job: ExperimentJob,
) -> PostWithoutFfsIntuitiveRecord:
    """Run one Intuitive parameter config on the best without-FFS feature set."""

    if _POST_WITHOUT_FFS_INTUITIVE_CONTEXT is None:
        msg = "Post-without-FFS Intuitive worker was not initialized."
        raise RuntimeError(msg)
    if job.intuitive_params is None:
        msg = "intuitive_params is required for post-without-FFS Intuitive jobs."
        raise ValueError(msg)

    context = _POST_WITHOUT_FFS_INTUITIVE_CONTEXT
    alpha = job.intuitive_params.view_weight_alpha
    view_weighted_distance_matrix = (
        context.view_weighted_distance_matrices.get(alpha)
        if alpha is not None and context.view_weighted_distance_matrices is not None
        else None
    )
    try:
        clustering = run_intuitive_kprototypes(
            continuous_df=context.h_fused_df,
            binary_df=context.binary_df,
            n_clusters=context.n_clusters,
            random_state=context.random_state,
            mu_param=job.intuitive_params.mu_param,
            gamma=job.intuitive_params.gamma,
            beta=job.intuitive_params.beta,
            distance_matrix=context.distance_matrix,
            view_weight_alpha=alpha,
            view_weighted_distance_matrix=view_weighted_distance_matrix,
            verbose=False,
        )
    except ValueError as error:
        if not is_empty_cluster_error(error):
            raise
        return make_post_without_ffs_intuitive_record(
            job=job,
            context=context,
            clustering=None,
            status="failed_empty_cluster",
            error_message=str(error),
        )

    if any(size == 1 for size in clustering.cluster_sizes):
        return make_post_without_ffs_intuitive_record(
            job=job,
            context=context,
            clustering=clustering,
            status="failed_singleton_cluster",
            error_message=(
                "Singleton clusters are not considered valid segmentation results."
            ),
        )

    return make_post_without_ffs_intuitive_record(
        job=job,
        context=context,
        clustering=clustering,
        status="ok",
        error_message=None,
    )

def run_post_without_ffs_intuitive_job_safely(
    job: ExperimentJob,
) -> PostWithoutFfsIntuitiveRecord | None:
    """Run one post-without-FFS job and retry unexpected failures later."""

    try:
        return run_post_without_ffs_intuitive_job(job)
    except Exception:
        logger.exception(
            "Skipping failed post-without-FFS Intuitive job_index={}. It will "
            "be retried on the next resume run.",
            job.job_index,
        )
        return None


def emit_record(
    record: ExperimentRecord,
    records: list[ExperimentRecord],
    queue: Any | None,
) -> None:
    """Store a completed record locally and optionally publish it to the parent."""

    records.append(record)
    if queue is not None:
        queue.put(record)


def run_without_ffs_job(
    h_fused_df: pd.DataFrame,
    group: ExperimentGroup,
    job: ExperimentJob,
    n_clusters: int,
    random_state: int,
    baseline_score: float,
    param_workers: int,
) -> WithoutFfsRecord:
    """Run one without-FFS K-Prototypes support configuration."""

    del param_workers

    logger.info(
        "Running without-FFS: strategy={} n_bins={} min_support={}",
        group.strategy,
        group.n_bins,
        job.min_support,
    )
    fpmax_features = extract_fpmax_features(
        df=h_fused_df,
        n_bins=group.n_bins,
        bin_labels=BIN_LABELS_BY_SIZE[group.n_bins],
        strategy=group.strategy,
        min_support=job.min_support,
        drop_original_numeric=True,
    )
    if fpmax_features.features.empty:
        return make_empty_without_ffs_record(
            group=group,
            job=job,
            baseline_score=baseline_score,
        )

    clustering = run_kprototypes(
        continuous_df=h_fused_df,
        binary_df=fpmax_features.features,
        n_clusters=n_clusters,
        random_state=random_state,
        verbose=False,
    )

    return make_without_ffs_record(
        group=group,
        job=job,
        fpmax_features=fpmax_features,
        clustering=clustering,
    )


def run_without_ffs_intuitive_native_job(
    h_fused_df: pd.DataFrame,
    group: ExperimentGroup,
    job: ExperimentJob,
    n_clusters: int,
    random_state: int,
    baseline_score: float,
    param_workers: int,
    two_stage: bool = True,
    init_strategy: str = "farthest_first",
    strict_init: bool = False,
) -> WithoutFfsIntuitiveNativeRecord:
    """Run Intuitive directly on one without-FFS FP-Max support configuration."""

    logger.info(
        "Running without-FFS/intuitive-native: strategy={} n_bins={} min_support={}",
        group.strategy,
        group.n_bins,
        job.min_support,
    )
    fpmax_features = extract_fpmax_features(
        df=h_fused_df,
        n_bins=group.n_bins,
        bin_labels=BIN_LABELS_BY_SIZE[group.n_bins],
        strategy=group.strategy,
        min_support=job.min_support,
        drop_original_numeric=True,
    )
    if fpmax_features.features.empty:
        return make_empty_without_ffs_intuitive_native_record(
            group=group,
            job=job,
            baseline_score=baseline_score,
        )

    selection = select_native_intuitive_trials(
        h_fused_df=h_fused_df,
        binary_df=fpmax_features.features,
        n_clusters=n_clusters,
        random_state=random_state,
        param_workers=param_workers,
        two_stage=two_stage,
        init_strategy=init_strategy,
        strict_init=strict_init,
    )
    best_trial = selection.best_trial
    clustering = best_trial.clustering
    trial_records = make_without_ffs_intuitive_native_trial_records(
        group=group,
        job=job,
        fpmax_features=fpmax_features,
        trials=selection.trials,
        best_trial=best_trial,
    )
    return WithoutFfsIntuitiveNativeRecord(
        job_index=job.job_index,
        strategy=group.strategy,
        n_bins=group.n_bins,
        min_support=job.min_support,
        silhouette_score=(clustering.score if clustering else np.nan),
        n_selected_features=len(fpmax_features.features.columns),
        selected_features=", ".join(fpmax_features.features.columns),
        cluster_sizes=(clustering.cluster_sizes if clustering else []),
        mu_param=best_trial.mu_param,
        gamma=best_trial.gamma,
        beta=best_trial.beta,
        status=best_trial.status,
        error_message=best_trial.error_message,
        trial_records=trial_records,
    )

def run_without_ffs_intuitive_view_weighted_native_job(
    h_fused_df: pd.DataFrame,
    group: ExperimentGroup,
    job: ExperimentJob,
    n_clusters: int,
    random_state: int,
    baseline_score: float,
    param_workers: int,
    two_stage: bool = True,
    init_strategy: str = "farthest_first",
    strict_init: bool = False,
) -> WithoutFfsIntuitiveViewWeightedNativeRecord:
    """Run view-weighted Intuitive on one without-FFS FP-Max config."""

    logger.info(
        "Running without-FFS/intuitive-view-weighted-native: "
        "strategy={} n_bins={} min_support={}",
        group.strategy,
        group.n_bins,
        job.min_support,
    )
    fpmax_features = extract_fpmax_features(
        df=h_fused_df,
        n_bins=group.n_bins,
        bin_labels=BIN_LABELS_BY_SIZE[group.n_bins],
        strategy=group.strategy,
        min_support=job.min_support,
        drop_original_numeric=True,
    )
    if fpmax_features.features.empty:
        return make_empty_without_ffs_intuitive_view_weighted_native_record(
            group=group,
            job=job,
            baseline_score=baseline_score,
        )

    selection = select_native_intuitive_trials(
        h_fused_df=h_fused_df,
        binary_df=fpmax_features.features,
        n_clusters=n_clusters,
        random_state=random_state,
        param_workers=param_workers,
        intuitive_param_grid=default_view_weighted_intuitive_param_grid(),
        two_stage=two_stage,
        init_strategy=init_strategy,
        strict_init=strict_init,
    )
    best_trial = selection.best_trial
    clustering = best_trial.clustering
    final_score = trial_selection_score(best_trial) if clustering else np.nan
    trial_records = make_without_ffs_intuitive_native_trial_records(
        group=group,
        job=job,
        fpmax_features=fpmax_features,
        trials=selection.trials,
        best_trial=best_trial,
    )
    return WithoutFfsIntuitiveViewWeightedNativeRecord(
        job_index=job.job_index,
        strategy=group.strategy,
        n_bins=group.n_bins,
        min_support=job.min_support,
        final_score=final_score,
        view_weighted_score=(
            best_trial.view_weighted_score
            if best_trial.view_weighted_score is not None
            else np.nan
        ),
        score_delta=final_score - baseline_score,
        n_selected_features=len(fpmax_features.features.columns),
        selected_features=", ".join(fpmax_features.features.columns),
        cluster_sizes=(clustering.cluster_sizes if clustering else []),
        alpha=best_trial.view_weight_alpha,
        mu_param=best_trial.mu_param,
        gamma=best_trial.gamma,
        beta=best_trial.beta,
        status=best_trial.status,
        error_message=best_trial.error_message,
        trial_records=trial_records,
    )

def run_ffs_job(
    h_fused_df: pd.DataFrame,
    group: ExperimentGroup,
    job: ExperimentJob,
    n_clusters: int,
    random_state: int,
    baseline_score: float,
    param_workers: int,
    candidate_workers: int = 1,
) -> FfsRecord:
    """Run one support value with forward feature selection."""

    del param_workers

    logger.info(
        "Running FFS: strategy={} n_bins={} min_support={}",
        group.strategy,
        group.n_bins,
        job.min_support,
    )
    fpmax_features = extract_fpmax_features(
        df=h_fused_df,
        n_bins=group.n_bins,
        bin_labels=BIN_LABELS_BY_SIZE[group.n_bins],
        strategy=group.strategy,
        min_support=job.min_support,
        drop_original_numeric=True,
    )
    selected = run_forward_selection(
        continuous_df=h_fused_df,
        binary_df=fpmax_features.features,
        n_clusters=n_clusters,
        random_state=random_state,
        baseline_score=baseline_score,
        verbose=False,
        candidate_workers=candidate_workers,
    )
    sizes = (
        cluster_sizes(selected.labels, n_clusters)
        if selected.labels is not None
        else []
    )
    has_singleton_cluster = any(size == 1 for size in sizes)
    if has_singleton_cluster:
        final_score = np.nan
        status = "failed_singleton_cluster"
        error_message = (
            "Singleton clusters are not considered valid segmentation results."
        )
    elif selected.labels is None:
        final_score = baseline_score
        status = (
            "baseline_no_features"
            if fpmax_features.features.empty
            else "baseline_no_improvement"
        )
        error_message = None
    else:
        final_score = selected.score
        status = "ok"
        error_message = None

    return FfsRecord(
        job_index=job.job_index,
        strategy=group.strategy,
        n_bins=group.n_bins,
        min_support=job.min_support,
        final_score=final_score,
        n_selected_features=len(selected.selected_feature_names),
        selected_features=", ".join(selected.selected_feature_names),
        init=selected.init,
        cluster_sizes=sizes,
        status=status,
        error_message=error_message,
    )


def run_ffs_intuitive_native_job(
    h_fused_df: pd.DataFrame,
    group: ExperimentGroup,
    job: ExperimentJob,
    n_clusters: int,
    random_state: int,
    baseline_score: float,
    param_workers: int,
    two_stage: bool = True,
    init_strategy: str = "farthest_first",
    strict_init: bool = False,
    candidate_workers: int = 1,
) -> FfsIntuitiveNativeRecord:
    """Run forward feature selection with Intuitive as the evaluator."""

    logger.info(
        "Running FFS/intuitive-native: strategy={} n_bins={} min_support={}",
        group.strategy,
        group.n_bins,
        job.min_support,
    )
    fpmax_features = extract_fpmax_features(
        df=h_fused_df,
        n_bins=group.n_bins,
        bin_labels=BIN_LABELS_BY_SIZE[group.n_bins],
        strategy=group.strategy,
        min_support=job.min_support,
        drop_original_numeric=True,
    )
    selected = run_native_intuitive_forward_selection(
        continuous_df=h_fused_df,
        binary_df=fpmax_features.features,
        n_clusters=n_clusters,
        random_state=random_state,
        baseline_score=baseline_score,
        param_workers=param_workers,
        two_stage=two_stage,
        init_strategy=init_strategy,
        strict_init=strict_init,
        candidate_workers=candidate_workers,
    )
    if not selected.selected_feature_names:
        status = (
            "baseline_no_features"
            if fpmax_features.features.empty
            else selected.status
        )
        error_message = (
            None if fpmax_features.features.empty else selected.error_message
        )
        trial_records = make_ffs_intuitive_native_trial_records(
            group=group,
            job=job,
            candidate_traces=selected.candidate_traces,
            final_best_trial=selected.trial,
        )
        return replace(
            make_empty_ffs_intuitive_native_record(
                group=group,
                job=job,
                baseline_score=baseline_score,
                status=status,
                error_message=error_message,
            ),
            trial_records=trial_records,
        )

    best_trial = selected.trial
    clustering = best_trial.clustering if best_trial else None
    final_score = selected.score if clustering else np.nan
    trial_records = make_ffs_intuitive_native_trial_records(
        group=group,
        job=job,
        candidate_traces=selected.candidate_traces,
        final_best_trial=best_trial,
    )
    return FfsIntuitiveNativeRecord(
        job_index=job.job_index,
        strategy=group.strategy,
        n_bins=group.n_bins,
        min_support=job.min_support,
        final_score=final_score,
        score_delta=final_score - baseline_score,
        n_selected_features=len(selected.selected_feature_names),
        selected_features=", ".join(selected.selected_feature_names),
        cluster_sizes=(clustering.cluster_sizes if clustering else []),
        mu_param=(best_trial.mu_param if best_trial else None),
        gamma=(best_trial.gamma if best_trial else None),
        beta=(best_trial.beta if best_trial else None),
        status=(best_trial.status if best_trial else "failed_all_params"),
        error_message=(
            best_trial.error_message
            if best_trial
            else "No Intuitive parameter combination produced a valid result."
        ),
        trial_records=trial_records,
    )


def run_ffs_intuitive_view_weighted_native_job(
    h_fused_df: pd.DataFrame,
    group: ExperimentGroup,
    job: ExperimentJob,
    n_clusters: int,
    random_state: int,
    baseline_score: float,
    param_workers: int,
    two_stage: bool = True,
    init_strategy: str = "farthest_first",
    strict_init: bool = False,
    candidate_workers: int = 1,
) -> FfsIntuitiveViewWeightedNativeRecord:
    """Run FFS with view-weighted Intuitive as the evaluator."""

    logger.info(
        "Running FFS/intuitive-view-weighted-native: "
        "strategy={} n_bins={} min_support={}",
        group.strategy,
        group.n_bins,
        job.min_support,
    )
    fpmax_features = extract_fpmax_features(
        df=h_fused_df,
        n_bins=group.n_bins,
        bin_labels=BIN_LABELS_BY_SIZE[group.n_bins],
        strategy=group.strategy,
        min_support=job.min_support,
        drop_original_numeric=True,
    )
    selected = run_native_intuitive_forward_selection(
        continuous_df=h_fused_df,
        binary_df=fpmax_features.features,
        n_clusters=n_clusters,
        random_state=random_state,
        baseline_score=baseline_score,
        param_workers=param_workers,
        intuitive_param_grid=default_view_weighted_intuitive_param_grid(),
        two_stage=two_stage,
        init_strategy=init_strategy,
        strict_init=strict_init,
        candidate_workers=candidate_workers,
    )
    if not selected.selected_feature_names:
        status = (
            "baseline_no_features"
            if fpmax_features.features.empty
            else selected.status
        )
        error_message = (
            None if fpmax_features.features.empty else selected.error_message
        )
        trial_records = make_ffs_intuitive_native_trial_records(
            group=group,
            job=job,
            candidate_traces=selected.candidate_traces,
            final_best_trial=selected.trial,
        )
        return replace(
            make_empty_ffs_intuitive_view_weighted_native_record(
                group=group,
                job=job,
                baseline_score=baseline_score,
                status=status,
                error_message=error_message,
            ),
            trial_records=trial_records,
        )

    best_trial = selected.trial
    clustering = best_trial.clustering if best_trial else None
    final_score = selected.score if clustering else np.nan
    trial_records = make_ffs_intuitive_native_trial_records(
        group=group,
        job=job,
        candidate_traces=selected.candidate_traces,
        final_best_trial=best_trial,
    )
    return FfsIntuitiveViewWeightedNativeRecord(
        job_index=job.job_index,
        strategy=group.strategy,
        n_bins=group.n_bins,
        min_support=job.min_support,
        final_score=final_score,
        view_weighted_score=(
            best_trial.view_weighted_score
            if best_trial and best_trial.view_weighted_score is not None
            else np.nan
        ),
        score_delta=final_score - baseline_score,
        n_selected_features=len(selected.selected_feature_names),
        selected_features=", ".join(selected.selected_feature_names),
        cluster_sizes=(clustering.cluster_sizes if clustering else []),
        alpha=(best_trial.view_weight_alpha if best_trial else None),
        mu_param=(best_trial.mu_param if best_trial else None),
        gamma=(best_trial.gamma if best_trial else None),
        beta=(best_trial.beta if best_trial else None),
        status=(best_trial.status if best_trial else "failed_all_params"),
        error_message=(
            best_trial.error_message
            if best_trial
            else "No Intuitive parameter combination produced a valid result."
        ),
        trial_records=trial_records,
    )

def run_without_ffs_intuitive_exhaustive_native_job(
    h_fused_df: pd.DataFrame,
    group: ExperimentGroup,
    job: ExperimentJob,
    n_clusters: int,
    random_state: int,
    baseline_score: float,
    param_workers: int,
) -> WithoutFfsIntuitiveNativeRecord:
    """Run exhaustive Intuitive directly on one without-FFS config."""

    return run_without_ffs_intuitive_native_job(
        h_fused_df,
        group,
        job,
        n_clusters,
        random_state,
        baseline_score,
        param_workers,
        two_stage=False,
    )

def run_without_ffs_intuitive_view_weighted_exhaustive_native_job(
    h_fused_df: pd.DataFrame,
    group: ExperimentGroup,
    job: ExperimentJob,
    n_clusters: int,
    random_state: int,
    baseline_score: float,
    param_workers: int,
) -> WithoutFfsIntuitiveViewWeightedNativeRecord:
    """Run exhaustive view-weighted Intuitive on one without-FFS config."""

    return run_without_ffs_intuitive_view_weighted_native_job(
        h_fused_df,
        group,
        job,
        n_clusters,
        random_state,
        baseline_score,
        param_workers,
        two_stage=False,
    )

def run_ffs_intuitive_exhaustive_native_job(
    h_fused_df: pd.DataFrame,
    group: ExperimentGroup,
    job: ExperimentJob,
    n_clusters: int,
    random_state: int,
    baseline_score: float,
    param_workers: int,
    candidate_workers: int = 1,
) -> FfsIntuitiveNativeRecord:
    """Run FFS with exhaustive native Intuitive evaluator."""

    return run_ffs_intuitive_native_job(
        h_fused_df,
        group,
        job,
        n_clusters,
        random_state,
        baseline_score,
        param_workers,
        two_stage=False,
        candidate_workers=candidate_workers,
    )

def run_ffs_intuitive_view_weighted_exhaustive_native_job(
    h_fused_df: pd.DataFrame,
    group: ExperimentGroup,
    job: ExperimentJob,
    n_clusters: int,
    random_state: int,
    baseline_score: float,
    param_workers: int,
    candidate_workers: int = 1,
) -> FfsIntuitiveViewWeightedNativeRecord:
    """Run FFS with exhaustive view-weighted native Intuitive evaluator."""

    return run_ffs_intuitive_view_weighted_native_job(
        h_fused_df,
        group,
        job,
        n_clusters,
        random_state,
        baseline_score,
        param_workers,
        two_stage=False,
        candidate_workers=candidate_workers,
    )

def run_without_ffs_intuitive_paper_native_job(
    h_fused_df: pd.DataFrame,
    group: ExperimentGroup,
    job: ExperimentJob,
    n_clusters: int,
    random_state: int,
    baseline_score: float,
    param_workers: int,
) -> WithoutFfsIntuitiveNativeRecord:
    """Run native Intuitive with Algorithm-1 paper initialization."""

    return run_without_ffs_intuitive_native_job(
        h_fused_df,
        group,
        job,
        n_clusters,
        random_state,
        baseline_score,
        param_workers,
        init_strategy="paper",
        strict_init=True,
    )

def run_without_ffs_intuitive_paper_exhaustive_native_job(
    h_fused_df: pd.DataFrame,
    group: ExperimentGroup,
    job: ExperimentJob,
    n_clusters: int,
    random_state: int,
    baseline_score: float,
    param_workers: int,
) -> WithoutFfsIntuitiveNativeRecord:
    """Run exhaustive native Intuitive with paper initialization."""

    return run_without_ffs_intuitive_native_job(
        h_fused_df,
        group,
        job,
        n_clusters,
        random_state,
        baseline_score,
        param_workers,
        two_stage=False,
        init_strategy="paper",
        strict_init=True,
    )

def run_without_ffs_intuitive_view_weighted_paper_native_job(
    h_fused_df: pd.DataFrame,
    group: ExperimentGroup,
    job: ExperimentJob,
    n_clusters: int,
    random_state: int,
    baseline_score: float,
    param_workers: int,
) -> WithoutFfsIntuitiveViewWeightedNativeRecord:
    """Run view-weighted native Intuitive with paper initialization."""

    return run_without_ffs_intuitive_view_weighted_native_job(
        h_fused_df,
        group,
        job,
        n_clusters,
        random_state,
        baseline_score,
        param_workers,
        init_strategy="paper",
        strict_init=True,
    )

def run_without_ffs_intuitive_view_weighted_paper_exhaustive_native_job(
    h_fused_df: pd.DataFrame,
    group: ExperimentGroup,
    job: ExperimentJob,
    n_clusters: int,
    random_state: int,
    baseline_score: float,
    param_workers: int,
) -> WithoutFfsIntuitiveViewWeightedNativeRecord:
    """Run exhaustive view-weighted native Intuitive with paper init."""

    return run_without_ffs_intuitive_view_weighted_native_job(
        h_fused_df,
        group,
        job,
        n_clusters,
        random_state,
        baseline_score,
        param_workers,
        two_stage=False,
        init_strategy="paper",
        strict_init=True,
    )

def run_ffs_intuitive_paper_native_job(
    h_fused_df: pd.DataFrame,
    group: ExperimentGroup,
    job: ExperimentJob,
    n_clusters: int,
    random_state: int,
    baseline_score: float,
    param_workers: int,
    candidate_workers: int = 1,
) -> FfsIntuitiveNativeRecord:
    """Run Intuitive-native FFS with Algorithm-1 paper initialization."""

    return run_ffs_intuitive_native_job(
        h_fused_df,
        group,
        job,
        n_clusters,
        random_state,
        baseline_score,
        param_workers,
        init_strategy="paper",
        strict_init=True,
        candidate_workers=candidate_workers,
    )

def run_ffs_intuitive_paper_exhaustive_native_job(
    h_fused_df: pd.DataFrame,
    group: ExperimentGroup,
    job: ExperimentJob,
    n_clusters: int,
    random_state: int,
    baseline_score: float,
    param_workers: int,
    candidate_workers: int = 1,
) -> FfsIntuitiveNativeRecord:
    """Run exhaustive Intuitive-native FFS with paper initialization."""

    return run_ffs_intuitive_native_job(
        h_fused_df,
        group,
        job,
        n_clusters,
        random_state,
        baseline_score,
        param_workers,
        two_stage=False,
        init_strategy="paper",
        strict_init=True,
        candidate_workers=candidate_workers,
    )

def run_ffs_intuitive_view_weighted_paper_native_job(
    h_fused_df: pd.DataFrame,
    group: ExperimentGroup,
    job: ExperimentJob,
    n_clusters: int,
    random_state: int,
    baseline_score: float,
    param_workers: int,
    candidate_workers: int = 1,
) -> FfsIntuitiveViewWeightedNativeRecord:
    """Run view-weighted Intuitive-native FFS with paper initialization."""

    return run_ffs_intuitive_view_weighted_native_job(
        h_fused_df,
        group,
        job,
        n_clusters,
        random_state,
        baseline_score,
        param_workers,
        init_strategy="paper",
        strict_init=True,
        candidate_workers=candidate_workers,
    )

def run_ffs_intuitive_view_weighted_paper_exhaustive_native_job(
    h_fused_df: pd.DataFrame,
    group: ExperimentGroup,
    job: ExperimentJob,
    n_clusters: int,
    random_state: int,
    baseline_score: float,
    param_workers: int,
    candidate_workers: int = 1,
) -> FfsIntuitiveViewWeightedNativeRecord:
    """Run exhaustive view-weighted Intuitive-native FFS with paper init."""

    return run_ffs_intuitive_view_weighted_native_job(
        h_fused_df,
        group,
        job,
        n_clusters,
        random_state,
        baseline_score,
        param_workers,
        two_stage=False,
        init_strategy="paper",
        strict_init=True,
        candidate_workers=candidate_workers,
    )
def drain_queue(
    queue: Any,
    existing_df: pd.DataFrame,
    records: dict[int, ExperimentRecord],
    save_path: Path,
    label: str,
    columns: list[str],
    existing_trial_df: pd.DataFrame,
    trial_records: dict[
        tuple[int, int | None, str | None, int], NativeIntuitiveTrialRecord
    ],
    trial_save_path: Path | None,
    trial_columns: list[str] | None,
) -> None:
    """Drain finished records from workers and save them immediately."""

    while True:
        try:
            record = queue.get_nowait()
        except Empty:
            break

        records[record.job_index] = record
        save_results(existing_df, records, save_path, label, columns)
        save_trial_records_if_any(
            record,
            existing_trial_df,
            trial_records,
            trial_save_path,
            trial_columns,
            label,
        )
