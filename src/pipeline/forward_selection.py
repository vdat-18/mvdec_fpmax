"""Forward selection for FP-Max binary features with K-Prototypes clustering."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

from config import RANDOM_STATE
from pipeline.clustering import (
    DEFAULT_INIT_METHODS,
    make_mixed_features,
    run_kprototypes,
)

DEFAULT_FFS_MIN_IMPROVEMENT = 0.0
# Gain a candidate must exceed in the FP-Max study; the legacy default accepts
# floating-point noise and lets greedy selection chase it.
STUDY_FFS_MIN_IMPROVEMENT = 0.005


@dataclass(frozen=True)
class ForwardSelectionResult:
    """Result of greedy forward selection over binary features."""

    matrix: np.ndarray
    selected_feature_names: list[str]
    labels: np.ndarray | None
    score: float
    best_observed_score: float
    init: str | None
    silhouette_sample_std: float
    silhouette_negative_fraction: float
    n_candidates_evaluated: int = 0


@dataclass(frozen=True)
class ForwardSelectionCandidateJob:
    """One candidate binary feature set to evaluate during FFS."""

    idx: int
    feature_name: str
    candidate_names: tuple[str, ...]


@dataclass(frozen=True)
class ForwardSelectionCandidateContext:
    """Shared inputs for K-Prototypes FFS candidate evaluation."""

    continuous_df: pd.DataFrame
    binary_df: pd.DataFrame
    n_clusters: int
    init_methods: tuple[str, ...]
    random_state: int


@dataclass(frozen=True)
class ForwardSelectionCandidateResult:
    """Evaluation result for one K-Prototypes FFS candidate."""

    idx: int
    candidate_names: tuple[str, ...]
    labels: np.ndarray | None
    init: str | None
    score: float
    best_observed_score: float
    logs: tuple[tuple[str, str, float], ...]
    silhouette_sample_std: float
    silhouette_negative_fraction: float


def ensure_binary_features(binary_df: pd.DataFrame) -> pd.DataFrame:
    """Convert binary-like columns to integer 0/1 columns."""

    out = binary_df.copy()
    for column in out.columns:
        out[column] = out[column].astype(int)
    return out


_FORWARD_SELECTION_CANDIDATE_CONTEXT: ForwardSelectionCandidateContext | None = None


def init_forward_selection_candidate_worker(
    context: ForwardSelectionCandidateContext,
) -> None:
    """Initialize per-process inputs for K-Prototypes FFS candidates."""

    global _FORWARD_SELECTION_CANDIDATE_CONTEXT
    _FORWARD_SELECTION_CANDIDATE_CONTEXT = context


def run_forward_selection_candidate(
    candidate: ForwardSelectionCandidateJob,
) -> ForwardSelectionCandidateResult:
    """Evaluate one candidate feature set for K-Prototypes FFS."""

    if _FORWARD_SELECTION_CANDIDATE_CONTEXT is None:
        msg = "Forward-selection candidate worker was not initialized."
        raise RuntimeError(msg)

    context = _FORWARD_SELECTION_CANDIDATE_CONTEXT
    candidate_binary_df = context.binary_df[list(candidate.candidate_names)]
    try:
        result = run_kprototypes(
            context.continuous_df,
            candidate_binary_df,
            context.n_clusters,
            init_methods=context.init_methods,
            random_state=context.random_state,
            verbose=False,
        )
    except RuntimeError:
        logger.warning(
            "Skip candidate {}: no valid K-Prototypes partition.",
            candidate.feature_name,
        )
        return ForwardSelectionCandidateResult(
            idx=candidate.idx,
            candidate_names=candidate.candidate_names,
            labels=None,
            init=None,
            score=-np.inf,
            best_observed_score=-np.inf,
            logs=(),
            silhouette_sample_std=np.nan,
            silhouette_negative_fraction=np.nan,
        )

    return ForwardSelectionCandidateResult(
        idx=candidate.idx,
        candidate_names=candidate.candidate_names,
        labels=result.labels,
        init=result.init,
        score=result.score,
        best_observed_score=result.score,
        logs=((candidate.feature_name, result.init, result.score),),
        silhouette_sample_std=result.silhouette_sample_std,
        silhouette_negative_fraction=result.silhouette_negative_fraction,
    )


def run_forward_selection(
    continuous_df: pd.DataFrame,
    binary_df: pd.DataFrame,
    n_clusters: int,
    baseline_score: float,
    init_methods: tuple[str, ...] = DEFAULT_INIT_METHODS,
    random_state: int = RANDOM_STATE,
    min_improvement: float = DEFAULT_FFS_MIN_IMPROVEMENT,
    verbose: bool = True,
    candidate_workers: int = 1,
) -> ForwardSelectionResult:
    """Greedily keep binary features that improve K-Prototypes silhouette."""

    if candidate_workers < 1:
        msg = "candidate_workers must be at least 1."
        raise ValueError(msg)

    continuous_df = continuous_df.copy()
    binary_df = ensure_binary_features(binary_df)

    for column in continuous_df.columns:
        continuous_df[column] = pd.to_numeric(continuous_df[column], errors="raise")

    remaining = list(range(binary_df.shape[1]))
    selected_names: list[str] = []
    best_score = float(baseline_score)
    best_labels: np.ndarray | None = None
    best_init: str | None = None
    best_sample_std = np.nan
    best_negative_fraction = np.nan
    best_observed_score = -np.inf
    n_candidates_evaluated = 0
    candidate_context = ForwardSelectionCandidateContext(
        continuous_df=continuous_df,
        binary_df=binary_df,
        n_clusters=n_clusters,
        init_methods=init_methods,
        random_state=random_state,
    )

    if verbose:
        logger.info("Start forward selection | baseline = {:.4f}", best_score)

    while remaining:
        local_best: tuple[int, list[str], np.ndarray, str, float, float] | None = None
        local_best_score = -np.inf
        candidate_jobs = [
            ForwardSelectionCandidateJob(
                idx=idx,
                feature_name=str(binary_df.columns[idx]),
                candidate_names=tuple(selected_names + [binary_df.columns[idx]]),
            )
            for idx in list(remaining)
        ]

        if candidate_workers > 1:
            max_workers = min(candidate_workers, len(candidate_jobs))
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=init_forward_selection_candidate_worker,
                initargs=(candidate_context,),
            ) as executor:
                candidate_results = list(
                    executor.map(run_forward_selection_candidate, candidate_jobs)
                )
        else:
            init_forward_selection_candidate_worker(candidate_context)
            candidate_results = [
                run_forward_selection_candidate(candidate)
                for candidate in candidate_jobs
            ]

        n_candidates_evaluated += len(candidate_results)
        for result in candidate_results:
            for feature_name, init, score in result.logs:
                if verbose:
                    logger.info(
                        "Feature {} | init={:<5} | silhouette={:.4f}",
                        feature_name,
                        init,
                        score,
                    )

            if result.best_observed_score > best_observed_score:
                best_observed_score = result.best_observed_score

            if (
                result.labels is not None
                and result.init is not None
                and result.score > local_best_score
            ):
                local_best_score = result.score
                local_best = (
                    result.idx,
                    list(result.candidate_names),
                    result.labels,
                    result.init,
                    result.silhouette_sample_std,
                    result.silhouette_negative_fraction,
                )

        if local_best is None or not improves_score(
            local_best_score,
            best_score,
            min_improvement,
        ):
            if verbose:
                logger.info("Stop: no remaining feature improves enough.")
            break

        (
            selected_idx,
            selected_names,
            best_labels,
            best_init,
            best_sample_std,
            best_negative_fraction,
        ) = local_best
        best_score = local_best_score
        remaining.remove(selected_idx)

        if verbose:
            logger.info(
                "Keep {} | silhouette = {:.4f}",
                binary_df.columns[selected_idx],
                best_score,
            )

    final_df, _, _ = make_mixed_features(
        continuous_df=continuous_df,
        binary_df=binary_df[selected_names],
    )

    return ForwardSelectionResult(
        matrix=final_df.to_numpy(),
        selected_feature_names=selected_names,
        labels=best_labels,
        score=best_score,
        best_observed_score=float(best_observed_score),
        init=best_init,
        silhouette_sample_std=float(best_sample_std),
        silhouette_negative_fraction=float(best_negative_fraction),
        n_candidates_evaluated=n_candidates_evaluated,
    )


def improves_score(
    candidate_score: float,
    current_score: float,
    min_improvement: float = DEFAULT_FFS_MIN_IMPROVEMENT,
) -> bool:
    """Return whether a candidate strictly exceeds the current score threshold."""

    if min_improvement < 0:
        msg = "min_improvement must be non-negative."
        raise ValueError(msg)
    return candidate_score > current_score + min_improvement
