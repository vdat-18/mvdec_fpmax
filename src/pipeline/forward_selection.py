"""Forward selection for FP-Max binary features with K-Prototypes clustering."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
from loguru import logger

from config import RANDOM_STATE
from pipeline.clustering import (
    DEFAULT_INIT_METHODS,
    compute_gower_distance,
    compute_silhouette,
    make_mixed_features,
)


@dataclass(frozen=True)
class ForwardSelectionResult:
    """Result of greedy forward selection over binary features."""

    matrix: np.ndarray
    selected_feature_names: list[str]
    labels: np.ndarray | None
    score: float
    best_observed_score: float
    init: str | None


def ensure_binary_features(binary_df: pd.DataFrame) -> pd.DataFrame:
    """Convert binary-like columns to integer 0/1 columns."""

    out = binary_df.copy()
    for column in out.columns:
        out[column] = out[column].astype(int)
    return out


def run_forward_selection(
    continuous_df: pd.DataFrame,
    binary_df: pd.DataFrame,
    n_clusters: int = 5,
    init_methods: tuple[str, ...] = DEFAULT_INIT_METHODS,
    random_state: int = RANDOM_STATE,
    baseline_score: float = 0.4069,
    min_improvement: float = 1e-6,
    verbose: bool = True,
) -> ForwardSelectionResult:
    """Greedily keep binary features that improve K-Prototypes silhouette."""

    continuous_df = continuous_df.copy()
    binary_df = ensure_binary_features(binary_df)

    for column in continuous_df.columns:
        continuous_df[column] = pd.to_numeric(continuous_df[column], errors="raise")

    remaining = list(range(binary_df.shape[1]))
    selected_names: list[str] = []
    best_score = float(baseline_score)
    best_labels: np.ndarray | None = None
    best_init: str | None = None
    best_observed_score = -np.inf

    if verbose:
        logger.info("Start forward selection | baseline = {:.4f}", best_score)

    while remaining:
        local_best: tuple[int, list[str], np.ndarray, str] | None = None
        local_best_score = -np.inf

        for idx in list(remaining):
            feature_name = binary_df.columns[idx]
            candidate_names = selected_names + [feature_name]
            candidate_binary_df = binary_df[candidate_names]
            combined_df, categorical_idx, _ = make_mixed_features(
                continuous_df=continuous_df,
                binary_df=candidate_binary_df,
            )
            distance_matrix = compute_gower_distance(combined_df)

            if not np.isfinite(distance_matrix).all():
                msg = f"Gower matrix contains non-finite values for {feature_name}."
                raise ValueError(msg)

            for init in init_methods:
                model = KPrototypes(
                    n_clusters=n_clusters,
                    init=init,
                    random_state=random_state,
                    verbose=0,
                )
                labels = model.fit_predict(
                    combined_df.to_numpy(),
                    categorical=categorical_idx,
                )
                score = compute_silhouette(distance_matrix, labels)

                if verbose:
                    logger.info(
                        "Feature {} | init={:<5} | silhouette={:.4f}",
                        feature_name,
                        init,
                        score,
                    )

                if score > best_observed_score:
                    best_observed_score = score

                if score > local_best_score:
                    local_best_score = score
                    local_best = (idx, candidate_names, labels, init)

        if local_best is None or local_best_score < best_score + min_improvement:
            if verbose:
                logger.info("Stop: no remaining feature improves enough.")
            break

        selected_idx, selected_names, best_labels, best_init = local_best
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
    )
