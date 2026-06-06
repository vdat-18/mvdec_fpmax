"""K-Prototypes clustering and mixed-type validation metrics."""

from dataclasses import dataclass

import gower
import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
from loguru import logger
from sklearn.metrics import silhouette_score

from config import RANDOM_STATE

DEFAULT_INIT_METHODS = ("huang", "cao")


@dataclass(frozen=True)
class KPrototypesResult:
    """Best K-Prototypes result for one feature set."""

    matrix: np.ndarray
    binary_feature_names: list[str]
    labels: np.ndarray
    score: float
    init: str


def compute_gower_distance(df: pd.DataFrame) -> np.ndarray:
    """Compute a Gower distance matrix for mixed numeric and categorical data."""

    return gower.gower_matrix(df)


def compute_silhouette(distance_matrix: np.ndarray, labels: np.ndarray) -> float:
    """Compute silhouette score from a precomputed distance matrix."""

    return float(silhouette_score(distance_matrix, labels, metric="precomputed"))


def make_mixed_features(
    continuous_df: pd.DataFrame,
    binary_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[int], list[str]]:
    """Combine continuous and binary data for K-Prototypes."""

    n_numeric = continuous_df.shape[1]
    binary_feature_names = binary_df.columns.tolist()
    combined_df = pd.concat(
        [
            continuous_df.reset_index(drop=True),
            binary_df.astype(object).reset_index(drop=True),
        ],
        axis=1,
    )
    categorical_idx = list(range(n_numeric, n_numeric + len(binary_feature_names)))

    return combined_df, categorical_idx, binary_feature_names


def run_kprototypes(
    continuous_df: pd.DataFrame,
    binary_df: pd.DataFrame,
    n_clusters: int = 5,
    init_methods: tuple[str, ...] = DEFAULT_INIT_METHODS,
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
) -> KPrototypesResult:
    """Cluster continuous data with the provided binary features."""

    combined_df, categorical_idx, binary_feature_names = make_mixed_features(
        continuous_df=continuous_df,
        binary_df=binary_df,
    )
    distance_matrix = compute_gower_distance(combined_df)

    best_result: KPrototypesResult | None = None
    for init in init_methods:
        model = KPrototypes(
            n_clusters=n_clusters,
            init=init,
            random_state=random_state,
            verbose=0,
        )
        labels = model.fit_predict(combined_df.to_numpy(), categorical=categorical_idx)
        score = compute_silhouette(distance_matrix, labels)

        if verbose:
            logger.info("Init: {:<5} | silhouette: {:.4f}", init, score)

        if best_result is None or score > best_result.score:
            best_result = KPrototypesResult(
                matrix=combined_df.to_numpy(),
                binary_feature_names=binary_feature_names,
                labels=labels,
                score=score,
                init=init,
            )

    if best_result is None:
        msg = "No clustering result was produced."
        raise RuntimeError(msg)

    return best_result
