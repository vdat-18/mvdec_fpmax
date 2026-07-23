"""K-Prototypes clustering and mixed-type validation metrics."""

import hashlib
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import gower
import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
from loguru import logger
from sklearn.metrics import silhouette_samples, silhouette_score

from config import KPROTOTYPES_N_INIT, RANDOM_STATE
from intuitive_kprototypes import IntuitiveKPrototypes

DEFAULT_INIT_METHODS = ("huang", "cao")
MIXED_DISTANCE_CONTRACT = "gower_numeric_asymmetric_binary_v1"
SYMMETRIC_DISTANCE_CONTRACT = "gower_numeric_symmetric_binary_v1"
ClusteringBackend = Literal["kprototypes", "intuitive"]


@dataclass(frozen=True)
class KPrototypesResult:
    """Best K-Prototypes result for one feature set."""

    matrix: np.ndarray
    binary_feature_names: list[str]
    labels: np.ndarray
    score: float
    init: str
    fit_time_seconds: float
    cluster_sizes: list[int]
    labels_hash: str
    silhouette_sample_std: float
    silhouette_negative_fraction: float


@dataclass(frozen=True)
class IntuitiveClusteringResult:
    """Intuitive-K-prototypes result for one feature set."""

    matrix: np.ndarray
    binary_feature_names: list[str]
    labels: np.ndarray
    score: float
    init: str | None
    fit_time_seconds: float
    cluster_sizes: list[int]
    labels_hash: str
    converged: bool
    n_iter: int
    init_strategy: str
    lambda_init: float | None
    mu_param: float
    gamma: float
    beta: float
    empty_cluster_policy: str
    min_cluster_size: int
    initial_prototype_indices: list[int]
    final_cost: float
    silhouette_sample_std: float
    silhouette_negative_fraction: float
    view_weight_alpha: float | None = None
    view_weighted_score: float | None = None


MixedClusteringResult = KPrototypesResult | IntuitiveClusteringResult
DEFAULT_INTUITIVE_INIT_STRATEGY = "farthest_first"
DEFAULT_INTUITIVE_STRICT_INIT = False


def labels_hash(labels: np.ndarray) -> str:
    """Return a stable short hash for a label assignment."""

    labels_arr = np.asarray(labels, dtype=np.int64)
    digest = hashlib.sha256(labels_arr.tobytes()).hexdigest()
    return digest[:16]


def cluster_sizes(labels: np.ndarray, n_clusters: int) -> list[int]:
    """Return cluster sizes with empty clusters represented as zeros."""

    return np.bincount(np.asarray(labels, dtype=int), minlength=n_clusters).tolist()


def compute_gower_distance(df: pd.DataFrame) -> np.ndarray:
    """Compute range-normalized Gower distance for numeric features."""

    if df.shape[1] == 0:
        msg = "At least one feature is required to compute Gower distance."
        raise ValueError(msg)
    if len(df.select_dtypes(include=[np.number]).columns) != df.shape[1]:
        msg = "Numeric Gower distance requires numeric feature columns."
        raise ValueError(msg)
    return np.asarray(gower.gower_matrix(df.reset_index(drop=True)), dtype=float)


def compute_mixed_gower_distance(
    continuous_df: pd.DataFrame,
    binary_df: pd.DataFrame,
) -> np.ndarray:
    """Combine numeric Gower and asymmetric binary contributions by feature."""

    if len(continuous_df) != len(binary_df):
        msg = "Continuous and binary views must contain the same rows."
        raise ValueError(msg)

    numeric_count = continuous_df.shape[1]
    binary_count = binary_df.shape[1]
    if numeric_count == 0 and binary_count == 0:
        msg = "At least one view is required to compute mixed Gower distance."
        raise ValueError(msg)

    numeric_distance = (
        compute_gower_distance(continuous_df)
        if numeric_count
        else np.zeros((len(binary_df), len(binary_df)), dtype=float)
    )
    if binary_count == 0:
        return numeric_distance

    binary_distance, binary_union_count = _asymmetric_binary_distance(binary_df)
    numerator = numeric_distance * numeric_count + binary_distance * binary_union_count
    denominator = numeric_count + binary_union_count
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )


def compute_symmetric_mixed_gower_distance(
    continuous_df: pd.DataFrame,
    binary_df: pd.DataFrame,
) -> np.ndarray:
    """Compute the symmetric-Gower ablation for one mixed feature set."""

    combined_df, _, _ = make_mixed_features(continuous_df, binary_df)
    return np.asarray(gower.gower_matrix(combined_df), dtype=float)


def _asymmetric_binary_distance(
    binary_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Jaccard distance and pairwise nonzero-union feature counts."""

    binary = binary_df.to_numpy(dtype=float)
    if not np.isfinite(binary).all() or not np.isin(binary, (0.0, 1.0)).all():
        msg = "Asymmetric binary features must contain only finite 0/1 values."
        raise ValueError(msg)

    binary = binary.astype(np.int64, copy=False)
    row_sums = binary.sum(axis=1, dtype=np.int64)
    intersection = binary @ binary.T
    union_count = row_sums[:, None] + row_sums[None, :] - intersection
    mismatch_count = row_sums[:, None] + row_sums[None, :] - 2 * intersection
    distance = np.divide(
        mismatch_count,
        union_count,
        out=np.zeros_like(union_count, dtype=float),
        where=union_count > 0,
    )
    return distance, union_count


def compute_view_weighted_gower_distance(
    continuous_df: pd.DataFrame,
    binary_df: pd.DataFrame,
    view_weight_alpha: float,
) -> np.ndarray:
    """Compute a pairwise distance that balances latent and pattern views."""

    if not 0 <= view_weight_alpha <= 1:
        msg = "view_weight_alpha must satisfy 0 <= view_weight_alpha <= 1."
        raise ValueError(msg)

    has_numeric = continuous_df.shape[1] > 0
    has_binary = binary_df.shape[1] > 0
    if has_numeric and has_binary:
        numeric_distance = compute_gower_distance(continuous_df.reset_index(drop=True))
        binary_distance, binary_union_count = _asymmetric_binary_distance(
            binary_df.reset_index(drop=True),
        )
        binary_available = binary_union_count > 0
        numerator = view_weight_alpha * numeric_distance + (
            (1.0 - view_weight_alpha) * binary_distance * binary_available
        )
        denominator = view_weight_alpha + ((1.0 - view_weight_alpha) * binary_available)
        return np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0,
        )
    if has_numeric:
        return compute_gower_distance(continuous_df.reset_index(drop=True))
    if has_binary:
        binary_distance, _ = _asymmetric_binary_distance(
            binary_df.reset_index(drop=True),
        )
        return binary_distance

    msg = "At least one view is required to compute a distance matrix."
    raise ValueError(msg)


def compute_silhouette(distance_matrix: np.ndarray, labels: np.ndarray) -> float:
    """Compute silhouette score from a precomputed distance matrix."""

    return float(silhouette_score(distance_matrix, labels, metric="precomputed"))


def compute_silhouette_diagnostics(
    distance_matrix: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float, float]:
    """Return mean, sample dispersion, and negative-sample share for Silhouette."""

    samples = silhouette_samples(distance_matrix, labels, metric="precomputed")
    return (
        float(np.mean(samples)),
        float(np.std(samples)),
        float(np.mean(samples < 0.0)),
    )


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
    n_clusters: int,
    init_methods: tuple[str, ...] = DEFAULT_INIT_METHODS,
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
) -> KPrototypesResult:
    """Cluster continuous data with the provided binary features."""

    combined_df, categorical_idx, binary_feature_names = make_mixed_features(
        continuous_df=continuous_df,
        binary_df=binary_df,
    )
    distance_matrix = compute_mixed_gower_distance(continuous_df, binary_df)

    best_result: KPrototypesResult | None = None
    for init in init_methods:
        fit_start = perf_counter()
        model = KPrototypes(
            n_clusters=n_clusters,
            init=init,
            n_init=KPROTOTYPES_N_INIT,
            random_state=random_state,
            verbose=0,
        )
        labels = model.fit_predict(combined_df.to_numpy(), categorical=categorical_idx)
        fit_time = perf_counter() - fit_start
        score, sample_std, negative_fraction = compute_silhouette_diagnostics(
            distance_matrix,
            labels,
        )

        if verbose:
            logger.info(
                "backend=kprototypes init={:<5} silhouette={:.4f} "
                "time={:.3f}s sizes={}",
                init,
                score,
                fit_time,
                cluster_sizes(labels, n_clusters),
            )

        if best_result is None or score > best_result.score:
            best_result = KPrototypesResult(
                matrix=combined_df.to_numpy(),
                binary_feature_names=binary_feature_names,
                labels=labels,
                score=score,
                init=init,
                fit_time_seconds=fit_time,
                cluster_sizes=cluster_sizes(labels, n_clusters),
                labels_hash=labels_hash(labels),
                silhouette_sample_std=sample_std,
                silhouette_negative_fraction=negative_fraction,
            )

    if best_result is None:
        msg = "No clustering result was produced."
        raise RuntimeError(msg)

    return best_result


def run_intuitive_kprototypes(
    continuous_df: pd.DataFrame,
    binary_df: pd.DataFrame,
    n_clusters: int,
    random_state: int = RANDOM_STATE,
    lambda_init: float | None = None,
    mu_param: float = 0.5,
    gamma: float = 0.5,
    beta: float = 2.0,
    max_iter: int = 100,
    init_strategy: str = DEFAULT_INTUITIVE_INIT_STRATEGY,
    strict_init: bool = DEFAULT_INTUITIVE_STRICT_INIT,
    empty_cluster_policy: str = "raise",
    min_cluster_size: int = 1,
    distance_matrix: np.ndarray | None = None,
    view_weight_alpha: float | None = None,
    view_weighted_distance_matrix: np.ndarray | None = None,
    verbose: bool = True,
) -> IntuitiveClusteringResult:
    """Cluster continuous and binary features with Intuitive-K-prototypes."""

    combined_df, _, binary_feature_names = make_mixed_features(
        continuous_df=continuous_df,
        binary_df=binary_df,
    )
    if distance_matrix is None:
        distance_matrix = compute_mixed_gower_distance(continuous_df, binary_df)

    X_num = continuous_df.to_numpy(dtype=float)
    X_cat = binary_df.astype(str).to_numpy(dtype=object)
    model_lambda_init = 0.8 if lambda_init is None else lambda_init
    model = IntuitiveKPrototypes(
        n_clusters=n_clusters,
        lambda_init=model_lambda_init,
        mu_param=mu_param,
        gamma=gamma,
        beta=beta,
        max_iter=max_iter,
        random_state=random_state,
        init_strategy=init_strategy,
        strict_init=strict_init,
        empty_cluster_policy=empty_cluster_policy,
        min_cluster_size=min_cluster_size,
        view_weight_alpha=view_weight_alpha,
    )
    result = model.fit(X_num=X_num, X_cat=X_cat).result_
    labels = result.labels
    score, sample_std, negative_fraction = compute_silhouette_diagnostics(
        distance_matrix,
        labels,
    )
    view_weighted_score = None
    if view_weight_alpha is not None:
        if view_weighted_distance_matrix is None:
            view_weighted_distance_matrix = compute_view_weighted_gower_distance(
                continuous_df,
                binary_df,
                view_weight_alpha,
            )
        view_weighted_score = compute_silhouette(
            view_weighted_distance_matrix,
            labels,
        )
    sizes = cluster_sizes(labels, n_clusters)
    final_cost = float(result.distances[np.arange(len(labels)), labels].sum())

    if verbose:
        logger.info(
            "backend=intuitive silhouette={:.4f} time={:.3f}s "
            "converged={} iter={} sizes={}",
            score,
            result.fit_time_seconds,
            result.converged,
            result.n_iter,
            sizes,
        )

    return IntuitiveClusteringResult(
        matrix=combined_df.to_numpy(),
        binary_feature_names=binary_feature_names,
        labels=labels,
        score=score,
        init=None,
        fit_time_seconds=float(result.fit_time_seconds),
        cluster_sizes=sizes,
        labels_hash=labels_hash(labels),
        converged=bool(result.converged),
        n_iter=int(result.n_iter),
        init_strategy=init_strategy,
        lambda_init=lambda_init if init_strategy == "paper" else None,
        mu_param=mu_param,
        gamma=gamma,
        beta=beta,
        empty_cluster_policy=empty_cluster_policy,
        min_cluster_size=min_cluster_size,
        initial_prototype_indices=result.initial_prototype_indices.astype(int).tolist(),
        final_cost=final_cost,
        silhouette_sample_std=sample_std,
        silhouette_negative_fraction=negative_fraction,
        view_weight_alpha=view_weight_alpha,
        view_weighted_score=view_weighted_score,
    )
