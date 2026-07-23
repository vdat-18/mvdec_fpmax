"""Intuitive-K-prototypes clustering backend.

This module implements the equations from:

Wang, H., & Mi, J. (2025). Intuitive-K-prototypes: A mixed data clustering
algorithm with intuitionistic distribution centroid. Pattern Recognition,
158, 111062.

The implementation is intentionally explicit. It favors equation-level
traceability over speed because this implementation follows the paper formulas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Literal

import numpy as np

EmptyClusterPolicy = Literal["raise", "farthest"]
InitialPrototypeStrategy = Literal["paper", "farthest_first"]
NonMembershipStrategy = Literal["paper", "table"]


@dataclass(frozen=True)
class IntuitionisticCentroids:
    """Categorical centroids from Eq. (2)-(4)."""

    values: tuple[np.ndarray, ...]
    mu: tuple[np.ndarray, ...]
    nu: tuple[np.ndarray, ...]
    value_to_index: tuple[dict[Any, int], ...]


@dataclass(frozen=True)
class AttributeWeights:
    """Attribute-weight state for Eq. (15)-(23)."""

    numeric: np.ndarray
    categorical: np.ndarray
    numeric_scaled: np.ndarray
    categorical_scaled: np.ndarray
    numeric_phi: np.ndarray
    categorical_phi: np.ndarray
    numeric_complexity: np.ndarray
    numeric_similarity: np.ndarray
    categorical_complexity: np.ndarray
    categorical_similarity: np.ndarray


@dataclass(frozen=True)
class NumericComplexityDetails:
    """Trace values for numerical Eq. (5)-(6)."""

    coefficients: np.ndarray
    normalized_coefficients: np.ndarray
    complexity: np.ndarray


@dataclass(frozen=True)
class NumericSimilarityDetails:
    """Trace values for numerical Eq. (7)-(14)."""

    means: np.ndarray
    r_global: np.ndarray
    r_local: np.ndarray
    d_global: np.ndarray
    d_local: np.ndarray
    pair_indices: tuple[tuple[int, int], ...]
    pair_weights: np.ndarray
    pair_similarity: np.ndarray
    similarity: np.ndarray


@dataclass(frozen=True)
class IterationRecord:
    """One iteration of the full Algorithm 2 loop."""

    iteration: int
    changed: int
    cost: float
    elapsed_seconds: float


@dataclass(frozen=True)
class IntuitiveKPrototypesResult:
    """Final result returned by ``IntuitiveKPrototypes.fit``."""

    labels: np.ndarray
    numeric_centroids: np.ndarray
    categorical_centroids: IntuitionisticCentroids | None
    weights: AttributeWeights
    distances: np.ndarray
    history: tuple[IterationRecord, ...]
    n_iter: int
    converged: bool
    fit_time_seconds: float
    initial_prototype_indices: np.ndarray


def _as_2d_float(X: Any | None, name: str) -> np.ndarray:
    if X is None:
        return np.empty((0, 0), dtype=float)
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        msg = f"{name} must be a 2D array."
        raise ValueError(msg)
    return arr


def _as_2d_object(X: Any | None, name: str) -> np.ndarray:
    if X is None:
        return np.empty((0, 0), dtype=object)
    arr = np.asarray(X, dtype=object)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        msg = f"{name} must be a 2D array."
        raise ValueError(msg)
    return arr


def _validate_mixed_inputs(
    X_num: np.ndarray,
    X_cat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if X_num.size == 0 and X_cat.size == 0:
        msg = "At least one of X_num or X_cat must be non-empty."
        raise ValueError(msg)

    n_num = X_num.shape[0] if X_num.size else None
    n_cat = X_cat.shape[0] if X_cat.size else None
    if n_num is not None and n_cat is not None and n_num != n_cat:
        msg = "X_num and X_cat must have the same number of rows."
        raise ValueError(msg)

    if X_num.size == 0:
        X_num = np.empty((n_cat, 0), dtype=float)
    if X_cat.size == 0:
        X_cat = np.empty((n_num, 0), dtype=object)
    return X_num, X_cat


def _validate_labels(labels: np.ndarray, n: int, n_clusters: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    if labels.shape != (n,):
        msg = "labels must have shape (n_samples,)."
        raise ValueError(msg)
    if labels.min(initial=0) < 0 or labels.max(initial=0) >= n_clusters:
        msg = "labels must be in 0..n_clusters-1."
        raise ValueError(msg)
    return labels


def _cluster_sizes(labels: np.ndarray, n_clusters: int) -> np.ndarray:
    return np.bincount(labels, minlength=n_clusters).astype(int)


def _require_non_empty_clusters(labels: np.ndarray, n_clusters: int) -> np.ndarray:
    sizes = _cluster_sizes(labels, n_clusters)
    empty = np.flatnonzero(sizes == 0)
    if len(empty):
        msg = f"Empty clusters are not defined by the paper: {empty.tolist()}."
        raise ValueError(msg)
    return sizes


def _repair_small_clusters(
    labels: np.ndarray,
    distances: np.ndarray,
    n_clusters: int,
    min_cluster_size: int = 1,
) -> np.ndarray:
    """Move high-cost donor points into undersized clusters.

    The paper does not define empty-cluster recovery. This conservative repair
    is opt-in and keeps the default estimator behavior strict.
    """

    if min_cluster_size < 1:
        msg = "min_cluster_size must be at least 1."
        raise ValueError(msg)
    repaired = np.asarray(labels, dtype=int).copy()
    if n_clusters * min_cluster_size > len(repaired):
        msg = (
            "Cannot enforce min_cluster_size because "
            "n_clusters * min_cluster_size exceeds n_samples."
        )
        raise ValueError(msg)

    sizes = _cluster_sizes(repaired, n_clusters)
    small = np.flatnonzero(sizes < min_cluster_size)
    if len(small) == 0:
        return repaired

    assigned_cost = distances[np.arange(len(repaired)), repaired].astype(float)
    for cluster in small:
        while sizes[cluster] < min_cluster_size:
            donor_mask = sizes[repaired] > min_cluster_size
            if not np.any(donor_mask):
                msg = "Cannot repair small clusters without a large enough donor."
                raise ValueError(msg)
            donor_candidates = np.flatnonzero(donor_mask)
            donor = donor_candidates[int(np.argmax(assigned_cost[donor_candidates]))]
            sizes[repaired[donor]] -= 1
            repaired[donor] = int(cluster)
            sizes[cluster] += 1
            assigned_cost[donor] = -np.inf

    _require_non_empty_clusters(repaired, n_clusters)
    return repaired


def mixed_initial_distance_matrix(
    X_num: Any | None,
    X_cat: Any | None,
    prototypes_num: Any | None,
    prototypes_cat: Any | None,
) -> np.ndarray:
    """Return Eq. (1) distances from rows to prototype rows."""

    X_num_arr = _as_2d_float(X_num, "X_num")
    X_cat_arr = _as_2d_object(X_cat, "X_cat")
    X_num_arr, X_cat_arr = _validate_mixed_inputs(X_num_arr, X_cat_arr)

    proto_num = _as_2d_float(prototypes_num, "prototypes_num")
    proto_cat = _as_2d_object(prototypes_cat, "prototypes_cat")
    proto_num, proto_cat = _validate_mixed_inputs(proto_num, proto_cat)

    if X_num_arr.shape[1] != proto_num.shape[1]:
        msg = "X_num and prototypes_num must have the same number of columns."
        raise ValueError(msg)
    if X_cat_arr.shape[1] != proto_cat.shape[1]:
        msg = "X_cat and prototypes_cat must have the same number of columns."
        raise ValueError(msg)

    n_samples = X_num_arr.shape[0]
    n_prototypes = proto_num.shape[0]
    distances = np.zeros((n_samples, n_prototypes), dtype=float)

    if X_num_arr.shape[1]:
        distances += np.abs(
            X_num_arr[:, None, :] - proto_num[None, :, :]
        ).sum(axis=2)
    if X_cat_arr.shape[1]:
        distances += (X_cat_arr[:, None, :] != proto_cat[None, :, :]).sum(axis=2)
    return distances


def select_initial_prototypes(
    X_num: Any | None,
    X_cat: Any | None,
    n_clusters: int,
    lambda_init: float = 0.8,
    random_state: int | None = None,
    strict: bool = True,
) -> np.ndarray:
    """Select initial prototype indexes using Algorithm 1."""

    if n_clusters < 1:
        msg = "n_clusters must be at least 1."
        raise ValueError(msg)
    if not 0 < lambda_init <= 1:
        msg = "lambda_init must satisfy 0 < lambda_init <= 1."
        raise ValueError(msg)

    X_num_arr = _as_2d_float(X_num, "X_num")
    X_cat_arr = _as_2d_object(X_cat, "X_cat")
    X_num_arr, X_cat_arr = _validate_mixed_inputs(X_num_arr, X_cat_arr)
    n_samples = X_num_arr.shape[0]
    if n_clusters > n_samples:
        msg = "n_clusters cannot exceed n_samples."
        raise ValueError(msg)
    if n_clusters == 1:
        return np.array([0], dtype=int)

    rng = np.random.default_rng(random_state)
    idx0 = int(rng.integers(0, n_samples))
    dist_to_x0 = mixed_initial_distance_matrix(
        X_num_arr,
        X_cat_arr,
        X_num_arr[[idx0]],
        X_cat_arr[[idx0]],
    )[:, 0]
    idx1 = int(np.argmax(dist_to_x0))

    dist_to_x1 = mixed_initial_distance_matrix(
        X_num_arr,
        X_cat_arr,
        X_num_arr[[idx1]],
        X_cat_arr[[idx1]],
    )[:, 0]
    idx2 = int(np.argmax(dist_to_x1))
    d_approx = float(dist_to_x1[idx2])
    d_lambda = lambda_init * d_approx / n_clusters

    selected = [idx1]
    if idx2 != idx1:
        selected.append(idx2)

    while len(selected) < n_clusters:
        chosen: int | None = None
        for idx in range(n_samples):
            if idx in selected:
                continue
            distances = mixed_initial_distance_matrix(
                X_num_arr[[idx]],
                X_cat_arr[[idx]],
                X_num_arr[selected],
                X_cat_arr[selected],
            )[0]
            if np.all(distances >= d_lambda):
                chosen = idx
                break

        if chosen is None:
            if strict:
                msg = (
                    "Algorithm 1 could not find enough prototypes with "
                    f"lambda_init={lambda_init} and d_lambda={d_lambda:.6g}."
                )
                raise ValueError(msg)
            remaining = [idx for idx in range(n_samples) if idx not in selected]
            dist_to_selected = mixed_initial_distance_matrix(
                X_num_arr[remaining],
                X_cat_arr[remaining],
                X_num_arr[selected],
                X_cat_arr[selected],
            )
            chosen = remaining[int(np.argmax(dist_to_selected.min(axis=1)))]

        selected.append(chosen)

    return np.asarray(selected, dtype=int)


def select_farthest_first_prototypes(
    X_num: Any | None,
    X_cat: Any | None,
    n_clusters: int,
    random_state: int | None = None,
) -> np.ndarray:
    """Select initial prototype indexes with greedy farthest-first traversal."""

    if n_clusters < 1:
        msg = "n_clusters must be at least 1."
        raise ValueError(msg)

    X_num_arr = _as_2d_float(X_num, "X_num")
    X_cat_arr = _as_2d_object(X_cat, "X_cat")
    X_num_arr, X_cat_arr = _validate_mixed_inputs(X_num_arr, X_cat_arr)
    n_samples = X_num_arr.shape[0]
    if n_clusters > n_samples:
        msg = "n_clusters cannot exceed n_samples."
        raise ValueError(msg)
    if n_clusters == 1:
        return np.array([0], dtype=int)

    rng = np.random.default_rng(random_state)
    idx0 = int(rng.integers(0, n_samples))
    dist_to_x0 = mixed_initial_distance_matrix(
        X_num_arr,
        X_cat_arr,
        X_num_arr[[idx0]],
        X_cat_arr[[idx0]],
    )[:, 0]
    first = int(np.argmax(dist_to_x0))
    selected = [first]

    min_distances = mixed_initial_distance_matrix(
        X_num_arr,
        X_cat_arr,
        X_num_arr[[first]],
        X_cat_arr[[first]],
    )[:, 0]
    min_distances[first] = -np.inf

    while len(selected) < n_clusters:
        next_idx = int(np.argmax(min_distances))
        selected.append(next_idx)
        new_distances = mixed_initial_distance_matrix(
            X_num_arr,
            X_cat_arr,
            X_num_arr[[next_idx]],
            X_cat_arr[[next_idx]],
        )[:, 0]
        min_distances = np.minimum(min_distances, new_distances)
        min_distances[selected] = -np.inf

    return np.asarray(selected, dtype=int)


def compute_intuitionistic_centroids(
    X_cat: Any,
    labels: Any,
    n_clusters: int,
    non_membership: NonMembershipStrategy = "paper",
) -> IntuitionisticCentroids:
    """Compute Definition 1, Eq. (2)-(4)."""

    if non_membership not in {"paper", "table"}:
        msg = "non_membership must be either 'paper' or 'table'."
        raise ValueError(msg)

    X_cat_arr = _as_2d_object(X_cat, "X_cat")
    n_samples, n_cat = X_cat_arr.shape
    labels_arr = _validate_labels(np.asarray(labels), n_samples, n_clusters)
    sizes = _require_non_empty_clusters(labels_arr, n_clusters)

    values: list[np.ndarray] = []
    value_to_index: list[dict[Any, int]] = []
    mu_all: list[np.ndarray] = []
    nu_all: list[np.ndarray] = []

    for j in range(n_cat):
        vals = np.unique(X_cat_arr[:, j])
        values.append(vals)
        value_to_index.append({value: idx for idx, value in enumerate(vals)})
        t = len(vals)
        counts = np.array([(X_cat_arr[:, j] == value).sum() for value in vals])
        counts_by_cluster = np.zeros((n_clusters, t), dtype=float)
        for cluster in range(n_clusters):
            mask = labels_arr == cluster
            for value_idx, value in enumerate(vals):
                counts_by_cluster[cluster, value_idx] = np.sum(
                    mask & (X_cat_arr[:, j] == value)
                )

        mu = np.zeros((n_clusters, t), dtype=float)
        nu = np.zeros((n_clusters, t), dtype=float)
        for cluster in range(n_clusters):
            for value_idx in range(t):
                global_count = counts[value_idx]
                inter = counts_by_cluster[cluster, value_idx]
                mu[cluster, value_idx] = (inter**2) / (
                    global_count * sizes[cluster]
                )

                total = 0.0
                for other_cluster in range(n_clusters):
                    if other_cluster == cluster:
                        continue
                    other_count = counts_by_cluster[other_cluster, value_idx]
                    if non_membership == "paper":
                        total += (other_count / global_count) ** 2
                    elif inter == 0:
                        total = 1.0
                        break
                    else:
                        total += (other_count / sizes[other_cluster]) ** 2
                nu[cluster, value_idx] = total

        if np.any(mu < -1e-12) or np.any(nu < -1e-12):
            msg = "IDC contains negative membership values."
            raise RuntimeError(msg)
        if np.any(mu + nu > 1 + 1e-10):
            msg = "IDC violates mu + nu <= 1."
            raise RuntimeError(msg)
        mu_all.append(mu)
        nu_all.append(nu)

    return IntuitionisticCentroids(
        values=tuple(values),
        mu=tuple(mu_all),
        nu=tuple(nu_all),
        value_to_index=tuple(value_to_index),
    )


def numeric_centroids(
    X_num: Any,
    labels: Any,
    n_clusters: int,
) -> np.ndarray:
    """Return numerical cluster means."""

    X_num_arr = _as_2d_float(X_num, "X_num")
    labels_arr = _validate_labels(np.asarray(labels), X_num_arr.shape[0], n_clusters)
    _require_non_empty_clusters(labels_arr, n_clusters)
    centroids = np.zeros((n_clusters, X_num_arr.shape[1]), dtype=float)
    for cluster in range(n_clusters):
        centroids[cluster] = X_num_arr[labels_arr == cluster].mean(axis=0)
    return centroids


def numeric_complexity(
    X_num: Any,
    labels: Any,
    n_clusters: int,
) -> np.ndarray:
    """Compute numerical intra-cluster complexity, Eq. (5)-(6)."""

    return numeric_complexity_details(X_num, labels, n_clusters).complexity


def numeric_complexity_details(
    X_num: Any,
    labels: Any,
    n_clusters: int,
) -> NumericComplexityDetails:
    """Return trace values for numerical intra-cluster complexity."""

    X_num_arr = _as_2d_float(X_num, "X_num")
    n_samples, n_num = X_num_arr.shape
    if n_num == 0:
        empty = np.empty((n_clusters, 0), dtype=float)
        return NumericComplexityDetails(
            coefficients=empty,
            normalized_coefficients=empty,
            complexity=np.empty(0, dtype=float),
        )
    labels_arr = _validate_labels(np.asarray(labels), n_samples, n_clusters)
    sizes = _require_non_empty_clusters(labels_arr, n_clusters)

    cv = np.zeros((n_clusters, n_num), dtype=float)
    for cluster in range(n_clusters):
        rows = X_num_arr[labels_arr == cluster]
        means = rows.mean(axis=0)
        stds = rows.std(axis=0)
        cv[cluster] = np.divide(
            stds,
            np.abs(means),
            out=np.zeros_like(stds, dtype=float),
            where=np.abs(means) > 1e-12,
        )

    max_cv = cv.max(axis=0)
    cv_norm = np.divide(
        cv,
        max_cv,
        out=np.zeros_like(cv, dtype=float),
        where=np.abs(max_cv) > 1e-12,
    )
    alpha = sizes / n_samples
    return NumericComplexityDetails(
        coefficients=cv,
        normalized_coefficients=cv_norm,
        complexity=alpha @ cv_norm,
    )


def numeric_similarity(
    X_num: Any,
    labels: Any,
    n_clusters: int,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute numerical inter-cluster similarity, Eq. (7)-(14)."""

    return numeric_similarity_details(X_num, labels, n_clusters, eps).similarity


def numeric_similarity_details(
    X_num: Any,
    labels: Any,
    n_clusters: int,
    eps: float = 1e-12,
) -> NumericSimilarityDetails:
    """Return trace values for numerical inter-cluster similarity."""

    X_num_arr = _as_2d_float(X_num, "X_num")
    n_samples, n_num = X_num_arr.shape
    if n_num == 0:
        empty_cluster_attr = np.empty((n_clusters, 0), dtype=float)
        empty_pair_attr = np.empty((0, 0), dtype=float)
        return NumericSimilarityDetails(
            means=empty_cluster_attr,
            r_global=empty_cluster_attr,
            r_local=empty_cluster_attr,
            d_global=np.empty((n_clusters, n_clusters, 0), dtype=float),
            d_local=np.empty((n_clusters, n_clusters, 0), dtype=float),
            pair_indices=(),
            pair_weights=np.empty(0, dtype=float),
            pair_similarity=empty_pair_attr,
            similarity=np.empty(0, dtype=float),
        )
    labels_arr = _validate_labels(np.asarray(labels), n_samples, n_clusters)
    sizes = _require_non_empty_clusters(labels_arr, n_clusters)

    means = numeric_centroids(X_num_arr, labels_arr, n_clusters)
    r_global = np.zeros((n_clusters, n_num), dtype=float)
    r_local = np.zeros((n_clusters, n_num), dtype=float)
    good_masks: list[list[np.ndarray]] = [
        [np.empty(0, dtype=bool) for _ in range(n_num)] for _ in range(n_clusters)
    ]

    for cluster in range(n_clusters):
        rows = X_num_arr[labels_arr == cluster]
        for j in range(n_num):
            dev = np.abs(rows[:, j] - means[cluster, j])
            r_global[cluster, j] = dev.mean()
            good = dev <= r_global[cluster, j] + eps
            good_masks[cluster][j] = good
            r_local[cluster, j] = dev[good].mean() if np.any(good) else 0.0

    d_global = np.zeros((n_clusters, n_clusters, n_num), dtype=float)
    d_local = np.zeros((n_clusters, n_clusters, n_num), dtype=float)
    for src in range(n_clusters):
        rows = X_num_arr[labels_arr == src]
        for dst in range(n_clusters):
            if src == dst:
                continue
            for j in range(n_num):
                d_global[src, dst, j] = np.abs(rows[:, j] - means[dst, j]).mean()
                good_rows = rows[good_masks[src][j]]
                if len(good_rows):
                    d_local[src, dst, j] = np.abs(
                        good_rows[:, j] - means[dst, j]
                    ).mean()

    pair_weights = _pair_weights(sizes)
    pair_indices = tuple((c1, c2) for c1, c2, _ in pair_weights)
    pair_weight_values = np.array([theta for _, _, theta in pair_weights])
    pair_similarity = np.zeros((len(pair_weights), n_num), dtype=float)
    similarity = np.zeros(n_num, dtype=float)
    for j in range(n_num):
        total = 0.0
        for pair_idx, (c1, c2, theta) in enumerate(pair_weights):
            d_star = d_local[c1, c2, j] * d_local[c2, c1, j]
            if abs(d_star) <= eps:
                s_pair = 1.0
            else:
                denom_21 = max(eps, d_global[c2, c1, j] * d_local[c2, c1, j])
                denom_12 = max(eps, d_global[c1, c2, j] * d_local[c1, c2, j])
                h = (
                    r_global[c1, j] * r_local[c1, j] / denom_21
                    + r_global[c2, j] * r_local[c2, j] / denom_12
                )
                s_pair = 1.0 - np.exp(-h)
            pair_similarity[pair_idx, j] = s_pair
            total += theta * s_pair
        similarity[j] = total
    return NumericSimilarityDetails(
        means=means,
        r_global=r_global,
        r_local=r_local,
        d_global=d_global,
        d_local=d_local,
        pair_indices=pair_indices,
        pair_weights=pair_weight_values,
        pair_similarity=np.clip(pair_similarity, 0.0, 1.0),
        similarity=np.clip(similarity, 0.0, 1.0),
    )


def categorical_complexity(
    centroids: IntuitionisticCentroids,
    labels: Any,
    n_clusters: int,
) -> np.ndarray:
    """Compute categorical intra-cluster complexity, Eq. (17)-(18)."""

    labels_arr = np.asarray(labels, dtype=int)
    sizes = _require_non_empty_clusters(labels_arr, n_clusters)
    alpha = sizes / len(labels_arr)
    complexity = np.zeros(len(centroids.values), dtype=float)

    for j in range(len(centroids.values)):
        entropies = np.zeros(n_clusters, dtype=float)
        for cluster in range(n_clusters):
            mins = np.minimum(centroids.mu[j][cluster], centroids.nu[j][cluster])
            maxs = np.maximum(centroids.mu[j][cluster], centroids.nu[j][cluster])
            denom = maxs.sum()
            entropies[cluster] = 0.0 if denom <= 1e-12 else mins.sum() / denom
        complexity[j] = alpha @ entropies
    return np.clip(complexity, 0.0, 1.0)


def categorical_similarity(
    centroids: IntuitionisticCentroids,
    labels: Any,
    n_clusters: int,
) -> np.ndarray:
    """Compute categorical inter-cluster similarity, Eq. (19)-(20)."""

    labels_arr = np.asarray(labels, dtype=int)
    sizes = _require_non_empty_clusters(labels_arr, n_clusters)
    pair_weights = _pair_weights(sizes)
    similarity = np.zeros(len(centroids.values), dtype=float)

    for j, vals in enumerate(centroids.values):
        t_values = len(vals)
        total = 0.0
        for c1, c2, theta in pair_weights:
            diff_1 = centroids.mu[j][c1] - centroids.nu[j][c1]
            diff_2 = centroids.mu[j][c2] - centroids.nu[j][c2]
            s_pair = 1.0 - np.abs(diff_1 - diff_2).sum() / (2.0 * t_values)
            total += theta * s_pair
        similarity[j] = total
    return np.clip(similarity, 0.0, 1.0)


def _pair_weights(sizes: np.ndarray) -> list[tuple[int, int, float]]:
    pairs = [
        (i, j, float(sizes[i] + sizes[j]))
        for i in range(len(sizes) - 1)
        for j in range(i + 1, len(sizes))
    ]
    denom = sum(weight for _, _, weight in pairs)
    if denom <= 0:
        return [(i, j, 0.0) for i, j, _ in pairs]
    return [(i, j, weight / denom) for i, j, weight in pairs]


def _inverse_phi_weights(phi: np.ndarray, beta: float) -> np.ndarray:
    if beta <= 1:
        msg = "beta must be greater than 1."
        raise ValueError(msg)
    phi = np.asarray(phi, dtype=float)
    if len(phi) == 0:
        return np.empty(0, dtype=float)
    if np.any(~np.isfinite(phi)):
        msg = "phi contains non-finite values."
        raise ValueError(msg)
    if np.any(phi < -1e-12):
        msg = "phi must be non-negative."
        raise ValueError(msg)

    phi = np.maximum(phi, 0.0)
    positive = phi > 0.0
    weights = np.zeros_like(phi, dtype=float)
    if not np.any(positive):
        return weights

    exponent = -1.0 / (beta - 1.0)
    positive_phi = phi[positive]
    log_scores = exponent * np.log(positive_phi)
    scores = np.exp(log_scores - log_scores.max())
    weights[positive] = scores / scores.sum()
    return weights


def numeric_weights(
    X_num: Any,
    labels: Any,
    n_clusters: int,
    gamma: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return unscaled weights, phi, complexity, and similarity for numerics."""

    complexity = numeric_complexity(X_num, labels, n_clusters)
    similarity = numeric_similarity(X_num, labels, n_clusters)
    phi = gamma * complexity + (1.0 - gamma) * similarity
    return _inverse_phi_weights(phi, beta), phi, complexity, similarity


def categorical_weights(
    centroids: IntuitionisticCentroids,
    labels: Any,
    n_clusters: int,
    gamma: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return unscaled weights, phi, complexity, and similarity for categories."""

    complexity = categorical_complexity(centroids, labels, n_clusters)
    similarity = categorical_similarity(centroids, labels, n_clusters)
    phi = gamma * complexity + (1.0 - gamma) * similarity
    return _inverse_phi_weights(phi, beta), phi, complexity, similarity


def compute_attribute_weights(
    X_num: np.ndarray,
    centroids_cat: IntuitionisticCentroids | None,
    labels: np.ndarray,
    n_clusters: int,
    gamma: float,
    beta: float,
) -> AttributeWeights:
    n_num = X_num.shape[1]
    n_cat = len(centroids_cat.values) if centroids_cat is not None else 0

    if n_num:
        w_num, phi_num, comp_num, sim_num = numeric_weights(
            X_num, labels, n_clusters, gamma, beta
        )
    else:
        w_num = phi_num = comp_num = sim_num = np.empty(0, dtype=float)

    if n_cat and centroids_cat is not None:
        w_cat, phi_cat, comp_cat, sim_cat = categorical_weights(
            centroids_cat, labels, n_clusters, gamma, beta
        )
    else:
        w_cat = phi_cat = comp_cat = sim_cat = np.empty(0, dtype=float)

    return AttributeWeights(
        numeric=w_num,
        categorical=w_cat,
        numeric_scaled=w_num * n_num,
        categorical_scaled=w_cat * n_cat,
        numeric_phi=phi_num,
        categorical_phi=phi_cat,
        numeric_complexity=comp_num,
        numeric_similarity=sim_num,
        categorical_complexity=comp_cat,
        categorical_similarity=sim_cat,
    )


def categorical_attribute_distances(
    X_cat: Any,
    centroids: IntuitionisticCentroids,
    mu_param: float = 0.5,
) -> np.ndarray:
    """Return unweighted per-attribute Eq. (26) distances.

    Output shape is ``(n_samples, n_clusters, n_categorical_attributes)``.
    """

    if not 0 <= mu_param <= 1:
        msg = "mu_param must satisfy 0 <= mu_param <= 1."
        raise ValueError(msg)

    X_cat_arr = _as_2d_object(X_cat, "X_cat")
    n_samples, n_cat = X_cat_arr.shape
    if n_cat != len(centroids.values):
        msg = "X_cat column count does not match centroids."
        raise ValueError(msg)
    n_clusters = centroids.mu[0].shape[0] if n_cat else 0
    distances = np.zeros((n_samples, n_clusters, n_cat), dtype=float)

    for j in range(n_cat):
        sum_mu = centroids.mu[j].sum(axis=1)
        for i in range(n_samples):
            value = X_cat_arr[i, j]
            if value not in centroids.value_to_index[j]:
                msg = f"Unseen categorical value {value!r} in column {j}."
                raise ValueError(msg)
            value_idx = centroids.value_to_index[j][value]
            d1 = sum_mu - centroids.mu[j][:, value_idx]
            d2 = centroids.nu[j][:, value_idx]
            distances[i, :, j] = mu_param * d1 + (1.0 - mu_param) * d2
    return distances


def categorical_distance_matrix(
    X_cat: Any,
    centroids: IntuitionisticCentroids | None,
    weights_cat_scaled: np.ndarray,
    mu_param: float,
    n_clusters: int,
) -> np.ndarray:
    """Return weighted categorical distance matrix for Eq. (29)."""

    X_cat_arr = _as_2d_object(X_cat, "X_cat")
    if X_cat_arr.shape[1] == 0:
        return np.zeros((X_cat_arr.shape[0], n_clusters), dtype=float)
    if centroids is None:
        msg = "centroids are required when X_cat is non-empty."
        raise ValueError(msg)
    per_attr = categorical_attribute_distances(X_cat_arr, centroids, mu_param)
    return np.einsum("ikj,j->ik", per_attr, weights_cat_scaled)


def numeric_distance_matrix(
    X_num: Any,
    centroids: np.ndarray,
    weights_num_scaled: np.ndarray,
) -> np.ndarray:
    """Return weighted numerical distance matrix for Eq. (29)."""

    X_num_arr = _as_2d_float(X_num, "X_num")
    if X_num_arr.shape[1] == 0:
        return np.zeros((X_num_arr.shape[0], centroids.shape[0]), dtype=float)
    return np.einsum(
        "ikj,j->ik",
        np.abs(X_num_arr[:, None, :] - centroids[None, :, :]),
        weights_num_scaled,
    )


@dataclass
class IntuitiveKPrototypes:
    """Estimator-style implementation of Algorithm 2."""

    n_clusters: int
    lambda_init: float = 0.8
    mu_param: float = 0.5
    gamma: float = 0.5
    beta: float = 2.0
    max_iter: int = 30
    random_state: int | None = 42
    init_strategy: InitialPrototypeStrategy = "paper"
    strict_init: bool = True
    non_membership: NonMembershipStrategy = "paper"
    empty_cluster_policy: EmptyClusterPolicy = "raise"
    min_cluster_size: int = 1
    view_weight_alpha: float | None = None
    result_: IntuitiveKPrototypesResult | None = field(default=None, init=False)

    def fit(
        self,
        X_num: Any | None = None,
        X_cat: Any | None = None,
    ) -> IntuitiveKPrototypes:
        fit_start = perf_counter()
        if self.empty_cluster_policy not in {"raise", "farthest"}:
            msg = "empty_cluster_policy must be either 'raise' or 'farthest'."
            raise ValueError(msg)
        if self.init_strategy not in {"paper", "farthest_first"}:
            msg = "init_strategy must be either 'paper' or 'farthest_first'."
            raise ValueError(msg)
        if self.n_clusters < 1:
            msg = "n_clusters must be at least 1."
            raise ValueError(msg)
        if self.max_iter < 1:
            msg = "max_iter must be at least 1."
            raise ValueError(msg)
        if self.min_cluster_size < 1:
            msg = "min_cluster_size must be at least 1."
            raise ValueError(msg)
        if not 0 <= self.mu_param <= 1:
            msg = "mu_param must satisfy 0 <= mu_param <= 1."
            raise ValueError(msg)
        if not 0 <= self.gamma <= 1:
            msg = "gamma must satisfy 0 <= gamma <= 1."
            raise ValueError(msg)
        if (
            self.view_weight_alpha is not None
            and not 0 <= self.view_weight_alpha <= 1
        ):
            msg = "view_weight_alpha must satisfy 0 <= view_weight_alpha <= 1."
            raise ValueError(msg)

        X_num_arr = _as_2d_float(X_num, "X_num")
        X_cat_arr = _as_2d_object(X_cat, "X_cat")
        X_num_arr, X_cat_arr = _validate_mixed_inputs(X_num_arr, X_cat_arr)
        if (
            self.empty_cluster_policy == "farthest"
            and self.n_clusters * self.min_cluster_size > X_num_arr.shape[0]
        ):
            msg = (
                "Cannot enforce min_cluster_size because "
                "n_clusters * min_cluster_size exceeds n_samples."
            )
            raise ValueError(msg)

        prototype_idx = (
            select_farthest_first_prototypes(
                X_num_arr,
                X_cat_arr,
                n_clusters=self.n_clusters,
                random_state=self.random_state,
            )
            if self.init_strategy == "farthest_first"
            else select_initial_prototypes(
                X_num_arr,
                X_cat_arr,
                n_clusters=self.n_clusters,
                lambda_init=self.lambda_init,
                random_state=self.random_state,
                strict=self.strict_init,
            )
        )
        initial_distances = mixed_initial_distance_matrix(
            X_num_arr,
            X_cat_arr,
            X_num_arr[prototype_idx],
            X_cat_arr[prototype_idx],
        )
        labels = initial_distances.argmin(axis=1).astype(int)
        if self.empty_cluster_policy == "farthest":
            labels = _repair_small_clusters(
                labels,
                initial_distances,
                self.n_clusters,
                self.min_cluster_size,
            )
        _require_non_empty_clusters(labels, self.n_clusters)

        history: list[IterationRecord] = []
        converged = False
        final_state = self._state_from_labels(X_num_arr, X_cat_arr, labels)

        for iteration in range(1, self.max_iter + 1):
            iteration_start = perf_counter()
            state = self._state_from_labels(X_num_arr, X_cat_arr, labels)
            distances = state["distances"]
            new_labels = distances.argmin(axis=1).astype(int)
            if self.empty_cluster_policy == "farthest":
                new_labels = _repair_small_clusters(
                    new_labels,
                    distances,
                    self.n_clusters,
                    self.min_cluster_size,
                )
            changed = int(np.sum(new_labels != labels))
            cost = float(distances[np.arange(len(labels)), labels].sum())
            history.append(
                IterationRecord(
                    iteration=iteration,
                    changed=changed,
                    cost=cost,
                    elapsed_seconds=perf_counter() - iteration_start,
                )
            )
            final_state = state

            if changed == 0:
                converged = True
                break

            _require_non_empty_clusters(new_labels, self.n_clusters)
            labels = new_labels

        if not converged:
            final_state = self._state_from_labels(X_num_arr, X_cat_arr, labels)

        self.result_ = IntuitiveKPrototypesResult(
            labels=labels.copy(),
            numeric_centroids=final_state["numeric_centroids"],
            categorical_centroids=final_state["categorical_centroids"],
            weights=final_state["weights"],
            distances=final_state["distances"],
            history=tuple(history),
            n_iter=len(history),
            converged=converged,
            fit_time_seconds=perf_counter() - fit_start,
            initial_prototype_indices=prototype_idx,
        )
        return self

    def fit_predict(
        self,
        X_num: Any | None = None,
        X_cat: Any | None = None,
    ) -> np.ndarray:
        return self.fit(X_num=X_num, X_cat=X_cat).result_.labels.copy()

    def _state_from_labels(
        self,
        X_num: np.ndarray,
        X_cat: np.ndarray,
        labels: np.ndarray,
    ) -> dict[str, Any]:
        _require_non_empty_clusters(labels, self.n_clusters)
        num_centroids = numeric_centroids(X_num, labels, self.n_clusters)
        cat_centroids = (
            compute_intuitionistic_centroids(
                X_cat,
                labels,
                self.n_clusters,
                non_membership=self.non_membership,
            )
            if X_cat.shape[1]
            else None
        )
        weights = compute_attribute_weights(
            X_num,
            cat_centroids,
            labels,
            n_clusters=self.n_clusters,
            gamma=self.gamma,
            beta=self.beta,
        )
        if self.view_weight_alpha is None:
            d_num = numeric_distance_matrix(
                X_num,
                num_centroids,
                weights.numeric_scaled,
            )
            d_cat = categorical_distance_matrix(
                X_cat,
                cat_centroids,
                weights.categorical_scaled,
                self.mu_param,
                self.n_clusters,
            )
            distances = d_num + d_cat
        else:
            d_num = numeric_distance_matrix(X_num, num_centroids, weights.numeric)
            d_cat = categorical_distance_matrix(
                X_cat,
                cat_centroids,
                weights.categorical,
                self.mu_param,
                self.n_clusters,
            )
            if X_num.shape[1] and X_cat.shape[1]:
                distances = (
                    self.view_weight_alpha * d_num
                    + (1.0 - self.view_weight_alpha) * d_cat
                )
            elif X_num.shape[1]:
                distances = d_num
            else:
                distances = d_cat
        if not np.isfinite(distances).all():
            msg = "Distance matrix contains non-finite values."
            raise ValueError(msg)
        return {
            "numeric_centroids": num_centroids,
            "categorical_centroids": cat_centroids,
            "weights": weights,
            "distances": distances,
        }
