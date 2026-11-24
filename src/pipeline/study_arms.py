"""Fit one clustering arm (algorithm plus optional FP-Max itemsets) on study data."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd

from intuitive_kprototypes import IntuitiveKPrototypes
from pipeline.clustering import fit_kprototypes_by_cost, is_valid_partition
from pipeline.study_config import StudyConfig
from pipeline.study_data import StudyData

ALGORITHMS = ("kmeans", "kprototypes", "intuitive")
STATUS_OK = "ok"
STATUS_INVALID = "invalid_partition"
STATUS_ERROR = "error"
STATUS_NOT_APPLICABLE = "not_applicable"
RESTART_SEED_STRIDE = 1000


@dataclass(frozen=True)
class ArmFit:
    """Outcome of one arm fit; labels are set only when the partition is valid."""

    labels: np.ndarray | None
    status: str
    cost: float
    error: str | None
    fit_seconds: float
    init: str | None


def _failure(status: str, error: str | None, started: float) -> ArmFit:
    return ArmFit(
        labels=None,
        status=status,
        cost=float("nan"),
        error=error,
        fit_seconds=perf_counter() - started,
        init=None,
    )


def _categorical_frame(data: StudyData, itemsets: pd.DataFrame | None) -> pd.DataFrame:
    """Original categorical attributes plus itemset indicators, as strings."""

    blocks = [data.categorical.astype(str)]
    if itemsets is not None and itemsets.shape[1]:
        blocks.append(itemsets.astype(int).astype(str))
    return pd.concat(blocks, axis=1).astype(object)


def _fit_kprototypes_family(
    data: StudyData,
    algorithm: str,
    itemsets: pd.DataFrame | None,
    seed: int,
    study: StudyConfig,
    started: float,
) -> ArmFit:
    if algorithm == "kmeans":
        categorical = pd.DataFrame(index=data.numeric.index)
    else:
        categorical = _categorical_frame(data, itemsets)
    numeric_cols = list(data.numeric.columns)
    combined = pd.concat([data.numeric, categorical], axis=1)
    categorical_idx = list(
        range(data.numeric.shape[1], data.numeric.shape[1] + categorical.shape[1])
    )
    try:
        fit = fit_kprototypes_by_cost(
            combined,
            categorical_idx,
            numeric_cols,
            data.n_clusters,
            study.kprototypes_init_methods,
            random_state=seed,
        )
    except (RuntimeError, ValueError) as error:
        status = STATUS_INVALID if "No valid" in str(error) else STATUS_ERROR
        return _failure(status, str(error), started)
    return ArmFit(
        labels=fit.labels,
        status=STATUS_OK,
        cost=fit.cost,
        error=None,
        fit_seconds=perf_counter() - started,
        init=fit.init,
    )


def _fit_intuitive(
    data: StudyData,
    itemsets: pd.DataFrame | None,
    seed: int,
    study: StudyConfig,
    started: float,
    restarts: int,
) -> ArmFit:
    params = study.intuitive.for_dataset(data.name)
    categorical = _categorical_frame(data, itemsets)
    X_num = data.numeric.to_numpy(dtype=float) if data.numeric.shape[1] else None
    X_cat = categorical.to_numpy(dtype=object) if categorical.shape[1] else None

    best_labels: np.ndarray | None = None
    best_cost = float("inf")
    last_error: str | None = None
    n_invalid = 0
    for restart in range(restarts):
        model = IntuitiveKPrototypes(
            n_clusters=data.n_clusters,
            mu_param=params.mu,
            gamma=params.gamma,
            beta=params.beta,
            max_iter=study.intuitive.max_iter,
            random_state=seed * RESTART_SEED_STRIDE + restart,
            strict_init=False,
            empty_cluster_policy="farthest",
            min_cluster_size=1,
            init_strategy=study.intuitive.init_strategy,
            mu_convention=study.intuitive.mu_convention,
        )
        try:
            result = model.fit(X_num=X_num, X_cat=X_cat).result_
        except (RuntimeError, ValueError) as error:
            last_error = str(error)
            continue
        labels = np.asarray(result.labels)
        if not is_valid_partition(labels, data.n_clusters):
            n_invalid += 1
            continue
        cost = float(result.distances[np.arange(len(labels)), labels].sum())
        if cost < best_cost:
            best_labels, best_cost = labels, cost

    if best_labels is None:
        # "error" only when no restart produced a partition at all
        status = STATUS_ERROR if last_error and not n_invalid else STATUS_INVALID
        return _failure(status, last_error, started)
    return ArmFit(
        labels=best_labels,
        status=STATUS_OK,
        cost=best_cost,
        error=None,
        fit_seconds=perf_counter() - started,
        init=study.intuitive.init_strategy,
    )


def fit_arm(
    data: StudyData,
    algorithm: str,
    itemsets: pd.DataFrame | None,
    seed: int,
    study: StudyConfig,
    restarts: int | None = None,
) -> ArmFit:
    """Fit ``algorithm`` on the data plus optional itemset columns.

    ``restarts`` overrides the Intuitive restart count (other algorithms ignore it).

    Failures of the model fit itself (``RuntimeError``, ``ValueError``) become a
    status and message; anything else is a bug and propagates. True labels are never
    read here.
    """

    if algorithm not in ALGORITHMS:
        msg = f"algorithm must be one of {ALGORITHMS}, got {algorithm!r}."
        raise ValueError(msg)
    started = perf_counter()
    if algorithm == "kmeans" and data.numeric.shape[1] == 0:
        return _failure(STATUS_NOT_APPLICABLE, "no numeric attributes", started)
    if algorithm == "intuitive":
        n_restarts = study.intuitive.restarts if restarts is None else restarts
        return _fit_intuitive(data, itemsets, seed, study, started, n_restarts)
    return _fit_kprototypes_family(data, algorithm, itemsets, seed, study, started)
