"""Forward selection of FP-Max itemsets for the study arms.

The selection criterion is a label-free silhouette computed in the candidate's own
Gower space (original attributes plus the itemsets under test). It steers selection
only and must never be reported as clustering quality; reported scores come from
``reference_evaluation``.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from pipeline.clustering import is_valid_partition
from pipeline.reference_evaluation import gower_distance_matrix
from pipeline.study_arms import STATUS_OK, fit_arm
from pipeline.study_config import StudyConfig
from pipeline.study_data import StudyData

SUBSAMPLE_SEED = 0


@dataclass(frozen=True)
class FfsResult:
    """Selected itemset columns and the labels of the final accepted step."""

    selected: list[str]
    labels: np.ndarray
    n_candidates_evaluated: int
    final_score: float


def candidate_silhouette(
    data: StudyData,
    itemsets: pd.DataFrame,
    labels: np.ndarray,
    max_rows: int,
) -> float:
    """Silhouette of ``labels`` under Gower on attributes plus ``itemsets``.

    Rows above ``max_rows`` are subsampled with a fixed seed. Returns ``nan`` when
    the labelling has fewer than two clusters or a singleton cluster in the sample.
    """

    labels = np.asarray(labels)
    n_rows = len(labels)
    if n_rows > max_rows:
        rng = np.random.default_rng(SUBSAMPLE_SEED)
        rows = np.sort(rng.choice(n_rows, size=max_rows, replace=False))
    else:
        rows = np.arange(n_rows)
    sample_labels = labels[rows]
    n_clusters = len(np.unique(labels))
    if n_clusters < 2 or not is_valid_partition(sample_labels, n_clusters):
        return float("nan")

    blocks = [data.numeric, data.categorical]
    if itemsets.shape[1]:
        blocks.append(itemsets.astype(int).astype(str))
    frame = pd.concat(blocks, axis=1).iloc[rows]
    distance = gower_distance_matrix(frame)
    return float(silhouette_score(distance, sample_labels, metric="precomputed"))


def forward_select_itemsets(
    data: StudyData,
    algorithm: str,
    itemsets: pd.DataFrame,
    seed: int,
    study: StudyConfig,
    baseline_labels: np.ndarray,
) -> FfsResult:
    """Greedily add the itemset that most improves the candidate-space silhouette.

    A candidate is accepted only when it beats the current score by more than
    ``study.ffs_min_improvement``; failed or invalid fits never win. With nothing
    accepted the result carries ``baseline_labels`` unchanged.
    """

    if algorithm == "kmeans":
        msg = "Forward selection needs an algorithm that uses itemsets."
        raise ValueError(msg)

    empty = itemsets.iloc[:, :0]
    current_score = candidate_silhouette(
        data, empty, baseline_labels, study.ffs_max_rows
    )
    current_labels = np.asarray(baseline_labels)
    selected: list[str] = []
    remaining = list(itemsets.columns)
    n_evaluated = 0

    while remaining and not np.isnan(current_score):
        best_name: str | None = None
        best_score = -np.inf
        best_labels: np.ndarray | None = None
        for name in remaining:
            candidate = itemsets[selected + [name]]
            fit = fit_arm(data, algorithm, candidate, seed, study)
            n_evaluated += 1
            if fit.status != STATUS_OK or fit.labels is None:
                continue
            score = candidate_silhouette(
                data, candidate, fit.labels, study.ffs_max_rows
            )
            if not np.isnan(score) and score > best_score:
                best_name, best_score, best_labels = name, score, fit.labels
        if (
            best_name is None
            or best_labels is None
            or best_score <= current_score + study.ffs_min_improvement
        ):
            break
        selected.append(best_name)
        remaining.remove(best_name)
        current_score, current_labels = best_score, best_labels

    return FfsResult(
        selected=selected,
        labels=current_labels,
        n_candidates_evaluated=n_evaluated,
        final_score=float(current_score),
    )
