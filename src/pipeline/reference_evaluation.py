"""Fixed reference-space scoring for clustering results.

Every method (with or without FP-Max) is scored in the same distance space, built
once from the dataset's original attributes. The space never contains a method's own
FP-Max columns, so scores are comparable across methods and cannot be inflated by
adding features that mirror the clustering.
"""

from dataclasses import dataclass

import gower
import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from pipeline.clustering import is_valid_partition

DEFAULT_MAX_ROWS = 3000
EMBEDDING_DIMENSIONS = 10


@dataclass(frozen=True)
class ReferenceSpace:
    """Pairwise distances, a Euclidean embedding and the rows they describe."""

    distance: np.ndarray
    embedding: np.ndarray
    row_index: np.ndarray
    n_rows: int


@dataclass(frozen=True)
class InternalScores:
    """Label-free quality scores in the reference space (nan if invalid)."""

    silhouette: float
    davies_bouldin: float
    calinski_harabasz: float


def _classical_mds(distance: np.ndarray, n_components: int) -> np.ndarray:
    """Embed a distance matrix with classical (Torgerson) multidimensional scaling."""

    n = distance.shape[0]
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ (distance**2) @ centering
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1][:n_components]
    top_values = np.clip(values[order], 0.0, None)
    return vectors[:, order] * np.sqrt(top_values)


def build_reference_distance(
    df: pd.DataFrame,
    max_rows: int = DEFAULT_MAX_ROWS,
    random_state: int = 0,
) -> ReferenceSpace:
    """Build the Gower reference space, subsampling rows above ``max_rows``."""

    n_rows = len(df)
    if n_rows > max_rows:
        rng = np.random.default_rng(random_state)
        row_index = np.sort(rng.choice(n_rows, size=max_rows, replace=False))
    else:
        row_index = np.arange(n_rows)
    sample = df.iloc[row_index].reset_index(drop=True)
    # gower cannot interpret pandas' extension string dtype; plain object works.
    text_columns = [
        c for c in sample.columns if not pd.api.types.is_numeric_dtype(sample[c])
    ]
    sample = sample.astype({column: object for column in text_columns})
    distance = np.asarray(gower.gower_matrix(sample), dtype=float)
    embedding = _classical_mds(distance, min(EMBEDDING_DIMENSIONS, len(sample) - 1))
    return ReferenceSpace(
        distance=distance,
        embedding=embedding,
        row_index=row_index,
        n_rows=n_rows,
    )


def score_internal(space: ReferenceSpace, labels: np.ndarray) -> InternalScores:
    """Score a labelling of all rows; invalid partitions give nan for every score."""

    labels = np.asarray(labels)
    if len(labels) != space.n_rows:
        msg = f"labels must cover all {space.n_rows} rows, got {len(labels)}."
        raise ValueError(msg)
    sample_labels = labels[space.row_index]
    n_clusters = len(np.unique(sample_labels))
    if n_clusters < 2 or not is_valid_partition(sample_labels, n_clusters):
        return InternalScores(np.nan, np.nan, np.nan)
    return InternalScores(
        silhouette=float(
            silhouette_score(space.distance, sample_labels, metric="precomputed")
        ),
        davies_bouldin=float(davies_bouldin_score(space.embedding, sample_labels)),
        calinski_harabasz=float(
            calinski_harabasz_score(space.embedding, sample_labels)
        ),
    )


def pairwise_ari_stability(label_sets: list[np.ndarray]) -> float:
    """Mean adjusted Rand index over all pairs of labellings."""

    if len(label_sets) < 2:
        msg = "At least two labellings are required."
        raise ValueError(msg)
    scores = [
        adjusted_rand_score(label_sets[i], label_sets[j])
        for i in range(len(label_sets))
        for j in range(i + 1, len(label_sets))
    ]
    return float(np.mean(scores))
