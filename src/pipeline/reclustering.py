"""Recluster fused MvDEC embeddings and compare saved assignments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.metrics.cluster import contingency_matrix


@dataclass(frozen=True)
class KMeansConfig:
    """K-Means parameters recorded by an MvDEC artifact."""

    n_clusters: int
    n_init: int
    random_state: int


@dataclass(frozen=True)
class ReclusteringResult:
    """Assignments and agreement metrics from reclustering one embedding."""

    labels_raw: np.ndarray
    labels_aligned: np.ndarray
    silhouette: float
    ari: float
    nmi: float
    changed_count: int
    inertia: float
    n_iter: int


def artifact_kmeans_config(raw: dict) -> KMeansConfig:
    """Read and cross-check K-Means settings stored in an artifact."""

    nested = raw.get("config")
    if not isinstance(nested, dict):
        msg = "MvDEC artifact must contain a config dictionary."
        raise ValueError(msg)

    required_top_level = ("n_clusters", "kmeans_n_init")
    missing = [key for key in required_top_level if key not in raw]
    if "seed" not in nested:
        missing.append("config.seed")
    if missing:
        msg = f"MvDEC artifact is missing K-Means settings: {missing}."
        raise ValueError(msg)

    n_clusters = int(raw["n_clusters"])
    n_init = int(raw["kmeans_n_init"])
    random_state = int(nested["seed"])
    if n_clusters < 2 or n_init < 1:
        msg = "MvDEC artifact requires n_clusters >= 2 and kmeans_n_init >= 1."
        raise ValueError(msg)

    if nested.get("n_clusters") != n_clusters:
        msg = "Top-level and config n_clusters values do not match."
        raise ValueError(msg)
    if nested.get("kmeans_n_init") != n_init:
        msg = "Top-level and config kmeans_n_init values do not match."
        raise ValueError(msg)

    return KMeansConfig(
        n_clusters=n_clusters,
        n_init=n_init,
        random_state=random_state,
    )


def align_cluster_labels(
    reference: np.ndarray,
    candidate: np.ndarray,
    expected_clusters: int,
) -> np.ndarray:
    """Align candidate cluster IDs to reference IDs by maximum overlap."""

    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    if reference.ndim != 1 or candidate.ndim != 1:
        msg = "Cluster labels must be one-dimensional arrays."
        raise ValueError(msg)
    if len(reference) != len(candidate):
        msg = "Reference and candidate labels must have the same length."
        raise ValueError(msg)

    reference_ids = np.unique(reference)
    candidate_ids = np.unique(candidate)
    if len(reference_ids) != expected_clusters:
        msg = (
            "Saved assignments contain "
            f"{len(reference_ids)} clusters; expected {expected_clusters}."
        )
        raise ValueError(msg)
    if len(candidate_ids) != expected_clusters:
        msg = (
            "Reclustered assignments contain "
            f"{len(candidate_ids)} clusters; expected {expected_clusters}."
        )
        raise ValueError(msg)

    overlap = contingency_matrix(reference, candidate)
    reference_positions, candidate_positions = linear_sum_assignment(
        overlap,
        maximize=True,
    )
    label_map = {
        candidate_ids[candidate_position]: reference_ids[reference_position]
        for reference_position, candidate_position in zip(
            reference_positions,
            candidate_positions,
            strict=True,
        )
    }
    return np.asarray([label_map[label] for label in candidate])


def recluster_embedding(
    h_fused: np.ndarray,
    artifact_labels: np.ndarray,
    config: KMeansConfig,
) -> ReclusteringResult:
    """Run the same K-Means call used by MvDEC and measure agreement."""

    h_fused = np.asarray(h_fused)
    artifact_labels = np.asarray(artifact_labels)
    if h_fused.ndim != 2 or len(h_fused) != len(artifact_labels):
        msg = "h_fused must be 2D and match the saved label count."
        raise ValueError(msg)
    if len(h_fused) <= config.n_clusters:
        msg = "Silhouette evaluation requires more samples than clusters."
        raise ValueError(msg)

    model = KMeans(
        n_clusters=config.n_clusters,
        n_init=config.n_init,
        random_state=config.random_state,
    )
    labels_raw = model.fit_predict(h_fused)
    labels_aligned = align_cluster_labels(
        artifact_labels,
        labels_raw,
        expected_clusters=config.n_clusters,
    )

    return ReclusteringResult(
        labels_raw=labels_raw,
        labels_aligned=labels_aligned,
        silhouette=float(silhouette_score(h_fused, labels_raw)),
        ari=float(adjusted_rand_score(artifact_labels, labels_raw)),
        nmi=float(normalized_mutual_info_score(artifact_labels, labels_raw)),
        changed_count=int(np.count_nonzero(artifact_labels != labels_aligned)),
        inertia=float(model.inertia_),
        n_iter=int(model.n_iter_),
    )


def build_assignment_frame(
    source_df: pd.DataFrame,
    h_fused_df: pd.DataFrame,
    artifact_labels: np.ndarray,
    result: ReclusteringResult,
    *,
    h_view1: np.ndarray | None = None,
    h_view2: np.ndarray | None = None,
) -> pd.DataFrame:
    """Combine source features, fused features, and comparable cluster labels."""

    row_count = len(source_df)
    arrays = (h_fused_df, artifact_labels, result.labels_raw, result.labels_aligned)
    if any(len(values) != row_count for values in arrays):
        msg = "Source, embedding, and cluster assignment row counts must match."
        raise ValueError(msg)
    if (h_view1 is None) != (h_view2 is None):
        msg = "View embeddings must either both be provided or both be omitted."
        raise ValueError(msg)
    if h_view1 is not None and (len(h_view1) != row_count or len(h_view2) != row_count):
        msg = "View embedding row counts must match the source dataset."
        raise ValueError(msg)

    feature_frame = source_df.reset_index(drop=True).copy()
    feature_frame.insert(0, "original_index", source_df.index.to_numpy())
    embedding_frames = [h_fused_df.reset_index(drop=True)]
    if h_view1 is not None and h_view2 is not None:
        embedding_frames = [
            pd.DataFrame(
                h_view1,
                columns=[f"view1_{index}" for index in range(1, h_view1.shape[1] + 1)],
            ),
            pd.DataFrame(
                h_view2,
                columns=[f"view2_{index}" for index in range(1, h_view2.shape[1] + 1)],
            ),
            *embedding_frames,
        ]
    label_frame = pd.DataFrame(
        {
            "artifact_cluster": np.asarray(artifact_labels, dtype=int),
            "reclustered_cluster_raw": np.asarray(result.labels_raw, dtype=int),
            "reclustered_cluster_aligned": np.asarray(
                result.labels_aligned,
                dtype=int,
            ),
        }
    )
    return pd.concat(
        [feature_frame, *embedding_frames, label_frame],
        axis=1,
    )


def cluster_sizes(labels: np.ndarray) -> dict[int, int]:
    """Return cluster sizes keyed by their actual label IDs."""

    ids, counts = np.unique(labels, return_counts=True)
    return {int(label): int(count) for label, count in zip(ids, counts, strict=True)}
