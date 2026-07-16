"""Tests for reclustering and comparing saved MvDEC embeddings."""

import numpy as np
import pandas as pd
import pytest

from pipeline.reclustering import (
    KMeansConfig,
    ReclusteringResult,
    align_cluster_labels,
    artifact_kmeans_config,
    build_assignment_frame,
    recluster_embedding,
)


def test_align_cluster_labels_handles_id_permutation():
    reference = np.array([0, 0, 1, 1, 2, 2])
    candidate = np.array([2, 2, 0, 0, 1, 1])

    aligned = align_cluster_labels(reference, candidate, expected_clusters=3)

    np.testing.assert_array_equal(aligned, reference)


def test_align_cluster_labels_rejects_cluster_count_mismatch():
    reference = np.array([0, 0, 1, 1])
    candidate = np.array([0, 0, 0, 0])

    with pytest.raises(ValueError, match="Reclustered assignments contain 1 clusters"):
        align_cluster_labels(reference, candidate, expected_clusters=2)


def test_artifact_kmeans_config_rejects_inconsistent_metadata():
    raw = {
        "n_clusters": 3,
        "kmeans_n_init": 100,
        "config": {"n_clusters": 4, "kmeans_n_init": 100, "seed": 42},
    }

    with pytest.raises(ValueError, match="n_clusters values do not match"):
        artifact_kmeans_config(raw)


def test_recluster_embedding_reproduces_separated_assignments():
    h_fused = np.array(
        [
            [-5.1, -5.0],
            [-4.9, -5.0],
            [0.0, 5.0],
            [0.1, 5.1],
            [5.0, -5.0],
            [5.1, -4.9],
        ]
    )
    config = KMeansConfig(n_clusters=3, n_init=20, random_state=42)
    first_run = recluster_embedding(
        h_fused,
        artifact_labels=np.array([0, 0, 1, 1, 2, 2]),
        config=config,
    )

    assert first_run.ari == pytest.approx(1.0)
    assert first_run.nmi == pytest.approx(1.0)
    assert first_run.changed_count == 0


def test_build_assignment_frame_preserves_rows_and_columns():
    source_df = pd.DataFrame({"feature": [10.0, 20.0]}, index=[7, 9])
    h_fused_df = pd.DataFrame({"fused_1": [0.1, 0.2]})
    result = ReclusteringResult(
        labels_raw=np.array([1, 0]),
        labels_aligned=np.array([0, 1]),
        silhouette=0.5,
        ari=1.0,
        nmi=1.0,
        changed_count=0,
        inertia=0.1,
        n_iter=2,
    )

    frame = build_assignment_frame(
        source_df,
        h_fused_df,
        artifact_labels=np.array([0, 1]),
        result=result,
    )

    assert frame.columns.tolist() == [
        "original_index",
        "feature",
        "fused_1",
        "artifact_cluster",
        "reclustered_cluster_raw",
        "reclustered_cluster_aligned",
    ]
    assert frame["original_index"].tolist() == [7, 9]
    assert frame["feature"].tolist() == [10.0, 20.0]
