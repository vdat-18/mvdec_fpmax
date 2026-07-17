"""Tests for reclustering and comparing saved MvDEC embeddings."""

import numpy as np
import pandas as pd
import pytest

from pipeline.interpretation import attach_row_mapping, build_cluster_profiles
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


def test_build_assignment_frame_includes_both_view_embeddings():
    """Interpretation exports retain both view-specific latent embeddings."""

    source_df = pd.DataFrame({"feature": [10.0, 20.0]})
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
        h_view1=np.array([[1.0], [2.0]]),
        h_view2=np.array([[3.0], [4.0]]),
    )

    assert frame[["view1_1", "view2_1", "fused_1"]].to_dict("list") == {
        "view1_1": [1.0, 2.0],
        "view2_1": [3.0, 4.0],
        "fused_1": [0.1, 0.2],
    }


def test_attach_row_mapping_requires_complete_one_to_one_ids():
    """Mapping joins fail before labels can be attached to the wrong shops."""

    assignments = pd.DataFrame(
        {
            "original_index": [0, 1],
            "artifact_cluster": [1, 0],
        }
    )
    mapping = pd.DataFrame(
        {
            "csv_row_index": [0, 1],
            "Store Name": ["Shop A", "Shop B"],
        }
    )

    interpretation = attach_row_mapping(assignments, mapping)

    assert interpretation["Store Name"].tolist() == ["Shop A", "Shop B"]
    assert interpretation["artifact_cluster"].tolist() == [1, 0]

    incomplete_mapping = mapping.iloc[:1]
    with pytest.raises(ValueError, match="row IDs do not match"):
        attach_row_mapping(assignments, incomplete_mapping)

    duplicate_mapping = pd.concat([mapping.iloc[:1], mapping.iloc[:1]])
    with pytest.raises(ValueError, match="row key must be unique"):
        attach_row_mapping(assignments, duplicate_mapping)


def test_build_cluster_profiles_summarizes_raw_numeric_features():
    """Profiles expose raw distributions without assigning semantic names."""

    interpretation = pd.DataFrame(
        {
            "artifact_cluster": [0, 0, 1],
            "Followers": [10, 30, 100],
            "Total Revenue": [100.0, 300.0, 900.0],
        }
    )

    profiles = build_cluster_profiles(
        interpretation,
        ["Followers", "Total Revenue"],
    )

    followers_cluster_zero = profiles.loc[
        (profiles["cluster"] == 0) & (profiles["feature"] == "Followers")
    ].iloc[0]
    assert followers_cluster_zero["cluster_sample_count"] == 2
    assert followers_cluster_zero["cluster_share"] == pytest.approx(2 / 3)
    assert followers_cluster_zero["median"] == 20
    assert followers_cluster_zero["q1"] == 15
    assert followers_cluster_zero["q3"] == 25
