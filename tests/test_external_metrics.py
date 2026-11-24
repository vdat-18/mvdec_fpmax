"""Tests for shared labeled-benchmark clustering metrics."""

import numpy as np
import pytest

from pipeline.external_metrics import compute_external_metrics


def test_external_metrics_are_invariant_to_cluster_id_permutation() -> None:
    """Hungarian ACC and NMI must ignore arbitrary cluster identifiers."""

    metrics = compute_external_metrics(
        [0, 0, 1, 1],
        [7, 7, 3, 3],
        n_clusters=2,
    )

    assert metrics.acc == 1.0
    assert metrics.nmi == 1.0


def test_external_metrics_reject_missing_clusters() -> None:
    """Public benchmark runs must use every configured cluster."""

    with pytest.raises(ValueError, match="exactly n_clusters"):
        compute_external_metrics(
            np.array([0, 0, 1, 1]),
            np.array([0, 0, 0, 0]),
            n_clusters=2,
        )


def test_external_metrics_report_ari() -> None:
    """ARI is 1 for identical partitions and below 1 for a mixed one."""

    perfect = compute_external_metrics([0, 0, 1, 1], [5, 5, 2, 2], n_clusters=2)
    mixed = compute_external_metrics([0, 0, 1, 1], [0, 1, 0, 1], n_clusters=2)

    assert perfect.ari == 1.0
    assert mixed.ari < 0.1


def test_external_metrics_tolerate_empty_predicted_cluster_when_allowed() -> None:
    """A collapsed partition must still be scoreable instead of crashing."""

    metrics = compute_external_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0, 0, 0, 0]),
        n_clusters=2,
        allow_missing_clusters=True,
    )

    assert metrics.acc == 0.5
    assert metrics.ari == 0.0
