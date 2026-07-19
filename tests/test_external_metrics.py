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
