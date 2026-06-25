"""Tests for the two documented Eq. (4) interpretations."""

import numpy as np
import pytest
from intuitive_kprototypes import compute_intuitionistic_centroids


def test_paper_non_membership_keeps_intuitionistic_constraint():
    X_cat = np.asarray([["a"]] * 80 + [["b"]] * 10 + [["a"]] * 9 + [["c"]])
    labels = np.asarray([0] * 90 + [1] * 10)

    idc = compute_intuitionistic_centroids(
        X_cat,
        labels,
        n_clusters=2,
        non_membership="paper",
    )

    assert np.all(idc.mu[0] + idc.nu[0] <= 1 + 1e-12)


def test_table_non_membership_is_only_safe_for_paper_tables():
    X_cat = np.asarray([["a"]] * 80 + [["b"]] * 10 + [["a"]] * 9 + [["c"]])
    labels = np.asarray([0] * 90 + [1] * 10)

    with pytest.raises(RuntimeError, match="mu \\+ nu"):
        compute_intuitionistic_centroids(
            X_cat,
            labels,
            n_clusters=2,
            non_membership="table",
        )
