"""Tests for the single Intuitive-K-prototypes package used by the pipeline."""

from pathlib import Path

import numpy as np
import pytest

import intuitive_kprototypes
from intuitive_kprototypes import IntuitiveKPrototypes, select_initial_prototypes


def test_single_package_is_loaded_from_src():
    package_dir = Path(intuitive_kprototypes.__file__).resolve().parent
    assert package_dir.parts[-2:] == ("src", "intuitive_kprototypes")
    assert "paper_replication" not in package_dir.parts


def test_farthest_first_picks_extremes_on_a_line():
    X = np.array([[0.0], [1.0], [10.0], [11.0]])
    idx = select_initial_prototypes(
        X, None, n_clusters=2, random_state=0, strategy="farthest_first"
    )
    assert max(X[idx, 0]) - min(X[idx, 0]) >= 10.0


def test_view_weight_alpha_one_ignores_categorical_block():
    X_num = [[0.0], [0.1], [1.0], [1.1]]
    X_cat = [["a"], ["b"], ["a"], ["b"]]
    model = IntuitiveKPrototypes(
        n_clusters=2,
        view_weight_alpha=1.0,
        random_state=0,
        init_strategy="farthest_first",
    ).fit(X_num=X_num, X_cat=X_cat)
    labels = model.result_.labels
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_view_weight_alpha_is_validated():
    with pytest.raises(ValueError, match="view_weight_alpha"):
        IntuitiveKPrototypes(n_clusters=2, view_weight_alpha=1.5).fit(
            X_num=[[0.0], [1.0], [2.0]], X_cat=None
        )


def test_negative_numeric_means_give_finite_weights():
    X = np.array([[-1.0, -2.0], [-1.1, -2.2], [-5.0, -9.0], [-5.2, -9.1]])
    model = IntuitiveKPrototypes(
        n_clusters=2, random_state=0, init_strategy="farthest_first"
    ).fit(X_num=X, X_cat=None)
    assert np.isfinite(model.result_.weights.numeric).all()
