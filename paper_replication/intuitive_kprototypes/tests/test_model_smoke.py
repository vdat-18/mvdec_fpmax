"""Smoke tests for the full IntuitiveKPrototypes loop."""

import numpy as np
from intuitive_kprototypes import IntuitiveKPrototypes


def test_full_loop_mixed_data_smoke():
    X_num = np.asarray(
        [
            [1.0, 1.2],
            [1.1, 1.0],
            [0.9, 1.1],
            [8.0, 7.8],
            [8.2, 8.1],
            [7.9, 8.0],
        ]
    )
    X_cat = np.asarray(
        [
            ["a", "x"],
            ["a", "x"],
            ["a", "y"],
            ["b", "z"],
            ["b", "z"],
            ["b", "y"],
        ],
        dtype=object,
    )

    model = IntuitiveKPrototypes(
        n_clusters=2,
        lambda_init=0.5,
        mu_param=0.5,
        gamma=0.5,
        beta=2.0,
        max_iter=10,
        random_state=0,
        strict_init=True,
    )
    labels = model.fit_predict(X_num=X_num, X_cat=X_cat)

    assert labels.shape == (6,)
    assert set(labels.tolist()) == {0, 1}
    assert model.result_ is not None
    assert np.isfinite(model.result_.distances).all()
    assert np.isclose(model.result_.weights.numeric_scaled.sum(), X_num.shape[1])
    assert np.isclose(model.result_.weights.categorical_scaled.sum(), X_cat.shape[1])
    assert model.result_.fit_time_seconds >= 0.0
    assert np.isfinite(model.result_.fit_time_seconds)
    assert len(model.result_.history) == model.result_.n_iter
    assert all(record.elapsed_seconds >= 0.0 for record in model.result_.history)


def test_full_loop_categorical_only_smoke():
    X_cat = np.asarray(
        [
            ["a", "x"],
            ["a", "x"],
            ["b", "x"],
            ["c", "y"],
            ["c", "y"],
            ["d", "y"],
        ],
        dtype=object,
    )
    model = IntuitiveKPrototypes(
        n_clusters=2,
        lambda_init=0.5,
        max_iter=10,
        random_state=1,
        strict_init=False,
    )
    labels = model.fit_predict(X_cat=X_cat)

    assert labels.shape == (6,)
    assert set(labels.tolist()) == {0, 1}
    assert model.result_ is not None
    assert model.result_.numeric_centroids.shape == (2, 0)
    assert np.isclose(model.result_.weights.categorical_scaled.sum(), X_cat.shape[1])
    assert model.result_.fit_time_seconds >= 0.0


def test_empty_cluster_farthest_policy_repairs_degenerate_assignment():
    X_cat = np.asarray([["a"]] * 6, dtype=object)
    model = IntuitiveKPrototypes(
        n_clusters=3,
        lambda_init=0.5,
        max_iter=2,
        random_state=0,
        strict_init=False,
        empty_cluster_policy="farthest",
        min_cluster_size=2,
    )
    labels = model.fit_predict(X_cat=X_cat)

    assert labels.shape == (6,)
    assert set(labels.tolist()) == {0, 1, 2}
    assert model.result_ is not None
    assert np.bincount(labels, minlength=3).min() >= 2
