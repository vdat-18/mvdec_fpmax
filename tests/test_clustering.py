"""Tests for the clustering helpers shared by the FP-Max study."""

import numpy as np
import pandas as pd
import pytest

from pipeline import clustering


def test_minmax_scale_block_bounds_and_constant_column():
    df = pd.DataFrame({"a": [-3.0, 0.0, 3.0], "b": [5.0, 5.0, 5.0]})

    scaled = clustering.minmax_scale_block(df)

    assert scaled["a"].tolist() == [0.0, 0.5, 1.0]
    assert scaled["b"].tolist() == [0.0, 0.0, 0.0]


def test_run_intuitive_passes_mu_convention(monkeypatch):
    captured = {}
    real = clustering.IntuitiveKPrototypes

    def recording_model(**kwargs):
        captured.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(clustering, "IntuitiveKPrototypes", recording_model)
    continuous = pd.DataFrame({"x": [0.0, 0.1, 1.0, 1.1, 0.05, 1.05]})
    binary = pd.DataFrame({"f": [1, 1, 0, 0, 1, 0]})

    result = clustering.run_intuitive_kprototypes(
        continuous,
        binary,
        n_clusters=2,
        mu_convention="swapped",
        verbose=False,
    )

    assert captured["mu_convention"] == "swapped"
    assert result.mu_convention == "swapped"
    assert np.unique(result.labels).size == 2


def _two_blobs() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    labels = np.repeat([0, 1], 20)
    continuous = pd.DataFrame(
        rng.normal(0.0, 0.1, size=(40, 2)) + labels[:, None] * 3.0,
        columns=["x", "y"],
    )
    binary = pd.DataFrame({"f": labels, "g": 1 - labels})
    return continuous, binary


class _FakeKPrototypes:
    """Stand-in returning scripted (labels, cost) per init."""

    script: dict[str, tuple[np.ndarray, float]] = {}

    def __init__(self, init, **_kwargs):
        self.init = init
        self.cost_ = float("nan")

    def fit_predict(self, _matrix, categorical):
        labels, self.cost_ = self.script[self.init]
        return labels


def test_is_valid_partition():
    assert clustering.is_valid_partition(np.array([0, 0, 1, 1]), 2)
    assert not clustering.is_valid_partition(np.array([0, 0, 0, 1]), 2)
    assert not clustering.is_valid_partition(np.array([0, 0, 0, 0]), 2)
    assert not clustering.is_valid_partition(np.array([0, 0, 1, 1, 2, 2]), 2)


def test_empty_binary_block_uses_kmeans_equivalent():
    continuous, _ = _two_blobs()

    result = clustering.run_kprototypes(
        continuous, pd.DataFrame(index=continuous.index), 2, verbose=False
    )

    assert result.backend == "kmeans_equivalent"
    assert clustering.is_valid_partition(result.labels, 2)


def test_init_choice_uses_cost_not_silhouette(monkeypatch):
    continuous, binary = _two_blobs()
    good = np.repeat([0, 1], 20)
    # same partition shape, but "huang" claims a higher cost than "cao"
    _FakeKPrototypes.script = {"huang": (good, 10.0), "cao": (good, 5.0)}
    monkeypatch.setattr(clustering, "KPrototypes", _FakeKPrototypes)

    result = clustering.run_kprototypes(continuous, binary, 2, verbose=False)

    assert result.init == "cao"
    assert result.cost == 5.0
    assert result.backend == "kprototypes"


def test_invalid_restart_is_skipped_even_when_cheaper(monkeypatch):
    continuous, binary = _two_blobs()
    good = np.repeat([0, 1], 20)
    singleton = np.zeros(40, dtype=int)
    singleton[0] = 1
    _FakeKPrototypes.script = {"huang": (singleton, 1.0), "cao": (good, 9.0)}
    monkeypatch.setattr(clustering, "KPrototypes", _FakeKPrototypes)

    result = clustering.run_kprototypes(continuous, binary, 2, verbose=False)

    assert result.init == "cao"


def test_all_invalid_restarts_raise(monkeypatch):
    continuous, binary = _two_blobs()
    singleton = np.zeros(40, dtype=int)
    singleton[0] = 1
    _FakeKPrototypes.script = {"huang": (singleton, 1.0), "cao": (singleton, 2.0)}
    monkeypatch.setattr(clustering, "KPrototypes", _FakeKPrototypes)

    with pytest.raises(RuntimeError, match="No valid clustering"):
        clustering.run_kprototypes(continuous, binary, 2, verbose=False)


def test_init_that_cannot_initialize_is_skipped(monkeypatch):
    continuous, binary = _two_blobs()
    good = np.repeat([0, 1], 20)

    class Flaky(_FakeKPrototypes):
        def fit_predict(self, _matrix, categorical):
            if self.init == "huang":
                msg = "Clustering algorithm could not initialize."
                raise ValueError(msg)
            return super().fit_predict(_matrix, categorical)

    Flaky.script = {"cao": (good, 3.0)}
    monkeypatch.setattr(clustering, "KPrototypes", Flaky)

    result = clustering.run_kprototypes(continuous, binary, 2, verbose=False)

    assert result.init == "cao"


def test_unrelated_value_error_is_not_swallowed(monkeypatch):
    continuous, binary = _two_blobs()

    class Broken(_FakeKPrototypes):
        def fit_predict(self, _matrix, categorical):
            msg = "bad input shape"
            raise ValueError(msg)

    monkeypatch.setattr(clustering, "KPrototypes", Broken)

    with pytest.raises(ValueError, match="bad input shape"):
        clustering.run_kprototypes(continuous, binary, 2, verbose=False)


def test_categorical_only_uses_kmodes():
    rng = np.random.default_rng(1)
    group = np.repeat([0, 1], 20)
    values = np.where(
        rng.random((40, 3)) < 0.9, group[:, None], 1 - group[:, None]
    ).astype(str)
    frame = pd.DataFrame(values, columns=["a", "b", "c"]).astype(object)

    fit = clustering.fit_kprototypes_by_cost(
        frame,
        [0, 1, 2],
        [],
        2,
        ("huang", "cao"),
        random_state=0,
    )

    assert fit.backend == "kmodes"
    assert clustering.is_valid_partition(fit.labels, 2)


def test_fit_kprototypes_by_cost_matches_run_kprototypes():
    continuous, binary = _two_blobs()
    combined, categorical_idx, _ = clustering.make_mixed_features(continuous, binary)

    fit = clustering.fit_kprototypes_by_cost(
        combined,
        categorical_idx,
        list(continuous.columns),
        2,
        clustering.DEFAULT_INIT_METHODS,
        random_state=clustering.RANDOM_STATE,
    )
    result = clustering.run_kprototypes(continuous, binary, 2, verbose=False)

    assert np.array_equal(fit.labels, result.labels)
    assert fit.init == result.init
    assert fit.backend == "kprototypes"
