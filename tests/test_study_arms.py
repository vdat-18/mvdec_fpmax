"""Tests for arm fitting."""

import numpy as np
import pandas as pd
import pytest

from pipeline import study_arms
from pipeline.study_arms import fit_arm
from pipeline.study_config import DEFAULT_STUDY_CONFIG_PATH, load_study_config
from pipeline.study_data import StudyData

STUDY = load_study_config(DEFAULT_STUDY_CONFIG_PATH)


def _data(*, numeric=True, categorical=True, labels=True, name="iris") -> StudyData:
    rng = np.random.default_rng(0)
    n = 40
    group = np.repeat([0, 1], n // 2)
    num = pd.DataFrame(
        {
            "x": group * 0.8 + rng.normal(0, 0.05, n),
            "y": group * 0.8 + rng.normal(0, 0.05, n),
        }
    )
    flip = rng.random(n) < 0.05
    cat = pd.DataFrame({"c": np.where(group ^ flip, "u", "v")})
    return StudyData(
        name=name,
        backbone="raw",
        numeric=num if numeric else pd.DataFrame(index=range(n)),
        categorical=cat if categorical else pd.DataFrame(index=range(n)),
        labels=group if labels else None,
        n_clusters=2,
        reference_df=pd.concat([num, cat], axis=1),
        reference_metric="gower",
    )


def _itemsets(n=40) -> pd.DataFrame:
    return pd.DataFrame(
        {"fp|a": np.repeat([0, 1], n // 2), "fp|b": np.tile([0, 1], n // 2)}
    )


def test_kmeans_not_applicable_without_numeric_columns():
    fit = fit_arm(_data(numeric=False), "kmeans", None, 0, STUDY)

    assert fit.status == "not_applicable"
    assert fit.labels is None


def test_kmeans_ignores_categorical_and_itemsets():
    fit = fit_arm(_data(), "kmeans", _itemsets(), 0, STUDY)

    assert fit.status == "ok"
    assert fit.init == "k-means++"


def test_kprototypes_categorical_only_runs_via_kmodes():
    fit = fit_arm(_data(numeric=False), "kprototypes", None, 0, STUDY)

    assert fit.status == "ok"
    assert len(np.unique(fit.labels)) == 2


def test_kprototypes_mixed_with_itemsets_runs():
    fit = fit_arm(_data(), "kprototypes", _itemsets(), 0, STUDY)

    assert fit.status == "ok"


class _RecordingModel:
    """Intuitive stand-in that records its inputs and returns scripted labels."""

    calls: list[dict] = []
    scripts: list[tuple[np.ndarray, float]] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        _RecordingModel.calls.append(kwargs)

    def fit(self, X_num=None, X_cat=None):
        _RecordingModel.calls[-1]["X_num"] = X_num
        _RecordingModel.calls[-1]["X_cat"] = X_cat
        index = len(_RecordingModel.calls) - 1
        labels, cost = _RecordingModel.scripts[index % len(_RecordingModel.scripts)]
        distances = np.full((len(labels), 2), cost / len(labels))
        self.result_ = type("R", (), {"labels": labels, "distances": distances})()
        return self


@pytest.fixture
def recording(monkeypatch):
    _RecordingModel.calls = []
    good = np.repeat([0, 1], 20)
    _RecordingModel.scripts = [(good, 5.0)]
    monkeypatch.setattr(study_arms, "IntuitiveKPrototypes", _RecordingModel)
    return _RecordingModel


def test_intuitive_uses_study_params_and_swapped_mu(recording):
    fit_arm(_data(name="bcw"), "intuitive", None, 3, STUDY)

    first = recording.calls[0]
    assert first["mu_convention"] == "swapped"
    assert first["init_strategy"] == "farthest_first"
    assert first["beta"] == 8.0
    assert first["mu_param"] == 0.2
    assert first["gamma"] == 0.4
    assert [c["random_state"] for c in recording.calls] == [3000 + r for r in range(5)]


def test_intuitive_keeps_lowest_cost_valid_restart(recording):
    good_cheap = np.repeat([0, 1], 20)
    other = np.tile([0, 1], 20)
    singleton = np.zeros(40, dtype=int)
    singleton[0] = 1
    recording.scripts = [
        (singleton, 1.0),
        (other, 9.0),
        (good_cheap, 2.0),
        (other, 7.0),
        (other, 8.0),
    ]

    fit = fit_arm(_data(), "intuitive", None, 0, STUDY)

    assert fit.status == "ok"
    assert fit.cost == pytest.approx(2.0)
    assert np.array_equal(fit.labels, good_cheap)


def test_itemsets_change_the_intuitive_categorical_block(recording):
    data = _data()

    fit_arm(data, "intuitive", _itemsets(), 0, STUDY)

    assert recording.calls[0]["X_cat"].shape == (40, data.categorical.shape[1] + 2)
    assert recording.calls[0]["X_num"].shape == (40, 2)


def test_intuitive_all_invalid_restarts_report_invalid_partition(recording):
    singleton = np.zeros(40, dtype=int)
    singleton[0] = 1
    recording.scripts = [(singleton, 1.0)]

    fit = fit_arm(_data(), "intuitive", None, 0, STUDY)

    assert fit.status == "invalid_partition"
    assert fit.labels is None


def test_fit_error_is_recorded_not_raised(monkeypatch):
    class Broken:
        def __init__(self, **_kwargs):
            pass

        def fit(self, **_kwargs):
            msg = "bad data"
            raise ValueError(msg)

    monkeypatch.setattr(study_arms, "IntuitiveKPrototypes", Broken)

    fit = fit_arm(_data(), "intuitive", None, 0, STUDY)

    assert fit.status == "error"
    assert "bad data" in fit.error


def test_unexpected_exception_propagates(monkeypatch):
    class Buggy:
        def __init__(self, **_kwargs):
            pass

        def fit(self, **_kwargs):
            raise TypeError("programming error")

    monkeypatch.setattr(study_arms, "IntuitiveKPrototypes", Buggy)

    with pytest.raises(TypeError):
        fit_arm(_data(), "intuitive", None, 0, STUDY)


def test_fit_arm_works_without_labels():
    fit = fit_arm(_data(labels=False), "kprototypes", None, 0, STUDY)

    assert fit.status == "ok"


def test_unknown_algorithm_is_rejected():
    with pytest.raises(ValueError, match="algorithm"):
        fit_arm(_data(), "dbscan", None, 0, STUDY)


def test_real_intuitive_fit_separates_clear_groups():
    fit = fit_arm(_data(), "intuitive", _itemsets(), 0, STUDY)

    assert fit.status == "ok"
    accuracy = max(
        np.mean(fit.labels == np.repeat([0, 1], 20)),
        np.mean(fit.labels != np.repeat([0, 1], 20)),
    )
    assert accuracy > 0.9


def test_mixed_error_and_invalid_restarts_report_invalid_partition(monkeypatch):
    singleton = np.zeros(40, dtype=int)
    singleton[0] = 1
    calls = {"n": 0}

    class Mixed:
        def __init__(self, **_kwargs):
            pass

        def fit(self, **_kwargs):
            calls["n"] += 1
            if calls["n"] % 2:
                msg = "empty cluster"
                raise ValueError(msg)
            distances = np.ones((40, 2))
            self.result_ = type(
                "R", (), {"labels": singleton, "distances": distances}
            )()
            return self

    monkeypatch.setattr(study_arms, "IntuitiveKPrototypes", Mixed)

    fit = fit_arm(_data(), "intuitive", None, 0, STUDY)

    assert fit.status == "invalid_partition"
