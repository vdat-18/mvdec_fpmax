"""Tests for the study forward selection."""

import numpy as np
import pandas as pd
import pytest

from pipeline import study_ffs
from pipeline.study_arms import ArmFit
from pipeline.study_config import DEFAULT_STUDY_CONFIG_PATH, load_study_config
from pipeline.study_data import StudyData
from pipeline.study_ffs import candidate_silhouette, forward_select_itemsets

STUDY = load_study_config(DEFAULT_STUDY_CONFIG_PATH)
GOOD = np.repeat([0, 1], 20)
OTHER = np.tile([0, 1], 20)


def _data(labels=True) -> StudyData:
    rng = np.random.default_rng(0)
    num = pd.DataFrame({"x": GOOD * 0.8 + rng.normal(0, 0.05, 40), "y": rng.random(40)})
    cat = pd.DataFrame({"c": np.where(GOOD == 0, "u", "v")})
    return StudyData(
        name="toy",
        backbone="raw",
        numeric=num,
        categorical=cat,
        labels=GOOD if labels else None,
        n_clusters=2,
        reference_df=pd.concat([num, cat], axis=1),
        reference_metric="gower",
    )


def _itemsets(names=("a", "b", "c")) -> pd.DataFrame:
    return pd.DataFrame({name: np.tile([0, 1], 20) for name in names})


def _ok(labels) -> ArmFit:
    return ArmFit(labels, "ok", 1.0, None, 0.0, "farthest_first")


def _failed(status="error") -> ArmFit:
    return ArmFit(None, status, float("nan"), "boom", 0.0, None)


def _script(monkeypatch, fits, scores, baseline):
    """fits/scores map the candidate's last column name to an ArmFit / score."""

    def fake_fit(data, algorithm, candidate, seed, study, restarts=None):
        return fits[candidate.columns[-1]]

    def fake_score(data, itemsets, labels, max_rows):
        if itemsets.shape[1] == 0:
            return baseline
        return scores[itemsets.columns[-1]]

    monkeypatch.setattr(study_ffs, "fit_arm", fake_fit)
    monkeypatch.setattr(study_ffs, "candidate_silhouette", fake_score)


def _run(**kwargs):
    return forward_select_itemsets(
        _data(), "intuitive", _itemsets(), 0, STUDY, baseline_labels=OTHER, **kwargs
    )


def test_ffs_stops_when_gain_is_below_threshold(monkeypatch):
    fits = {n: _ok(GOOD) for n in "abc"}
    _script(monkeypatch, fits, {"a": 0.404, "b": 0.402, "c": 0.401}, baseline=0.4)

    result = _run()

    assert result.selected == []


def test_ffs_accepts_best_candidate_then_stops(monkeypatch):
    fits = {n: _ok(GOOD) for n in "abc"}
    _script(monkeypatch, fits, {"a": 0.41, "b": 0.45, "c": 0.43}, baseline=0.4)

    result = _run()

    assert result.selected == ["b"]
    assert result.final_score == pytest.approx(0.45)
    assert np.array_equal(result.labels, GOOD)


def test_ffs_skips_invalid_and_error_candidates(monkeypatch):
    fits = {"a": _failed("invalid_partition"), "b": _failed("error"), "c": _ok(GOOD)}
    _script(monkeypatch, fits, {"a": 0.99, "b": 0.99, "c": 0.5}, baseline=0.4)

    result = _run()

    assert result.selected == ["c"]


def test_ffs_with_no_improvement_returns_baseline_labels(monkeypatch):
    fits = {n: _ok(GOOD) for n in "abc"}
    _script(monkeypatch, fits, {"a": 0.1, "b": 0.2, "c": 0.3}, baseline=0.4)

    result = _run()

    assert result.selected == []
    assert np.array_equal(result.labels, OTHER)
    assert result.final_score == pytest.approx(0.4)


def test_ffs_counts_every_candidate_fit(monkeypatch):
    fits = {n: _ok(GOOD) for n in "abc"}
    _script(monkeypatch, fits, {"a": 0.1, "b": 0.2, "c": 0.3}, baseline=0.4)

    assert _run().n_candidates_evaluated == 3


def test_ffs_rejects_kmeans():
    with pytest.raises(ValueError, match="itemsets"):
        forward_select_itemsets(
            _data(), "kmeans", _itemsets(), 0, STUDY, baseline_labels=OTHER
        )


def test_ffs_runs_without_labels(monkeypatch):
    fits = {n: _ok(GOOD) for n in "abc"}
    _script(monkeypatch, fits, {"a": 0.5, "b": 0.1, "c": 0.1}, baseline=0.4)

    result = forward_select_itemsets(
        _data(labels=False), "intuitive", _itemsets(), 0, STUDY, OTHER
    )

    assert result.selected == ["a"]


def test_candidate_silhouette_nan_for_singleton_partition():
    labels = np.zeros(40, dtype=int)
    labels[0] = 1

    score = candidate_silhouette(_data(), _itemsets().iloc[:, :0], labels, 1500)

    assert np.isnan(score)


def test_candidate_silhouette_rewards_the_true_structure():
    good = candidate_silhouette(_data(), _itemsets().iloc[:, :0], GOOD, 1500)
    bad = candidate_silhouette(_data(), _itemsets().iloc[:, :0], OTHER, 1500)

    assert good > bad


def test_candidate_silhouette_subsample_is_deterministic():
    first = candidate_silhouette(_data(), _itemsets(), GOOD, 25)
    second = candidate_silhouette(_data(), _itemsets(), GOOD, 25)

    assert first == second


def test_ffs_flags_an_invalid_baseline(monkeypatch):
    singleton = np.zeros(40, dtype=int)
    singleton[0] = 1

    result = forward_select_itemsets(
        _data(), "intuitive", _itemsets(), 0, STUDY, baseline_labels=singleton
    )

    assert result.baseline_valid is False
    assert result.selected == []
    assert result.n_candidates_evaluated == 0


def _recording_script(monkeypatch, scores, labels_by_restarts=None):
    """Scripted fit_arm and score that record (columns, restarts) of every fit."""

    calls: list[tuple[tuple[str, ...], int | None]] = []

    def fake_fit(data, algorithm, candidate, seed, study, restarts=None):
        calls.append((tuple(candidate.columns), restarts))
        if labels_by_restarts is not None and restarts is None:
            return _ok(labels_by_restarts)
        return _ok(GOOD)

    def fake_score(data, itemsets, labels, max_rows):
        if itemsets.shape[1] == 0:
            return 0.4
        # every extra selected column adds 0.1 so greedy steps keep improving
        return scores[itemsets.columns[-1]] + 0.1 * (itemsets.shape[1] - 1)

    monkeypatch.setattr(study_ffs, "fit_arm", fake_fit)
    monkeypatch.setattr(study_ffs, "candidate_silhouette", fake_score)
    return calls


def test_ffs_only_considers_the_first_candidates(monkeypatch):
    import dataclasses

    study = dataclasses.replace(STUDY, ffs_max_candidates=2)
    calls = _recording_script(monkeypatch, {"a": 0.1, "b": 0.1, "c": 0.9})

    result = forward_select_itemsets(
        _data(), "kprototypes", _itemsets(), 0, study, baseline_labels=OTHER
    )

    assert result.selected == []
    assert {name for columns, _ in calls for name in columns} == {"a", "b"}
    assert result.n_candidates_evaluated == 2


def test_ffs_stops_after_max_steps(monkeypatch):
    import dataclasses

    study = dataclasses.replace(STUDY, ffs_max_steps=2, ffs_min_improvement=0.0)
    _recording_script(monkeypatch, {"a": 0.51, "b": 0.52, "c": 0.53})

    result = forward_select_itemsets(
        _data(), "kprototypes", _itemsets(), 0, study, baseline_labels=OTHER
    )

    assert len(result.selected) == 2


def test_ffs_candidates_use_reduced_restarts_and_final_refit_uses_full(monkeypatch):
    import dataclasses

    study = dataclasses.replace(STUDY, ffs_restarts=1)
    final = np.tile([1, 0], 20)
    calls = _recording_script(
        monkeypatch, {"a": 0.9, "b": 0.1, "c": 0.1}, labels_by_restarts=final
    )

    result = forward_select_itemsets(
        _data(), "intuitive", _itemsets(), 0, study, baseline_labels=OTHER
    )

    assert result.selected == ["a"]
    candidate_calls = [r for columns, r in calls[:-1]]
    assert set(candidate_calls) == {1}
    assert calls[-1] == (("a",), None)
    assert np.array_equal(result.labels, final)


def test_ffs_refit_failure_keeps_last_labels(monkeypatch):
    import dataclasses

    study = dataclasses.replace(STUDY, ffs_restarts=1)

    def fake_fit(data, algorithm, candidate, seed, study, restarts=None):
        if restarts is None:
            return _failed("invalid_partition")
        return _ok(GOOD)

    monkeypatch.setattr(study_ffs, "fit_arm", fake_fit)
    monkeypatch.setattr(
        study_ffs,
        "candidate_silhouette",
        lambda data, itemsets, labels, max_rows: 0.4 if itemsets.shape[1] == 0 else 0.9,
    )

    result = forward_select_itemsets(
        _data(), "intuitive", _itemsets(), 0, study, baseline_labels=OTHER
    )

    assert result.selected
    assert np.array_equal(result.labels, GOOD)
