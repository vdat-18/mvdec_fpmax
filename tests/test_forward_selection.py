"""Tests for forward feature selection over FP-Max binary features."""

import numpy as np
import pandas as pd

from pipeline import clustering, forward_selection

GOOD = np.array([0, 0, 0, 0, 1, 1, 1, 1])
SINGLETON = np.array([0, 0, 0, 0, 0, 0, 0, 1])


def _data() -> tuple[pd.DataFrame, pd.DataFrame]:
    continuous = pd.DataFrame({"x": [0.0, 0.1, 0.2, 0.3, 4.8, 4.9, 5.0, 5.1]})
    binary = pd.DataFrame(
        {
            "a": [1, 1, 1, 1, 0, 0, 0, 0],
            "b": [1, 1, 1, 0, 0, 0, 0, 0],
            "c": [1, 0, 1, 0, 1, 0, 1, 0],
        }
    )
    return continuous, binary


def _patch_fit(monkeypatch, labels_by_feature: dict[str, np.ndarray]) -> None:
    """Make the K-Prototypes fit return scripted labels keyed by the last feature."""

    class Fake:
        cost_ = 1.0

        def __init__(self, **_kwargs) -> None:
            pass

        def fit_predict(self, matrix, categorical):
            # the candidate's newest binary column is the last matrix column
            column = np.asarray(matrix)[:, -1].astype(int)
            for name, labels in labels_by_feature.items():
                if np.array_equal(column, _data()[1][name].to_numpy()):
                    return labels
            return GOOD

    monkeypatch.setattr(clustering, "KPrototypes", Fake)


def _run(**kwargs):
    continuous, binary = _data()
    return forward_selection.run_forward_selection(
        continuous,
        binary,
        n_clusters=2,
        init_methods=("cao",),
        verbose=False,
        **kwargs,
    )


def test_ffs_stops_below_min_improvement(monkeypatch) -> None:
    monkeypatch.setattr(
        forward_selection,
        "run_kprototypes",
        lambda *a, **k: _scripted_result(0.401),
    )

    result = _run(baseline_score=0.4, min_improvement=0.005)

    assert result.selected_feature_names == []


def _scripted_result(score: float) -> clustering.KPrototypesResult:
    return clustering.KPrototypesResult(
        matrix=np.zeros((8, 1)),
        binary_feature_names=[],
        labels=GOOD,
        score=score,
        init="cao",
        fit_time_seconds=0.0,
        cluster_sizes=[4, 4],
        labels_hash="x",
        silhouette_sample_std=0.0,
        silhouette_negative_fraction=0.0,
    )


def test_ffs_skips_candidate_with_singleton_partition(monkeypatch) -> None:
    _patch_fit(monkeypatch, {"a": SINGLETON, "b": GOOD, "c": SINGLETON})

    result = _run(baseline_score=-1.0, min_improvement=0.0)

    assert result.selected_feature_names[0] == "b"


def test_ffs_all_invalid_candidates_do_not_crash(monkeypatch) -> None:
    _patch_fit(
        monkeypatch,
        {"a": SINGLETON, "b": SINGLETON, "c": SINGLETON},
    )

    result = _run(baseline_score=-1.0, min_improvement=0.0)

    assert result.selected_feature_names == []


def test_ffs_counts_candidates(monkeypatch) -> None:
    _patch_fit(
        monkeypatch,
        {"a": SINGLETON, "b": SINGLETON, "c": SINGLETON},
    )

    result = _run(baseline_score=-1.0, min_improvement=0.0)

    assert result.n_candidates_evaluated == 3


def test_study_min_improvement_is_a_real_threshold() -> None:
    assert forward_selection.STUDY_FFS_MIN_IMPROVEMENT == 0.005
    assert not forward_selection.improves_score(
        0.504, 0.5, forward_selection.STUDY_FFS_MIN_IMPROVEMENT
    )
