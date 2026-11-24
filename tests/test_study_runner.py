"""Tests for the study runner."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from pipeline import study_runner
from pipeline.reference_evaluation import build_reference_distance
from pipeline.study_arms import ArmFit
from pipeline.study_config import DEFAULT_STUDY_CONFIG_PATH, load_study_config
from pipeline.study_data import StudyData
from pipeline.study_itemsets import FpmaxSetting, ItemsetBlock
from pipeline.study_runner import (
    STUDY_COLUMNS,
    build_grid_blocks,
    run_group,
    select_fpmax_setting,
)

BASE = load_study_config(DEFAULT_STUDY_CONFIG_PATH)
STUDY = dataclasses.replace(
    BASE,
    selection_seeds=(100, 101),
    report_seeds=(0, 1),
    fpmax_strategies=("uniform",),
    fpmax_n_bins=(3,),
    fpmax_min_supports=(0.6, 0.4),
)
GOOD = np.repeat([0, 1], 20)


def _data(*, labels=True, categorical=True) -> StudyData:
    rng = np.random.default_rng(0)
    num = pd.DataFrame(
        {
            "x": GOOD * 0.8 + rng.normal(0, 0.05, 40),
            "y": GOOD * 0.8 + rng.normal(0, 0.05, 40),
        }
    )
    cat = (
        pd.DataFrame({"c": np.where(GOOD == 0, "u", "v")})
        if categorical
        else pd.DataFrame(index=range(40))
    )
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


def _read(path) -> pd.DataFrame:
    return pd.read_csv(path)


def test_group_writes_all_arms_for_every_report_seed(tmp_path):
    path = run_group(_data(), "kprototypes", STUDY, tmp_path)

    frame = _read(path)
    report = frame[frame["phase"] == "report"]

    assert list(frame.columns) == list(STUDY_COLUMNS)
    assert set(report["arm"]) == {"none", "fpmax", "fpmax_ffs"}
    for arm in ("none", "fpmax", "fpmax_ffs"):
        assert sorted(report[report["arm"] == arm]["seed"]) == [0, 1]
    assert not set(report["seed"]) & set(STUDY.selection_seeds)
    assert set(frame[frame["phase"] == "select"]["seed"].dropna()) <= set(
        STUDY.selection_seeds
    )
    assert report["status"].notna().all()


def test_kmeans_group_has_only_the_none_arm(tmp_path):
    frame = _read(run_group(_data(), "kmeans", STUDY, tmp_path))

    assert set(frame["arm"]) == {"none"}
    assert set(frame["phase"]) == {"report"}


def test_labels_archive_contains_valid_partitions_only(tmp_path):
    path = run_group(_data(), "kprototypes", STUDY, tmp_path)

    archive = np.load(path.with_suffix(".labels.npz"))

    assert "none__0" in archive.files
    assert all(len(np.unique(archive[name])) == 2 for name in archive.files)


def test_unlabeled_data_has_nan_external_metrics_but_reference_scores(tmp_path):
    frame = _read(run_group(_data(labels=False), "kprototypes", STUDY, tmp_path))
    ok = frame[(frame["status"] == "ok") & (frame["arm"] == "none")]

    assert ok["acc"].isna().all()
    assert ok["ari"].isna().all()
    assert ok["ref_silhouette"].notna().all()


def test_labelled_data_reports_external_metrics(tmp_path):
    frame = _read(run_group(_data(), "kprototypes", STUDY, tmp_path))
    ok = frame[(frame["status"] == "ok") & (frame["arm"] == "none")]

    assert (ok["acc"] > 0.9).all()
    assert ok["ari"].notna().all()


def _scripted_fit(table):
    def fake(data, algorithm, itemsets, seed, study):
        key = (None if itemsets is None else itemsets.shape[1], seed)
        labels = table.get(key, GOOD)
        if labels is None:
            return ArmFit(None, "error", float("nan"), "boom", 0.0, None)
        return ArmFit(labels, "ok", 1.0, None, 0.0, "x")

    return fake


def test_setting_choice_ignores_report_seed_results(monkeypatch):
    data = _data()
    reference = build_reference_distance(data.reference_df)
    blocks = build_grid_blocks(data, STUDY)
    good_settings = [s for s, b in blocks.items() if b.status == "ok"]
    assert good_settings

    baseline = select_fpmax_setting(data, "kprototypes", STUDY, reference, blocks)
    # sabotage every report-seed fit; selection must not change
    sabotaged = {(n, seed): None for n in range(0, 80) for seed in STUDY.report_seeds}
    monkeypatch.setattr(study_runner, "fit_arm", _scripted_fit(sabotaged))
    again = select_fpmax_setting(data, "kprototypes", STUDY, reference, blocks)

    assert again.setting == baseline.setting


def test_failed_fit_is_a_row_with_status_and_error(tmp_path, monkeypatch):
    table = {(None, 0): None}
    monkeypatch.setattr(study_runner, "fit_arm", _scripted_fit(table))

    frame = _read(run_group(_data(), "kprototypes", STUDY, tmp_path))
    failed = frame[(frame["arm"] == "none") & (frame["seed"] == 0)]

    assert len(failed) == 1
    assert failed.iloc[0]["status"] == "error"
    assert failed.iloc[0]["error"] == "boom"
    ffs = frame[(frame["arm"] == "fpmax_ffs") & (frame["seed"] == 0)]
    assert ffs.iloc[0]["status"] == "baseline_unavailable"


def test_no_valid_setting_marks_fpmax_arms(tmp_path, monkeypatch):
    def fake(data, algorithm, itemsets, seed, study):
        if itemsets is not None and seed in study.selection_seeds:
            return ArmFit(None, "invalid_partition", float("nan"), None, 0.0, None)
        return ArmFit(GOOD, "ok", 1.0, None, 0.0, "x")

    monkeypatch.setattr(study_runner, "fit_arm", fake)

    frame = _read(run_group(_data(), "kprototypes", STUDY, tmp_path))
    report = frame[(frame["phase"] == "report") & (frame["arm"] != "none")]

    assert set(report["status"]) == {"no_valid_fpmax_setting"}
    assert len(report) == 4


def test_setting_selection_runs_without_labels():
    data = _data(labels=False)
    reference = build_reference_distance(data.reference_df)
    blocks = build_grid_blocks(data, STUDY)

    choice = select_fpmax_setting(data, "kprototypes", STUDY, reference, blocks)

    assert choice.setting is not None


def test_skip_rule_applies_in_grid_order(monkeypatch):
    calls: list[float] = []

    def fake_build(data, setting, max_itemsets):
        calls.append(setting.min_support)
        status = "too_many_itemsets" if setting.min_support <= 0.6 else "ok"
        return ItemsetBlock(pd.DataFrame(index=range(40)), status, 99)

    monkeypatch.setattr(study_runner, "build_itemset_block", fake_build)
    study = dataclasses.replace(STUDY, fpmax_min_supports=(0.8, 0.6, 0.4, 0.2))

    blocks = build_grid_blocks(_data(), study)

    assert calls == [0.8, 0.6]
    assert blocks[FpmaxSetting("uniform", 3, 0.4)].status == (
        "skipped_after_too_many_itemsets"
    )
    assert blocks[FpmaxSetting("uniform", 3, 0.2)].status == (
        "skipped_after_too_many_itemsets"
    )


def test_unknown_algorithm_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="algorithm"):
        run_group(_data(), "dbscan", STUDY, tmp_path)
