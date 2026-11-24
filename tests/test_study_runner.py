"""Tests for the study runner."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from pipeline import study_runner
from pipeline.study_arms import ArmFit
from pipeline.study_config import DEFAULT_STUDY_CONFIG_PATH, load_study_config
from pipeline.study_data import StudyData
from pipeline.study_itemsets import FpmaxSetting, ItemsetBlock
from pipeline.study_runner import (
    STUDY_COLUMNS,
    build_grid_blocks,
    itemset_signature,
    run_group,
)

BASE = load_study_config(DEFAULT_STUDY_CONFIG_PATH)
STUDY = dataclasses.replace(
    BASE,
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


def test_every_ok_setting_is_its_own_arm_on_every_report_seed(tmp_path):
    data = _data()
    blocks = build_grid_blocks(data, STUDY)
    ok_settings = [s for s, b in blocks.items() if b.status == "ok"]
    assert len(ok_settings) >= 1

    frame = _read(run_group(data, "kprototypes", STUDY, tmp_path))

    assert list(frame.columns) == list(STUDY_COLUMNS)
    assert sorted(frame[frame["arm"] == "none"]["seed"]) == [0, 1]
    for setting in ok_settings:
        mask = (frame["strategy"] == setting.strategy) & (
            frame["min_support"] == setting.min_support
        )
        for arm in ("fpmax", "fpmax_ffs"):
            rows = frame[mask & (frame["arm"] == arm)]
            assert sorted(rows["seed"]) == [0, 1]
            assert rows["itemset_signature"].notna().all()
    assert frame["status"].notna().all()


def test_settings_without_itemsets_get_one_seedless_row(tmp_path):
    study = dataclasses.replace(STUDY, fpmax_min_supports=(1.0, 0.6))

    frame = _read(run_group(_data(), "kprototypes", study, tmp_path))
    empty = frame[(frame["arm"] == "fpmax") & (frame["status"] == "no_itemsets")]

    assert len(empty) >= 1
    assert empty["seed"].isna().all()
    assert 1.0 in set(empty["min_support"])


def test_kmeans_group_has_only_the_none_arm(tmp_path):
    frame = _read(run_group(_data(), "kmeans", STUDY, tmp_path))

    assert set(frame["arm"]) == {"none"}


def test_labels_archive_contains_valid_partitions_only(tmp_path):
    path = run_group(_data(), "kprototypes", STUDY, tmp_path)

    archive = np.load(path.with_suffix(".labels.npz"))

    assert "none__0" in archive.files
    assert any(name.startswith("fpmax__uniform-3-") for name in archive.files)
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
    def fake(data, algorithm, itemsets, seed, study, restarts=None):
        key = (None if itemsets is None else itemsets.shape[1], seed)
        labels = table.get(key, GOOD)
        if labels is None:
            return ArmFit(None, "error", float("nan"), "boom", 0.0, None)
        return ArmFit(labels, "ok", 1.0, None, 0.0, "x")

    return fake


def test_failed_fit_is_a_row_with_status_and_error(tmp_path, monkeypatch):
    monkeypatch.setattr(study_runner, "fit_arm", _scripted_fit({(None, 0): None}))

    frame = _read(run_group(_data(), "kprototypes", STUDY, tmp_path))
    failed = frame[(frame["arm"] == "none") & (frame["seed"] == 0)]

    assert len(failed) == 1
    assert failed.iloc[0]["status"] == "error"
    assert failed.iloc[0]["error"] == "boom"
    ffs = frame[(frame["arm"] == "fpmax_ffs") & (frame["seed"] == 0)]
    assert set(ffs["status"]) == {"baseline_unavailable"}


def test_systemic_failure_raises_but_keeps_the_rows_written(tmp_path, monkeypatch):
    def always_fail(data, algorithm, itemsets, seed, study, restarts=None):
        return ArmFit(None, "error", float("nan"), "bad config", 0.0, None)

    monkeypatch.setattr(study_runner, "fit_arm", always_fail)

    with pytest.raises(RuntimeError, match="bad config"):
        run_group(_data(), "kprototypes", STUDY, tmp_path)

    frame = pd.read_csv(tmp_path / "toy__kprototypes.csv")
    assert (frame["status"] != "ok").all()
    assert len(frame) > 0


def test_ffs_with_invalid_baseline_is_flagged(tmp_path, monkeypatch):
    singleton = np.zeros(40, dtype=int)
    singleton[0] = 1

    def fake(data, algorithm, itemsets, seed, study, restarts=None):
        labels = singleton if itemsets is None else GOOD
        return ArmFit(labels, "ok", 1.0, None, 0.0, "x")

    monkeypatch.setattr(study_runner, "fit_arm", fake)
    monkeypatch.setattr("pipeline.study_ffs.fit_arm", fake)

    frame = pd.read_csv(run_group(_data(), "kprototypes", STUDY, tmp_path))
    ffs = frame[frame["arm"] == "fpmax_ffs"]

    assert set(ffs["status"]) == {"ffs_baseline_invalid"}


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


def test_categorical_only_data_uses_a_single_strategy_and_bin_count():
    study = dataclasses.replace(
        STUDY,
        fpmax_strategies=("uniform", "quantile"),
        fpmax_n_bins=(3, 5),
        fpmax_min_supports=(0.6, 0.4),
    )
    data = dataclasses.replace(_data(), numeric=pd.DataFrame(index=range(40)))

    blocks = build_grid_blocks(data, study)

    assert len(blocks) == 2
    assert {(s.strategy, s.n_bins) for s in blocks} == {("uniform", 3)}


def test_itemset_signature_identifies_identical_blocks():
    a = ItemsetBlock(pd.DataFrame({"fp|x": [0, 1], "fp|y": [1, 0]}), "ok", 2)
    same = ItemsetBlock(pd.DataFrame({"fp|y": [1, 0], "fp|x": [0, 1]}), "ok", 2)
    other = ItemsetBlock(pd.DataFrame({"fp|x": [0, 1]}), "ok", 1)

    assert itemset_signature(a) == itemset_signature(same)
    assert itemset_signature(a) != itemset_signature(other)


def test_unknown_algorithm_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="algorithm"):
        run_group(_data(), "dbscan", STUDY, tmp_path)


def test_stem_suffix_changes_file_names_and_shards_do_not_collide(tmp_path):
    first = run_group(_data(), "kmeans", STUDY, tmp_path, stem_suffix="__seeds-0")
    second = run_group(_data(), "kmeans", STUDY, tmp_path, stem_suffix="__seeds-1")

    assert first.name == "toy__kmeans__seeds-0.csv"
    assert second.name == "toy__kmeans__seeds-1.csv"
    assert first.exists() and second.exists()
    assert first.with_suffix(".labels.npz").exists()


def test_rerunning_a_shard_overwrites_its_files(tmp_path):
    run_group(_data(), "kmeans", STUDY, tmp_path, stem_suffix="__seeds-0")
    path = run_group(_data(), "kmeans", STUDY, tmp_path, stem_suffix="__seeds-0")

    assert len(pd.read_csv(path)) == len(STUDY.report_seeds)
