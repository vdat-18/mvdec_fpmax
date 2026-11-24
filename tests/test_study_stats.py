"""Tests for the study statistics."""

import numpy as np
import pandas as pd
import pytest

from pipeline.reference_evaluation import pairwise_ari_stability
from pipeline.study_runner import STUDY_COLUMNS
from pipeline.study_stats import (
    dataset_level,
    holm_adjust,
    load_results,
    paired_deltas,
    primary_table,
    rank_biserial,
    setting_table,
    stability_table,
    status_table,
    wilcoxon_p,
)


def _row(dataset, algorithm, arm, seed, ari, *, signature="", support=0.5, status="ok"):
    row = dict.fromkeys(STUDY_COLUMNS, np.nan)
    row.update(
        dataset=dataset,
        backbone="raw",
        algorithm=algorithm,
        arm=arm,
        seed=seed,
        status=status,
        error="",
        ari=ari,
        nmi=ari,
        itemset_signature=signature,
    )
    if arm != "none":
        row.update(strategy="uniform", n_bins=3, min_support=support)
    return row


def _frame(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(STUDY_COLUMNS))


def test_holm_adjust_matches_the_step_down_rule():
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert np.isnan(holm_adjust([0.01, float("nan")])[1])


def test_rank_biserial_extremes():
    assert rank_biserial(np.array([0.1, 0.2, 0.3])) == 1.0
    assert rank_biserial(np.array([-0.1, -0.2])) == -1.0
    assert rank_biserial(np.array([0.0, 0.0])) == 0.0


def test_wilcoxon_p_is_one_when_all_deltas_are_zero():
    assert wilcoxon_p(np.zeros(5)) == 1.0
    assert wilcoxon_p(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])) == pytest.approx(2 / 64)


def test_load_results_drops_seedless_rows_repeated_across_shards(tmp_path):
    seedless = _row("iris", "intuitive", "fpmax", np.nan, np.nan, status="no_itemsets")
    for shard in (0, 1):
        rows = [_row("iris", "intuitive", "none", shard, 0.5), seedless]
        _frame(rows).to_csv(
            tmp_path / f"iris__intuitive__seeds-{shard}.csv", index=False
        )

    results = load_results(tmp_path)

    assert (results["status"] == "no_itemsets").sum() == 1
    assert sorted(results[results["arm"] == "none"]["seed"]) == [0, 1]


def test_paired_deltas_pairs_by_seed_and_collapses_identical_signatures():
    rows = []
    for seed, base in [(0, 0.5), (1, 0.6)]:
        rows.append(_row("d", "intuitive", "none", seed, base))
        rows.append(_row("d", "intuitive", "fpmax", seed, base + 0.1, signature="A"))
        rows.append(
            _row(
                "d", "intuitive", "fpmax", seed, base + 0.1, signature="A", support=0.4
            )
        )
    rows.append(_row("d", "intuitive", "fpmax", 0, 0.9, signature="B", status="error"))

    deltas = paired_deltas(_frame(rows), "ari")

    assert len(deltas) == 2
    assert set(deltas["itemset_signature"]) == {"A"}
    assert deltas["delta"].tolist() == pytest.approx([0.1, 0.1])


def test_dataset_level_takes_the_median_over_settings():
    rows = []
    for seed in (0, 1):
        rows.append(_row("d", "intuitive", "none", seed, 0.5))
        for signature, gain in [("A", 0.3), ("B", 0.0), ("C", -0.1)]:
            rows.append(
                _row("d", "intuitive", "fpmax", seed, 0.5 + gain, signature=signature)
            )

    level = dataset_level(paired_deltas(_frame(rows), "ari"))

    assert level.iloc[0]["delta"] == pytest.approx(0.0)
    assert level.iloc[0]["n_settings"] == 3


def test_primary_table_reports_wins_p_and_holm():
    rows = []
    for dataset in "abcdef":
        for seed in (0, 1):
            rows.append(_row(dataset, "intuitive", "none", seed, 0.5))
            rows.append(
                _row(
                    dataset,
                    "intuitive",
                    "fpmax",
                    seed,
                    0.5 + 0.05 * (ord(dataset) - 96),
                    signature="S",
                )
            )

    table = primary_table(_frame(rows), ("ari", "nmi"))

    ari = table[table["metric"] == "ari"].iloc[0]
    assert ari["wins"] == 6 and ari["losses"] == 0
    assert ari["p_value"] == pytest.approx(2 / 64)
    assert ari["effect_rank_biserial"] == 1.0
    assert (table["p_holm"] >= table["p_value"]).all()
    assert len(table) == 2


def test_setting_table_summarises_the_distribution_over_settings():
    rows = []
    for seed in (0, 1):
        rows.append(_row("d", "intuitive", "none", seed, 0.5))
        for signature, gain in [("A", 0.3), ("B", -0.1), ("C", 0.1)]:
            rows.append(
                _row("d", "intuitive", "fpmax", seed, 0.5 + gain, signature=signature)
            )

    table = setting_table(_frame(rows), "ari")

    row = table.iloc[0]
    assert row["n_settings"] == 3
    assert row["share_positive"] == pytest.approx(2 / 3)
    assert row["max_delta"] == pytest.approx(0.3)
    assert row["min_delta"] == pytest.approx(-0.1)


def test_stability_table_uses_saved_labels(tmp_path):
    a = np.array([0, 0, 1, 1])
    relabelled = np.array([1, 1, 0, 0])
    different = np.array([0, 0, 1, 0])
    rows = []
    for seed, labels in [(0, a), (1, relabelled), (2, different)]:
        rows.append(_row("d", "intuitive", "none", seed, 0.5))
        np.savez(
            tmp_path / f"d__intuitive__seeds-{seed}.labels.npz",
            **{f"none__{seed}": labels},
        )

    table = stability_table(tmp_path, _frame(rows))

    none = table[table["arm"] == "none"].iloc[0]
    expected = pairwise_ari_stability([a, relabelled, different])
    assert none["stability_ari"] == pytest.approx(expected)
    assert 0.0 < expected < 1.0


def test_status_table_keeps_failures_visible():
    rows = [
        _row("d", "intuitive", "none", 0, 0.5),
        _row("d", "intuitive", "fpmax", 0, np.nan, status="error"),
    ]

    table = status_table(_frame(rows))

    assert set(table["status"]) == {"ok", "error"}
