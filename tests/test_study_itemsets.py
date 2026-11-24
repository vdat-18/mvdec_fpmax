"""Tests for the FP-Max itemset block."""

import numpy as np
import pandas as pd

from pipeline.study_data import StudyData
from pipeline.study_itemsets import (
    FpmaxSetting,
    build_itemset_block,
)


def _data(*, numeric=True, categorical=True, labels=True) -> StudyData:
    rng = np.random.default_rng(0)
    n = 40
    group = np.repeat([0, 1], n // 2)
    num = pd.DataFrame(
        {
            "x": group * 0.8 + rng.normal(0, 0.05, n),
            "y": group * 0.8 + rng.normal(0, 0.05, n),
        }
    )
    cat = pd.DataFrame({"c": np.where(group == 0, "u", "v")})
    return StudyData(
        name="toy",
        backbone="raw",
        numeric=num if numeric else pd.DataFrame(index=range(n)),
        categorical=cat if categorical else pd.DataFrame(index=range(n)),
        labels=group if labels else None,
        n_clusters=2,
        reference_df=pd.concat([num, cat], axis=1),
        reference_metric="gower",
    )


SETTING = FpmaxSetting(strategy="uniform", n_bins=3, min_support=0.3)


def test_block_has_only_binary_itemset_columns_with_prefix():
    block = build_itemset_block(_data(), SETTING, max_itemsets=200)

    assert block.status == "ok"
    assert block.n_itemsets == block.columns.shape[1] > 0
    assert all(name.startswith("fp|") for name in block.columns)
    assert set(np.unique(block.columns.to_numpy())) <= {0, 1}
    assert len(block.columns) == 40


def test_constant_itemset_columns_are_dropped():
    block = build_itemset_block(_data(), SETTING, max_itemsets=200)

    assert (block.columns.nunique() > 1).all()


def test_too_many_itemsets_is_reported_not_raised():
    block = build_itemset_block(_data(), SETTING, max_itemsets=1)

    assert block.status == "too_many_itemsets"
    assert block.columns.shape[1] == 0
    assert block.n_itemsets > 1


def test_no_itemsets_status_at_unreachable_support():
    block = build_itemset_block(
        _data(), FpmaxSetting("uniform", 5, 1.0), max_itemsets=200
    )

    assert block.status == "no_itemsets"
    assert block.columns.shape == (40, 0)


def test_block_is_deterministic():
    first = build_itemset_block(_data(), SETTING, max_itemsets=200).columns
    second = build_itemset_block(_data(), SETTING, max_itemsets=200).columns

    pd.testing.assert_frame_equal(first, second)


def test_categorical_only_data_is_supported():
    block = build_itemset_block(_data(numeric=False), SETTING, max_itemsets=200)

    assert block.status in {"ok", "no_itemsets"}


def test_block_ignores_labels():
    with_labels = build_itemset_block(_data(labels=True), SETTING, 200).columns
    without_labels = build_itemset_block(_data(labels=False), SETTING, 200).columns

    pd.testing.assert_frame_equal(with_labels, without_labels)
