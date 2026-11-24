"""FP-Max itemset features for study data."""

from dataclasses import dataclass

import pandas as pd

from pipeline.fpmax import (
    extract_fpmax_features_from_prepared,
    prepare_fpmax_data,
)
from pipeline.study_data import StudyData

ITEMSET_PREFIX = "fp|"
STATUS_OK = "ok"
STATUS_NO_ITEMSETS = "no_itemsets"
STATUS_TOO_MANY = "too_many_itemsets"


@dataclass(frozen=True)
class FpmaxSetting:
    """One FP-Max configuration from the study grid."""

    strategy: str
    n_bins: int
    min_support: float


@dataclass(frozen=True)
class ItemsetBlock:
    """Binary itemset indicator columns (zero columns unless status is ok)."""

    columns: pd.DataFrame
    status: str
    n_itemsets: int


def build_itemset_block(
    data: StudyData,
    setting: FpmaxSetting,
    max_itemsets: int,
) -> ItemsetBlock:
    """Mine maximal frequent itemsets over the numeric and categorical attributes.

    Numeric columns are binned, categorical columns are one-hot encoded, and each
    maximal itemset becomes a 0/1 column. Constant columns (an itemset every row
    satisfies) carry no information and are dropped. A mining result over
    ``max_itemsets`` or with no itemsets returns an empty block with that status so
    the caller can record it. True labels are never used.
    """

    attributes = pd.concat([data.numeric, data.categorical], axis=1)
    prepared = prepare_fpmax_data(
        df=attributes,
        numeric_cols=list(data.numeric.columns),
        n_bins=setting.n_bins,
        strategy=setting.strategy,
    )
    mined = extract_fpmax_features_from_prepared(
        prepared=prepared,
        min_support=setting.min_support,
        drop_original_numeric=True,
    )
    itemset_columns = [
        column for column in mined.features.columns if column not in attributes.columns
    ]
    block = mined.features[itemset_columns].astype(int)
    block = block.loc[:, block.nunique() > 1]
    block.columns = [f"{ITEMSET_PREFIX}{name}" for name in block.columns]
    block = block.reset_index(drop=True)

    empty = pd.DataFrame(index=block.index)
    if block.shape[1] == 0:
        return ItemsetBlock(empty, STATUS_NO_ITEMSETS, 0)
    if block.shape[1] > max_itemsets:
        return ItemsetBlock(empty, STATUS_TOO_MANY, int(block.shape[1]))
    return ItemsetBlock(block, STATUS_OK, int(block.shape[1]))
