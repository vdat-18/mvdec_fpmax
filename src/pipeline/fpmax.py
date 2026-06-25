"""Derive binary features from maximal frequent itemsets."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import fpmax
from sklearn.preprocessing import KBinsDiscretizer

DiscretizeStrategy = Literal["uniform", "quantile", "kmeans"]

BIN_LABELS_BY_SIZE = {
    3: ["low", "medium", "high"],
    5: ["verylow", "low", "medium", "high", "veryhigh"],
    7: ["megalow", "verylow", "low", "medium", "high", "veryhigh", "megahigh"],
}
DEFAULT_STRATEGIES: tuple[DiscretizeStrategy, ...] = ("uniform", "quantile", "kmeans")
DEFAULT_SUPPORTS = np.round(np.arange(0.10, 1.01, 0.05), 2)[::-1]


@dataclass(frozen=True)
class FpmaxFeatures:
    """Binary features extracted from maximal frequent itemsets."""

    features: pd.DataFrame
    itemsets: pd.DataFrame


@dataclass(frozen=True)
class FpmaxPreparedData:
    """Discretized and one-hot encoded data reused across support values."""

    discrete_df: pd.DataFrame
    onehot: pd.DataFrame
    numeric_cols: list[str]


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric column names in dataframe order."""

    return [
        column for column in df.columns if np.issubdtype(df[column].dtype, np.number)
    ]


def discretize_features(
    df: pd.DataFrame,
    columns: list[str],
    n_bins: int,
    strategy: DiscretizeStrategy,
    bin_labels: list[str] | None = None,
) -> pd.DataFrame:
    """Return a copy where selected numeric columns are replaced by bin labels."""

    if bin_labels is None:
        bin_labels = BIN_LABELS_BY_SIZE[n_bins]

    if len(bin_labels) != n_bins:
        msg = "bin_labels must have the same length as n_bins."
        raise ValueError(msg)

    df_discrete = df.copy()
    if not columns:
        return df_discrete

    discretizer = KBinsDiscretizer(
        n_bins=n_bins,
        encode="ordinal",
        strategy=strategy,
    )
    binned = discretizer.fit_transform(df_discrete[columns]).astype(int)
    df_bins = pd.DataFrame(binned, columns=columns, index=df_discrete.index)

    for column in columns:
        df_discrete[column] = df_bins[column].map(lambda value: bin_labels[value])

    return df_discrete


def extract_fpmax_features(
    df: pd.DataFrame,
    numeric_cols: list[str] | None = None,
    n_bins: int = 7,
    bin_labels: list[str] | None = None,
    strategy: DiscretizeStrategy = "kmeans",
    min_support: float = 0.2,
    drop_original_numeric: bool = True,
) -> FpmaxFeatures:
    """Discretize numeric columns and convert FP-Max itemsets into binary features."""

    prepared = prepare_fpmax_data(
        df=df,
        numeric_cols=numeric_cols,
        n_bins=n_bins,
        bin_labels=bin_labels,
        strategy=strategy,
    )
    return extract_fpmax_features_from_prepared(
        prepared=prepared,
        min_support=min_support,
        drop_original_numeric=drop_original_numeric,
    )


def prepare_fpmax_data(
    df: pd.DataFrame,
    numeric_cols: list[str] | None = None,
    n_bins: int = 7,
    bin_labels: list[str] | None = None,
    strategy: DiscretizeStrategy = "kmeans",
) -> FpmaxPreparedData:
    """Discretize once and build one-hot data reusable for many support values."""

    if numeric_cols is None:
        numeric_cols = get_numeric_columns(df)

    df_discrete = discretize_features(
        df=df,
        columns=numeric_cols,
        n_bins=n_bins,
        strategy=strategy,
        bin_labels=bin_labels,
    )
    onehot = pd.get_dummies(df_discrete, dtype=bool)
    return FpmaxPreparedData(
        discrete_df=df_discrete,
        onehot=onehot,
        numeric_cols=numeric_cols,
    )


def extract_fpmax_features_from_prepared(
    prepared: FpmaxPreparedData,
    min_support: float = 0.2,
    drop_original_numeric: bool = True,
) -> FpmaxFeatures:
    """Run FP-Max on prepared data and convert itemsets into binary features."""

    itemsets = fpmax(
        prepared.onehot,
        min_support=min_support,
        use_colnames=True,
    )
    if not itemsets.empty:
        itemsets = itemsets.assign(
            feature_name=itemsets["itemsets"].map(lambda items: "+".join(sorted(items)))
        ).sort_values(
            by=["support", "feature_name"],
            ascending=[False, True],
        )

    features = prepared.discrete_df.copy()
    if not itemsets.empty:
        new_features = {
            row.feature_name: prepared.onehot.loc[:, sorted(row.itemsets)]
            .all(axis=1)
            .astype(int)
            for row in itemsets.itertuples(index=False)
        }
        features = pd.concat([features, pd.DataFrame(new_features)], axis=1)
        itemsets = itemsets.drop(columns="feature_name")

    if drop_original_numeric and prepared.numeric_cols:
        features = features.drop(columns=prepared.numeric_cols)

    return FpmaxFeatures(features=features, itemsets=itemsets)
