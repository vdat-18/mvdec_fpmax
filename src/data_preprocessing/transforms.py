"""Scaling, winsorization, and DBSCAN filtering for Tiki preprocessing."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

RIGHT_OUTLIER_COLUMNS = ["Followers", "Total Revenue", "Med Review", "Bad Review"]
LEFT_OUTLIER_COLUMNS = ["Rating Quality", "Good Review"]


@dataclass(frozen=True)
class DbscanResult:
    """Preprocessed DBSCAN labels and the non-noise subset."""

    labels: np.ndarray
    clusters_info: dict[int, int]
    non_noise_df: pd.DataFrame


def standardize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply notebook-equivalent Z-score normalization."""

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    return pd.DataFrame(scaled, columns=df.columns, index=df.index)


def winsorize_series(
    series: pd.Series, lower_limit: float, upper_limit: float
) -> pd.Series:
    """Winsorize a numeric series using count-based percentile limits."""

    values = series.to_numpy(copy=True)
    valid_mask = ~pd.isna(values)
    valid_values = values[valid_mask]
    n_values = len(valid_values)
    if n_values == 0:
        return series.copy()

    sorted_values = np.sort(valid_values)
    lower_count = int(lower_limit * n_values)
    upper_count = int(upper_limit * n_values)

    if lower_count > 0:
        lower_bound = sorted_values[lower_count]
        valid_values = np.maximum(valid_values, lower_bound)
    if upper_count > 0:
        upper_bound = sorted_values[n_values - upper_count - 1]
        valid_values = np.minimum(valid_values, upper_bound)

    out = series.copy()
    out.loc[valid_mask] = valid_values
    return out


def winsorize_preprocessed_features(df: pd.DataFrame) -> pd.DataFrame:
    """Winsorize the scaled columns with the same limits as the notebook."""

    out = df.copy()
    for column in RIGHT_OUTLIER_COLUMNS:
        out[column] = winsorize_series(out[column], lower_limit=0, upper_limit=0.05)
    for column in LEFT_OUTLIER_COLUMNS:
        out[column] = winsorize_series(out[column], lower_limit=0.05, upper_limit=0)
    return out


def compute_k_distance(
    df: pd.DataFrame,
    min_samples: int = 14,
) -> np.ndarray:
    """Return sorted k-distance values used by the notebook elbow plot."""

    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(df)
    distances, _ = neighbors_fit.kneighbors(df)
    return np.sort(distances[:, min_samples - 1])


def remove_dbscan_noise(
    df: pd.DataFrame,
    eps: float = 1.05,
    min_samples: int = 14,
) -> DbscanResult:
    """Run DBSCAN and return only non-noise rows."""

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(df)
    unique_labels, counts = np.unique(labels, return_counts=True)
    clusters_info = {
        int(label): int(count)
        for label, count in zip(unique_labels, counts, strict=True)
    }
    non_noise_df = df.loc[labels != -1].copy()

    return DbscanResult(
        labels=labels,
        clusters_info=clusters_info,
        non_noise_df=non_noise_df,
    )
