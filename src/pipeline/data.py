"""Load preprocessed data and cached MvDEC representation output."""

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from config import FUSED_REPRESENTATION_PATH, H_FUSED_COLUMNS, PREPROCESSED_DATA_PATH


@dataclass(frozen=True)
class PreprocessedDataset:
    """Preprocessed tabular dataset used by the clustering pipeline."""

    df: pd.DataFrame
    X: np.ndarray
    input_dim: int


@dataclass(frozen=True)
class MvdecResult:
    """Cached MvDEC representation output loaded from the legacy notebook run."""

    raw: dict
    h_fused: np.ndarray
    h_fused_df: pd.DataFrame
    labels: np.ndarray
    init: str
    score: float
    iteration: int
    score_check: float


def load_preprocessed_dataset(
    data_path: Path = PREPROCESSED_DATA_PATH,
) -> PreprocessedDataset:
    """Load the preprocessed CSV dataset as a dataframe and float32 matrix."""

    df = pd.read_csv(data_path)
    X = df.values.astype(np.float32)

    return PreprocessedDataset(
        df=df,
        X=X,
        input_dim=X.shape[1],
    )


def load_mvdec_result(
    result_path: Path = FUSED_REPRESENTATION_PATH,
    data_path: Path = PREPROCESSED_DATA_PATH,
) -> MvdecResult:
    """Load the cached MvDEC result and recompute its silhouette score."""

    with result_path.open("rb") as file:
        best_result = pickle.load(file)

    h_fused = np.asarray(best_result["h_fused"])
    if h_fused.ndim != 2:
        msg = "h_fused must be a 2D array."
        raise ValueError(msg)
    if h_fused.shape[1] != len(H_FUSED_COLUMNS):
        msg = (
            "h_fused column count does not match H_FUSED_COLUMNS: "
            f"{h_fused.shape[1]} != {len(H_FUSED_COLUMNS)}."
        )
        raise ValueError(msg)
    if not np.isfinite(h_fused).all():
        msg = "h_fused contains NaN or infinite values."
        raise ValueError(msg)

    labels = np.asarray(best_result["labels"])
    if labels.ndim != 1:
        msg = "MvDEC labels must be a 1D array."
        raise ValueError(msg)
    if len(labels) != h_fused.shape[0]:
        msg = (
            "MvDEC labels length does not match h_fused rows: "
            f"{len(labels)} != {h_fused.shape[0]}."
        )
        raise ValueError(msg)

    preprocessed_rows = len(pd.read_csv(data_path))
    if preprocessed_rows != h_fused.shape[0]:
        msg = (
            "Preprocessed dataset row count does not match h_fused rows: "
            f"{preprocessed_rows} != {h_fused.shape[0]}."
        )
        raise ValueError(msg)

    h_fused_df = pd.DataFrame(h_fused, columns=H_FUSED_COLUMNS)
    score_check = silhouette_score(h_fused, labels)

    return MvdecResult(
        raw=best_result,
        h_fused=h_fused,
        h_fused_df=h_fused_df,
        labels=labels,
        init=best_result["init"],
        score=float(best_result["score"]),
        iteration=int(best_result["iteration"]),
        score_check=float(score_check),
    )
