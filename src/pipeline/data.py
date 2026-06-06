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


def load_mvdec_result(result_path: Path = FUSED_REPRESENTATION_PATH) -> MvdecResult:
    """Load the cached MvDEC result and recompute its silhouette score."""

    with result_path.open("rb") as file:
        best_result = pickle.load(file)

    h_fused = best_result["h_fused"]
    h_fused_df = pd.DataFrame(h_fused, columns=H_FUSED_COLUMNS)
    labels = best_result["labels"]
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
