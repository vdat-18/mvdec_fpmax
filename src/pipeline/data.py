"""Load preprocessed data and cached MvDEC representation output."""

import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from config import FUSED_REPRESENTATION_PATH, H_FUSED_COLUMNS, PREPROCESSED_DATA_PATH


def fused_embedding_columns(n_columns: int) -> list[str]:
    """Return stable fused embedding column names for any representation width."""

    if n_columns == len(H_FUSED_COLUMNS):
        return H_FUSED_COLUMNS
    return [f"fused_{index}" for index in range(1, n_columns + 1)]


@dataclass(frozen=True)
class PreprocessedDataset:
    """Preprocessed tabular dataset used by the clustering pipeline."""

    df: pd.DataFrame
    X: np.ndarray
    input_dim: int


@dataclass(frozen=True)
class MvdecResult:
    """Cached MvDEC representation output."""

    raw: dict
    h_fused: np.ndarray
    h_fused_df: pd.DataFrame
    labels: np.ndarray
    h_view1: np.ndarray | None
    h_view2: np.ndarray | None
    fpmax_representation: np.ndarray | None
    view_concat_representation: np.ndarray | None
    init: str
    score: float
    iteration: int
    score_check: float


def _legacy_concat_width(best_result: dict) -> int | None:
    input_dim = best_result.get("input_dim")
    config = best_result.get("config", {})
    if not isinstance(config, dict):
        return None
    latent_dim = config.get("view1_latent_dim")
    if input_dim is None or latent_dim is None:
        return None
    return int(input_dim) + int(latent_dim)


def _validate_mvdec2025_contract(best_result: dict, h_fused: np.ndarray) -> None:
    view_output_layout = best_result.get("view_output_layout")
    if view_output_layout is not None and not re.fullmatch(
        r"eq5_compatible_\d+_plus_\d+",
        view_output_layout,
    ):
        msg = (
            "Unexpected MvDEC 2025 view_output_layout: "
            f"{view_output_layout!r}."
        )
        raise ValueError(msg)

    final_training_objective = best_result.get("final_training_objective")
    if final_training_objective not in {
        None,
        "dekm2021_greedy_cluster_loss_after_reconstruction_pretrain",
    }:
        msg = (
            "Unexpected MvDEC 2025 final_training_objective: "
            f"{final_training_objective!r}."
        )
        raise ValueError(msg)

    h_view1 = np.asarray(best_result.get("h_view1"))
    h_view2 = np.asarray(best_result.get("h_view2"))
    if h_view1.shape != h_fused.shape or h_view2.shape != h_fused.shape:
        msg = (
            "MvDEC 2025 artifact must store h_view1, h_view2, and h_fused "
            f"with the same shape; got {h_view1.shape}, {h_view2.shape}, "
            f"and {h_fused.shape}."
        )
        raise ValueError(msg)

    fusion_dim = best_result.get("fusion_dim", best_result.get("latent_dim"))
    if fusion_dim is not None and h_fused.shape[1] != int(fusion_dim):
        msg = (
            "h_fused width must match fusion_dim for MvDEC 2025 artifacts: "
            f"{h_fused.shape[1]} != {fusion_dim}."
        )
        raise ValueError(msg)

    expected = (h_view1 + h_view2) / 2
    if not np.allclose(h_fused, expected, rtol=1e-5, atol=1e-6):
        msg = "h_fused must equal (h_view1 + h_view2) / 2 for MvDEC 2025 artifacts."
        raise ValueError(msg)


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
    allow_legacy_concat: bool = False,
) -> MvdecResult:
    """Load the cached MvDEC result and recompute its silhouette score."""

    with result_path.open("rb") as file:
        best_result = pickle.load(file)

    h_fused = np.asarray(best_result["h_fused"])
    if h_fused.ndim != 2:
        msg = "h_fused must be a 2D array."
        raise ValueError(msg)
    if not np.isfinite(h_fused).all():
        msg = "h_fused contains NaN or infinite values."
        raise ValueError(msg)

    fusion_contract = best_result.get("fusion_contract")
    if fusion_contract in {
        "mvdec2025_encoder_average",
        "mvdec2025_figure_output_average",
    }:
        _validate_mvdec2025_contract(best_result, h_fused)
    elif (
        not allow_legacy_concat
        and h_fused.shape[1] == _legacy_concat_width(best_result)
    ):
        msg = (
            "This artifact looks like a legacy concat representation "
            "(input features + latent features), but h_fused must be the "
            "MvDEC figure-based fused view output. Regenerate it from "
            "src/representation_learning/MVDEC_dense.py or load it with "
            "allow_legacy_concat=True only for legacy experiments."
        )
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

    h_fused_df = pd.DataFrame(
        h_fused,
        columns=fused_embedding_columns(h_fused.shape[1]),
    )
    score_check = silhouette_score(h_fused, labels)

    return MvdecResult(
        raw=best_result,
        h_fused=h_fused,
        h_fused_df=h_fused_df,
        labels=labels,
        h_view1=(
            np.asarray(best_result["h_view1"]) if "h_view1" in best_result else None
        ),
        h_view2=(
            np.asarray(best_result["h_view2"]) if "h_view2" in best_result else None
        ),
        fpmax_representation=(
            np.asarray(best_result["fpmax_representation"])
            if "fpmax_representation" in best_result
            else None
        ),
        view_concat_representation=(
            np.asarray(best_result["view_concat_representation"])
            if "view_concat_representation" in best_result
            else None
        ),
        init=best_result["init"],
        score=float(best_result["score"]),
        iteration=int(best_result["iteration"]),
        score_check=float(score_check),
    )
