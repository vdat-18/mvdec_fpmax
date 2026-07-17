"""Load preprocessed data and cached MvDEC representation output."""

import hashlib
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from config import FUSED_REPRESENTATION_PATH, H_FUSED_COLUMNS, PREPROCESSED_DATA_PATH
from pipeline.clustering import compute_gower_distance, compute_silhouette_diagnostics


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
    evaluation_score: float
    evaluation_sample_std: float
    evaluation_negative_fraction: float


def _legacy_concat_width(best_result: dict) -> int | None:
    input_dim = best_result.get("input_dim")
    config = best_result.get("config", {})
    if not isinstance(config, dict):
        return None
    latent_dim = config.get("view1_latent_dim")
    if input_dim is None or latent_dim is None:
        return None
    return int(input_dim) + int(latent_dim)


def file_sha256(path: Path) -> str:
    """Return a cross-platform hash, normalizing text line endings."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        if path.suffix.lower() in {".csv", ".tsv", ".txt"}:
            for line in file:
                digest.update(line.replace(b"\r\n", b"\n"))
        else:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _validate_source_dataset(
    best_result: dict,
    data_path: Path,
    preprocessed_df: pd.DataFrame,
) -> None:
    """Validate source columns and ordered content when metadata is available."""

    preprocessing = best_result.get("preprocessing")
    if not isinstance(preprocessing, dict):
        return

    source_sha256 = preprocessing.get("source_sha256")
    if source_sha256 is not None and source_sha256 != file_sha256(data_path):
        msg = (
            "MvDEC source CSV hash does not match the ordered dataset used to "
            "train the artifact."
        )
        raise ValueError(msg)

    feature_columns = preprocessing.get("feature_columns")
    if feature_columns is not None and feature_columns != list(preprocessed_df.columns):
        msg = "MvDEC source feature columns do not match the training dataset."
        raise ValueError(msg)


def _validate_airpollution_preprocessing(
    best_result: dict,
    data_path: Path,
    preprocessed_df: pd.DataFrame,
) -> None:
    if best_result.get("dataset") != "AIRPOLLUTION":
        return

    greedy_eigen_direction = best_result.get("greedy_eigen_direction")
    if greedy_eigen_direction not in {"largest", "smallest"}:
        msg = (
            "Air Pollution MvDEC artifact must declare greedy_eigen_direction "
            "as 'largest' or 'smallest'."
        )
        raise ValueError(msg)
    if best_result.get("eigenvalue_order") != "ascending":
        msg = "Air Pollution MvDEC artifact must use ascending eigenvalue order."
        raise ValueError(msg)
    fusion_dim = int(best_result.get("fusion_dim", 0))
    expected_eigen_index = fusion_dim - 1 if greedy_eigen_direction == "largest" else 0
    if best_result.get("greedy_eigen_index") != expected_eigen_index:
        msg = "MvDEC artifact greedy eigen index does not match its direction."
        raise ValueError(msg)
    greedy_target_mode = best_result.get("greedy_target_mode")
    if greedy_target_mode not in {"selected_dimension_only", "frozen_snapshot"}:
        msg = (
            "Air Pollution MvDEC artifact must declare greedy_target_mode as "
            "'selected_dimension_only' or 'frozen_snapshot'."
        )
        raise ValueError(msg)
    config = best_result.get("config", {})
    if not isinstance(config, dict) or any(
        (
            config.get("eigenvalue_order") != "ascending",
            config.get("greedy_eigen_direction") != greedy_eigen_direction,
            config.get("greedy_eigen_index") != expected_eigen_index,
            config.get("greedy_target_mode") != greedy_target_mode,
        )
    ):
        msg = "MvDEC artifact eigen direction does not match its config metadata."
        raise ValueError(msg)
    if config.get("kmeans_n_init") != 100:
        msg = "Air Pollution MvDEC artifact must use fixed K-Means n_init=100."
        raise ValueError(msg)
    batch_size = int(config.get("batch_size", 0))
    if batch_size < 1:
        msg = "Air Pollution MvDEC artifact must record a positive batch_size."
        raise ValueError(msg)
    expected_batches_per_epoch = (len(preprocessed_df) + batch_size - 1) // batch_size
    if any(
        (
            best_result.get("kmeans_refresh_policy") != "one_epoch",
            config.get("kmeans_refresh_policy") != "one_epoch",
            best_result.get("batches_per_epoch") != expected_batches_per_epoch,
            config.get("batches_per_epoch") != expected_batches_per_epoch,
            config.get("update_interval") != expected_batches_per_epoch,
        )
    ):
        msg = "Air Pollution MvDEC artifact must refresh K-Means once per full epoch."
        raise ValueError(msg)
    max_refinement_epochs = int(config.get("max_refinement_epochs", 0))
    refinement_epochs_completed = int(
        best_result.get("refinement_epochs_completed", -1)
    )
    stop_reason = best_result.get("stop_reason")
    if any(
        (
            max_refinement_epochs < 1,
            stop_reason not in {"converged_assignment", "max_epochs_reached"},
            config.get("stop_reason") != stop_reason,
            config.get("refinement_epochs_completed") != refinement_epochs_completed,
            not 0 <= refinement_epochs_completed <= max_refinement_epochs,
            config.get("max_training_steps")
            != max_refinement_epochs * expected_batches_per_epoch,
            stop_reason == "max_epochs_reached"
            and refinement_epochs_completed != max_refinement_epochs,
        )
    ):
        msg = "Air Pollution MvDEC artifact has inconsistent training stop metadata."
        raise ValueError(msg)

    preprocessing = best_result.get("preprocessing")
    if not isinstance(preprocessing, dict):
        msg = (
            "Air Pollution MvDEC artifact is missing preprocessing metadata; "
            "regenerate it with column-wise Min-Max scaling."
        )
        raise ValueError(msg)
    if preprocessing.get("method") != "minmax":
        msg = (
            "Air Pollution MvDEC artifact must use column-wise Min-Max scaling; "
            f"got {preprocessing.get('method')!r}."
        )
        raise ValueError(msg)
    if preprocessing.get("feature_range") != [0.0, 1.0]:
        msg = "Air Pollution Min-Max feature_range must be [0.0, 1.0]."
        raise ValueError(msg)

    columns = list(preprocessed_df.columns)
    if preprocessing.get("feature_columns") != columns:
        msg = "Air Pollution scaler feature columns do not match the source CSV."
        raise ValueError(msg)
    data_min = np.asarray(preprocessing.get("data_min", []), dtype=float)
    data_max = np.asarray(preprocessing.get("data_max", []), dtype=float)
    if len(data_min) != len(columns) or len(data_max) != len(columns):
        msg = "Air Pollution scaler min/max metadata does not match the input width."
        raise ValueError(msg)
    source_values = preprocessed_df.to_numpy(dtype=float)
    if not np.allclose(data_min, source_values.min(axis=0)) or not np.allclose(
        data_max,
        source_values.max(axis=0),
    ):
        msg = "Air Pollution scaler min/max metadata does not match the source CSV."
        raise ValueError(msg)


def _validate_mvdec2025_contract(best_result: dict, h_fused: np.ndarray) -> None:
    view_output_layout = best_result.get("view_output_layout")
    if view_output_layout is not None and not re.fullmatch(
        r"eq5_compatible_\d+_plus_\d+",
        view_output_layout,
    ):
        msg = f"Unexpected MvDEC 2025 view_output_layout: {view_output_layout!r}."
        raise ValueError(msg)

    final_training_objective = best_result.get("final_training_objective")
    if final_training_objective not in {
        None,
        "dekm2021_greedy_cluster_loss_after_reconstruction_pretrain",
        "mvdec2025_joint_reconstruction_kmeans_orthonormal_greedy_loss",
        "mvdec2025_joint_reconstruction_kmeans_greedy_l3_trace_logged",
        "mvdec2025_latent_joint_reconstruction_kmeans_greedy_l3_trace_logged",
        "mvdec2025_latent_joint_reconstruction_greedy_l3_trace_logged",
        "mvdec2025_latent_joint_reconstruction_kmeans_l3_trace_logged",
        "mvdec2025_latent_joint_reconstruction_l3_trace_logged",
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

    if best_result.get("fusion_contract") == "mvdec2025_encoder_average":
        config = best_result.get("config", {})
        if not isinstance(config, dict):
            config = {}
        latent_dim = best_result.get("view1_latent_dim", config.get("view1_latent_dim"))
        if latent_dim is not None and h_fused.shape[1] != int(latent_dim):
            msg = (
                "h_fused width must match view1_latent_dim for MvDEC 2025 "
                f"encoder-average artifacts: {h_fused.shape[1]} != {latent_dim}."
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
    if fusion_contract == "mvdec2025_encoder_average":
        _validate_mvdec2025_contract(best_result, h_fused)
    elif fusion_contract == "mvdec2025_figure_output_average":
        if not allow_legacy_concat:
            msg = (
                "This artifact uses the legacy full-view-output average. Regenerate "
                "it so h_fused contains only the averaged encoder embeddings."
            )
            raise ValueError(msg)
        _validate_mvdec2025_contract(best_result, h_fused)
    elif not allow_legacy_concat and h_fused.shape[1] == _legacy_concat_width(
        best_result
    ):
        msg = (
            "This artifact looks like a legacy concat representation "
            "(input features + latent features), but h_fused must be the "
            "MvDEC fused encoder embedding. Regenerate it from "
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

    preprocessed_df = pd.read_csv(data_path)
    preprocessed_rows = len(preprocessed_df)
    if preprocessed_rows != h_fused.shape[0]:
        msg = (
            "Preprocessed dataset row count does not match h_fused rows: "
            f"{preprocessed_rows} != {h_fused.shape[0]}."
        )
        raise ValueError(msg)

    _validate_source_dataset(best_result, data_path, preprocessed_df)
    _validate_airpollution_preprocessing(best_result, data_path, preprocessed_df)

    h_fused_df = pd.DataFrame(
        h_fused,
        columns=fused_embedding_columns(h_fused.shape[1]),
    )
    score_check = silhouette_score(h_fused, labels)
    evaluation_distance = compute_gower_distance(h_fused_df)
    (
        evaluation_score,
        evaluation_sample_std,
        evaluation_negative_fraction,
    ) = compute_silhouette_diagnostics(evaluation_distance, labels)

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
        evaluation_score=evaluation_score,
        evaluation_sample_std=evaluation_sample_std,
        evaluation_negative_fraction=evaluation_negative_fraction,
    )
