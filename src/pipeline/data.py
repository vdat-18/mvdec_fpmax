"""Load preprocessed data and cached MvDEC representation output."""

import hashlib
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from config import H_FUSED_COLUMNS, PREPROCESSED_DATA_PATH
from pipeline.clustering import compute_gower_distance, compute_silhouette_diagnostics
from pipeline.mvdec_contract import (
    VIEW2_ARCHITECTURE_ID,
    VIEW2_ENCODER_BOTTLENECK_ARCHITECTURE_ID,
)
from pipeline.mvdec_runs import (
    RUN_MANIFEST_FILENAME,
    load_run_manifest,
    resolve_manifest_output_path,
)

PRIMARY_MVDEC_PROTOCOL_ID = "mvdec_dekm_consistent_v1"
PRIMARY_MVDEC_OBJECTIVE = "mvdec_dekm_consistent_l1_reconstruction_plus_l4_greedy"
PRIMARY_MVDEC_METHOD = "MvDEC-DEKM-consistent"
PUBLIC_REPRODUCTION_PROTOCOL_ID = "mvdec_2025_public_reproduction_v1"
PUBLIC_REPRODUCTION_OBJECTIVE = (
    "mvdec_2025_public_reproduction_release_"
    "l1_reconstruction_plus_l4_greedy_mse"
)
PUBLIC_REPRODUCTION_METHOD = "MvDEC-2025-public-reproduction"
PUBLIC_ENCODER_BOTTLENECK_PROTOCOL_ID = (
    "mvdec_2025_view2_encoder_bottleneck_v1"
)
PUBLIC_ENCODER_BOTTLENECK_METHOD = (
    "MvDEC-2025-View2-encoder-bottleneck-ablation"
)
PUBLIC_PROTOCOL_ARCHITECTURES = {
    PUBLIC_REPRODUCTION_PROTOCOL_ID: VIEW2_ARCHITECTURE_ID,
    PUBLIC_ENCODER_BOTTLENECK_PROTOCOL_ID: (
        VIEW2_ENCODER_BOTTLENECK_ARCHITECTURE_ID
    ),
}
PUBLIC_PROTOCOL_METHODS = {
    PUBLIC_REPRODUCTION_PROTOCOL_ID: PUBLIC_REPRODUCTION_METHOD,
    PUBLIC_ENCODER_BOTTLENECK_PROTOCOL_ID: PUBLIC_ENCODER_BOTTLENECK_METHOD,
}
PUBLIC_MVDEC_DATASETS = {"REUTERS", "20NEWS", "RCV1"}
PRIVATE_MVDEC_DATASETS = {"AIRPOLLUTION", "TIKI"}


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
    true_labels: np.ndarray | None
    row_indices: np.ndarray | None
    acc: float | None
    nmi: float | None
    source_sha256: str | None


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
            "regenerate it with an explicit preprocessing method."
        )
        raise ValueError(msg)
    preprocessing_method = preprocessing.get("method")
    expected_feature_ranges = {
        "minmax": [0.0, 1.0],
        "standard": None,
        "none": None,
    }
    if preprocessing_method not in expected_feature_ranges:
        msg = (
            f"Unsupported Air Pollution preprocessing method: {preprocessing_method!r}."
        )
        raise ValueError(msg)
    expected_feature_range = expected_feature_ranges[preprocessing_method]
    if preprocessing.get("feature_range") != expected_feature_range:
        msg = (
            "Air Pollution preprocessing feature_range does not match method "
            f"{preprocessing_method!r}."
        )
        raise ValueError(msg)

    columns = list(preprocessed_df.columns)
    if preprocessing.get("feature_columns") != columns:
        msg = "Air Pollution preprocessing columns do not match the source CSV."
        raise ValueError(msg)
    data_min = np.asarray(preprocessing.get("data_min", []), dtype=float)
    data_max = np.asarray(preprocessing.get("data_max", []), dtype=float)
    if len(data_min) != len(columns) or len(data_max) != len(columns):
        msg = "Air Pollution preprocessing min/max metadata has the wrong width."
        raise ValueError(msg)
    source_values = preprocessed_df.to_numpy(dtype=float)
    if not np.allclose(data_min, source_values.min(axis=0)) or not np.allclose(
        data_max,
        source_values.max(axis=0),
    ):
        msg = (
            "Air Pollution preprocessing min/max metadata does not match the "
            "source CSV."
        )
        raise ValueError(msg)
    if preprocessing_method == "standard":
        data_mean = np.asarray(preprocessing.get("data_mean", []), dtype=float)
        data_std = np.asarray(preprocessing.get("data_std", []), dtype=float)
        if len(data_mean) != len(columns) or len(data_std) != len(columns):
            msg = "Air Pollution standard-scaling metadata has the wrong width."
            raise ValueError(msg)
        if not np.allclose(data_mean, source_values.mean(axis=0)) or not np.allclose(
            data_std,
            source_values.std(axis=0),
        ):
            msg = (
                "Air Pollution standard-scaling mean/std metadata does not match "
                "the source CSV."
            )
            raise ValueError(msg)


def _protocol_contract_sha256(contract: dict[str, object]) -> str:
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_public_reproduction_contract(
    best_result: dict,
    contract: dict,
    config: dict,
) -> None:
    """Validate the immutable historical public-reproduction schedule."""

    objective = contract["objective"]
    eigen = contract["eigen"]
    greedy_target = contract["greedy_target"]
    stopping = contract["stopping"]
    schedule = contract.get("schedule")
    protocol_id = best_result.get("protocol_id")
    expected_method = PUBLIC_PROTOCOL_METHODS[protocol_id]
    expected_loss_terms = {
        "L1_reconstruction": {"weight": 1.0, "optimized": True},
        "L2_kmeans": {"weight": 0.0, "optimized": False},
        "L3_scatter_trace": {
            "weight": 0.0,
            "optimized": False,
            "mathematically_equivalent_to_L2": True,
        },
        "L4_greedy": {
            "weight": 1.0,
            "optimized": True,
            "reduction": ("mean_squared_dimensions_per_sample_sum_batch_gradient"),
        },
    }
    expected_schedule = {
        "scope": "public_datasets_only",
        "pretraining": {
            "epochs": 200,
            "batch_size": 256,
            "loss_reduction": "mean_squared_dimensions_per_sample",
            "shuffle_buffer": 8000,
        },
        "refinement": {
            "objective": "release_reconstruction_plus_greedy_mse",
            "batch_size": 256,
            "batching_policy": "sequential_release_order",
            "kmeans_refresh_policy": "fixed_10_updates",
            "update_interval": 10,
            "max_training_steps": 14000,
            "kmeans_n_init_policy": "initial_100_then_twice_previous_n_iter",
        },
        "final_evaluation": {
            "representations": ["view1", "view2", "fused"],
            "clustering": "independent_kmeans_n_init_100",
            "metrics": ["acc", "nmi"],
            "ground_truth_usage": "final_evaluation_only",
        },
    }
    expected_loss_weights = {
        "reconstruction": 1.0,
        "kmeans": 0.0,
        "scatter_trace_diagnostic": 0.0,
        "greedy": 1.0,
    }
    training_steps_completed = int(best_result.get("training_steps_completed", -1))
    batches_per_epoch = int(best_result.get("batches_per_epoch", 0))
    refinement_epochs_completed = int(
        best_result.get("refinement_epochs_completed", -1)
    )
    stop_reason = best_result.get("stop_reason")
    metric_names = (
        "view1_acc",
        "view1_nmi",
        "view2_acc",
        "view2_nmi",
        "fused_acc",
        "fused_nmi",
    )
    if any(
        (
            best_result.get("dataset") not in PUBLIC_MVDEC_DATASETS,
            best_result.get("algorithm_family") != "MvDEC",
            best_result.get("algorithm") != expected_method,
            best_result.get("method_name") != expected_method,
            objective.get("name") != PUBLIC_REPRODUCTION_OBJECTIVE,
            objective.get("loss_terms") != expected_loss_terms,
            eigen.get("order") != "ascending",
            eigen.get("direction") != "largest",
            greedy_target.get("mode") != "frozen_snapshot",
            schedule != expected_schedule,
            config.get("loss_weights") != expected_loss_weights,
            config.get("kmeans_refresh_policy") != "fixed_10_updates",
            config.get("refinement_batching_policy") != "sequential_release_order",
            config.get("update_interval") != 10,
            config.get("max_training_steps") != 14000,
            config.get("kmeans_n_init_policy")
            != "initial_100_then_twice_previous_n_iter",
            config.get("pretrain_loss_reduction")
            != "mean_squared_dimensions_per_sample",
            config.get("pretrain_shuffle_buffer") != 8000,
            config.get("refinement_objective")
            != "release_reconstruction_plus_greedy_mse",
            best_result.get("kmeans_refresh_policy") != "fixed_10_updates",
            best_result.get("refinement_batching_policy") != "sequential_release_order",
            best_result.get("kmeans_n_init_policy")
            != "initial_100_then_twice_previous_n_iter",
            best_result.get("iteration") != training_steps_completed,
            config.get("training_steps_completed") != training_steps_completed,
            batches_per_epoch < 1,
            config.get("batches_per_epoch") != batches_per_epoch,
            refinement_epochs_completed
            != training_steps_completed // max(batches_per_epoch, 1),
            config.get("refinement_epochs_completed")
            != refinement_epochs_completed,
            config.get("stop_reason") != stop_reason,
            not 0 <= training_steps_completed <= 14000,
            stop_reason not in {"converged_assignment", "max_epochs_reached"},
            stop_reason == "max_epochs_reached" and training_steps_completed != 14000,
            not np.isclose(
                float(stopping.get("assignment_change_tolerance", np.nan)),
                0.001,
            ),
            any(best_result.get(name) is None for name in metric_names),
            not np.isclose(
                float(best_result.get("acc", np.nan)),
                float(best_result.get("fused_acc", np.nan)),
            ),
            not np.isclose(
                float(best_result.get("nmi", np.nan)),
                float(best_result.get("fused_nmi", np.nan)),
            ),
        )
    ):
        msg = "MvDEC public-reproduction artifact violates its immutable protocol."
        raise ValueError(msg)


def _validate_mvdec_protocol_contract(best_result: dict) -> None:
    """Validate the resolved protocol identity and detect artifact tampering."""

    protocol_id = best_result.get("protocol_id")
    contract = best_result.get("protocol_contract")
    stored_hash = best_result.get("protocol_contract_sha256")
    config = best_result.get("config")
    if not isinstance(contract, dict) or not isinstance(config, dict):
        msg = (
            "MvDEC protocol artifacts require dictionary contract and config metadata."
        )
        raise ValueError(msg)
    if not isinstance(stored_hash, str) or stored_hash != _protocol_contract_sha256(
        contract
    ):
        msg = "MvDEC protocol contract SHA-256 is missing or does not match."
        raise ValueError(msg)
    if any(
        (
            contract.get("protocol_id") != protocol_id,
            config.get("protocol_id") != protocol_id,
            config.get("protocol_contract") != contract,
            config.get("protocol_contract_sha256") != stored_hash,
        )
    ):
        msg = "MvDEC protocol identity is inconsistent across artifact metadata."
        raise ValueError(msg)

    objective = contract.get("objective")
    architecture = contract.get("architecture")
    eigen = contract.get("eigen")
    greedy_target = contract.get("greedy_target")
    stopping = contract.get("stopping")
    if not all(
        isinstance(section, dict)
        for section in (objective, architecture, eigen, greedy_target, stopping)
    ):
        msg = (
            "MvDEC protocol contract is missing "
            "architecture/objective/eigen/target/stopping."
        )
        raise ValueError(msg)
    view2_architecture_id = architecture.get("view2")
    expected_architecture_id = PUBLIC_PROTOCOL_ARCHITECTURES.get(
        protocol_id,
        VIEW2_ARCHITECTURE_ID,
    )
    if any(
        (
            view2_architecture_id != expected_architecture_id,
            best_result.get("view2_architecture_id") != view2_architecture_id,
            config.get("view2_architecture_id") != view2_architecture_id,
        )
    ):
        msg = "MvDEC View2 architecture identity is missing or inconsistent."
        raise ValueError(msg)
    if objective.get("name") != best_result.get("final_training_objective"):
        msg = "MvDEC objective name does not match its protocol contract."
        raise ValueError(msg)

    if protocol_id not in {
        PRIMARY_MVDEC_PROTOCOL_ID,
        *PUBLIC_PROTOCOL_ARCHITECTURES,
        "custom",
    }:
        msg = f"Unsupported MvDEC protocol_id: {protocol_id!r}."
        raise ValueError(msg)
    if protocol_id in PUBLIC_PROTOCOL_ARCHITECTURES:
        _validate_public_reproduction_contract(best_result, contract, config)
        return
    if protocol_id == "custom":
        if not str(objective.get("name", "")).startswith("mvdec_custom_"):
            msg = "Custom MvDEC protocols must use an explicit custom objective name."
            raise ValueError(msg)
        return

    loss_terms = objective.get("loss_terms")
    expected_loss_terms = {
        "L1_reconstruction": {"weight": 1.0, "optimized": True},
        "L2_kmeans": {"weight": 0.0, "optimized": False},
        "L3_scatter_trace": {
            "weight": 0.0,
            "optimized": False,
            "mathematically_equivalent_to_L2": True,
        },
        "L4_greedy": {
            "weight": 1.0,
            "optimized": True,
            "reduction": "sum_squared_dimensions_then_mean_batch",
        },
    }
    expected_tolerance = (
        0.01 if best_result.get("dataset") in PRIVATE_MVDEC_DATASETS else 0.001
    )
    expected_loss_weights = {
        "reconstruction": 1.0,
        "kmeans": 0.0,
        "scatter_trace_diagnostic": 0.0,
        "greedy": 1.0,
    }
    if any(
        (
            best_result.get("algorithm_family") != "MvDEC",
            best_result.get("algorithm") != PRIMARY_MVDEC_METHOD,
            best_result.get("method_name") != PRIMARY_MVDEC_METHOD,
            objective.get("name") != PRIMARY_MVDEC_OBJECTIVE,
            loss_terms != expected_loss_terms,
            eigen.get("order") != "ascending",
            eigen.get("direction") != "largest",
            greedy_target.get("mode") != "frozen_snapshot",
            config.get("loss_weights") != expected_loss_weights,
            config.get("eigenvalue_order") != "ascending",
            config.get("greedy_eigen_direction") != "largest",
            config.get("greedy_target_mode") != "frozen_snapshot",
            best_result.get("eigenvalue_order") != "ascending",
            best_result.get("greedy_eigen_direction") != "largest",
            best_result.get("greedy_target_mode") != "frozen_snapshot",
            not np.isclose(
                float(config.get("assignment_change_tolerance", np.nan)),
                expected_tolerance,
            ),
            not np.isclose(
                float(stopping.get("assignment_change_tolerance", np.nan)),
                expected_tolerance,
            ),
        )
    ):
        msg = "MvDEC-DEKM-consistent artifact violates its immutable protocol."
        raise ValueError(msg)


def _validate_mvdec_run_manifest(best_result: dict, result_path: Path) -> None:
    """Require a complete run manifest that owns and hashes this artifact."""

    run_id = best_result.get("run_id")
    config_hash = best_result.get("config_hash")
    if not isinstance(run_id, str) or not isinstance(config_hash, str):
        msg = "Paper-ready MvDEC artifacts require run_id and config_hash."
        raise ValueError(msg)
    manifest_path = result_path.parent / RUN_MANIFEST_FILENAME
    manifest = load_run_manifest(manifest_path)
    artifact_config = best_result.get("config", {})
    manifest_config = manifest.get("config", {})
    output_sha256 = manifest.get("output_sha256", {}).get("artifact")
    if any(
        (
            manifest.get("status") != "complete",
            manifest.get("run_id") != run_id,
            manifest.get("config_hash") != config_hash,
            manifest.get("dataset") != best_result.get("dataset"),
            manifest.get("protocol_id") != best_result.get("protocol_id"),
            artifact_config.get("run_id") != run_id,
            artifact_config.get("config_hash") != config_hash,
            artifact_config.get("seed") != manifest.get("seed"),
            manifest_config.get("protocol_contract")
            != best_result.get("protocol_contract"),
            resolve_manifest_output_path(manifest, "artifact").resolve()
            != result_path.resolve(),
            output_sha256 != file_sha256(result_path),
        )
    ):
        msg = "MvDEC artifact does not match its complete run manifest."
        raise ValueError(msg)


def _validate_mvdec2025_contract(
    best_result: dict,
    h_fused: np.ndarray,
    result_path: Path,
) -> None:
    view_output_layout = best_result.get("view_output_layout")
    if view_output_layout is not None and not re.fullmatch(
        r"eq5_compatible_\d+_plus_\d+",
        view_output_layout,
    ):
        msg = f"Unexpected MvDEC 2025 view_output_layout: {view_output_layout!r}."
        raise ValueError(msg)

    if best_result.get("protocol_id") is None:
        msg = (
            "Legacy MvDEC artifact has no protocol contract. Regenerate it with "
            "mvdec_dekm_consistent_v1 before using it in the paper pipeline."
        )
        raise ValueError(msg)
    _validate_mvdec_protocol_contract(best_result)
    _validate_mvdec_run_manifest(best_result, result_path)

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
    result_path: Path,
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
        _validate_mvdec2025_contract(best_result, h_fused, result_path)
    elif fusion_contract == "mvdec2025_figure_output_average":
        if not allow_legacy_concat:
            msg = (
                "This artifact uses the legacy full-view-output average. Regenerate "
                "it so h_fused contains only the averaged encoder embeddings."
            )
            raise ValueError(msg)
        _validate_mvdec2025_contract(best_result, h_fused, result_path)
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

    true_labels = (
        np.asarray(best_result["true_labels"])
        if best_result.get("true_labels") is not None
        else None
    )
    if true_labels is not None and (
        true_labels.ndim != 1 or len(true_labels) != len(labels)
    ):
        msg = "MvDEC true_labels must be one-dimensional and match artifact rows."
        raise ValueError(msg)
    row_indices = (
        np.asarray(best_result["row_indices"], dtype=int)
        if best_result.get("row_indices") is not None
        else None
    )
    if row_indices is not None and not np.array_equal(
        row_indices,
        np.arange(len(labels)),
    ):
        msg = "MvDEC row_indices must use canonical order 0..n_samples - 1."
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
        true_labels=true_labels,
        row_indices=row_indices,
        acc=(float(best_result["acc"]) if best_result.get("acc") is not None else None),
        nmi=(float(best_result["nmi"]) if best_result.get("nmi") is not None else None),
        source_sha256=best_result.get("source_sha256"),
    )
