"""Load preprocessed data and cached MvDEC representation output."""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from config import H_FUSED_COLUMNS, PREPROCESSED_DATA_PATH
from pipeline.clustering import compute_gower_distance, compute_silhouette_diagnostics
from pipeline.dekm_checkpoint import validate_dekm_f_checkpoint_provenance
from pipeline.mvdec_calibration import (
    CENTERED_RMS_CALIBRATION_CLAIM_SCOPE,
    CENTERED_RMS_CALIBRATION_METHOD,
    CENTERED_RMS_CALIBRATION_SOURCE_PHASE,
    CenteredRmsCalibration,
)
from pipeline.mvdec_contract import (
    ALIGNED_BALANCED_FULL_BATCH_PROTOCOL_ID,
    ALIGNED_BALANCED_NO_ALIGNMENT_FULL_BATCH_PROTOCOL_ID,
    ALIGNED_BALANCED_PROTOCOL_ALIGNMENT_MODES,
    ALIGNED_BALANCED_PROTOCOL_IDS,
    ALIGNED_BALANCED_PROTOCOL_LOSS_SCALE_POLICIES,
    ALIGNED_BALANCED_PROTOCOL_SETTINGS,
    ALIGNED_BALANCED_WEAK_ALIGNMENT_FULL_BATCH_PROTOCOL_ID,
    ALIGNMENT_FULL_BATCH_CENTERED_ROW_L2,
    ALIGNMENT_UNCENTERED_ROW_L2,
    CALIBRATED_INDEPENDENT_DEKM_VIEW2_PROTOCOL_ID,
    DECISION_PROVENANCE_CLASSIFICATIONS,
    EQUATION11_PROTOCOL_IDS,
    EQUATION11_PROTOCOL_SETTINGS,
    EXTERNAL_VIEW1_PROTOCOL_IDS,
    FROZEN_DEKM_F_VIEW1_RELEASE_L2_PROTOCOL_ID,
    FROZEN_VIEW1_PROTOCOL_IDS,
    FUSION_ENCODER_AVERAGE,
    FUSION_ENCODER_CENTERED_RMS_CALIBRATED_AVERAGE,
    FUSION_ENCODER_CONCATENATION,
    FUSION_ENCODER_L2_NORMALIZED_AVERAGE,
    FUSION_ENCODER_L2_NORMALIZED_CONCATENATION,
    IMPORTED_DEKM_F_VIEW1_JOINT_EQ11_RELEASE_L2_PROTOCOL_ID,
    IMPORTED_JOINT_VIEW1_PROTOCOL_IDS,
    MIN_REFINEMENT_ALIGNED_VIEW1_PROTOCOL_ID,
    MIN_REFINEMENT_FROZEN_VIEW1_PROTOCOL_ID,
    MIN_REFINEMENT_JOINT_VIEW1_PROTOCOL_ID,
    MIN_REFINEMENT_SCRATCH_CONCAT_PROTOCOL_ID,
    MIN_REFINEMENT_SCRATCH_L2_CONCAT_PROTOCOL_ID,
    MIN_REFINEMENT_SCRATCH_PAPER_RMS_PROTOCOL_ID,
    MIN_REFINEMENT_SCRATCH_PROTOCOL_ID,
    MIN_REFINEMENT_SCRATCH_TABLE1_SUBSET_PROTOCOL_ID,
    MIN_REFINEMENT_SCRATCH_MINIBATCH_PROTOCOL_ID,
    MIN_REFINEMENT_SCRATCH_NO_L1_PROTOCOL_ID,
    MIN_REFINEMENT_SCRATCH_L1_TENTH_PROTOCOL_ID,
    MIN_REFINEMENT_FROZEN_VIEW1_L2_CONCAT_PROTOCOL_ID,
    MIN_REFINEMENT_FROZEN_VIEW1_L4_ONLY_PROTOCOL_ID,
    MIN_REFINEMENT_FROZEN_VIEW1_LOSS_TENTH_PROTOCOL_ID,
    MIN_REFINEMENT_FROZEN_VIEW1_REFRESH_50_PROTOCOL_ID,
    MIN_REFINEMENT_FROZEN_VIEW1_REFINE_LR_1E4_PROTOCOL_ID,
    MIN_REFINEMENT_EVEN_COLUMNS_PROTOCOL_ID,
    MIN_REFINEMENT_WIDE_VIEW2_PROTOCOL_ID,
    INDEPENDENT_DEKM_VIEW2_PROTOCOL_ID,
    INDEPENDENT_DEKM_VIEW2_PROTOCOL_IDS,
    INDEPENDENT_DEKM_VIEW2_SEED_SWEEP_PROTOCOL_ID,
    LEGACY_SCALED_ALIGNMENT_FULL_BATCH_PROTOCOL_ID,
    LEGACY_SCALED_CENTERED_ALIGNMENT_FULL_BATCH_PROTOCOL_ID,
    LEGACY_SCALED_CENTERED_SCALE_MATCHED_ALIGNMENT_FULL_BATCH_PROTOCOL_ID,
    LEGACY_SCALED_WEAK_ALIGNMENT_FULL_BATCH_PROTOCOL_ID,
    LITERAL_EQUATION11_PROTOCOL_ID,
    PAPER_INFORMED_FULL_BATCH_PROTOCOL_ID,
    PAPER_RESOLVED_FULL_BATCH_PROTOCOL_ID,
    PAPER_RESOLVED_RELEASE_L2_FULL_BATCH_PROTOCOL_ID,
    REFINEMENT_PROGRESS_UNIT_KMEANS_REFRESH_WINDOWS,
    REFINEMENT_REPRESENTATION_FUSED,
    REFINEMENT_REPRESENTATION_VIEW2,
    VIEW1_INITIALIZATION_IMPORTED_DEKM_F,
    VIEW1_INITIALIZATION_PRETRAINED_MVDEC,
    VIEW1_REFINEMENT_FROZEN_DEKM_F,
    VIEW1_REFINEMENT_JOINT,
    VIEW2_ARCHITECTURE_ID,
    VIEW2_EVEN_COLUMNS_ARCHITECTURE_ID,
    VIEW2_DIRECT_JOINT_HEAD_ARCHITECTURE_ID,
    VIEW2_ENCODER_BOTTLENECK_ARCHITECTURE_ID,
    VIEW_INPUT_CONTRACT,
    balanced_batch_count,
    representation_metric_semantics,
    view1_metric_semantics,
)
from pipeline.mvdec_integrity import (
    validate_frozen_view1_proof,
    validate_view1_refinement_transition,
)
from pipeline.mvdec_runs import (
    RUN_MANIFEST_FILENAME,
    load_run_manifest,
    resolve_manifest_output_path,
)
from pipeline.public_artifacts import load_public_artifact

PRIMARY_MVDEC_PROTOCOL_ID = "mvdec_dekm_consistent_v1"
PRIMARY_MVDEC_OBJECTIVE = "mvdec_dekm_consistent_l1_reconstruction_plus_l4_greedy"
PRIMARY_MVDEC_METHOD = "MvDEC-DEKM-consistent"
PUBLIC_DEKM_CONSISTENT_PROTOCOL_ID = "mvdec_dekm_consistent_public_v1"
PUBLIC_DEKM_CONSISTENT_METHOD = "MvDEC-DEKM-consistent-public"
PUBLIC_REPRODUCTION_PROTOCOL_ID = "mvdec_2025_public_reproduction_v1"
PUBLIC_REPRODUCTION_OBJECTIVE = (
    "mvdec_2025_public_reproduction_release_l1_reconstruction_plus_l4_greedy_mse"
)
PUBLIC_REPRODUCTION_METHOD = "MvDEC-2025-public-reproduction"
PUBLIC_ENCODER_BOTTLENECK_PROTOCOL_ID = "mvdec_2025_view2_encoder_bottleneck_v1"
PUBLIC_ENCODER_BOTTLENECK_METHOD = "MvDEC-2025-View2-encoder-bottleneck-ablation"
PUBLIC_DIRECT_JOINT_HEAD_PROTOCOL_ID = "mvdec_2025_view2_direct_23_split_v1"
PUBLIC_DIRECT_JOINT_HEAD_METHOD = "MvDEC-2025-View2-direct-joint-head-ablation"
PUBLIC_DIRECT_JOINT_HEAD_L2NORM_PROTOCOL_ID = (
    "mvdec_2025_view2_direct_23_split_l2norm_average_v1"
)
PUBLIC_DIRECT_JOINT_HEAD_L2NORM_METHOD = (
    "MvDEC-2025-View2-direct-joint-head-L2norm-average-ablation"
)
LITERAL_EQUATION11_OBJECTIVE = "mvdec_2025_equation11_l1_plus_l2_plus_l3_plus_l4"
ALIGNED_BALANCED_OBJECTIVE = "mvdec_aligned_balanced_l1_plus_l2_plus_l4_plus_l5"
LITERAL_EQUATION11_METHOD = "MvDEC-2025-Eq11-literal"
EQUATION11_AMBIGUITY_METHOD = "MvDEC-2025-Eq11-ambiguity-experiment"
PAPER_RESOLVED_METHOD = "MvDEC-2025-paper-resolved-full-batch"
PAPER_RESOLVED_RELEASE_L2_METHOD = (
    "MvDEC-2025-paper-resolved-release-L2-full-batch-diagnostic"
)
FROZEN_DEKM_F_VIEW1_RELEASE_L2_METHOD = (
    "MvDEC-2025-frozen-DEKM_F-View1-release-L2-diagnostic"
)
IMPORTED_DEKM_F_VIEW1_JOINT_EQ11_RELEASE_L2_METHOD = (
    "MvDEC-2025-imported-DEKM_F-View1-joint-Eq11-true-bottleneck-release-L2-diagnostic"
)
MIN_REFINEMENT_FROZEN_VIEW1_METHOD = (
    "MvDEC-2025-frozen-DEKM_F-View1-Eq11-true-bottleneck-min500-release-L2-diagnostic"
)
MIN_REFINEMENT_ALIGNED_VIEW1_METHOD = (
    "MvDEC-2025-frozen-DEKM_F-View1-true-bottleneck-centered-alignment-min500-"
    "release-L2-ablation"
)
MIN_REFINEMENT_EVEN_COLUMNS_METHOD = (
    "MvDEC-2025-frozen-DEKM_F-View1-true-bottleneck-even-columns-View2-min500-"
    "release-L2-ablation"
)
MIN_REFINEMENT_WIDE_VIEW2_METHOD = (
    "MvDEC-2025-frozen-DEKM_F-View1-true-bottleneck-wide-View2-min500-"
    "release-L2-ablation"
)
MIN_REFINEMENT_SCRATCH_METHOD = (
    "MvDEC-2025-scratch-both-views-true-bottleneck-min500-release-L2-diagnostic"
)
MIN_REFINEMENT_SCRATCH_CONCAT_METHOD = (
    "MvDEC-2025-scratch-both-views-true-bottleneck-figure2-concat-min500-"
    "release-L2-diagnostic"
)
MIN_REFINEMENT_SCRATCH_L2_CONCAT_METHOD = (
    "MvDEC-2025-scratch-both-views-true-bottleneck-figure2-L2-concat-min500-"
    "release-L2-diagnostic"
)
MIN_REFINEMENT_SCRATCH_PAPER_RMS_METHOD = (
    "MvDEC-2025-scratch-both-views-true-bottleneck-min500-paper-RMS-diagnostic"
)
MIN_REFINEMENT_SCRATCH_MINIBATCH_METHOD = (
    "MvDEC-2025-scratch-both-views-true-bottleneck-min500-minibatch256-"
    "release-L2-diagnostic"
)
MIN_REFINEMENT_SCRATCH_NO_L1_METHOD = (
    "MvDEC-2025-scratch-both-views-true-bottleneck-min500-no-L1-release-L2-diagnostic"
)
MIN_REFINEMENT_SCRATCH_L1_TENTH_METHOD = (
    "MvDEC-2025-scratch-both-views-true-bottleneck-min500-L1-0.1-"
    "release-L2-diagnostic"
)
MIN_REFINEMENT_FROZEN_VIEW1_L2_CONCAT_METHOD = (
    "MvDEC-2025-frozen-DEKM_F-View1-true-bottleneck-figure2-L2-concat-min500-"
    "release-L2-diagnostic"
)
MIN_REFINEMENT_FROZEN_VIEW1_L4_ONLY_METHOD = (
    "MvDEC-2025-frozen-DEKM_F-View1-true-bottleneck-L4-only-min500-"
    "release-L2-diagnostic"
)
MIN_REFINEMENT_FROZEN_VIEW1_LOSS_TENTH_METHOD = (
    "MvDEC-2025-frozen-DEKM_F-View1-true-bottleneck-L2-L3-0.1-min500-"
    "release-L2-diagnostic"
)
MIN_REFINEMENT_FROZEN_VIEW1_REFRESH_50_METHOD = (
    "MvDEC-2025-frozen-DEKM_F-View1-true-bottleneck-KMeans-refresh-50-min500-"
    "release-L2-diagnostic"
)
MIN_REFINEMENT_FROZEN_VIEW1_REFINE_LR_1E4_METHOD = (
    "MvDEC-2025-frozen-DEKM_F-View1-true-bottleneck-refine-lr-1e-4-min500-"
    "release-L2-diagnostic"
)
MIN_REFINEMENT_SCRATCH_TABLE1_SUBSET_METHOD = (
    "MvDEC-2025-scratch-both-views-true-bottleneck-min500-Table1-subset-"
    "release-L2-diagnostic"
)
MIN_REFINEMENT_JOINT_VIEW1_METHOD = (
    "MvDEC-2025-imported-DEKM_F-View1-joint-Eq11-true-bottleneck-min500-release-L2-diagnostic"
)
INDEPENDENT_DEKM_VIEW2_METHOD = (
    "MvDEC-2025-independent-DEKM-View2-release-L2-diagnostic"
)
INDEPENDENT_DEKM_VIEW2_SEED_SWEEP_METHOD = (
    "MvDEC-2025-independent-DEKM-View2-seed-sweep-release-L2-diagnostic"
)
CALIBRATED_INDEPENDENT_DEKM_VIEW2_METHOD = (
    "MvDEC-2025-independent-DEKM-View2-centered-RMS-calibrated-diagnostic"
)
INDEPENDENT_DEKM_VIEW2_OBJECTIVE = (
    "mvdec_independent_view2_dekm_greedy_mse_then_eq4_fusion"
)
PAPER_INFORMED_METHOD = "MvDEC-2025-paper-informed-full-batch"
ALIGNED_BALANCED_METHOD = "MvDEC-2025-aligned-balanced-full-batch"
ALIGNED_BALANCED_WEAK_ALIGNMENT_METHOD = (
    "MvDEC-2025-aligned-balanced-alignment-005-full-batch"
)
ALIGNED_BALANCED_NO_ALIGNMENT_METHOD = (
    "MvDEC-2025-aligned-balanced-alignment-000-full-batch"
)
LEGACY_SCALED_WEAK_ALIGNMENT_METHOD = (
    "MvDEC-2025-legacy-scaled-alignment-005-full-batch"
)
LEGACY_SCALED_ALIGNMENT_METHOD = "MvDEC-2025-legacy-scaled-alignment-100-full-batch"
LEGACY_SCALED_CENTERED_ALIGNMENT_METHOD = (
    "MvDEC-2025-legacy-scaled-centered-alignment-100-full-batch"
)
LEGACY_SCALED_CENTERED_SCALE_MATCHED_ALIGNMENT_METHOD = (
    "MvDEC-2025-legacy-scaled-centered-alignment-10000-full-batch"
)
PUBLIC_PROTOCOL_ARCHITECTURES = {
    PUBLIC_DEKM_CONSISTENT_PROTOCOL_ID: VIEW2_ARCHITECTURE_ID,
    PUBLIC_REPRODUCTION_PROTOCOL_ID: VIEW2_ARCHITECTURE_ID,
    INDEPENDENT_DEKM_VIEW2_PROTOCOL_ID: VIEW2_ARCHITECTURE_ID,
    INDEPENDENT_DEKM_VIEW2_SEED_SWEEP_PROTOCOL_ID: VIEW2_ARCHITECTURE_ID,
    CALIBRATED_INDEPENDENT_DEKM_VIEW2_PROTOCOL_ID: VIEW2_ARCHITECTURE_ID,
    PUBLIC_ENCODER_BOTTLENECK_PROTOCOL_ID: (VIEW2_ENCODER_BOTTLENECK_ARCHITECTURE_ID),
    PUBLIC_DIRECT_JOINT_HEAD_PROTOCOL_ID: VIEW2_DIRECT_JOINT_HEAD_ARCHITECTURE_ID,
    PUBLIC_DIRECT_JOINT_HEAD_L2NORM_PROTOCOL_ID: (
        VIEW2_DIRECT_JOINT_HEAD_ARCHITECTURE_ID
    ),
    **{
        protocol_id: VIEW2_DIRECT_JOINT_HEAD_ARCHITECTURE_ID
        for protocol_id in EQUATION11_PROTOCOL_IDS
    },
    **{
        protocol_id: VIEW2_DIRECT_JOINT_HEAD_ARCHITECTURE_ID
        for protocol_id in ALIGNED_BALANCED_PROTOCOL_IDS
    },
    IMPORTED_DEKM_F_VIEW1_JOINT_EQ11_RELEASE_L2_PROTOCOL_ID: VIEW2_ARCHITECTURE_ID,
    MIN_REFINEMENT_FROZEN_VIEW1_PROTOCOL_ID: VIEW2_ARCHITECTURE_ID,
    MIN_REFINEMENT_JOINT_VIEW1_PROTOCOL_ID: VIEW2_ARCHITECTURE_ID,
    MIN_REFINEMENT_SCRATCH_PROTOCOL_ID: VIEW2_ARCHITECTURE_ID,
    MIN_REFINEMENT_SCRATCH_CONCAT_PROTOCOL_ID: VIEW2_ARCHITECTURE_ID,
    MIN_REFINEMENT_SCRATCH_L2_CONCAT_PROTOCOL_ID: VIEW2_ARCHITECTURE_ID,
    MIN_REFINEMENT_SCRATCH_PAPER_RMS_PROTOCOL_ID: VIEW2_ARCHITECTURE_ID,
    MIN_REFINEMENT_SCRATCH_TABLE1_SUBSET_PROTOCOL_ID: VIEW2_ARCHITECTURE_ID,
    MIN_REFINEMENT_SCRATCH_MINIBATCH_PROTOCOL_ID: VIEW2_ARCHITECTURE_ID,
    MIN_REFINEMENT_SCRATCH_NO_L1_PROTOCOL_ID: VIEW2_ARCHITECTURE_ID,
    MIN_REFINEMENT_SCRATCH_L1_TENTH_PROTOCOL_ID: VIEW2_ARCHITECTURE_ID,
    MIN_REFINEMENT_FROZEN_VIEW1_L2_CONCAT_PROTOCOL_ID: VIEW2_ARCHITECTURE_ID,
    MIN_REFINEMENT_FROZEN_VIEW1_L4_ONLY_PROTOCOL_ID: VIEW2_ARCHITECTURE_ID,
    MIN_REFINEMENT_FROZEN_VIEW1_LOSS_TENTH_PROTOCOL_ID: VIEW2_ARCHITECTURE_ID,
    MIN_REFINEMENT_FROZEN_VIEW1_REFRESH_50_PROTOCOL_ID: VIEW2_ARCHITECTURE_ID,
    MIN_REFINEMENT_FROZEN_VIEW1_REFINE_LR_1E4_PROTOCOL_ID: VIEW2_ARCHITECTURE_ID,
    MIN_REFINEMENT_WIDE_VIEW2_PROTOCOL_ID: VIEW2_ARCHITECTURE_ID,
    MIN_REFINEMENT_EVEN_COLUMNS_PROTOCOL_ID: VIEW2_EVEN_COLUMNS_ARCHITECTURE_ID,
    MIN_REFINEMENT_ALIGNED_VIEW1_PROTOCOL_ID: VIEW2_ARCHITECTURE_ID,
}
PUBLIC_PROTOCOL_METHODS = {
    PUBLIC_DEKM_CONSISTENT_PROTOCOL_ID: PUBLIC_DEKM_CONSISTENT_METHOD,
    PUBLIC_REPRODUCTION_PROTOCOL_ID: PUBLIC_REPRODUCTION_METHOD,
    PUBLIC_ENCODER_BOTTLENECK_PROTOCOL_ID: PUBLIC_ENCODER_BOTTLENECK_METHOD,
    PUBLIC_DIRECT_JOINT_HEAD_PROTOCOL_ID: PUBLIC_DIRECT_JOINT_HEAD_METHOD,
    PUBLIC_DIRECT_JOINT_HEAD_L2NORM_PROTOCOL_ID: (
        PUBLIC_DIRECT_JOINT_HEAD_L2NORM_METHOD
    ),
    **{
        protocol_id: EQUATION11_AMBIGUITY_METHOD
        for protocol_id in EQUATION11_PROTOCOL_IDS
    },
    LITERAL_EQUATION11_PROTOCOL_ID: LITERAL_EQUATION11_METHOD,
    IMPORTED_DEKM_F_VIEW1_JOINT_EQ11_RELEASE_L2_PROTOCOL_ID: (
        IMPORTED_DEKM_F_VIEW1_JOINT_EQ11_RELEASE_L2_METHOD
    ),
    MIN_REFINEMENT_FROZEN_VIEW1_PROTOCOL_ID: MIN_REFINEMENT_FROZEN_VIEW1_METHOD,
    MIN_REFINEMENT_JOINT_VIEW1_PROTOCOL_ID: MIN_REFINEMENT_JOINT_VIEW1_METHOD,
    MIN_REFINEMENT_SCRATCH_PROTOCOL_ID: MIN_REFINEMENT_SCRATCH_METHOD,
    MIN_REFINEMENT_SCRATCH_CONCAT_PROTOCOL_ID: MIN_REFINEMENT_SCRATCH_CONCAT_METHOD,
    MIN_REFINEMENT_SCRATCH_L2_CONCAT_PROTOCOL_ID: (
        MIN_REFINEMENT_SCRATCH_L2_CONCAT_METHOD
    ),
    MIN_REFINEMENT_SCRATCH_PAPER_RMS_PROTOCOL_ID: (
        MIN_REFINEMENT_SCRATCH_PAPER_RMS_METHOD
    ),
    MIN_REFINEMENT_SCRATCH_TABLE1_SUBSET_PROTOCOL_ID: (
        MIN_REFINEMENT_SCRATCH_TABLE1_SUBSET_METHOD
    ),
    MIN_REFINEMENT_SCRATCH_MINIBATCH_PROTOCOL_ID: (
        MIN_REFINEMENT_SCRATCH_MINIBATCH_METHOD
    ),
    MIN_REFINEMENT_SCRATCH_NO_L1_PROTOCOL_ID: MIN_REFINEMENT_SCRATCH_NO_L1_METHOD,
    MIN_REFINEMENT_SCRATCH_L1_TENTH_PROTOCOL_ID: MIN_REFINEMENT_SCRATCH_L1_TENTH_METHOD,
    MIN_REFINEMENT_FROZEN_VIEW1_L2_CONCAT_PROTOCOL_ID: (
        MIN_REFINEMENT_FROZEN_VIEW1_L2_CONCAT_METHOD
    ),
    MIN_REFINEMENT_FROZEN_VIEW1_L4_ONLY_PROTOCOL_ID: (
        MIN_REFINEMENT_FROZEN_VIEW1_L4_ONLY_METHOD
    ),
    MIN_REFINEMENT_FROZEN_VIEW1_LOSS_TENTH_PROTOCOL_ID: (
        MIN_REFINEMENT_FROZEN_VIEW1_LOSS_TENTH_METHOD
    ),
    MIN_REFINEMENT_FROZEN_VIEW1_REFRESH_50_PROTOCOL_ID: (
        MIN_REFINEMENT_FROZEN_VIEW1_REFRESH_50_METHOD
    ),
    MIN_REFINEMENT_FROZEN_VIEW1_REFINE_LR_1E4_PROTOCOL_ID: (
        MIN_REFINEMENT_FROZEN_VIEW1_REFINE_LR_1E4_METHOD
    ),
    MIN_REFINEMENT_WIDE_VIEW2_PROTOCOL_ID: MIN_REFINEMENT_WIDE_VIEW2_METHOD,
    MIN_REFINEMENT_EVEN_COLUMNS_PROTOCOL_ID: MIN_REFINEMENT_EVEN_COLUMNS_METHOD,
    MIN_REFINEMENT_ALIGNED_VIEW1_PROTOCOL_ID: MIN_REFINEMENT_ALIGNED_VIEW1_METHOD,
    PAPER_RESOLVED_FULL_BATCH_PROTOCOL_ID: PAPER_RESOLVED_METHOD,
    PAPER_RESOLVED_RELEASE_L2_FULL_BATCH_PROTOCOL_ID: (
        PAPER_RESOLVED_RELEASE_L2_METHOD
    ),
    FROZEN_DEKM_F_VIEW1_RELEASE_L2_PROTOCOL_ID: (FROZEN_DEKM_F_VIEW1_RELEASE_L2_METHOD),
    INDEPENDENT_DEKM_VIEW2_PROTOCOL_ID: INDEPENDENT_DEKM_VIEW2_METHOD,
    INDEPENDENT_DEKM_VIEW2_SEED_SWEEP_PROTOCOL_ID: (
        INDEPENDENT_DEKM_VIEW2_SEED_SWEEP_METHOD
    ),
    CALIBRATED_INDEPENDENT_DEKM_VIEW2_PROTOCOL_ID: (
        CALIBRATED_INDEPENDENT_DEKM_VIEW2_METHOD
    ),
    PAPER_INFORMED_FULL_BATCH_PROTOCOL_ID: PAPER_INFORMED_METHOD,
    ALIGNED_BALANCED_FULL_BATCH_PROTOCOL_ID: ALIGNED_BALANCED_METHOD,
    ALIGNED_BALANCED_WEAK_ALIGNMENT_FULL_BATCH_PROTOCOL_ID: (
        ALIGNED_BALANCED_WEAK_ALIGNMENT_METHOD
    ),
    ALIGNED_BALANCED_NO_ALIGNMENT_FULL_BATCH_PROTOCOL_ID: (
        ALIGNED_BALANCED_NO_ALIGNMENT_METHOD
    ),
    LEGACY_SCALED_WEAK_ALIGNMENT_FULL_BATCH_PROTOCOL_ID: (
        LEGACY_SCALED_WEAK_ALIGNMENT_METHOD
    ),
    LEGACY_SCALED_ALIGNMENT_FULL_BATCH_PROTOCOL_ID: LEGACY_SCALED_ALIGNMENT_METHOD,
    LEGACY_SCALED_CENTERED_ALIGNMENT_FULL_BATCH_PROTOCOL_ID: (
        LEGACY_SCALED_CENTERED_ALIGNMENT_METHOD
    ),
    LEGACY_SCALED_CENTERED_SCALE_MATCHED_ALIGNMENT_FULL_BATCH_PROTOCOL_ID: (
        LEGACY_SCALED_CENTERED_SCALE_MATCHED_ALIGNMENT_METHOD
    ),
}
PUBLIC_PROTOCOL_FUSIONS = {
    protocol_id: FUSION_ENCODER_AVERAGE for protocol_id in PUBLIC_PROTOCOL_ARCHITECTURES
}
PUBLIC_PROTOCOL_FUSIONS[PUBLIC_DIRECT_JOINT_HEAD_L2NORM_PROTOCOL_ID] = (
    FUSION_ENCODER_L2_NORMALIZED_AVERAGE
)
PUBLIC_PROTOCOL_FUSIONS[MIN_REFINEMENT_SCRATCH_CONCAT_PROTOCOL_ID] = (
    FUSION_ENCODER_CONCATENATION
)
PUBLIC_PROTOCOL_FUSIONS[MIN_REFINEMENT_SCRATCH_L2_CONCAT_PROTOCOL_ID] = (
    FUSION_ENCODER_L2_NORMALIZED_CONCATENATION
)
PUBLIC_PROTOCOL_FUSIONS[MIN_REFINEMENT_FROZEN_VIEW1_L2_CONCAT_PROTOCOL_ID] = (
    FUSION_ENCODER_L2_NORMALIZED_CONCATENATION
)
PUBLIC_PROTOCOL_FUSIONS[CALIBRATED_INDEPENDENT_DEKM_VIEW2_PROTOCOL_ID] = (
    FUSION_ENCODER_CENTERED_RMS_CALIBRATED_AVERAGE
)
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


def _validate_calibration_metadata(
    best_result: dict,
    h_view1: np.ndarray | None = None,
) -> CenteredRmsCalibration | None:
    """Validate calibrated-fusion metadata at the artifact trust boundary."""

    config_value = best_result.get("config", {})
    if not isinstance(config_value, dict):
        raise ValueError("MvDEC artifact config must be a dictionary.")
    config = config_value
    protocol_id = best_result.get("protocol_id")
    artifact_metadata = best_result.get("calibration")
    config_metadata = config.get("calibration")
    is_calibrated = protocol_id == CALIBRATED_INDEPENDENT_DEKM_VIEW2_PROTOCOL_ID
    if not is_calibrated:
        if artifact_metadata is not None or config_metadata is not None:
            raise ValueError(
                "Non-calibrated MvDEC artifacts must not contain calibration metadata."
            )
        return None
    calibration = CenteredRmsCalibration.from_metadata(artifact_metadata)
    if config_metadata != calibration.to_metadata():
        raise ValueError("MvDEC calibration metadata is inconsistent with config.")
    if calibration.claim_scope != CENTERED_RMS_CALIBRATION_CLAIM_SCOPE:
        raise ValueError("MvDEC calibration claim scope is invalid.")
    if h_view1 is not None and (
        calibration.n_samples != h_view1.shape[0]
        or calibration.latent_width != h_view1.shape[1]
    ):
        raise ValueError("MvDEC calibration dimensions do not match h_view1.")
    return calibration


def _validate_frozen_view1_proof_metadata(
    best_result: dict,
    h_view1: np.ndarray,
) -> dict[str, object] | None:
    """Validate an optional frozen-View1 proof and require it for calibration."""

    config_value = best_result.get("config", {})
    if not isinstance(config_value, dict):
        raise ValueError("MvDEC artifact config must be a dictionary.")
    config = config_value
    protocol_id = best_result.get("protocol_id")
    artifact_proof = best_result.get("frozen_view1_proof")
    config_proof = config.get("frozen_view1_proof")
    is_frozen = protocol_id in FROZEN_VIEW1_PROTOCOL_IDS
    is_calibrated = protocol_id == CALIBRATED_INDEPENDENT_DEKM_VIEW2_PROTOCOL_ID
    if not is_frozen:
        if artifact_proof is not None or config_proof is not None:
            raise ValueError(
                "Non-frozen MvDEC artifacts must not contain a View1 proof."
            )
        return None
    if artifact_proof is None:
        if config_proof is not None or is_calibrated:
            raise ValueError("Calibrated MvDEC artifacts require a View1 proof.")
        return None
    validated = validate_frozen_view1_proof(artifact_proof, h_view1)
    if config_proof != validated:
        raise ValueError("Frozen View1 proof is inconsistent with artifact config.")
    return validated


def _validate_view1_refinement_transition_metadata(
    best_result: dict,
    h_view1: np.ndarray,
) -> dict[str, object] | None:
    """Require transition evidence only for imported jointly refined View1."""

    config_value = best_result.get("config", {})
    if not isinstance(config_value, dict):
        raise ValueError("MvDEC artifact config must be a dictionary.")
    config = config_value
    protocol_id = best_result.get("protocol_id")
    artifact_transition = best_result.get("view1_refinement_transition")
    config_transition = config.get("view1_refinement_transition")
    is_imported_joint = protocol_id in IMPORTED_JOINT_VIEW1_PROTOCOL_IDS
    if not is_imported_joint:
        if artifact_transition is not None or config_transition is not None:
            raise ValueError(
                "Only imported-joint MvDEC artifacts may contain a View1 "
                "refinement transition."
            )
        return None
    validated = validate_view1_refinement_transition(
        artifact_transition,
        h_view1,
    )
    if config_transition != validated:
        raise ValueError(
            "View1 refinement transition is inconsistent with artifact config."
        )
    return validated


def _validate_public_protocol_contract(
    best_result: dict,
    contract: dict,
    config: dict,
) -> None:
    """Validate an immutable bounded public-dataset protocol."""

    objective = contract["objective"]
    eigen = contract["eigen"]
    greedy_target = contract["greedy_target"]
    stopping = contract["stopping"]
    schedule = contract.get("schedule")
    protocol_id = best_result.get("protocol_id")
    expected_method = PUBLIC_PROTOCOL_METHODS[protocol_id]
    is_dekm_consistent = protocol_id == PUBLIC_DEKM_CONSISTENT_PROTOCOL_ID
    is_equation11 = protocol_id in EQUATION11_PROTOCOL_IDS
    is_independent_dekm_view2 = protocol_id in INDEPENDENT_DEKM_VIEW2_PROTOCOL_IDS
    is_calibrated_independent_view2 = (
        protocol_id == CALIBRATED_INDEPENDENT_DEKM_VIEW2_PROTOCOL_ID
    )
    is_external_view1 = protocol_id in EXTERNAL_VIEW1_PROTOCOL_IDS
    is_frozen_view1 = protocol_id in FROZEN_VIEW1_PROTOCOL_IDS
    expected_initialization_policy = (
        VIEW1_INITIALIZATION_IMPORTED_DEKM_F
        if is_external_view1
        else VIEW1_INITIALIZATION_PRETRAINED_MVDEC
    )
    expected_view1_policy = (
        VIEW1_REFINEMENT_FROZEN_DEKM_F if is_frozen_view1 else VIEW1_REFINEMENT_JOINT
    )
    expected_metric_semantics = representation_metric_semantics(
        expected_view1_policy,
        expected_initialization_policy,
    )
    is_aligned_balanced = protocol_id in ALIGNED_BALANCED_PROTOCOL_IDS
    alignment_weight = ALIGNED_BALANCED_PROTOCOL_SETTINGS.get(protocol_id, 0.0)
    expected_alignment_mode = ALIGNED_BALANCED_PROTOCOL_ALIGNMENT_MODES.get(protocol_id)
    legacy_alignment_default = (
        ALIGNMENT_UNCENTERED_ROW_L2
        if expected_alignment_mode == ALIGNMENT_UNCENTERED_ROW_L2
        else None
    )
    artifact_alignment_mode = best_result.get(
        "alignment_mode", legacy_alignment_default
    )
    config_alignment_mode = config.get("alignment_mode", legacy_alignment_default)
    expected_loss_scale_policy = ALIGNED_BALANCED_PROTOCOL_LOSS_SCALE_POLICIES.get(
        protocol_id
    )
    if is_aligned_balanced:
        expected_objective = ALIGNED_BALANCED_OBJECTIVE
    elif is_equation11:
        expected_objective = LITERAL_EQUATION11_OBJECTIVE
    elif is_independent_dekm_view2:
        expected_objective = INDEPENDENT_DEKM_VIEW2_OBJECTIVE
    elif is_dekm_consistent:
        expected_objective = PRIMARY_MVDEC_OBJECTIVE
    else:
        expected_objective = PUBLIC_REPRODUCTION_OBJECTIVE
    expected_l4_reduction = (
        "sum_squared_dimensions_then_mean_batch"
        if is_dekm_consistent or is_equation11 or is_aligned_balanced
        else "mean_squared_dimensions_per_sample_sum_batch_gradient"
    )
    expected_pretrain_reduction = (
        "sum_squared_dimensions_per_sample"
        if is_dekm_consistent
        else "mean_squared_dimensions_per_sample"
    )
    expected_pretrain_shuffle_buffer = None if is_dekm_consistent else 8000
    expected_refinement_objective = (
        "independent_view2_dekm_greedy_mse"
        if is_independent_dekm_view2
        else "joint_reconstruction_plus_greedy"
        if is_dekm_consistent or is_equation11 or is_aligned_balanced
        else "release_reconstruction_plus_greedy_mse"
    )
    expected_batching_policy = (
        "balanced_shuffled_each_epoch"
        if is_dekm_consistent
        else "sequential_release_order"
    )
    expected_kmeans_n_init_policy = (
        "fixed_100" if is_dekm_consistent else "initial_100_then_twice_previous_n_iter"
    )
    if is_aligned_balanced:
        expected_loss_terms = {
            "L1_reconstruction": {"weight": 1.0, "optimized": True},
            "L2_kmeans": {"weight": 1.0, "optimized": True},
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
            "L5_cross_view_alignment": {
                "weight": alignment_weight,
                "optimized": alignment_weight > 0,
                "definition": (
                    "mean_batch(||L2_normalize(h_view1-mean_batch(h_view1)) - "
                    "L2_normalize(h_view2-mean_batch(h_view2))||^2)"
                    if expected_alignment_mode == ALIGNMENT_FULL_BATCH_CENTERED_ROW_L2
                    else "mean_batch(||L2_normalize(h_view1) - "
                    "L2_normalize(h_view2)||^2)"
                ),
            },
        }
    elif is_independent_dekm_view2:
        expected_loss_terms = {
            "L1_reconstruction": {"weight": 0.0, "optimized": False},
            "L2_kmeans": {"weight": 0.0, "optimized": False},
            "L3_scatter_trace": {
                "weight": 0.0,
                "optimized": False,
                "mathematically_equivalent_to_L2": True,
            },
            "L4_greedy": {
                "weight": 1.0,
                "optimized": True,
                "reduction": expected_l4_reduction,
            },
        }
    elif is_equation11:
        expected_loss_terms = {
            "L1_reconstruction": {
                "weight": 1.0,
                "optimized": True,
                "definition": ("mean_batch(||x-xhat_view1||^2 + ||x-xhat_view2||^2)"),
                "reduction": "sum_squared_dimensions_then_mean_batch",
                "source_classification": "paper_explicit",
            },
            "L2_kmeans": {
                "weight": 1.0,
                "optimized": True,
                "definition": "mean_batch(||h-u_assignment||^2)",
                "reduction": "sum_squared_dimensions_then_mean_batch",
                "source_classification": "paper_explicit",
            },
            "L3_scatter_trace": {
                "weight": 1.0,
                "optimized": True,
                "mathematically_equivalent_to_L2": True,
                "definition": "mean_batch(||(h-u_assignment)V||^2)",
                "reduction": "sum_squared_dimensions_then_mean_batch",
                "source_classification": "paper_explicit",
                "double_counting_note": (
                    "With orthonormal V this equals L2, so literal Eq. 11 "
                    "counts the same distance twice."
                ),
            },
            "L4_greedy": {
                "weight": 1.0,
                "optimized": True,
                "reduction": "sum_squared_dimensions_then_mean_batch",
                "definition": "mean_batch(||Y_prime-hV||^2)",
                "source_classification": "paper_explicit",
            },
        }
    else:
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
                "reduction": expected_l4_reduction,
            },
        }
    expected_eigen_direction = (
        "smallest"
        if is_aligned_balanced
        else EQUATION11_PROTOCOL_SETTINGS[protocol_id][0]
        if is_equation11
        else "largest"
    )
    expected_target_mode = (
        "selected_dimension_only"
        if is_aligned_balanced
        else EQUATION11_PROTOCOL_SETTINGS[protocol_id][1]
        if is_equation11
        else "frozen_snapshot"
    )
    expected_batch_policy = (
        "full_dataset"
        if is_aligned_balanced or is_independent_dekm_view2
        else EQUATION11_PROTOCOL_SETTINGS[protocol_id][2]
        if is_equation11
        else "fixed_256"
    )
    expected_batch_size = (
        "full_dataset" if expected_batch_policy == "full_dataset" else 256
    )
    n_samples = int(best_result.get("n_samples", 0))
    expected_resolved_batch_size = (
        n_samples if expected_batch_policy == "full_dataset" else 256
    )
    if n_samples < 1 or expected_resolved_batch_size < 1:
        expected_batches_per_epoch = 0
    elif expected_batch_policy == "full_dataset":
        expected_batches_per_epoch = 1
    elif expected_batching_policy == "balanced_shuffled_each_epoch":
        expected_batches_per_epoch = balanced_batch_count(
            n_samples,
            expected_resolved_batch_size,
        )
    else:
        expected_batches_per_epoch = (
            n_samples + expected_resolved_batch_size - 1
        ) // expected_resolved_batch_size
    expected_update_interval = 10
    expected_max_training_steps = 14_000
    expected_schedule = {
        "scope": "public_datasets_only",
        "pretraining": {
            "epochs": 200,
            "batch_size": 256,
            "loss_reduction": expected_pretrain_reduction,
            "shuffle_buffer": expected_pretrain_shuffle_buffer,
        },
        "refinement": {
            "objective": expected_refinement_objective,
            "batch_size": expected_batch_size,
            "batching_policy": expected_batching_policy,
            "kmeans_refresh_policy": "fixed_10_updates",
            "update_interval": expected_update_interval,
            "max_training_steps": expected_max_training_steps,
            "kmeans_n_init_policy": expected_kmeans_n_init_policy,
        },
        "final_evaluation": {
            "representations": ["view1", "view2", "fused"],
            "clustering": "independent_kmeans_n_init_100",
            "metrics": ["acc", "nmi"],
            "ground_truth_usage": "final_evaluation_only",
            "metric_semantics": expected_metric_semantics,
            "view1_semantics": view1_metric_semantics(
                expected_view1_policy,
                expected_initialization_policy,
            ),
        },
    }
    if is_external_view1:
        expected_schedule["pretraining"]["view1_initialization_policy"] = (
            expected_initialization_policy
        )
        expected_schedule["refinement"]["view1_refinement_policy"] = (
            expected_view1_policy
        )
    if is_independent_dekm_view2:
        if is_calibrated_independent_view2:
            expected_schedule["refinement"].update(
                {
                    "representation": REFINEMENT_REPRESENTATION_FUSED,
                    "fusion_timing": "fixed_calibration_before_refinement",
                    "calibration_policy": {
                        "method": CENTERED_RMS_CALIBRATION_METHOD,
                        "source_phase": CENTERED_RMS_CALIBRATION_SOURCE_PHASE,
                        "label_free": True,
                        "calibration_count": 1,
                        "recomputed_during_refinement": False,
                        "claim_scope": CENTERED_RMS_CALIBRATION_CLAIM_SCOPE,
                    },
                }
            )
        else:
            expected_schedule["refinement"].update(
                {
                    "representation": REFINEMENT_REPRESENTATION_VIEW2,
                    "fusion_timing": "final_evaluation_only",
                }
            )
    if (
        (is_equation11 and protocol_id != LITERAL_EQUATION11_PROTOCOL_ID)
        or is_aligned_balanced
        or is_independent_dekm_view2
    ):
        expected_schedule["refinement"]["batch_size_policy"] = expected_batch_policy
    expected_loss_weights = {
        "reconstruction": 0.0 if is_independent_dekm_view2 else 1.0,
        "kmeans": 1.0 if is_equation11 or is_aligned_balanced else 0.0,
        "scatter_trace_diagnostic": 1.0 if is_equation11 else 0.0,
        "greedy": 1.0,
        **({"cross_view_alignment": alignment_weight} if is_aligned_balanced else {}),
    }
    training_steps_completed = int(best_result.get("training_steps_completed", -1))
    batches_per_epoch = int(best_result.get("batches_per_epoch", 0))
    refinement_epochs_completed = int(
        best_result.get("refinement_epochs_completed", -1)
    )
    stop_reason = best_result.get("stop_reason")
    refinement_progress_unit = best_result.get("refinement_progress_unit")
    if refinement_progress_unit is None:
        expected_refinement_progress = training_steps_completed // max(
            batches_per_epoch, 1
        )
    elif refinement_progress_unit == REFINEMENT_PROGRESS_UNIT_KMEANS_REFRESH_WINDOWS:
        expected_refinement_progress = (
            training_steps_completed // expected_update_interval
        )
    else:
        expected_refinement_progress = -1
    metric_names = (
        "view1_acc",
        "view1_nmi",
        "view2_acc",
        "view2_nmi",
        "fused_acc",
        "fused_nmi",
    )
    external_checkpoint = best_result.get("external_view1_checkpoint")
    config_external_checkpoint = config.get("external_view1_checkpoint")
    if is_external_view1:
        try:
            validated_external_checkpoint = validate_dekm_f_checkpoint_provenance(
                external_checkpoint,
                expected_source_sha256=best_result.get("source_sha256"),
            )
        except ValueError as error:
            raise ValueError(
                "Imported DEKM_F View1 artifact has invalid checkpoint provenance."
            ) from error
        if validated_external_checkpoint != config_external_checkpoint:
            raise ValueError(
                "Imported DEKM_F View1 artifact has inconsistent checkpoint provenance."
            )
    elif external_checkpoint is not None or config_external_checkpoint is not None:
        raise ValueError(
            "Ordinary MvDEC artifact must not contain View1 checkpoint provenance."
        )
    if any(
        (
            best_result.get("dataset") not in PUBLIC_MVDEC_DATASETS,
            best_result.get("algorithm_family") != "MvDEC",
            best_result.get("algorithm") != expected_method,
            best_result.get("method_name") != expected_method,
            is_external_view1
            and contract.get("claim_scope")
            != "controlled_diagnostic_not_exact_paper_truth",
            objective.get("name") != expected_objective,
            objective.get("loss_terms") != expected_loss_terms,
            (
                objective.get("loss_scale_policy") != expected_loss_scale_policy
                if is_aligned_balanced
                else "loss_scale_policy" in objective
            ),
            (
                config.get("loss_scale_policy") != expected_loss_scale_policy
                if is_aligned_balanced
                else "loss_scale_policy" in config
            ),
            (
                artifact_alignment_mode != expected_alignment_mode
                if is_aligned_balanced
                else "alignment_mode" in best_result
            ),
            (
                config_alignment_mode != expected_alignment_mode
                if is_aligned_balanced
                else "alignment_mode" in config
            ),
            contract.get("architecture", {}).get("view_input_contract")
            != VIEW_INPUT_CONTRACT,
            schedule.get("final_evaluation", {}).get("metric_semantics")
            != expected_metric_semantics,
            schedule.get("final_evaluation", {}).get("view1_semantics")
            != view1_metric_semantics(
                expected_view1_policy,
                expected_initialization_policy,
            ),
            is_external_view1
            and schedule.get("pretraining", {}).get("view1_initialization_policy")
            != expected_initialization_policy,
            is_external_view1
            and schedule.get("refinement", {}).get("view1_refinement_policy")
            != expected_view1_policy,
            best_result.get("view_input_contract") != VIEW_INPUT_CONTRACT,
            config.get("view_input_contract") != VIEW_INPUT_CONTRACT,
            is_external_view1
            and best_result.get("view1_initialization_policy")
            != expected_initialization_policy,
            is_external_view1
            and config.get("view1_initialization_policy")
            != expected_initialization_policy,
            is_external_view1
            and best_result.get("view1_refinement_policy") != expected_view1_policy,
            is_external_view1
            and config.get("view1_refinement_policy") != expected_view1_policy,
            is_independent_dekm_view2
            and not is_calibrated_independent_view2
            and best_result.get("refinement_representation")
            != REFINEMENT_REPRESENTATION_VIEW2,
            is_independent_dekm_view2
            and not is_calibrated_independent_view2
            and config.get("refinement_representation")
            != REFINEMENT_REPRESENTATION_VIEW2,
            is_calibrated_independent_view2
            and best_result.get(
                "refinement_representation", REFINEMENT_REPRESENTATION_FUSED
            )
            != REFINEMENT_REPRESENTATION_FUSED,
            is_calibrated_independent_view2
            and config.get("refinement_representation", REFINEMENT_REPRESENTATION_FUSED)
            != REFINEMENT_REPRESENTATION_FUSED,
            not is_independent_dekm_view2
            and "refinement_representation" in best_result,
            not is_independent_dekm_view2 and "refinement_representation" in config,
            not is_external_view1
            and best_result.get(
                "view1_initialization_policy", VIEW1_INITIALIZATION_PRETRAINED_MVDEC
            )
            != VIEW1_INITIALIZATION_PRETRAINED_MVDEC,
            not is_external_view1
            and config.get(
                "view1_initialization_policy", VIEW1_INITIALIZATION_PRETRAINED_MVDEC
            )
            != VIEW1_INITIALIZATION_PRETRAINED_MVDEC,
            not is_external_view1
            and best_result.get("view1_refinement_policy", VIEW1_REFINEMENT_JOINT)
            != VIEW1_REFINEMENT_JOINT,
            not is_external_view1
            and config.get("view1_refinement_policy", VIEW1_REFINEMENT_JOINT)
            != VIEW1_REFINEMENT_JOINT,
            best_result.get("representation_metric_semantics")
            != expected_metric_semantics,
            config.get("representation_metric_semantics") != expected_metric_semantics,
            eigen.get("order") != "ascending",
            eigen.get("direction") != expected_eigen_direction,
            greedy_target.get("mode") != expected_target_mode,
            schedule != expected_schedule,
            config.get("loss_weights") != expected_loss_weights,
            config.get("batch_size") != expected_resolved_batch_size,
            config.get("kmeans_refresh_policy") != "fixed_10_updates",
            config.get("refinement_batching_policy") != expected_batching_policy,
            config.get("update_interval") != expected_update_interval,
            config.get("max_training_steps") != expected_max_training_steps,
            config.get("kmeans_n_init_policy") != expected_kmeans_n_init_policy,
            config.get("kmeans_n_init") != 100,
            best_result.get("kmeans_n_init") != 100,
            config.get("pretrain_loss_reduction") != expected_pretrain_reduction,
            config.get("pretrain_shuffle_buffer") != expected_pretrain_shuffle_buffer,
            config.get("refinement_objective") != expected_refinement_objective,
            best_result.get("kmeans_refresh_policy") != "fixed_10_updates",
            best_result.get("refinement_batching_policy") != expected_batching_policy,
            best_result.get("kmeans_n_init_policy") != expected_kmeans_n_init_policy,
            best_result.get("iteration") != training_steps_completed,
            config.get("training_steps_completed") != training_steps_completed,
            n_samples < 1,
            batches_per_epoch != expected_batches_per_epoch,
            config.get("batches_per_epoch") != batches_per_epoch,
            refinement_epochs_completed != expected_refinement_progress,
            config.get("refinement_epochs_completed") != refinement_epochs_completed,
            config.get("refinement_progress_unit") != refinement_progress_unit,
            config.get("stop_reason") != stop_reason,
            not 0 <= training_steps_completed <= expected_max_training_steps,
            stop_reason not in {"converged_assignment", "max_epochs_reached"},
            stop_reason == "max_epochs_reached"
            and training_steps_completed != expected_max_training_steps,
            not np.isclose(
                float(stopping.get("assignment_change_tolerance", np.nan)),
                0.001,
            ),
            (is_equation11 or is_independent_dekm_view2)
            and not contract.get("decision_provenance"),
            is_equation11
            and protocol_id == LITERAL_EQUATION11_PROTOCOL_ID
            and (
                "refinement_batch_size_policy" in best_result
                or "resolved_refinement_batch_size" in best_result
                or "refinement_batch_size_policy" in config
                or "resolved_refinement_batch_size" in config
            ),
            (is_equation11 or is_aligned_balanced or is_independent_dekm_view2)
            and protocol_id != LITERAL_EQUATION11_PROTOCOL_ID
            and (
                best_result.get("refinement_batch_size_policy") != expected_batch_policy
                or config.get("refinement_batch_size_policy") != expected_batch_policy
                or best_result.get("resolved_refinement_batch_size")
                != expected_resolved_batch_size
                or config.get("resolved_refinement_batch_size")
                != expected_resolved_batch_size
            ),
            (is_equation11 or is_independent_dekm_view2)
            and {
                item.get("classification")
                for item in contract.get("decision_provenance", {}).values()
            }
            - DECISION_PROVENANCE_CLASSIFICATIONS,
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
        msg = "MvDEC public artifact violates its immutable protocol."
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
    expected_fusion_contract = PUBLIC_PROTOCOL_FUSIONS.get(
        protocol_id,
        FUSION_ENCODER_AVERAGE,
    )
    contract_fusion = architecture.get("fusion", FUSION_ENCODER_AVERAGE)
    if any(
        (
            contract_fusion != expected_fusion_contract,
            best_result.get("fusion_contract") != expected_fusion_contract,
            config.get("fusion_contract", FUSION_ENCODER_AVERAGE)
            != expected_fusion_contract,
        )
    ):
        msg = "MvDEC fusion identity is missing or inconsistent."
        raise ValueError(msg)
    if protocol_id in PUBLIC_PROTOCOL_ARCHITECTURES:
        _validate_public_protocol_contract(best_result, contract, config)
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
    protocol_id = best_result.get("protocol_id")
    is_external_view1 = protocol_id in EXTERNAL_VIEW1_PROTOCOL_IDS
    is_frozen_view1 = protocol_id in FROZEN_VIEW1_PROTOCOL_IDS
    is_imported_joint = protocol_id in IMPORTED_JOINT_VIEW1_PROTOCOL_IDS
    expected_initialization_policy = (
        VIEW1_INITIALIZATION_IMPORTED_DEKM_F
        if is_external_view1
        else VIEW1_INITIALIZATION_PRETRAINED_MVDEC
    )
    expected_refinement_policy = (
        VIEW1_REFINEMENT_FROZEN_DEKM_F if is_frozen_view1 else VIEW1_REFINEMENT_JOINT
    )
    metadata_locations = (
        best_result,
        artifact_config,
        manifest,
        manifest_config,
    )
    provenance_fields = (
        "training_commit",
        "source_id",
        "source_sha256",
        "n_samples",
        "n_features",
        "n_classes",
        "view_input_contract",
        "representation_metric_semantics",
    )
    provenance_mismatch = any(
        manifest.get(field) != best_result.get(field)
        or manifest_config.get(field) != best_result.get(field)
        or artifact_config.get(field) != best_result.get(field)
        for field in provenance_fields
    )
    if is_external_view1:
        provenance_mismatch = provenance_mismatch or any(
            location.get("view1_initialization_policy")
            != expected_initialization_policy
            or location.get("view1_refinement_policy")
            != expected_refinement_policy
            for location in metadata_locations
        )
    else:
        provenance_mismatch = provenance_mismatch or any(
            location.get(
                "view1_initialization_policy",
                VIEW1_INITIALIZATION_PRETRAINED_MVDEC,
            )
            != VIEW1_INITIALIZATION_PRETRAINED_MVDEC
            or location.get("view1_refinement_policy", VIEW1_REFINEMENT_JOINT)
            != VIEW1_REFINEMENT_JOINT
            for location in metadata_locations
        )

    checkpoint_values = tuple(
        location.get("external_view1_checkpoint") for location in metadata_locations
    )
    if is_external_view1:
        try:
            validated_external_checkpoint = validate_dekm_f_checkpoint_provenance(
                checkpoint_values[0],
                expected_source_sha256=best_result.get("source_sha256"),
            )
        except ValueError:
            external_checkpoint_mismatch = True
        else:
            external_checkpoint_mismatch = any(
                value != validated_external_checkpoint for value in checkpoint_values
            )
    else:
        external_checkpoint_mismatch = any(
            value is not None for value in checkpoint_values
        )

    is_calibrated = protocol_id == CALIBRATED_INDEPENDENT_DEKM_VIEW2_PROTOCOL_ID
    calibration_values = tuple(
        location.get("calibration") for location in metadata_locations
    )
    if is_calibrated:
        try:
            validated_calibration = CenteredRmsCalibration.from_metadata(
                calibration_values[0]
            ).to_metadata()
        except ValueError:
            calibration_mismatch = True
        else:
            calibration_mismatch = any(
                value != validated_calibration for value in calibration_values
            )
    else:
        calibration_mismatch = any(value is not None for value in calibration_values)

    proof_values = tuple(
        location.get("frozen_view1_proof") for location in metadata_locations
    )
    if is_frozen_view1:
        proof_mismatch = any(value is not None for value in proof_values) and (
            proof_values[0] is None
            or any(value != proof_values[0] for value in proof_values)
        )
        if not proof_mismatch and proof_values[0] is not None:
            try:
                validate_frozen_view1_proof(
                    proof_values[0],
                    np.asarray(best_result.get("h_view1")),
                )
            except ValueError:
                proof_mismatch = True
        if is_calibrated and proof_values[0] is None:
            proof_mismatch = True
    else:
        proof_mismatch = any(value is not None for value in proof_values)

    transition_values = tuple(
        location.get("view1_refinement_transition")
        for location in metadata_locations
    )
    if is_imported_joint:
        transition_mismatch = transition_values[0] is None or any(
            value != transition_values[0] for value in transition_values
        )
        if not transition_mismatch:
            try:
                validate_view1_refinement_transition(
                    transition_values[0],
                    np.asarray(best_result.get("h_view1")),
                )
            except ValueError:
                transition_mismatch = True
    else:
        transition_mismatch = any(
            value is not None for value in transition_values
        )

    required_public_provenance_fields = provenance_fields + (
        (
            "view1_initialization_policy",
            "view1_refinement_policy",
        )
        if is_external_view1
        else ()
    )
    public_provenance_missing = best_result.get(
        "dataset"
    ) in PUBLIC_MVDEC_DATASETS and any(
        best_result.get(field) is None for field in required_public_provenance_fields
    )
    if any(
        (
            manifest.get("status") != "complete",
            manifest.get("run_id") != run_id,
            manifest.get("config_hash") != config_hash,
            manifest.get("dataset") != best_result.get("dataset"),
            manifest.get("protocol_id") != protocol_id,
            artifact_config.get("run_id") != run_id,
            artifact_config.get("config_hash") != config_hash,
            artifact_config.get("seed") != manifest.get("seed"),
            manifest_config.get("protocol_contract")
            != best_result.get("protocol_contract"),
            resolve_manifest_output_path(manifest, "artifact").resolve()
            != result_path.resolve(),
            output_sha256 != file_sha256(result_path),
            provenance_mismatch,
            external_checkpoint_mismatch,
            calibration_mismatch,
            proof_mismatch,
            transition_mismatch,
            public_provenance_missing,
        )
    ):
        msg = "MvDEC artifact does not match its complete run manifest."
        raise ValueError(msg)

def _l2_normalize_rows(latent: np.ndarray) -> np.ndarray:
    squared_norm = np.sum(np.square(latent), axis=1, keepdims=True)
    return latent / np.sqrt(np.maximum(squared_norm, 1e-12))


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
    if (
        h_view1.ndim != 2
        or h_view2.ndim != 2
        or h_view1.shape[0] != h_fused.shape[0]
        or h_view2.shape[0] != h_fused.shape[0]
        or h_view1.shape != h_view2.shape
    ):
        msg = (
            "MvDEC 2025 artifact must store compatible two-dimensional view "
            f"latents; got {h_view1.shape}, {h_view2.shape}, and {h_fused.shape}."
        )
        raise ValueError(msg)

    calibration = _validate_calibration_metadata(best_result, h_view1)
    _validate_frozen_view1_proof_metadata(best_result, h_view1)
    _validate_view1_refinement_transition_metadata(best_result, h_view1)

    fusion_dim = best_result.get("fusion_dim", best_result.get("latent_dim"))
    if fusion_dim is not None and h_fused.shape[1] != int(fusion_dim):
        msg = (
            "h_fused width must match fusion_dim for MvDEC 2025 artifacts: "
            f"{h_fused.shape[1]} != {fusion_dim}."
        )
        raise ValueError(msg)

    fusion_contract = best_result.get("fusion_contract")
    if fusion_contract == FUSION_ENCODER_AVERAGE:
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
    elif fusion_contract == FUSION_ENCODER_L2_NORMALIZED_AVERAGE:
        expected = (_l2_normalize_rows(h_view1) + _l2_normalize_rows(h_view2)) / 2
    elif fusion_contract == FUSION_ENCODER_CONCATENATION:
        expected = np.concatenate([h_view1, h_view2], axis=1)
    elif fusion_contract == FUSION_ENCODER_L2_NORMALIZED_CONCATENATION:
        expected = np.concatenate(
            [_l2_normalize_rows(h_view1), _l2_normalize_rows(h_view2)], axis=1
        )
    elif fusion_contract == FUSION_ENCODER_CENTERED_RMS_CALIBRATED_AVERAGE:
        if calibration is None:
            raise ValueError("Calibrated MvDEC fusion requires calibration metadata.")
        expected = (h_view1 + calibration.view2_scale_factor * h_view2) / 2
    else:
        msg = f"Unsupported MvDEC fusion contract: {fusion_contract!r}."
        raise ValueError(msg)

    fusion_matches = (
        np.array_equal(h_fused, expected)
        if fusion_contract == FUSION_ENCODER_CENTERED_RMS_CALIBRATED_AVERAGE
        else np.allclose(h_fused, expected, rtol=1e-5, atol=1e-6)
    )
    if h_fused.shape != expected.shape or not fusion_matches:
        msg = "h_fused does not match the declared MvDEC fusion contract."
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


def _load_validated_mvdec_artifact(
    result_path: Path,
    allow_legacy_concat: bool,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Load one artifact and validate its persisted contract without diagnostics."""

    best_result = load_public_artifact(result_path)
    if not isinstance(best_result, dict):
        msg = f"MvDEC artifact must contain a dictionary: {result_path}."
        raise ValueError(msg)

    h_fused = np.asarray(best_result["h_fused"])
    if h_fused.ndim != 2:
        msg = "h_fused must be a 2D array."
        raise ValueError(msg)
    if not np.isfinite(h_fused).all():
        msg = "h_fused contains NaN or infinite values."
        raise ValueError(msg)

    fusion_contract = best_result.get("fusion_contract")
    if fusion_contract in {
        FUSION_ENCODER_AVERAGE,
        FUSION_ENCODER_CONCATENATION,
        FUSION_ENCODER_L2_NORMALIZED_AVERAGE,
        FUSION_ENCODER_L2_NORMALIZED_CONCATENATION,
    }:
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

    return best_result, h_fused, labels, true_labels, row_indices


def validate_mvdec_artifact(
    result_path: Path,
    allow_legacy_concat: bool = False,
) -> dict:
    """Validate one MvDEC artifact without allocating pairwise diagnostics."""

    best_result, _, _, _, _ = _load_validated_mvdec_artifact(
        result_path,
        allow_legacy_concat,
    )
    return best_result


def load_mvdec_result(
    result_path: Path,
    data_path: Path = PREPROCESSED_DATA_PATH,
    allow_legacy_concat: bool = False,
) -> MvdecResult:
    """Load the cached MvDEC result and recompute its silhouette score."""

    best_result, h_fused, labels, true_labels, row_indices = (
        _load_validated_mvdec_artifact(result_path, allow_legacy_concat)
    )
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
