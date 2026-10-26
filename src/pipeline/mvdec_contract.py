"""Versioned architecture identifiers shared by MvDEC runners and loaders."""

VIEW2_ARCHITECTURE_ID = "mvdec2025_post_skip_latent_bottleneck_v2"
VIEW2_EVEN_COLUMNS_ARCHITECTURE_ID = (
    "mvdec2025_post_skip_latent_bottleneck_even_columns_v1"
)
VIEW2_ENCODER_BOTTLENECK_ARCHITECTURE_ID = (
    "mvdec2025_encoder_bottleneck_skip_decoder_v1"
)
VIEW2_DIRECT_JOINT_HEAD_ARCHITECTURE_ID = (
    "mvdec2025_post_skip_direct_latent_reconstruction_head_v1"
)
FUSION_ENCODER_AVERAGE = "mvdec2025_encoder_average"
FUSION_ENCODER_L2_NORMALIZED_AVERAGE = "mvdec2025_encoder_l2_normalized_average"
FUSION_ENCODER_CENTERED_RMS_CALIBRATED_AVERAGE = (
    "mvdec2025_encoder_centered_global_rms_calibrated_average"
)
FUSION_ENCODER_CONCATENATION = "mvdec2025_figure2_encoder_concatenation"
FUSION_FIGURE1_FULL_OUTPUT_AVERAGE = "mvdec2025_figure1_full_output_average_diagnostic"
FUSION_FIGURE2_FULL_OUTPUT_CONCATENATION = (
    "mvdec2025_figure2_full_output_concatenation_diagnostic"
)
PAPER_FIGURE_FUSION_CONTRACTS = (
    FUSION_FIGURE1_FULL_OUTPUT_AVERAGE,
    FUSION_FIGURE2_FULL_OUTPUT_CONCATENATION,
)
VIEW_INPUT_CONTRACT = "same_full_feature_matrix_both_views_v1"
DECISION_PROVENANCE_CLASSIFICATIONS = frozenset(
    {
        "paper_explicit",
        "dekm_inherited",
        "implementation_assumption",
        "paper_explicit_conflicting_statement",
        "paper_resolved_interpretation",
        "paper_table_inference",
        "ablation",
        "cross_paper_inference",
        "controlled_architecture_diagnostic",
    }
)
VIEW1_INITIALIZATION_PRETRAINED_MVDEC = "pretrained_mvdec_view1"
VIEW1_INITIALIZATION_IMPORTED_DEKM_F = "imported_dekm_f_view1"
VIEW1_REFINEMENT_JOINT = "jointly_refined_views"
VIEW1_REFINEMENT_FROZEN_DEKM_F = "frozen_imported_dekm_f_view1"
REFINEMENT_REPRESENTATION_FUSED = "fused_latent"
REFINEMENT_REPRESENTATION_VIEW2 = "view2_latent_only"
REPRESENTATION_METRIC_SEMANTICS = "jointly_trained_views_independent_kmeans_v1"
FROZEN_VIEW1_REPRESENTATION_METRIC_SEMANTICS = (
    "frozen_dekm_f_view1_refined_view2_independent_kmeans_v1"
)
IMPORTED_JOINT_VIEW1_REPRESENTATION_METRIC_SEMANTICS = (
    "imported_dekm_f_view1_jointly_refined_views_independent_kmeans_v1"
)


def representation_metric_semantics(
    view1_refinement_policy: str,
    view1_initialization_policy: str = VIEW1_INITIALIZATION_PRETRAINED_MVDEC,
) -> str:
    """Return metric semantics for the selected View1 refinement policy."""

    if view1_refinement_policy == VIEW1_REFINEMENT_JOINT:
        if view1_initialization_policy == VIEW1_INITIALIZATION_IMPORTED_DEKM_F:
            return IMPORTED_JOINT_VIEW1_REPRESENTATION_METRIC_SEMANTICS
        return REPRESENTATION_METRIC_SEMANTICS
    if view1_refinement_policy == VIEW1_REFINEMENT_FROZEN_DEKM_F:
        return FROZEN_VIEW1_REPRESENTATION_METRIC_SEMANTICS
    raise ValueError(
        f"Unsupported View1 refinement policy: {view1_refinement_policy!r}."
    )


def view1_metric_semantics(
    view1_refinement_policy: str,
    view1_initialization_policy: str = VIEW1_INITIALIZATION_PRETRAINED_MVDEC,
) -> str:
    """Describe whether View1 is jointly trained or imported and frozen."""

    if view1_refinement_policy == VIEW1_REFINEMENT_JOINT:
        if view1_initialization_policy == VIEW1_INITIALIZATION_IMPORTED_DEKM_F:
            return "imported_dekm_f_checkpoint_then_jointly_refined"
        return "jointly_trained_view1_not_dekm_f"
    if view1_refinement_policy == VIEW1_REFINEMENT_FROZEN_DEKM_F:
        return "frozen_imported_dekm_f_checkpoint_unchanged_during_refinement"
    raise ValueError(
        f"Unsupported View1 refinement policy: {view1_refinement_policy!r}."
    )


REFINEMENT_PROGRESS_UNIT_DATASET_EPOCHS = "dataset_epochs"
REFINEMENT_PROGRESS_UNIT_KMEANS_REFRESH_WINDOWS = "kmeans_refresh_windows"
ALIGNMENT_UNCENTERED_ROW_L2 = "uncentered_row_l2"
ALIGNMENT_FULL_BATCH_CENTERED_ROW_L2 = "full_batch_centered_row_l2"


def paper_figure_fusion_metadata(
    fusion_contract: str,
    view_output_width: int,
) -> dict[str, object]:
    """Describe a non-training diagnostic for the paper's conflicting figures."""

    if view_output_width < 1:
        raise ValueError("view_output_width must be positive.")
    if fusion_contract == FUSION_FIGURE1_FULL_OUTPUT_AVERAGE:
        return {
            "fusion_contract": fusion_contract,
            "paper_source": "MvDEC 2025 Figure 1",
            "operation": "elementwise_average",
            "input_width_per_view": view_output_width,
            "output_width": view_output_width,
            "claim_scope": "post_training_diagnostic_not_exact_paper_truth",
        }
    if fusion_contract == FUSION_FIGURE2_FULL_OUTPUT_CONCATENATION:
        return {
            "fusion_contract": fusion_contract,
            "paper_source": "MvDEC 2025 Figure 2 final layer",
            "operation": "column_concatenation",
            "input_width_per_view": view_output_width,
            "output_width": 2 * view_output_width,
            "claim_scope": "post_training_diagnostic_not_exact_paper_truth",
        }
    raise ValueError(
        f"Unsupported MvDEC paper-figure fusion contract: {fusion_contract!r}."
    )


LITERAL_EQUATION11_PROTOCOL_ID = "mvdec_2025_equation11_literal_v1"
EQUATION11_SMALLEST_FROZEN_PROTOCOL_ID = "mvdec_2025_equation11_smallest_frozen_v1"
EQUATION11_LARGEST_SELECTED_ONLY_PROTOCOL_ID = (
    "mvdec_2025_equation11_largest_selected_only_v1"
)
EQUATION11_SMALLEST_SELECTED_ONLY_PROTOCOL_ID = (
    "mvdec_2025_equation11_smallest_selected_only_v1"
)
EQUATION11_LARGEST_FROZEN_FULL_BATCH_PROTOCOL_ID = (
    "mvdec_2025_equation11_largest_frozen_full_batch_v1"
)
EQUATION11_SMALLEST_FROZEN_FULL_BATCH_PROTOCOL_ID = (
    "mvdec_2025_equation11_smallest_frozen_full_batch_v1"
)
EQUATION11_LARGEST_SELECTED_ONLY_FULL_BATCH_PROTOCOL_ID = (
    "mvdec_2025_equation11_largest_selected_only_full_batch_v1"
)
PAPER_RESOLVED_FULL_BATCH_PROTOCOL_ID = "mvdec_2025_paper_resolved_full_batch_v1"
PAPER_RESOLVED_RELEASE_L2_FULL_BATCH_PROTOCOL_ID = (
    "mvdec_2025_paper_resolved_release_l2_full_batch_v1"
)
FROZEN_DEKM_F_VIEW1_RELEASE_L2_PROTOCOL_ID = (
    "mvdec_2025_frozen_dekm_f_view1_release_l2_v1"
)
IMPORTED_DEKM_F_VIEW1_JOINT_EQ11_RELEASE_L2_PROTOCOL_ID = (
    "mvdec_2025_frozen_dekm_f_view1_eq11_true_bottleneck_release_l2_seed42_v1"
)
MIN_REFINEMENT_FROZEN_VIEW1_PROTOCOL_ID = (
    "mvdec_2025_frozen_dekm_f_view1_eq11_true_bottleneck_min500_release_l2_seed42_v1"
)
MIN_REFINEMENT_JOINT_VIEW1_PROTOCOL_ID = (
    "mvdec_2025_imported_dekm_f_view1_joint_eq11_true_bottleneck_min500_"
    "release_l2_seed42_v1"
)
MIN_REFINEMENT_ALIGNED_VIEW1_PROTOCOL_ID = (
    "mvdec_2025_frozen_dekm_f_view1_true_bottleneck_centered_alignment_min500_"
    "release_l2_seed42_v1"
)
MIN_REFINEMENT_SCRATCH_PROTOCOL_ID = (
    "mvdec_2025_scratch_both_views_true_bottleneck_min500_release_l2_v1"
)
MIN_REFINEMENT_SCRATCH_CONCAT_PROTOCOL_ID = (
    "mvdec_2025_scratch_both_views_true_bottleneck_figure2_concat_min500_"
    "release_l2_v1"
)
MIN_REFINEMENT_WIDE_VIEW2_PROTOCOL_ID = (
    "mvdec_2025_frozen_dekm_f_view1_true_bottleneck_wide_view2_min500_"
    "release_l2_seed42_v1"
)
MIN_REFINEMENT_WIDE_VIEW2_BASE_UNITS = 256
MIN_REFINEMENT_EVEN_COLUMNS_PROTOCOL_ID = (
    "mvdec_2025_frozen_dekm_f_view1_true_bottleneck_even_columns_view2_min500_"
    "release_l2_seed42_v1"
)
MIN_REFINEMENT_TRAINING_STEPS = 500
INDEPENDENT_DEKM_VIEW2_PROTOCOL_ID = (
    "mvdec_2025_frozen_dekm_f_view1_independent_dekm_view2_release_l2_v1"
)
CALIBRATED_INDEPENDENT_DEKM_VIEW2_PROTOCOL_ID = (
    "mvdec_2025_frozen_dekm_f_view1_independent_dekm_view2_"
    "centered_rms_calibrated_release_l2_v1"
)
INDEPENDENT_DEKM_VIEW2_SEED_SWEEP_PROTOCOL_ID = (
    "mvdec_2025_frozen_dekm_f_s42_view1_independent_dekm_view2_seed_sweep_"
    "release_l2_v1"
)
INDEPENDENT_DEKM_VIEW2_PROTOCOL_IDS = frozenset(
    {
        INDEPENDENT_DEKM_VIEW2_PROTOCOL_ID,
        INDEPENDENT_DEKM_VIEW2_SEED_SWEEP_PROTOCOL_ID,
        CALIBRATED_INDEPENDENT_DEKM_VIEW2_PROTOCOL_ID,
    }
)
FROZEN_VIEW1_PROTOCOL_IDS = frozenset(
    {
        FROZEN_DEKM_F_VIEW1_RELEASE_L2_PROTOCOL_ID,
        IMPORTED_DEKM_F_VIEW1_JOINT_EQ11_RELEASE_L2_PROTOCOL_ID,
        MIN_REFINEMENT_FROZEN_VIEW1_PROTOCOL_ID,
        MIN_REFINEMENT_ALIGNED_VIEW1_PROTOCOL_ID,
        MIN_REFINEMENT_WIDE_VIEW2_PROTOCOL_ID,
        MIN_REFINEMENT_EVEN_COLUMNS_PROTOCOL_ID,
        *INDEPENDENT_DEKM_VIEW2_PROTOCOL_IDS,
    }
)
# Imported-View1 protocols whose View1 stays trainable; joint-transition
# validation applies only to these IDs.
IMPORTED_JOINT_VIEW1_PROTOCOL_IDS = frozenset(
    {MIN_REFINEMENT_JOINT_VIEW1_PROTOCOL_ID}
)
EXTERNAL_VIEW1_PROTOCOL_IDS = frozenset(
    {
        *FROZEN_VIEW1_PROTOCOL_IDS,
        *IMPORTED_JOINT_VIEW1_PROTOCOL_IDS,
    }
)
PAPER_INFORMED_FULL_BATCH_PROTOCOL_ID = "mvdec_2025_paper_informed_full_batch_v2"
ALIGNED_BALANCED_FULL_BATCH_PROTOCOL_ID = "mvdec_2025_aligned_balanced_full_batch_v1"
ALIGNED_BALANCED_WEAK_ALIGNMENT_FULL_BATCH_PROTOCOL_ID = (
    "mvdec_2025_aligned_balanced_alignment_005_full_batch_v1"
)
ALIGNED_BALANCED_NO_ALIGNMENT_FULL_BATCH_PROTOCOL_ID = (
    "mvdec_2025_aligned_balanced_alignment_000_full_batch_v1"
)
LEGACY_SCALED_WEAK_ALIGNMENT_FULL_BATCH_PROTOCOL_ID = (
    "mvdec_2025_legacy_scaled_alignment_005_full_batch_v1"
)
LEGACY_SCALED_ALIGNMENT_FULL_BATCH_PROTOCOL_ID = (
    "mvdec_2025_legacy_scaled_alignment_100_full_batch_v1"
)
LEGACY_SCALED_CENTERED_ALIGNMENT_FULL_BATCH_PROTOCOL_ID = (
    "mvdec_2025_legacy_scaled_centered_alignment_100_full_batch_v1"
)
LEGACY_SCALED_CENTERED_SCALE_MATCHED_ALIGNMENT_FULL_BATCH_PROTOCOL_ID = (
    "mvdec_2025_legacy_scaled_centered_alignment_10000_full_batch_v1"
)
ALIGNED_BALANCED_PROTOCOL_SETTINGS = {
    ALIGNED_BALANCED_FULL_BATCH_PROTOCOL_ID: 1.0,
    ALIGNED_BALANCED_WEAK_ALIGNMENT_FULL_BATCH_PROTOCOL_ID: 0.05,
    ALIGNED_BALANCED_NO_ALIGNMENT_FULL_BATCH_PROTOCOL_ID: 0.0,
    LEGACY_SCALED_WEAK_ALIGNMENT_FULL_BATCH_PROTOCOL_ID: 0.05,
    LEGACY_SCALED_ALIGNMENT_FULL_BATCH_PROTOCOL_ID: 1.0,
    LEGACY_SCALED_CENTERED_ALIGNMENT_FULL_BATCH_PROTOCOL_ID: 1.0,
    LEGACY_SCALED_CENTERED_SCALE_MATCHED_ALIGNMENT_FULL_BATCH_PROTOCOL_ID: 100.0,
    MIN_REFINEMENT_ALIGNED_VIEW1_PROTOCOL_ID: 1.0,
}
ALIGNED_BALANCED_PROTOCOL_LOSS_SCALE_POLICIES = {
    ALIGNED_BALANCED_FULL_BATCH_PROTOCOL_ID: "mean_squared_dimensions",
    ALIGNED_BALANCED_WEAK_ALIGNMENT_FULL_BATCH_PROTOCOL_ID: ("mean_squared_dimensions"),
    ALIGNED_BALANCED_NO_ALIGNMENT_FULL_BATCH_PROTOCOL_ID: ("mean_squared_dimensions"),
    LEGACY_SCALED_WEAK_ALIGNMENT_FULL_BATCH_PROTOCOL_ID: ("sum_squared_dimensions"),
    LEGACY_SCALED_ALIGNMENT_FULL_BATCH_PROTOCOL_ID: "sum_squared_dimensions",
    LEGACY_SCALED_CENTERED_ALIGNMENT_FULL_BATCH_PROTOCOL_ID: ("sum_squared_dimensions"),
    LEGACY_SCALED_CENTERED_SCALE_MATCHED_ALIGNMENT_FULL_BATCH_PROTOCOL_ID: (
        "sum_squared_dimensions"
    ),
    MIN_REFINEMENT_ALIGNED_VIEW1_PROTOCOL_ID: "sum_squared_dimensions",
}
ALIGNED_BALANCED_PROTOCOL_ALIGNMENT_MODES = {
    protocol_id: ALIGNMENT_UNCENTERED_ROW_L2
    for protocol_id in ALIGNED_BALANCED_PROTOCOL_SETTINGS
}
for protocol_id in (
    LEGACY_SCALED_CENTERED_ALIGNMENT_FULL_BATCH_PROTOCOL_ID,
    LEGACY_SCALED_CENTERED_SCALE_MATCHED_ALIGNMENT_FULL_BATCH_PROTOCOL_ID,
    MIN_REFINEMENT_ALIGNED_VIEW1_PROTOCOL_ID,
):
    ALIGNED_BALANCED_PROTOCOL_ALIGNMENT_MODES[protocol_id] = (
        ALIGNMENT_FULL_BATCH_CENTERED_ROW_L2
    )
ALIGNED_BALANCED_PROTOCOL_IDS = frozenset(ALIGNED_BALANCED_PROTOCOL_SETTINGS)

EQUATION11_PROTOCOL_SETTINGS = {
    LITERAL_EQUATION11_PROTOCOL_ID: ("largest", "frozen_snapshot", "fixed_256"),
    EQUATION11_SMALLEST_FROZEN_PROTOCOL_ID: (
        "smallest",
        "frozen_snapshot",
        "fixed_256",
    ),
    EQUATION11_LARGEST_SELECTED_ONLY_PROTOCOL_ID: (
        "largest",
        "selected_dimension_only",
        "fixed_256",
    ),
    EQUATION11_SMALLEST_SELECTED_ONLY_PROTOCOL_ID: (
        "smallest",
        "selected_dimension_only",
        "fixed_256",
    ),
    EQUATION11_LARGEST_FROZEN_FULL_BATCH_PROTOCOL_ID: (
        "largest",
        "frozen_snapshot",
        "full_dataset",
    ),
    EQUATION11_SMALLEST_FROZEN_FULL_BATCH_PROTOCOL_ID: (
        "smallest",
        "frozen_snapshot",
        "full_dataset",
    ),
    EQUATION11_LARGEST_SELECTED_ONLY_FULL_BATCH_PROTOCOL_ID: (
        "largest",
        "selected_dimension_only",
        "full_dataset",
    ),
    PAPER_RESOLVED_FULL_BATCH_PROTOCOL_ID: (
        "smallest",
        "selected_dimension_only",
        "full_dataset",
    ),
    PAPER_RESOLVED_RELEASE_L2_FULL_BATCH_PROTOCOL_ID: (
        "smallest",
        "selected_dimension_only",
        "full_dataset",
    ),
    FROZEN_DEKM_F_VIEW1_RELEASE_L2_PROTOCOL_ID: (
        "smallest",
        "selected_dimension_only",
        "full_dataset",
    ),
    IMPORTED_DEKM_F_VIEW1_JOINT_EQ11_RELEASE_L2_PROTOCOL_ID: (
        "smallest",
        "selected_dimension_only",
        "full_dataset",
    ),
    MIN_REFINEMENT_FROZEN_VIEW1_PROTOCOL_ID: (
        "smallest",
        "selected_dimension_only",
        "full_dataset",
    ),
    MIN_REFINEMENT_JOINT_VIEW1_PROTOCOL_ID: (
        "smallest",
        "selected_dimension_only",
        "full_dataset",
    ),
    MIN_REFINEMENT_EVEN_COLUMNS_PROTOCOL_ID: (
        "smallest",
        "selected_dimension_only",
        "full_dataset",
    ),
    MIN_REFINEMENT_WIDE_VIEW2_PROTOCOL_ID: (
        "smallest",
        "selected_dimension_only",
        "full_dataset",
    ),
    MIN_REFINEMENT_SCRATCH_PROTOCOL_ID: (
        "smallest",
        "selected_dimension_only",
        "full_dataset",
    ),
    MIN_REFINEMENT_SCRATCH_CONCAT_PROTOCOL_ID: (
        "smallest",
        "selected_dimension_only",
        "full_dataset",
    ),
    PAPER_INFORMED_FULL_BATCH_PROTOCOL_ID: (
        "smallest",
        "selected_dimension_only",
        "full_dataset",
    ),
}
EQUATION11_PROTOCOL_IDS = frozenset(EQUATION11_PROTOCOL_SETTINGS)


def balanced_batch_count(n_samples: int, batch_size: int) -> int:
    """Return the closest positive batch count to the requested batch size."""

    if n_samples < 1 or batch_size < 1:
        raise ValueError("n_samples and batch_size must be positive.")
    return max(1, (n_samples + batch_size // 2) // batch_size)
