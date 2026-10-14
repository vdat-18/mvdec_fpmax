"""Versioned architecture identifiers shared by MvDEC runners and loaders."""

VIEW2_ARCHITECTURE_ID = "mvdec2025_post_skip_latent_bottleneck_v2"
VIEW2_ENCODER_BOTTLENECK_ARCHITECTURE_ID = (
    "mvdec2025_encoder_bottleneck_skip_decoder_v1"
)
VIEW2_DIRECT_JOINT_HEAD_ARCHITECTURE_ID = (
    "mvdec2025_post_skip_direct_latent_reconstruction_head_v1"
)
FUSION_ENCODER_AVERAGE = "mvdec2025_encoder_average"
FUSION_ENCODER_L2_NORMALIZED_AVERAGE = "mvdec2025_encoder_l2_normalized_average"
VIEW_INPUT_CONTRACT = "same_full_feature_matrix_both_views_v1"
REPRESENTATION_METRIC_SEMANTICS = "jointly_trained_views_independent_kmeans_v1"


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
PAPER_INFORMED_FULL_BATCH_PROTOCOL_ID = "mvdec_2025_paper_informed_full_batch_v2"
ALIGNED_BALANCED_FULL_BATCH_PROTOCOL_ID = (
    "mvdec_2025_aligned_balanced_full_batch_v1"
)
ALIGNED_BALANCED_WEAK_ALIGNMENT_FULL_BATCH_PROTOCOL_ID = (
    "mvdec_2025_aligned_balanced_alignment_005_full_batch_v1"
)
ALIGNED_BALANCED_NO_ALIGNMENT_FULL_BATCH_PROTOCOL_ID = (
    "mvdec_2025_aligned_balanced_alignment_000_full_batch_v1"
)
LEGACY_SCALED_WEAK_ALIGNMENT_FULL_BATCH_PROTOCOL_ID = (
    "mvdec_2025_legacy_scaled_alignment_005_full_batch_v1"
)
ALIGNED_BALANCED_PROTOCOL_SETTINGS = {
    ALIGNED_BALANCED_FULL_BATCH_PROTOCOL_ID: 1.0,
    ALIGNED_BALANCED_WEAK_ALIGNMENT_FULL_BATCH_PROTOCOL_ID: 0.05,
    ALIGNED_BALANCED_NO_ALIGNMENT_FULL_BATCH_PROTOCOL_ID: 0.0,
    LEGACY_SCALED_WEAK_ALIGNMENT_FULL_BATCH_PROTOCOL_ID: 0.05,
}
ALIGNED_BALANCED_PROTOCOL_LOSS_SCALE_POLICIES = {
    ALIGNED_BALANCED_FULL_BATCH_PROTOCOL_ID: "mean_squared_dimensions",
    ALIGNED_BALANCED_WEAK_ALIGNMENT_FULL_BATCH_PROTOCOL_ID: (
        "mean_squared_dimensions"
    ),
    ALIGNED_BALANCED_NO_ALIGNMENT_FULL_BATCH_PROTOCOL_ID: (
        "mean_squared_dimensions"
    ),
    LEGACY_SCALED_WEAK_ALIGNMENT_FULL_BATCH_PROTOCOL_ID: (
        "sum_squared_dimensions"
    ),
}
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
