"""Versioned architecture identifiers shared by MvDEC runners and loaders."""

VIEW2_ARCHITECTURE_ID = "mvdec2025_post_skip_latent_bottleneck_v2"
VIEW2_ENCODER_BOTTLENECK_ARCHITECTURE_ID = (
    "mvdec2025_encoder_bottleneck_skip_decoder_v1"
)
VIEW2_DIRECT_JOINT_HEAD_ARCHITECTURE_ID = (
    "mvdec2025_post_skip_direct_latent_reconstruction_head_v1"
)
FUSION_ENCODER_AVERAGE = "mvdec2025_encoder_average"
FUSION_ENCODER_L2_NORMALIZED_AVERAGE = (
    "mvdec2025_encoder_l2_normalized_average"
)


LITERAL_EQUATION11_PROTOCOL_ID = "mvdec_2025_equation11_literal_v1"
EQUATION11_SMALLEST_FROZEN_PROTOCOL_ID = (
    "mvdec_2025_equation11_smallest_frozen_v1"
)
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

EQUATION11_PROTOCOL_SETTINGS = {
    LITERAL_EQUATION11_PROTOCOL_ID: ("largest", "frozen_snapshot", "fixed_256"),
    EQUATION11_SMALLEST_FROZEN_PROTOCOL_ID: (
        "smallest", "frozen_snapshot", "fixed_256"
    ),
    EQUATION11_LARGEST_SELECTED_ONLY_PROTOCOL_ID: (
        "largest", "selected_dimension_only", "fixed_256"
    ),
    EQUATION11_SMALLEST_SELECTED_ONLY_PROTOCOL_ID: (
        "smallest", "selected_dimension_only", "fixed_256"
    ),
    EQUATION11_LARGEST_FROZEN_FULL_BATCH_PROTOCOL_ID: (
        "largest", "frozen_snapshot", "full_dataset"
    ),
    EQUATION11_SMALLEST_FROZEN_FULL_BATCH_PROTOCOL_ID: (
        "smallest", "frozen_snapshot", "full_dataset"
    ),
    EQUATION11_LARGEST_SELECTED_ONLY_FULL_BATCH_PROTOCOL_ID: (
        "largest", "selected_dimension_only", "full_dataset"
    ),
    PAPER_RESOLVED_FULL_BATCH_PROTOCOL_ID: (
        "smallest", "selected_dimension_only", "full_dataset"
    ),
}
EQUATION11_PROTOCOL_IDS = frozenset(EQUATION11_PROTOCOL_SETTINGS)
