"""Tests for shared MvDEC architecture and ambiguity contracts."""

import pytest

from pipeline import data
from pipeline.mvdec_contract import (
    ALIGNED_BALANCED_FULL_BATCH_PROTOCOL_ID,
    ALIGNED_BALANCED_NO_ALIGNMENT_FULL_BATCH_PROTOCOL_ID,
    ALIGNED_BALANCED_PROTOCOL_ALIGNMENT_MODES,
    ALIGNED_BALANCED_PROTOCOL_LOSS_SCALE_POLICIES,
    ALIGNED_BALANCED_PROTOCOL_SETTINGS,
    ALIGNED_BALANCED_WEAK_ALIGNMENT_FULL_BATCH_PROTOCOL_ID,
    ALIGNMENT_FULL_BATCH_CENTERED_ROW_L2,
    ALIGNMENT_UNCENTERED_ROW_L2,
    CALIBRATED_INDEPENDENT_DEKM_VIEW2_PROTOCOL_ID,
    EQUATION11_PROTOCOL_IDS,
    EQUATION11_PROTOCOL_SETTINGS,
    FROZEN_DEKM_F_VIEW1_RELEASE_L2_PROTOCOL_ID,
    FROZEN_VIEW1_PROTOCOL_IDS,
    FROZEN_VIEW1_REPRESENTATION_METRIC_SEMANTICS,
    FUSION_ENCODER_CENTERED_RMS_CALIBRATED_AVERAGE,
    FUSION_FIGURE1_FULL_OUTPUT_AVERAGE,
    FUSION_FIGURE2_FULL_OUTPUT_CONCATENATION,
    IMPORTED_DEKM_F_VIEW1_JOINT_EQ11_RELEASE_L2_PROTOCOL_ID,
    MIN_REFINEMENT_ALIGNED_VIEW1_PROTOCOL_ID,
    MIN_REFINEMENT_FROZEN_VIEW1_PROTOCOL_ID,
    MIN_REFINEMENT_JOINT_VIEW1_PROTOCOL_ID,
    MIN_REFINEMENT_SCRATCH_PROTOCOL_ID,
    MIN_REFINEMENT_EVEN_COLUMNS_PROTOCOL_ID,
    MIN_REFINEMENT_WIDE_VIEW2_PROTOCOL_ID,
    MIN_REFINEMENT_TRAINING_STEPS,
    INDEPENDENT_DEKM_VIEW2_PROTOCOL_ID,
    LEGACY_SCALED_ALIGNMENT_FULL_BATCH_PROTOCOL_ID,
    LEGACY_SCALED_CENTERED_ALIGNMENT_FULL_BATCH_PROTOCOL_ID,
    LEGACY_SCALED_CENTERED_SCALE_MATCHED_ALIGNMENT_FULL_BATCH_PROTOCOL_ID,
    LEGACY_SCALED_WEAK_ALIGNMENT_FULL_BATCH_PROTOCOL_ID,
    PAPER_INFORMED_FULL_BATCH_PROTOCOL_ID,
    PAPER_RESOLVED_FULL_BATCH_PROTOCOL_ID,
    PAPER_RESOLVED_RELEASE_L2_FULL_BATCH_PROTOCOL_ID,
    VIEW1_REFINEMENT_FROZEN_DEKM_F,
    VIEW2_ARCHITECTURE_ID,
    VIEW2_EVEN_COLUMNS_ARCHITECTURE_ID,
    VIEW2_DIRECT_JOINT_HEAD_ARCHITECTURE_ID,
    balanced_batch_count,
    paper_figure_fusion_metadata,
)


def test_equation11_contract_is_complete_two_by_two_by_two_grid() -> None:
    assert (
        IMPORTED_DEKM_F_VIEW1_JOINT_EQ11_RELEASE_L2_PROTOCOL_ID
        in EQUATION11_PROTOCOL_IDS
    )
    assert set(EQUATION11_PROTOCOL_SETTINGS.values()) == {
        (direction, target, batch_policy)
        for direction in ("largest", "smallest")
        for target in ("frozen_snapshot", "selected_dimension_only")
        for batch_policy in ("fixed_256", "full_dataset")
    }


def test_all_equation11_artifacts_have_loader_contracts() -> None:
    for protocol_id in EQUATION11_PROTOCOL_IDS:
        expected_architecture = (
            VIEW2_EVEN_COLUMNS_ARCHITECTURE_ID
            if protocol_id == MIN_REFINEMENT_EVEN_COLUMNS_PROTOCOL_ID
            else VIEW2_ARCHITECTURE_ID
            if protocol_id
            in {
                IMPORTED_DEKM_F_VIEW1_JOINT_EQ11_RELEASE_L2_PROTOCOL_ID,
                MIN_REFINEMENT_FROZEN_VIEW1_PROTOCOL_ID,
                MIN_REFINEMENT_JOINT_VIEW1_PROTOCOL_ID,
                MIN_REFINEMENT_SCRATCH_PROTOCOL_ID,
                MIN_REFINEMENT_WIDE_VIEW2_PROTOCOL_ID,
            }
            else VIEW2_DIRECT_JOINT_HEAD_ARCHITECTURE_ID
        )
        assert data.PUBLIC_PROTOCOL_ARCHITECTURES[protocol_id] == expected_architecture
        assert protocol_id in data.PUBLIC_PROTOCOL_METHODS
        assert protocol_id in data.PUBLIC_PROTOCOL_FUSIONS


def test_paper_informed_v2_preserves_historical_setting_without_overclaiming() -> None:
    assert (
        EQUATION11_PROTOCOL_SETTINGS[PAPER_INFORMED_FULL_BATCH_PROTOCOL_ID]
        == (EQUATION11_PROTOCOL_SETTINGS[PAPER_RESOLVED_FULL_BATCH_PROTOCOL_ID])
    )


def test_release_l2_diagnostic_preserves_paper_resolved_schedule() -> None:
    assert (
        EQUATION11_PROTOCOL_SETTINGS[PAPER_RESOLVED_RELEASE_L2_FULL_BATCH_PROTOCOL_ID]
        == EQUATION11_PROTOCOL_SETTINGS[PAPER_RESOLVED_FULL_BATCH_PROTOCOL_ID]
    )


def test_frozen_dekm_f_view1_protocol_has_immutable_loader_contract() -> None:
    assert (
        EQUATION11_PROTOCOL_SETTINGS[FROZEN_DEKM_F_VIEW1_RELEASE_L2_PROTOCOL_ID]
        == EQUATION11_PROTOCOL_SETTINGS[
            PAPER_RESOLVED_RELEASE_L2_FULL_BATCH_PROTOCOL_ID
        ]
    )
    assert (
        data.PUBLIC_PROTOCOL_ARCHITECTURES[FROZEN_DEKM_F_VIEW1_RELEASE_L2_PROTOCOL_ID]
        == VIEW2_DIRECT_JOINT_HEAD_ARCHITECTURE_ID
    )
    assert FROZEN_DEKM_F_VIEW1_RELEASE_L2_PROTOCOL_ID in data.PUBLIC_PROTOCOL_METHODS
    assert FROZEN_DEKM_F_VIEW1_RELEASE_L2_PROTOCOL_ID in data.PUBLIC_PROTOCOL_FUSIONS
    assert VIEW1_REFINEMENT_FROZEN_DEKM_F == "frozen_imported_dekm_f_view1"
    assert FROZEN_VIEW1_REPRESENTATION_METRIC_SEMANTICS == (
        "frozen_dekm_f_view1_refined_view2_independent_kmeans_v1"
    )


def test_independent_dekm_view2_protocol_has_immutable_loader_contract() -> None:
    assert FROZEN_VIEW1_PROTOCOL_IDS == {
        FROZEN_DEKM_F_VIEW1_RELEASE_L2_PROTOCOL_ID,
        IMPORTED_DEKM_F_VIEW1_JOINT_EQ11_RELEASE_L2_PROTOCOL_ID,
        MIN_REFINEMENT_FROZEN_VIEW1_PROTOCOL_ID,
        MIN_REFINEMENT_ALIGNED_VIEW1_PROTOCOL_ID,
        MIN_REFINEMENT_WIDE_VIEW2_PROTOCOL_ID,
        MIN_REFINEMENT_EVEN_COLUMNS_PROTOCOL_ID,
        INDEPENDENT_DEKM_VIEW2_PROTOCOL_ID,
        CALIBRATED_INDEPENDENT_DEKM_VIEW2_PROTOCOL_ID,
    }
    assert (
        data.PUBLIC_PROTOCOL_ARCHITECTURES[INDEPENDENT_DEKM_VIEW2_PROTOCOL_ID]
        == VIEW2_ARCHITECTURE_ID
    )
    assert INDEPENDENT_DEKM_VIEW2_PROTOCOL_ID in data.PUBLIC_PROTOCOL_METHODS
    assert INDEPENDENT_DEKM_VIEW2_PROTOCOL_ID in data.PUBLIC_PROTOCOL_FUSIONS
    assert (
        data.PUBLIC_PROTOCOL_ARCHITECTURES[
            CALIBRATED_INDEPENDENT_DEKM_VIEW2_PROTOCOL_ID
        ]
        == VIEW2_ARCHITECTURE_ID
    )
    assert CALIBRATED_INDEPENDENT_DEKM_VIEW2_PROTOCOL_ID in data.PUBLIC_PROTOCOL_METHODS
    assert (
        data.PUBLIC_PROTOCOL_FUSIONS[CALIBRATED_INDEPENDENT_DEKM_VIEW2_PROTOCOL_ID]
        == FUSION_ENCODER_CENTERED_RMS_CALIBRATED_AVERAGE
    )


def test_aligned_balanced_protocols_have_public_loader_contracts() -> None:
    assert ALIGNED_BALANCED_PROTOCOL_SETTINGS == {
        ALIGNED_BALANCED_FULL_BATCH_PROTOCOL_ID: 1.0,
        ALIGNED_BALANCED_WEAK_ALIGNMENT_FULL_BATCH_PROTOCOL_ID: 0.05,
        ALIGNED_BALANCED_NO_ALIGNMENT_FULL_BATCH_PROTOCOL_ID: 0.0,
        LEGACY_SCALED_WEAK_ALIGNMENT_FULL_BATCH_PROTOCOL_ID: 0.05,
        LEGACY_SCALED_ALIGNMENT_FULL_BATCH_PROTOCOL_ID: 1.0,
        LEGACY_SCALED_CENTERED_ALIGNMENT_FULL_BATCH_PROTOCOL_ID: 1.0,
        LEGACY_SCALED_CENTERED_SCALE_MATCHED_ALIGNMENT_FULL_BATCH_PROTOCOL_ID: 100.0,
        MIN_REFINEMENT_ALIGNED_VIEW1_PROTOCOL_ID: 1.0,
    }
    assert ALIGNED_BALANCED_PROTOCOL_LOSS_SCALE_POLICIES == {
        ALIGNED_BALANCED_FULL_BATCH_PROTOCOL_ID: "mean_squared_dimensions",
        ALIGNED_BALANCED_WEAK_ALIGNMENT_FULL_BATCH_PROTOCOL_ID: (
            "mean_squared_dimensions"
        ),
        ALIGNED_BALANCED_NO_ALIGNMENT_FULL_BATCH_PROTOCOL_ID: (
            "mean_squared_dimensions"
        ),
        LEGACY_SCALED_WEAK_ALIGNMENT_FULL_BATCH_PROTOCOL_ID: ("sum_squared_dimensions"),
        LEGACY_SCALED_ALIGNMENT_FULL_BATCH_PROTOCOL_ID: ("sum_squared_dimensions"),
        LEGACY_SCALED_CENTERED_ALIGNMENT_FULL_BATCH_PROTOCOL_ID: (
            "sum_squared_dimensions"
        ),
        LEGACY_SCALED_CENTERED_SCALE_MATCHED_ALIGNMENT_FULL_BATCH_PROTOCOL_ID: (
            "sum_squared_dimensions"
        ),
        MIN_REFINEMENT_ALIGNED_VIEW1_PROTOCOL_ID: "sum_squared_dimensions",
    }
    assert ALIGNED_BALANCED_PROTOCOL_ALIGNMENT_MODES == {
        **{
            protocol_id: ALIGNMENT_UNCENTERED_ROW_L2
            for protocol_id in ALIGNED_BALANCED_PROTOCOL_SETTINGS
        },
        LEGACY_SCALED_CENTERED_ALIGNMENT_FULL_BATCH_PROTOCOL_ID: (
            ALIGNMENT_FULL_BATCH_CENTERED_ROW_L2
        ),
        LEGACY_SCALED_CENTERED_SCALE_MATCHED_ALIGNMENT_FULL_BATCH_PROTOCOL_ID: (
            ALIGNMENT_FULL_BATCH_CENTERED_ROW_L2
        ),
        MIN_REFINEMENT_ALIGNED_VIEW1_PROTOCOL_ID: (
            ALIGNMENT_FULL_BATCH_CENTERED_ROW_L2
        ),
    }
    for protocol_id in ALIGNED_BALANCED_PROTOCOL_SETTINGS:
        expected_architecture = (
            VIEW2_ARCHITECTURE_ID
            if protocol_id == MIN_REFINEMENT_ALIGNED_VIEW1_PROTOCOL_ID
            else VIEW2_DIRECT_JOINT_HEAD_ARCHITECTURE_ID
        )
        assert data.PUBLIC_PROTOCOL_ARCHITECTURES[protocol_id] == expected_architecture
        assert protocol_id in data.PUBLIC_PROTOCOL_METHODS
        assert protocol_id in data.PUBLIC_PROTOCOL_FUSIONS


def test_balanced_batch_count_matches_training_contract() -> None:
    assert balanced_batch_count(18_846, 256) == 74
    assert balanced_batch_count(3_765, 256) == 15
    with pytest.raises(ValueError, match="positive"):
        balanced_batch_count(0, 256)


@pytest.mark.parametrize(
    ("fusion_contract", "paper_source", "output_width"),
    [
        (
            FUSION_FIGURE1_FULL_OUTPUT_AVERAGE,
            "MvDEC 2025 Figure 1",
            23,
        ),
        (
            FUSION_FIGURE2_FULL_OUTPUT_CONCATENATION,
            "MvDEC 2025 Figure 2 final layer",
            46,
        ),
    ],
)
def test_paper_figure_fusion_metadata_is_explicitly_diagnostic(
    fusion_contract,
    paper_source,
    output_width,
) -> None:
    metadata = paper_figure_fusion_metadata(fusion_contract, 23)

    assert metadata["paper_source"] == paper_source
    assert metadata["input_width_per_view"] == 23
    assert metadata["output_width"] == output_width
    assert metadata["claim_scope"] == ("post_training_diagnostic_not_exact_paper_truth")
