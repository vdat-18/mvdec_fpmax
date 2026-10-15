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
    EQUATION11_PROTOCOL_IDS,
    EQUATION11_PROTOCOL_SETTINGS,
    FUSION_FIGURE1_FULL_OUTPUT_AVERAGE,
    FUSION_FIGURE2_FULL_OUTPUT_CONCATENATION,
    LEGACY_SCALED_ALIGNMENT_FULL_BATCH_PROTOCOL_ID,
    LEGACY_SCALED_CENTERED_ALIGNMENT_FULL_BATCH_PROTOCOL_ID,
    LEGACY_SCALED_CENTERED_SCALE_MATCHED_ALIGNMENT_FULL_BATCH_PROTOCOL_ID,
    LEGACY_SCALED_WEAK_ALIGNMENT_FULL_BATCH_PROTOCOL_ID,
    PAPER_INFORMED_FULL_BATCH_PROTOCOL_ID,
    PAPER_RESOLVED_FULL_BATCH_PROTOCOL_ID,
    VIEW2_DIRECT_JOINT_HEAD_ARCHITECTURE_ID,
    balanced_batch_count,
    paper_figure_fusion_metadata,
)


def test_equation11_contract_is_complete_two_by_two_by_two_grid() -> None:
    assert len(EQUATION11_PROTOCOL_IDS) == 9
    assert set(EQUATION11_PROTOCOL_SETTINGS.values()) == {
        (direction, target, batch_policy)
        for direction in ("largest", "smallest")
        for target in ("frozen_snapshot", "selected_dimension_only")
        for batch_policy in ("fixed_256", "full_dataset")
    }


def test_all_equation11_artifacts_have_loader_contracts() -> None:
    for protocol_id in EQUATION11_PROTOCOL_IDS:
        assert data.PUBLIC_PROTOCOL_ARCHITECTURES[protocol_id] == (
            VIEW2_DIRECT_JOINT_HEAD_ARCHITECTURE_ID
        )
        assert protocol_id in data.PUBLIC_PROTOCOL_METHODS
        assert protocol_id in data.PUBLIC_PROTOCOL_FUSIONS


def test_paper_informed_v2_preserves_historical_setting_without_overclaiming() -> None:
    assert (
        EQUATION11_PROTOCOL_SETTINGS[PAPER_INFORMED_FULL_BATCH_PROTOCOL_ID]
        == (EQUATION11_PROTOCOL_SETTINGS[PAPER_RESOLVED_FULL_BATCH_PROTOCOL_ID])
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
    }
    assert ALIGNED_BALANCED_PROTOCOL_LOSS_SCALE_POLICIES == {
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
        LEGACY_SCALED_ALIGNMENT_FULL_BATCH_PROTOCOL_ID: (
            "sum_squared_dimensions"
        ),
        LEGACY_SCALED_CENTERED_ALIGNMENT_FULL_BATCH_PROTOCOL_ID: (
            "sum_squared_dimensions"
        ),
        LEGACY_SCALED_CENTERED_SCALE_MATCHED_ALIGNMENT_FULL_BATCH_PROTOCOL_ID: (
            "sum_squared_dimensions"
        ),
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
    }
    for protocol_id in ALIGNED_BALANCED_PROTOCOL_SETTINGS:
        assert data.PUBLIC_PROTOCOL_ARCHITECTURES[protocol_id] == (
            VIEW2_DIRECT_JOINT_HEAD_ARCHITECTURE_ID
        )
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
    assert metadata["claim_scope"] == (
        "post_training_diagnostic_not_exact_paper_truth"
    )
