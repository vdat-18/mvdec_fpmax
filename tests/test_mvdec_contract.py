"""Tests for shared MvDEC architecture and ambiguity contracts."""

import pytest

from pipeline import data
from pipeline.mvdec_contract import (
    EQUATION11_PROTOCOL_IDS,
    EQUATION11_PROTOCOL_SETTINGS,
    PAPER_INFORMED_FULL_BATCH_PROTOCOL_ID,
    PAPER_RESOLVED_FULL_BATCH_PROTOCOL_ID,
    VIEW2_DIRECT_JOINT_HEAD_ARCHITECTURE_ID,
    balanced_batch_count,
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


def test_balanced_batch_count_matches_training_contract() -> None:
    assert balanced_batch_count(18_846, 256) == 74
    assert balanced_batch_count(3_765, 256) == 15
    with pytest.raises(ValueError, match="positive"):
        balanced_batch_count(0, 256)
