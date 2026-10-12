"""Tests for shared MvDEC architecture and ambiguity contracts."""

from pipeline import data
from pipeline.mvdec_contract import (
    EQUATION11_PROTOCOL_IDS,
    EQUATION11_PROTOCOL_SETTINGS,
    VIEW2_DIRECT_JOINT_HEAD_ARCHITECTURE_ID,
)


def test_equation11_contract_is_complete_two_by_two_by_two_grid() -> None:
    assert len(EQUATION11_PROTOCOL_IDS) == 8
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
