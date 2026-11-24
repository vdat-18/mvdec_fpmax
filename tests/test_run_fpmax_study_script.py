"""Tests for the study CLI helpers."""

import pytest

from pipeline.study_runner import shard_suffix
from scripts.run_fpmax_study import parse_seed_subset

ALLOWED = (0, 1, 2, 3)


def test_parse_seed_subset_accepts_configured_seeds():
    assert parse_seed_subset("0,3", ALLOWED) == (0, 3)
    assert shard_suffix((0, 3)) == "__seeds-0-3"


@pytest.mark.parametrize("text", ["9", "0,0", "", "a,1", "1,7"])
def test_parse_seed_subset_rejects_bad_input(text):
    with pytest.raises(ValueError, match="seeds"):
        parse_seed_subset(text, ALLOWED)
