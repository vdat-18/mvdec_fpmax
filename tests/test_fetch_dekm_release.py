"""Tests for resilient retrieval of the official DEKM release files."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts/fetch_dekm_release.py"
SPEC = importlib.util.spec_from_file_location("fetch_dekm_release", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
fetch_dekm_release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fetch_dekm_release)


class FailingGdown:
    """Simulate Google Drive throttling without making a network request."""

    @staticmethod
    def download(**_kwargs):
        raise RuntimeError("rate limited")


def test_rcv1_bundled_fallbacks_have_canonical_hashes() -> None:
    for filename in ("test", "validation"):
        path = fetch_dekm_release.BUNDLED_FALLBACK_ROOT / "RCV1" / filename
        expected = fetch_dekm_release.CANONICAL_SHA256[("RCV1", filename)]
        assert fetch_dekm_release.sha256(path) == expected


def test_download_file_uses_verified_fallback(tmp_path) -> None:
    fallback = fetch_dekm_release.BUNDLED_FALLBACK_ROOT / "RCV1" / "test"
    destination = tmp_path / "RCV1" / "test"

    retrieval = fetch_dekm_release.download_file(
        FailingGdown(),
        "unused-file-id",
        destination,
        False,
        fallback_path=fallback,
        expected_sha256=fetch_dekm_release.CANONICAL_SHA256[("RCV1", "test")],
    )

    assert retrieval == "bundled_canonical_copy"
    assert destination.read_bytes() == fallback.read_bytes()


def test_download_file_rejects_unverified_fallback(tmp_path) -> None:
    fallback = tmp_path / "bad-fallback"
    fallback.write_text("corrupt", encoding="ascii")

    with pytest.raises(RuntimeError, match="failed verification"):
        fetch_dekm_release.download_file(
            FailingGdown(),
            "unused-file-id",
            tmp_path / "output",
            False,
            fallback_path=fallback,
            expected_sha256="0" * 64,
        )
