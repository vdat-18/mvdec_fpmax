"""Tests for fail-closed public dataset downloads."""

import hashlib
import io

import numpy as np
import pytest

from scripts import fetch_public_datasets as fetcher


def test_download_file_verifies_checksum_before_atomic_replace(
    tmp_path,
    monkeypatch,
) -> None:
    payload = b"verified source bytes"
    expected = hashlib.sha256(payload).hexdigest()
    output_path = tmp_path / "source.bin"
    monkeypatch.setattr(
        fetcher, "urlopen", lambda *_args, **_kwargs: io.BytesIO(payload)
    )

    fetcher.download_file(
        "https://example.invalid/source.bin",
        output_path,
        force=False,
        expected_sha256=expected,
    )

    assert output_path.read_bytes() == payload


def test_download_file_rejects_tampered_cached_source(tmp_path, monkeypatch) -> None:
    output_path = tmp_path / "source.bin"
    output_path.write_bytes(b"tampered")
    monkeypatch.setattr(
        fetcher,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("cache mismatch must fail closed"),
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        fetcher.download_file(
            "https://example.invalid/source.bin",
            output_path,
            force=False,
            expected_sha256=hashlib.sha256(b"expected").hexdigest(),
        )


def test_download_file_keeps_existing_source_when_replacement_hash_fails(
    tmp_path,
    monkeypatch,
) -> None:
    output_path = tmp_path / "source.bin"
    output_path.write_bytes(b"previous source")
    monkeypatch.setattr(
        fetcher,
        "urlopen",
        lambda *_args, **_kwargs: io.BytesIO(b"untrusted replacement"),
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        fetcher.download_file(
            "https://example.invalid/source.bin",
            output_path,
            force=True,
            expected_sha256=hashlib.sha256(b"expected replacement").hexdigest(),
        )

    assert output_path.read_bytes() == b"previous source"
    assert not output_path.with_name("source.bin.tmp").exists()


def test_reuters10k_uses_revision_pinned_checksum_verified_sources(
    tmp_path,
    monkeypatch,
) -> None:
    raw_dir = tmp_path / "reuters10k"
    raw_dir.mkdir()
    payload = {
        "data": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        "label": np.array(["a", "b"]),
    }
    np.save(raw_dir / "train.npy", payload, allow_pickle=True)
    np.save(raw_dir / "test.npy", payload, allow_pickle=True)
    calls = []

    def capture_download(url, path, force, expected_sha256=None):
        calls.append((url, path.name, force, expected_sha256))

    monkeypatch.setattr(fetcher, "download_file", capture_download)

    dataset = fetcher.parse_reuters10k(raw_dir, force=False)

    assert dataset.X.shape == (2, 2)
    assert any(fetcher.REUTERS10K_REVISION in note for note in dataset.notes)
    assert any(fetcher.REUTERS10K_SHA256["train.npy"] in note for note in dataset.notes)
    assert calls == [
        (
            "https://huggingface.co/datasets/wwydmanski/reuters10k/resolve/"
            f"{fetcher.REUTERS10K_REVISION}/train.npy",
            "train.npy",
            False,
            fetcher.REUTERS10K_SHA256["train.npy"],
        ),
        (
            "https://huggingface.co/datasets/wwydmanski/reuters10k/resolve/"
            f"{fetcher.REUTERS10K_REVISION}/test.npy",
            "test.npy",
            False,
            fetcher.REUTERS10K_SHA256["test.npy"],
        ),
    ]
