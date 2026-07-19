"""Tests for the canonical DEKM public dataset loader."""

import numpy as np
import pytest

from pipeline.public_data import load_dekm_public_dataset


def test_reuters_release_loader_preserves_rows_and_normalizes(tmp_path) -> None:
    """Reuters must come only from the release arrays in canonical order."""

    dataset_dir = tmp_path / "REUTERS"
    dataset_dir.mkdir()
    X = np.zeros((4, 2000), dtype=np.float32)
    X[:, 0] = [1.0, 2.0, 3.0, 4.0]
    X[:, 1] = [0.0, 2.0, 4.0, 6.0]
    y = np.array([0, 1, 2, 3], dtype=np.int32)
    np.save(dataset_dir / "10k_feature.npy", X)
    np.save(dataset_dir / "10k_target.npy", y)

    first = load_dekm_public_dataset("REUTERS", tmp_path)
    second = load_dekm_public_dataset("reuters", tmp_path)

    np.testing.assert_allclose(np.linalg.norm(first.X, axis=1), 1.0)
    np.testing.assert_array_equal(first.y, y)
    assert first.source_sha256 == second.source_sha256
    assert first.spec.n_clusters == 4


def test_20news_release_loader_uses_local_npz_files_only(tmp_path) -> None:
    """20NEWS must concatenate the exact local train/test release arrays."""

    dataset_dir = tmp_path / "20NEWS"
    dataset_dir.mkdir()
    train_X = np.zeros((10, 2000), dtype=np.float32)
    test_X = np.zeros((10, 2000), dtype=np.float32)
    train_X[:, 0] = 1.0
    test_X[:, 1] = 1.0
    np.savez(dataset_dir / "train_data.npz", train_X)
    np.savez(dataset_dir / "test_data.npz", test_X)
    np.savez(dataset_dir / "train_label.npz", np.arange(10))
    np.savez(dataset_dir / "test_label.npz", np.arange(10, 20))

    dataset = load_dekm_public_dataset("20NEWS", tmp_path)

    assert dataset.X.shape == (20, 2000)
    np.testing.assert_array_equal(dataset.y, np.arange(20))
    np.testing.assert_allclose(dataset.X[:10, 0], 1.0)
    np.testing.assert_allclose(dataset.X[10:, 1], 1.0)


def test_rcv1_loader_fails_fast_when_local_sklearn_cache_is_unreadable(
    tmp_path,
    monkeypatch,
) -> None:
    """The loader must not silently download or substitute another RCV1 copy."""

    dataset_dir = tmp_path / "RCV1"
    dataset_dir.mkdir()
    (dataset_dir / "test").write_text("0\n", encoding="utf-8")
    (dataset_dir / "validation").write_text("1\n", encoding="utf-8")

    def fail_fetch(*, download_if_missing):
        assert download_if_missing is False
        raise ValueError("corrupt cache")

    monkeypatch.setattr("sklearn.datasets.fetch_rcv1", fail_fetch)

    with pytest.raises(RuntimeError, match="cache is missing or unreadable"):
        load_dekm_public_dataset("RCV1", tmp_path)
