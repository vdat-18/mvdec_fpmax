"""Tests for safe legacy representation dataset loading."""

import numpy as np
import pytest
from scipy import sparse

from representation_learning.utils import get_xy


def _write_20news_files(root, *, object_data: bool = False) -> None:
    dataset_dir = root / "20NEWS"
    dataset_dir.mkdir()
    dtype = object if object_data else np.float64
    train = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=dtype)
    test = np.array([[3.0, 4.0]], dtype=dtype)
    np.savez(dataset_dir / "train_data.npz", train)
    np.savez(dataset_dir / "test_data.npz", test)
    np.savez(dataset_dir / "train_label.npz", np.array([0, 1]))
    np.savez(dataset_dir / "test_label.npz", np.array([1]))


def test_get_xy_loads_numeric_20news_npz_without_pickle(tmp_path) -> None:
    _write_20news_files(tmp_path)

    X, y = get_xy("20NEWS", f"{tmp_path}/", log_print=False)

    np.testing.assert_allclose(np.linalg.norm(X, axis=1), np.ones(3))
    assert sorted(y.tolist()) == [0, 1, 1]


def test_get_xy_rejects_pickled_object_20news_npz(tmp_path) -> None:
    _write_20news_files(tmp_path, object_data=True)

    with pytest.raises(ValueError, match="Object arrays cannot be loaded"):
        get_xy("20NEWS", f"{tmp_path}/", log_print=False)


def test_get_xy_does_not_hide_unexpected_sparse_loader_failures(
    tmp_path,
    monkeypatch,
) -> None:
    _write_20news_files(tmp_path)

    def fail_unexpectedly(_path):
        raise RuntimeError("unexpected sparse loader failure")

    monkeypatch.setattr(sparse, "load_npz", fail_unexpectedly)

    with pytest.raises(RuntimeError, match="unexpected sparse loader failure"):
        get_xy("20NEWS", f"{tmp_path}/", log_print=False)
