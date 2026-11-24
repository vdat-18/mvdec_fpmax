"""Tests for StudyData loaders."""

import json
import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pipeline import study_data
from pipeline.study_data import (
    load_application_study_data,
    load_dekm_study_data,
    load_uci_study_data,
)


def _write_uci(tmp_path, numeric_cols, categorical_cols):
    processed = tmp_path / "processed"
    processed.mkdir()
    frame = pd.DataFrame(
        {
            "n1": [0.0, 2.0, 4.0, 10.0],
            "n2": [5.0, 5.0, 5.0, 5.0],
            "c1": ["a", "b", "a", "b"],
            "target": ["x", "x", "y", "y"],
        }
    )
    frame.to_csv(processed / "toy.csv", index=False)
    manifest = [
        {
            "key": "toy",
            "n_clusters": 2,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "target_col": "target",
            "processed_csv": "processed\\toy.csv",
        }
    ]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def test_uci_mixed_dataset_splits_and_scales(tmp_path):
    data = load_uci_study_data("toy", _write_uci(tmp_path, ["n1", "n2"], ["c1"]))

    assert data.numeric["n1"].tolist() == [0.0, 0.2, 0.4, 1.0]
    assert data.numeric["n2"].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert data.categorical["c1"].tolist() == ["a", "b", "a", "b"]
    assert len(data.labels) == 4
    assert list(data.reference_df.columns) == ["n1", "n2", "c1"]
    assert data.reference_metric == "gower"
    assert data.backbone == "raw"


def test_uci_categorical_only_has_empty_numeric_block(tmp_path):
    data = load_uci_study_data("toy", _write_uci(tmp_path, [], ["c1"]))

    assert data.numeric.shape == (4, 0)
    assert data.categorical.shape == (4, 1)


def test_uci_unknown_key_is_rejected(tmp_path):
    _write_uci(tmp_path, ["n1"], [])

    with pytest.raises(ValueError, match="missing"):
        load_uci_study_data("nope", tmp_path)


def test_application_data_is_unlabeled(tmp_path):
    path = tmp_path / "app.csv"
    pd.DataFrame({"a": [1.0, 3.0, 5.0], "b": [-2.0, 0.0, 2.0]}).to_csv(
        path, index=False
    )

    data = load_application_study_data("app", path, 2)

    assert data.labels is None
    assert data.numeric.to_numpy().min() == 0.0
    assert data.numeric.to_numpy().max() == 1.0
    assert data.categorical.shape == (3, 0)


def _dekm_fixture(tmp_path, monkeypatch, *, sha="sha", rows=6):
    X = np.arange(12, dtype=float).reshape(6, 2) + 1.0
    y = np.array([0, 0, 0, 1, 1, 1])
    dataset = SimpleNamespace(
        X=X,
        y=y,
        source_sha256="sha",
        spec=SimpleNamespace(n_clusters=2),
    )
    monkeypatch.setattr(
        study_data, "load_dekm_public_dataset", lambda *_a, **_k: dataset
    )
    artifact = {
        "embedding": np.arange(rows * 3, dtype=float).reshape(rows, 3),
        "source_sha256": sha,
    }
    path = tmp_path / "artifact.pkl"
    with path.open("wb") as file:
        pickle.dump(artifact, file, protocol=4)
    return path


def test_dekm_loader_uses_embedding_and_cosine_reference(tmp_path, monkeypatch):
    path = _dekm_fixture(tmp_path, monkeypatch)

    data = load_dekm_study_data(path, "REUTERS", tmp_path)

    assert data.backbone == "dekm"
    assert data.numeric.shape == (6, 3)
    assert data.numeric.to_numpy().max() == 1.0
    assert data.reference_metric == "cosine"
    assert data.reference_df.shape == (6, 2)
    assert data.labels.tolist() == [0, 0, 0, 1, 1, 1]


def test_dekm_loader_rejects_source_mismatch(tmp_path, monkeypatch):
    path = _dekm_fixture(tmp_path, monkeypatch, sha="other")

    with pytest.raises(ValueError, match="source"):
        load_dekm_study_data(path, "REUTERS", tmp_path)


def test_dekm_loader_rejects_row_count_mismatch(tmp_path, monkeypatch):
    path = _dekm_fixture(tmp_path, monkeypatch, rows=5)

    with pytest.raises(ValueError, match="rows"):
        load_dekm_study_data(path, "REUTERS", tmp_path)
