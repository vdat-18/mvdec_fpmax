"""Tests for StudyData loaders."""

import json
import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from pipeline import study_data
from pipeline.study_data import (
    load_application_study_data,
    load_dekm_study_bundle,
    load_uci_study_data,
    tfidf_backbone,
    write_dekm_study_bundle,
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
    assert data.reference_metric == "euclidean"


def _dekm_fixture(tmp_path, monkeypatch, *, sha="sha", rows=6, sparse=False):
    X = np.arange(12, dtype=float).reshape(6, 2) + 1.0
    y = np.array([0, 0, 0, 1, 1, 1])
    dataset = SimpleNamespace(
        X=csr_matrix(X) if sparse else X,
        y=y,
        source_sha256="sha",
        spec=SimpleNamespace(name="REUTERS", n_clusters=2),
    )
    monkeypatch.setattr(
        study_data, "load_dekm_public_dataset", lambda *_a, **_k: dataset
    )
    artifact = {
        "embedding": np.arange(rows * 3, dtype=float).reshape(rows, 3),
        "source_sha256": sha,
        "random_seed": 42,
    }
    path = tmp_path / "artifact.pkl"
    with path.open("wb") as file:
        pickle.dump(artifact, file, protocol=4)
    return path, X, y


def test_dekm_bundle_round_trip_keeps_rows_aligned(tmp_path, monkeypatch):
    path, X, y = _dekm_fixture(tmp_path, monkeypatch)

    bundle = write_dekm_study_bundle(
        path, "REUTERS", tmp_path, tmp_path / "b.npz", n_rows=4
    )
    data = load_dekm_study_bundle(bundle, n_rows=4)
    rows = np.load(bundle)["row_index"]

    assert data.name == "dekm_reuters"
    assert data.backbone == "dekm"
    assert data.numeric.shape == (4, 3)
    embedding = np.arange(len(X) * 3, dtype=float).reshape(len(X), 3)
    np.testing.assert_allclose(data.numeric.to_numpy(), embedding[rows])
    assert data.reference_metric == "cosine"
    np.testing.assert_allclose(data.reference_df.to_numpy(), X[rows])
    assert data.labels.tolist() == y[rows].tolist()
    assert int(np.load(bundle)["embedding_seed"]) == 42


def test_dekm_bundle_rows_depend_only_on_row_seed(tmp_path, monkeypatch):
    path, _, _ = _dekm_fixture(tmp_path, monkeypatch)

    first = write_dekm_study_bundle(path, "REUTERS", tmp_path, tmp_path / "1.npz", 3)
    second = write_dekm_study_bundle(path, "REUTERS", tmp_path, tmp_path / "2.npz", 3)

    assert np.load(first)["row_index"].tolist() == np.load(second)["row_index"].tolist()


def test_dekm_bundle_accepts_sparse_tfidf(tmp_path, monkeypatch):
    path, X, _ = _dekm_fixture(tmp_path, monkeypatch, sparse=True)

    bundle = write_dekm_study_bundle(path, "REUTERS", tmp_path, tmp_path / "b.npz", 6)

    np.testing.assert_allclose(load_dekm_study_bundle(bundle).reference_df, X)


def test_dekm_bundle_rejects_source_mismatch(tmp_path, monkeypatch):
    path, _, _ = _dekm_fixture(tmp_path, monkeypatch, sha="other")

    with pytest.raises(ValueError, match="source"):
        write_dekm_study_bundle(path, "REUTERS", tmp_path, tmp_path / "b.npz", 4)


def test_dekm_bundle_rejects_row_count_mismatch(tmp_path, monkeypatch):
    path, _, _ = _dekm_fixture(tmp_path, monkeypatch, rows=5)

    with pytest.raises(ValueError, match="rows"):
        write_dekm_study_bundle(path, "REUTERS", tmp_path, tmp_path / "b.npz", 4)


def test_dekm_bundle_rejects_too_many_rows(tmp_path, monkeypatch):
    path, _, _ = _dekm_fixture(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="n_rows"):
        write_dekm_study_bundle(path, "REUTERS", tmp_path, tmp_path / "b.npz", 7)


def test_dekm_bundle_load_checks_expected_rows(tmp_path, monkeypatch):
    path, _, _ = _dekm_fixture(tmp_path, monkeypatch)
    bundle = write_dekm_study_bundle(path, "REUTERS", tmp_path, tmp_path / "b.npz", 4)

    with pytest.raises(ValueError, match="expects 5"):
        load_dekm_study_bundle(bundle, n_rows=5)


def test_tfidf_backbone_clusters_the_same_rows_on_their_tfidf(tmp_path, monkeypatch):
    path, X, y = _dekm_fixture(tmp_path, monkeypatch)
    bundle = write_dekm_study_bundle(path, "REUTERS", tmp_path, tmp_path / "b.npz", 4)
    dekm = load_dekm_study_bundle(bundle, n_rows=4)

    data = tfidf_backbone(dekm)

    assert data.name == "tfidf_reuters"
    assert data.backbone == "tfidf"
    np.testing.assert_allclose(data.numeric.to_numpy(), dekm.reference_df.to_numpy())
    assert data.labels.tolist() == dekm.labels.tolist()
    assert data.reference_metric == "cosine"
    np.testing.assert_allclose(
        data.reference_df.to_numpy(), dekm.reference_df.to_numpy()
    )


def test_tfidf_backbone_needs_dekm_rows(tmp_path):
    data = load_uci_study_data("toy", _write_uci(tmp_path, ["n1"], ["c1"]))

    with pytest.raises(ValueError, match="DEKM"):
        tfidf_backbone(data)
