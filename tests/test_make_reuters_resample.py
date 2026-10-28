import importlib.util
import io
import json
from pathlib import Path

import numpy as np

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "make_reuters_resample.py"
)
SPEC = importlib.util.spec_from_file_location("make_reuters_resample", SCRIPT_PATH)
reuters = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reuters)


def test_single_label_documents_keeps_one_root_category_only():
    qrels = [
        "CCAT 1 1",
        "GCAT 2 1",
        "ECAT 2 1",
        "MCAT 3 1",
        "C15 3 1",
        "E11 4 1",
    ]
    assert reuters.single_label_documents(qrels) == {1: 0, 3: 2}


def test_iter_documents_parses_lyrl_token_format():
    lines = [".I 7\n", ".W\n", "alpha beta\n", "gamma\n", ".I 9\n", ".W\n", "delta\n"]
    assert list(reuters.iter_documents(lines)) == [
        (7, "alpha beta\ngamma\n"),
        (9, "delta\n"),
    ]


def test_dense_sample_scales_l2_rows_by_sqrt_feature_count():
    docs = ["alpha beta gamma", "alpha alpha delta", "beta gamma delta", "eps zeta"]
    tfidf = reuters.tfidf_features(docs)
    x, y = reuters.dense_sample(tfidf, np.array([0, 1, 2, 3]), np.array([2, 0]))
    np.testing.assert_allclose(
        np.linalg.norm(x, axis=1), np.sqrt(reuters.N_FEATURES), rtol=1e-6
    )
    assert y.tolist() == [2, 0]


def test_wayback_snapshot_uses_first_successful_capture(monkeypatch):
    rows = [["urlkey", "timestamp", "original"], ["k", "20120101000000", "x"]]
    requested = []

    def fake_urlopen(url, timeout):
        requested.append(url)
        return io.BytesIO(json.dumps(rows).encode())

    monkeypatch.setattr(reuters.urllib.request, "urlopen", fake_urlopen)
    assert reuters._wayback_snapshot("jmlr.org/a.gz") == (
        "https://web.archive.org/web/20120101000000id_/http://jmlr.org/a.gz"
    )
    assert "statuscode%3A200" in requested[0]


def test_wayback_snapshot_returns_none_without_capture(monkeypatch):
    monkeypatch.setattr(
        reuters.urllib.request, "urlopen", lambda url, timeout: io.BytesIO(b"[]")
    )
    assert reuters._wayback_snapshot("jmlr.org/a.gz") is None


def test_canonical_rows_permute_the_whole_corpus_before_taking_the_sample():
    rows = reuters.canonical_rows(reuters.N_SAMPLES + 500)
    assert len(rows) == reuters.N_SAMPLES
    assert len(set(rows.tolist())) == reuters.N_SAMPLES
    assert rows.max() >= reuters.N_SAMPLES
    np.testing.assert_array_equal(rows, reuters.canonical_rows(reuters.N_SAMPLES + 500))
