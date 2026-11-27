"""Tests for the study CLI helpers."""

import pytest

from pipeline.study_config import DEFAULT_STUDY_CONFIG_PATH, load_study_config
from pipeline.study_runner import shard_suffix
from scripts import run_fpmax_study
from scripts.run_fpmax_study import load_named_dataset, parse_seed_subset

ALLOWED = (0, 1, 2, 3)


def test_parse_seed_subset_accepts_configured_seeds():
    assert parse_seed_subset("0,3", ALLOWED) == (0, 3)
    assert shard_suffix((0, 3)) == "__seeds-0-3"


@pytest.mark.parametrize("text", ["9", "0,0", "", "a,1", "1,7"])
def test_parse_seed_subset_rejects_bad_input(text):
    with pytest.raises(ValueError, match="seeds"):
        parse_seed_subset(text, ALLOWED)


def test_dekm_dataset_needs_a_single_seed():
    study = load_study_config(DEFAULT_STUDY_CONFIG_PATH)

    with pytest.raises(ValueError, match="one seed at a time"):
        load_named_dataset("dekm_reuters", study)


def test_dekm_dataset_loads_the_bundle_of_its_embedding_seed(monkeypatch):
    study = load_study_config(DEFAULT_STUDY_CONFIG_PATH)
    seen = {}

    def fake_load(path, n_rows=None):
        seen.update(path=path, n_rows=n_rows)
        return "data"

    monkeypatch.setattr(run_fpmax_study, "load_dekm_study_bundle", fake_load)

    assert load_named_dataset("dekm_rcv1", study, 3) == "data"
    assert seen["path"].name == "dekm_rcv1__seed-45.npz"
    assert seen["n_rows"] == 2000


def test_tfidf_flag_runs_kmeans_on_the_tfidf_rows(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(run_fpmax_study, "load_named_dataset", lambda *a: "dekm rows")
    monkeypatch.setattr(
        run_fpmax_study, "tfidf_backbone", lambda data: f"tfidf of {data}"
    )

    def fake_run_group(data, algorithm, study, out_dir, stem_suffix=""):
        seen.update(data=data, algorithm=algorithm)
        return tmp_path

    monkeypatch.setattr(run_fpmax_study, "run_group", fake_run_group)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run",
            "--dataset",
            "dekm_reuters",
            "--algorithm",
            "kmeans",
            "--seeds",
            "2",
            "--out-dir",
            str(tmp_path),
            "--tfidf",
        ],
    )

    run_fpmax_study.main()

    assert seen == {"data": "tfidf of dekm rows", "algorithm": "kmeans"}


def test_tfidf_flag_is_kmeans_only(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run",
            "--dataset",
            "dekm_reuters",
            "--algorithm",
            "kprototypes",
            "--out-dir",
            str(tmp_path),
            "--tfidf",
        ],
    )

    with pytest.raises(SystemExit):
        run_fpmax_study.main()
