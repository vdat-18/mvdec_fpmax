"""Tests for the FP-Max study configuration."""

import json

import pytest

from pipeline.study_config import DEFAULT_STUDY_CONFIG_PATH, load_study_config


def _payload() -> dict:
    return json.loads(DEFAULT_STUDY_CONFIG_PATH.read_text(encoding="utf-8"))


def _write(tmp_path, payload):
    path = tmp_path / "study.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_default_config_loads_and_matches_constraints():
    study = load_study_config(DEFAULT_STUDY_CONFIG_PATH)

    assert study.selection_seeds == (100, 101, 102)
    assert study.report_seeds == tuple(range(10))
    assert study.max_itemsets == 60
    assert study.ffs_min_improvement == 0.005
    assert study.ffs_max_rows == 1500
    assert study.intuitive.mu_convention == "swapped"
    assert study.intuitive.restarts == 5
    assert study.intuitive.for_dataset("heart_cleveland_5").mu == 0.44
    assert study.intuitive.for_dataset("unseen").beta == 2.0
    assert set(study.applications) == {"tiki", "airpollution"}
    assert len(study.uci_datasets) == 8


def test_seed_sets_must_be_disjoint(tmp_path):
    payload = _payload()
    payload["seeds"]["selection"] = [0, 100]

    with pytest.raises(ValueError, match="disjoint"):
        load_study_config(_write(tmp_path, payload))


def test_unknown_or_missing_fields_rejected(tmp_path):
    payload = _payload()
    payload["extra"] = 1

    with pytest.raises(ValueError, match="root"):
        load_study_config(_write(tmp_path, payload))


def test_strategy_kmeans_is_rejected(tmp_path):
    payload = _payload()
    payload["fpmax"]["strategies"] = ["kmeans"]

    with pytest.raises(ValueError, match="strategies"):
        load_study_config(_write(tmp_path, payload))


def test_invalid_dataset_params_are_rejected(tmp_path):
    payload = _payload()
    payload["intuitive"]["per_dataset"]["iris"]["beta"] = 1.0

    with pytest.raises(ValueError, match="beta"):
        load_study_config(_write(tmp_path, payload))
