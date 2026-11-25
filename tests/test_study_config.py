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

    assert study.report_seeds == tuple(range(10))
    assert study.max_itemsets == 60
    assert study.ffs_min_improvement == 0.005
    assert study.ffs_max_rows == 1500
    assert study.intuitive.mu_convention == "swapped"
    assert study.intuitive.restarts == 3
    assert study.intuitive.for_dataset("heart_cleveland_5").mu == 0.44
    assert study.intuitive.for_dataset("unseen").beta == 2.0
    assert set(study.applications) == {"tiki", "airpollution"}
    assert len(study.uci_datasets) == 8


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


def test_min_supports_must_be_strictly_descending(tmp_path):
    payload = _payload()
    payload["fpmax"]["min_supports"] = [0.2, 0.4]

    with pytest.raises(ValueError, match="descending"):
        load_study_config(_write(tmp_path, payload))


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("intuitive", "init_strategy"), "farthest-first", "init_strategy"),
        (("seeds", "report"), [-1, 1], "seeds"),
        (("seeds", "report"), [], "seeds"),
        (("intuitive", "restarts"), 1001, "restarts"),
        (("ffs", "max_candidates"), 0, "max_candidates"),
        (("ffs", "max_steps"), 0, "max_steps"),
        (("ffs", "max_rows"), 1, "max_rows"),
        (("data", "uci_datasets"), ["iris", "iris"], "unique"),
        (("data", "dekm", "datasets"), ["MNIST"], "dekm.datasets"),
        (("data", "dekm", "embedding_seed_offset"), -1, "embedding_seed_offset"),
        (("data", "dekm", "n_rows"), 5, "n_rows"),
    ],
)
def test_invalid_values_are_rejected(tmp_path, path, value, match):
    payload = _payload()
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValueError, match=match):
        load_study_config(_write(tmp_path, payload))


def test_per_dataset_typo_is_rejected(tmp_path):
    payload = _payload()
    payload["intuitive"]["per_dataset"]["irsi"] = {"beta": 2.0, "mu": 0.5, "gamma": 0.5}

    with pytest.raises(ValueError, match="unknown datasets"):
        load_study_config(_write(tmp_path, payload))


def test_cost_caps_are_loaded():
    study = load_study_config(DEFAULT_STUDY_CONFIG_PATH)

    assert study.ffs_max_candidates == 15
    assert study.ffs_max_steps == 5
    assert study.ffs_restarts == 1
    assert study.intuitive.restarts == 3


def test_ffs_restarts_cannot_exceed_intuitive_restarts(tmp_path):
    payload = _payload()
    payload["ffs"]["restarts"] = 9

    with pytest.raises(ValueError, match="restarts"):
        load_study_config(_write(tmp_path, payload))


def test_dekm_pairs_each_report_seed_with_one_embedding_seed():
    dekm = load_study_config(DEFAULT_STUDY_CONFIG_PATH).dekm

    assert dekm.study_names == ("dekm_reuters", "dekm_rcv1", "dekm_20news")
    assert dekm.bundle_name("dekm_reuters", 0) == "dekm_reuters__seed-42.npz"
    assert dekm.bundle_name("dekm_20news", 9) == "dekm_20news__seed-51.npz"
    with pytest.raises(ValueError, match="DEKM"):
        dekm.bundle_name("iris", 0)


def test_dataset_names_must_not_collide_across_sources(tmp_path):
    payload = _payload()
    payload["data"]["uci_datasets"].append("dekm_reuters")

    with pytest.raises(ValueError, match="unique across"):
        load_study_config(_write(tmp_path, payload))
