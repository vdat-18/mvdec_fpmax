import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "mvdec_fusion_weight_sweep", SCRIPTS / "mvdec_fusion_weight_sweep.py"
)
sweep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sweep)


def test_unit_rms_centres_and_scales():
    h = np.random.default_rng(0).normal(3.0, 5.0, size=(50, 4))
    scaled = sweep.unit_rms(h)
    np.testing.assert_allclose(scaled.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.mean(scaled**2), 1.0)


def test_weighted_concat_matches_weighted_squared_distance():
    rng = np.random.default_rng(1)
    h1, h2 = rng.normal(size=(2, 3)), rng.normal(size=(2, 5))
    fused = sweep.fuse(h1, h2, 0.3, "concat")
    expected = 0.3 * np.sum((h1[0] - h1[1]) ** 2) + 0.7 * np.sum((h2[0] - h2[1]) ** 2)
    np.testing.assert_allclose(np.sum((fused[0] - fused[1]) ** 2), expected)


def test_sweep_endpoints_equal_single_views(tmp_path):
    rng = np.random.default_rng(2)
    y = np.repeat([0, 1], 20)
    h1 = rng.normal(size=(40, 2)) + y[:, None] * 6.0
    h2 = rng.normal(size=(40, 2))
    run = tmp_path / "seed_42"
    run.mkdir()
    art = {
        "true_labels": y,
        "n_clusters": 2,
        "config": {"seed": 42},
        "dataset": "REUTERS",
        "h_view1": h1,
        "h_view2": h2,
    }
    (run / "artifact.pkl").write_bytes(pickle.dumps(art))
    rows = sweep.sweep_run(run / "artifact.pkl", np.array([0.0, 1.0]), 5)["rows"]
    by_weight = {(r["fusion"], r["w_view1"]): r for r in rows}
    assert by_weight[("average", 1.0)]["acc"] == 1.0
    assert by_weight[("concat", 1.0)]["acc"] == 1.0
    assert by_weight[("average", 0.0)]["acc"] < 1.0
