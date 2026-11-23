import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "mvdec_view2_latent_probe", SCRIPTS / "mvdec_view2_latent_probe.py"
)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def _two_blob_latents(n: int = 60):
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], n // 2)
    informative = rng.normal(size=(n, 3)) + 4.0 * y[:, None]
    noise = rng.normal(size=(n, 5))
    return informative, noise, y


def test_probe_seed_reports_view_scores_and_a_label_free_midpoint():
    h1, h2, y = _two_blob_latents()
    result = probe.probe_seed(h1, h2, y, seed=0, steps=5)

    assert result["view1"]["acc"] > result["view2"]["acc"]
    assert result["label_free"]["w_view1"] == probe.LABEL_FREE_WEIGHT
    assert len(result["rows"]) == 5
    best = result["best_nmi_label_guided"]
    assert best["nmi"] == max(row["nmi"] for row in result["rows"])
