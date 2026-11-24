"""Tests for the clustering helpers shared by the FP-Max study."""

import numpy as np
import pandas as pd

from pipeline import clustering


def test_minmax_scale_block_bounds_and_constant_column():
    df = pd.DataFrame({"a": [-3.0, 0.0, 3.0], "b": [5.0, 5.0, 5.0]})

    scaled = clustering.minmax_scale_block(df)

    assert scaled["a"].tolist() == [0.0, 0.5, 1.0]
    assert scaled["b"].tolist() == [0.0, 0.0, 0.0]


def test_run_intuitive_passes_mu_convention(monkeypatch):
    captured = {}
    real = clustering.IntuitiveKPrototypes

    def recording_model(**kwargs):
        captured.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(clustering, "IntuitiveKPrototypes", recording_model)
    continuous = pd.DataFrame({"x": [0.0, 0.1, 1.0, 1.1, 0.05, 1.05]})
    binary = pd.DataFrame({"f": [1, 1, 0, 0, 1, 0]})

    result = clustering.run_intuitive_kprototypes(
        continuous,
        binary,
        n_clusters=2,
        mu_convention="swapped",
        verbose=False,
    )

    assert captured["mu_convention"] == "swapped"
    assert result.mu_convention == "swapped"
    assert np.unique(result.labels).size == 2
