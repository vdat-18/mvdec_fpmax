"""Tests for the DEKM-release K-Means benchmark runner."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _load_public_kmeans():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_public_kmeans_baseline.py"
    )
    spec = importlib.util.spec_from_file_location("public_kmeans", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import public K-Means runner: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_kmeans_baseline_records_seed_assignments_and_acc(monkeypatch) -> None:
    """One public K-Means run must expose row labels and external metrics."""

    public_kmeans = _load_public_kmeans()
    monkeypatch.setattr(public_kmeans, "N_INIT", 2)
    inputs = public_kmeans.DatasetInputs(
        name="TEST",
        title="Test",
        n_clusters=2,
        X=pd.DataFrame(
            {
                "x": [0.0, 0.1, 0.2, 4.8, 4.9, 5.0],
                "y": [0.0, 0.1, 0.0, 5.0, 4.9, 5.0],
            }
        ),
        y=pd.DataFrame({"label": [0, 0, 0, 1, 1, 1]}),
        metadata={},
        source_sha256="e" * 64,
    )

    result, assignments = public_kmeans.run_kmeans_baseline(
        inputs,
        "k-means++",
        random_state=17,
    )

    assert result.random_state == 17
    assert result.acc == 1.0
    assert result.nmi == 1.0
    assert assignments["random_seed"].unique().tolist() == [17]
    assert assignments["sample_index"].tolist() == list(range(6))
