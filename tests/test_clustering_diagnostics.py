"""Tests for Silhouette-only clustering diagnostics."""

import inspect

import numpy as np
from sklearn.metrics import silhouette_samples

import pipeline.clustering as clustering
import pipeline.experiments as experiments
import pipeline.forward_selection as forward_selection
from pipeline.clustering import compute_silhouette_diagnostics
from pipeline.experiments import serialize_cluster_assignments


def test_silhouette_diagnostics_match_per_sample_silhouettes() -> None:
    """Diagnostics must derive only from standard Silhouette samples."""

    points = np.array([[0.0], [1.0], [4.0], [5.0]])
    distance_matrix = np.abs(points - points.T)
    labels = np.array([0, 1, 0, 1])

    score, sample_std, negative_fraction = compute_silhouette_diagnostics(
        distance_matrix,
        labels,
    )
    samples = silhouette_samples(distance_matrix, labels, metric="precomputed")

    assert score == np.mean(samples)
    assert sample_std == np.std(samples)
    assert negative_fraction == np.mean(samples < 0.0)


def test_cluster_assignments_preserve_preprocessed_row_order() -> None:
    """Serialized labels must round-trip without changing row order."""

    assert serialize_cluster_assignments(np.array([2, 0, 1])) == "[2,0,1]"


def test_cluster_count_and_baseline_are_required_at_low_level_apis() -> None:
    """Pipeline APIs must not silently fall back to Tiki-specific defaults."""

    modules = (clustering, experiments, forward_selection)
    for module in modules:
        for value in vars(module).values():
            if not inspect.isfunction(value) or value.__module__ != module.__name__:
                continue
            parameters = inspect.signature(value).parameters
            for parameter_name in ("n_clusters", "baseline_score"):
                if parameter_name in parameters:
                    assert (
                        parameters[parameter_name].default is inspect.Parameter.empty
                    ), f"{value.__name__}.{parameter_name} still has a default"
