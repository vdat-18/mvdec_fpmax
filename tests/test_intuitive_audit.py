"""Regression tests for Intuitive weights and persisted model diagnostics."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

import pipeline.clustering as clustering
from intuitive_kprototypes.model import _inverse_phi_weights
from pipeline.experiments import (
    INTUITIVE_MODEL_AUDIT_COLUMNS,
    POST_FFS_INTUITIVE_COLUMNS,
    ExperimentJob,
    IntuitiveParams,
    PostFfsIntuitiveContext,
    make_post_ffs_intuitive_record,
)


def test_zero_phi_receives_zero_weight_as_printed_in_paper() -> None:
    """Eq. (16)/(22) excludes zero-phi attributes from normalization."""

    weights = _inverse_phi_weights(np.array([0.0, 0.5, 1.0]), beta=2.0)

    np.testing.assert_allclose(weights, [0.0, 2.0 / 3.0, 1.0 / 3.0])
    np.testing.assert_allclose(
        _inverse_phi_weights(np.zeros(3), beta=2.0),
        np.zeros(3),
    )


def test_intuitive_preserves_raw_latent_and_exposes_final_phi_weights(
    monkeypatch,
) -> None:
    """The wrapper keeps fused latent values unchanged and names diagnostics."""

    captured: dict[str, np.ndarray] = {}

    class FakeIntuitiveKPrototypes:
        def __init__(self, **_kwargs) -> None:
            pass

        def fit(self, X_num, X_cat):
            captured["X_num"] = X_num.copy()
            captured["X_cat"] = X_cat.copy()
            labels = np.array([0, 0, 0, 1, 1, 1])
            self.result_ = SimpleNamespace(
                labels=labels,
                weights=SimpleNamespace(
                    numeric_phi=np.array([0.25, 0.75]),
                    categorical_phi=np.array([0.5]),
                    numeric=np.array([0.75, 0.25]),
                    categorical=np.array([1.0]),
                    numeric_scaled=np.array([1.5, 0.5]),
                    categorical_scaled=np.array([1.0]),
                ),
                distances=np.zeros((len(labels), 2)),
                fit_time_seconds=0.1,
                converged=True,
                n_iter=2,
                initial_prototype_indices=np.array([0, 3]),
            )
            return self

    monkeypatch.setattr(clustering, "IntuitiveKPrototypes", FakeIntuitiveKPrototypes)
    monkeypatch.setattr(
        clustering,
        "compute_silhouette_diagnostics",
        lambda _distance, _labels: (0.4, 0.1, 0.0),
    )
    continuous_df = pd.DataFrame(
        {
            "latent_view_1": [-3.0, -2.0, -1.0, 5.0, 6.0, 7.0],
            "latent_view_2": [10.0, 9.0, 8.0, 2.0, 1.0, 0.0],
        }
    )
    binary_df = pd.DataFrame({"pattern_1": [0, 0, 0, 1, 1, 1]})

    result = clustering.run_intuitive_kprototypes(
        continuous_df=continuous_df,
        binary_df=binary_df,
        n_clusters=2,
        distance_matrix=np.zeros((6, 6)),
        verbose=False,
    )

    np.testing.assert_array_equal(captured["X_num"], continuous_df.to_numpy())
    assert result.final_numeric_phi == {
        "latent_view_1": 0.25,
        "latent_view_2": 0.75,
    }
    assert result.final_categorical_weights == {"pattern_1": 1.0}
    assert result.final_numeric_weights_scaled == {
        "latent_view_1": 1.5,
        "latent_view_2": 0.5,
    }


def test_post_ffs_record_persists_final_phi_weights_as_json() -> None:
    """Every post-FFS result row must retain interpretable model diagnostics."""

    model_result = SimpleNamespace(
        score=0.4,
        cluster_sizes=[3, 3],
        init_strategy="farthest_first",
        fit_time_seconds=0.1,
        labels_hash="labels",
        converged=True,
        n_iter=2,
        initial_prototype_indices=[0, 3],
        final_cost=1.5,
        silhouette_sample_std=0.1,
        silhouette_negative_fraction=0.0,
        labels=np.array([0, 0, 0, 1, 1, 1]),
        view_weighted_score=None,
        final_numeric_phi={"latent_view_1": 0.25},
        final_categorical_phi={"pattern_1": 0.5},
        final_numeric_weights={"latent_view_1": 1.0},
        final_categorical_weights={"pattern_1": 1.0},
        final_numeric_weights_scaled={"latent_view_1": 1.0},
        final_categorical_weights_scaled={"pattern_1": 1.0},
    )
    context = PostFfsIntuitiveContext(
        h_fused_df=pd.DataFrame({"latent_view_1": range(6)}),
        binary_df=pd.DataFrame({"pattern_1": [0, 0, 0, 1, 1, 1]}),
        distance_matrix=np.zeros((6, 6)),
        view_weighted_distance_matrices=None,
        ffs_job_index=7,
        strategy="quantile",
        n_bins=3,
        min_support=0.1,
        ffs_score=0.3,
        selected_features="pattern_1",
        n_selected_features=1,
        n_clusters=2,
        random_state=44,
    )
    record = make_post_ffs_intuitive_record(
        job=ExperimentJob(
            job_index=0,
            min_support=0.1,
            intuitive_params=IntuitiveParams(mu_param=0.5, gamma=0.5, beta=2.0),
        ),
        context=context,
        clustering=model_result,
        status="ok",
        error_message=None,
    )

    assert set(INTUITIVE_MODEL_AUDIT_COLUMNS).issubset(POST_FFS_INTUITIVE_COLUMNS)
    assert json.loads(record.final_numeric_phi) == {"latent_view_1": 0.25}
    assert json.loads(record.final_categorical_weights) == {"pattern_1": 1.0}
