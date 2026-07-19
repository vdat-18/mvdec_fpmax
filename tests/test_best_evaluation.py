"""Tests for fixed-configuration multi-seed evaluation."""

import json
from types import SimpleNamespace

import pandas as pd

from pipeline.best_evaluation import (
    SelectedConfiguration,
    aggregate_seed_evaluation,
    evaluate_fixed_configuration,
    export_seed_interpretation,
    select_result_row,
)
from pipeline.clustering import (
    MIXED_DISTANCE_CONTRACT,
    SYMMETRIC_DISTANCE_CONTRACT,
)


def _configuration() -> SelectedConfiguration:
    return SelectedConfiguration(
        source_job_index=7,
        strategy="uniform",
        n_bins=3,
        min_support=0.2,
        selected_features=("pattern",),
        init="cao",
        mu_param=None,
        gamma=None,
        beta=None,
        alpha=None,
    )


def test_select_result_row_uses_requested_score_column(tmp_path) -> None:
    """Automatic selection must maximize only the explicit score column."""

    results_path = tmp_path / "results.csv"
    pd.DataFrame(
        [
            {"job_index": 1, "status": "ok", "final_score": 0.4},
            {"job_index": 2, "status": "ok", "final_score": 0.7},
            {"job_index": 3, "status": "failed", "final_score": 0.9},
        ]
    ).to_csv(results_path, index=False)

    selected = select_result_row(results_path, "final_score")

    assert int(selected["job_index"]) == 2


def test_evaluate_fixed_configuration_outputs_both_distance_contracts() -> None:
    """Each seed must be scored with symmetric and asymmetric Silhouette."""

    h_fused_df = pd.DataFrame(
        {
            "x": [0.0, 0.1, 0.2, 4.8, 4.9, 5.0],
            "y": [0.0, 0.1, 0.0, 5.0, 4.9, 5.0],
        }
    )
    binary_df = pd.DataFrame({"pattern": [1, 1, 1, 0, 0, 0]})

    runs = evaluate_fixed_configuration(
        backend="kprototypes",
        h_fused_df=h_fused_df,
        binary_df=binary_df,
        mvdec_labels=pd.Series([0, 1, 0, 1, 0, 1]).to_numpy(),
        configuration=_configuration(),
        n_clusters=2,
        seeds=(11, 12),
        true_labels=pd.Series([0, 0, 0, 1, 1, 1]).to_numpy(),
    )

    assert len(runs) == 4
    assert set(runs["distance_contract"]) == {
        MIXED_DISTANCE_CONTRACT,
        SYMMETRIC_DISTANCE_CONTRACT,
    }
    assert set(runs["status"]) == {"ok"}
    assert set(runs["random_seed"]) == {11, 12}
    assert set(runs["source_job_index"]) == {7}
    assert set(runs["selected_features"]) == {"pattern"}
    assert (runs["silhouette_delta_vs_mvdec"] > 0).all()
    assert (runs["acc"] > runs["mvdec_reference_acc"]).all()
    assert (runs["nmi"] > runs["mvdec_reference_nmi"]).all()
    assert (
        runs["silhouette_delta_vs_mvdec"]
        == runs["silhouette_score"] - runs["mvdec_reference_silhouette_score"]
    ).all()

    summary = aggregate_seed_evaluation(runs)
    assert set(summary["successful_runs"]) == {2}
    assert set(summary["failed_runs"]) == {0}
    assert set(summary["source_job_index"]) == {7}
    assert (
        summary["silhouette_delta_vs_mvdec_mean"]
        == summary["silhouette_mean"] - summary["mvdec_reference_silhouette_score"]
    ).all()
    assert (summary["acc_delta_vs_mvdec_mean"] > 0).all()
    assert (summary["nmi_delta_vs_mvdec_mean"] > 0).all()


def test_export_seed_interpretation_joins_mapping_and_profiles(tmp_path) -> None:
    """One selected seed must produce row-level and cluster-profile outputs."""

    data_path = tmp_path / "data.csv"
    mapping_path = tmp_path / "mapping.csv"
    output_path = tmp_path / "interpretation.csv"
    profile_path = tmp_path / "profiles.csv"
    pd.DataFrame({"scaled": [0.1, 0.2, 0.8]}).to_csv(data_path, index=False)
    pd.DataFrame(
        {
            "csv_row_index": [0, 1, 2],
            "Store Name": ["A", "B", "C"],
            "Revenue": [10.0, 20.0, 100.0],
        }
    ).to_csv(mapping_path, index=False)
    runs = pd.DataFrame(
        [
            {
                "random_seed": 42,
                "distance_contract": MIXED_DISTANCE_CONTRACT,
                "status": "ok",
                "cluster_assignments": json.dumps([0, 0, 1]),
            }
        ]
    )
    mvdec = SimpleNamespace(h_fused_df=pd.DataFrame({"fused_1": [0.0, 0.1, 1.0]}))

    export_seed_interpretation(
        runs=runs,
        seed=42,
        mvdec=mvdec,
        data_path=data_path,
        mapping_path=mapping_path,
        output_path=output_path,
        profile_output_path=profile_path,
        backend="kprototypes",
        configuration=_configuration(),
    )

    interpretation = pd.read_csv(output_path)
    profiles = pd.read_csv(profile_path)
    assert interpretation["Store Name"].tolist() == ["A", "B", "C"]
    assert interpretation["cluster"].tolist() == [0, 0, 1]
    assert "preprocessed_scaled" in interpretation.columns
    assert set(profiles["cluster"]) == {0, 1}
