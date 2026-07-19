"""Validate one selected experiment configuration across seeds and distances."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from config import FUSED_REPRESENTATION_PATH, PREPROCESSED_DATA_PATH
from pipeline.clustering import (
    MIXED_DISTANCE_CONTRACT,
    SYMMETRIC_DISTANCE_CONTRACT,
    compute_mixed_gower_distance,
    compute_silhouette_diagnostics,
    compute_symmetric_mixed_gower_distance,
    run_intuitive_kprototypes,
    run_kprototypes,
)
from pipeline.data import MvdecResult, load_mvdec_result
from pipeline.experiment_context import artifact_n_clusters, build_experiment_context
from pipeline.experiments import (
    parse_selected_features,
    resolve_rebuilt_feature_names,
    serialize_cluster_assignments,
)
from pipeline.external_metrics import compute_external_metrics
from pipeline.fpmax import BIN_LABELS_BY_SIZE, extract_fpmax_features
from pipeline.interpretation import attach_row_mapping, build_cluster_profiles
from pipeline.io import replace_with_retry

EVALUATION_CONTRACTS = (
    MIXED_DISTANCE_CONTRACT,
    SYMMETRIC_DISTANCE_CONTRACT,
)


@dataclass(frozen=True)
class SelectedConfiguration:
    """Validated configuration reconstructed from one summary CSV row."""

    source_job_index: int
    strategy: str
    n_bins: int
    min_support: float
    selected_features: tuple[str, ...]
    init: str | None
    mu_param: float | None
    gamma: float | None
    beta: float | None
    alpha: float | None


def _optional_float(row: pd.Series, column: str) -> float | None:
    """Return one finite optional float from a result row."""

    if column not in row or pd.isna(row[column]):
        return None
    value = float(row[column])
    if not np.isfinite(value):
        msg = f"Result column must be finite when provided: {column}."
        raise ValueError(msg)
    return value


def select_result_row(
    results_path: Path,
    score_column: str,
    job_index: int | None = None,
) -> pd.Series:
    """Return one successful row by job index or maximum requested score."""

    results = pd.read_csv(results_path)
    required = {"job_index", "status", score_column}
    missing = sorted(required - set(results.columns))
    if missing:
        msg = f"Result CSV is missing required columns: {missing}."
        raise ValueError(msg)

    successful = results[results["status"] == "ok"].copy()
    if job_index is not None:
        selected = successful[successful["job_index"].astype(int) == job_index]
        if len(selected) != 1:
            msg = f"Expected one successful result for job_index={job_index}."
            raise ValueError(msg)
        return selected.iloc[0]

    successful[score_column] = pd.to_numeric(
        successful[score_column],
        errors="coerce",
    )
    successful = successful[np.isfinite(successful[score_column])]
    if successful.empty:
        msg = f"No successful finite rows are available for {score_column}."
        raise ValueError(msg)
    return successful.loc[successful[score_column].idxmax()]


def parse_selected_configuration(row: pd.Series) -> SelectedConfiguration:
    """Parse the persisted fields needed to reproduce one clustering run."""

    required = {"job_index", "strategy", "n_bins", "min_support", "selected_features"}
    missing = sorted(required - set(row.index))
    if missing:
        msg = f"Selected result row is missing configuration fields: {missing}."
        raise ValueError(msg)

    selected_features = tuple(parse_selected_features(row["selected_features"]))
    if not selected_features:
        msg = "Selected result row has no FP-Max features to evaluate."
        raise ValueError(msg)

    init = None
    if "init" in row and not pd.isna(row["init"]):
        init = str(row["init"])
    return SelectedConfiguration(
        source_job_index=int(row["job_index"]),
        strategy=str(row["strategy"]),
        n_bins=int(row["n_bins"]),
        min_support=float(row["min_support"]),
        selected_features=selected_features,
        init=init,
        mu_param=_optional_float(row, "mu_param"),
        gamma=_optional_float(row, "gamma"),
        beta=_optional_float(row, "beta"),
        alpha=_optional_float(row, "alpha"),
    )


def rebuild_selected_binary_features(
    h_fused_df: pd.DataFrame,
    configuration: SelectedConfiguration,
) -> pd.DataFrame:
    """Rebuild and select the exact persisted FP-Max feature set."""

    if configuration.n_bins not in BIN_LABELS_BY_SIZE:
        msg = f"Unsupported n_bins in selected result: {configuration.n_bins}."
        raise ValueError(msg)
    fpmax_features = extract_fpmax_features(
        df=h_fused_df,
        n_bins=configuration.n_bins,
        bin_labels=BIN_LABELS_BY_SIZE[configuration.n_bins],
        strategy=configuration.strategy,
        min_support=configuration.min_support,
        drop_original_numeric=True,
    )
    rebuilt_names = resolve_rebuilt_feature_names(
        selected_feature_names=list(configuration.selected_features),
        rebuilt_feature_names=fpmax_features.features.columns.tolist(),
    )
    return fpmax_features.features[rebuilt_names]


def _cluster_for_seed(
    *,
    backend: str,
    h_fused_df: pd.DataFrame,
    binary_df: pd.DataFrame,
    configuration: SelectedConfiguration,
    n_clusters: int,
    seed: int,
    init_strategy: str,
    strict_init: bool,
) -> np.ndarray:
    """Re-run one fixed feature/parameter configuration for one seed."""

    if backend == "kprototypes":
        if configuration.init is None:
            msg = "K-Prototypes evaluation requires the persisted init value."
            raise ValueError(msg)
        result = run_kprototypes(
            continuous_df=h_fused_df,
            binary_df=binary_df,
            n_clusters=n_clusters,
            init_methods=(configuration.init,),
            random_state=seed,
            verbose=False,
        )
        return result.labels

    if backend == "intuitive":
        params = (
            configuration.mu_param,
            configuration.gamma,
            configuration.beta,
        )
        if any(value is None for value in params):
            msg = "Intuitive evaluation requires mu_param, gamma, and beta."
            raise ValueError(msg)
        result = run_intuitive_kprototypes(
            continuous_df=h_fused_df,
            binary_df=binary_df,
            n_clusters=n_clusters,
            random_state=seed,
            mu_param=configuration.mu_param,
            gamma=configuration.gamma,
            beta=configuration.beta,
            view_weight_alpha=configuration.alpha,
            init_strategy=init_strategy,
            strict_init=strict_init,
            verbose=False,
        )
        return result.labels

    msg = f"Unsupported evaluation backend: {backend!r}."
    raise ValueError(msg)


def _configuration_metadata(
    configuration: SelectedConfiguration,
) -> dict[str, object]:
    """Return persisted fields that identify one fixed configuration."""

    return {
        "source_job_index": configuration.source_job_index,
        "strategy": configuration.strategy,
        "n_bins": configuration.n_bins,
        "min_support": configuration.min_support,
        "n_selected_features": len(configuration.selected_features),
        "selected_features": ", ".join(configuration.selected_features),
        "init": configuration.init,
        "mu_param": configuration.mu_param,
        "gamma": configuration.gamma,
        "beta": configuration.beta,
        "alpha": configuration.alpha,
    }


def evaluate_fixed_configuration(
    *,
    backend: str,
    h_fused_df: pd.DataFrame,
    binary_df: pd.DataFrame,
    mvdec_labels: np.ndarray,
    configuration: SelectedConfiguration,
    n_clusters: int,
    seeds: tuple[int, ...],
    true_labels: np.ndarray | None = None,
    init_strategy: str = "farthest_first",
    strict_init: bool = False,
) -> pd.DataFrame:
    """Evaluate one fixed clustering configuration across seeds and contracts."""

    if not seeds or len(set(seeds)) != len(seeds):
        msg = "Seeds must be a non-empty list of unique integers."
        raise ValueError(msg)

    distance_matrices = {
        MIXED_DISTANCE_CONTRACT: compute_mixed_gower_distance(
            h_fused_df,
            binary_df,
        ),
        SYMMETRIC_DISTANCE_CONTRACT: compute_symmetric_mixed_gower_distance(
            h_fused_df,
            binary_df,
        ),
    }
    mvdec_labels = np.asarray(mvdec_labels, dtype=int)
    if mvdec_labels.ndim != 1 or len(mvdec_labels) != len(h_fused_df):
        msg = "MvDEC reference labels must match the selected representation rows."
        raise ValueError(msg)
    if set(np.unique(mvdec_labels)) != set(range(n_clusters)):
        msg = "MvDEC reference labels must use every cluster ID from 0 to K - 1."
        raise ValueError(msg)
    mvdec_sizes = np.bincount(mvdec_labels, minlength=n_clusters).tolist()
    if any(size == 1 for size in mvdec_sizes):
        msg = "MvDEC reference labels must contain all clusters without singletons."
        raise ValueError(msg)
    true_labels_array = None
    reference_external = {
        "mvdec_reference_acc": np.nan,
        "mvdec_reference_nmi": np.nan,
    }
    if true_labels is not None:
        true_labels_array = np.asarray(true_labels)
        if true_labels_array.ndim != 1 or len(true_labels_array) != len(h_fused_df):
            msg = "True labels must match the selected representation rows."
            raise ValueError(msg)
        reference_metrics = compute_external_metrics(
            true_labels_array,
            mvdec_labels,
            n_clusters=n_clusters,
        )
        reference_external = {
            "mvdec_reference_acc": reference_metrics.acc,
            "mvdec_reference_nmi": reference_metrics.nmi,
        }
    mvdec_assignments = serialize_cluster_assignments(mvdec_labels)
    reference_by_contract: dict[str, dict[str, object]] = {}
    for contract, distance_matrix in distance_matrices.items():
        reference_score, reference_std, reference_negative = (
            compute_silhouette_diagnostics(distance_matrix, mvdec_labels)
        )
        reference_by_contract[contract] = {
            **reference_external,
            "mvdec_reference_silhouette_score": reference_score,
            "mvdec_reference_sample_std": reference_std,
            "mvdec_reference_negative_fraction": reference_negative,
            "mvdec_reference_cluster_sizes": json.dumps(
                mvdec_sizes,
                separators=(",", ":"),
            ),
            "mvdec_reference_cluster_assignments": mvdec_assignments,
        }
    configuration_metadata = _configuration_metadata(configuration)
    records: list[dict[str, object]] = []
    for seed in seeds:
        try:
            labels = _cluster_for_seed(
                backend=backend,
                h_fused_df=h_fused_df,
                binary_df=binary_df,
                configuration=configuration,
                n_clusters=n_clusters,
                seed=seed,
                init_strategy=init_strategy,
                strict_init=strict_init,
            )
        except ValueError as error:
            for contract in EVALUATION_CONTRACTS:
                records.append(
                    {
                        **configuration_metadata,
                        "random_seed": seed,
                        "backend": backend,
                        "distance_contract": contract,
                        **reference_by_contract[contract],
                        "silhouette_score": np.nan,
                        "silhouette_delta_vs_mvdec": np.nan,
                        "silhouette_sample_std": np.nan,
                        "silhouette_negative_fraction": np.nan,
                        "acc": np.nan,
                        "nmi": np.nan,
                        "acc_delta_vs_mvdec": np.nan,
                        "nmi_delta_vs_mvdec": np.nan,
                        "cluster_sizes": "[]",
                        "cluster_assignments": "",
                        "status": "failed_value_error",
                        "error_message": str(error),
                    }
                )
            continue

        sizes = np.bincount(labels.astype(int), minlength=n_clusters).tolist()
        assignments = serialize_cluster_assignments(labels)
        if any(size == 1 for size in sizes):
            for contract in EVALUATION_CONTRACTS:
                records.append(
                    {
                        **configuration_metadata,
                        "random_seed": seed,
                        "backend": backend,
                        "distance_contract": contract,
                        **reference_by_contract[contract],
                        "silhouette_score": np.nan,
                        "silhouette_delta_vs_mvdec": np.nan,
                        "silhouette_sample_std": np.nan,
                        "silhouette_negative_fraction": np.nan,
                        "acc": np.nan,
                        "nmi": np.nan,
                        "acc_delta_vs_mvdec": np.nan,
                        "nmi_delta_vs_mvdec": np.nan,
                        "cluster_sizes": json.dumps(
                            sizes,
                            separators=(",", ":"),
                        ),
                        "cluster_assignments": assignments,
                        "status": "failed_singleton_cluster",
                        "error_message": (
                            "Singleton clusters are not valid segmentation results."
                        ),
                    }
                )
            continue

        for contract, distance_matrix in distance_matrices.items():
            reference_score = float(
                reference_by_contract[contract]["mvdec_reference_silhouette_score"]
            )
            try:
                score, sample_std, negative_fraction = compute_silhouette_diagnostics(
                    distance_matrix,
                    labels,
                )
            except ValueError as error:
                records.append(
                    {
                        **configuration_metadata,
                        "random_seed": seed,
                        "backend": backend,
                        "distance_contract": contract,
                        **reference_by_contract[contract],
                        "silhouette_score": np.nan,
                        "silhouette_delta_vs_mvdec": np.nan,
                        "silhouette_sample_std": np.nan,
                        "silhouette_negative_fraction": np.nan,
                        "acc": np.nan,
                        "nmi": np.nan,
                        "acc_delta_vs_mvdec": np.nan,
                        "nmi_delta_vs_mvdec": np.nan,
                        "cluster_sizes": json.dumps(
                            sizes,
                            separators=(",", ":"),
                        ),
                        "cluster_assignments": assignments,
                        "status": "failed_silhouette",
                        "error_message": str(error),
                    }
                )
                continue
            external_fields = {
                "acc": np.nan,
                "nmi": np.nan,
                "acc_delta_vs_mvdec": np.nan,
                "nmi_delta_vs_mvdec": np.nan,
            }
            if true_labels_array is not None:
                external_metrics = compute_external_metrics(
                    true_labels_array,
                    labels,
                    n_clusters=n_clusters,
                )
                external_fields = {
                    "acc": external_metrics.acc,
                    "nmi": external_metrics.nmi,
                    "acc_delta_vs_mvdec": (
                        external_metrics.acc - reference_external["mvdec_reference_acc"]
                    ),
                    "nmi_delta_vs_mvdec": (
                        external_metrics.nmi - reference_external["mvdec_reference_nmi"]
                    ),
                }
            records.append(
                {
                    **configuration_metadata,
                    "random_seed": seed,
                    "backend": backend,
                    "distance_contract": contract,
                    **reference_by_contract[contract],
                    "silhouette_score": score,
                    "silhouette_delta_vs_mvdec": score - reference_score,
                    "silhouette_sample_std": sample_std,
                    "silhouette_negative_fraction": negative_fraction,
                    **external_fields,
                    "cluster_sizes": json.dumps(sizes, separators=(",", ":")),
                    "cluster_assignments": assignments,
                    "status": "ok",
                    "error_message": None,
                }
            )
    return pd.DataFrame.from_records(records)


def aggregate_seed_evaluation(runs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate successful Silhouette results by distance contract."""

    records: list[dict[str, object]] = []
    metadata_columns = [
        "dataset",
        "method",
        "source_sha256",
        "source_job_index",
        "backend",
        "strategy",
        "n_bins",
        "min_support",
        "n_selected_features",
        "selected_features",
        "init",
        "mu_param",
        "gamma",
        "beta",
        "alpha",
    ]
    metadata = {
        column: runs.iloc[0][column]
        for column in metadata_columns
        if column in runs.columns
    }
    for contract in EVALUATION_CONTRACTS:
        contract_runs = runs[runs["distance_contract"] == contract]
        successful = contract_runs[contract_runs["status"] == "ok"]
        scores = pd.to_numeric(successful["silhouette_score"], errors="coerce").dropna()
        deltas = pd.to_numeric(
            successful["silhouette_delta_vs_mvdec"],
            errors="coerce",
        ).dropna()
        acc_values = pd.to_numeric(successful["acc"], errors="coerce").dropna()
        nmi_values = pd.to_numeric(successful["nmi"], errors="coerce").dropna()
        acc_deltas = pd.to_numeric(
            successful["acc_delta_vs_mvdec"],
            errors="coerce",
        ).dropna()
        nmi_deltas = pd.to_numeric(
            successful["nmi_delta_vs_mvdec"],
            errors="coerce",
        ).dropna()
        reference_row = contract_runs.iloc[0]
        records.append(
            {
                **metadata,
                "distance_contract": contract,
                "mvdec_reference_silhouette_score": reference_row[
                    "mvdec_reference_silhouette_score"
                ],
                "mvdec_reference_sample_std": reference_row[
                    "mvdec_reference_sample_std"
                ],
                "mvdec_reference_negative_fraction": reference_row[
                    "mvdec_reference_negative_fraction"
                ],
                "mvdec_reference_acc": reference_row["mvdec_reference_acc"],
                "mvdec_reference_nmi": reference_row["mvdec_reference_nmi"],
                "requested_runs": len(contract_runs),
                "successful_runs": len(scores),
                "failed_runs": len(contract_runs) - len(scores),
                "silhouette_mean": scores.mean(),
                "silhouette_std_across_seeds": (
                    scores.std(ddof=1) if len(scores) > 1 else 0.0
                ),
                "silhouette_min": scores.min(),
                "silhouette_max": scores.max(),
                "silhouette_delta_vs_mvdec_mean": deltas.mean(),
                "silhouette_delta_vs_mvdec_std": (
                    deltas.std(ddof=1) if len(deltas) > 1 else 0.0
                ),
                "silhouette_delta_vs_mvdec_min": deltas.min(),
                "silhouette_delta_vs_mvdec_max": deltas.max(),
                "acc_mean": acc_values.mean(),
                "acc_std_across_seeds": (
                    acc_values.std(ddof=1) if len(acc_values) > 1 else 0.0
                ),
                "acc_delta_vs_mvdec_mean": acc_deltas.mean(),
                "nmi_mean": nmi_values.mean(),
                "nmi_std_across_seeds": (
                    nmi_values.std(ddof=1) if len(nmi_values) > 1 else 0.0
                ),
                "nmi_delta_vs_mvdec_mean": nmi_deltas.mean(),
                "sample_std_mean": pd.to_numeric(
                    successful["silhouette_sample_std"],
                    errors="coerce",
                ).mean(),
                "negative_fraction_mean": pd.to_numeric(
                    successful["silhouette_negative_fraction"],
                    errors="coerce",
                ).mean(),
            }
        )
    return pd.DataFrame.from_records(records)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Atomically write one evaluation artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    frame.to_csv(tmp_path, index=False)
    replace_with_retry(tmp_path, path)


def export_seed_interpretation(
    *,
    runs: pd.DataFrame,
    seed: int,
    mvdec: MvdecResult,
    data_path: Path,
    mapping_path: Path,
    output_path: Path,
    profile_output_path: Path | None,
    backend: str,
    configuration: SelectedConfiguration,
) -> None:
    """Export mapped row assignments and raw numeric profiles for one seed."""

    selected = runs[
        (runs["random_seed"].astype(int) == seed)
        & (runs["distance_contract"] == MIXED_DISTANCE_CONTRACT)
        & (runs["status"] == "ok")
    ]
    if len(selected) != 1:
        msg = f"No unique successful asymmetric evaluation exists for seed={seed}."
        raise ValueError(msg)

    labels = np.asarray(json.loads(selected.iloc[0]["cluster_assignments"]), dtype=int)
    preprocessed = pd.read_csv(data_path)
    if len(labels) != len(preprocessed) or len(labels) != len(mvdec.h_fused_df):
        msg = "Interpretation labels do not match preprocessed and fused rows."
        raise ValueError(msg)

    assignments = preprocessed.add_prefix("preprocessed_")
    assignments.insert(0, "original_index", np.arange(len(assignments)))
    assignments["cluster"] = labels
    assignments["random_seed"] = seed
    assignments["backend"] = backend
    assignments["distance_contract"] = MIXED_DISTANCE_CONTRACT
    assignments["source_job_index"] = configuration.source_job_index
    assignments["selected_features"] = ", ".join(configuration.selected_features)
    assignments = attach_row_mapping(assignments, pd.read_csv(mapping_path))
    _write_csv(assignments, output_path)

    if profile_output_path is not None:
        mapping = pd.read_csv(mapping_path)
        excluded = {"csv_row_index", "merged_row_index"}
        raw_numeric_columns = [
            column
            for column in mapping.select_dtypes(include="number").columns
            if column not in excluded and mapping[column].notna().any()
        ]
        profiles = build_cluster_profiles(
            assignments,
            raw_numeric_columns,
            cluster_column="cluster",
        )
        _write_csv(profiles, profile_output_path)


def parse_args() -> argparse.Namespace:
    """Parse the best-configuration stability evaluation command."""

    parser = argparse.ArgumentParser(
        description=(
            "Re-run one selected experiment row across seeds, compare symmetric "
            "and asymmetric Silhouette, and optionally export interpretation."
        )
    )
    parser.add_argument("--results-path", type=Path, required=True)
    parser.add_argument(
        "--backend", choices=["kprototypes", "intuitive"], required=True
    )
    parser.add_argument("--score-column", required=True)
    parser.add_argument("--job-index", type=int)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument(
        "--representation-path",
        type=Path,
        default=FUSED_REPRESENTATION_PATH,
    )
    parser.add_argument("--data-path", type=Path, default=PREPROCESSED_DATA_PATH)
    parser.add_argument("--runs-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument(
        "--init-strategy",
        choices=["farthest_first", "paper"],
        default="farthest_first",
    )
    parser.add_argument("--strict-init", action="store_true")
    parser.add_argument("--mapping-path", type=Path)
    parser.add_argument("--interpretation-output", type=Path)
    parser.add_argument("--profile-output", type=Path)
    parser.add_argument("--interpretation-seed", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    """Validate cross-argument contracts before running expensive work."""

    if args.interpretation_output is not None:
        if args.mapping_path is None or args.interpretation_seed is None:
            msg = (
                "--interpretation-output requires --mapping-path and "
                "--interpretation-seed."
            )
            raise ValueError(msg)
    if args.profile_output is not None and args.interpretation_output is None:
        msg = "--profile-output requires --interpretation-output."
        raise ValueError(msg)
    if (
        args.interpretation_seed is not None
        and args.interpretation_seed not in args.seeds
    ):
        msg = "--interpretation-seed must be included in --seeds."
        raise ValueError(msg)

    outputs = [args.runs_output, args.summary_output]
    if args.interpretation_output is not None:
        outputs.append(args.interpretation_output)
    if args.profile_output is not None:
        outputs.append(args.profile_output)
    existing = [path for path in outputs if path.exists()]
    if existing and not args.force:
        msg = f"Output already exists: {existing}. Use --force to overwrite it."
        raise FileExistsError(msg)


def main() -> None:
    """Run multi-seed validation, distance ablation, and interpretation export."""

    args = parse_args()
    _validate_args(args)
    mvdec = load_mvdec_result(args.representation_path, args.data_path)
    build_experiment_context(
        artifact=mvdec.raw,
        data_path=args.data_path,
        representation_path=args.representation_path,
        requested_output_dir=args.results_path.parent,
    )
    selected_row = select_result_row(
        args.results_path,
        args.score_column,
        args.job_index,
    )
    configuration = parse_selected_configuration(selected_row)
    binary_df = rebuild_selected_binary_features(mvdec.h_fused_df, configuration)
    seeds = tuple(args.seeds)
    runs = evaluate_fixed_configuration(
        backend=args.backend,
        h_fused_df=mvdec.h_fused_df,
        binary_df=binary_df,
        mvdec_labels=mvdec.labels,
        configuration=configuration,
        n_clusters=artifact_n_clusters(mvdec.raw),
        seeds=seeds,
        true_labels=mvdec.true_labels,
        init_strategy=args.init_strategy,
        strict_init=args.strict_init,
    )
    runs.insert(0, "dataset", mvdec.raw.get("dataset"))
    runs.insert(1, "method", "MiMvDEC")
    runs.insert(2, "source_sha256", mvdec.source_sha256)
    summary = aggregate_seed_evaluation(runs)
    _write_csv(runs, args.runs_output)
    _write_csv(summary, args.summary_output)

    if args.interpretation_output is not None:
        export_seed_interpretation(
            runs=runs,
            seed=args.interpretation_seed,
            mvdec=mvdec,
            data_path=args.data_path,
            mapping_path=args.mapping_path,
            output_path=args.interpretation_output,
            profile_output_path=args.profile_output,
            backend=args.backend,
            configuration=configuration,
        )

    logger.info("Saved per-seed evaluation to {}", args.runs_output)
    logger.info("Saved seed summary to {}", args.summary_output)


if __name__ == "__main__":
    main()
