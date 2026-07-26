"""Run seed-isolated native Intuitive FP-Max experiments."""

import argparse
import hashlib
import json
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger

import config as project_config
from pipeline.data import file_sha256, load_mvdec_result
from pipeline.experiment_context import artifact_n_clusters, build_experiment_context
from pipeline.experiments import (
    INTUITIVE_MODEL_AUDIT_COLUMNS,
    IntuitiveParams,
    run_ffs_intuitive_native,
    run_ffs_intuitive_view_weighted_native,
    run_without_ffs_intuitive_native,
    run_without_ffs_intuitive_view_weighted_native,
)
from pipeline.intuitive_protocol import (
    DEFAULT_PROTOCOL_CONFIG_PATH,
    PARAMETER_SEARCH,
    PRIMARY_PROTOCOL_ID,
    IntuitiveProtocol,
    load_intuitive_protocol,
)
from pipeline.private_without_ffs import (
    DEFAULT_CONFIG_PATH,
    load_experiment_spec,
    resolve_seed_artifact,
)

IntuitiveSource = Literal["without-ffs", "ffs"]


def _parameter_grid(protocol: IntuitiveProtocol) -> tuple[IntuitiveParams, ...]:
    """Build the exact parameter grid declared by a versioned protocol."""

    alphas: tuple[float | None, ...] = protocol.view_weight_alphas or (None,)
    return tuple(
        IntuitiveParams(
            mu_param=mu_param,
            gamma=gamma,
            beta=beta,
            view_weight_alpha=alpha,
        )
        for alpha, mu_param, gamma, beta in product(
            alphas,
            protocol.mu_params,
            protocol.gammas,
            protocol.betas,
        )
    )


def _contract_hash(contract: dict[str, object]) -> str:
    """Return a stable identifier for an Intuitive run contract."""

    serialized = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _result_path(
    output_dir: Path,
    source: IntuitiveSource,
    view_weighted: bool,
) -> Path:
    """Return the method-specific result CSV in the shared seed directory."""

    if source == "ffs":
        configured_path = (
            project_config.FFS_INTUITIVE_VIEW_WEIGHTED_NATIVE_RESULTS_PATH
            if view_weighted
            else project_config.FFS_INTUITIVE_NATIVE_RESULTS_PATH
        )
    else:
        configured_path = (
            project_config.WITHOUT_FFS_INTUITIVE_VIEW_WEIGHTED_NATIVE_RESULTS_PATH
            if view_weighted
            else project_config.WITHOUT_FFS_INTUITIVE_NATIVE_RESULTS_PATH
        )
    filename = configured_path.name
    return output_dir / filename


def _build_contract(
    *,
    dataset: str,
    seed: int,
    data_path: Path,
    representation_path: Path,
    n_clusters: int,
    baseline_score: float,
    protocol: IntuitiveProtocol,
    protocol_config_path: Path,
    parameter_grid: tuple[IntuitiveParams, ...],
) -> dict[str, object]:
    """Build shared provenance for both Intuitive result sources."""

    return {
        "dataset": dataset,
        "seed": seed,
        "protocol_id": protocol.protocol_id,
        "protocol": asdict(protocol),
        "protocol_config_path": str(protocol_config_path.resolve()),
        "protocol_config_sha256": file_sha256(protocol_config_path),
        "n_clusters": n_clusters,
        "data_path": str(data_path.resolve()),
        "data_sha256": file_sha256(data_path),
        "representation_path": str(representation_path.resolve()),
        "representation_sha256": file_sha256(representation_path),
        "baseline_numeric_gower_silhouette": baseline_score,
        "model_audit_columns": INTUITIVE_MODEL_AUDIT_COLUMNS,
        "parameter_grid": [asdict(params) for params in parameter_grid],
    }


def run_configured_intuitive(
    dataset: str,
    seed: int,
    source: IntuitiveSource,
    config_path: Path = DEFAULT_CONFIG_PATH,
    protocol_id: str = PRIMARY_PROTOCOL_ID,
    protocol_config_path: Path = DEFAULT_PROTOCOL_CONFIG_PATH,
    workers: int | None = None,
    param_workers: int = 1,
    candidate_workers: int = 1,
    resume: bool = False,
) -> pd.DataFrame:
    """Run a native Intuitive FP-Max workflow for one configured artifact seed."""

    if source not in ("without-ffs", "ffs"):
        msg = f"Unsupported Intuitive workflow source: {source!r}."
        raise ValueError(msg)
    selected_workers = 1 if workers is None else workers
    if selected_workers < 1:
        msg = "workers must be at least 1."
        raise ValueError(msg)
    if param_workers < 1:
        msg = "param_workers must be at least 1."
        raise ValueError(msg)
    if candidate_workers < 1:
        msg = "candidate_workers must be at least 1."
        raise ValueError(msg)
    if source == "without-ffs" and candidate_workers != 1:
        msg = "candidate_workers only applies to the FFS workflow."
        raise ValueError(msg)
    parallel_axes = sum(
        value > 1 for value in (selected_workers, param_workers, candidate_workers)
    )
    if parallel_axes > 1:
        msg = (
            "Use only one parallel axis above 1: workers, param_workers, "
            "or candidate_workers."
        )
        raise ValueError(msg)

    spec = load_experiment_spec(config_path, dataset.upper())
    if seed not in spec.seeds:
        msg = f"Seed {seed} is not configured for {spec.dataset}: {spec.seeds}."
        raise ValueError(msg)

    protocol = load_intuitive_protocol(protocol_config_path, protocol_id)
    parameter_grid = _parameter_grid(protocol)
    view_weighted = bool(protocol.view_weight_alphas)
    representation_path = resolve_seed_artifact(spec, seed)
    seed_output_dir = spec.output_root / f"seed_{seed}"
    output_dir = seed_output_dir
    save_path = _result_path(output_dir, source, view_weighted)

    mvdec_result = load_mvdec_result(representation_path, spec.data_path)
    n_clusters = artifact_n_clusters(mvdec_result.raw)
    contract = _build_contract(
        dataset=spec.dataset,
        seed=seed,
        data_path=spec.data_path,
        representation_path=representation_path,
        n_clusters=n_clusters,
        baseline_score=mvdec_result.evaluation_score,
        protocol=protocol,
        protocol_config_path=protocol_config_path,
        parameter_grid=parameter_grid,
    )
    experiment = build_experiment_context(
        artifact=mvdec_result.raw,
        data_path=spec.data_path,
        representation_path=representation_path,
        requested_output_dir=output_dir,
        random_state=seed,
        workflow_id=protocol.protocol_id,
        workflow_metadata={
            "contract_hash": _contract_hash(contract),
            "contract": contract,
        },
    )
    workflow_name = (
        f"{source.replace('-', '_')}_intuitive_view_weighted_native"
        if view_weighted
        else f"{source.replace('-', '_')}_intuitive_native"
    )
    logger.info(
        "start:dataset={}; workflow={}; protocol={}; seed={}; "
        "workers={}; param_workers={}; candidate_workers={}",
        spec.dataset,
        workflow_name,
        protocol.protocol_id,
        seed,
        selected_workers,
        param_workers,
        candidate_workers,
    )
    if source == "ffs" and view_weighted:
        results = run_ffs_intuitive_view_weighted_native(
            h_fused_df=mvdec_result.h_fused_df,
            save_path=save_path,
            strategies=protocol.fpmax_strategies,
            n_bins_options=protocol.fpmax_n_bins,
            supports=np.asarray(protocol.fpmax_min_supports),
            n_clusters=experiment.n_clusters,
            random_state=seed,
            baseline_score=mvdec_result.evaluation_score,
            resume=resume,
            workers=selected_workers,
            param_workers=param_workers,
            candidate_workers=candidate_workers,
            intuitive_param_grid=parameter_grid,
            two_stage=protocol.parameter_search == PARAMETER_SEARCH,
            min_improvement=protocol.ffs_min_improvement,
            init_strategy=protocol.initialization_strategy,
            strict_init=protocol.strict_initialization,
            non_membership=protocol.non_membership,
            max_iter=protocol.max_iter,
            empty_cluster_policy=protocol.empty_cluster_policy,
            min_cluster_size=protocol.min_cluster_size,
            persist_trials=False,
        )
    elif source == "ffs":
        results = run_ffs_intuitive_native(
            h_fused_df=mvdec_result.h_fused_df,
            save_path=save_path,
            strategies=protocol.fpmax_strategies,
            n_bins_options=protocol.fpmax_n_bins,
            supports=np.asarray(protocol.fpmax_min_supports),
            n_clusters=experiment.n_clusters,
            random_state=seed,
            baseline_score=mvdec_result.evaluation_score,
            resume=resume,
            workers=selected_workers,
            param_workers=param_workers,
            candidate_workers=candidate_workers,
            intuitive_param_grid=parameter_grid,
            two_stage=protocol.parameter_search == PARAMETER_SEARCH,
            min_improvement=protocol.ffs_min_improvement,
            init_strategy=protocol.initialization_strategy,
            strict_init=protocol.strict_initialization,
            non_membership=protocol.non_membership,
            max_iter=protocol.max_iter,
            empty_cluster_policy=protocol.empty_cluster_policy,
            min_cluster_size=protocol.min_cluster_size,
            persist_trials=False,
        )
    elif view_weighted:
        results = run_without_ffs_intuitive_view_weighted_native(
            h_fused_df=mvdec_result.h_fused_df,
            save_path=save_path,
            strategies=protocol.fpmax_strategies,
            n_bins_options=protocol.fpmax_n_bins,
            supports=np.asarray(protocol.fpmax_min_supports),
            n_clusters=experiment.n_clusters,
            random_state=seed,
            baseline_score=mvdec_result.evaluation_score,
            resume=resume,
            workers=selected_workers,
            param_workers=param_workers,
            intuitive_param_grid=parameter_grid,
            two_stage=protocol.parameter_search == PARAMETER_SEARCH,
            init_strategy=protocol.initialization_strategy,
            strict_init=protocol.strict_initialization,
            non_membership=protocol.non_membership,
            max_iter=protocol.max_iter,
            empty_cluster_policy=protocol.empty_cluster_policy,
            min_cluster_size=protocol.min_cluster_size,
            persist_trials=False,
        )
    else:
        results = run_without_ffs_intuitive_native(
            h_fused_df=mvdec_result.h_fused_df,
            save_path=save_path,
            strategies=protocol.fpmax_strategies,
            n_bins_options=protocol.fpmax_n_bins,
            supports=np.asarray(protocol.fpmax_min_supports),
            n_clusters=experiment.n_clusters,
            random_state=seed,
            baseline_score=mvdec_result.evaluation_score,
            resume=resume,
            workers=selected_workers,
            param_workers=param_workers,
            intuitive_param_grid=parameter_grid,
            two_stage=protocol.parameter_search == PARAMETER_SEARCH,
            init_strategy=protocol.initialization_strategy,
            strict_init=protocol.strict_initialization,
            non_membership=protocol.non_membership,
            max_iter=protocol.max_iter,
            empty_cluster_policy=protocol.empty_cluster_policy,
            min_cluster_size=protocol.min_cluster_size,
            persist_trials=False,
        )
    logger.info(
        "complete:dataset={}; workflow={}; protocol={}; seed={}; rows={}",
        spec.dataset,
        workflow_name,
        protocol.protocol_id,
        seed,
        len(results),
    )
    return results


def parse_args() -> argparse.Namespace:
    """Parse the private Intuitive runner command line."""

    parser = argparse.ArgumentParser(
        description="Run native Intuitive over the configured FP-Max grid."
    )
    parser.add_argument("dataset", help="Dataset key from the private config.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--source",
        choices=("without-ffs", "ffs"),
        default="ffs",
        help="Run native Intuitive with all FP-Max features or Intuitive-native FFS.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--protocol", default=PRIMARY_PROTOCOL_ID)
    parser.add_argument(
        "--protocol-config",
        type=Path,
        default=DEFAULT_PROTOCOL_CONFIG_PATH,
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--param-workers", type=int, default=1)
    parser.add_argument("--candidate-workers", type=int, default=1)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only when the Intuitive provenance contract is unchanged.",
    )
    return parser.parse_args()


def main() -> None:
    """Run one configured private Intuitive workflow."""

    args = parse_args()
    run_configured_intuitive(
        dataset=args.dataset,
        seed=args.seed,
        source=args.source,
        config_path=args.config,
        protocol_id=args.protocol,
        protocol_config_path=args.protocol_config,
        workers=args.workers,
        param_workers=args.param_workers,
        candidate_workers=args.candidate_workers,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
