"""Run seed-isolated native Intuitive FP-Max experiments."""

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger

import config as project_config
from pipeline.data import file_sha256, load_mvdec_result
from pipeline.experiment_context import build_experiment_context
from pipeline.experiments import (
    FFS_INTUITIVE_NATIVE_COLUMNS,
    INTUITIVE_MODEL_AUDIT_COLUMNS,
    WITHOUT_FFS_INTUITIVE_NATIVE_COLUMNS,
    IntuitiveParams,
    native_trial_sidecar_for,
    run_ffs_intuitive_native,
    run_without_ffs_intuitive_native,
)
from pipeline.intuitive_protocol import (
    DEFAULT_PROTOCOL_CONFIG_PATH,
    PRIMARY_PROTOCOL_ID,
    IntuitiveProtocol,
    load_intuitive_protocol,
)
from pipeline.io import result_csv_path
from pipeline.private_without_ffs import (
    DEFAULT_CONFIG_PATH,
    load_experiment_spec,
    resolve_seed_artifact,
)

IntuitiveSource = Literal["without-ffs", "ffs"]
MANIFEST_SCHEMA_VERSION = 4


def _parameter_grid(protocol: IntuitiveProtocol) -> tuple[IntuitiveParams, ...]:
    """Build the exact parameter grid declared by a versioned protocol."""

    return tuple(
        IntuitiveParams(mu_param=mu_param, gamma=gamma, beta=beta)
        for mu_param, gamma, beta in product(
            protocol.mu_params,
            protocol.gammas,
            protocol.betas,
        )
    )


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    """Write one JSON manifest without exposing a partial file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _contract_hash(contract: dict[str, object]) -> str:
    """Return a stable identifier for an Intuitive run contract."""

    serialized = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _csv_data_row_count(path: Path) -> int:
    """Count CSV data rows without loading a potentially large trial file."""

    with path.open(encoding="utf-8", newline="") as file:
        row_count = sum(1 for _row in csv.reader(file))
    return max(row_count - 1, 0)


def _result_path(output_dir: Path, source: IntuitiveSource) -> Path:
    """Return the protocol-scoped native Intuitive result CSV."""

    filename = (
        project_config.FFS_INTUITIVE_NATIVE_RESULTS_PATH.name
        if source == "ffs"
        else project_config.WITHOUT_FFS_INTUITIVE_NATIVE_RESULTS_PATH.name
    )
    return output_dir / filename


def _result_columns(source: IntuitiveSource) -> list[str]:
    """Return the persisted schema for one native Intuitive workflow."""

    return (
        FFS_INTUITIVE_NATIVE_COLUMNS
        if source == "ffs"
        else WITHOUT_FFS_INTUITIVE_NATIVE_COLUMNS
    )


def _build_contract(
    *,
    dataset: str,
    seed: int,
    source: IntuitiveSource,
    data_path: Path,
    representation_path: Path,
    n_clusters: int,
    baseline_score: float,
    protocol: IntuitiveProtocol,
    protocol_config_path: Path,
    parameter_grid: tuple[IntuitiveParams, ...],
    workers: int,
    param_workers: int,
    candidate_workers: int,
) -> dict[str, object]:
    """Build the immutable provenance contract for one Intuitive grid."""

    return {
        "dataset": dataset,
        "seed": seed,
        "source": source,
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
        "execution": {
            "workers": workers,
            "param_workers": param_workers,
            "candidate_workers": candidate_workers,
        },
    }


def _prepare_manifest(
    path: Path,
    contract: dict[str, object],
    output_path: Path,
    resume: bool,
) -> dict[str, object]:
    """Validate resume provenance and mark the workflow as running."""

    contract_hash = _contract_hash(contract)
    if resume:
        if not path.is_file():
            msg = f"Cannot resume Intuitive output without its manifest: {path}."
            raise FileNotFoundError(msg)
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("contract_hash") != contract_hash:
            msg = f"Intuitive manifest contract does not match: {path}."
            raise ValueError(msg)

    payload: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "running",
        "contract_hash": contract_hash,
        "contract": contract,
        "output": {"results_path": str(output_path.resolve())},
    }
    _write_json_atomic(path, payload)
    return payload


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
    representation_path = resolve_seed_artifact(spec, seed)
    seed_output_dir = spec.output_root / f"seed_{seed}"
    output_dir = seed_output_dir / protocol.protocol_id
    save_path = _result_path(output_dir, source)
    trial_path, _trial_columns = native_trial_sidecar_for(
        save_path,
        _result_columns(source),
    )
    if not resume and trial_path is not None:
        trial_path.unlink(missing_ok=True)

    mvdec_result = load_mvdec_result(representation_path, spec.data_path)
    experiment = build_experiment_context(
        artifact=mvdec_result.raw,
        data_path=spec.data_path,
        representation_path=representation_path,
        requested_output_dir=output_dir,
        random_state=seed,
    )
    contract = _build_contract(
        dataset=spec.dataset,
        seed=seed,
        source=source,
        data_path=spec.data_path,
        representation_path=representation_path,
        n_clusters=experiment.n_clusters,
        baseline_score=mvdec_result.evaluation_score,
        protocol=protocol,
        protocol_config_path=protocol_config_path,
        parameter_grid=parameter_grid,
        workers=selected_workers,
        param_workers=param_workers,
        candidate_workers=candidate_workers,
    )
    manifest_name = f"{source.replace('-', '_')}_intuitive_native_manifest.json"
    manifest_path = output_dir / manifest_name
    manifest = _prepare_manifest(manifest_path, contract, save_path, resume)

    logger.info(
        "start:dataset={}; workflow={}_intuitive_native; protocol={}; seed={}; "
        "workers={}; param_workers={}; candidate_workers={}",
        spec.dataset,
        source.replace("-", "_"),
        protocol.protocol_id,
        seed,
        selected_workers,
        param_workers,
        candidate_workers,
    )
    start = perf_counter()
    try:
        if source == "ffs":
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
                two_stage=False,
                min_improvement=protocol.ffs_min_improvement,
                init_strategy=protocol.initialization_strategy,
                strict_init=protocol.strict_initialization,
                non_membership=protocol.non_membership,
                max_iter=protocol.max_iter,
                empty_cluster_policy=protocol.empty_cluster_policy,
                min_cluster_size=protocol.min_cluster_size,
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
                two_stage=False,
                init_strategy=protocol.initialization_strategy,
                strict_init=protocol.strict_initialization,
                non_membership=protocol.non_membership,
                max_iter=protocol.max_iter,
                empty_cluster_policy=protocol.empty_cluster_policy,
                min_cluster_size=protocol.min_cluster_size,
            )
    except Exception as error:
        manifest.update(
            {
                "status": "failed",
                "runtime_seconds": perf_counter() - start,
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
            }
        )
        _write_json_atomic(manifest_path, manifest)
        raise

    csv_path = result_csv_path(save_path)
    status_counts = (
        results["status"].value_counts(dropna=False).to_dict()
        if "status" in results
        else {}
    )
    output_metadata: dict[str, object] = {
        "results_path": str(csv_path.resolve()),
        "results_sha256": file_sha256(csv_path),
        "row_count": len(results),
        "status_counts": status_counts,
    }
    if trial_path is not None and trial_path.is_file():
        output_metadata.update(
            {
                "trials_path": str(trial_path.resolve()),
                "trials_sha256": file_sha256(trial_path),
                "trial_row_count": _csv_data_row_count(trial_path),
            }
        )
    manifest.update(
        {
            "status": "complete",
            "runtime_seconds": perf_counter() - start,
            "output": output_metadata,
        }
    )
    _write_json_atomic(manifest_path, manifest)
    logger.info(
        "complete:dataset={}; workflow={}_intuitive_native; protocol={}; seed={}; "
        "rows={}",
        spec.dataset,
        source.replace("-", "_"),
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
