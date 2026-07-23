"""Run seed-isolated post-clustering Intuitive experiments."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Literal

import pandas as pd
from loguru import logger

import config as project_config
from pipeline.clustering import (
    DEFAULT_INTUITIVE_INIT_STRATEGY,
    DEFAULT_INTUITIVE_STRICT_INIT,
    MIXED_DISTANCE_CONTRACT,
)
from pipeline.data import file_sha256, load_mvdec_result
from pipeline.experiment_context import build_experiment_context
from pipeline.experiments import (
    default_intuitive_param_grid,
    run_post_ffs_intuitive,
    run_post_without_ffs_intuitive,
)
from pipeline.io import result_csv_path
from pipeline.private_without_ffs import (
    DEFAULT_CONFIG_PATH,
    load_experiment_spec,
    resolve_seed_artifact,
)

IntuitiveSource = Literal["without-ffs", "ffs"]
MANIFEST_SCHEMA_VERSION = 1


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


def _source_paths(output_dir: Path, source: IntuitiveSource) -> tuple[Path, Path]:
    """Return source and destination CSV paths for one Intuitive workflow."""

    if source == "ffs":
        return (
            output_dir / project_config.FFS_RESULTS_PATH.name,
            output_dir / project_config.POST_FFS_INTUITIVE_RESULTS_PATH.name,
        )
    return (
        output_dir / project_config.WITHOUT_FFS_KPROTOTYPES_RESULTS_PATH.name,
        output_dir / project_config.POST_WITHOUT_FFS_INTUITIVE_RESULTS_PATH.name,
    )


def _build_contract(
    *,
    dataset: str,
    seed: int,
    source: IntuitiveSource,
    data_path: Path,
    representation_path: Path,
    source_path: Path,
    n_clusters: int,
) -> dict[str, object]:
    """Build the immutable provenance contract for one Intuitive grid."""

    parameter_grid = [asdict(params) for params in default_intuitive_param_grid()]
    return {
        "dataset": dataset,
        "seed": seed,
        "source": source,
        "n_clusters": n_clusters,
        "data_path": str(data_path.resolve()),
        "data_sha256": file_sha256(data_path),
        "representation_path": str(representation_path.resolve()),
        "representation_sha256": file_sha256(representation_path),
        "source_results_path": str(source_path.resolve()),
        "source_results_sha256": file_sha256(source_path),
        "distance_contract": MIXED_DISTANCE_CONTRACT,
        "initialization": {
            "strategy": DEFAULT_INTUITIVE_INIT_STRATEGY,
            "strict": DEFAULT_INTUITIVE_STRICT_INIT,
        },
        "parameter_grid": parameter_grid,
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
    param_workers: int = 1,
    resume: bool = False,
) -> pd.DataFrame:
    """Run the full post-Intuitive grid for one configured artifact seed."""

    if param_workers < 1:
        msg = "param_workers must be at least 1."
        raise ValueError(msg)

    spec = load_experiment_spec(config_path, dataset.upper())
    if seed not in spec.seeds:
        msg = f"Seed {seed} is not configured for {spec.dataset}: {spec.seeds}."
        raise ValueError(msg)

    representation_path = resolve_seed_artifact(spec, seed)
    output_dir = spec.output_root / f"seed_{seed}"
    source_path, save_path = _source_paths(output_dir, source)
    if not source_path.is_file():
        msg = f"Missing {source} source results: {source_path}."
        raise FileNotFoundError(msg)

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
        source_path=source_path,
        n_clusters=experiment.n_clusters,
    )
    manifest_name = f"post_{source.replace('-', '_')}_intuitive_manifest.json"
    manifest_path = output_dir / manifest_name
    manifest = _prepare_manifest(manifest_path, contract, save_path, resume)

    logger.info(
        "start:dataset={}; workflow=post_{}_intuitive; seed={}; param_workers={}",
        spec.dataset,
        source.replace("-", "_"),
        seed,
        param_workers,
    )
    start = perf_counter()
    try:
        if source == "ffs":
            results = run_post_ffs_intuitive(
                h_fused_df=mvdec_result.h_fused_df,
                save_path=save_path,
                ffs_path=source_path,
                n_clusters=experiment.n_clusters,
                random_state=seed,
                resume=resume,
                param_workers=param_workers,
            )
        else:
            results = run_post_without_ffs_intuitive(
                h_fused_df=mvdec_result.h_fused_df,
                save_path=save_path,
                without_ffs_path=source_path,
                n_clusters=experiment.n_clusters,
                random_state=seed,
                resume=resume,
                param_workers=param_workers,
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
    manifest.update(
        {
            "status": "complete",
            "runtime_seconds": perf_counter() - start,
            "output": {
                "results_path": str(csv_path.resolve()),
                "results_sha256": file_sha256(csv_path),
                "row_count": len(results),
                "status_counts": status_counts,
            },
        }
    )
    _write_json_atomic(manifest_path, manifest)
    logger.info(
        "complete:dataset={}; workflow=post_{}_intuitive; seed={}; rows={}",
        spec.dataset,
        source.replace("-", "_"),
        seed,
        len(results),
    )
    return results


def parse_args() -> argparse.Namespace:
    """Parse the private Intuitive runner command line."""

    parser = argparse.ArgumentParser(
        description="Run post-clustering Intuitive over every usable source row."
    )
    parser.add_argument("dataset", help="Dataset key from the private config.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--source",
        choices=("without-ffs", "ffs"),
        default="ffs",
        help="Reuse every usable row from without-FFS or FFS results.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--param-workers", type=int, default=1)
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
        param_workers=args.param_workers,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
