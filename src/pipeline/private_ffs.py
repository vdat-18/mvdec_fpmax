"""Run the configured FFS grid for one private-dataset MvDEC seed."""

import argparse
from pathlib import Path

from loguru import logger

import config as project_config
from pipeline.data import load_mvdec_result
from pipeline.experiment_context import build_experiment_context
from pipeline.experiments import run_ffs
from pipeline.private_without_ffs import (
    DEFAULT_CONFIG_PATH,
    load_experiment_spec,
    resolve_seed_artifact,
)


def run_configured_ffs(
    dataset: str,
    seed: int,
    config_path=DEFAULT_CONFIG_PATH,
    workers: int | None = None,
    candidate_workers: int = 1,
    resume: bool = False,
) -> None:
    """Run the complete FFS grid for one configured seed."""

    spec = load_experiment_spec(config_path, dataset.upper())
    if seed not in spec.seeds:
        msg = f"Seed {seed} is not configured for {spec.dataset}: {spec.seeds}."
        raise ValueError(msg)
    selected_workers = spec.workers if workers is None else workers
    if selected_workers < 1:
        msg = "workers must be at least 1."
        raise ValueError(msg)
    if candidate_workers < 1:
        msg = "candidate_workers must be at least 1."
        raise ValueError(msg)

    representation_path = resolve_seed_artifact(spec, seed)
    output_dir = spec.output_root / f"seed_{seed}"
    logger.info("start:dataset={}; workflow=ffs; seed={}", spec.dataset, seed)

    mvdec_result = load_mvdec_result(representation_path, spec.data_path)
    experiment = build_experiment_context(
        artifact=mvdec_result.raw,
        data_path=spec.data_path,
        representation_path=representation_path,
        requested_output_dir=output_dir,
        random_state=seed,
    )
    run_ffs(
        h_fused_df=mvdec_result.h_fused_df,
        save_path=experiment.result_path(project_config.FFS_RESULTS_PATH),
        n_clusters=experiment.n_clusters,
        random_state=seed,
        baseline_score=mvdec_result.evaluation_score,
        resume=resume,
        workers=selected_workers,
        candidate_workers=candidate_workers,
    )
    logger.info("complete:dataset={}; workflow=ffs; seed={}", spec.dataset, seed)


def parse_args() -> argparse.Namespace:
    """Parse the configured private FFS command."""

    parser = argparse.ArgumentParser(
        description="Run the FFS grid for one configured private MvDEC seed."
    )
    parser.add_argument("dataset", help="Dataset key from the private config.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--candidate-workers", type=int, default=1)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume existing rows instead of rerunning the FFS grid.",
    )
    return parser.parse_args()


def main() -> None:
    """Run one configured private FFS workflow."""

    args = parse_args()
    run_configured_ffs(
        dataset=args.dataset,
        seed=args.seed,
        config_path=args.config,
        workers=args.workers,
        candidate_workers=args.candidate_workers,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
