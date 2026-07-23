"""Run configured private-dataset without-FFS experiments across MvDEC seeds."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

import config as project_config
from pipeline.data import load_mvdec_result
from pipeline.experiment_context import build_experiment_context
from pipeline.experiments import run_without_ffs

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_DIR / "configs" / "private_experiments.json"
CONFIG_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PrivateExperimentSpec:
    """Validated paths and runtime settings for one private dataset."""

    dataset: str
    data_path: Path
    representation_root: Path
    output_root: Path
    seeds: tuple[int, ...]
    workers: int


def _project_path(value: object, field: str, project_dir: Path) -> Path:
    """Resolve one required repository-relative path from configuration."""

    if not isinstance(value, str) or not value.strip():
        msg = f"Private experiment field {field!r} must be a non-empty path."
        raise ValueError(msg)
    path = Path(value)
    return path if path.is_absolute() else project_dir / path


def load_experiment_spec(
    config_path: Path,
    dataset: str,
    project_dir: Path = PROJECT_DIR,
) -> PrivateExperimentSpec:
    """Load and validate one private experiment specification."""

    with config_path.open(encoding="utf-8") as file:
        registry = json.load(file)
    if registry.get("schema_version") != CONFIG_SCHEMA_VERSION:
        msg = f"Unsupported private experiment config schema: {config_path}."
        raise ValueError(msg)

    datasets = registry.get("datasets")
    if not isinstance(datasets, dict) or dataset not in datasets:
        available = sorted(datasets) if isinstance(datasets, dict) else []
        msg = f"Unknown private dataset {dataset!r}; available datasets: {available}."
        raise ValueError(msg)
    raw_spec = datasets[dataset]
    if not isinstance(raw_spec, dict):
        msg = f"Private dataset configuration must be an object: {dataset}."
        raise ValueError(msg)

    raw_seeds = raw_spec.get("seeds")
    if not isinstance(raw_seeds, list) or not raw_seeds:
        msg = f"Private dataset {dataset} requires a non-empty seeds list."
        raise ValueError(msg)
    seeds = tuple(int(seed) for seed in raw_seeds)
    if len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        msg = f"Private dataset {dataset} seeds must be unique non-negative integers."
        raise ValueError(msg)

    workers = int(raw_spec.get("workers", 0))
    if workers < 1:
        msg = f"Private dataset {dataset} workers must be at least 1."
        raise ValueError(msg)

    return PrivateExperimentSpec(
        dataset=dataset,
        data_path=_project_path(raw_spec.get("data_path"), "data_path", project_dir),
        representation_root=_project_path(
            raw_spec.get("representation_root"),
            "representation_root",
            project_dir,
        ),
        output_root=_project_path(
            raw_spec.get("output_root"),
            "output_root",
            project_dir,
        ),
        seeds=seeds,
        workers=workers,
    )


def resolve_seed_artifact(spec: PrivateExperimentSpec, seed: int) -> Path:
    """Return the single MvDEC artifact configured for one seed."""

    seed_directory = f"seed_{seed}"
    artifacts = sorted(
        path
        for path in spec.representation_root.rglob("artifact.pkl")
        if path.parent.name == seed_directory
    )
    if len(artifacts) != 1:
        msg = (
            f"Expected exactly one {spec.dataset} artifact for seed {seed} under "
            f"{spec.representation_root}, found {len(artifacts)}."
        )
        raise FileNotFoundError(msg)
    return artifacts[0]


def run_configured_experiment(
    spec: PrivateExperimentSpec,
    workers: int | None = None,
    resume: bool = False,
) -> None:
    """Run without-FFS K-Prototypes for every configured MvDEC seed."""

    selected_workers = spec.workers if workers is None else workers
    if selected_workers < 1:
        msg = "workers must be at least 1."
        raise ValueError(msg)
    if not spec.data_path.is_file():
        msg = f"Missing preprocessed dataset: {spec.data_path}."
        raise FileNotFoundError(msg)

    for seed in spec.seeds:
        representation_path = resolve_seed_artifact(spec, seed)
        output_dir = spec.output_root / f"seed_{seed}"
        logger.info(
            "start:dataset={}; workflow=without_ffs; seed={}",
            spec.dataset,
            seed,
        )

        mvdec_result = load_mvdec_result(representation_path, spec.data_path)
        experiment = build_experiment_context(
            artifact=mvdec_result.raw,
            data_path=spec.data_path,
            representation_path=representation_path,
            requested_output_dir=output_dir,
            random_state=seed,
        )
        run_without_ffs(
            h_fused_df=mvdec_result.h_fused_df,
            save_path=experiment.result_path(
                project_config.WITHOUT_FFS_KPROTOTYPES_RESULTS_PATH
            ),
            n_clusters=experiment.n_clusters,
            baseline_score=mvdec_result.evaluation_score,
            random_state=seed,
            workers=selected_workers,
            resume=resume,
        )
        logger.info(
            "complete:dataset={}; workflow=without_ffs; seed={}",
            spec.dataset,
            seed,
        )


def parse_args() -> argparse.Namespace:
    """Parse the configured private without-FFS command."""

    parser = argparse.ArgumentParser(
        description="Run without-FFS FP-Max for all configured MvDEC seeds."
    )
    parser.add_argument("dataset", help="Dataset key from the private config.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume existing rows instead of rerunning the configured grid.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the configured private without-FFS workflow."""

    args = parse_args()
    spec = load_experiment_spec(args.config, args.dataset.upper())
    run_configured_experiment(spec, workers=args.workers, resume=args.resume)


if __name__ == "__main__":
    main()
