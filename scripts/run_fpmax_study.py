"""Run one (dataset, algorithm) group of the FP-Max study.

One process per group, so groups can be sharded across background processes.
"""

import argparse
import dataclasses
from pathlib import Path

from pipeline.study_arms import ALGORITHMS
from pipeline.study_config import (
    DEFAULT_STUDY_CONFIG_PATH,
    PROJECT_DIR,
    StudyConfig,
    load_study_config,
)
from pipeline.study_data import (
    StudyData,
    load_application_study_data,
    load_dekm_study_bundle,
    load_uci_study_data,
)
from pipeline.study_runner import run_group, shard_suffix


def load_named_dataset(
    name: str,
    study: StudyConfig,
    report_seed: int | None = None,
) -> StudyData:
    """Build StudyData for a configured UCI, application or DEKM dataset.

    A DEKM dataset has one embedding per report seed, so it needs ``report_seed``.
    """

    if name in study.dekm.study_names:
        if report_seed is None:
            msg = f"{name} has one embedding per seed; run it one seed at a time."
            raise ValueError(msg)
        path = (
            PROJECT_DIR
            / study.dekm.bundle_dir
            / study.dekm.bundle_name(name, report_seed)
        )
        return load_dekm_study_bundle(path, n_rows=study.dekm.n_rows)
    if name in study.uci_datasets:
        return load_uci_study_data(name, PROJECT_DIR / study.uci_data_dir)
    if name in study.applications:
        spec = study.applications[name]
        return load_application_study_data(
            name, PROJECT_DIR / spec.csv_path, spec.n_clusters
        )
    known = sorted([*study.uci_datasets, *study.applications, *study.dekm.study_names])
    msg = f"Unknown dataset {name!r}; configured datasets: {known}."
    raise ValueError(msg)


def parse_seed_subset(text: str, allowed: tuple[int, ...]) -> tuple[int, ...]:
    """Parse ``"0,3"`` into seeds that must all be configured report seeds."""

    try:
        seeds = tuple(int(part) for part in text.split(",") if part.strip())
    except ValueError as error:
        msg = f"--seeds must be comma-separated integers, got {text!r}."
        raise ValueError(msg) from error
    unknown = sorted(set(seeds) - set(allowed))
    if not seeds or unknown or len(set(seeds)) != len(seeds):
        msg = f"--seeds must be unique configured report seeds {allowed}; got {text!r}."
        raise ValueError(msg)
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_STUDY_CONFIG_PATH)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--algorithm", required=True, choices=ALGORITHMS)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--report-seeds",
        type=int,
        default=None,
        help="Use only the first N report seeds (smoke runs).",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated subset of the report seeds (one shard of the group).",
    )
    args = parser.parse_args()

    study = load_study_config(args.config)
    suffix = ""
    if args.seeds is not None:
        seeds = parse_seed_subset(args.seeds, study.report_seeds)
        study = dataclasses.replace(study, report_seeds=seeds)
        suffix = shard_suffix(seeds)
    if args.report_seeds is not None:
        if args.report_seeds < 1:
            parser.error("--report-seeds must be at least 1.")
        study = dataclasses.replace(
            study, report_seeds=study.report_seeds[: args.report_seeds]
        )
    single_seed = study.report_seeds[0] if len(study.report_seeds) == 1 else None
    data = load_named_dataset(args.dataset, study, single_seed)
    print(run_group(data, args.algorithm, study, args.out_dir, stem_suffix=suffix))


if __name__ == "__main__":
    main()
