"""Run the FP-Max study as independent subprocesses, one per (dataset, algorithm, seed).

``multiprocessing`` pools are unreliable on this machine, so jobs are plain
subprocesses kept at a fixed worker count. Jobs whose output CSV already exists are
skipped, so rerunning the same command resumes an interrupted run.
"""

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from pipeline.study_config import (
    DEFAULT_STUDY_CONFIG_PATH,
    PROJECT_DIR,
    StudyConfig,
    load_study_config,
)
from pipeline.study_runner import shard_suffix

RUN_SCRIPT = PROJECT_DIR / "scripts" / "run_fpmax_study.py"
ALGORITHM_CHOICES = ("kmeans", "kprototypes", "intuitive")
# Attribute-less numeric blocks make K-Means inapplicable (see the loaders).
NO_NUMERIC_DATASETS = frozenset({"bcw", "soybean_small_no_constants"})
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


@dataclass(frozen=True)
class Job:
    """One (dataset, algorithm, seed) unit of work."""

    dataset: str
    algorithm: str
    seed: int


def build_jobs(
    study: StudyConfig,
    datasets: list[str],
    algorithms: list[str],
) -> list[Job]:
    """Every dataset x algorithm x report seed; K-Means only with numeric data."""

    jobs = []
    for dataset in datasets:
        for algorithm in algorithms:
            if algorithm == "kmeans" and dataset in NO_NUMERIC_DATASETS:
                continue
            jobs.extend(Job(dataset, algorithm, seed) for seed in study.report_seeds)
    return jobs


def job_output_path(out_dir: Path, job: Job) -> Path:
    """CSV the job writes (the existence of this file marks the job as finished)."""

    suffix = shard_suffix((job.seed,))
    return Path(out_dir) / f"{job.dataset}__{job.algorithm}{suffix}.csv"


def job_command(config: Path, out_dir: Path, job: Job) -> list[str]:
    """Command line that runs one job."""

    return [
        sys.executable,
        str(RUN_SCRIPT),
        "--config",
        str(config),
        "--dataset",
        job.dataset,
        "--algorithm",
        job.algorithm,
        "--out-dir",
        str(out_dir),
        "--seeds",
        str(job.seed),
    ]


def _worker_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(dict.fromkeys(THREAD_VARIABLES, "1"))
    return environment


def run_jobs(
    jobs: list[Job],
    config: Path,
    out_dir: Path,
    workers: int,
) -> list[Job]:
    """Run pending jobs with at most ``workers`` subprocesses; return failures."""

    log_dir = Path(out_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    pending = [job for job in jobs if not job_output_path(out_dir, job).exists()]
    print(
        f"{len(jobs)} jobs, {len(jobs) - len(pending)} done, {len(pending)} to run"
    )
    running: list[tuple[Job, subprocess.Popen, object]] = []
    failed: list[Job] = []
    finished = 0
    while pending or running:
        while pending and len(running) < workers:
            job = pending.pop(0)
            log = (log_dir / f"{job_output_path(out_dir, job).stem}.log").open("w")
            process = subprocess.Popen(
                job_command(config, out_dir, job),
                stdout=log,
                stderr=subprocess.STDOUT,
                env=_worker_environment(),
                cwd=PROJECT_DIR,
            )
            running.append((job, process, log))
        time.sleep(2)
        still_running = []
        for job, process, log in running:
            code = process.poll()
            if code is None:
                still_running.append((job, process, log))
                continue
            log.close()
            finished += 1
            if code != 0:
                failed.append(job)
            print(f"[{finished}/{len(jobs)}] {job} exit={code}", flush=True)
        running = still_running
    return failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_STUDY_CONFIG_PATH)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--algorithms", nargs="*", default=list(ALGORITHM_CHOICES))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    study = load_study_config(args.config)
    datasets = args.datasets or [*study.uci_datasets, *study.applications]
    unknown = sorted(set(datasets) - {*study.uci_datasets, *study.applications})
    if unknown:
        parser.error(f"Unknown datasets: {unknown}")
    bad = sorted(set(args.algorithms) - set(ALGORITHM_CHOICES))
    if bad:
        parser.error(f"Unknown algorithms: {bad}")
    if args.workers < 1:
        parser.error("--workers must be at least 1.")

    jobs = build_jobs(study, datasets, args.algorithms)
    if args.dry_run:
        for job in jobs:
            print(job_output_path(args.out_dir, job).name)
        print(f"{len(jobs)} jobs")
        return 0
    failed = run_jobs(jobs, args.config, args.out_dir, args.workers)
    print(f"done: {len(jobs)} jobs, {len(failed)} failed")
    for job in failed:
        print(f"FAILED {job}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
