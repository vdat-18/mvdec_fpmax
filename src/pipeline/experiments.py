"""Experiment runners for MiMvDEC with and without forward feature selection."""

from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from time import sleep
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from config import FFS_RESULTS_PATH, RANDOM_STATE, WITHOUT_FFS_RESULTS_PATH
from pipeline.clustering import run_kprototypes
from pipeline.forward_selection import run_forward_selection
from pipeline.fpmax import (
    BIN_LABELS_BY_SIZE,
    DEFAULT_STRATEGIES,
    DEFAULT_SUPPORTS,
    DiscretizeStrategy,
    extract_fpmax_features,
)
from pipeline.io import empty_results, is_job_done, load_results, save_results


@dataclass(frozen=True)
class ExperimentJob:
    """One support value within a strategy and bin group."""

    job_index: int
    min_support: float


@dataclass(frozen=True)
class ExperimentGroup:
    """One worker group sharing a strategy and bin count."""

    group_index: int
    strategy: DiscretizeStrategy
    n_bins: int
    jobs: tuple[ExperimentJob, ...]


@dataclass(frozen=True)
class WithoutFfsRecord:
    """One MiMvDEC without-FFS sensitivity result row."""

    job_index: int
    group_index: int
    strategy: str
    n_bins: int
    min_support: float
    silhouette_score: float
    n_itemsets: int
    selected_features: str
    init: str | None


@dataclass(frozen=True)
class FfsRecord:
    """One MiMvDEC with-FFS sensitivity result row."""

    job_index: int
    group_index: int
    strategy: str
    n_bins: int
    min_support: float
    silhouette_score: float
    n_itemsets: int
    selected_features: str
    init: str | None
    final_score: float
    best_observed_score: float


JobRunner = Callable[
    [pd.DataFrame, ExperimentGroup, ExperimentJob, int, int, float],
    WithoutFfsRecord | FfsRecord,
]


def run_without_ffs(
    h_fused_df: pd.DataFrame,
    save_path: Path = WITHOUT_FFS_RESULTS_PATH,
    strategies: tuple[DiscretizeStrategy, ...] = DEFAULT_STRATEGIES,
    n_bins_options: tuple[int, ...] = (3, 5, 7),
    supports: np.ndarray = DEFAULT_SUPPORTS,
    n_clusters: int = 5,
    random_state: int = RANDOM_STATE,
    baseline_score: float = 0.4069,
    resume: bool = True,
    workers: int = 3,
    limit: int | None = None,
) -> pd.DataFrame:
    """Run MiMvDEC without forward feature selection."""

    return run_experiment_grid(
        h_fused_df=h_fused_df,
        save_path=save_path,
        columns=[
            "job_index",
            "group_index",
            "strategy",
            "n_bins",
            "min_support",
            "silhouette_score",
            "n_itemsets",
            "selected_features",
            "init",
        ],
        job_runner=run_without_ffs_job,
        strategies=strategies,
        n_bins_options=n_bins_options,
        supports=supports,
        n_clusters=n_clusters,
        random_state=random_state,
        baseline_score=baseline_score,
        resume=resume,
        workers=workers,
        limit=limit,
        label="without-FFS",
    )


def run_ffs(
    h_fused_df: pd.DataFrame,
    save_path: Path = FFS_RESULTS_PATH,
    strategies: tuple[DiscretizeStrategy, ...] = DEFAULT_STRATEGIES,
    n_bins_options: tuple[int, ...] = (3, 5, 7),
    supports: np.ndarray = DEFAULT_SUPPORTS,
    n_clusters: int = 5,
    random_state: int = RANDOM_STATE,
    baseline_score: float = 0.4069,
    resume: bool = True,
    workers: int = 3,
    limit: int | None = None,
) -> pd.DataFrame:
    """Run MiMvDEC with forward feature selection."""

    return run_experiment_grid(
        h_fused_df=h_fused_df,
        save_path=save_path,
        columns=[
            "job_index",
            "group_index",
            "strategy",
            "n_bins",
            "min_support",
            "silhouette_score",
            "n_itemsets",
            "selected_features",
            "init",
            "final_score",
            "best_observed_score",
        ],
        job_runner=run_ffs_job,
        strategies=strategies,
        n_bins_options=n_bins_options,
        supports=supports,
        n_clusters=n_clusters,
        random_state=random_state,
        baseline_score=baseline_score,
        resume=resume,
        workers=workers,
        limit=limit,
        label="FFS",
    )


def run_experiment_grid(
    h_fused_df: pd.DataFrame,
    save_path: Path,
    columns: list[str],
    job_runner: JobRunner,
    strategies: tuple[DiscretizeStrategy, ...],
    n_bins_options: tuple[int, ...],
    supports: np.ndarray,
    n_clusters: int,
    random_state: int,
    baseline_score: float,
    resume: bool,
    workers: int,
    limit: int | None,
    label: str,
) -> pd.DataFrame:
    """Run grouped sensitivity search and save CSV plus JSONL outputs."""

    existing_df = load_results(save_path, columns) if resume else empty_results(columns)
    groups = build_experiment_groups(strategies, n_bins_options, supports, limit)
    records: dict[int, WithoutFfsRecord | FfsRecord] = {}
    pending_groups = filter_pending_groups(groups, existing_df)

    logger.info(
        "Running {} {} groups with workers={}", len(pending_groups), label, workers
    )
    if workers <= 1:
        for group in pending_groups:
            for record in run_group_sequentially(
                h_fused_df,
                group,
                n_clusters,
                random_state,
                baseline_score,
                job_runner,
            ):
                records[record.job_index] = record
                save_results(existing_df, records, save_path, label)
    else:
        run_groups_parallel(
            h_fused_df=h_fused_df,
            groups=pending_groups,
            n_clusters=n_clusters,
            random_state=random_state,
            baseline_score=baseline_score,
            workers=workers,
            existing_df=existing_df,
            records=records,
            save_path=save_path,
            label=label,
            job_runner=job_runner,
        )

    return save_results(existing_df, records, save_path, label)


def build_experiment_groups(
    strategies: tuple[DiscretizeStrategy, ...],
    n_bins_options: tuple[int, ...],
    supports: np.ndarray,
    limit: int | None = None,
) -> list[ExperimentGroup]:
    """Build worker groups ordered by strategy, n_bins, then support."""

    groups = []
    job_index = 0
    for strategy in strategies:
        for n_bins in n_bins_options:
            support_jobs = []
            for min_support in supports:
                if limit is not None and job_index >= limit:
                    break

                support_jobs.append(
                    ExperimentJob(
                        job_index=job_index,
                        min_support=float(min_support),
                    )
                )
                job_index += 1

            if support_jobs:
                groups.append(
                    ExperimentGroup(
                        group_index=len(groups),
                        strategy=strategy,
                        n_bins=n_bins,
                        jobs=tuple(support_jobs),
                    )
                )

            if limit is not None and job_index >= limit:
                break

        if limit is not None and job_index >= limit:
            break

    return groups


def filter_pending_groups(
    groups: list[ExperimentGroup],
    existing_df: pd.DataFrame,
) -> list[ExperimentGroup]:
    """Remove completed support jobs while keeping group execution order."""

    pending_groups = []
    for group in groups:
        pending_jobs = tuple(
            job for job in group.jobs if not is_job_done(existing_df, job.job_index)
        )
        if pending_jobs:
            pending_groups.append(
                ExperimentGroup(
                    group_index=group.group_index,
                    strategy=group.strategy,
                    n_bins=group.n_bins,
                    jobs=pending_jobs,
                )
            )
    return pending_groups


def run_groups_parallel(
    h_fused_df: pd.DataFrame,
    groups: list[ExperimentGroup],
    n_clusters: int,
    random_state: int,
    baseline_score: float,
    workers: int,
    existing_df: pd.DataFrame,
    records: dict[int, WithoutFfsRecord | FfsRecord],
    save_path: Path,
    label: str,
    job_runner: JobRunner,
) -> None:
    """Run groups in parallel and save each support result from the main process."""

    import multiprocessing as mp

    with mp.Manager() as manager:
        queue = manager.Queue()
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    run_group_sequentially,
                    h_fused_df,
                    group,
                    n_clusters,
                    random_state,
                    baseline_score,
                    job_runner,
                    queue,
                ): group
                for group in groups
            }

            while futures:
                done, _ = wait(
                    futures,
                    timeout=0.2,
                    return_when=FIRST_COMPLETED,
                )
                drain_queue(queue, existing_df, records, save_path, label)

                for future in done:
                    futures.pop(future)
                    future.result()

                sleep(0.05)

            drain_queue(queue, existing_df, records, save_path, label)


def run_group_sequentially(
    h_fused_df: pd.DataFrame,
    group: ExperimentGroup,
    n_clusters: int,
    random_state: int,
    baseline_score: float,
    job_runner: JobRunner,
    queue: Any | None = None,
) -> list[WithoutFfsRecord | FfsRecord]:
    """Run support values sequentially for one strategy and bin group."""

    records = []
    for job in group.jobs:
        record = job_runner(
            h_fused_df,
            group,
            job,
            n_clusters,
            random_state,
            baseline_score,
        )
        records.append(record)

        if queue is not None:
            queue.put(record)

    return records


def run_without_ffs_job(
    h_fused_df: pd.DataFrame,
    group: ExperimentGroup,
    job: ExperimentJob,
    n_clusters: int,
    random_state: int,
    baseline_score: float,
) -> WithoutFfsRecord:
    """Run one support value without forward feature selection."""

    logger.info(
        "Running without-FFS: strategy={} n_bins={} min_support={}",
        group.strategy,
        group.n_bins,
        job.min_support,
    )
    fpmax_features = extract_fpmax_features(
        df=h_fused_df,
        n_bins=group.n_bins,
        bin_labels=BIN_LABELS_BY_SIZE[group.n_bins],
        strategy=group.strategy,
        min_support=job.min_support,
        drop_original_numeric=True,
    )
    if fpmax_features.features.empty:
        return WithoutFfsRecord(
            job_index=job.job_index,
            group_index=group.group_index,
            strategy=group.strategy,
            n_bins=group.n_bins,
            min_support=job.min_support,
            silhouette_score=baseline_score,
            n_itemsets=0,
            selected_features="",
            init=None,
        )

    clustering = run_kprototypes(
        continuous_df=h_fused_df,
        binary_df=fpmax_features.features,
        n_clusters=n_clusters,
        random_state=random_state,
        verbose=False,
    )

    return WithoutFfsRecord(
        job_index=job.job_index,
        group_index=group.group_index,
        strategy=group.strategy,
        n_bins=group.n_bins,
        min_support=job.min_support,
        silhouette_score=clustering.score,
        n_itemsets=len(fpmax_features.itemsets),
        selected_features=", ".join(clustering.binary_feature_names),
        init=clustering.init,
    )


def run_ffs_job(
    h_fused_df: pd.DataFrame,
    group: ExperimentGroup,
    job: ExperimentJob,
    n_clusters: int,
    random_state: int,
    baseline_score: float,
) -> FfsRecord:
    """Run one support value with forward feature selection."""

    logger.info(
        "Running FFS: strategy={} n_bins={} min_support={}",
        group.strategy,
        group.n_bins,
        job.min_support,
    )
    fpmax_features = extract_fpmax_features(
        df=h_fused_df,
        n_bins=group.n_bins,
        bin_labels=BIN_LABELS_BY_SIZE[group.n_bins],
        strategy=group.strategy,
        min_support=job.min_support,
        drop_original_numeric=True,
    )
    selected = run_forward_selection(
        continuous_df=h_fused_df,
        binary_df=fpmax_features.features,
        n_clusters=n_clusters,
        random_state=random_state,
        baseline_score=baseline_score,
        verbose=False,
    )
    score_to_save = (
        selected.best_observed_score
        if np.isfinite(selected.best_observed_score)
        else selected.score
    )

    return FfsRecord(
        job_index=job.job_index,
        group_index=group.group_index,
        strategy=group.strategy,
        n_bins=group.n_bins,
        min_support=job.min_support,
        silhouette_score=float(score_to_save),
        n_itemsets=len(fpmax_features.itemsets),
        selected_features=", ".join(selected.selected_feature_names),
        init=selected.init,
        final_score=selected.score,
        best_observed_score=selected.best_observed_score,
    )


def drain_queue(
    queue: Any,
    existing_df: pd.DataFrame,
    records: dict[int, WithoutFfsRecord | FfsRecord],
    save_path: Path,
    label: str,
) -> None:
    """Drain finished records from workers and save them immediately."""

    while True:
        try:
            record = queue.get_nowait()
        except Empty:
            break

        records[record.job_index] = record
        save_results(existing_df, records, save_path, label)
