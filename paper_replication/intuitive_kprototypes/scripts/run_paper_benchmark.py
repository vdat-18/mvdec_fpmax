"""Run a 50-run UCI benchmark shaped like Tables 11-18 of the paper.

This script is intentionally separate from ``run_uci_smoke.py``. The smoke
script answers "does it run?"; this script answers "how close are we to the
paper's repeated-run tables?"

The paper does not publish source code, seeds, or every preprocessing choice,
so the output should be read as an experiment-level replication audit rather
than an exact reproduction guarantee.
"""

# ruff: noqa: E402,I001

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    fowlkes_mallows_score,
    normalized_mutual_info_score,
)

SANDBOX_DIR = Path(__file__).resolve().parents[1]
SANDBOX_SRC = SANDBOX_DIR / "src"
sys.path.insert(0, str(SANDBOX_SRC))

from intuitive_kprototypes import IntuitiveKPrototypes  # noqa: E402
from run_uci_smoke import (  # noqa: E402
    DEFAULT_DATA_DIR,
    _load_manifest,
    _split_features,
    clustering_accuracy,
    object_precision_recall_f1,
)


METRIC_NAMES = ("precision", "recall", "accuracy", "f1", "ari", "fmi", "nmi")


@dataclass(frozen=True)
class DatasetConfig:
    """Paper-facing experiment config for one Table 10 dataset variant."""

    key: str
    table: str
    lambda_init: float = 0.3
    mu_param: float = 0.5
    gamma: float = 0.5
    beta: float = 2.0


PAPER_CONFIGS: dict[str, DatasetConfig] = {
    "iris": DatasetConfig(
        key="iris",
        table="Table 11",
        gamma=0.0,
    ),
    "ionosphere": DatasetConfig(
        key="ionosphere",
        table="Table 12",
        gamma=0.0,
    ),
    "bcw": DatasetConfig(
        key="bcw",
        table="Table 13",
        mu_param=0.2,
        gamma=0.4,
    ),
    "soybean_small": DatasetConfig(
        key="soybean_small",
        table="Table 14",
        gamma=0.5,
    ),
    "soybean_small_no_constants": DatasetConfig(
        key="soybean_small_no_constants",
        table="Table 14 alternate",
        gamma=0.5,
    ),
    "australian_credit_approval": DatasetConfig(
        key="australian_credit_approval",
        table="Table 15",
        mu_param=0.2,
        gamma=0.7,
    ),
    "heart_cleveland_2": DatasetConfig(
        key="heart_cleveland_2",
        table="Table 16",
        mu_param=0.3,
        gamma=0.2,
    ),
    "chess_ten_fifteen": DatasetConfig(
        key="chess_ten_fifteen",
        table="Table 17",
        mu_param=0.2,
        gamma=0.0,
    ),
    "heart_cleveland_5": DatasetConfig(
        key="heart_cleveland_5",
        table="Table 18",
        mu_param=0.3,
        gamma=0.5,
    ),
}


PAPER_WITH_SELECT_METRICS: dict[str, dict[str, float]] = {
    "iris": {
        "precision": 0.960,
        "recall": 0.960,
        "accuracy": 0.960,
        "f1": 0.960,
        "ari": 0.886,
        "fmi": 0.923,
        "nmi": 0.864,
    },
    "ionosphere": {
        "precision": 0.836,
        "recall": 0.730,
        "accuracy": 0.800,
        "f1": 0.780,
        "ari": 0.334,
        "fmi": 0.740,
        "nmi": 0.275,
    },
    "bcw": {
        "precision": 0.949,
        "recall": 0.939,
        "accuracy": 0.950,
        "f1": 0.944,
        "ari": 0.807,
        "fmi": 0.914,
        "nmi": 0.697,
    },
    "soybean_small": {
        "precision": 1.000,
        "recall": 1.000,
        "accuracy": 1.000,
        "f1": 1.000,
        "ari": 1.000,
        "fmi": 1.000,
        "nmi": 1.000,
    },
    "australian_credit_approval": {
        "precision": 0.859,
        "recall": 0.862,
        "accuracy": 0.856,
        "f1": 0.861,
        "ari": 0.507,
        "fmi": 0.755,
        "nmi": 0.429,
    },
    "heart_cleveland_2": {
        "precision": 0.828,
        "recall": 0.820,
        "accuracy": 0.825,
        "f1": 0.824,
        "ari": 0.420,
        "fmi": 0.715,
        "nmi": 0.333,
    },
    "chess_ten_fifteen": {
        "precision": 0.774,
        "recall": 0.751,
        "accuracy": 0.756,
        "f1": 0.763,
        "ari": 0.347,
        "fmi": 0.683,
        "nmi": 0.316,
    },
    "heart_cleveland_5": {
        "precision": 0.371,
        "recall": 0.357,
        "accuracy": 0.547,
        "f1": 0.365,
        "ari": 0.356,
        "fmi": 0.580,
        "nmi": 0.240,
    },
}


DEFAULT_DATASETS = (
    "iris",
    "ionosphere",
    "bcw",
    "soybean_small",
    "australian_credit_approval",
    "heart_cleveland_2",
    "chess_ten_fifteen",
    "heart_cleveland_5",
)


def _metric_row(y_true: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    precision, recall, f1 = object_precision_recall_f1(y_true, labels)
    return {
        "precision": precision,
        "recall": recall,
        "accuracy": clustering_accuracy(y_true, labels),
        "f1": f1,
        "ari": adjusted_rand_score(y_true, labels),
        "fmi": fowlkes_mallows_score(y_true, labels),
        "nmi": normalized_mutual_info_score(y_true, labels),
    }


def _run_single_seed(
    *,
    config: DatasetConfig,
    meta: dict[str, Any],
    X_num: np.ndarray | None,
    X_cat: np.ndarray | None,
    y_true: np.ndarray,
    seed: int,
    max_iter: int,
    restarts_per_run: int,
    empty_cluster_policy: str,
    require_converged: bool,
    min_cluster_size: int,
) -> dict[str, Any]:
    best_result = None
    best_seed = None
    best_cost = np.inf
    best_fit_time = np.nan
    total_fit_time = 0.0

    for offset in range(restarts_per_run):
        current_seed = seed + offset
        model = IntuitiveKPrototypes(
            n_clusters=int(meta["n_clusters"]),
            lambda_init=config.lambda_init,
            mu_param=config.mu_param,
            gamma=config.gamma,
            beta=config.beta,
            max_iter=max_iter,
            random_state=current_seed,
            strict_init=False,
            empty_cluster_policy=empty_cluster_policy,
            min_cluster_size=min_cluster_size,
        )
        result = model.fit(X_num=X_num, X_cat=X_cat).result_
        total_fit_time += float(result.fit_time_seconds)
        cost = float(
            result.distances[np.arange(len(result.labels)), result.labels].sum()
        )
        if cost < best_cost:
            best_result = result
            best_seed = current_seed
            best_cost = cost
            best_fit_time = float(result.fit_time_seconds)

    labels = best_result.labels
    cluster_sizes = np.bincount(labels, minlength=int(meta["n_clusters"]))
    if require_converged and not best_result.converged:
        msg = "Run did not converge."
        raise ValueError(msg)
    if int(cluster_sizes.min()) < min_cluster_size:
        msg = (
            f"Run has a cluster smaller than min_cluster_size={min_cluster_size}: "
            f"{cluster_sizes.tolist()}."
        )
        raise ValueError(msg)
    return {
        "seed": best_seed,
        "cost": best_cost,
        "converged": bool(best_result.converged),
        "n_iter": int(best_result.n_iter),
        "fit_time_seconds": best_fit_time,
        "total_restart_fit_time_seconds": total_fit_time,
        "cluster_sizes": cluster_sizes.tolist(),
        **_metric_row(y_true, labels),
    }


def _summarize(
    *,
    config: DatasetConfig,
    meta: dict[str, Any],
    rows: list[dict[str, Any]],
    failures: list[str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "dataset": config.key,
        "table": config.table,
        "runs_ok": len(rows),
        "runs_failed": len(failures),
        "rows": int(meta["n_samples"]),
        "numeric": len(meta["numeric_cols"]),
        "categorical": len(meta["categorical_cols"]),
        "k": int(meta["n_clusters"]),
        "lambda_init": config.lambda_init,
        "mu_param": config.mu_param,
        "gamma": config.gamma,
        "beta": config.beta,
        "converged_runs": sum(bool(row["converged"]) for row in rows),
        "singleton_runs": sum(min(row["cluster_sizes"]) <= 1 for row in rows),
        "attempts": sum(int(row.get("attempts", 1)) for row in rows) + len(failures),
        "rejected_attempts": len(failures),
    }
    fit_times = np.asarray([row["fit_time_seconds"] for row in rows], dtype=float)
    total_fit_times = np.asarray(
        [row["total_restart_fit_time_seconds"] for row in rows],
        dtype=float,
    )
    summary["fit_time_seconds_mean"] = (
        float(fit_times.mean()) if len(fit_times) else np.nan
    )
    summary["fit_time_seconds_std"] = (
        float(fit_times.std(ddof=0)) if len(fit_times) else np.nan
    )
    summary["total_restart_fit_time_seconds"] = (
        float(total_fit_times.sum()) if len(total_fit_times) else np.nan
    )
    for name in METRIC_NAMES:
        values = np.asarray([row[name] for row in rows], dtype=float)
        summary[f"{name}_mean"] = float(values.mean()) if len(values) else np.nan
        summary[f"{name}_std"] = float(values.std(ddof=0)) if len(values) else np.nan

        paper = PAPER_WITH_SELECT_METRICS.get(config.key, {}).get(name)
        summary[f"{name}_paper"] = paper if paper is not None else np.nan
        summary[f"{name}_delta"] = (
            summary[f"{name}_mean"] - paper if paper is not None else np.nan
        )
    return summary


def _format_summary(summary: dict[str, Any]) -> str:
    return (
        f"{summary['dataset']}: ok={summary['runs_ok']}, "
        f"failed={summary['runs_failed']}, "
        f"time={summary['fit_time_seconds_mean']:.3f}s, "
        f"AC={summary['accuracy_mean']:.3f}"
        f"({summary['accuracy_delta']:+.3f}), "
        f"ARI={summary['ari_mean']:.3f}"
        f"({summary['ari_delta']:+.3f}), "
        f"NMI={summary['nmi_mean']:.3f}"
        f"({summary['nmi_delta']:+.3f})"
    )


def _write_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(summaries[0]) if summaries else []
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True))
            file.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory produced by fetch_paper_uci_datasets.py.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        help=(
            "Dataset key to run. May be repeated. Use 'all' for every "
            "paper-configured dataset."
        ),
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=50,
        help="Number of repeated runs per dataset.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument(
        "--restarts-per-run",
        type=int,
        default=1,
        help="Optional cost-based restarts inside each repeated run.",
    )
    parser.add_argument(
        "--empty-cluster-policy",
        choices=["raise", "farthest"],
        default="farthest",
        help=(
            "Recovery policy for real benchmark runs. The estimator default "
            "is 'raise'; this benchmark defaults to 'farthest' because the "
            "paper does not specify empty-cluster handling."
        ),
    )
    parser.add_argument(
        "--valid-runs-only",
        action="store_true",
        help=(
            "Retry seeds until each recorded run converges and satisfies "
            "--min-cluster-size. This is useful for auditing how much of the "
            "paper gap is caused by singleton or non-converged runs."
        ),
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=2,
        help="Minimum cluster size when --valid-runs-only is enabled.",
    )
    parser.add_argument(
        "--max-attempts-per-run",
        type=int,
        default=100,
        help="Maximum seed attempts allowed to produce one recorded run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SANDBOX_DIR / "output",
        help="Directory for CSV/JSONL benchmark artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.runs < 1:
        msg = "--runs must be at least 1."
        raise ValueError(msg)
    if args.restarts_per_run < 1:
        msg = "--restarts-per-run must be at least 1."
        raise ValueError(msg)
    if args.min_cluster_size < 1:
        msg = "--min-cluster-size must be at least 1."
        raise ValueError(msg)
    if args.max_attempts_per_run < 1:
        msg = "--max-attempts-per-run must be at least 1."
        raise ValueError(msg)

    data_dir = args.data_dir.resolve()
    manifest = _load_manifest(data_dir)
    if args.dataset == ["all"]:
        dataset_keys = list(DEFAULT_DATASETS)
    else:
        dataset_keys = args.dataset or list(DEFAULT_DATASETS)

    unknown = sorted(set(dataset_keys) - set(PAPER_CONFIGS))
    if unknown:
        msg = f"Unknown benchmark dataset keys: {unknown}."
        raise ValueError(msg)

    summaries: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for key in dataset_keys:
        if key not in manifest:
            msg = f"Dataset {key!r} is missing from {data_dir / 'manifest.json'}."
            raise ValueError(msg)

        config = PAPER_CONFIGS[key]
        meta = manifest[key]
        frame = pd.read_csv(data_dir / meta["processed_csv"])
        X_num, X_cat, y_true = _split_features(frame, meta)

        rows: list[dict[str, Any]] = []
        failures: list[str] = []
        for run_idx in range(args.runs):
            row = None
            seed_stride = (
                args.max_attempts_per_run
                if args.valid_runs_only
                else args.restarts_per_run
            )
            first_seed = args.seed + run_idx * seed_stride
            for attempt in range(1, args.max_attempts_per_run + 1):
                seed = first_seed + attempt - 1
                try:
                    row = _run_single_seed(
                        config=config,
                        meta=meta,
                        X_num=X_num,
                        X_cat=X_cat,
                        y_true=y_true,
                        seed=seed,
                        max_iter=args.max_iter,
                        restarts_per_run=args.restarts_per_run,
                        empty_cluster_policy=args.empty_cluster_policy,
                        require_converged=args.valid_runs_only,
                        min_cluster_size=(
                            args.min_cluster_size if args.valid_runs_only else 1
                        ),
                    )
                except ValueError as exc:
                    failures.append(f"run={run_idx + 1}, seed={seed}: {exc}")
                    if args.valid_runs_only:
                        continue
                    break
                row["attempts"] = attempt
                break
            if row is None:
                continue
            row = {"dataset": key, "run": run_idx + 1, **row}
            rows.append(row)
            detail_rows.append(row)

        summary = _summarize(config=config, meta=meta, rows=rows, failures=failures)
        summaries.append(summary)
        print(_format_summary(summary))
        if failures:
            print(f"  first failure: {failures[0]}")

    output_dir = args.output_dir.resolve()
    _write_csv(output_dir / "paper_benchmark_summary.csv", summaries)
    _write_jsonl(output_dir / "paper_benchmark_runs.jsonl", detail_rows)
    print(f"Wrote {output_dir / 'paper_benchmark_summary.csv'}")
    print(f"Wrote {output_dir / 'paper_benchmark_runs.jsonl'}")


if __name__ == "__main__":
    main()
