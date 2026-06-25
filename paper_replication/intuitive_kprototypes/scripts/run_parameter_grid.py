"""Run a small parameter grid for difficult UCI replication datasets."""

# ruff: noqa: E402,I001

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import sys
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

SANDBOX_DIR = Path(__file__).resolve().parents[1]
SANDBOX_SRC = SANDBOX_DIR / "src"
sys.path.insert(0, str(SANDBOX_SRC))

from intuitive_kprototypes import IntuitiveKPrototypes  # noqa: E402
from run_paper_benchmark import PAPER_WITH_SELECT_METRICS  # noqa: E402
from run_uci_smoke import (  # noqa: E402
    DEFAULT_DATA_DIR,
    _load_manifest,
    _split_features,
    clustering_accuracy,
)


DEFAULT_DATASETS = (
    "soybean_small",
    "australian_credit_approval",
    "heart_cleveland_2",
    "chess_ten_fifteen",
    "heart_cleveland_5",
)
PARAMETER_GRIDS: dict[str, dict[str, list[float]]] = {
    "quick": {
        "mu_param": [0.3, 0.5, 0.8],
        "gamma": [0.0, 0.5, 1.0],
        "lambda_init": [0.2, 0.3, 0.5],
        "beta": [2.0, 3.0],
    },
    "balanced": {
        "mu_param": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0],
        "gamma": [0.0, 0.2, 0.5, 0.7, 1.0],
        "lambda_init": [0.1, 0.2, 0.3, 0.5, 0.8, 1.0],
        "beta": [1.5, 2.0, 3.0, 5.0],
    },
    "paper": {
        "mu_param": [0.1, 0.2, 0.3, 0.4],
        "gamma": [0.0, 0.2, 0.4, 0.5, 0.7, 1.0],
        "lambda_init": [0.3],
        "beta": [2.0],
    },
}


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _grid_values(args: argparse.Namespace, name: str) -> list[float]:
    override = getattr(args, name)
    if override is not None:
        return _parse_float_list(override)
    return PARAMETER_GRIDS[args.grid][name]


def _run_one(
    *,
    meta: dict[str, Any],
    X_num: np.ndarray | None,
    X_cat: np.ndarray | None,
    y_true: np.ndarray,
    params: dict[str, float],
    seed: int,
    max_iter: int,
    min_cluster_size: int,
    valid_runs_only: bool,
) -> dict[str, Any]:
    model = IntuitiveKPrototypes(
        n_clusters=int(meta["n_clusters"]),
        lambda_init=params["lambda_init"],
        mu_param=params["mu_param"],
        gamma=params["gamma"],
        beta=params["beta"],
        max_iter=max_iter,
        random_state=seed,
        strict_init=False,
        empty_cluster_policy="farthest",
        min_cluster_size=min_cluster_size,
    )
    result = model.fit(X_num=X_num, X_cat=X_cat).result_
    sizes = np.bincount(result.labels, minlength=int(meta["n_clusters"]))
    if valid_runs_only and not result.converged:
        msg = "Run did not converge."
        raise ValueError(msg)
    if valid_runs_only and int(sizes.min()) < min_cluster_size:
        msg = f"Run has small clusters: {sizes.tolist()}."
        raise ValueError(msg)
    return {
        "accuracy": clustering_accuracy(y_true, result.labels),
        "ari": adjusted_rand_score(y_true, result.labels),
        "nmi": normalized_mutual_info_score(y_true, result.labels),
        "converged": bool(result.converged),
        "min_cluster": int(sizes.min()),
        "fit_time_seconds": float(result.fit_time_seconds),
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "runs_ok": 0,
            "converged_runs": 0,
            "singleton_runs": 0,
            "accuracy_mean": np.nan,
            "ari_mean": np.nan,
            "nmi_mean": np.nan,
            "fit_time_seconds_mean": np.nan,
        }
    return {
        "runs_ok": len(rows),
        "converged_runs": sum(bool(row["converged"]) for row in rows),
        "singleton_runs": sum(int(row["min_cluster"]) <= 1 for row in rows),
        "accuracy_mean": float(np.mean([row["accuracy"] for row in rows])),
        "ari_mean": float(np.mean([row["ari"] for row in rows])),
        "nmi_mean": float(np.mean([row["nmi"] for row in rows])),
        "fit_time_seconds_mean": float(
            np.mean([row["fit_time_seconds"] for row in rows])
        ),
    }


def _with_paper_deltas(row: dict[str, Any]) -> dict[str, Any]:
    paper = PAPER_WITH_SELECT_METRICS.get(str(row["dataset"]), {})
    metrics = ("accuracy", "ari", "nmi")
    result = dict(row)
    squared: list[float] = []
    for metric in metrics:
        value = float(row[f"{metric}_mean"])
        target = paper.get(metric)
        result[f"{metric}_paper"] = target if target is not None else np.nan
        if target is None or not np.isfinite(value):
            result[f"{metric}_delta"] = np.nan
            continue
        delta = value - target
        result[f"{metric}_delta"] = delta
        squared.append(delta * delta)
    result["paper_distance"] = (
        float(np.sqrt(np.mean(squared))) if squared else np.nan
    )
    return result


def _run_grid_task(task: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    data_dir = Path(task["data_dir"])
    key = task["dataset"]
    params = task["params"]
    meta = task["meta"]
    frame = pd.read_csv(data_dir / meta["processed_csv"])
    X_num, X_cat, y_true = _split_features(frame, meta)

    rows: list[dict[str, Any]] = []
    failures = 0
    for run_idx in range(task["runs"]):
        first_seed = task["seed"] + run_idx * task["max_attempts_per_run"]
        for attempt in range(task["max_attempts_per_run"]):
            try:
                row = _run_one(
                    meta=meta,
                    X_num=X_num,
                    X_cat=X_cat,
                    y_true=y_true,
                    params=params,
                    seed=first_seed + attempt,
                    max_iter=task["max_iter"],
                    min_cluster_size=task["min_cluster_size"],
                    valid_runs_only=task["valid_runs_only"],
                )
            except ValueError:
                failures += 1
                if task["valid_runs_only"]:
                    continue
                break
            rows.append(row)
            break

    summary = {
        "dataset": key,
        **params,
        "failures": failures,
        **_summarize(rows),
    }
    return int(task["index"]), _with_paper_deltas(summary)


def _best_report(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    report: list[dict[str, Any]] = []
    datasets = sorted({str(row["dataset"]) for row in rows})
    for dataset in datasets:
        current = [
            row for row in rows if row["dataset"] == dataset and row["runs_ok"] > 0
        ]
        if not current:
            continue
        selectors = {
            "closest_to_paper": lambda row: -float(row["paper_distance"]),
            "best_accuracy": lambda row: float(row["accuracy_mean"]),
            "best_ari": lambda row: float(row["ari_mean"]),
            "best_nmi": lambda row: float(row["nmi_mean"]),
        }
        seen: set[tuple[Any, ...]] = set()
        for rank_name, selector in selectors.items():
            best = max(current, key=selector)
            key = (
                dataset,
                best["mu_param"],
                best["gamma"],
                best["lambda_init"],
                best["beta"],
                rank_name,
            )
            if key in seen:
                continue
            seen.add(key)
            report.append({"rank": rank_name, **best})
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--dataset", action="append")
    parser.add_argument(
        "--grid",
        choices=sorted(PARAMETER_GRIDS),
        default="quick",
        help=(
            "Named parameter grid. Individual parameter arguments override "
            "the selected preset."
        ),
    )
    parser.add_argument("--mu-param")
    parser.add_argument("--gamma")
    parser.add_argument("--lambda-init")
    parser.add_argument("--beta")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--min-cluster-size", type=int, default=2)
    parser.add_argument("--max-attempts-per-run", type=int, default=50)
    parser.add_argument("--valid-runs-only", action="store_true")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help=(
            "Number of worker processes for concurrent grid execution. "
            "Use 1 for deterministic sequential execution."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=SANDBOX_DIR / "output" / "parameter_grid.csv",
    )
    parser.add_argument(
        "--best-report-csv",
        type=Path,
        default=SANDBOX_DIR / "output" / "best_config_report.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_jobs < 1:
        msg = "--n-jobs must be at least 1."
        raise ValueError(msg)
    data_dir = args.data_dir.resolve()
    manifest = _load_manifest(data_dir)
    datasets = args.dataset or list(DEFAULT_DATASETS)
    combos = list(
        product(
            _grid_values(args, "mu_param"),
            _grid_values(args, "gamma"),
            _grid_values(args, "lambda_init"),
            _grid_values(args, "beta"),
        )
    )

    tasks: list[dict[str, Any]] = []
    task_index = 0
    for key in datasets:
        meta = manifest[key]
        for mu_param, gamma, lambda_init, beta in combos:
            params = {
                "mu_param": mu_param,
                "gamma": gamma,
                "lambda_init": lambda_init,
                "beta": beta,
            }
            tasks.append(
                {
                    "index": task_index,
                    "data_dir": str(data_dir),
                    "dataset": key,
                    "meta": meta,
                    "params": params,
                    "runs": args.runs,
                    "seed": args.seed,
                    "max_iter": args.max_iter,
                    "min_cluster_size": args.min_cluster_size,
                    "max_attempts_per_run": args.max_attempts_per_run,
                    "valid_runs_only": args.valid_runs_only,
                }
            )
            task_index += 1

    indexed_rows: list[tuple[int, dict[str, Any]]] = []
    if args.n_jobs == 1:
        for task in tasks:
            indexed_rows.append(_run_grid_task(task))
    else:
        with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
            futures = [executor.submit(_run_grid_task, task) for task in tasks]
            for future in as_completed(futures):
                indexed_rows.append(future.result())

    output_rows = [row for _, row in sorted(indexed_rows, key=lambda item: item[0])]
    for summary in output_rows:
        print(
            f"{summary['dataset']} mu={summary['mu_param']} "
            f"gamma={summary['gamma']}: "
            f"ok={summary['runs_ok']} AC={summary['accuracy_mean']:.3f} "
            f"ARI={summary['ari_mean']:.3f} NMI={summary['nmi_mean']:.3f} "
            f"time={summary['fit_time_seconds_mean']:.3f}s"
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {args.output_csv.resolve()}")

    best_rows = _best_report(output_rows)
    args.best_report_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.best_report_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(best_rows[0]))
        writer.writeheader()
        writer.writerows(best_rows)
    print(f"Wrote {args.best_report_csv.resolve()}")


if __name__ == "__main__":
    main()
