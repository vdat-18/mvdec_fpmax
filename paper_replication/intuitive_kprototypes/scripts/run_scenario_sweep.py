"""Sweep unspecified implementation choices to see which reproduce Tables 11-18.

The paper leaves several choices open (initial prototypes, mu orientation,
non-membership rule, lambda, gamma, empty-cluster handling, Soybean columns).
This script runs scenario grids over those choices and scores each scenario by
its mean absolute gap to the paper's reported accuracy, ARI and NMI.

Scenarios are only evidence about which reading is plausible. The paper's
numbers are the selection target, so a winning scenario must be checked on
datasets it was not picked on (see ``--holdout``), and must never be presented
as the authors' implementation.
"""

# ruff: noqa: E402,I001

from __future__ import annotations

import argparse
import itertools
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SANDBOX_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from intuitive_kprototypes import IntuitiveKPrototypes
from run_paper_benchmark import (
    DEFAULT_DATASETS,
    PAPER_CONFIGS,
    PAPER_NO_SELECT_ACCURACY_STD,
    PAPER_NO_SELECT_METRICS,
    PAPER_WITH_SELECT_METRICS,
)
from run_uci_smoke import (
    DEFAULT_DATA_DIR,
    _load_manifest,
    _split_features,
    clustering_accuracy,
)
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

METRICS = ("accuracy", "ari", "nmi")
BASELINE = {
    "mu_conv": "eq26",
    "init": "alg1",
    "lam": None,  # None keeps the paper-config lambda_init
    "nm": "paper",
    "gamma": None,  # None keeps the per-dataset paper gamma
    "max_iter": 30,
    "policy": "repair",
    "soy": "all",
    "soy_mu": None,  # None keeps the config mu; a value is passed to the code as-is
    "restarts": 1,  # fits per run; the lowest assigned-distance cost is kept
}
KNOB_VALUES = {
    "mu_conv": ["eq26", "swapped"],
    "init": ["alg1", "alg1rand", "random", "farthest"],
    "lam": [None, 0.5, 0.8, 1.0],
    "nm": ["paper", "table"],
    "gamma": [None, 0.0, 0.5, 1.0],
    "max_iter": [30, 100],
    "policy": ["repair", "retry"],
    "soy": ["all", "noconst"],
    "soy_mu": [None, 0.8],
    "restarts": [1, 5, 10, 20],
}


# Scenario knob -> model ``init_strategy``.
INIT_STRATEGY_BY_KNOB = {
    "alg1": "first",
    "alg1rand": "random_candidate",
    "random": "random",
    "farthest": "farthest",
}


def _dataset_key(key: str, scenario: dict[str, Any]) -> str:
    if key == "soybean_small" and scenario["soy"] == "noconst":
        return "soybean_small_no_constants"
    return key


def run_cell(task: tuple[dict[str, Any], str, int]) -> dict[str, Any]:
    """Run one (scenario, dataset) cell and return mean metrics."""

    scenario, key, runs = task
    manifest = _load_manifest(DEFAULT_DATA_DIR)
    data_key = _dataset_key(key, scenario)
    meta = manifest[data_key]
    frame = pd.read_csv(DEFAULT_DATA_DIR / meta["processed_csv"])
    X_num, X_cat, y = _split_features(frame, meta)
    config = PAPER_CONFIGS[key]
    mu, mu_convention = config.mu_param, scenario["mu_conv"]
    if key == "soybean_small" and scenario["soy_mu"] is not None:
        # Table 14 lists no mu: pass the chosen value to the code unchanged.
        mu, mu_convention = scenario["soy_mu"], "eq26"
    gamma = config.gamma if scenario["gamma"] is None else scenario["gamma"]
    lam = config.lambda_init if scenario["lam"] is None else scenario["lam"]

    n_clusters = int(meta["n_clusters"])

    def fit_once(seed_base: int):
        """One fit; the retry policy reseeds until converged and non-degenerate."""

        result = None
        for attempt in range(20 if scenario["policy"] == "retry" else 1):
            est = IntuitiveKPrototypes(
                n_clusters=n_clusters,
                lambda_init=lam,
                mu_param=mu,
                gamma=gamma,
                beta=config.beta,
                max_iter=scenario["max_iter"],
                random_state=seed_base + attempt,
                strict_init=False,
                non_membership=scenario["nm"],
                empty_cluster_policy="farthest",
                min_cluster_size=1,
                init_strategy=INIT_STRATEGY_BY_KNOB[scenario["init"]],
                mu_convention=mu_convention,
            )
            try:
                result = est.fit(X_num=X_num, X_cat=X_cat).result_
            except (ValueError, RuntimeError):
                result = None
                continue
            sizes = np.bincount(result.labels, minlength=n_clusters)
            if scenario["policy"] != "retry" or (
                result.converged and sizes.min() >= 2
            ):
                break
        return result

    def assigned_cost(result) -> float:
        rows_idx = np.arange(len(result.labels))
        return float(result.distances[rows_idx, result.labels].sum())

    rows = []
    for run in range(runs):
        # Restarts keep the lowest-cost fit, as in the benchmark script.
        fits = [
            fit_once(42 + run * 1000 + offset * 50)
            for offset in range(scenario["restarts"])
        ]
        fits = [fit for fit in fits if fit is not None]
        if not fits:
            continue
        result = min(fits, key=assigned_cost)
        sizes = np.bincount(result.labels, minlength=n_clusters)
        rows.append(
            {
                "accuracy": clustering_accuracy(y, result.labels),
                "ari": adjusted_rand_score(y, result.labels),
                "nmi": normalized_mutual_info_score(y, result.labels),
                "collapsed": int(sizes.min() <= 2),
            }
        )
    # Random initial prototypes correspond to the paper's plain
    # "Intuitive-K-prototypes" row; Algorithm 1 to "with select prototypes".
    paper = (
        PAPER_NO_SELECT_METRICS if scenario["init"] == "random"
        else PAPER_WITH_SELECT_METRICS
    )[key]
    out: dict[str, Any] = {"dataset": key, "runs_ok": len(rows)}
    if not rows:
        return {**out, "gap": np.nan, "collapsed": np.nan,
                **{m: np.nan for m in METRICS}}
    for m in METRICS:
        out[m] = float(np.mean([r[m] for r in rows]))
    out["collapsed"] = int(sum(r["collapsed"] for r in rows))
    out["gap"] = float(np.mean([abs(out[m] - paper[m]) for m in METRICS]))
    out["paper_accuracy"] = paper["accuracy"]
    out["accuracy_std"] = float(np.std([r["accuracy"] for r in rows], ddof=1))
    out["paper_accuracy_std"] = (
        PAPER_NO_SELECT_ACCURACY_STD[key] if scenario["init"] == "random" else np.nan
    )
    return out


def build_scenarios(grid: str) -> list[dict[str, Any]]:
    if grid == "ofat":
        scenarios = [dict(BASELINE)]
        for knob, values in KNOB_VALUES.items():
            for value in values:
                if value == BASELINE[knob]:
                    continue
                scenarios.append({**BASELINE, knob: value})
        return scenarios
    if grid == "full":
        names = ["mu_conv", "init", "lam", "nm", "gamma", "soy"]
        return [
            {**BASELINE, **dict(zip(names, combo, strict=True))}
            for combo in itertools.product(*(KNOB_VALUES[n] for n in names))
        ]
    if grid == "init":
        return [
            {**BASELINE, "mu_conv": "swapped", "soy": "noconst", "soy_mu": 0.8,
             "init": init}
            for init in KNOB_VALUES["init"]
        ]
    if grid == "restarts":
        base = {**BASELINE, "mu_conv": "swapped", "soy": "noconst", "soy_mu": 0.8}
        return [
            {**base, "init": init, "restarts": restarts}
            for init in ("random", "farthest")
            for restarts in (1, 5, 10)
        ]
    if grid == "rows":
        names = ["mu_conv", "init", "soy"]
        return [
            {**BASELINE, **dict(zip(names, combo, strict=True))}
            for combo in itertools.product(
                KNOB_VALUES["mu_conv"], KNOB_VALUES["init"], KNOB_VALUES["soy"]
            )
        ]
    if grid == "focus":
        names = ["mu_conv", "nm", "init", "soy", "gamma"]
        values = {
            "mu_conv": KNOB_VALUES["mu_conv"],
            "nm": KNOB_VALUES["nm"],
            "init": KNOB_VALUES["init"],
            "soy": KNOB_VALUES["soy"],
            "gamma": [None, 0.5],
        }
        return [
            {**BASELINE, **dict(zip(names, combo, strict=True))}
            for combo in itertools.product(*(values[n] for n in names))
        ]
    msg = f"Unknown grid {grid!r}."
    raise ValueError(msg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--grid",
        choices=["ofat", "focus", "rows", "init", "restarts", "full"],
        default="ofat",
    )
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--datasets", nargs="*", default=list(DEFAULT_DATASETS),
        help="Datasets to score on.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=SANDBOX_DIR / "output" / "scenarios"
    )
    args = parser.parse_args()

    scenarios = build_scenarios(args.grid)
    tasks = [(s, k, args.runs) for s in scenarios for k in args.datasets]
    print(f"{len(scenarios)} scenarios x {len(args.datasets)} datasets "
          f"x {args.runs} runs", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        cells = list(pool.map(run_cell, tasks, chunksize=1))

    rows = []
    for (scenario, _key, _runs), cell in zip(tasks, cells, strict=True):
        rows.append({**{f"s_{k}": v for k, v in scenario.items()}, **cell})
    frame = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / f"cells_{args.grid}.csv", index=False)

    knob_cols = [f"s_{k}" for k in BASELINE]
    summary = (
        frame.groupby(knob_cols, dropna=False)
        .agg(mean_gap=("gap", "mean"), max_gap=("gap", "max"),
             collapsed=("collapsed", "sum"))
        .reset_index()
        .sort_values("mean_gap")
    )
    summary.to_csv(args.output_dir / f"summary_{args.grid}.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(summary.head(15).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
