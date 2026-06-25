"""Run Intuitive-K-prototypes on fetched UCI paper datasets.

Run the downloader first:

    uv run --link-mode=copy --with ucimlrepo python \
        paper_replication/intuitive_kprototypes/scripts/fetch_paper_uci_datasets.py

Then run this smoke script from the repository root:

    uv run python paper_replication/intuitive_kprototypes/scripts/run_uci_smoke.py

These runs are sanity checks. The paper reports 50-run averages and does not
publish source code, seeds, or every preprocessing detail, so exact metric
equality is not expected from this script.
"""

# ruff: noqa: E402,I001

from __future__ import annotations

import argparse
import json
import sys
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    fowlkes_mallows_score,
    normalized_mutual_info_score,
    pair_confusion_matrix,
    precision_recall_fscore_support,
)

SANDBOX_DIR = Path(__file__).resolve().parents[1]
SANDBOX_SRC = SANDBOX_DIR / "src"
sys.path.insert(0, str(SANDBOX_SRC))

from intuitive_kprototypes import IntuitiveKPrototypes  # noqa: E402


DEFAULT_DATA_DIR = SANDBOX_DIR / "data" / "uci"
DEFAULT_DATASETS = (
    "iris",
    "bcw",
    "australian_credit_approval",
    "heart_cleveland_2",
)


def _load_manifest(data_dir: Path) -> dict[str, dict[str, Any]]:
    path = data_dir / "manifest.json"
    if not path.exists():
        msg = (
            f"Missing manifest: {path}. Run fetch_paper_uci_datasets.py first."
        )
        raise FileNotFoundError(msg)
    with path.open("r", encoding="utf-8") as file:
        rows = json.load(file)
    return {row["key"]: row for row in rows}


def _split_features(
    frame: pd.DataFrame,
    meta: dict[str, Any],
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray]:
    numeric_cols = meta["numeric_cols"]
    categorical_cols = meta["categorical_cols"]
    target_col = meta["target_col"]

    X_num = (
        frame[numeric_cols].to_numpy(dtype=float)
        if numeric_cols
        else None
    )
    X_cat = (
        frame[categorical_cols].astype(str).to_numpy(dtype=object)
        if categorical_cols
        else None
    )
    y_true = frame[target_col].astype(str).to_numpy()
    return X_num, X_cat, y_true


def clustering_accuracy(y_true: np.ndarray, labels: np.ndarray) -> float:
    """Best label-permutation accuracy for small-k clustering outputs."""

    _, score = _best_mapped_labels(y_true, labels)
    return score


def _best_mapped_labels(
    y_true: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Map cluster ids to class labels with maximum object-level accuracy."""

    true_values = np.unique(y_true)
    pred_values = np.unique(labels)
    n_classes = max(len(true_values), len(pred_values))

    true_to_idx = {value: idx for idx, value in enumerate(true_values)}
    pred_to_idx = {value: idx for idx, value in enumerate(pred_values)}
    contingency = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_true, labels, strict=True):
        contingency[pred_to_idx[pred], true_to_idx[true]] += 1

    best = 0
    best_perm: tuple[int, ...] | None = None
    for perm in permutations(range(n_classes)):
        score = sum(
            contingency[pred_idx, true_idx]
            for pred_idx, true_idx in enumerate(perm)
        )
        if score > best:
            best = score
            best_perm = perm

    fallback = true_values[0]
    mapped = np.empty(len(labels), dtype=true_values.dtype)
    for idx, pred in enumerate(labels):
        pred_idx = pred_to_idx[pred]
        if best_perm is not None and pred_idx < len(best_perm):
            true_idx = best_perm[pred_idx]
            mapped[idx] = (
                true_values[true_idx] if true_idx < len(true_values) else fallback
            )
        else:
            mapped[idx] = fallback
    return mapped, best / len(y_true)


def object_precision_recall_f1(
    y_true: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float, float]:
    """Macro PR/RE/F1 after best cluster-to-label mapping.

    The paper describes TP/FP/FN over data objects rather than object pairs.
    """

    mapped, _ = _best_mapped_labels(y_true, labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        mapped,
        average="macro",
        zero_division=0,
    )
    return float(precision), float(recall), float(f1)


def pair_precision_recall_f1(
    y_true: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float, float]:
    pair_counts = pair_confusion_matrix(y_true, labels)
    tn, fp, fn, tp = pair_counts.ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return float(precision), float(recall), float(f1)


def _run_one(
    *,
    data_dir: Path,
    meta: dict[str, Any],
    lambda_init: float,
    mu_param: float,
    gamma: float,
    beta: float,
    max_iter: int,
    seed: int,
    restarts: int,
) -> dict[str, Any]:
    frame = pd.read_csv(data_dir / meta["processed_csv"])
    X_num, X_cat, y_true = _split_features(frame, meta)

    best_result = None
    best_seed = None
    best_cost = np.inf
    errors: list[str] = []
    for offset in range(restarts):
        current_seed = seed + offset
        model = IntuitiveKPrototypes(
            n_clusters=int(meta["n_clusters"]),
            lambda_init=lambda_init,
            mu_param=mu_param,
            gamma=gamma,
            beta=beta,
            max_iter=max_iter,
            random_state=current_seed,
            strict_init=False,
        )
        try:
            result = model.fit(X_num=X_num, X_cat=X_cat).result_
        except ValueError as exc:
            errors.append(f"seed={current_seed}: {exc}")
            continue
        cost = float(
            result.distances[np.arange(len(result.labels)), result.labels].sum()
        )
        if cost < best_cost:
            best_result = result
            best_seed = current_seed
            best_cost = cost

    if best_result is None:
        msg = "All restarts failed:\n" + "\n".join(errors)
        raise RuntimeError(msg)

    labels = best_result.labels
    precision, recall, f1 = object_precision_recall_f1(y_true, labels)
    cluster_sizes = np.bincount(labels, minlength=int(meta["n_clusters"]))
    return {
        "dataset": meta["key"],
        "seed": best_seed,
        "rows": len(y_true),
        "numeric": len(meta["numeric_cols"]),
        "categorical": len(meta["categorical_cols"]),
        "k": int(meta["n_clusters"]),
        "converged": bool(best_result.converged),
        "n_iter": int(best_result.n_iter),
        "fit_time_seconds": float(best_result.fit_time_seconds),
        "cost": best_cost,
        "cluster_sizes": cluster_sizes.tolist(),
        "accuracy": clustering_accuracy(y_true, labels),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ari": adjusted_rand_score(y_true, labels),
        "fmi": fowlkes_mallows_score(y_true, labels),
        "nmi": normalized_mutual_info_score(y_true, labels),
    }


def _format_row(row: dict[str, Any]) -> str:
    return (
        f"{row['dataset']}: rows={row['rows']}, "
        f"num={row['numeric']}, cat={row['categorical']}, k={row['k']}, "
        f"seed={row['seed']}, converged={row['converged']}, "
        f"iter={row['n_iter']}, time={row['fit_time_seconds']:.3f}s, "
        f"sizes={row['cluster_sizes']}, "
        f"AC={row['accuracy']:.3f}, PR={row['precision']:.3f}, "
        f"RE={row['recall']:.3f}, F1={row['f1']:.3f}, "
        f"ARI={row['ari']:.3f}, FMI={row['fmi']:.3f}, NMI={row['nmi']:.3f}"
    )


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
            "manifest dataset. Defaults to a small smoke subset."
        ),
    )
    parser.add_argument("--lambda-init", type=float, default=0.3)
    parser.add_argument("--mu-param", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--restarts", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    manifest = _load_manifest(data_dir)
    if args.dataset == ["all"]:
        dataset_keys = list(manifest)
    else:
        dataset_keys = args.dataset or list(DEFAULT_DATASETS)

    unknown = sorted(set(dataset_keys) - set(manifest))
    if unknown:
        msg = f"Unknown dataset keys: {unknown}. Available: {sorted(manifest)}"
        raise ValueError(msg)

    for key in dataset_keys:
        try:
            row = _run_one(
                data_dir=data_dir,
                meta=manifest[key],
                lambda_init=args.lambda_init,
                mu_param=args.mu_param,
                gamma=args.gamma,
                beta=args.beta,
                max_iter=args.max_iter,
                seed=args.seed,
                restarts=args.restarts,
            )
        except RuntimeError as exc:
            print(f"{key}: FAILED - {exc}")
            continue
        print(_format_row(row))


if __name__ == "__main__":
    main()
