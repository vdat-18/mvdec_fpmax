"""Label-guided ceiling of weighted two-view fusion on saved MvDEC latents.

Controlled diagnostic only. For each finished run, both views are centred and
scaled to unit RMS, then fused for every weight ``w`` in [0, 1] either as a
weighted average ``w*h1 + (1-w)*h2`` or as a weighted concatenation
``[sqrt(w)*h1, sqrt(1-w)*h2]`` (K-Means on it minimises ``w*d1^2 + (1-w)*d2^2``).
Labels pick the best ``w`` per run, so the result is an upper bound for any
fixed-weight linear fusion of these two views, not a reproduction protocol.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from mvdec_hypothesis_latent_checks import PAPER, hungarian_acc, kmeans
from sklearn.metrics import normalized_mutual_info_score

FUSIONS = ("average", "concat")


def unit_rms(h: np.ndarray) -> np.ndarray:
    centred = h - h.mean(axis=0)
    return centred / np.sqrt(np.mean(centred**2))


def fuse(h1: np.ndarray, h2: np.ndarray, weight: float, fusion: str) -> np.ndarray:
    if fusion == "average":
        return weight * h1 + (1.0 - weight) * h2
    return np.concatenate([np.sqrt(weight) * h1, np.sqrt(1.0 - weight) * h2], axis=1)


def sweep_run(path: Path, weights: np.ndarray, n_init: int) -> dict:
    art = pickle.loads(path.read_bytes())
    y = np.asarray(art["true_labels"])
    k = int(art["n_clusters"])
    seed = int(art["config"]["seed"])
    h1, h2 = unit_rms(art["h_view1"]), unit_rms(art["h_view2"])
    rows = []
    for fusion in FUSIONS:
        for weight in weights:
            labels = kmeans(fuse(h1, h2, float(weight), fusion), k, seed, n_init)
            rows.append(
                {
                    "fusion": fusion,
                    "w_view1": round(float(weight), 4),
                    "acc": hungarian_acc(y, labels),
                    "nmi": float(normalized_mutual_info_score(y, labels)),
                }
            )
    return {"run": path.parent.name, "dataset": art["dataset"], "rows": rows}


def ceiling(runs: list[dict], paper_fused: tuple[float, float]) -> dict:
    rows = [row | {"run": run["run"]} for run in runs for row in run["rows"]]
    acc, nmi = paper_fused
    return {
        "best_nmi": max(rows, key=lambda row: row["nmi"]),
        "best_acc": max(rows, key=lambda row: row["acc"]),
        "points": len(rows),
        "points_reaching_paper_fused": sum(
            row["acc"] >= acc and row["nmi"] >= nmi for row in rows
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots", type=Path, nargs="+", help="Folders searched for */artifact.pkl"
    )
    parser.add_argument("--steps", type=int, default=21)
    parser.add_argument("--n-init", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    weights = np.linspace(0.0, 1.0, args.steps)
    paths = sorted(p for root in args.roots for p in root.rglob("artifact.pkl"))
    runs = [sweep_run(path, weights, args.n_init) for path in paths]
    dataset = runs[0]["dataset"]
    result = {
        "dataset": dataset,
        "paper": PAPER[dataset],
        "claim_scope": "label_guided_ceiling_not_reproduction",
        "ceiling": ceiling(runs, PAPER[dataset]["fused"]),
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps({"paper": result["paper"], "ceiling": result["ceiling"]}, indent=1)
    )


if __name__ == "__main__":
    main()
