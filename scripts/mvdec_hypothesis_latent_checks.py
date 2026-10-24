"""CPU hypothesis checks on saved MvDEC S2 latents (controlled diagnostic only).

Every check reads the saved ``h_view1``/``h_view2``/``h_fused`` of finished runs and
asks whether a reporting or evaluation choice the paper leaves open (NMI
normalisation, ACC definition, Fig. 2 concatenation, K-Means restart selection,
independent best-ACC/best-NMI selection) can explain MvDEC Table 2. Labels are
used only for evaluation; label-guided selections are reported as such.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

PAPER = {
    "REUTERS": {"view1": (0.7642, 0.5999), "view2": (0.6501, 0.5129), "fused": (0.7813, 0.6141)},
    "RCV1": {"view1": (0.6212, 0.3598), "view2": (0.6235, 0.3659), "fused": (0.7193, 0.4885)},
    "20NEWS": {"view1": (0.4120, 0.4376), "view2": (0.4086, 0.4032), "fused": (0.4163, 0.4543)},
}
NMI_METHODS = ("arithmetic", "geometric", "min", "max")


def hungarian_acc(truth: np.ndarray, pred: np.ndarray) -> float:
    _, t = np.unique(truth, return_inverse=True)
    _, p = np.unique(pred, return_inverse=True)
    table = np.zeros((p.max() + 1, t.max() + 1), dtype=np.int64)
    np.add.at(table, (p, t), 1)
    rows, cols = linear_sum_assignment(-table)
    return float(table[rows, cols].sum() / len(truth))


def purity(truth: np.ndarray, pred: np.ndarray) -> float:
    _, t = np.unique(truth, return_inverse=True)
    _, p = np.unique(pred, return_inverse=True)
    table = np.zeros((p.max() + 1, t.max() + 1), dtype=np.int64)
    np.add.at(table, (p, t), 1)
    return float(table.max(axis=1).sum() / len(truth))


def kmeans(h: np.ndarray, k: int, seed: int, n_init: int) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit_predict(h)


def analyse_run(path: Path, restarts: int) -> dict:
    art = pickle.loads(path.read_bytes())
    y = np.asarray(art["true_labels"])
    k = int(art["n_clusters"])
    seed = int(art["config"]["seed"])
    out: dict = {"seed": seed, "dataset": art["dataset"]}
    reps = {
        "view1": art["h_view1"],
        "view2": art["h_view2"],
        "fused": art["h_fused"],
        "concat_fig2": art["view_concat_representation"],
    }
    for name, h in reps.items():
        labels = kmeans(h, k, seed, 100)
        row = {"acc": hungarian_acc(y, labels), "purity": purity(y, labels)}
        for method in NMI_METHODS:
            row[f"nmi_{method}"] = float(
                normalized_mutual_info_score(y, labels, average_method=method)
            )
        out[name] = row
    # Stored final fused assignment (what the run reported).
    out["fused_stored"] = {
        "acc": hungarian_acc(y, art["labels"]),
        "nmi_arithmetic": float(normalized_mutual_info_score(y, art["labels"])),
    }
    # Label-guided best K-Means restart (n_init=1) on the fused latent.
    accs, nmis = [], []
    for r in range(restarts):
        labels = kmeans(art["h_fused"], k, r, 1)
        accs.append(hungarian_acc(y, labels))
        nmis.append(float(normalized_mutual_info_score(y, labels)))
    out["fused_kmeans_restarts"] = {
        "n": restarts,
        "max_acc": max(accs),
        "max_nmi": max(nmis),
        "nmi_at_max_acc": nmis[int(np.argmax(accs))],
    }
    return out


def summarise(runs: list[dict]) -> dict:
    summary: dict = {}
    for rep in ("view1", "view2", "fused", "concat_fig2"):
        block = {}
        for key in ("acc", "purity", *(f"nmi_{m}" for m in NMI_METHODS)):
            vals = np.array([r[rep][key] for r in runs])
            block[key] = {"mean": float(vals.mean()), "max": float(vals.max())}
        summary[rep] = block
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Folder with seed_*/artifact.pkl")
    parser.add_argument("--restarts", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = [analyse_run(p, args.restarts) for p in sorted(args.root.glob("seed_*/artifact.pkl"))]
    dataset = runs[0]["dataset"]
    result = {"dataset": dataset, "paper": PAPER[dataset], "runs": runs, "summary": summarise(runs)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=1))


if __name__ == "__main__":
    main()
