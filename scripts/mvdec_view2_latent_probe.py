"""Probe a wider View2 latent beside the frozen DEKM_F View1 (diagnostic only).

MvDEC does not state View2's latent size and Eq. 4 averages the two latents, so a
different size cannot be tested through joint refinement. This post-hoc probe
pretrains View2 alone with the given ``--latent`` size, takes the frozen DEKM_F
View1 latent, and fuses them as a weighted concatenation. The weight ``0.5`` is
label-free; the best weight over the grid is picked with labels and is a ceiling,
not a reproduction protocol.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from mvdec_fusion_weight_sweep import fuse, unit_rms
from mvdec_hypothesis_latent_checks import PAPER, hungarian_acc, kmeans
from mvdec_view2_screen import load_shuffled_reuters, pretrain_view2_latent
from sklearn.metrics import normalized_mutual_info_score

ROOT = Path(__file__).resolve().parents[1]
VIEW1_LATENT = 10
LABEL_FREE_WEIGHT = 0.5


def score(y: np.ndarray, h: np.ndarray, seed: int) -> dict:
    labels = kmeans(h, len(np.unique(y)), seed, n_init=20)
    return {
        "acc": hungarian_acc(y, labels),
        "nmi": float(normalized_mutual_info_score(y, labels)),
    }


def probe_seed(h1: np.ndarray, h2: np.ndarray, y: np.ndarray, seed: int, steps: int):
    h1, h2 = unit_rms(h1), unit_rms(h2)
    rows = [
        {"w_view1": round(float(weight), 4)}
        | score(y, fuse(h1, h2, float(weight), "concat"), seed)
        for weight in np.linspace(0.0, 1.0, steps)
    ]
    return {
        "seed": seed,
        "view1": score(y, h1, seed),
        "view2": score(y, h2, seed),
        "label_free": next(r for r in rows if r["w_view1"] == LABEL_FREE_WEIGHT),
        "best_nmi_label_guided": max(rows, key=lambda row: row["nmi"]),
        "rows": rows,
    }


def frozen_view1_latent(mvdec, x: np.ndarray, weights: Path) -> np.ndarray:
    mvdec.hidden_units = VIEW1_LATENT
    model = mvdec.model_view1(weights_path=weights)
    return model.predict(x, batch_size=1024, verbose=0)[:, :VIEW1_LATENT]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view1-weights", type=Path, required=True)
    parser.add_argument("--latent", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--steps", type=int, default=21)
    parser.add_argument(
        "--dataset-root", type=Path, default="external_repos/DEKM/datasets"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src" / "representation_learning")]
    import MVDEC_dense as mvdec

    x, y = load_shuffled_reuters(mvdec, args.dataset_root, seed=42)
    h1 = frozen_view1_latent(mvdec, x, args.view1_weights)
    config = {"epochs": args.epochs, "lr": 0.001, "latent": args.latent, "width": 64}
    runs = []
    for seed in args.seed:
        h2, _ = pretrain_view2_latent(mvdec, x, config, seed)
        runs.append(probe_seed(h1, h2, y, seed, args.steps))
        print(
            json.dumps({k: v for k, v in runs[-1].items() if k != "rows"}), flush=True
        )
    result = {
        "claim_scope": "post_hoc_probe_not_reproduction",
        "paper": PAPER["REUTERS"],
        "config": config,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
