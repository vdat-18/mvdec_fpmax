"""Screen standalone View2 pretraining settings (controlled diagnostic only).

Trains the S1 true-bottleneck View2 autoencoder on its own, with no View1 and no
refinement, then scores K-Means on its latent against the labels. Labels are used
for evaluation only, so the numbers say which pretraining setting makes View2 a
stronger view; they are not a reproduction protocol. Each ``--config`` is a
comma-separated ``key=value`` list with keys ``epochs`` (default 200), ``lr``
(default 0.001), ``latent`` (default 10) and ``width`` (View2 stem, default 64).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {"epochs": 200, "lr": 0.001, "latent": 10, "width": 64}
KEY_TYPES = {"epochs": int, "lr": float, "latent": int, "width": int}


def parse_config(text: str) -> dict:
    """Parse ``epochs=1000,lr=0.0001`` into a full config dict."""

    config = dict(DEFAULTS)
    for item in filter(None, text.split(",")):
        key, _, value = item.partition("=")
        if key not in KEY_TYPES or not value:
            raise ValueError(f"Unsupported View2 screen setting: {item!r}.")
        config[key] = KEY_TYPES[key](value)
    return config


def screen_config(mvdec, x, y, config: dict, seed: int, n_clusters: int) -> dict:
    import tensorflow as tf

    from pipeline.external_metrics import compute_external_metrics

    protocol = mvdec.MIN_REFINEMENT_FROZEN_VIEW1_PROTOCOL
    mvdec.hidden_units = config["latent"]
    mvdec.view2_base_units = config["width"]
    mvdec.configure_deterministic_runtime(seed)
    started = time.time()
    model = mvdec.model_view2(
        load_weights=False, architecture_id=protocol.view2_architecture_id
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["lr"]),
        loss=mvdec.pretraining_loss(protocol),
    )
    dataset = mvdec.make_pretraining_dataset(x, seed, protocol=protocol)
    history = model.fit(dataset, epochs=config["epochs"], verbose=0, shuffle=False)
    latent = model.predict(x, batch_size=1024, verbose=0)[:, : config["latent"]]
    labels = KMeans(n_clusters=n_clusters, n_init=100, random_state=seed).fit_predict(
        latent
    )
    metrics = compute_external_metrics(y, labels, n_clusters=n_clusters)
    return {
        **config,
        "seed": seed,
        "final_loss": float(history.history["loss"][-1]),
        "view2_acc": float(metrics.acc),
        "view2_nmi": float(metrics.nmi),
        "seconds": round(time.time() - started, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dataset-root", type=Path, default="external_repos/DEKM/datasets"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configs = [parse_config(text) for text in args.config]

    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src" / "representation_learning")]
    import MVDEC_dense as mvdec
    import tensorflow as tf

    from pipeline.public_data import load_dekm_public_dataset

    dataset = load_dekm_public_dataset(
        "REUTERS", args.dataset_root, normalization=mvdec.RELEASE_L2_NORMALIZATION
    )
    rows = tf.random.shuffle(np.arange(len(dataset.X)), seed=args.seed).numpy()
    x, y = dataset.X[rows], dataset.y[rows]
    mvdec.input_shape = x.shape[1]
    results = []
    for config in configs:
        result = screen_config(mvdec, x, y, config, args.seed, n_clusters=4)
        print(json.dumps(result), flush=True)
        results.append(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "claim_scope": "pretrain_only_screen_not_reproduction",
                    "runs": results,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
