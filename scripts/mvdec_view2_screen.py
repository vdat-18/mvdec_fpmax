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


def pretrain_view2_latent(
    mvdec, x, config: dict, seed: int
) -> tuple[np.ndarray, float]:
    """Pretrain the S1 View2 alone and return its latent and final loss."""

    import tensorflow as tf

    protocol = mvdec.MIN_REFINEMENT_FROZEN_VIEW1_PROTOCOL
    mvdec.hidden_units = config["latent"]
    mvdec.view2_base_units = config["width"]
    mvdec.configure_deterministic_runtime(seed)
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
    return latent, float(history.history["loss"][-1])


def load_shuffled_reuters(mvdec, dataset_root: Path, seed: int):
    """Load REUTERS with the release L2 normalisation and a seeded row shuffle."""

    import tensorflow as tf

    from pipeline.public_data import load_dekm_public_dataset

    dataset = load_dekm_public_dataset(
        "REUTERS", dataset_root, normalization=mvdec.RELEASE_L2_NORMALIZATION
    )
    rows = tf.random.shuffle(np.arange(len(dataset.X)), seed=seed).numpy()
    mvdec.input_shape = dataset.X.shape[1]
    return dataset.X[rows], dataset.y[rows]


def screen_config(mvdec, x, y, config: dict, seed: int, n_clusters: int) -> dict:
    from pipeline.external_metrics import compute_external_metrics

    started = time.time()
    latent, final_loss = pretrain_view2_latent(mvdec, x, config, seed)
    labels = KMeans(n_clusters=n_clusters, n_init=100, random_state=seed).fit_predict(
        latent
    )
    metrics = compute_external_metrics(y, labels, n_clusters=n_clusters)
    return {
        **config,
        "seed": seed,
        "final_loss": final_loss,
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

    x, y = load_shuffled_reuters(mvdec, args.dataset_root, args.seed)
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
