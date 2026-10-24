"""Label-aware per-checkpoint metric trajectory for MvDEC diagnostics.

The trajectory is a pure observer: it never feeds back into training, does not
touch the global NumPy or TensorFlow random state, and is written to a sidecar
JSON-lines file outside the run manifest. It exists only to test the hypothesis
that a "best performance" row was picked from intermediate checkpoints, as the
DEKM release code prints ACC/NMI at every K-Means refresh.

Enable it by setting ``MVDEC_LABEL_TRAJECTORY_LOG`` to a file path. When the
variable is unset nothing is computed, so protocol contracts and config hashes
are unaffected either way.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from pipeline.external_metrics import compute_external_metrics

TRAJECTORY_ENV_VAR = "MVDEC_LABEL_TRAJECTORY_LOG"
VIEW_KMEANS_N_INIT = 10


def trajectory_path() -> Path | None:
    value = os.environ.get(TRAJECTORY_ENV_VAR, "").strip()
    return Path(value) if value else None


def record_checkpoint(
    path: Path,
    *,
    seed: int,
    step: int,
    n_clusters: int,
    true_labels: np.ndarray,
    fused_assignment: np.ndarray,
    h_view1: np.ndarray,
    h_view2: np.ndarray,
    n_changed: int,
) -> dict:
    """Append ACC/NMI of the fused refinement assignment and of each view."""

    row: dict = {"seed": int(seed), "step": int(step), "n_changed": int(n_changed)}
    fused = compute_external_metrics(
        true_labels, fused_assignment, n_clusters=n_clusters
    )
    row["fused_acc"], row["fused_nmi"] = fused.acc, fused.nmi
    for name, latent in (("view1", h_view1), ("view2", h_view2)):
        labels = KMeans(
            n_clusters=n_clusters, n_init=VIEW_KMEANS_N_INIT, random_state=seed
        ).fit_predict(latent)
        metrics = compute_external_metrics(true_labels, labels, n_clusters=n_clusters)
        row[f"{name}_acc"], row[f"{name}_nmi"] = metrics.acc, metrics.nmi
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    return row
