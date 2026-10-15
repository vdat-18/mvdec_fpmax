"""Persist row-aligned clustering outputs for the labeled public benchmarks."""

import pickle
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd

from pipeline.external_metrics import ExternalMetrics
from pipeline.io import replace_with_retry

PUBLIC_ARTIFACT_SCHEMA_VERSION = 1

_ALLOWED_PICKLE_GLOBALS = {
    ("numpy", "dtype"): np.dtype,
    ("numpy", "ndarray"): np.ndarray,
    ("numpy._core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
    ("numpy._core.multiarray", "scalar"): np._core.multiarray.scalar,
    # NumPy 1.x used this module path. NumPy 2 keeps it load-compatible.
    ("numpy.core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
    ("numpy.core.multiarray", "scalar"): np._core.multiarray.scalar,
}


class _PublicArtifactUnpickler(pickle.Unpickler):
    """Deserialize artifact containers without importing arbitrary globals."""

    def find_class(self, module: str, name: str) -> object:
        allowed = _ALLOWED_PICKLE_GLOBALS.get((module, name))
        if allowed is None:
            msg = f"Pickle global {module}.{name} is forbidden in public artifacts."
            raise pickle.UnpicklingError(msg)
        return allowed


def _load_restricted_pickle(file: BinaryIO) -> object:
    payload = _PublicArtifactUnpickler(file).load()
    if file.read(1):
        msg = "Public artifact contains trailing bytes after its pickle payload."
        raise pickle.UnpicklingError(msg)
    return payload


def load_public_artifact(path: Path) -> object:
    """Load one NumPy artifact while rejecting executable pickle globals."""

    with path.open("rb") as file:
        return _load_restricted_pickle(file)


def restore_original_order(values: object, shuffled_indices: object) -> np.ndarray:
    """Restore shuffled row values to canonical DEKM release order."""

    array = np.asarray(values)
    indices = np.asarray(shuffled_indices, dtype=int)
    if array.shape[0] != len(indices):
        msg = "Values and shuffled_indices must contain the same number of rows."
        raise ValueError(msg)
    if not np.array_equal(np.sort(indices), np.arange(len(indices))):
        msg = "shuffled_indices must be a permutation of 0..n_samples - 1."
        raise ValueError(msg)
    restored = np.empty_like(array)
    restored[indices] = array
    return restored


def public_run_stem(method: str, dataset: str, seed: int | None, run: int) -> str:
    """Return a collision-free filename stem for one benchmark run."""

    seed_part = f"seed_{seed}" if seed is not None else f"run_{run:02d}"
    return f"{method.lower()}_{dataset.lower()}_{seed_part}"


def public_assignment_frame(
    *,
    dataset: str,
    method: str,
    labels: np.ndarray,
    true_labels: np.ndarray,
    random_seed: int | None,
) -> pd.DataFrame:
    """Build a canonical row-level assignment table."""

    if len(labels) != len(true_labels):
        msg = "Predicted and true labels must contain the same number of rows."
        raise ValueError(msg)
    return pd.DataFrame(
        {
            "dataset": dataset,
            "method": method,
            "sample_index": np.arange(len(labels), dtype=int),
            "random_seed": random_seed,
            "cluster": np.asarray(labels, dtype=int),
            "true_label": np.asarray(true_labels),
        }
    )


def build_public_artifact(
    *,
    dataset: str,
    method: str,
    n_clusters: int,
    random_seed: int | None,
    source_sha256: str,
    labels: np.ndarray,
    true_labels: np.ndarray,
    metrics: ExternalMetrics,
    representation: np.ndarray | None = None,
    representation_key: str = "representation",
) -> dict[str, object]:
    """Build the shared public benchmark artifact payload."""

    labels = np.asarray(labels, dtype=int)
    true_labels = np.asarray(true_labels)
    if len(labels) != len(true_labels):
        msg = "Public artifact labels and true_labels must have equal length."
        raise ValueError(msg)
    artifact: dict[str, object] = {
        "schema_version": PUBLIC_ARTIFACT_SCHEMA_VERSION,
        "algorithm": method,
        "dataset": dataset,
        "n_clusters": int(n_clusters),
        "n_samples": int(len(labels)),
        "random_seed": random_seed,
        "source_sha256": source_sha256,
        "labels": labels,
        "true_labels": true_labels,
        "row_indices": np.arange(len(labels), dtype=int),
        "acc": metrics.acc,
        "nmi": metrics.nmi,
        "config": {
            "seed": random_seed,
            "n_clusters": int(n_clusters),
        },
    }
    if representation is not None:
        values = np.asarray(representation)
        if values.ndim != 2 or len(values) != len(labels):
            msg = "Public artifact representation must match label rows."
            raise ValueError(msg)
        artifact[representation_key] = values
        artifact["representation_key"] = representation_key
    return artifact


def write_public_artifact(payload: dict[str, object], path: Path) -> None:
    """Atomically write one trusted local public benchmark pickle."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    with temporary_path.open("wb") as file:
        pickle.dump(payload, file)
    replace_with_retry(temporary_path, path)


def write_public_frame(frame: pd.DataFrame, path: Path) -> None:
    """Atomically write one public benchmark CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    frame.to_csv(temporary_path, index=False)
    replace_with_retry(temporary_path, path)


def write_public_assignments(frame: pd.DataFrame, path: Path) -> None:
    """Atomically write one row-level public assignment CSV."""

    write_public_frame(frame, path)
