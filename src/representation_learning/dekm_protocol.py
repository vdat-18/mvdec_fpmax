"""Pure DEKM 2021 released-code protocol helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

DEKM_MODERN_PROTOCOL_ID = "dekm_modern_deterministic_v1"
DEKM_RELEASE_2021_PROTOCOL_ID = "dekm_2021_release_seeded_modern_runtime_v1"
DEKM_RELEASED_PROTOCOL_ID = DEKM_MODERN_PROTOCOL_ID


@dataclass(frozen=True)
class DekmConfig:
    """Runtime settings that define one reproducible DEKM experiment."""

    protocol_id: str = DEKM_MODERN_PROTOCOL_ID
    kernel_initializer: str = "glorot_uniform"
    eigen_solver: str = "symmetric_eigh"
    kmeans_random_state: str = "seeded"
    deterministic_runtime: bool = True
    data_shuffle: str = "direct_seed"
    pretrain_epochs: int = 200
    pretrain_batch_size: int = 256
    refinement_batch_size: int = 256
    update_interval: int = 10
    max_refinement_steps: int = 14_000
    assignment_change_tolerance: float = 0.001
    initial_kmeans_n_init: int = 100
    latent_dim: int = 10

    def validate(self) -> None:
        """Reject settings that cannot represent a valid DEKM run."""

        positive_fields = {
            "pretrain_epochs": self.pretrain_epochs,
            "pretrain_batch_size": self.pretrain_batch_size,
            "refinement_batch_size": self.refinement_batch_size,
            "update_interval": self.update_interval,
            "max_refinement_steps": self.max_refinement_steps,
            "initial_kmeans_n_init": self.initial_kmeans_n_init,
            "latent_dim": self.latent_dim,
        }
        invalid = [name for name, value in positive_fields.items() if value < 1]
        if invalid:
            raise ValueError(f"DEKM settings must be positive: {invalid}.")
        if self.protocol_id not in {
            DEKM_MODERN_PROTOCOL_ID,
            DEKM_RELEASE_2021_PROTOCOL_ID,
        }:
            raise ValueError(f"Unsupported DEKM protocol: {self.protocol_id}.")
        if self.kernel_initializer not in {"glorot_uniform", "uniform"}:
            raise ValueError("Unsupported DEKM kernel initializer.")
        if self.eigen_solver not in {"symmetric_eigh", "general_eig"}:
            raise ValueError("Unsupported DEKM eigen solver.")
        if self.kmeans_random_state not in {"seeded", "global"}:
            raise ValueError("Unsupported DEKM K-Means random-state policy.")
        if self.data_shuffle not in {"direct_seed", "release_random_0_99"}:
            raise ValueError("Unsupported DEKM data-shuffle policy.")
        if not 0 <= self.assignment_change_tolerance < 1:
            msg = "assignment_change_tolerance must be in the range [0, 1)."
            raise ValueError(msg)

    def contract(self) -> dict[str, object]:
        """Return the immutable released-code behavior plus runtime values."""

        return {
            "protocol_id": self.protocol_id,
            "architecture": {
                "encoder_units": [500, 500, 2000, self.latent_dim],
                "decoder_units": [2000, 500, 500],
                "hidden_activation": "relu",
                "kernel_initializer": self.kernel_initializer,
            },
            "pretraining": {
                "objective": "reconstruction_mse_mean_dimensions",
                "optimizer": "adam_default",
                "shuffle_buffer": 8000,
            },
            "refinement": {
                "objective": "frozen_transformed_embedding_mse",
                "optimizer": "adam_default",
                "batching": "sequential_wraparound",
                "eigenvalue_order": "ascending",
                "eigen_solver": self.eigen_solver,
                "greedy_eigen_direction": "largest_last",
                "kmeans_refresh_policy": "dynamic_2x_previous_iterations",
                "kmeans_random_state": self.kmeans_random_state,
            },
            "randomness": {
                "deterministic_runtime": self.deterministic_runtime,
                "data_shuffle": self.data_shuffle,
            },
            "runtime": asdict(self),
        }


def config_hash(config: DekmConfig) -> str:
    """Return a stable identifier for one DEKM runtime contract."""

    serialized = json.dumps(config.contract(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def sorted_eigendecomposition(
    matrix: np.ndarray,
    solver: str = "symmetric_eigh",
) -> tuple[np.ndarray, np.ndarray]:
    """Return ascending eigenpairs using the selected DEKM implementation."""

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("The DEKM scatter matrix must be square.")
    if solver == "symmetric_eigh":
        symmetric = (values + values.T) / 2
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    elif solver == "general_eig":
        eigenvalues, eigenvectors = np.linalg.eig(values)
        eigenvalues = np.real_if_close(eigenvalues).real
        eigenvectors = np.real_if_close(eigenvectors).real
    else:
        raise ValueError(f"Unsupported DEKM eigen solver: {solver}.")
    order = np.argsort(eigenvalues)
    return eigenvalues[order], eigenvectors[:, order]


def within_cluster_scatter(
    embeddings: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    """Compute the within-cluster scatter matrix used by released DEKM."""

    embeddings = np.asarray(embeddings, dtype=float)
    labels = np.asarray(labels, dtype=int)
    centroids = np.asarray(centroids, dtype=float)
    if embeddings.ndim != 2 or centroids.ndim != 2:
        raise ValueError("Embeddings and centroids must be two-dimensional.")
    if len(embeddings) != len(labels):
        raise ValueError("Embeddings and labels must contain the same rows.")
    if embeddings.shape[1] != centroids.shape[1]:
        raise ValueError("Embeddings and centroids must share their latent dimension.")

    scatter = np.zeros((embeddings.shape[1], embeddings.shape[1]), dtype=float)
    for cluster_index in range(len(centroids)):
        residuals = embeddings[labels == cluster_index] - centroids[cluster_index]
        scatter += residuals.T @ residuals
    return scatter


def build_frozen_greedy_target(
    transformed_embeddings: np.ndarray,
    transformed_centroids: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Replace only the released code's last/largest transformed dimension."""

    target = np.array(transformed_embeddings, dtype=float, copy=True)
    centroids = np.asarray(transformed_centroids, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if target.ndim != 2 or centroids.ndim != 2:
        raise ValueError("Transformed embeddings and centroids must be 2D.")
    if len(target) != len(labels):
        raise ValueError("Transformed embeddings and labels must have equal rows.")
    target[:, -1] = centroids[labels, -1]
    return target


def assignment_change_count(
    previous_labels: np.ndarray,
    current_labels: np.ndarray,
    n_clusters: int,
) -> int:
    """Count changes after optimally aligning the previous cluster identifiers."""

    previous = np.asarray(previous_labels, dtype=int)
    current = np.asarray(current_labels, dtype=int)
    if previous.shape != current.shape or previous.ndim != 1:
        raise ValueError("Cluster assignments must be equal-length 1D arrays.")
    if np.any((current < 0) | (current >= n_clusters)):
        raise ValueError("Current cluster assignments are outside the cluster range.")
    if np.any((previous < 0) | (previous >= n_clusters)):
        return int(len(current))

    contingency = np.zeros((n_clusters, n_clusters), dtype=np.int64)
    np.add.at(contingency, (current, previous), 1)
    current_ids, previous_ids = linear_sum_assignment(-contingency)
    previous_to_current = dict(
        zip(previous_ids.tolist(), current_ids.tolist(), strict=True)
    )
    aligned_previous = np.asarray(
        [previous_to_current[label] for label in previous],
        dtype=int,
    )
    return int(np.count_nonzero(current != aligned_previous))


def sequential_batch_indices(
    n_samples: int,
    batch_size: int,
    batch_index: int,
) -> tuple[np.ndarray, int]:
    """Return one released-code sequential batch and the next wrapped index."""

    if n_samples < 1 or batch_size < 1 or batch_index < 0:
        raise ValueError("n_samples, batch_size, and batch_index must be valid.")
    start = batch_index * batch_size
    if start >= n_samples:
        start = 0
        batch_index = 0
    stop = min(start + batch_size, n_samples)
    next_index = batch_index + 1 if stop < n_samples else 0
    return np.arange(start, stop, dtype=int), next_index
