"""Run the DEKM 2021 dense released-code protocol reproducibly."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.cluster import KMeans

from config import DEKM_DATASET_DIR, PUBLIC_BENCHMARK_OUTPUT_DIR
from pipeline.external_metrics import compute_external_metrics
from pipeline.public_artifacts import (
    build_public_artifact,
    public_assignment_frame,
    restore_original_order,
    write_public_artifact,
    write_public_assignments,
    write_public_frame,
)
from pipeline.public_data import PublicDataset, load_dekm_public_dataset
from representation_learning.dekm_protocol import (
    DEKM_MODERN_PROTOCOL_ID,
    DEKM_RELEASE_2021_PROTOCOL_ID,
    DekmConfig,
    assignment_change_count,
    build_frozen_greedy_target,
    config_hash,
    sequential_batch_indices,
    sorted_eigendecomposition,
    within_cluster_scatter,
)

OFFICIAL_REPOSITORY = "https://github.com/spdj2271/DEKM"
DETERMINISTIC_RUNTIME_POLICY = "keras_seeded_tf_deterministic_v1"


@dataclass(frozen=True)
class RefinementResult:
    """Final released-code refinement state for one shuffled run."""

    embedding: np.ndarray
    labels: np.ndarray
    training_log: pd.DataFrame
    steps_completed: int
    refreshes_completed: int
    stop_reason: str


def parse_args() -> argparse.Namespace:
    """Parse the DEKM reproduction command line."""

    parser = argparse.ArgumentParser(
        description="Reproduce the DEKM 2021 dense public benchmark protocol."
    )
    parser.add_argument("dataset", choices=("REUTERS", "20NEWS", "RCV1"))
    parser.add_argument(
        "--protocol",
        choices=("modern-deterministic", "release-2021"),
        default="modern-deterministic",
    )
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-root", type=Path, default=DEKM_DATASET_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PUBLIC_BENCHMARK_OUTPUT_DIR / "dekm",
    )
    parser.add_argument("--pretrain-epochs", type=int, default=200)
    parser.add_argument("--pretrain-batch-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--update-interval", type=int, default=10)
    parser.add_argument("--max-refinement-steps", type=int, default=14_000)
    parser.add_argument("--assignment-tolerance", type=float)
    parser.add_argument("--initial-kmeans-n-init", type=int, default=100)
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail before training when TensorFlow cannot see a GPU.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing completed seed directory.",
    )
    return parser.parse_args()


def import_tensorflow() -> Any:
    """Import the optional GPU dependency with an actionable error."""

    try:
        import tensorflow as tf
    except ImportError as error:
        msg = (
            "TensorFlow is required for DEKM representation learning. "
            "Install it in the GPU runtime before running this command."
        )
        raise RuntimeError(msg) from error
    return tf


def enable_tensorflow_determinism(tf: Any) -> None:
    """Require deterministic TensorFlow kernels for repeatable GPU runs."""

    tf.config.experimental.enable_op_determinism()


def build_config(args: argparse.Namespace) -> DekmConfig:
    """Build the selected modern or release-faithful DEKM contract."""

    common = {
        "pretrain_epochs": args.pretrain_epochs,
        "pretrain_batch_size": args.pretrain_batch_size,
        "update_interval": args.update_interval,
        "max_refinement_steps": args.max_refinement_steps,
        "initial_kmeans_n_init": args.initial_kmeans_n_init,
    }
    if args.protocol == "release-2021":
        config = DekmConfig(
            protocol_id=DEKM_RELEASE_2021_PROTOCOL_ID,
            kernel_initializer="uniform",
            eigen_solver="general_eig",
            kmeans_random_state="global",
            deterministic_runtime=False,
            data_shuffle="release_random_0_99",
            refinement_batch_size=(
                args.batch_size
                if args.batch_size is not None
                else 10_000
                if args.dataset == "REUTERS"
                else 256
            ),
            assignment_change_tolerance=(
                args.assignment_tolerance
                if args.assignment_tolerance is not None
                else 0.005
            ),
            **common,
        )
    else:
        config = DekmConfig(
            protocol_id=DEKM_MODERN_PROTOCOL_ID,
            kernel_initializer="glorot_uniform",
            eigen_solver="symmetric_eigh",
            kmeans_random_state="seeded",
            deterministic_runtime=True,
            data_shuffle="direct_seed",
            refinement_batch_size=(
                args.batch_size if args.batch_size is not None else 256
            ),
            assignment_change_tolerance=(
                args.assignment_tolerance
                if args.assignment_tolerance is not None
                else 0.001
            ),
            **common,
        )
    config.validate()
    return config


def build_model(
    tf: Any,
    input_dim: int,
    latent_dim: int,
    kernel_initializer: str,
) -> Any:
    """Build the released dense DEKM autoencoder and concatenated output."""

    layers = tf.keras.layers
    inputs = layers.Input(shape=(input_dim,), name="features")
    values = inputs
    for index, units in enumerate((500, 500, 2000), start=1):
        values = layers.Dense(
            units,
            activation="relu",
            kernel_initializer=kernel_initializer,
            name=f"encoder_dense_{index}",
        )(values)
    embedding = layers.Dense(
        latent_dim,
        kernel_initializer=kernel_initializer,
        name="embedding",
    )(values)

    values = embedding
    for index, units in enumerate((2000, 500, 500), start=1):
        values = layers.Dense(
            units,
            activation="relu",
            kernel_initializer=kernel_initializer,
            name=f"decoder_dense_{index}",
        )(values)
    reconstruction = layers.Dense(
        input_dim,
        kernel_initializer=kernel_initializer,
        name="reconstruction",
    )(values)
    outputs = layers.Concatenate(name="dekm_output")([embedding, reconstruction])
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="dekm_dense")


def reconstruction_loss(tf: Any, latent_dim: int):
    """Return the released pretraining MSE averaged over input dimensions."""

    def loss(y_true: Any, y_pred: Any) -> Any:
        reconstructed = y_pred[:, latent_dim:]
        flattened = tf.reshape(y_true, tf.shape(reconstructed))
        return tf.reduce_mean(tf.square(flattened - reconstructed), axis=1)

    return loss


def pretrain(
    tf: Any,
    model: Any,
    features: np.ndarray,
    config: DekmConfig,
    seed: int,
    weights_path: Path,
) -> None:
    """Pretrain the autoencoder with the released reconstruction objective."""

    options = tf.data.Options()
    options.deterministic = True
    dataset = (
        tf.data.Dataset.from_tensor_slices((features, features))
        .shuffle(
            8000,
            seed=seed if config.data_shuffle == "direct_seed" else None,
            reshuffle_each_iteration=True,
        )
        .batch(config.pretrain_batch_size)
    )
    if config.deterministic_runtime:
        dataset = dataset.with_options(options)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=reconstruction_loss(tf, config.latent_dim),
    )
    model.fit(dataset, epochs=config.pretrain_epochs, verbose=2)
    model.save_weights(weights_path)


def run_refinement(
    tf: Any,
    model: Any,
    features: np.ndarray,
    true_labels: np.ndarray,
    n_clusters: int,
    config: DekmConfig,
    seed: int,
) -> RefinementResult:
    """Run the official alternating K-Means and greedy DEKM updates."""

    optimizer = tf.keras.optimizers.Adam()
    previous_labels = np.full(len(features), -1, dtype=int)
    labels = previous_labels.copy()
    embedding = np.empty((len(features), config.latent_dim), dtype=np.float32)
    transformed_embedding = np.empty_like(embedding)
    transformed_centroids = np.empty((n_clusters, config.latent_dim), dtype=float)
    eigenvectors = np.eye(config.latent_dim, dtype=float)
    batch_index = 0
    kmeans_n_init = config.initial_kmeans_n_init
    loss_mean = np.nan
    logs: list[dict[str, object]] = []
    stop_reason = "max_refinement_steps"
    refreshes_completed = 0
    steps_completed = 0

    for step in range(config.max_refinement_steps):
        if step % config.update_interval == 0:
            outputs = model(features, training=False).numpy()
            embedding = outputs[:, : config.latent_dim]
            kmeans = KMeans(
                n_clusters=n_clusters,
                n_init=kmeans_n_init,
                random_state=(seed if config.kmeans_random_state == "seeded" else None),
            ).fit(embedding)
            labels = kmeans.labels_.astype(int)
            centroids = kmeans.cluster_centers_
            changed = assignment_change_count(previous_labels, labels, n_clusters)
            previous_labels = labels.copy()

            scatter = within_cluster_scatter(embedding, labels, centroids)
            eigenvalues, eigenvectors = sorted_eigendecomposition(
                scatter,
                solver=config.eigen_solver,
            )
            transformed_embedding = embedding @ eigenvectors
            transformed_centroids = centroids @ eigenvectors
            metrics = compute_external_metrics(
                true_labels,
                labels,
                n_clusters=n_clusters,
            )
            refreshes_completed += 1
            logs.append(
                {
                    "refresh": refreshes_completed,
                    "step": step,
                    "acc": metrics.acc,
                    "nmi": metrics.nmi,
                    "loss_mean": loss_mean,
                    "changed_assignments": changed,
                    "changed_fraction": changed / len(features),
                    "kmeans_n_init": kmeans_n_init,
                    "kmeans_n_iter": int(kmeans.n_iter_),
                    "kmeans_inertia": float(kmeans.inertia_),
                    "smallest_eigenvalue": float(eigenvalues[0]),
                    "largest_eigenvalue": float(eigenvalues[-1]),
                    "cluster_sizes": json.dumps(
                        np.bincount(labels, minlength=n_clusters).astype(int).tolist()
                    ),
                }
            )
            print(
                f"refresh={refreshes_completed} step={step} "
                f"acc={metrics.acc:.5f} nmi={metrics.nmi:.5f} "
                f"changed={changed}/{len(features)} loss={loss_mean}"
            )
            kmeans_n_init = max(1, int(kmeans.n_iter_) * 2)
            if changed <= len(features) * config.assignment_change_tolerance:
                stop_reason = "assignment_tolerance"
                steps_completed = step
                break

        indices, batch_index = sequential_batch_indices(
            len(features),
            config.refinement_batch_size,
            batch_index,
        )
        batch_target = build_frozen_greedy_target(
            transformed_embedding[indices],
            transformed_centroids,
            labels[indices],
        ).astype(np.float32)
        batch_features = features[indices]
        with tf.GradientTape() as tape:
            outputs = model(batch_features, training=True)
            predicted_embedding = outputs[:, : config.latent_dim]
            predicted_transformed = tf.matmul(
                predicted_embedding,
                tf.convert_to_tensor(eigenvectors, dtype=predicted_embedding.dtype),
            )
            per_sample_loss = tf.reduce_mean(
                tf.square(tf.convert_to_tensor(batch_target) - predicted_transformed),
                axis=1,
            )
            gradient_loss = tf.reduce_sum(per_sample_loss)
        gradients = tape.gradient(gradient_loss, model.trainable_variables)
        optimizer.apply_gradients(
            zip(gradients, model.trainable_variables, strict=True)
        )
        loss_mean = float(tf.reduce_mean(per_sample_loss).numpy())
        steps_completed = step + 1

    return RefinementResult(
        embedding=np.asarray(embedding),
        labels=np.asarray(labels),
        training_log=pd.DataFrame.from_records(logs),
        steps_completed=steps_completed,
        refreshes_completed=refreshes_completed,
        stop_reason=stop_reason,
    )


def file_sha256(path: Path) -> str:
    """Hash one persisted output file."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, object]) -> None:
    """Atomically write stable JSON metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def runtime_metadata(tf: Any, deterministic_runtime: bool = True) -> dict[str, object]:
    """Capture the environment needed to interpret numerical differences."""

    gpu_devices = [device.name for device in tf.config.list_physical_devices("GPU")]
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scikit_learn": sklearn_version,
        "tensorflow": tf.__version__,
        "gpu_devices": gpu_devices,
        "deterministic_runtime_policy": (
            DETERMINISTIC_RUNTIME_POLICY if deterministic_runtime else "disabled"
        ),
        "tensorflow_op_determinism": deterministic_runtime,
        "tf_data_deterministic": deterministic_runtime,
    }


def prepare_run_directory(path: Path, force: bool) -> bool:
    """Create an isolated seed directory or skip an existing completed run."""

    manifest_path = path / "manifest.json"
    if manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") == "complete":
            print(f"Skipping completed run: {path}")
            return False
        msg = f"Incomplete run directory already exists: {path}. Use --force."
        raise FileExistsError(msg)
    if path.exists() and force:
        for child in path.iterdir():
            if child.is_file():
                child.unlink()
            else:
                msg = f"Refusing to replace nested directory in run path: {child}"
                raise ValueError(msg)
    path.mkdir(parents=True, exist_ok=True)
    return True


def run_one_seed(
    tf: Any,
    dataset: PublicDataset,
    config: DekmConfig,
    seed: int,
    run_dir: Path,
    force: bool,
) -> dict[str, object] | None:
    """Execute and persist one seed-isolated DEKM run."""

    if not prepare_run_directory(run_dir, force):
        return None

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    started = time.time()
    if config.data_shuffle == "release_random_0_99":
        shuffle_seed = int(np.random.randint(100))
    else:
        shuffle_seed = seed
    row_indices = tf.random.shuffle(
        np.arange(len(dataset.X), dtype=int),
        seed=shuffle_seed,
    ).numpy()
    features = dataset.X[row_indices]
    true_labels = dataset.y[row_indices]

    model = build_model(
        tf,
        dataset.X.shape[1],
        config.latent_dim,
        config.kernel_initializer,
    )
    pretrain_path = run_dir / "pretrain.weights.h5"
    final_path = run_dir / "final.weights.h5"
    pretrain(tf, model, features, config, seed, pretrain_path)
    result = run_refinement(
        tf,
        model,
        features,
        true_labels,
        dataset.spec.n_clusters,
        config,
        seed,
    )
    model.save_weights(final_path)

    embedding = restore_original_order(result.embedding, row_indices)
    labels = restore_original_order(result.labels, row_indices)
    metrics = compute_external_metrics(
        dataset.y,
        labels,
        n_clusters=dataset.spec.n_clusters,
    )
    artifact_path = run_dir / "artifact.pkl"
    assignments_path = run_dir / "assignments.csv"
    training_log_path = run_dir / "training_log.csv"

    artifact = build_public_artifact(
        dataset=dataset.spec.name,
        method="DEKM",
        n_clusters=dataset.spec.n_clusters,
        random_seed=seed,
        source_sha256=dataset.source_sha256,
        labels=labels,
        true_labels=dataset.y,
        metrics=metrics,
        representation=embedding,
        representation_key="embedding",
    )
    artifact.update(
        {
            "algorithm_family": "DEKM",
            "protocol_id": config.protocol_id,
            "protocol_contract": config.contract(),
            "config_hash": config_hash(config),
            "steps_completed": result.steps_completed,
            "refreshes_completed": result.refreshes_completed,
            "stop_reason": result.stop_reason,
            "row_indices": np.arange(len(labels), dtype=int),
        }
    )
    write_public_artifact(artifact, artifact_path)
    write_public_assignments(
        public_assignment_frame(
            dataset=dataset.spec.name,
            method="DEKM",
            labels=labels,
            true_labels=dataset.y,
            random_seed=seed,
        ),
        assignments_path,
    )
    write_public_frame(result.training_log, training_log_path)

    elapsed = time.time() - started
    output_paths = {
        "artifact": artifact_path,
        "assignments": assignments_path,
        "training_log": training_log_path,
        "pretrain_weights": pretrain_path,
        "final_weights": final_path,
    }
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "dataset": dataset.spec.name,
        "method": "DEKM",
        "protocol_id": config.protocol_id,
        "official_repository": OFFICIAL_REPOSITORY,
        "seed": seed,
        "config": asdict(config),
        "config_hash": config_hash(config),
        "source_sha256": dataset.source_sha256,
        "n_samples": len(dataset.X),
        "n_features": dataset.X.shape[1],
        "n_clusters": dataset.spec.n_clusters,
        "metrics": {"acc": metrics.acc, "nmi": metrics.nmi},
        "steps_completed": result.steps_completed,
        "refreshes_completed": result.refreshes_completed,
        "stop_reason": result.stop_reason,
        "fit_time_seconds": elapsed,
        "runtime": runtime_metadata(tf, config.deterministic_runtime),
        "paths": {name: path.name for name, path in output_paths.items()},
        "output_sha256": {
            name: file_sha256(path) for name, path in output_paths.items()
        },
    }
    write_json(run_dir / "manifest.json", manifest)
    print(
        f"complete dataset={dataset.spec.name} seed={seed} "
        f"acc={metrics.acc:.5f} nmi={metrics.nmi:.5f} "
        f"seconds={elapsed:.1f}"
    )
    return {
        "dataset": dataset.spec.name,
        "method": "DEKM",
        "protocol_id": config.protocol_id,
        "config_hash": config_hash(config),
        "seed": seed,
        "n_samples": len(dataset.X),
        "n_features": dataset.X.shape[1],
        "n_clusters": dataset.spec.n_clusters,
        "source_sha256": dataset.source_sha256,
        "acc": metrics.acc,
        "nmi": metrics.nmi,
        "steps_completed": result.steps_completed,
        "refreshes_completed": result.refreshes_completed,
        "stop_reason": result.stop_reason,
        "fit_time_seconds": elapsed,
        "run_dir": str(run_dir),
    }


def main() -> None:
    """Run one or more consecutive DEKM seeds and aggregate their metrics."""

    args = parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be at least 1.")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative.")
    config = build_config(args)

    tf = import_tensorflow()
    if config.deterministic_runtime:
        enable_tensorflow_determinism(tf)
    runtime = runtime_metadata(tf, config.deterministic_runtime)
    print(json.dumps(runtime, indent=2))
    if args.require_gpu and not runtime["gpu_devices"]:
        raise RuntimeError("--require-gpu was set, but TensorFlow found no GPU.")

    dataset = load_dekm_public_dataset(args.dataset, args.dataset_root)
    hash_prefix = config_hash(config)[:12]
    config_root = (
        args.output_dir / dataset.spec.name.lower() / config.protocol_id / hash_prefix
    )
    records: list[dict[str, object]] = []
    for seed in range(args.seed, args.seed + args.runs):
        record = run_one_seed(
            tf,
            dataset,
            config,
            seed,
            config_root / f"seed_{seed}",
            args.force,
        )
        if record is not None:
            records.append(record)

    existing_path = config_root / "dekm_runs.csv"
    if existing_path.exists():
        existing = pd.read_csv(existing_path)
    else:
        existing = pd.DataFrame()
    combined = pd.concat(
        [existing, pd.DataFrame.from_records(records)],
        ignore_index=True,
    )
    if not combined.empty:
        combined = combined.drop_duplicates(subset=["seed"], keep="last").sort_values(
            "seed"
        )
        write_public_frame(combined, existing_path)
        summary = pd.DataFrame(
            [
                {
                    "dataset": dataset.spec.name,
                    "method": "DEKM",
                    "protocol_id": config.protocol_id,
                    "config_hash": config_hash(config),
                    "completed_runs": len(combined),
                    "acc_mean": combined["acc"].mean(),
                    "acc_std": combined["acc"].std(ddof=1)
                    if len(combined) > 1
                    else 0.0,
                    "nmi_mean": combined["nmi"].mean(),
                    "nmi_std": combined["nmi"].std(ddof=1)
                    if len(combined) > 1
                    else 0.0,
                    "source_sha256": dataset.source_sha256,
                }
            ]
        )
        write_public_frame(summary, config_root / "dekm_summary.csv")


if __name__ == "__main__":
    main()
