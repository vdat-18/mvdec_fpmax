"""Train and save the legacy MvDEC fused representation artifact."""

from __future__ import annotations

import json
import pickle
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


@dataclass(frozen=True)
class MvDECRepresentationConfig:
    """Hyperparameters matching the legacy Colab MvDEC notebook."""

    seed: int = 42
    n_iterations: int = 20
    n_clusters: int = 5
    batch_size: int = 128
    epochs: int = 100
    validation_split: float = 0.1
    early_stopping_patience: int = 10
    learning_rate: float = 1e-4
    view1_latent_dim: int = 4
    kmeans_n_init: int = 10
    init_methods: tuple[str, ...] = ("k-means++", "random")


def require_tensorflow() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    """Import TensorFlow lazily and return the Keras objects used here."""

    try:
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.layers import Concatenate, Dense, Input
        from tensorflow.keras.models import Model
        from tensorflow.keras.optimizers import Adam
    except ModuleNotFoundError as error:  # pragma: no cover - depends on runtime.
        msg = (
            "TensorFlow is required for MvDEC representation learning. "
            "Run this stage in Colab GPU and install dependencies with "
            "`uv pip install tensorflow`, or install tensorflow manually."
        )
        raise ModuleNotFoundError(msg) from error

    return tf, EarlyStopping, Concatenate, Dense, Input, Model, Adam


def tensorflow_gpu_names() -> list[str]:
    """Return TensorFlow-visible GPU device names."""

    tf, *_ = require_tensorflow()
    return [device.name for device in tf.config.list_physical_devices("GPU")]


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy, and TensorFlow for reproducible runs."""

    tf, *_ = require_tensorflow()
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_view1_models(
    input_dim: int,
    latent_dim: int,
) -> tuple[Any, Any]:
    """Build the DEKM-style MLP autoencoder and embedding model."""

    _, _, Concatenate, Dense, Input, Model, _ = require_tensorflow()
    inputs = Input(shape=(input_dim,), name="view1_input")

    encoded = Dense(250, activation="relu")(inputs)
    encoded = Dense(250, activation="relu")(encoded)
    encoded = Dense(1000, activation="relu")(encoded)
    bottleneck = Dense(latent_dim, activation="linear", name="bottleneck_view1")(
        encoded
    )

    decoded = Dense(1000, activation="relu")(bottleneck)
    decoded = Dense(250, activation="relu")(decoded)
    decoded = Dense(250, activation="relu")(decoded)
    reconstruction = Dense(
        input_dim,
        activation="linear",
        name="reconstruction_view1",
    )(decoded)

    autoencoder = Model(inputs=inputs, outputs=reconstruction, name="View1_Autoencoder")
    embedding_output = Concatenate(name="view1_embedding")([bottleneck, reconstruction])
    embedding_model = Model(
        inputs=inputs,
        outputs=embedding_output,
        name="View1_EmbeddingModel",
    )
    return autoencoder, embedding_model


def build_view2_models(input_dim: int, embedding_dim: int) -> tuple[Any, Any]:
    """Build the U-Net-style autoencoder and embedding model."""

    _, _, Concatenate, Dense, Input, Model, _ = require_tensorflow()
    inputs = Input(shape=(input_dim,), name="view2_input")

    x1 = Dense(32, activation="relu")(inputs)
    x2 = Dense(32, activation="relu")(x1)
    x3 = Dense(64, activation="relu")(x2)
    x4 = Dense(64, activation="relu")(x3)
    x5 = Dense(128, activation="relu")(x4)
    x6 = Dense(128, activation="relu")(x5)
    x7 = Dense(256, activation="relu")(x6)
    x8 = Dense(256, activation="relu")(x7)
    x9 = Dense(512, activation="relu")(x8)
    x10 = Dense(256, activation="relu")(x9)
    x11 = Dense(128, activation="relu")(x10)

    decoded = Concatenate()([x8, x11])
    decoded = Dense(256, activation="relu")(decoded)
    decoded = Dense(128, activation="relu")(decoded)
    decoded = Dense(64, activation="relu")(decoded)
    decoded = Concatenate()([decoded, x6])
    decoded = Dense(128, activation="relu")(decoded)
    decoded = Dense(64, activation="relu")(decoded)
    decoded = Dense(32, activation="relu")(decoded)
    decoded = Concatenate()([decoded, x3])
    decoded = Dense(64, activation="relu")(decoded)
    decoded = Dense(32, activation="relu")(decoded)
    decoded = Dense(16, activation="relu")(decoded)
    decoded = Concatenate()([decoded, x1])
    decoded = Dense(32, activation="relu")(decoded)

    embedding = Dense(embedding_dim, activation=None, name="view2_embedding")(decoded)
    reconstruction = Dense(
        input_dim,
        activation="linear",
        name="reconstruction_view2",
    )(embedding)

    autoencoder = Model(inputs=inputs, outputs=reconstruction, name="View2_Autoencoder")
    embedding_model = Model(
        inputs=inputs, outputs=embedding, name="View2_EmbeddingModel"
    )
    return autoencoder, embedding_model


def validate_input_matrix(X: np.ndarray) -> None:
    """Validate the continuous feature matrix before representation learning."""

    if X.ndim != 2:
        msg = f"X must be 2D, got shape {X.shape}."
        raise ValueError(msg)
    if X.shape[0] < 2:
        msg = "X must contain at least two rows."
        raise ValueError(msg)
    if not np.isfinite(X).all():
        msg = "X contains NaN or infinite values."
        raise ValueError(msg)


def make_history_row(
    iteration: int,
    init_method: str,
    score: float,
    labels: np.ndarray,
) -> dict[str, Any]:
    """Build one auditable representation-learning history row."""

    return {
        "iteration": iteration,
        "init": init_method,
        "silhouette": float(score),
        "cluster_sizes": json.dumps(
            np.bincount(labels.astype(int)).astype(int).tolist()
        ),
    }


def train_mvdec_representation(
    X: np.ndarray,
    config: MvDECRepresentationConfig,
    expected_fused_dim: int | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Train MvDEC views and return the legacy fused-representation artifact."""

    _, EarlyStopping, _, _, _, _, Adam = require_tensorflow()
    validate_input_matrix(X)
    set_random_seed(config.seed)

    best_score = -np.inf
    best_labels: np.ndarray | None = None
    best_init: str | None = None
    best_iteration = -1
    best_fused: np.ndarray | None = None
    history_rows: list[dict[str, Any]] = []
    scores_by_init = {init: [] for init in config.init_methods}
    embedding_dim = X.shape[1] + config.view1_latent_dim
    if expected_fused_dim is not None and embedding_dim != expected_fused_dim:
        msg = (
            "Configured fused representation dimension does not match downstream "
            f"schema: {embedding_dim} != {expected_fused_dim}."
        )
        raise ValueError(msg)

    for iteration in range(1, config.n_iterations + 1):
        logger.info("Representation iteration {}/{}", iteration, config.n_iterations)
        auto_v1, embed_v1 = build_view1_models(
            input_dim=X.shape[1],
            latent_dim=config.view1_latent_dim,
        )
        auto_v2, embed_v2 = build_view2_models(
            input_dim=X.shape[1],
            embedding_dim=embedding_dim,
        )
        auto_v1.compile(optimizer=Adam(config.learning_rate), loss="mse")
        auto_v2.compile(optimizer=Adam(config.learning_rate), loss="mse")

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
        )
        auto_v1.fit(
            X,
            X,
            batch_size=config.batch_size,
            epochs=config.epochs,
            validation_split=config.validation_split,
            callbacks=[early_stop],
            verbose=0,
        )
        auto_v2.fit(
            X,
            X,
            batch_size=config.batch_size,
            epochs=config.epochs,
            validation_split=config.validation_split,
            callbacks=[early_stop],
            verbose=0,
        )

        h1 = embed_v1.predict(X, verbose=0)
        h2 = embed_v2.predict(X, verbose=0)
        if h1.shape != h2.shape:
            msg = f"View embedding shape mismatch: {h1.shape} != {h2.shape}."
            raise ValueError(msg)
        h_fused = ((h1 + h2) / 2).astype(np.float32)
        if expected_fused_dim is not None and h_fused.shape[1] != expected_fused_dim:
            msg = (
                "Fused representation dimension does not match downstream schema: "
                f"{h_fused.shape[1]} != {expected_fused_dim}."
            )
            raise ValueError(msg)

        for init_method in config.init_methods:
            kmeans = KMeans(
                n_clusters=config.n_clusters,
                init=init_method,
                n_init=config.kmeans_n_init,
                random_state=config.seed,
            )
            labels = kmeans.fit_predict(h_fused)
            score = float(silhouette_score(h_fused, labels))
            scores_by_init[init_method].append(score)
            history_rows.append(make_history_row(iteration, init_method, score, labels))
            logger.info("{} silhouette={:.4f}", init_method, score)

            if score > best_score:
                best_score = score
                best_labels = labels.astype(int)
                best_fused = h_fused
                best_init = init_method
                best_iteration = iteration

    if best_fused is None or best_labels is None or best_init is None:
        msg = "Representation learning did not produce a valid fused embedding."
        raise RuntimeError(msg)

    result = {
        "h_fused": best_fused,
        "labels": best_labels,
        "init": best_init,
        "score": float(best_score),
        "iteration": int(best_iteration),
        "config": asdict(config),
        "scores_by_init": scores_by_init,
    }
    return result, pd.DataFrame(history_rows)


def save_representation(result: dict[str, Any], output_path: Path) -> None:
    """Write the fused representation artifact as a pickle file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        pickle.dump(result, file)


def load_feature_matrix(input_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Load the preprocessed continuous feature matrix from CSV."""

    if not input_path.exists():
        msg = f"Missing preprocessed data file: {input_path}"
        raise FileNotFoundError(msg)
    df = pd.read_csv(input_path)
    invalid_columns = [
        column for column in df.columns if not pd.api.types.is_numeric_dtype(df[column])
    ]
    if invalid_columns:
        msg = f"Input has non-numeric columns: {invalid_columns}."
        raise ValueError(msg)
    X = df.to_numpy(dtype=np.float32)
    validate_input_matrix(X)
    return df, X
