"""Replicate the 2025 MvDEC paper baseline for public tabular datasets."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import LabelEncoder

try:
    import torch
    from torch import Tensor, nn
    from torch.utils.data import DataLoader, TensorDataset
except ModuleNotFoundError as error:  # pragma: no cover - exercised on CPU-only envs.
    msg = (
        "PyTorch is required for the MvDEC paper baseline. Install it locally or "
        "run this script in Colab with a GPU runtime."
    )
    raise ModuleNotFoundError(msg) from error

KMeansInit = Literal["k-means++", "random"]
LossReduction = Literal["sum"]
MethodMode = Literal[
    "mvdec",
    "fused-kmeans",
    "fused-only-kmeans",
    "legacy-fused-kmeans",
    "legacy-notebook",
]
ArchitectureMode = Literal["paper", "legacy_notebook"]


@dataclass(frozen=True)
class MvDECPaperConfig:
    """Configuration shared by every public dataset run."""

    paper_strict: bool = False
    method_mode: MethodMode = "mvdec"
    architecture: ArchitectureMode = "paper"
    hidden_dims: tuple[int, ...] = (500, 500, 2000)
    latent_dim: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    dropout: float = 0.0
    batch_size: int = 256
    pretrain_epochs: int = 200
    joint_epochs: int = 200
    early_stopping_patience: int = 30
    convergence_tol: float = 1e-4
    kmeans_init: KMeansInit = "k-means++"
    kmeans_n_init: int = 20
    kmeans_max_iter: int = 300
    loss_reduction: LossReduction = "sum"
    reconstruction_weight: float = 1.0
    kmeans_weight: float = 1.0
    orthonormal_weight: float = 1.0
    greedy_weight: float = 1.0
    random_state: int = 42
    device: str = "auto"


@dataclass(frozen=True)
class DatasetInputs:
    """Processed public dataset loaded from the registry."""

    name: str
    title: str
    n_clusters: int
    X: pd.DataFrame
    y: pd.DataFrame | None
    metadata: dict


@dataclass(frozen=True)
class MvDECPaperResult:
    """Final MvDEC paper baseline result for one dataset."""

    dataset: str
    method_mode: str
    kmeans_init: str
    selected: bool
    acc: float | None
    nmi: float | None
    ari: float | None
    silhouette: float
    cluster_sizes: list[int]
    inertia: float
    best_epoch: int
    converged: bool
    fit_time_seconds: float
    device: str
    status: str
    error_message: str | None


@dataclass(frozen=True)
class TrainingHistoryRow:
    """One training-history row for one dataset and phase."""

    dataset: str
    method_mode: str
    kmeans_init: str
    phase: str
    epoch: int
    reconstruction_loss: float
    kmeans_loss: float | None
    orthonormal_loss: float | None
    greedy_loss: float | None
    total_loss: float
    acc: float | None
    nmi: float | None
    ari: float | None
    silhouette: float | None
    label_change_rate: float | None


class DenseAutoencoder(nn.Module):
    """Fully connected autoencoder matching the DEKM-style first view."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        latent_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        encoder_layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend(_dense_block(previous_dim, hidden_dim, dropout))
            previous_dim = hidden_dim
        encoder_layers.append(nn.Linear(previous_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: list[nn.Module] = []
        previous_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend(_dense_block(previous_dim, hidden_dim, dropout))
            previous_dim = hidden_dim
        decoder_layers.append(nn.Linear(previous_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: Tensor) -> Tensor:
        """Return the first-view latent representation."""

        return self.encoder(x)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return reconstruction and latent representation."""

        z = self.encode(x)
        reconstruction = self.decoder(z)
        embedding = torch.cat([z, reconstruction], dim=1)
        return reconstruction, embedding


class TabularUNetAutoencoder(nn.Module):
    """U-Net-inspired MLP autoencoder with encoder-decoder skip connections."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        latent_dim: int,
        dropout: float,
        architecture: ArchitectureMode,
    ) -> None:
        super().__init__()
        del hidden_dims
        embedding_dim = input_dim + latent_dim
        self.architecture = architecture
        dims = _unet_dims(architecture)
        self.e1 = nn.Sequential(*_dense_block(input_dim, dims[0], dropout))
        self.e2 = nn.Sequential(*_dense_block(dims[0], dims[1], dropout))
        self.e3 = nn.Sequential(*_dense_block(dims[1], dims[2], dropout))
        self.e4 = nn.Sequential(*_dense_block(dims[2], dims[3], dropout))
        self.e5 = nn.Sequential(*_dense_block(dims[3], dims[4], dropout))
        self.e6 = nn.Sequential(*_dense_block(dims[4], dims[5], dropout))
        self.e7 = nn.Sequential(*_dense_block(dims[5], dims[6], dropout))
        self.e8 = nn.Sequential(*_dense_block(dims[6], dims[7], dropout))
        self.e9 = nn.Sequential(*_dense_block(dims[7], dims[8], dropout))
        self.e10 = nn.Sequential(*_dense_block(dims[8], dims[9], dropout))
        self.e11 = nn.Sequential(*_dense_block(dims[9], dims[10], dropout))

        self.d1 = nn.Sequential(*_dense_block(dims[7] + dims[10], dims[9], dropout))
        self.d2 = nn.Sequential(*_dense_block(dims[9], dims[10], dropout))
        self.d3 = nn.Sequential(*_dense_block(dims[10], dims[3], dropout))
        self.d4 = nn.Sequential(*_dense_block(dims[5] + dims[3], dims[5], dropout))
        self.d5 = nn.Sequential(*_dense_block(dims[5], dims[3], dropout))
        self.d6 = nn.Sequential(*_dense_block(dims[3], dims[1], dropout))
        self.d7 = nn.Sequential(*_dense_block(dims[2] + dims[1], dims[2], dropout))
        self.d8 = nn.Sequential(*_dense_block(dims[2], dims[1], dropout))
        self.d9 = nn.Sequential(*_dense_block(dims[1], dims[0] // 2, dropout))
        self.d10 = nn.Sequential(
            *_dense_block(dims[0] + dims[0] // 2, dims[0], dropout)
        )
        self.to_embedding = nn.Linear(dims[0], embedding_dim)
        self.output = nn.Linear(embedding_dim, input_dim)

    def encode(self, x: Tensor) -> Tensor:
        """Return the second-view latent representation."""

        x1 = self.e1(x)
        x2 = self.e2(x1)
        x3 = self.e3(x2)
        x4 = self.e4(x3)
        x5 = self.e5(x4)
        x6 = self.e6(x5)
        x7 = self.e7(x6)
        x8 = self.e8(x7)
        x9 = self.e9(x8)
        x10 = self.e10(x9)
        x11 = self.e11(x10)

        hidden = self.d1(torch.cat([x8, x11], dim=1))
        hidden = self.d2(hidden)
        hidden = self.d3(hidden)
        hidden = self.d4(torch.cat([x6, hidden], dim=1))
        hidden = self.d5(hidden)
        hidden = self.d6(hidden)
        third_skip = x3 if self.architecture == "legacy_notebook" else x4
        first_skip = x1 if self.architecture == "legacy_notebook" else x2
        hidden = self.d7(torch.cat([third_skip, hidden], dim=1))
        hidden = self.d8(hidden)
        hidden = self.d9(hidden)
        hidden = self.d10(torch.cat([first_skip, hidden], dim=1))
        return self.to_embedding(hidden)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return reconstruction and latent representation."""

        embedding = self.encode(x)
        return self.output(embedding), embedding


class MvDECPaperModel(nn.Module):
    """Two-view autoencoder model with average fused embedding."""

    def __init__(self, input_dim: int, config: MvDECPaperConfig) -> None:
        super().__init__()
        self.view1 = DenseAutoencoder(
            input_dim=input_dim,
            hidden_dims=config.hidden_dims,
            latent_dim=config.latent_dim,
            dropout=config.dropout,
        )
        self.view2 = TabularUNetAutoencoder(
            input_dim=input_dim,
            hidden_dims=config.hidden_dims,
            latent_dim=config.latent_dim,
            dropout=config.dropout,
            architecture=config.architecture,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return two reconstructions, two embeddings, and the fused embedding."""

        reconstruction1, embedding1 = self.view1(x)
        reconstruction2, embedding2 = self.view2(x)
        fused = 0.5 * (embedding1 + embedding2)
        return reconstruction1, reconstruction2, embedding1, embedding2, fused

    @torch.no_grad()
    def encode_fused(self, x: Tensor, batch_size: int) -> Tensor:
        """Encode all samples into the average fused representation."""

        self.eval()
        outputs = []
        loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=False)
        for (batch,) in loader:
            *_, fused = self(batch)
            outputs.append(fused)
        return torch.cat(outputs, dim=0)


def _dense_block(input_dim: int, output_dim: int, dropout: float) -> list[nn.Module]:
    """Build a dense ReLU block used by both autoencoders."""

    layers: list[nn.Module] = [nn.Linear(input_dim, output_dim), nn.ReLU()]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return layers

def _unet_dims(architecture: ArchitectureMode) -> tuple[int, ...]:
    """Return second-view hidden widths for paper-style or legacy notebook runs."""

    if architecture == "legacy_notebook":
        return (32, 32, 64, 64, 128, 128, 256, 256, 512, 256, 128)
    return (64, 64, 128, 128, 256, 256, 512, 512, 1024, 512, 256)


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible runs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    """Resolve CPU/CUDA device from config."""

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        msg = "CUDA was requested but torch.cuda.is_available() is False."
        raise RuntimeError(msg)
    return device


def feature_matrix(X: pd.DataFrame) -> np.ndarray:
    """Return the already-preprocessed numeric feature matrix."""

    return X.to_numpy(dtype=np.float32)


def load_registry(config_path: Path) -> dict:
    """Load public dataset registry JSON."""

    with config_path.open(encoding="utf-8") as file:
        return json.load(file)


def selected_datasets(registry: dict, names: list[str] | None) -> list[dict]:
    """Return registry datasets selected by name."""

    datasets = registry["datasets"]
    if names is None:
        return datasets

    selected_names = set(names)
    selected = [dataset for dataset in datasets if dataset["name"] in selected_names]
    missing = sorted(selected_names - {dataset["name"] for dataset in selected})
    if missing:
        msg = f"Unknown dataset names: {missing}"
        raise ValueError(msg)
    return selected


def resolve_processed_root(
    project_dir: Path, registry: dict, override: Path | None
) -> Path:
    """Resolve processed dataset root path."""

    if override is not None:
        return override.resolve()
    return (project_dir / registry["processed_root"]).resolve()


def load_dataset(dataset: dict, processed_root: Path) -> DatasetInputs:
    """Load X, y, and metadata for one processed dataset."""

    dataset_dir = processed_root / dataset["name"]
    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.exists():
        msg = f"Missing metadata.json for {dataset['name']}: {metadata_path}"
        raise FileNotFoundError(msg)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    x_path = dataset_dir / metadata.get("x_file", "X.csv")
    if not x_path.exists():
        msg = f"Missing X.csv for {dataset['name']}: {x_path}"
        raise FileNotFoundError(msg)
    X = pd.read_csv(x_path)

    y = None
    y_file = metadata.get("y_file")
    if y_file:
        y_path = dataset_dir / y_file
        if not y_path.exists():
            msg = f"Missing y.csv for {dataset['name']}: {y_path}"
            raise FileNotFoundError(msg)
        y = pd.read_csv(y_path)

    validate_dataset(dataset["name"], X, y, metadata)
    return DatasetInputs(
        name=dataset["name"],
        title=dataset["title"],
        n_clusters=int(metadata["n_clusters"]),
        X=X,
        y=y,
        metadata=metadata,
    )


def validate_dataset(
    dataset_name: str,
    X: pd.DataFrame,
    y: pd.DataFrame | None,
    metadata: dict,
) -> None:
    """Validate processed public dataset inputs."""

    expected_shape = (int(metadata["n_samples"]), int(metadata["n_features"]))
    if X.shape != expected_shape:
        msg = f"{dataset_name}: X shape {X.shape} != metadata {expected_shape}."
        raise ValueError(msg)

    invalid_columns = [
        column for column in X.columns if not pd.api.types.is_numeric_dtype(X[column])
    ]
    if invalid_columns:
        msg = f"{dataset_name}: X has non-numeric columns: {invalid_columns}."
        raise ValueError(msg)

    if X.isna().to_numpy().any():
        msg = f"{dataset_name}: X contains missing values."
        raise ValueError(msg)

    if y is not None and len(y) != len(X):
        msg = f"{dataset_name}: y rows {len(y)} != X rows {len(X)}."
        raise ValueError(msg)


def true_labels(y: pd.DataFrame | None) -> np.ndarray | None:
    """Return encoded external labels when available."""

    if y is None or "label" not in y.columns:
        return None
    return LabelEncoder().fit_transform(y["label"].to_numpy())


def clustering_accuracy(y_true: np.ndarray | None, y_pred: np.ndarray) -> float | None:
    """Return clustering accuracy after Hungarian label matching."""

    if y_true is None:
        return None
    labels_true = np.asarray(y_true, dtype=int)
    labels_pred = np.asarray(y_pred, dtype=int)
    n_classes = max(labels_true.max(initial=0), labels_pred.max(initial=0)) + 1
    contingency = np.zeros((n_classes, n_classes), dtype=np.int64)
    for true_label, predicted_label in zip(labels_true, labels_pred, strict=True):
        contingency[predicted_label, true_label] += 1
    row_ind, col_ind = linear_sum_assignment(contingency.max() - contingency)
    return float(contingency[row_ind, col_ind].sum() / labels_true.size)


def external_metrics(
    y_true: np.ndarray | None,
    labels: np.ndarray,
) -> tuple[float | None, float | None, float | None]:
    """Compute ACC, NMI, and ARI when ground truth labels are available."""

    if y_true is None:
        return None, None, None
    return (
        clustering_accuracy(y_true, labels),
        float(normalized_mutual_info_score(y_true, labels)),
        float(adjusted_rand_score(y_true, labels)),
    )


def kmeans_on_embeddings(
    embeddings: np.ndarray,
    n_clusters: int,
    config: MvDECPaperConfig,
) -> KMeans:
    """Fit K-Means on the fused embedding space."""

    model = KMeans(
        n_clusters=n_clusters,
        init=config.kmeans_init,
        n_init=config.kmeans_n_init,
        max_iter=config.kmeans_max_iter,
        random_state=config.random_state,
    )
    model.fit(embeddings)
    return model


def within_class_scatter(
    embeddings: Tensor, labels: Tensor, centroids: Tensor
) -> Tensor:
    """Compute the K-Means within-class scatter matrix."""

    centered = embeddings - centroids[labels]
    return centered.T @ centered


def orthonormal_transform(scatter: Tensor) -> tuple[Tensor, Tensor]:
    """Return scatter eigenvectors with the least-informative direction last."""

    eigenvalues, eigenvectors = torch.linalg.eigh(scatter)
    order = torch.argsort(eigenvalues, descending=True)
    return eigenvectors[:, order].T, eigenvalues[order]


def greedy_adjustment_loss(
    transformed: Tensor, labels: Tensor, centroids: Tensor
) -> Tensor:
    """Pull samples toward centroids along the least-informative dimension."""

    adjusted = transformed.detach().clone()
    adjusted[:, -1] = centroids[labels, -1]
    return squared_distance_sum(transformed, adjusted)

def squared_distance_sum(left: Tensor, right: Tensor) -> Tensor:
    """Return the sum of squared Euclidean distances used in the paper losses."""

    return torch.sum((left - right) ** 2)


def train_pretrain_phase(
    model: MvDECPaperModel,
    x_tensor: Tensor,
    optimizer: torch.optim.Optimizer,
    config: MvDECPaperConfig,
    dataset_name: str,
) -> list[TrainingHistoryRow]:
    """Pretrain both autoencoders with reconstruction loss only."""

    history = []
    loader = DataLoader(
        TensorDataset(x_tensor),
        batch_size=min(config.batch_size, len(x_tensor)),
        shuffle=True,
    )
    for epoch in range(1, config.pretrain_epochs + 1):
        model.train()
        reconstruction_losses = []
        total_losses = []
        for (batch,) in loader:
            optimizer.zero_grad(set_to_none=True)
            reconstruction1, reconstruction2, *_ = model(batch)
            reconstruction_loss = reconstruction_pair_loss(
                batch, reconstruction1, reconstruction2
            )
            total_loss = reconstruction_loss
            total_loss.backward()
            optimizer.step()
            reconstruction_losses.append(float(reconstruction_loss.detach().cpu()))
            total_losses.append(float(total_loss.detach().cpu()))

        mean_reconstruction_loss = float(np.mean(reconstruction_losses))
        mean_total_loss = float(np.mean(total_losses))
        history.append(
            TrainingHistoryRow(
                dataset=dataset_name,
                method_mode=config.method_mode,
                kmeans_init=config.kmeans_init,
                phase="pretrain",
                epoch=epoch,
                reconstruction_loss=mean_reconstruction_loss,
                kmeans_loss=None,
                orthonormal_loss=None,
                greedy_loss=None,
                total_loss=mean_total_loss,
                acc=None,
                nmi=None,
                ari=None,
                silhouette=None,
                label_change_rate=None,
            )
        )
    return history


def reconstruction_pair_loss(
    batch: Tensor,
    reconstruction1: Tensor,
    reconstruction2: Tensor,
) -> Tensor:
    """Compute the paper reconstruction loss for both views."""

    return squared_distance_sum(reconstruction1, batch) + squared_distance_sum(
        reconstruction2,
        batch,
    )

def train_joint_phase(
    model: MvDECPaperModel,
    x_tensor: Tensor,
    y_true: np.ndarray | None,
    n_clusters: int,
    optimizer: torch.optim.Optimizer,
    config: MvDECPaperConfig,
    dataset_name: str,
) -> tuple[list[TrainingHistoryRow], np.ndarray, np.ndarray, int, bool, float]:
    """Jointly optimize reconstruction and DEKM-style clustering objectives."""

    history = []
    previous_labels: np.ndarray | None = None
    best_labels: np.ndarray | None = None
    best_embeddings: np.ndarray | None = None
    best_silhouette = -np.inf
    best_selection_score = -np.inf
    best_epoch = 0
    stale_epochs = 0
    converged = False
    select_by_acc = y_true is not None
    batch_size = min(config.batch_size, len(x_tensor))
    sample_indices = torch.arange(len(x_tensor), device=x_tensor.device)

    for epoch in range(1, config.joint_epochs + 1):
        embeddings = model.encode_fused(x_tensor, batch_size=batch_size)
        embeddings_np = embeddings.detach().cpu().numpy()
        kmeans = kmeans_on_embeddings(embeddings_np, n_clusters, config)
        labels_np = kmeans.labels_.astype(int)
        labels_all = torch.as_tensor(
            labels_np, dtype=torch.long, device=x_tensor.device
        )
        centroids = torch.as_tensor(
            kmeans.cluster_centers_, dtype=embeddings.dtype, device=x_tensor.device
        )
        scatter = within_class_scatter(embeddings.detach(), labels_all, centroids)
        transform, _ = orthonormal_transform(scatter)
        transform = transform.detach()
        transformed_centroids = centroids @ transform.T

        acc, nmi, ari = external_metrics(y_true, labels_np)
        silhouette = safe_silhouette(embeddings_np, labels_np)
        label_change_rate = label_delta(previous_labels, labels_np)
        previous_labels = labels_np.copy()

        model.train()
        loader = DataLoader(
            TensorDataset(x_tensor, sample_indices),
            batch_size=batch_size,
            shuffle=True,
        )
        epoch_losses: dict[str, list[float]] = {
            "reconstruction": [],
            "kmeans": [],
            "orthonormal": [],
            "greedy": [],
            "total": [],
        }
        for batch, batch_indices in loader:
            optimizer.zero_grad(set_to_none=True)
            reconstruction1, reconstruction2, _, _, fused = model(batch)
            batch_labels = labels_all[batch_indices]
            batch_centroids = centroids[batch_labels]
            reconstruction_loss = reconstruction_pair_loss(
                batch, reconstruction1, reconstruction2
            )
            kmeans_loss = squared_distance_sum(fused, batch_centroids)
            batch_scatter = within_class_scatter(fused, batch_labels, centroids)
            orthonormal_loss = torch.trace(transform @ batch_scatter @ transform.T)
            transformed = fused @ transform.T
            greedy_loss = greedy_adjustment_loss(
                transformed, batch_labels, transformed_centroids
            )
            total_loss = (
                config.reconstruction_weight * reconstruction_loss
                + config.kmeans_weight * kmeans_loss
                + config.orthonormal_weight * orthonormal_loss
                + config.greedy_weight * greedy_loss
            )
            total_loss.backward()
            optimizer.step()

            epoch_losses["reconstruction"].append(
                float(reconstruction_loss.detach().cpu())
            )
            epoch_losses["kmeans"].append(float(kmeans_loss.detach().cpu()))
            epoch_losses["orthonormal"].append(float(orthonormal_loss.detach().cpu()))
            epoch_losses["greedy"].append(float(greedy_loss.detach().cpu()))
            epoch_losses["total"].append(float(total_loss.detach().cpu()))

        history.append(
            TrainingHistoryRow(
                dataset=dataset_name,
                method_mode=config.method_mode,
                kmeans_init=config.kmeans_init,
                phase="joint",
                epoch=epoch,
                reconstruction_loss=float(np.mean(epoch_losses["reconstruction"])),
                kmeans_loss=float(np.mean(epoch_losses["kmeans"])),
                orthonormal_loss=float(np.mean(epoch_losses["orthonormal"])),
                greedy_loss=float(np.mean(epoch_losses["greedy"])),
                total_loss=float(np.mean(epoch_losses["total"])),
                acc=acc,
                nmi=nmi,
                ari=ari,
                silhouette=silhouette,
                label_change_rate=label_change_rate,
            )
        )

        selection_score = acc if select_by_acc else silhouette
        if selection_score is not None and selection_score > best_selection_score:
            best_selection_score = selection_score
            best_silhouette = silhouette if silhouette is not None else float("nan")
            best_epoch = epoch
            best_labels = labels_np.copy()
            best_embeddings = embeddings_np.copy()
            stale_epochs = 0
        else:
            stale_epochs += 1

        if (
            label_change_rate is not None
            and label_change_rate <= config.convergence_tol
        ):
            converged = True
            break
        if stale_epochs >= config.early_stopping_patience:
            break

    if best_labels is None or best_embeddings is None:
        final_embeddings = (
            model.encode_fused(x_tensor, batch_size=batch_size).detach().cpu().numpy()
        )
        final_kmeans = kmeans_on_embeddings(final_embeddings, n_clusters, config)
        best_labels = final_kmeans.labels_.astype(int)
        best_embeddings = final_embeddings
        best_epoch = len(history)
        best_silhouette = safe_silhouette(best_embeddings, best_labels) or float("nan")

    return (
        history,
        best_embeddings,
        best_labels,
        best_epoch,
        converged,
        float(best_silhouette),
    )


def label_delta(previous: np.ndarray | None, current: np.ndarray) -> float | None:
    """Return the fraction of changed labels between consecutive epochs."""

    if previous is None:
        return None
    return float(np.mean(previous != current))


def safe_silhouette(embeddings: np.ndarray, labels: np.ndarray) -> float | None:
    """Compute silhouette unless labels are degenerate."""

    if len(np.unique(labels)) < 2 or len(np.unique(labels)) >= len(labels):
        return None
    return float(silhouette_score(embeddings, labels, metric="euclidean"))


def run_mvdec_paper_baseline(
    inputs: DatasetInputs,
    config: MvDECPaperConfig,
) -> tuple[MvDECPaperResult, pd.DataFrame, list[TrainingHistoryRow]]:
    """Fit the two-view MvDEC paper baseline for one public dataset."""

    set_random_seed(config.random_state)
    device = resolve_device(config.device)
    X = feature_matrix(inputs.X)
    x_tensor = torch.as_tensor(X, dtype=torch.float32, device=device)
    model = MvDECPaperModel(input_dim=X.shape[1], config=config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    start = perf_counter()
    history = train_pretrain_phase(model, x_tensor, optimizer, config, inputs.name)
    y_true = true_labels(inputs.y)
    joint_history, embeddings, labels, best_epoch, converged, silhouette = (
        train_joint_phase(
            model=model,
            x_tensor=x_tensor,
            y_true=y_true,
            n_clusters=inputs.n_clusters,
            optimizer=optimizer,
            config=config,
            dataset_name=inputs.name,
        )
    )
    history.extend(joint_history)
    fit_time = perf_counter() - start

    final_kmeans = kmeans_on_embeddings(embeddings, inputs.n_clusters, config)
    labels = final_kmeans.labels_.astype(int)
    acc, nmi, ari = external_metrics(y_true, labels)
    silhouette = safe_silhouette(embeddings, labels) or float("nan")
    n_distinct_clusters = len(np.unique(labels))
    is_degenerate = n_distinct_clusters < inputs.n_clusters

    result = MvDECPaperResult(
        dataset=inputs.name,
        method_mode=config.method_mode,
        kmeans_init=config.kmeans_init,
        selected=True,
        acc=acc,
        nmi=nmi,
        ari=ari,
        silhouette=silhouette,
        cluster_sizes=np.bincount(labels, minlength=inputs.n_clusters)
        .astype(int)
        .tolist(),
        inertia=float(final_kmeans.inertia_),
        best_epoch=best_epoch,
        converged=converged,
        fit_time_seconds=float(fit_time),
        device=device_description(device),
        status="failed_degenerate_clusters" if is_degenerate else "ok",
        error_message=(
            f"K-Means found {n_distinct_clusters} distinct clusters, "
            f"expected {inputs.n_clusters}."
            if is_degenerate
            else None
        ),
    )
    return result, make_label_frame(
        inputs,
        labels,
        config.method_mode,
        config.kmeans_init,
    ), history

def run_fused_kmeans_baseline(
    inputs: DatasetInputs,
    config: MvDECPaperConfig,
) -> tuple[MvDECPaperResult, pd.DataFrame, list[TrainingHistoryRow]]:
    """Pretrain the two views, then run K-Means directly on fused embeddings."""

    set_random_seed(config.random_state)
    device = resolve_device(config.device)
    X = feature_matrix(inputs.X)
    x_tensor = torch.as_tensor(X, dtype=torch.float32, device=device)
    model = MvDECPaperModel(input_dim=X.shape[1], config=config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    start = perf_counter()
    history = train_pretrain_phase(model, x_tensor, optimizer, config, inputs.name)
    embeddings = (
        model.encode_fused(x_tensor, batch_size=min(config.batch_size, len(x_tensor)))
        .detach()
        .cpu()
        .numpy()
    )
    final_kmeans = kmeans_on_embeddings(embeddings, inputs.n_clusters, config)
    labels = final_kmeans.labels_.astype(int)
    fit_time = perf_counter() - start

    y_true = true_labels(inputs.y)
    acc, nmi, ari = external_metrics(y_true, labels)
    silhouette = safe_silhouette(embeddings, labels) or float("nan")
    n_distinct_clusters = len(np.unique(labels))
    is_degenerate = n_distinct_clusters < inputs.n_clusters

    result = MvDECPaperResult(
        dataset=inputs.name,
        method_mode=config.method_mode,
        kmeans_init=config.kmeans_init,
        selected=True,
        acc=acc,
        nmi=nmi,
        ari=ari,
        silhouette=silhouette,
        cluster_sizes=np.bincount(labels, minlength=inputs.n_clusters)
        .astype(int)
        .tolist(),
        inertia=float(final_kmeans.inertia_),
        best_epoch=config.pretrain_epochs,
        converged=False,
        fit_time_seconds=float(fit_time),
        device=device_description(device),
        status="failed_degenerate_clusters" if is_degenerate else "ok",
        error_message=(
            f"K-Means found {n_distinct_clusters} distinct clusters, "
            f"expected {inputs.n_clusters}."
            if is_degenerate
            else None
        ),
    )
    return result, make_label_frame(
        inputs,
        labels,
        config.method_mode,
        config.kmeans_init,
    ), history

def run_legacy_notebook_baseline(
    inputs: DatasetInputs,
    config: MvDECPaperConfig,
) -> tuple[MvDECPaperResult, pd.DataFrame, list[TrainingHistoryRow]]:
    """Run the TensorFlow pipeline from legacy/representation.ipynb."""

    from representation_learning.mvdec_representation import (  # noqa: PLC0415
        MvDECRepresentationConfig,
        train_mvdec_representation,
    )

    set_random_seed(config.random_state)
    X = feature_matrix(inputs.X)
    legacy_config = MvDECRepresentationConfig(
        seed=config.random_state,
        n_iterations=20,
        n_clusters=inputs.n_clusters,
        batch_size=128,
        epochs=100,
        validation_split=0.1,
        early_stopping_patience=10,
        learning_rate=1e-4,
        view1_latent_dim=4,
        kmeans_n_init=10,
        init_methods=("k-means++", "random"),
    )

    start = perf_counter()
    legacy_result, legacy_history = train_mvdec_representation(
        X=X,
        config=legacy_config,
        expected_fused_dim=None,
    )
    fit_time = perf_counter() - start

    embeddings = np.asarray(legacy_result["h_fused"], dtype=np.float32)
    labels = np.asarray(legacy_result["labels"], dtype=int)
    best_init = str(legacy_result["init"])
    y_true = true_labels(inputs.y)
    acc, nmi, ari = external_metrics(y_true, labels)
    silhouette = safe_silhouette(embeddings, labels) or float("nan")
    n_distinct_clusters = len(np.unique(labels))
    is_degenerate = n_distinct_clusters < inputs.n_clusters

    history = [
        TrainingHistoryRow(
            dataset=inputs.name,
            method_mode=config.method_mode,
            kmeans_init=str(row["init"]),
            phase="legacy_notebook",
            epoch=int(row["iteration"]),
            reconstruction_loss=float("nan"),
            kmeans_loss=None,
            orthonormal_loss=None,
            greedy_loss=None,
            total_loss=float("nan"),
            acc=None,
            nmi=None,
            ari=None,
            silhouette=float(row["silhouette"]),
            label_change_rate=None,
        )
        for row in legacy_history.to_dict("records")
    ]

    result = MvDECPaperResult(
        dataset=inputs.name,
        method_mode=config.method_mode,
        kmeans_init=best_init,
        selected=True,
        acc=acc,
        nmi=nmi,
        ari=ari,
        silhouette=silhouette,
        cluster_sizes=np.bincount(labels, minlength=inputs.n_clusters)
        .astype(int)
        .tolist(),
        inertia=float("nan"),
        best_epoch=int(legacy_result["iteration"]),
        converged=False,
        fit_time_seconds=float(fit_time),
        device="tensorflow",
        status="failed_degenerate_clusters" if is_degenerate else "ok",
        error_message=(
            f"K-Means found {n_distinct_clusters} distinct clusters, "
            f"expected {inputs.n_clusters}."
            if is_degenerate
            else None
        ),
    )
    return result, make_label_frame(
        inputs,
        labels,
        config.method_mode,
        best_init,
    ), history


def device_description(device: torch.device) -> str:
    """Return a readable device description for logs and workbooks."""

    if device.type == "cuda":
        return f"cuda:{torch.cuda.get_device_name(device)}"
    return "cpu"


def make_label_frame(
    inputs: DatasetInputs, labels: np.ndarray, method_mode: str, kmeans_init: str
) -> pd.DataFrame:
    """Build per-sample predicted and external-label output rows."""

    frame = pd.DataFrame(
        {
            "dataset": inputs.name,
            "method_mode": method_mode,
            "kmeans_init": kmeans_init,
            "sample_index": np.arange(len(labels), dtype=int),
            "mvdec_label": labels.astype(int),
        }
    )
    if inputs.y is not None and "label" in inputs.y.columns:
        frame["true_label"] = inputs.y["label"].to_numpy()
    return frame


def result_to_dict(result: MvDECPaperResult) -> dict:
    """Convert result dataclass to an Excel-friendly row."""

    row = asdict(result)
    row["cluster_sizes"] = json.dumps(row["cluster_sizes"])
    return row


def history_to_frame(history: list[TrainingHistoryRow]) -> pd.DataFrame:
    """Convert training history rows to a dataframe."""

    return pd.DataFrame([asdict(row) for row in history])


def metadata_frame(processed_root: Path, datasets: list[dict]) -> pd.DataFrame:
    """Build workbook metadata rows for selected datasets."""

    rows = []
    for dataset in datasets:
        metadata_path = processed_root / dataset["name"] / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "dataset": dataset["name"],
                "title": dataset["title"],
                "source": dataset["source"],
                "source_url": dataset["url"],
                "n_samples": metadata["n_samples"],
                "n_features": metadata["n_features"],
                "n_clusters": metadata["n_clusters"],
                "label_available": metadata["label_available"],
                "notes": " | ".join(metadata.get("notes", [])),
            }
        )
    return pd.DataFrame(rows)


def config_frame(config: MvDECPaperConfig) -> pd.DataFrame:
    """Build workbook rows describing the MvDEC paper configuration."""

    rows = asdict(config)
    rows["hidden_dims"] = json.dumps(list(config.hidden_dims))
    rows["paper_components"] = (
        "two autoencoders, average fused embedding, K-Means objective, "
        "within-class scatter eigenvectors, orthonormal transform, greedy loss"
    )
    rows["loss_reduction_note"] = (
        "Losses follow the paper equations as sums of squared distances on each "
        "mini-batch: L1 reconstruction, L2 K-Means, L3 Tr(V Sw V^T), and L4 "
        "greedy adjustment."
    )
    rows["best_epoch_selection_note"] = (
        "Datasets with ground-truth labels select the best epoch by ACC; unlabeled "
        "datasets select the best epoch by Silhouette."
    )
    rows["kmeans_init_grid_note"] = (
        "Baseline runners may grid-search K-Means init values and mark the best "
        "trial with selected=True."
    )
    rows["method_mode_note"] = (
        "mvdec uses the full joint objective L1+L2+L3+L4; fused-kmeans pretrains "
        "the two autoencoders, then clusters the average fused representation "
        "directly with K-Means; fused-only-kmeans is an explicit alias for the "
        "same no-L3/L4 comparison; legacy-fused-kmeans uses the legacy Tiki "
        "notebook architecture in PyTorch; legacy-notebook runs the TensorFlow "
        "notebook pipeline with 20 representation iterations."
    )
    return pd.DataFrame(
        [{"parameter": parameter, "value": value} for parameter, value in rows.items()]
    )


def paper_table_frame(results: list[MvDECPaperResult]) -> pd.DataFrame:
    """Return the compact ACC/NMI table used for paper-style comparison."""

    selected_results = [result for result in results if result.selected]
    return pd.DataFrame(
        [
            {
                "dataset": result.dataset,
                "method": result.method_mode,
                "kmeans_init": result.kmeans_init,
                "ACC": result.acc,
                "NMI": result.nmi,
                "status": result.status,
            }
            for result in selected_results
        ]
    )

def write_workbook(
    output_path: Path,
    results: list[MvDECPaperResult],
    label_frames: list[pd.DataFrame],
    history_rows: list[TrainingHistoryRow],
    metadata: pd.DataFrame,
    config: MvDECPaperConfig,
) -> None:
    """Write MvDEC paper baseline outputs into one Excel workbook."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([result_to_dict(result) for result in results])
    labels = (
        pd.concat(label_frames, ignore_index=True) if label_frames else pd.DataFrame()
    )
    history = history_to_frame(history_rows)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        paper_table_frame(results).to_excel(
            writer,
            sheet_name="paper_table",
            index=False,
        )
        labels.to_excel(writer, sheet_name="labels", index=False)
        history.to_excel(writer, sheet_name="training_history", index=False)
        metadata.to_excel(writer, sheet_name="metadata", index=False)
        config_frame(config).to_excel(writer, sheet_name="config", index=False)


def log_device(config: MvDECPaperConfig) -> None:
    """Log selected compute device before a run starts."""

    device = resolve_device(config.device)
    cuda_available = torch.cuda.is_available()
    cuda_count = torch.cuda.device_count() if cuda_available else 0
    logger.info(
        "Compute environment | requested_device={} resolved_device={} "
        "cuda_available={} cuda_device_count={}",
        config.device,
        device_description(device),
        cuda_available,
        cuda_count,
    )
    if cuda_available:
        for index in range(cuda_count):
            properties = torch.cuda.get_device_properties(index)
            logger.info(
                "CUDA device {} | name={} total_memory_gb={:.2f}",
                index,
                properties.name,
                properties.total_memory / (1024**3),
            )
