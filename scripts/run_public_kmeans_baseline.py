"""Run K-Means baselines for selected public datasets."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_DIR / "configs" / "public_datasets.json"
DEFAULT_OUTPUT_PATH = (
    PROJECT_DIR / "output" / "public_baselines" / "kmeans_baseline.xlsx"
)
RANDOM_STATE = 42
N_INIT = 100
MAX_ITER = 1000


@dataclass(frozen=True)
class DatasetInputs:
    """Loaded public dataset inputs."""

    name: str
    title: str
    n_clusters: int
    X: pd.DataFrame
    y: pd.DataFrame | None
    metadata: dict


@dataclass(frozen=True)
class KMeansBaselineResult:
    """K-Means result summary for one dataset."""

    dataset: str
    title: str
    n_samples: int
    n_features: int
    n_clusters: int
    scaler: str
    algorithm: str
    init: str
    n_init: int
    max_iter: int
    random_state: int
    silhouette: float
    ari: float | None
    nmi: float | None
    cluster_sizes: list[int]
    inertia: float
    n_iter: int
    fit_time_seconds: float
    status: str
    error_message: str | None


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(
        description="Run StandardScaler + K-Means baselines on public datasets."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Public dataset registry JSON path.",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=None,
        help="Override processed dataset root. Defaults to the registry value.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Excel workbook output path.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset names to run. Defaults to every registry entry.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing workbook.",
    )
    return parser.parse_args()


def load_registry(config_path: Path) -> dict:
    """Load the public dataset registry."""

    with config_path.open(encoding="utf-8") as file:
        return json.load(file)


def selected_datasets(registry: dict, names: list[str] | None) -> list[dict]:
    """Return registry entries selected by name."""

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


def resolve_processed_root(registry: dict, override: Path | None) -> Path:
    """Resolve the processed dataset root path."""

    if override is not None:
        return override.resolve()
    return (PROJECT_DIR / registry["processed_root"]).resolve()


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
    """Validate processed inputs before clustering."""

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
    """Return the external label column when available."""

    if y is None or "label" not in y.columns:
        return None
    return y["label"].to_numpy()


def run_kmeans_baseline(
    inputs: DatasetInputs,
) -> tuple[KMeansBaselineResult, pd.DataFrame]:
    """Fit StandardScaler + K-Means and return metrics and labels."""

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(inputs.X.to_numpy(dtype=float))

    model = KMeans(
        n_clusters=inputs.n_clusters,
        init="k-means++",
        n_init=N_INIT,
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
    )
    start = perf_counter()
    labels = model.fit_predict(X_scaled)
    fit_time = perf_counter() - start

    silhouette = float(silhouette_score(X_scaled, labels, metric="euclidean"))
    labels_true = true_labels(inputs.y)
    ari = (
        None if labels_true is None else float(adjusted_rand_score(labels_true, labels))
    )
    nmi = (
        None
        if labels_true is None
        else float(normalized_mutual_info_score(labels_true, labels))
    )
    cluster_sizes = (
        np.bincount(labels, minlength=inputs.n_clusters).astype(int).tolist()
    )

    result = KMeansBaselineResult(
        dataset=inputs.name,
        title=inputs.title,
        n_samples=len(inputs.X),
        n_features=inputs.X.shape[1],
        n_clusters=inputs.n_clusters,
        scaler="StandardScaler",
        algorithm="KMeans",
        init="k-means++",
        n_init=N_INIT,
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
        silhouette=silhouette,
        ari=ari,
        nmi=nmi,
        cluster_sizes=cluster_sizes,
        inertia=float(model.inertia_),
        n_iter=int(model.n_iter_),
        fit_time_seconds=float(fit_time),
        status="ok",
        error_message=None,
    )
    label_df = make_label_frame(inputs, labels)
    return result, label_df


def make_label_frame(inputs: DatasetInputs, labels: np.ndarray) -> pd.DataFrame:
    """Build a per-sample label output frame."""

    frame = pd.DataFrame(
        {
            "dataset": inputs.name,
            "sample_index": np.arange(len(labels), dtype=int),
            "kmeans_label": labels.astype(int),
        }
    )
    if inputs.y is not None and "label" in inputs.y.columns:
        frame["true_label"] = inputs.y["label"].to_numpy()
    return frame


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


def config_frame() -> pd.DataFrame:
    """Build workbook rows describing the baseline configuration."""

    return pd.DataFrame(
        [
            {"parameter": "scaler", "value": "StandardScaler"},
            {"parameter": "algorithm", "value": "KMeans"},
            {"parameter": "init", "value": "k-means++"},
            {"parameter": "n_init", "value": N_INIT},
            {"parameter": "max_iter", "value": MAX_ITER},
            {"parameter": "random_state", "value": RANDOM_STATE},
            {"parameter": "internal_metric", "value": "silhouette_euclidean"},
            {"parameter": "external_metrics", "value": "ARI, NMI"},
        ]
    )


def result_to_dict(result: KMeansBaselineResult) -> dict:
    """Convert one baseline result to an Excel-friendly row."""

    return {
        "dataset": result.dataset,
        "silhouette": result.silhouette,
        "ari": result.ari,
        "nmi": result.nmi,
        "cluster_sizes": json.dumps(result.cluster_sizes),
        "inertia": result.inertia,
        "n_iter": result.n_iter,
        "fit_time_seconds": result.fit_time_seconds,
        "status": result.status,
        "error_message": result.error_message,
    }


def write_workbook(
    output_path: Path,
    results: list[KMeansBaselineResult],
    label_frames: list[pd.DataFrame],
    metadata: pd.DataFrame,
) -> None:
    """Write all baseline outputs into one Excel workbook."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([result_to_dict(result) for result in results])
    labels = (
        pd.concat(label_frames, ignore_index=True) if label_frames else pd.DataFrame()
    )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        labels.to_excel(writer, sheet_name="labels", index=False)
        metadata.to_excel(writer, sheet_name="metadata", index=False)
        config_frame().to_excel(writer, sheet_name="config", index=False)


def main() -> None:
    """Run K-Means baselines and save the Excel workbook."""

    args = parse_args()
    if args.output.exists() and not args.force:
        msg = f"Output already exists: {args.output}. Use --force to overwrite."
        raise FileExistsError(msg)

    registry = load_registry(args.config)
    datasets = selected_datasets(registry, args.datasets)
    processed_root = resolve_processed_root(registry, args.processed_root)

    results: list[KMeansBaselineResult] = []
    label_frames: list[pd.DataFrame] = []
    for dataset in datasets:
        logger.info("Running K-Means baseline for {}", dataset["name"])
        inputs = load_dataset(dataset, processed_root)
        result, label_frame = run_kmeans_baseline(inputs)
        results.append(result)
        label_frames.append(label_frame)
        logger.info(
            "{} silhouette={:.4f} ari={} nmi={}",
            result.dataset,
            result.silhouette,
            None if result.ari is None else round(result.ari, 4),
            None if result.nmi is None else round(result.nmi, 4),
        )

    metadata = metadata_frame(processed_root, datasets)
    write_workbook(args.output, results, label_frames, metadata)
    logger.info("Saved K-Means baseline workbook to {}", args.output)


if __name__ == "__main__":
    main()


