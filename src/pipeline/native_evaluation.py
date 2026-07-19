"""Evaluate continuous baseline clusterings in their native representations."""

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from pipeline.clustering import (
    compute_gower_distance,
    compute_silhouette_diagnostics,
)
from pipeline.data import file_sha256
from pipeline.experiments import serialize_cluster_assignments
from pipeline.io import replace_with_retry

NATIVE_NUMERIC_GOWER_CONTRACT = "native_numeric_gower_v1"
CSV_LABEL_COLUMNS = ("cluster", "label", "kmeans_label")
CSV_ROW_ID_COLUMNS = ("orig_index", "original_index", "sample_index")
CSV_METADATA_COLUMNS = {
    "dataset",
    "kmeans_init",
    "random_seed",
    "seed",
    "true_label",
}


@dataclass(frozen=True)
class NativeRun:
    """One continuous native representation and its row-ordered labels."""

    method: str
    source_path: Path
    representation_name: str
    random_seed: int | None
    representation: pd.DataFrame
    labels: np.ndarray


def parse_source_spec(value: str) -> tuple[str, Path, str]:
    """Parse METHOD=PATH[#REPRESENTATION_KEY]."""

    method, separator, raw_source = value.partition("=")
    method = method.strip()
    raw_source = raw_source.strip()
    if not separator or not method or not raw_source:
        msg = "Native sources must use METHOD=PATH or METHOD=PATH#KEY."
        raise ValueError(msg)
    raw_path, key_separator, raw_key = raw_source.partition("#")
    representation_key = raw_key.strip() if key_separator else "h_fused"
    if not raw_path.strip() or not representation_key:
        msg = "Native source path and representation key must not be empty."
        raise ValueError(msg)
    return method, Path(raw_path.strip()), representation_key


def _validated_representation(values: object, source_path: Path) -> pd.DataFrame:
    """Return one finite two-dimensional numeric representation."""

    representation = np.asarray(values)
    if representation.ndim != 2 or representation.shape[1] == 0:
        msg = f"Native representation must be a non-empty 2D matrix: {source_path}."
        raise ValueError(msg)
    try:
        numeric = representation.astype(float)
    except (TypeError, ValueError) as error:
        msg = f"Native representation must be numeric: {source_path}."
        raise ValueError(msg) from error
    if not np.isfinite(numeric).all():
        msg = f"Native representation contains NaN or infinite values: {source_path}."
        raise ValueError(msg)
    return pd.DataFrame(
        numeric,
        columns=[f"feature_{index + 1}" for index in range(numeric.shape[1])],
    )


def _validated_labels(
    values: object,
    expected_rows: int,
    n_clusters: int,
    source_path: Path,
) -> np.ndarray:
    """Return contiguous cluster labels with no missing or singleton cluster."""

    labels_raw = np.asarray(values)
    if labels_raw.ndim != 1 or len(labels_raw) != expected_rows:
        msg = f"Labels do not match native representation rows: {source_path}."
        raise ValueError(msg)
    try:
        labels_float = labels_raw.astype(float)
    except (TypeError, ValueError) as error:
        msg = f"Cluster labels must be integers: {source_path}."
        raise ValueError(msg) from error
    if (
        not np.isfinite(labels_float).all()
        or not np.equal(
            labels_float,
            np.floor(labels_float),
        ).all()
    ):
        msg = f"Cluster labels must be finite integers: {source_path}."
        raise ValueError(msg)
    labels = labels_float.astype(int)
    if set(np.unique(labels)) != set(range(n_clusters)):
        msg = f"Labels must use every cluster ID from 0 to {n_clusters - 1}."
        raise ValueError(msg)
    sizes = np.bincount(labels, minlength=n_clusters)
    if (sizes == 1).any():
        msg = f"Singleton clusters are not valid native results: {source_path}."
        raise ValueError(msg)
    return labels


def _artifact_seed(artifact: dict) -> int | None:
    """Return a recorded artifact seed when available."""

    config = artifact.get("config")
    raw_seed = config.get("seed") if isinstance(config, dict) else None
    if raw_seed is None:
        raw_seed = artifact.get("random_seed")
    return None if raw_seed is None else int(raw_seed)


def _load_pickle_source(
    method: str,
    source_path: Path,
    representation_key: str,
    n_clusters: int,
) -> NativeRun:
    """Load a trusted local representation artifact."""

    with source_path.open("rb") as file:
        artifact = pickle.load(file)
    if not isinstance(artifact, dict):
        msg = f"Native artifact must contain a dictionary: {source_path}."
        raise ValueError(msg)
    config = artifact.get("config")
    recorded_counts = [artifact.get("n_clusters")]
    if isinstance(config, dict):
        recorded_counts.append(config.get("n_clusters"))
    if any(value is not None and int(value) != n_clusters for value in recorded_counts):
        msg = f"Artifact cluster count does not match n_clusters={n_clusters}."
        raise ValueError(msg)
    if representation_key not in artifact or "labels" not in artifact:
        msg = (
            f"Artifact must contain {representation_key!r} and 'labels': {source_path}."
        )
        raise ValueError(msg)
    representation = _validated_representation(
        artifact[representation_key],
        source_path,
    )
    labels = _validated_labels(
        artifact["labels"],
        len(representation),
        n_clusters,
        source_path,
    )
    return NativeRun(
        method=method,
        source_path=source_path,
        representation_name=representation_key,
        random_seed=_artifact_seed(artifact),
        representation=representation,
        labels=labels,
    )


def _unique_optional_integer(
    frame: pd.DataFrame, columns: tuple[str, ...]
) -> int | None:
    """Return one unique optional integer from the first available column."""

    column = next((candidate for candidate in columns if candidate in frame), None)
    if column is None:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna().unique()
    if len(values) > 1:
        msg = f"Native row-level CSV must contain at most one {column}."
        raise ValueError(msg)
    return int(values[0]) if len(values) == 1 else None


def _load_csv_source(
    method: str,
    source_path: Path,
    representation_key: str,
    n_clusters: int,
) -> NativeRun:
    """Load numeric representation columns and labels from one row-level CSV."""

    if representation_key != "h_fused":
        msg = "CSV native sources do not accept a #REPRESENTATION_KEY suffix."
        raise ValueError(msg)
    frame = pd.read_csv(source_path)
    label_column = next(
        (column for column in CSV_LABEL_COLUMNS if column in frame), None
    )
    if label_column is None:
        msg = f"Native CSV is missing cluster, label, or kmeans_label: {source_path}."
        raise ValueError(msg)

    row_id_column = next(
        (column for column in CSV_ROW_ID_COLUMNS if column in frame),
        None,
    )
    ordered = frame.copy()
    if row_id_column is not None:
        ordered[row_id_column] = pd.to_numeric(
            ordered[row_id_column],
            errors="raise",
        ).astype(int)
        ordered = ordered.sort_values(row_id_column)
        if ordered[row_id_column].tolist() != list(range(len(ordered))):
            msg = f"Native row IDs must equal 0..{len(ordered) - 1}: {source_path}."
            raise ValueError(msg)

    excluded = {
        label_column,
        *CSV_ROW_ID_COLUMNS,
        *CSV_METADATA_COLUMNS,
    }
    feature_columns = [column for column in ordered.columns if column not in excluded]
    if not feature_columns:
        msg = f"Native CSV contains no representation columns: {source_path}."
        raise ValueError(msg)
    try:
        numeric = ordered[feature_columns].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as error:
        msg = f"Native CSV representation columns must be numeric: {source_path}."
        raise ValueError(msg) from error
    representation = _validated_representation(numeric.to_numpy(), source_path)
    representation.columns = feature_columns
    labels = _validated_labels(
        ordered[label_column].to_numpy(),
        len(representation),
        n_clusters,
        source_path,
    )
    return NativeRun(
        method=method,
        source_path=source_path,
        representation_name=", ".join(feature_columns),
        random_seed=_unique_optional_integer(ordered, ("random_seed", "seed")),
        representation=representation,
        labels=labels,
    )


def load_native_source(
    method: str,
    source_path: Path,
    representation_key: str,
    n_clusters: int,
) -> NativeRun:
    """Load one pickle artifact or row-level CSV native result."""

    suffix = source_path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        return _load_pickle_source(
            method,
            source_path,
            representation_key,
            n_clusters,
        )
    if suffix == ".csv":
        return _load_csv_source(
            method,
            source_path,
            representation_key,
            n_clusters,
        )
    msg = f"Unsupported native source type {suffix!r}: {source_path}."
    raise ValueError(msg)


def evaluate_native_runs(runs: list[NativeRun], n_clusters: int) -> pd.DataFrame:
    """Compute numeric-Gower Silhouette diagnostics in each method's native space."""

    if n_clusters < 2:
        msg = "Native evaluation requires n_clusters >= 2."
        raise ValueError(msg)
    if not runs:
        msg = "At least one native source is required."
        raise ValueError(msg)
    records: list[dict[str, object]] = []
    for run in runs:
        distance_matrix = compute_gower_distance(run.representation)
        score, sample_std, negative_fraction = compute_silhouette_diagnostics(
            distance_matrix,
            run.labels,
        )
        sizes = np.bincount(run.labels, minlength=n_clusters).astype(int).tolist()
        records.append(
            {
                "method": run.method,
                "source_path": str(run.source_path),
                "source_sha256": file_sha256(run.source_path),
                "representation": run.representation_name,
                "random_seed": run.random_seed,
                "n_samples": len(run.representation),
                "n_features": run.representation.shape[1],
                "n_clusters": n_clusters,
                "distance_contract": NATIVE_NUMERIC_GOWER_CONTRACT,
                "silhouette_score": score,
                "silhouette_sample_std": sample_std,
                "silhouette_negative_fraction": negative_fraction,
                "cluster_sizes": json.dumps(sizes, separators=(",", ":")),
                "cluster_assignments": serialize_cluster_assignments(run.labels),
                "status": "ok",
                "error_message": None,
            }
        )
    return pd.DataFrame.from_records(records)


def aggregate_native_evaluation(runs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate native numeric-Gower results by method."""

    records: list[dict[str, object]] = []
    for method, group in runs.groupby("method", sort=True):
        scores = pd.to_numeric(group["silhouette_score"], errors="coerce").dropna()
        records.append(
            {
                "method": method,
                "distance_contract": NATIVE_NUMERIC_GOWER_CONTRACT,
                "requested_runs": len(group),
                "successful_runs": len(scores),
                "failed_runs": len(group) - len(scores),
                "silhouette_mean": scores.mean(),
                "silhouette_std_across_seeds": (
                    scores.std(ddof=1) if len(scores) > 1 else 0.0
                ),
                "silhouette_min": scores.min(),
                "silhouette_max": scores.max(),
                "sample_std_mean": pd.to_numeric(
                    group["silhouette_sample_std"],
                    errors="coerce",
                ).mean(),
                "negative_fraction_mean": pd.to_numeric(
                    group["silhouette_negative_fraction"],
                    errors="coerce",
                ).mean(),
            }
        )
    return pd.DataFrame.from_records(records)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Atomically write one native-evaluation CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    frame.to_csv(temporary_path, index=False)
    replace_with_retry(temporary_path, path)


def parse_args() -> argparse.Namespace:
    """Parse native continuous evaluation arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate continuous baseline clusterings with numeric-Gower "
            "Silhouette in each method's native representation."
        )
    )
    parser.add_argument("--n-clusters", type=int, required=True)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="METHOD=PATH[#KEY]",
        help=(
            "Repeat for each run. Trusted pickle artifacts default to h_fused "
            "and labels; CSV files require numeric representation columns and "
            "a cluster, label, or kmeans_label column."
        ),
    )
    parser.add_argument("--runs-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run native numeric-Gower evaluation for requested continuous baselines."""

    args = parse_args()
    if args.n_clusters < 2:
        msg = "Native evaluation requires n_clusters >= 2."
        raise ValueError(msg)
    outputs = (args.runs_output, args.summary_output)
    existing = [path for path in outputs if path.exists()]
    if existing and not args.force:
        msg = f"Output already exists: {existing}. Use --force to overwrite it."
        raise FileExistsError(msg)

    native_runs = []
    for source_spec in args.source:
        method, source_path, representation_key = parse_source_spec(source_spec)
        native_runs.append(
            load_native_source(
                method,
                source_path,
                representation_key,
                args.n_clusters,
            )
        )
    runs = evaluate_native_runs(native_runs, args.n_clusters)
    summary = aggregate_native_evaluation(runs)
    _write_csv(runs, args.runs_output)
    _write_csv(summary, args.summary_output)
    logger.info("Saved native evaluation runs to {}", args.runs_output)
    logger.info("Saved native evaluation summary to {}", args.summary_output)


if __name__ == "__main__":
    main()
