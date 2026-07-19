"""Compare cluster assignments on one shared numeric evaluation space."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import pairwise_distances

from config import PREPROCESSED_DATA_PATH
from pipeline.clustering import (
    compute_gower_distance,
    compute_silhouette_diagnostics,
)
from pipeline.data import file_sha256
from pipeline.io import replace_with_retry

COMMON_EUCLIDEAN_CONTRACT = "common_numeric_euclidean_v1"
COMMON_GOWER_CONTRACT = "common_numeric_gower_v1"
COMMON_DISTANCE_CONTRACTS = (
    COMMON_EUCLIDEAN_CONTRACT,
    COMMON_GOWER_CONTRACT,
)


@dataclass(frozen=True)
class AssignmentRun:
    """One row-ordered clustering assignment loaded from a source CSV."""

    method: str
    source_path: Path
    source_run_index: int
    random_seed: int | None
    labels: np.ndarray


def parse_source_spec(value: str) -> tuple[str, Path]:
    """Parse one METHOD=CSV source specification."""

    method, separator, raw_path = value.partition("=")
    method = method.strip()
    raw_path = raw_path.strip()
    if not separator or not method or not raw_path:
        msg = "Assignment sources must use the format METHOD=PATH.csv."
        raise ValueError(msg)
    return method, Path(raw_path)


def _validated_labels(values: object, expected_rows: int) -> np.ndarray:
    """Return finite non-negative integer labels with the expected row count."""

    labels_raw = np.asarray(values)
    if labels_raw.ndim != 1 or len(labels_raw) != expected_rows:
        msg = (
            "Cluster assignments must be one-dimensional and match the common "
            f"data rows: {labels_raw.shape} versus {expected_rows}."
        )
        raise ValueError(msg)
    try:
        labels_float = labels_raw.astype(float)
    except (TypeError, ValueError) as error:
        msg = "Cluster assignments must contain integer labels."
        raise ValueError(msg) from error
    if (
        not np.isfinite(labels_float).all()
        or (labels_float < 0).any()
        or not np.equal(labels_float, np.floor(labels_float)).all()
    ):
        msg = "Cluster assignments must contain finite non-negative integers."
        raise ValueError(msg)
    return labels_float.astype(int)


def _optional_seed(frame: pd.DataFrame) -> int | None:
    """Return one unique random seed from a row-level assignment CSV."""

    if "random_seed" not in frame.columns:
        return None
    seeds = pd.to_numeric(frame["random_seed"], errors="coerce").dropna().unique()
    if len(seeds) > 1:
        msg = "A row-level assignment CSV must contain at most one random_seed."
        raise ValueError(msg)
    return int(seeds[0]) if len(seeds) == 1 else None


def _load_row_level_assignments(
    method: str,
    source_path: Path,
    frame: pd.DataFrame,
    expected_rows: int,
) -> list[AssignmentRun]:
    """Load one assignment stored as a cluster column with one row per sample."""

    ordered = frame.copy()
    row_key = next(
        (column for column in ("orig_index", "original_index") if column in ordered),
        None,
    )
    if row_key is not None:
        ordered[row_key] = pd.to_numeric(ordered[row_key], errors="raise").astype(int)
        ordered = ordered.sort_values(row_key)
        if ordered[row_key].tolist() != list(range(expected_rows)):
            msg = f"Row-level assignment IDs must equal 0..{expected_rows - 1}."
            raise ValueError(msg)
    labels = _validated_labels(ordered["cluster"].to_numpy(), expected_rows)
    return [
        AssignmentRun(
            method=method,
            source_path=source_path,
            source_run_index=0,
            random_seed=_optional_seed(ordered),
            labels=labels,
        )
    ]


def _load_serialized_assignments(
    method: str,
    source_path: Path,
    frame: pd.DataFrame,
    expected_rows: int,
) -> list[AssignmentRun]:
    """Load one or more JSON cluster_assignments rows from an experiment CSV."""

    usable = frame.copy()
    if "status" in usable.columns:
        usable = usable[usable["status"] == "ok"]
    usable = usable[usable["cluster_assignments"].notna()]
    usable = usable[usable["cluster_assignments"].astype(str).str.strip() != ""]
    if usable.empty:
        msg = f"No successful cluster assignments found in {source_path}."
        raise ValueError(msg)

    parsed: list[tuple[int | None, str, np.ndarray]] = []
    for row in usable.itertuples(index=False):
        assignment_text = str(row.cluster_assignments)
        try:
            assignment_values = json.loads(assignment_text)
        except json.JSONDecodeError as error:
            msg = f"Invalid cluster_assignments JSON in {source_path}."
            raise ValueError(msg) from error
        labels = _validated_labels(assignment_values, expected_rows)
        raw_seed = getattr(row, "random_seed", None)
        seed = None if raw_seed is None or pd.isna(raw_seed) else int(raw_seed)
        canonical = json.dumps(labels.tolist(), separators=(",", ":"))
        parsed.append((seed, canonical, labels))

    unique: dict[tuple[int | None, str], np.ndarray] = {}
    assignments_by_seed: dict[int | None, set[str]] = {}
    for seed, canonical, labels in parsed:
        assignments_by_seed.setdefault(seed, set()).add(canonical)
        unique[(seed, canonical)] = labels
    conflicting = [
        seed for seed, values in assignments_by_seed.items() if len(values) > 1
    ]
    if conflicting:
        msg = (
            "Each method and seed must identify one fixed assignment; conflicting "
            f"seeds in {source_path}: {conflicting}."
        )
        raise ValueError(msg)
    if len(unique) > 1 and all(seed is None for seed, _ in unique):
        msg = "Multiple assignments require a random_seed column."
        raise ValueError(msg)

    return [
        AssignmentRun(
            method=method,
            source_path=source_path,
            source_run_index=index,
            random_seed=seed,
            labels=labels,
        )
        for index, ((seed, _), labels) in enumerate(unique.items())
    ]


def load_assignment_source(
    method: str,
    source_path: Path,
    expected_rows: int,
) -> list[AssignmentRun]:
    """Load row-level or serialized cluster assignments from one CSV."""

    frame = pd.read_csv(source_path)
    if "cluster_assignments" in frame.columns:
        return _load_serialized_assignments(
            method,
            source_path,
            frame,
            expected_rows,
        )
    if "cluster" in frame.columns:
        return _load_row_level_assignments(
            method,
            source_path,
            frame,
            expected_rows,
        )
    msg = f"Assignment CSV must contain cluster_assignments or cluster: {source_path}."
    raise ValueError(msg)


def load_common_numeric_data(data_path: Path) -> pd.DataFrame:
    """Load a finite all-numeric common evaluation dataset."""

    frame = pd.read_csv(data_path)
    if frame.empty or frame.shape[1] == 0:
        msg = "Common evaluation data must contain rows and numeric columns."
        raise ValueError(msg)
    numeric = frame.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        msg = "Common evaluation data contains NaN or infinite values."
        raise ValueError(msg)
    return numeric


def evaluate_common_assignments(
    common_df: pd.DataFrame,
    assignments: list[AssignmentRun],
    n_clusters: int,
) -> pd.DataFrame:
    """Score every assignment using the same Euclidean and numeric-Gower spaces."""

    if n_clusters < 2:
        msg = "Common evaluation requires n_clusters >= 2."
        raise ValueError(msg)
    if not assignments:
        msg = "At least one assignment source is required."
        raise ValueError(msg)

    distance_matrices = {
        COMMON_EUCLIDEAN_CONTRACT: pairwise_distances(
            common_df.to_numpy(dtype=float),
            metric="euclidean",
        ),
        COMMON_GOWER_CONTRACT: compute_gower_distance(common_df),
    }
    records: list[dict[str, object]] = []
    for assignment in assignments:
        labels = _validated_labels(assignment.labels, len(common_df))
        unique_labels, counts = np.unique(labels, return_counts=True)
        sizes = counts.astype(int).tolist()
        base = {
            "method": assignment.method,
            "source_path": str(assignment.source_path),
            "source_sha256": file_sha256(assignment.source_path),
            "source_run_index": assignment.source_run_index,
            "random_seed": assignment.random_seed,
            "n_clusters": n_clusters,
            "cluster_sizes": json.dumps(sizes, separators=(",", ":")),
            "cluster_assignments": json.dumps(
                labels.tolist(),
                separators=(",", ":"),
            ),
        }
        if len(unique_labels) != n_clusters:
            status = "failed_cluster_count"
            error_message = (
                f"Expected {n_clusters} clusters but found {len(unique_labels)}."
            )
        elif (counts == 1).any():
            status = "failed_singleton_cluster"
            error_message = "Singleton clusters are not valid comparison results."
        else:
            status = "ok"
            error_message = None

        for contract, distance_matrix in distance_matrices.items():
            record = {
                **base,
                "distance_contract": contract,
                "silhouette_score": np.nan,
                "silhouette_sample_std": np.nan,
                "silhouette_negative_fraction": np.nan,
                "status": status,
                "error_message": error_message,
            }
            if status == "ok":
                score, sample_std, negative_fraction = compute_silhouette_diagnostics(
                    distance_matrix, labels
                )
                record.update(
                    {
                        "silhouette_score": score,
                        "silhouette_sample_std": sample_std,
                        "silhouette_negative_fraction": negative_fraction,
                    }
                )
            records.append(record)
    return pd.DataFrame.from_records(records)


def aggregate_common_evaluation(runs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate successful common-space Silhouette results by method and metric."""

    records: list[dict[str, object]] = []
    for (method, contract), group in runs.groupby(
        ["method", "distance_contract"],
        sort=True,
        dropna=False,
    ):
        successful = group[group["status"] == "ok"]
        scores = pd.to_numeric(successful["silhouette_score"], errors="coerce").dropna()
        records.append(
            {
                "method": method,
                "distance_contract": contract,
                "n_clusters": int(group.iloc[0]["n_clusters"]),
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
                    successful["silhouette_sample_std"],
                    errors="coerce",
                ).mean(),
                "negative_fraction_mean": pd.to_numeric(
                    successful["silhouette_negative_fraction"],
                    errors="coerce",
                ).mean(),
            }
        )
    return pd.DataFrame.from_records(records)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Atomically write one common-evaluation CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    frame.to_csv(temporary_path, index=False)
    replace_with_retry(temporary_path, path)


def parse_args() -> argparse.Namespace:
    """Parse common-space evaluation arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Compare row-ordered clustering assignments using shared Euclidean "
            "and numeric-Gower Silhouette metrics."
        )
    )
    parser.add_argument("--data-path", type=Path, default=PREPROCESSED_DATA_PATH)
    parser.add_argument("--n-clusters", type=int, required=True)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="METHOD=PATH.csv",
        help=(
            "Repeat for each method. CSV may contain one row per sample in a "
            "cluster column or JSON cluster_assignments rows."
        ),
    )
    parser.add_argument("--runs-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run common-space evaluation for all requested assignment sources."""

    args = parse_args()
    outputs = (args.runs_output, args.summary_output)
    existing = [path for path in outputs if path.exists()]
    if existing and not args.force:
        msg = f"Output already exists: {existing}. Use --force to overwrite it."
        raise FileExistsError(msg)

    common_df = load_common_numeric_data(args.data_path)
    assignments: list[AssignmentRun] = []
    for source_spec in args.source:
        method, source_path = parse_source_spec(source_spec)
        assignments.extend(load_assignment_source(method, source_path, len(common_df)))
    runs = evaluate_common_assignments(common_df, assignments, args.n_clusters)
    runs.insert(2, "data_path", str(args.data_path))
    runs.insert(3, "data_sha256", file_sha256(args.data_path))
    summary = aggregate_common_evaluation(runs)
    summary.insert(2, "data_path", str(args.data_path))
    summary.insert(3, "data_sha256", file_sha256(args.data_path))
    _write_csv(runs, args.runs_output)
    _write_csv(summary, args.summary_output)
    logger.info("Saved common evaluation runs to {}", args.runs_output)
    logger.info("Saved common evaluation summary to {}", args.summary_output)


if __name__ == "__main__":
    main()
