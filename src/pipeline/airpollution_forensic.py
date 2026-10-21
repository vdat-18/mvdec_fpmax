"""Forensic audit of the Air Pollution matrix against the 2025 paper."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import __version__ as scipy_version
from scipy.optimize import linear_sum_assignment
from sklearn import __version__ as sklearn_version
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

PAPER_K = 4
PAPER_KMEANS_SILHOUETTE = 0.2829
PAPER_K4_DISTORTION = 149.646

SOURCE_COLUMNS = (
    "satellite_no2",
    "highway_primary",
    "high_traffic",
    "medium_traffic",
    "low_traffic",
    "landuse_industrial_variance",
    "landuse_industrial_distance_mean",
    "landuse_industrial_b5",
    "landuse_residential_b5",
    "landuse_forest_b5",
    "neighbours_no2_variance",
    "neighbours_no2_distance_mean",
    "neighbours_no2_distance_max",
)

PAPER_COLUMNS = (
    "highway_primary",
    "high_traffic",
    "medium_traffic",
    "low_traffic",
    "landuse_industrial_variance",
    "landuse_industrial_distance_mean",
    "landuse_industrial_b5",
    "landuse_residential_b5",
    "landuse_forest_b5",
    "neighbours_no2_variance",
    "neighbours_no2_distance_mean",
    "neighbours_no2_distance_max",
    "satellite_no2",
)

# Table 4, page 14. Values are transcribed in the printed column order.
PAPER_TABLE4 = np.asarray(
    [
        [
            0.0964,
            0.1276,
            0.1101,
            0.2885,
            0.6573,
            0.1500,
            0.0000,
            0.2500,
            0.0000,
            0.2270,
            0.0239,
            0.5950,
            0.3730,
        ],
        [
            0.1140,
            0.0728,
            0.1562,
            0.2524,
            0.6023,
            0.0000,
            0.0000,
            0.5000,
            0.0000,
            0.2690,
            0.2150,
            0.6840,
            0.4290,
        ],
        [
            0.0828,
            0.0355,
            0.1422,
            0.2163,
            0.6820,
            0.0500,
            0.0000,
            0.2500,
            0.0000,
            0.2890,
            0.2660,
            0.5280,
            0.4490,
        ],
        [
            0.0803,
            0.0612,
            0.0995,
            0.1803,
            0.5914,
            0.9500,
            0.0000,
            0.0000,
            0.1429,
            0.2610,
            0.3300,
            0.6520,
            0.4720,
        ],
        [
            0.0536,
            0.0901,
            0.0657,
            0.1444,
            0.6023,
            0.2500,
            0.1667,
            0.2500,
            0.1429,
            0.2760,
            0.3160,
            0.4800,
            0.4890,
        ],
        [
            0.0014,
            0.0455,
            0.0586,
            0.1087,
            0.5624,
            0.9500,
            0.1667,
            0.2500,
            0.1429,
            0.2100,
            0.2550,
            0.4700,
            0.4950,
        ],
    ],
    dtype=np.float64,
)

TRAFFIC_COLUMNS = PAPER_COLUMNS[:4]
LANDUSE_COLUMNS = PAPER_COLUMNS[4:9]
POLLUTION_COLUMNS = PAPER_COLUMNS[9:]
BOUNDED_LANDUSE_COLUMNS = (
    "landuse_industrial_b5",
    "landuse_residential_b5",
    "landuse_forest_b5",
)


@dataclass(frozen=True)
class Candidate:
    """One declared preprocessing hypothesis and its transformed matrix."""

    name: str
    description: str
    matrix: np.ndarray
    diagnostic_only: bool = False


@dataclass(frozen=True)
class TableMatch:
    """Unique nearest-row comparison with the six printed Table 4 rows."""

    mean_rmse: float
    max_rmse: float
    exact_rounded_rows: int
    source_row_indices: list[int]
    row_rmse: list[float]


@dataclass(frozen=True)
class CandidateResult:
    """Metrics for one preprocessing candidate."""

    name: str
    description: str
    diagnostic_only: bool
    silhouette_by_seed: dict[str, float]
    silhouette_mean: float
    silhouette_std: float
    silhouette_min: float
    silhouette_max: float
    paper_silhouette_gap_abs: float
    inertia_k4_by_seed: dict[str, float]
    cluster_sizes_by_seed: dict[str, list[int]]
    table4: TableMatch
    elbow_inertia: dict[str, float]
    scale_factor_to_paper_k4_distortion: float


def default_data_path() -> Path:
    """Return the repository's author-supplied Air Pollution CSV path."""

    return (
        Path(__file__).resolve().parents[2]
        / "data"
        / "preprocessed_data"
        / "data_demvk.csv"
    )


def load_airpollution_csv(path: Path) -> pd.DataFrame:
    """Load and strictly validate the author-supplied 13-column matrix."""

    if not path.is_file():
        raise FileNotFoundError(f"Air Pollution CSV not found: {path}")
    frame = pd.read_csv(path)
    actual = tuple(frame.columns)
    if actual != SOURCE_COLUMNS:
        raise ValueError(
            "Unexpected Air Pollution schema. "
            f"Expected {list(SOURCE_COLUMNS)}, got {list(actual)}."
        )
    if frame.empty:
        raise ValueError("Air Pollution CSV must contain at least one row.")
    try:
        numeric = frame.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Air Pollution CSV contains non-numeric values.") from exc
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Air Pollution CSV contains NaN or infinite values.")
    return numeric


def _as_finite_matrix(matrix: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(SOURCE_COLUMNS):
        raise ValueError(
            f"Candidate {name!r} must have shape (n, {len(SOURCE_COLUMNS)})."
        )
    if not np.isfinite(values).all():
        raise ValueError(f"Candidate {name!r} contains NaN or infinite values.")
    return values


def _row_minmax(matrix: np.ndarray) -> np.ndarray:
    row_min = matrix.min(axis=1, keepdims=True)
    row_range = matrix.max(axis=1, keepdims=True) - row_min
    safe_range = np.where(row_range == 0.0, 1.0, row_range)
    return (matrix - row_min) / safe_range


def _scale_selected_columns(
    matrix: np.ndarray,
    columns: tuple[str, ...],
    scaler_factory: Callable[[], object] = MinMaxScaler,
) -> np.ndarray:
    transformed = matrix.copy()
    indices = [SOURCE_COLUMNS.index(column) for column in columns]
    scaler = scaler_factory()
    transformed[:, indices] = scaler.fit_transform(transformed[:, indices])
    return transformed


def _primary_candidates(frame: pd.DataFrame) -> list[Candidate]:
    matrix = frame.to_numpy(dtype=np.float64, copy=True)
    semantic_columns = tuple(
        column for column in SOURCE_COLUMNS if column not in BOUNDED_LANDUSE_COLUMNS
    )
    transforms: list[tuple[str, str, Callable[[np.ndarray], np.ndarray]]] = [
        ("raw", "Author-supplied values without scaling.", lambda x: x.copy()),
        (
            "column_minmax",
            "Independent [0, 1] Min-Max scaling for every column.",
            lambda x: MinMaxScaler().fit_transform(x),
        ),
        (
            "standard",
            "Independent zero-mean/unit-variance scaling.",
            lambda x: StandardScaler().fit_transform(x),
        ),
        (
            "maxabs",
            "Independent MaxAbs scaling for every column.",
            lambda x: MaxAbsScaler().fit_transform(x),
        ),
        (
            "robust",
            "Independent median/IQR RobustScaler transformation.",
            lambda x: RobustScaler().fit_transform(x),
        ),
        (
            "row_l2",
            "L2 normalization applied independently to each sample.",
            lambda x: Normalizer(norm="l2").fit_transform(x),
        ),
        (
            "row_minmax",
            "Min-Max normalization applied independently to each sample.",
            _row_minmax,
        ),
        (
            "signed_log1p_minmax",
            "Signed log1p compression followed by column Min-Max scaling.",
            lambda x: MinMaxScaler().fit_transform(np.sign(x) * np.log1p(np.abs(x))),
        ),
        (
            "quantile_uniform",
            "Per-column empirical quantiles mapped to a uniform distribution.",
            lambda x: QuantileTransformer(
                n_quantiles=min(1000, len(x)),
                output_distribution="uniform",
                random_state=0,
            ).fit_transform(x),
        ),
        (
            "yeo_johnson_standard",
            "Per-column Yeo-Johnson transform with standardization.",
            lambda x: PowerTransformer(
                method="yeo-johnson", standardize=True
            ).fit_transform(x),
        ),
        (
            "semantic_selective_minmax",
            "Min-Max continuous fields while retaining buffered land-use ratios.",
            lambda x: _scale_selected_columns(x, semantic_columns),
        ),
        (
            "traffic_no2_selective_minmax",
            "Min-Max traffic and NO2 fields while retaining land-use values.",
            lambda x: _scale_selected_columns(x, TRAFFIC_COLUMNS + POLLUTION_COLUMNS),
        ),
        (
            "primary_road_only_minmax",
            "Min-Max only the primary-road distance as a narrow control.",
            lambda x: _scale_selected_columns(x, ("highway_primary",)),
        ),
    ]
    return [
        Candidate(name, description, _as_finite_matrix(transform(matrix), name))
        for name, description, transform in transforms
    ]


def _weighted_diagnostic_candidates(
    frame: pd.DataFrame,
    weight_values: tuple[float, ...],
) -> list[Candidate]:
    if not weight_values or any(
        value <= 0 or not np.isfinite(value) for value in weight_values
    ):
        raise ValueError("Diagnostic weights must be finite positive numbers.")
    base = MinMaxScaler().fit_transform(frame.to_numpy(dtype=np.float64))
    traffic_indices = [SOURCE_COLUMNS.index(column) for column in TRAFFIC_COLUMNS]
    pollution_indices = [SOURCE_COLUMNS.index(column) for column in POLLUTION_COLUMNS]
    candidates = []
    for pollution_weight in weight_values:
        for traffic_weight in weight_values:
            weighted = base.copy()
            weighted[:, pollution_indices] *= pollution_weight
            weighted[:, traffic_indices] *= traffic_weight
            name = (
                "diagnostic_weighted_minmax_"
                f"pollution_{pollution_weight:g}_traffic_{traffic_weight:g}"
            )
            candidates.append(
                Candidate(
                    name=name,
                    description=(
                        "Post-hoc sensitivity control: Min-Max base with "
                        f"pollution weight {pollution_weight:g}, traffic weight "
                        f"{traffic_weight:g}, and land-use weight 1."
                    ),
                    matrix=weighted,
                    diagnostic_only=True,
                )
            )
    return candidates


def build_candidates(
    frame: pd.DataFrame,
    *,
    include_weight_grid: bool = False,
    weight_values: tuple[float, ...] = (0.25, 0.5, 1.0, 1.5, 2.0, 4.0, 8.0),
) -> list[Candidate]:
    """Build bounded hypotheses without selecting on the paper target."""

    candidates = _primary_candidates(frame)
    if include_weight_grid:
        candidates.extend(_weighted_diagnostic_candidates(frame, weight_values))
    return candidates


def match_table4(
    matrix: np.ndarray,
    paper_rows: np.ndarray = PAPER_TABLE4,
) -> TableMatch:
    """Find a unique nearest source row for each printed Table 4 row."""

    values = _as_finite_matrix(matrix, "table4_match")
    expected = np.asarray(paper_rows, dtype=np.float64)
    if expected.ndim != 2 or expected.shape[1] != len(PAPER_COLUMNS):
        raise ValueError(f"Paper rows must have shape (n, {len(PAPER_COLUMNS)}).")
    if len(values) < len(expected):
        raise ValueError("The source matrix has fewer rows than the paper sample.")
    paper_order = [SOURCE_COLUMNS.index(column) for column in PAPER_COLUMNS]
    ordered = values[:, paper_order]
    differences = expected[:, np.newaxis, :] - ordered[np.newaxis, :, :]
    costs = np.sqrt(np.mean(np.square(differences), axis=2))
    paper_indices, source_indices = linear_sum_assignment(costs)
    order = np.argsort(paper_indices)
    paper_indices = paper_indices[order]
    source_indices = source_indices[order]
    row_rmse = costs[paper_indices, source_indices]
    exact = np.all(
        np.round(expected[paper_indices], 4) == np.round(ordered[source_indices], 4),
        axis=1,
    )
    return TableMatch(
        mean_rmse=float(row_rmse.mean()),
        max_rmse=float(row_rmse.max()),
        exact_rounded_rows=int(np.count_nonzero(exact)),
        source_row_indices=[int(index) for index in source_indices],
        row_rmse=[float(value) for value in row_rmse],
    )


def _fit_kmeans(
    matrix: np.ndarray,
    n_clusters: int,
    seed: int,
    n_init: int,
) -> tuple[KMeans, np.ndarray]:
    model = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=n_init,
        max_iter=1000,
        algorithm="lloyd",
        random_state=seed,
    )
    return model, model.fit_predict(matrix)


def evaluate_candidate(
    candidate: Candidate,
    *,
    seeds: tuple[int, ...],
    n_init: int,
    compute_elbow: bool = True,
) -> CandidateResult:
    """Evaluate K=4 stability, paper-row compatibility, and elbow inertia."""

    if not seeds:
        raise ValueError("At least one K-Means seed is required.")
    if n_init <= 0:
        raise ValueError("n_init must be positive.")
    matrix = _as_finite_matrix(candidate.matrix, candidate.name)
    if len(matrix) <= PAPER_K:
        raise ValueError("The matrix must have more rows than the paper's K=4.")

    silhouettes: dict[str, float] = {}
    inertias: dict[str, float] = {}
    cluster_sizes: dict[str, list[int]] = {}
    for seed in seeds:
        model, labels = _fit_kmeans(matrix, PAPER_K, seed, n_init)
        key = str(seed)
        silhouettes[key] = float(silhouette_score(matrix, labels))
        inertias[key] = float(model.inertia_)
        cluster_sizes[key] = np.bincount(labels, minlength=PAPER_K).astype(int).tolist()

    elbow: dict[str, float] = {}
    if compute_elbow:
        elbow_seed = seeds[0]
        for n_clusters in range(2, min(9, len(matrix) - 1) + 1):
            model, _ = _fit_kmeans(matrix, n_clusters, elbow_seed, n_init)
            elbow[str(n_clusters)] = float(model.inertia_)

    scores = np.asarray(list(silhouettes.values()))
    reference_inertia = inertias[str(seeds[0])]
    scale_factor = np.sqrt(PAPER_K4_DISTORTION / reference_inertia)
    return CandidateResult(
        name=candidate.name,
        description=candidate.description,
        diagnostic_only=candidate.diagnostic_only,
        silhouette_by_seed=silhouettes,
        silhouette_mean=float(scores.mean()),
        silhouette_std=float(scores.std()),
        silhouette_min=float(scores.min()),
        silhouette_max=float(scores.max()),
        paper_silhouette_gap_abs=float(abs(scores.mean() - PAPER_KMEANS_SILHOUETTE)),
        inertia_k4_by_seed=inertias,
        cluster_sizes_by_seed=cluster_sizes,
        table4=match_table4(matrix),
        elbow_inertia=elbow,
        scale_factor_to_paper_k4_distortion=float(scale_factor),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def build_report(
    data_path: Path,
    *,
    seeds: tuple[int, ...] = (42,),
    n_init: int = 100,
    include_weight_grid: bool = False,
    weight_values: tuple[float, ...] = (0.25, 0.5, 1.0, 1.5, 2.0, 4.0, 8.0),
) -> dict[str, object]:
    """Run the complete forensic preprocessing audit."""

    if not seeds:
        raise ValueError("At least one K-Means seed is required.")
    frame = load_airpollution_csv(data_path)
    candidates = build_candidates(
        frame,
        include_weight_grid=include_weight_grid,
        weight_values=weight_values,
    )
    results = [
        evaluate_candidate(
            candidate,
            seeds=seeds,
            n_init=n_init,
            compute_elbow=not candidate.diagnostic_only,
        )
        for candidate in candidates
    ]
    primary = [result for result in results if not result.diagnostic_only]
    closest_silhouette = min(
        primary, key=lambda result: result.paper_silhouette_gap_abs
    )
    closest_table = min(primary, key=lambda result: result.table4.mean_rmse)
    exact_table_candidates = [
        result.name
        for result in primary
        if result.table4.exact_rounded_rows == len(PAPER_TABLE4)
    ]
    joint_candidates = [
        result.name
        for result in primary
        if result.table4.exact_rounded_rows == len(PAPER_TABLE4)
        and result.paper_silhouette_gap_abs <= 0.005
    ]
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "generated_with": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn_version,
            "scipy": scipy_version,
        },
        "source": {
            "path": str(data_path.resolve()),
            "sha256": _sha256(data_path),
            "shape": [int(frame.shape[0]), int(frame.shape[1])],
            "columns": list(frame.columns),
        },
        "paper_reference": {
            "k": PAPER_K,
            "kmeans_silhouette": PAPER_KMEANS_SILHOUETTE,
            "k4_distortion": PAPER_K4_DISTORTION,
            "table4_rows": PAPER_TABLE4.tolist(),
            "table4_columns": list(PAPER_COLUMNS),
        },
        "protocol": {
            "seeds": list(seeds),
            "n_init": n_init,
            "max_iter": 1000,
            "algorithm": "lloyd",
            "elbow_seed": seeds[0],
            "elbow_k_values": list(range(2, 10)),
            "diagnostic_weight_grid_enabled": include_weight_grid,
            "diagnostic_weight_values": list(weight_values),
        },
        "candidates": [asdict(result) for result in results],
        "verdict": {
            "closest_primary_silhouette_candidate": closest_silhouette.name,
            "closest_primary_silhouette": closest_silhouette.silhouette_mean,
            "closest_primary_silhouette_gap_abs": (
                closest_silhouette.paper_silhouette_gap_abs
            ),
            "closest_primary_table4_candidate": closest_table.name,
            "closest_primary_table4_mean_rmse": closest_table.table4.mean_rmse,
            "primary_candidates_recovering_all_table4_rows": exact_table_candidates,
            "primary_candidates_jointly_matching_table4_and_silhouette": (
                joint_candidates
            ),
            "tested_transforms_identify_paper_preprocessing": bool(joint_candidates),
            "interpretation": (
                "A diagnostic weighted candidate near 0.2829 is not evidence of "
                "the paper pipeline; feature weights are non-identifiable from one "
                "reported score. A credible match must also recover Table 4."
            ),
        },
    }


def write_csv_reports(report: dict[str, object], output_dir: Path) -> None:
    """Write flat candidate and elbow tables for independent inspection."""

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_fields = (
        "name",
        "diagnostic_only",
        "silhouette_mean",
        "silhouette_std",
        "silhouette_min",
        "silhouette_max",
        "paper_silhouette_gap_abs",
        "table4_mean_rmse",
        "table4_max_rmse",
        "table4_exact_rounded_rows",
        "scale_factor_to_paper_k4_distortion",
        "description",
    )
    with (output_dir / "candidates.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=candidate_fields)
        writer.writeheader()
        for candidate in report["candidates"]:
            table4 = candidate["table4"]
            writer.writerow(
                {
                    "name": candidate["name"],
                    "diagnostic_only": candidate["diagnostic_only"],
                    "silhouette_mean": candidate["silhouette_mean"],
                    "silhouette_std": candidate["silhouette_std"],
                    "silhouette_min": candidate["silhouette_min"],
                    "silhouette_max": candidate["silhouette_max"],
                    "paper_silhouette_gap_abs": candidate["paper_silhouette_gap_abs"],
                    "table4_mean_rmse": table4["mean_rmse"],
                    "table4_max_rmse": table4["max_rmse"],
                    "table4_exact_rounded_rows": table4["exact_rounded_rows"],
                    "scale_factor_to_paper_k4_distortion": candidate[
                        "scale_factor_to_paper_k4_distortion"
                    ],
                    "description": candidate["description"],
                }
            )

    with (output_dir / "elbow_curves.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("name", "diagnostic_only", "k", "inertia"),
        )
        writer.writeheader()
        for candidate in report["candidates"]:
            for k, inertia in candidate["elbow_inertia"].items():
                writer.writerow(
                    {
                        "name": candidate["name"],
                        "diagnostic_only": candidate["diagnostic_only"],
                        "k": k,
                        "inertia": inertia,
                    }
                )


def write_markdown_report(report: dict[str, object], path: Path) -> None:
    """Write a concise human-readable audit report."""

    source = report["source"]
    protocol = report["protocol"]
    verdict = report["verdict"]
    lines = [
        "# Air Pollution Preprocessing Forensic Audit",
        "",
        "## Verdict",
        "",
        (
            "- Tested transparent preprocessing hypotheses identify the paper "
            f"pipeline: `{verdict['tested_transforms_identify_paper_preprocessing']}`."
        ),
        (
            "- Primary candidates recovering all six Table 4 rows after rounding: "
            f"`{verdict['primary_candidates_recovering_all_table4_rows']}`."
        ),
        (
            "- Primary candidates jointly matching Table 4 and the published "
            "K-Means silhouette: "
            f"`{verdict['primary_candidates_jointly_matching_table4_and_silhouette']}`."
        ),
        (
            "- Closest primary silhouette candidate: "
            f"`{verdict['closest_primary_silhouette_candidate']}` at "
            f"`{verdict['closest_primary_silhouette']:.6f}` "
            f"(absolute gap `{verdict['closest_primary_silhouette_gap_abs']:.6f}`)."
        ),
        (
            "- Closest primary Table 4 candidate: "
            f"`{verdict['closest_primary_table4_candidate']}` with mean RMSE "
            f"`{verdict['closest_primary_table4_mean_rmse']:.6f}`."
        ),
        "",
        "The author-supplied CSV is therefore valid provenance evidence, but the "
        "tested results do not establish that it is the exact 13-feature matrix "
        "used for the paper's reported experiments.",
        "",
        "## Source",
        "",
        f"- Shape: `{source['shape']}`",
        f"- SHA-256: `{source['sha256']}`",
        f"- K-Means seeds: `{protocol['seeds']}`",
        f"- K-Means n_init: `{protocol['n_init']}`",
        "",
        "## Candidate results",
        "",
        "| Candidate | Diagnostic only | Silhouette | Gap to 0.2829 | "
        "Table 4 RMSE | Exact rows |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for candidate in report["candidates"]:
        table4 = candidate["table4"]
        lines.append(
            f"| `{candidate['name']}` | {candidate['diagnostic_only']} | "
            f"{candidate['silhouette_mean']:.6f} | "
            f"{candidate['paper_silhouette_gap_abs']:.6f} | "
            f"{table4['mean_rmse']:.6f} | "
            f"{table4['exact_rounded_rows']}/{len(PAPER_TABLE4)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- Silhouette alone cannot identify preprocessing: many arbitrary "
            "feature-weight choices can move the score near 0.2829.",
            "- Inertia is scale-sensitive. The reported scale factor shows how a "
            "single global multiplier can force K=4 inertia to 149.646 without "
            "changing cluster assignments or silhouette.",
            "- Diagnostic weighted candidates are post-hoc sensitivity checks, "
            "not reproduction claims.",
            "- Failure to recover Table 4 under ordinary transforms is evidence "
            "of a missing feature-engineering/scaling snapshot or a different "
            "source matrix.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(report: dict[str, object], output_dir: Path) -> None:
    """Write JSON, Markdown, and CSV audit artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_markdown_report(report, output_dir / "report.md")
    write_csv_reports(report, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=default_data_path())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output") / "airpollution_forensic",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--n-init", type=int, default=100)
    parser.add_argument("--include-weight-grid", action="store_true")
    parser.add_argument(
        "--weight-values",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 1.0, 1.5, 2.0, 4.0, 8.0],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(
        args.data,
        seeds=tuple(args.seeds),
        n_init=args.n_init,
        include_weight_grid=args.include_weight_grid,
        weight_values=tuple(args.weight_values),
    )
    write_report(report, args.output_dir)
    print(args.output_dir / "report.json")
    print(args.output_dir / "report.md")
    print(args.output_dir / "candidates.csv")
    print(args.output_dir / "elbow_curves.csv")


if __name__ == "__main__":
    main()
