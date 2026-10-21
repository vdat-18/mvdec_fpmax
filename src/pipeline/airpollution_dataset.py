"""Build and audit a transparent candidate for the paper's 13-feature dataset."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

PAPER_ROWS = 3_765
PAPER_FEATURES = (
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


@dataclass(frozen=True)
class FeatureContract:
    """Paper evidence and clean-room implementation status for one feature."""

    name: str
    paper_definition: str
    cleanroom_source: str
    candidate_rule: str
    status: str
    blocker: str | None = None


def feature_contracts() -> tuple[FeatureContract, ...]:
    """Return the single authoritative mapping for the paper's 13 features."""

    traffic_blocker = (
        "The paper does not specify how overlapping workbook rows are aggregated."
    )
    landuse_blocker = (
        "The paper does not define rasterization, convolution mask, or edge handling."
    )
    return (
        FeatureContract(
            "highway_primary",
            "Distance to nearest primary road.",
            "Historical OpenStreetMap roads.",
            "Metric distance from each cell center to OSM highway=primary.",
            "candidate",
            "The paper does not disclose its OSM snapshot or tag rule.",
        ),
        FeatureContract(
            "high_traffic",
            "Distance to nearest high-traffic point.",
            "Luxembourg 2018 traffic workbook.",
            "Monthly average, >=600000 class, nearest-point distance.",
            "candidate",
            traffic_blocker,
        ),
        FeatureContract(
            "medium_traffic",
            "Distance to nearest moderate-traffic point.",
            "Luxembourg 2018 traffic workbook.",
            "Monthly average, [120000,600000) class, nearest distance.",
            "candidate",
            traffic_blocker,
        ),
        FeatureContract(
            "low_traffic",
            "Distance to nearest low-traffic point.",
            "Luxembourg 2018 traffic workbook.",
            "Monthly average, <120000 class, nearest-point distance.",
            "candidate",
            traffic_blocker,
        ),
        FeatureContract(
            "landuse_industrial_variance",
            "Neighbor industrial-zone variance within 5 km.",
            "Historical OpenStreetMap land-use polygons.",
            "Variance of a binary industrial raster within 5 km.",
            "candidate",
            landuse_blocker,
        ),
        FeatureContract(
            "landuse_industrial_distance_mean",
            "Neighbor industrial-zone distance to mean within 5 km.",
            "Historical OpenStreetMap land-use polygons.",
            "Center binary value minus its 5 km neighborhood mean.",
            "hypothesis",
            "The paper does not define 'distance to mean'.",
        ),
        FeatureContract(
            "landuse_industrial_b5",
            "Neighbor industrial zones buffered by 5 km.",
            "Historical OpenStreetMap land-use polygons.",
            "Mean binary industrial coverage within 5 km.",
            "candidate",
            landuse_blocker,
        ),
        FeatureContract(
            "landuse_residential_b5",
            "Neighbor residential zones buffered by 5 km.",
            "Historical OpenStreetMap land-use polygons.",
            "Mean binary residential coverage within 5 km.",
            "candidate",
            landuse_blocker,
        ),
        FeatureContract(
            "landuse_forest_b5",
            "Neighbor forest zones buffered by 5 km.",
            "Historical OpenStreetMap land-use polygons.",
            "Mean binary forest coverage within 5 km.",
            "candidate",
            landuse_blocker,
        ),
        FeatureContract(
            "neighbours_no2_variance",
            "Neighbor NO2 variance within 5 km.",
            "Paper-compatible ground-level NO2 raster.",
            "Population variance of valid neighboring cells within 5 km.",
            "blocked",
            "The WRF-Chem conversion is not reproducibly specified.",
        ),
        FeatureContract(
            "neighbours_no2_distance_mean",
            "Neighbor NO2 distance to mean within 5 km.",
            "Paper-compatible ground-level NO2 raster.",
            "Center NO2 minus valid-neighbor mean within 5 km.",
            "blocked",
            "The conversion and meaning/sign of distance are undisclosed.",
        ),
        FeatureContract(
            "neighbours_no2_distance_max",
            "Neighbor NO2 distance to maximum within 5 km.",
            "Paper-compatible ground-level NO2 raster.",
            "Center NO2 minus valid-neighbor maximum within 5 km.",
            "blocked",
            "The conversion and meaning/sign of distance are undisclosed.",
        ),
        FeatureContract(
            "satellite_no2",
            "Ground-level NO2 derived from monthly Sentinel-5P via WRF-Chem.",
            "Sentinel-5P column NO2 plus an authoritative conversion method.",
            "Monthly mean from 2018 through June 2021, then WRF-Chem conversion.",
            "blocked",
            "No executable WRF-Chem configuration or conversion is given.",
        ),
    )


def add_no2_neighbor_candidates(
    grid: pd.DataFrame, *, no2_column: str, radius_meters: float = 5_000.0
) -> pd.DataFrame:
    """Add explicit 5 km NO2 hypotheses without claiming the paper's exact rule."""

    required = {"x_luref", "y_luref", no2_column}
    missing = required - set(grid.columns)
    if missing:
        raise ValueError(f"Grid is missing required columns: {sorted(missing)}")
    if radius_meters <= 0 or not np.isfinite(radius_meters):
        raise ValueError("NO2 neighborhood radius must be finite and positive.")
    coordinates = grid[["x_luref", "y_luref"]].to_numpy(dtype=np.float64)
    values = grid[no2_column].to_numpy(dtype=np.float64)
    if not np.isfinite(coordinates).all():
        raise ValueError("Grid coordinates must be finite.")

    neighborhoods = cKDTree(coordinates).query_ball_point(coordinates, radius_meters)
    variance = np.full(len(grid), np.nan)
    distance_mean = np.full(len(grid), np.nan)
    distance_max = np.full(len(grid), np.nan)
    counts = np.zeros(len(grid), dtype=np.int64)
    for position, indices in enumerate(neighborhoods):
        neighbor_values = values[[index for index in indices if index != position]]
        neighbor_values = neighbor_values[np.isfinite(neighbor_values)]
        counts[position] = len(neighbor_values)
        if not np.isfinite(values[position]) or len(neighbor_values) == 0:
            continue
        variance[position] = np.var(neighbor_values)
        distance_mean[position] = values[position] - np.mean(neighbor_values)
        distance_max[position] = values[position] - np.max(neighbor_values)

    output = grid.copy()
    output["neighbours_no2_variance"] = variance
    output["neighbours_no2_distance_mean"] = distance_mean
    output["neighbours_no2_distance_max"] = distance_max
    output["neighbours_no2_count"] = counts
    return output


def audit_reproduction_readiness(
    *, grid_path: Path | None = None, no2_report_path: Path | None = None
) -> dict[str, object]:
    """Report what is reproduced, hypothesized, and still blocked."""

    contracts = feature_contracts()
    report: dict[str, object] = {
        "schema_version": 1,
        "target": {
            "paper": "2025_MvDEC air-pollution case study",
            "rows": PAPER_ROWS,
            "features": list(PAPER_FEATURES),
        },
        "claim": "cleanroom_candidate_not_exact_reproduction",
        "paper_ambiguities": [
            (
                "Table 3 lists 16 feature rows, while Section 5.3 and Table 4 "
                "describe a 13-feature sample matrix."
            ),
            (
                "Table 3 lists raw industrial, residential, and forest-zone "
                "features that are absent as separate columns in Table 4."
            ),
            (
                "The prose describes neighborhood average, maximum, and variance, "
                "while Table 4 labels the NO2 columns as variance, distance to mean, "
                "and distance to max."
            ),
            (
                "The paper states that grid resolution follows the lower Sentinel-5P "
                "resolution of 5.5 km x 3.5 km, but does not disclose grid origin, "
                "mask predicate, or the exact 3,765 retained cells."
            ),
        ],
        "feature_contracts": [asdict(contract) for contract in contracts],
        "blocked_features": [c.name for c in contracts if c.status == "blocked"],
        "exact_reproduction_ready": False,
    }
    if grid_path is not None:
        grid = pd.read_csv(grid_path)
        report["grid"] = {
            "path": str(grid_path.resolve()),
            "rows": len(grid),
            "paper_row_gap": len(grid) - PAPER_ROWS,
            "has_unique_grid_index": bool(
                "grid_index" in grid and not grid["grid_index"].duplicated().any()
            ),
        }
    if no2_report_path is not None:
        no2 = json.loads(no2_report_path.read_text(encoding="utf-8"))
        report["no2"] = {
            "path": str(no2_report_path.resolve()),
            "status": no2.get("status"),
            "products_processed": no2.get("products_processed"),
            "products_missing": no2.get("products_missing"),
            "cells_without_no2": no2.get("cells_without_no2"),
        }
    return report


def write_readiness_report(report: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "paper_dataset_contract.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# 2025 MvDEC Air-Pollution Dataset Reproduction Contract",
        "",
        "This separates paper evidence from clean-room hypotheses and does not claim",
        "that a candidate matrix is the authors' original dataset.",
        "",
        f"- Target shape: `{PAPER_ROWS} x {len(PAPER_FEATURES)}`",
        f"- Exact reproduction ready: `{report['exact_reproduction_ready']}`",
        f"- Claim: `{report['claim']}`",
        "",
        "## Paper Ambiguities",
        "",
    ]
    lines.extend(f"- {item}" for item in report["paper_ambiguities"])
    lines.extend(
        [
            "",
            "| Feature | Status | Candidate rule | Blocker |",
            "|---|---|---|---|",
        ]
    )
    for item in report["feature_contracts"]:
        lines.append(
            f"| `{item['name']}` | `{item['status']}` | "
            f"{item['candidate_rule']} | {item['blocker'] or '-'} |"
        )
    if "grid" in report:
        grid = report["grid"]
        lines += [
            "",
            "## Current Grid",
            "",
            f"- Rows: `{grid['rows']}`",
            f"- Gap to paper: `{grid['paper_row_gap']:+d}`",
        ]
    if "no2" in report:
        no2 = report["no2"]
        lines += [
            "",
            "## Current NO2 Stage",
            "",
            f"- Status: `{no2['status']}`",
            f"- Products processed: `{no2['products_processed']}`",
            f"- Products missing: `{no2['products_missing']}`",
            f"- Cells without NO2: `{no2['cells_without_no2']}`",
        ]
    (output_dir / "paper_dataset_contract.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path)
    parser.add_argument("--no2-report", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output") / "airpollution_reconstruction" / "paper_contract",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit_reproduction_readiness(
        grid_path=args.grid, no2_report_path=args.no2_report
    )
    write_readiness_report(report, args.output_dir)
    print(args.output_dir / "paper_dataset_contract.json")
    print(args.output_dir / "paper_dataset_contract.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
