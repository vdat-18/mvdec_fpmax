"""Quality-filter and grid downloaded Sentinel-5P L2 NO2 column products."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from pipeline.airpollution_sentinel import (
    SentinelProduct,
    default_inventory_path,
    default_raw_root,
    load_inventory,
    product_destination,
    verify_product_file,
)

NO2_VARIABLE = "nitrogendioxide_tropospheric_column"
QA_VARIABLE = "qa_value"
LATITUDE_VARIABLE = "latitude"
LONGITUDE_VARIABLE = "longitude"
NO2_UNITS = "mol m-2"
DEFAULT_QA_THRESHOLD = 0.75
DISTANCE_CRS = "EPSG:2169"


@dataclass(frozen=True)
class No2Pixels:
    """Quality-filtered pixel centers and tropospheric NO2 columns."""

    longitude: np.ndarray
    latitude: np.ndarray
    no2_mol_m2: np.ndarray
    input_pixels: int
    retained_pixels: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _finite_array(variable) -> np.ndarray:
    values = variable[:]
    if np.ma.isMaskedArray(values):
        values = values.filled(np.nan)
    return np.asarray(values, dtype=np.float64).squeeze()


def read_no2_pixels(
    path: Path,
    boundary,
    *,
    qa_threshold: float = DEFAULT_QA_THRESHOLD,
) -> No2Pixels:
    """Read official NO2 fields and retain quality pixels inside Luxembourg."""

    from netCDF4 import Dataset
    from shapely import contains_xy

    if not 0.0 <= qa_threshold <= 1.0:
        raise ValueError("NO2 QA threshold must be between 0 and 1.")
    with Dataset(path) as dataset:
        if "PRODUCT" not in dataset.groups:
            raise ValueError(f"Sentinel product lacks the PRODUCT group: {path}")
        product = dataset.groups["PRODUCT"]
        required = {
            LATITUDE_VARIABLE,
            LONGITUDE_VARIABLE,
            NO2_VARIABLE,
            QA_VARIABLE,
        }
        missing = required - product.variables.keys()
        if missing:
            raise ValueError(f"Sentinel PRODUCT group is missing: {sorted(missing)}")
        no2_variable = product.variables[NO2_VARIABLE]
        units = str(getattr(no2_variable, "units", ""))
        if units != NO2_UNITS:
            raise ValueError(
                f"Unexpected {NO2_VARIABLE} units: {units!r}; expected {NO2_UNITS!r}."
            )
        longitude = _finite_array(product.variables[LONGITUDE_VARIABLE])
        latitude = _finite_array(product.variables[LATITUDE_VARIABLE])
        no2 = _finite_array(no2_variable)
        qa = _finite_array(product.variables[QA_VARIABLE])

    if not (longitude.shape == latitude.shape == no2.shape == qa.shape):
        raise ValueError("Sentinel NO2, QA, latitude, and longitude shapes differ.")
    finite = np.isfinite(longitude) & np.isfinite(latitude) & np.isfinite(no2)
    finite &= np.isfinite(qa)
    quality = qa > qa_threshold
    inside = contains_xy(boundary, longitude, latitude)
    retained = finite & quality & inside
    return No2Pixels(
        longitude=longitude[retained],
        latitude=latitude[retained],
        no2_mol_m2=no2[retained],
        input_pixels=int(no2.size),
        retained_pixels=int(np.count_nonzero(retained)),
    )


def _validate_grid(grid: pd.DataFrame) -> None:
    required = {"grid_index", "x_luref", "y_luref"}
    missing = required - set(grid.columns)
    if missing:
        raise ValueError(f"Grid is missing required columns: {sorted(missing)}")
    if grid.empty or grid["grid_index"].duplicated().any():
        raise ValueError("Grid must contain unique, non-empty grid_index values.")
    coordinates = grid[["x_luref", "y_luref"]].to_numpy(dtype=np.float64)
    if not np.isfinite(coordinates).all():
        raise ValueError("Grid coordinates must be finite.")


def assign_pixels_to_grid(pixels: No2Pixels, grid: pd.DataFrame) -> np.ndarray:
    """Assign each retained pixel center to its nearest reconstructed grid cell."""

    from pyproj import Transformer

    _validate_grid(grid)
    if pixels.retained_pixels == 0:
        return np.empty(0, dtype=np.int64)
    transformer = Transformer.from_crs("EPSG:4326", DISTANCE_CRS, always_xy=True)
    x_coords, y_coords = transformer.transform(pixels.longitude, pixels.latitude)
    tree = cKDTree(grid[["x_luref", "y_luref"]].to_numpy(dtype=np.float64))
    _, positions = tree.query(np.column_stack((x_coords, y_coords)), k=1)
    return np.asarray(positions, dtype=np.int64)


def aggregate_no2_products(
    products: Sequence[SentinelProduct],
    raw_root: Path,
    grid: pd.DataFrame,
    boundary,
    *,
    qa_threshold: float = DEFAULT_QA_THRESHOLD,
    allow_partial: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Create monthly maps, then average monthly means with equal month weights."""

    _validate_grid(grid)
    available: list[tuple[SentinelProduct, Path]] = []
    missing: list[SentinelProduct] = []
    for product in products:
        path = product_destination(product, raw_root)
        if path.is_file():
            verify_product_file(path, product)
            available.append((product, path))
        else:
            missing.append(product)
    if missing and not allow_partial:
        raise FileNotFoundError(
            f"Missing {len(missing)} of {len(products)} inventoried Sentinel products. "
            "Download explicit products or pass --allow-partial for diagnostics."
        )
    if not available:
        raise FileNotFoundError("No downloaded Sentinel products are available.")

    cell_count = len(grid)
    monthly_sums: dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros(cell_count, dtype=np.float64)
    )
    monthly_counts: dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros(cell_count, dtype=np.int64)
    )
    input_pixels = 0
    retained_pixels = 0
    for product, path in available:
        pixels = read_no2_pixels(path, boundary, qa_threshold=qa_threshold)
        positions = assign_pixels_to_grid(pixels, grid)
        month = product.start[:7]
        np.add.at(monthly_sums[month], positions, pixels.no2_mol_m2)
        np.add.at(monthly_counts[month], positions, 1)
        input_pixels += pixels.input_pixels
        retained_pixels += pixels.retained_pixels

    monthly_records: list[pd.DataFrame] = []
    monthly_means: list[np.ndarray] = []
    for month in sorted(monthly_sums):
        counts = monthly_counts[month]
        means = np.divide(
            monthly_sums[month],
            counts,
            out=np.full(cell_count, np.nan, dtype=np.float64),
            where=counts > 0,
        )
        monthly_means.append(means)
        monthly_records.append(
            pd.DataFrame(
                {
                    "grid_index": grid["grid_index"].to_numpy(),
                    "month": month,
                    "satellite_no2_column_mol_m2": means,
                    "observation_count": counts,
                }
            )
        )

    monthly_matrix = np.vstack(monthly_means)
    valid_months = np.count_nonzero(np.isfinite(monthly_matrix), axis=0)
    total_observations = np.sum(np.vstack(list(monthly_counts.values())), axis=0)
    monthly_sum = np.nansum(monthly_matrix, axis=0)
    period_mean = np.divide(
        monthly_sum,
        valid_months,
        out=np.full(cell_count, np.nan, dtype=np.float64),
        where=valid_months > 0,
    )
    grid_output = grid.copy()
    grid_output["satellite_no2_column_mol_m2"] = period_mean
    grid_output["satellite_no2_valid_months"] = valid_months
    grid_output["satellite_no2_observations"] = total_observations
    monthly_output = pd.concat(monthly_records, ignore_index=True)
    report = {
        "schema_version": 1,
        "status": "tropospheric_column_not_ground_level_ppb",
        "products_in_inventory": len(products),
        "products_processed": len(available),
        "products_missing": len(missing),
        "missing_product_ids": [product.id for product in missing],
        "qa_filter": f"qa_value > {qa_threshold}",
        "no2_variable": NO2_VARIABLE,
        "units": NO2_UNITS,
        "spatial_assignment": "pixel_center_to_nearest_reconstructed_grid_center",
        "temporal_aggregation": (
            "observation_weighted_monthly_mean_then_equal_weight_mean_of_months"
        ),
        "input_pixels": input_pixels,
        "retained_quality_pixels_inside_boundary": retained_pixels,
        "months_processed": sorted(monthly_sums),
        "cells_without_no2": int(np.count_nonzero(valid_months == 0)),
        "methodological_limit": (
            "No WRF-Chem column-to-ground conversion is applied because the paper "
            "does not disclose a reproducible conversion protocol."
        ),
    }
    return grid_output, monthly_output, report


def write_no2_outputs(
    output_dir: Path,
    grid_output: pd.DataFrame,
    monthly_output: pd.DataFrame,
    report: dict[str, object],
    *,
    inventory_path: Path,
    grid_path: Path,
    boundary_path: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_output.to_csv(output_dir / "no2_grid.csv", index=False)
    monthly_output.to_csv(output_dir / "no2_monthly.csv", index=False)
    report["inputs"] = {
        "inventory": {
            "path": str(inventory_path.resolve()),
            "sha256": _sha256(inventory_path),
        },
        "grid": {"path": str(grid_path.resolve()), "sha256": _sha256(grid_path)},
        "boundary": {
            "path": str(boundary_path.resolve()),
            "sha256": _sha256(boundary_path),
        },
    }
    (output_dir / "no2_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    raw_root = default_raw_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=default_inventory_path())
    parser.add_argument("--raw-root", type=Path, default=raw_root)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument(
        "--boundary",
        type=Path,
        default=raw_root / "boundary" / "communes4326.geojson",
    )
    parser.add_argument("--qa-threshold", type=float, default=DEFAULT_QA_THRESHOLD)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output") / "airpollution_reconstruction" / "no2",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from pipeline.airpollution_reconstruction import load_country_boundary

    _, products = load_inventory(args.inventory)
    grid = pd.read_csv(args.grid)
    boundary = load_country_boundary(args.boundary)
    grid_output, monthly_output, report = aggregate_no2_products(
        products,
        args.raw_root,
        grid,
        boundary,
        qa_threshold=args.qa_threshold,
        allow_partial=args.allow_partial,
    )
    write_no2_outputs(
        args.output_dir,
        grid_output,
        monthly_output,
        report,
        inventory_path=args.inventory,
        grid_path=args.grid,
        boundary_path=args.boundary,
    )
    print(args.output_dir / "no2_grid.csv")
    print(args.output_dir / "no2_monthly.csv")
    print(args.output_dir / "no2_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
