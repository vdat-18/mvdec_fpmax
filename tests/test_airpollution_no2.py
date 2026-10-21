from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline import airpollution_no2 as no2
from pipeline.airpollution_sentinel import SentinelProduct, product_destination


def write_synthetic_product(
    path: Path,
    values: np.ndarray,
    qa: np.ndarray,
    longitude: np.ndarray,
    latitude: np.ndarray,
) -> None:
    from netCDF4 import Dataset

    path.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(path, "w") as dataset:
        product = dataset.createGroup("PRODUCT")
        product.createDimension("time", 1)
        product.createDimension("scanline", values.shape[0])
        product.createDimension("ground_pixel", values.shape[1])
        dimensions = ("time", "scanline", "ground_pixel")
        variables = {
            no2.NO2_VARIABLE: (values, no2.NO2_UNITS),
            no2.QA_VARIABLE: (qa, "1"),
            no2.LONGITUDE_VARIABLE: (longitude, "degrees_east"),
            no2.LATITUDE_VARIABLE: (latitude, "degrees_north"),
        }
        for name, (data, units) in variables.items():
            variable = product.createVariable(name, "f8", dimensions)
            variable.units = units
            variable[0, :, :] = data


def make_product_for_file(
    path: Path,
    *,
    start: str,
    product_id: str | None = None,
) -> SentinelProduct:
    return SentinelProduct(
        id=product_id or str(uuid.uuid4()),
        name=path.name,
        content_length=path.stat().st_size,
        start=start,
        end=start,
        s3_path=f"/synthetic/{path.name}",
        checksums={"MD5": hashlib.md5(path.read_bytes()).hexdigest()},
    )


def synthetic_name(day: str, orbit: str) -> str:
    compact = day.replace("-", "")
    return (
        f"S5P_RPRO_L2__NO2____{compact}T100000_{compact}T110000_"
        f"{orbit}_03_020400_20221108T000000.nc"
    )


def test_read_no2_pixels_applies_strict_qa_and_boundary_filter(tmp_path: Path) -> None:
    from shapely.geometry import box

    path = tmp_path / synthetic_name("2018-06-01", "03272")
    write_synthetic_product(
        path,
        values=np.array([[1.0, 2.0], [3.0, 4.0]]),
        qa=np.array([[0.8, 0.75], [0.9, 0.95]]),
        longitude=np.array([[6.00, 6.01], [6.02, 7.00]]),
        latitude=np.array([[49.50, 49.51], [49.52, 50.50]]),
    )

    pixels = no2.read_no2_pixels(
        path,
        box(5.9, 49.4, 6.1, 49.6),
        qa_threshold=0.75,
    )

    assert pixels.input_pixels == 4
    assert pixels.retained_pixels == 2
    assert np.array_equal(pixels.no2_mol_m2, [1.0, 3.0])


def test_read_no2_pixels_rejects_unexpected_units(tmp_path: Path) -> None:
    from netCDF4 import Dataset
    from shapely.geometry import box

    path = tmp_path / synthetic_name("2018-06-01", "03272")
    values = np.array([[1.0]])
    write_synthetic_product(
        path,
        values=values,
        qa=np.array([[0.9]]),
        longitude=np.array([[6.0]]),
        latitude=np.array([[49.5]]),
    )
    with Dataset(path, "a") as dataset:
        dataset.groups["PRODUCT"].variables[no2.NO2_VARIABLE].units = "ppb"

    with pytest.raises(ValueError, match="Unexpected.*units"):
        no2.read_no2_pixels(path, box(5.9, 49.4, 6.1, 49.6))


def test_monthly_maps_receive_equal_weight_in_period_mean(tmp_path: Path) -> None:
    from pyproj import Transformer
    from shapely.geometry import box

    raw_root = tmp_path / "raw"
    transformer = Transformer.from_crs("EPSG:4326", no2.DISTANCE_CRS, always_xy=True)
    x_coord, y_coord = transformer.transform(6.0, 49.5)
    grid = pd.DataFrame(
        {
            "grid_index": [0],
            "longitude": [6.0],
            "latitude": [49.5],
            "x_luref": [x_coord],
            "y_luref": [y_coord],
        }
    )

    definitions = [
        ("2018-06-01", "03272", np.array([[1.0, 1.0]])),
        ("2018-07-01", "03700", np.array([[3.0]])),
    ]
    products = []
    for day, orbit, values in definitions:
        provisional = SentinelProduct(
            id=str(uuid.uuid4()),
            name=synthetic_name(day, orbit),
            content_length=1,
            start=f"{day}T10:00:00.000000Z",
            end=f"{day}T11:00:00.000000Z",
            s3_path="/synthetic",
            checksums={"MD5": "0" * 32},
        )
        path = product_destination(provisional, raw_root)
        write_synthetic_product(
            path,
            values=values,
            qa=np.full(values.shape, 0.9),
            longitude=np.full(values.shape, 6.0),
            latitude=np.full(values.shape, 49.5),
        )
        products.append(
            make_product_for_file(
                path,
                start=provisional.start,
                product_id=provisional.id,
            )
        )

    grid_output, monthly_output, report = no2.aggregate_no2_products(
        products,
        raw_root,
        grid,
        box(5.9, 49.4, 6.1, 49.6),
    )

    assert grid_output.loc[0, "satellite_no2_column_mol_m2"] == pytest.approx(2.0)
    assert grid_output.loc[0, "satellite_no2_valid_months"] == 2
    assert grid_output.loc[0, "satellite_no2_observations"] == 3
    assert monthly_output["satellite_no2_column_mol_m2"].tolist() == [1.0, 3.0]
    assert report["status"] == "tropospheric_column_not_ground_level_ppb"


def test_aggregate_rejects_missing_products_unless_partial(tmp_path: Path) -> None:
    from shapely.geometry import box

    product = SentinelProduct(
        id=str(uuid.uuid4()),
        name=synthetic_name("2018-06-01", "03272"),
        content_length=1,
        start="2018-06-01T10:00:00.000000Z",
        end="2018-06-01T11:00:00.000000Z",
        s3_path="/synthetic",
        checksums={"MD5": "0" * 32},
    )
    grid = pd.DataFrame(
        {
            "grid_index": [0],
            "x_luref": [0.0],
            "y_luref": [0.0],
        }
    )

    with pytest.raises(FileNotFoundError, match="Missing 1 of 1"):
        no2.aggregate_no2_products(
            [product],
            tmp_path,
            grid,
            box(5.9, 49.4, 6.1, 49.6),
        )
    with pytest.raises(FileNotFoundError, match="No downloaded"):
        no2.aggregate_no2_products(
            [product],
            tmp_path,
            grid,
            box(5.9, 49.4, 6.1, 49.6),
            allow_partial=True,
        )


def test_grid_validation_rejects_duplicate_indices() -> None:
    grid = pd.DataFrame(
        {
            "grid_index": [1, 1],
            "x_luref": [0.0, 1.0],
            "y_luref": [0.0, 1.0],
        }
    )

    with pytest.raises(ValueError, match="unique"):
        no2.assign_pixels_to_grid(
            no2.No2Pixels(
                longitude=np.array([6.0]),
                latitude=np.array([49.5]),
                no2_mol_m2=np.array([1.0]),
                input_pixels=1,
                retained_pixels=1,
            ),
            grid,
        )
