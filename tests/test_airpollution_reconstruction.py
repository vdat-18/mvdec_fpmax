"""Tests for clean-room Air Pollution data reconstruction."""

import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
import pytest

from pipeline.airpollution_reconstruction import (
    TRAFFIC_HIGH_THRESHOLD,
    TRAFFIC_LOW_THRESHOLD,
    _compare_candidate,
    _grid_disk_neighbor_mean,
    _summarize_landuse_matches,
    add_traffic_distance_features,
    build_regular_grid,
    classify_traffic,
    load_geofabrik_landuse_geometries,
    minmax,
    quantile_rmse,
)


def test_traffic_thresholds_follow_paper_boundaries() -> None:
    assert classify_traffic(TRAFFIC_LOW_THRESHOLD - 1) == "low"
    assert classify_traffic(TRAFFIC_LOW_THRESHOLD) == "medium"
    assert classify_traffic(TRAFFIC_HIGH_THRESHOLD - 1) == "medium"
    assert classify_traffic(TRAFFIC_HIGH_THRESHOLD) == "high"


def test_minmax_handles_regular_and_constant_vectors() -> None:
    assert np.allclose(minmax(np.array([2.0, 4.0, 6.0])), [0.0, 0.5, 1.0])
    assert np.array_equal(minmax(np.array([3.0, 3.0])), [0.0, 0.0])


def test_regular_grid_uses_bbox_aligned_cell_centers() -> None:
    pytest.importorskip("geopandas")
    from shapely.geometry import box

    grid, metadata = build_regular_grid(box(6.0, 49.5, 6.03, 49.52), 0.01)

    assert metadata.longitude_bins == 3
    assert metadata.latitude_bins == 2
    assert metadata.bounding_box_cells == 6
    assert metadata.retained_cells == 6
    assert len(grid) == 6
    assert np.isfinite(grid[["x_luref", "y_luref"]].to_numpy()).all()


def test_load_geofabrik_landuse_geometries_reads_relation_complete_archive(
    tmp_path: Path,
) -> None:
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    layer_dir = tmp_path / "layer"
    layer_dir.mkdir()
    layer_path = layer_dir / "gis_osm_landuse_a_free_1.shp"
    frame = gpd.GeoDataFrame(
        {
            "fclass": ["industrial", "residential", "forest", "meadow"],
            "geometry": [
                box(6.00, 49.50, 6.01, 49.51),
                box(6.01, 49.50, 6.02, 49.51),
                box(6.02, 49.50, 6.03, 49.51),
                box(6.03, 49.50, 6.04, 49.51),
            ],
        },
        crs="EPSG:4326",
    )
    frame.to_file(layer_path)
    archive_path = tmp_path / "geofabrik.zip"
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        for component in layer_dir.iterdir():
            archive.write(component, arcname=component.name)

    geometries = load_geofabrik_landuse_geometries(
        archive_path,
        box(5.99, 49.49, 6.05, 49.52),
    )

    assert set(geometries) == {"industrial", "residential", "forest"}
    assert all(not geometry.is_empty for geometry in geometries.values())


def test_nearest_traffic_features_are_finite_and_scaled() -> None:
    grid = pd.DataFrame(
        {
            "grid_index": [0, 1, 2],
            "longitude": [0.0, 0.0, 0.0],
            "latitude": [0.0, 0.0, 0.0],
            "x_luref": [0.0, 5.0, 10.0],
            "y_luref": [0.0, 0.0, 0.0],
        }
    )
    stations = pd.DataFrame(
        {
            "x_luref": [0.0, 10.0, 5.0],
            "y_luref": [0.0, 0.0, 5.0],
            "all_rows_category": ["high", "medium", "low"],
        }
    )

    reconstructed, counts = add_traffic_distance_features(
        grid,
        stations,
        "all_rows",
    )

    assert counts == {"high": 1, "medium": 1, "low": 1}
    for column in ("high_traffic", "medium_traffic", "low_traffic"):
        assert reconstructed[column].between(0.0, 1.0).all()
        assert np.isfinite(reconstructed[column]).all()


def test_grid_disk_kernel_has_21_cell_interior_support() -> None:
    rows, columns = np.indices((5, 5))
    grid = pd.DataFrame(
        {
            "grid_row": rows.ravel(),
            "grid_column": columns.ravel(),
        }
    )
    values = np.zeros(25, dtype=np.float64)
    values[12] = 1.0

    means, supports = _grid_disk_neighbor_mean(grid, values, 2.5)

    assert supports[12] == 21
    assert means[12] == pytest.approx(1.0 / 21.0)
    assert supports[0] == 8


def test_grid_disk_kernel_rejects_non_positive_radius() -> None:
    grid = pd.DataFrame({"grid_row": [0], "grid_column": [0]})

    with pytest.raises(ValueError, match="radius must be positive"):
        _grid_disk_neighbor_mean(grid, np.array([1.0]), 0.0)


def test_quantile_rmse_is_zero_for_identical_distributions() -> None:
    values = np.array([3.0, 1.0, 2.0])

    assert quantile_rmse(values, values[::-1]) == pytest.approx(0.0)


def _author_comparison_frame() -> pd.DataFrame:
    values = np.array([0.0, 0.25, 0.75, 1.0])
    return pd.DataFrame(
        {
            "high_traffic": values,
            "medium_traffic": values[::-1],
            "low_traffic": [0.0, 0.0, 1.0, 1.0],
            "highway_primary": values,
            "landuse_industrial_b5": values,
            "landuse_residential_b5": values[::-1],
            "landuse_forest_b5": [0.0, 1.0, 0.0, 1.0],
            "landuse_industrial_distance_mean": values,
            "landuse_industrial_variance": [0.0, 0.1875, 0.1875, 0.0],
        }
    )


def test_compare_candidate_supports_traffic_only_frame() -> None:
    author = _author_comparison_frame()
    candidate = author[["high_traffic", "medium_traffic", "low_traffic"]].copy()

    traffic, road, landuse = _compare_candidate(candidate, author)

    assert set(traffic) == {"high_traffic", "medium_traffic", "low_traffic"}
    assert all(value == pytest.approx(0.0) for value in traffic.values())
    assert road == {}
    assert landuse == {}


def test_compare_candidate_supports_road_only_frame() -> None:
    author = _author_comparison_frame()
    candidate = pd.DataFrame(
        {
            "highway_primary_osm_motorway": author["highway_primary"],
            "highway_primary_osm_primary": author["highway_primary"][::-1].to_numpy(),
        }
    )

    traffic, road, landuse = _compare_candidate(candidate, author)

    assert traffic == {}
    assert set(road) == {"osm_motorway", "osm_primary"}
    assert all(value == pytest.approx(0.0) for value in road.values())
    assert landuse == {}


def test_compare_candidate_supports_landuse_only_frame() -> None:
    author = _author_comparison_frame()
    candidate = pd.DataFrame(
        {
            "landuse_industrial_radius5000_mean": author["landuse_industrial_b5"],
            "landuse_residential_queen8_mean": author["landuse_residential_b5"],
            "landuse_industrial_radius2500_variance": author[
                "landuse_industrial_variance"
            ],
        }
    )

    traffic, road, landuse = _compare_candidate(candidate, author)

    assert traffic == {}
    assert road == {}
    assert set(landuse) == {
        "industrial_radius5000_mean_to_b5",
        "residential_queen8_mean_to_b5",
        "industrial_radius2500_variance",
        "industrial_radius5000_mean",
    }
    assert all(value == pytest.approx(0.0) for value in landuse.values())


def test_compare_candidate_supports_multiple_feature_families() -> None:
    author = _author_comparison_frame()
    candidate = author[["high_traffic", "medium_traffic", "low_traffic"]].copy()
    candidate["highway_primary_osm_motorway"] = author["highway_primary"]
    candidate["landuse_forest_radius2500_mean"] = author["landuse_forest_b5"]

    traffic, road, landuse = _compare_candidate(candidate, author)

    assert len(traffic) == 3
    assert set(road) == {"osm_motorway"}
    assert set(landuse) == {"forest_radius2500_mean_to_b5"}


def test_compare_candidate_rejects_missing_author_target() -> None:
    author = _author_comparison_frame().drop(columns="highway_primary")
    candidate = pd.DataFrame({"highway_primary_osm_motorway": [0.0, 0.2, 0.8, 1.0]})

    with pytest.raises(
        ValueError, match="missing required road columns: highway_primary"
    ):
        _compare_candidate(candidate, author)


def test_compare_candidate_without_known_family_is_json_safe() -> None:
    result = _compare_candidate(
        pd.DataFrame({"grid_index": [0, 1]}),
        _author_comparison_frame(),
    )

    assert result == ({}, {}, {})
    json.dumps(result, allow_nan=False)


def test_landuse_summary_deduplicates_traffic_modes_and_selects_best() -> None:
    diagnostic = {
        "resolution_degrees": 0.01,
        "landuse_source": "overpass_ways",
        "landuse_quantile_rmse": {
            "industrial_disk2p5_mean_to_b5": 0.02,
            "industrial_radius5000_mean_to_b5": 0.01,
            "industrial_disk2p5_mean": 0.03,
            "industrial_disk2p5_variance": 0.004,
        },
    }

    summaries = _summarize_landuse_matches([diagnostic, diagnostic.copy()])

    assert len(summaries) == 3
    by_target = {item["target"]: item for item in summaries}
    assert by_target["industrial_b5"]["candidate"] == (
        "industrial_radius5000_mean_to_b5"
    )
    assert by_target["industrial_variance"]["quantile_rmse"] == pytest.approx(0.004)


def test_compare_candidate_rejects_partial_traffic_family() -> None:
    author = _author_comparison_frame()
    candidate = author[["high_traffic", "medium_traffic"]].copy()

    with pytest.raises(
        ValueError,
        match="complete feature family; missing: low_traffic",
    ):
        _compare_candidate(candidate, author)
