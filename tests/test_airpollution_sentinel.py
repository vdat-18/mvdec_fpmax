from __future__ import annotations

import hashlib
import io
import json
from dataclasses import replace
from datetime import date
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from pipeline import airpollution_sentinel as sentinel

PRODUCT_ID = "562b9de6-a368-4a55-ac44-4694fd0f768f"
PRODUCT_NAME = (
    "S5P_RPRO_L2__NO2____20180601T002121_20180601T020251_"
    "03272_03_020400_20221129T161605.nc"
)


def make_product(payload: bytes = b"sentinel product") -> sentinel.SentinelProduct:
    return sentinel.SentinelProduct(
        id=PRODUCT_ID,
        name=PRODUCT_NAME,
        content_length=len(payload),
        start="2018-06-01T00:42:56.000000Z",
        end="2018-06-01T01:41:19.000000Z",
        s3_path=f"/eodata/Sentinel-5P/TROPOMI/L2__NO2___/2018/06/01/{PRODUCT_NAME}",
        checksums={"MD5": hashlib.md5(payload).hexdigest()},
    )


def product_mapping(payload: bytes = b"sentinel product") -> dict[str, object]:
    product = make_product(payload)
    return {
        "Id": product.id,
        "Name": product.name,
        "ContentLength": product.content_length,
        "ContentDate": {"Start": product.start, "End": product.end},
        "S3Path": product.s3_path,
        "Checksum": [
            {
                "Algorithm": "MD5",
                "Value": product.checksums["MD5"],
            }
        ],
    }


def test_catalogue_filter_records_exact_period_and_luxembourg_bbox() -> None:
    result = sentinel.build_catalogue_filter(
        date(2018, 1, 1),
        date(2021, 7, 1),
    )

    assert "Collection/Name eq 'SENTINEL-5P'" in result
    assert "contains(Name,'L2__NO2___')" in result
    assert "POLYGON((5.73 49.44,6.54 49.44" in result
    assert "ContentDate/Start ge 2018-01-01T00:00:00.000Z" in result
    assert "ContentDate/Start lt 2021-07-01T00:00:00.000Z" in result


def test_catalogue_filter_rejects_invalid_period_and_bbox() -> None:
    with pytest.raises(ValueError, match="start must be earlier"):
        sentinel.build_catalogue_filter(date(2021, 1, 1), date(2021, 1, 1))
    with pytest.raises(ValueError, match="Invalid WGS84"):
        sentinel.build_catalogue_filter(
            date(2021, 1, 1),
            date(2021, 1, 2),
            (7.0, 49.0, 6.0, 50.0),
        )


def test_product_parser_requires_safe_no2_netcdf_and_md5() -> None:
    product = sentinel.SentinelProduct.from_mapping(product_mapping())

    assert product.id == PRODUCT_ID
    assert product.checksums["MD5"] == hashlib.md5(b"sentinel product").hexdigest()

    with pytest.raises(ValueError, match="Unsafe or unexpected"):
        sentinel.validate_product(replace(product, name="../unsafe.nc"))
    with pytest.raises(ValueError, match="valid catalogue MD5"):
        sentinel.validate_product(replace(product, checksums={}))


def test_query_catalogue_paginates_deterministically(monkeypatch) -> None:
    first = product_mapping(b"first")
    second_product = replace(
        make_product(b"second"),
        id="fb607047-5af9-489d-b9f3-c35d7e862204",
        name=PRODUCT_NAME.replace("03272", "03278"),
    )
    second = {
        "Id": second_product.id,
        "Name": second_product.name,
        "ContentLength": second_product.content_length,
        "ContentDate": {
            "Start": second_product.start,
            "End": second_product.end,
        },
        "S3Path": second_product.s3_path,
        "Checksum": [{"Algorithm": "MD5", "Value": second_product.checksums["MD5"]}],
    }
    pages = [[first], [second], []]
    requests = []

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        return io.BytesIO(json.dumps({"value": pages[len(requests) - 1]}).encode())

    monkeypatch.setattr(sentinel, "urlopen", fake_urlopen)

    _, products = sentinel.query_catalogue(
        date(2018, 6, 1),
        date(2018, 6, 2),
        page_size=1,
    )

    assert [product.id for product in products] == [first["Id"], second["Id"]]
    skips = [
        parse_qs(urlparse(request.full_url).query)["$skip"][0]
        for request, _ in requests
    ]
    assert skips == ["0", "1", "2"]
    assert all(request.get_method() == "GET" for request, _ in requests)


def test_build_inventory_is_json_safe(monkeypatch) -> None:
    monkeypatch.setattr(
        sentinel,
        "query_catalogue",
        lambda *_args, **_kwargs: ("filter", [make_product()]),
    )

    inventory = sentinel.build_inventory()

    assert inventory["product_count"] == 1
    assert inventory["total_content_bytes"] == len(b"sentinel product")
    json.dumps(inventory, allow_nan=False)


def test_valid_product_cache_does_not_require_token_or_network(
    tmp_path: Path, monkeypatch
) -> None:
    payload = b"valid sentinel cache"
    product = make_product(payload)
    destination = sentinel.product_destination(product, tmp_path)
    destination.parent.mkdir(parents=True)
    destination.write_bytes(payload)
    monkeypatch.delenv(sentinel.ACCESS_TOKEN_ENV, raising=False)

    def fail_urlopen(*_args, **_kwargs):
        raise AssertionError("A valid cache must not access the network.")

    monkeypatch.setattr(sentinel, "urlopen", fail_urlopen)

    assert sentinel.fetch_product(product, tmp_path) == destination


def test_tampered_product_cache_fails_without_network(
    tmp_path: Path, monkeypatch
) -> None:
    product = make_product(b"expected")
    destination = sentinel.product_destination(product, tmp_path)
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"tampered")

    def fail_urlopen(*_args, **_kwargs):
        raise AssertionError("A tampered cache must fail before network access.")

    monkeypatch.setattr(sentinel, "urlopen", fail_urlopen)

    with pytest.raises(ValueError, match="Size mismatch|Checksum mismatch"):
        sentinel.fetch_product(product, tmp_path, access_token="unused")


def test_download_requires_access_token(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv(sentinel.ACCESS_TOKEN_ENV, raising=False)

    with pytest.raises(ValueError, match=sentinel.ACCESS_TOKEN_ENV):
        sentinel.fetch_product(make_product(), tmp_path, source="cdse")


def test_public_s3_download_needs_no_token(tmp_path: Path, monkeypatch) -> None:
    payload = b"public sentinel object"
    product = make_product(payload)
    captured = []

    def fake_urlopen(request, timeout):
        captured.append((request, timeout))
        return io.BytesIO(payload)

    monkeypatch.delenv(sentinel.ACCESS_TOKEN_ENV, raising=False)
    monkeypatch.setattr(sentinel, "urlopen", fake_urlopen)

    path = sentinel.fetch_product(product, tmp_path, source="public-s3")

    assert path.read_bytes() == payload
    assert captured[0][0].full_url == sentinel.public_s3_url(product)
    assert captured[0][0].get_header("Authorization") is None
    assert "/RPRO/L2__NO2___/2018/06/01/" in captured[0][0].full_url


def test_failed_forced_download_preserves_existing_product(
    tmp_path: Path, monkeypatch
) -> None:
    payload = b"original sentinel"
    product = make_product(payload)
    destination = sentinel.product_destination(product, tmp_path)
    destination.parent.mkdir(parents=True)
    destination.write_bytes(payload)
    replacement = b"x" * len(payload)
    captured = []

    def fake_urlopen(request, timeout):
        captured.append((request, timeout))
        return io.BytesIO(replacement)

    monkeypatch.setattr(sentinel, "urlopen", fake_urlopen)

    with pytest.raises(ValueError, match="Checksum mismatch"):
        sentinel.fetch_product(
            product,
            tmp_path,
            force=True,
            access_token="secret-token",
            source="cdse",
        )

    assert destination.read_bytes() == payload
    assert captured[0][0].get_header("Authorization") == "Bearer secret-token"
    assert captured[0][0].get_method() == "GET"
    assert list(destination.parent.glob("*.tmp")) == []


def test_select_products_accepts_uuid_or_exact_name_and_deduplicates() -> None:
    product = make_product()

    selected = sentinel.select_products(
        [product],
        [product.id, product.name],
    )

    assert selected == [product]
    with pytest.raises(ValueError, match="Unknown Sentinel products"):
        sentinel.select_products([product], ["missing"])


def test_cli_download_requires_explicit_product_ids() -> None:
    with pytest.raises(SystemExit) as error:
        sentinel.main(["download"])

    assert error.value.code == 2
