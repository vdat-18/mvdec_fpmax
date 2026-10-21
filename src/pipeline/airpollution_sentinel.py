"""Inventory and fetch official Sentinel-5P L2 NO2 products for Luxembourg."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

CATALOGUE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
DOWNLOAD_URL = "https://download.dataspace.copernicus.eu/odata/v1/Products"
PUBLIC_S3_URL = "https://meeo-s5p.s3.eu-central-1.amazonaws.com"
COLLECTION_NAME = "SENTINEL-5P"
PRODUCT_NAME_TOKEN = "L2__NO2___"
TARGET_START = date(2018, 1, 1)
TARGET_END_EXCLUSIVE = date(2021, 7, 1)
LUXEMBOURG_BBOX = (5.73, 49.44, 6.54, 50.19)
USER_AGENT = "mvdec-fpmax-airpollution-reproduction/0.1"
ACCESS_TOKEN_ENV = "CDSE_ACCESS_TOKEN"
_BUFFER_SIZE = 1024 * 1024


@dataclass(frozen=True)
class SentinelProduct:
    """Immutable catalogue metadata needed to identify and verify one product."""

    id: str
    name: str
    content_length: int
    start: str
    end: str
    s3_path: str
    checksums: dict[str, str]

    @classmethod
    def from_mapping(cls, item: Mapping[str, object]) -> SentinelProduct:
        content_date = item.get("ContentDate")
        checksum_items = item.get("Checksum")
        if not isinstance(content_date, Mapping) or not isinstance(
            checksum_items, list
        ):
            raise ValueError("Catalogue product lacks dates or checksums.")
        checksums: dict[str, str] = {}
        for checksum in checksum_items:
            if not isinstance(checksum, Mapping):
                raise ValueError("Catalogue checksum entry must be an object.")
            algorithm = str(checksum.get("Algorithm", "")).upper()
            value = str(checksum.get("Value", "")).lower()
            if algorithm and value:
                checksums[algorithm] = value
        product = cls(
            id=str(item.get("Id", "")),
            name=str(item.get("Name", "")),
            content_length=int(item.get("ContentLength", 0)),
            start=str(content_date.get("Start", "")),
            end=str(content_date.get("End", "")),
            s3_path=str(item.get("S3Path", "")),
            checksums=checksums,
        )
        validate_product(product)
        return product


def default_raw_root() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "data"
        / "public"
        / "raw"
        / "airpollution_reconstruction"
    )


def default_inventory_path() -> Path:
    return default_raw_root() / "sentinel5p" / "no2_inventory.json"


def _iso_timestamp(day: date) -> str:
    return f"{day.isoformat()}T00:00:00.000Z"


def _validate_bbox(bbox: tuple[float, float, float, float]) -> None:
    west, south, east, north = bbox
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError(f"Invalid WGS84 bounding box: {bbox}")


def build_catalogue_filter(
    start: date,
    end_exclusive: date,
    bbox: tuple[float, float, float, float] = LUXEMBOURG_BBOX,
) -> str:
    """Build the exact historical OData filter persisted in the inventory."""

    if start >= end_exclusive:
        raise ValueError("Inventory start must be earlier than its exclusive end.")
    _validate_bbox(bbox)
    west, south, east, north = bbox
    polygon = (
        f"POLYGON(({west:g} {south:g},{east:g} {south:g},"
        f"{east:g} {north:g},{west:g} {north:g},{west:g} {south:g}))"
    )
    return (
        f"Collection/Name eq '{COLLECTION_NAME}' and "
        f"contains(Name,'{PRODUCT_NAME_TOKEN}') and "
        "OData.CSC.Intersects(area=geography'"
        f"SRID=4326;{polygon}') and "
        f"ContentDate/Start ge {_iso_timestamp(start)} and "
        f"ContentDate/Start lt {_iso_timestamp(end_exclusive)}"
    )


def validate_product(product: SentinelProduct) -> None:
    """Validate catalogue data before using it in paths or network requests."""

    try:
        uuid.UUID(product.id)
    except ValueError as error:
        raise ValueError(f"Invalid Sentinel product UUID: {product.id!r}") from error
    if (
        not product.name.endswith(".nc")
        or PRODUCT_NAME_TOKEN not in product.name
        or Path(product.name).name != product.name
    ):
        raise ValueError(
            f"Unsafe or unexpected Sentinel product name: {product.name!r}"
        )
    if product.content_length <= 0:
        raise ValueError(f"Invalid content length for {product.name!r}.")
    for value in (product.start, product.end):
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(
                f"Invalid Sentinel product timestamp: {value!r}"
            ) from error
    md5 = product.checksums.get("MD5", "")
    if len(md5) != 32 or any(character not in "0123456789abcdef" for character in md5):
        raise ValueError(f"Product {product.name!r} lacks a valid catalogue MD5.")


def query_catalogue(
    start: date = TARGET_START,
    end_exclusive: date = TARGET_END_EXCLUSIVE,
    bbox: tuple[float, float, float, float] = LUXEMBOURG_BBOX,
    *,
    page_size: int = 1000,
    max_products: int | None = None,
) -> tuple[str, list[SentinelProduct]]:
    """Query historical product metadata in deterministic chronological pages."""

    if page_size <= 0 or page_size > 1000:
        raise ValueError("Catalogue page size must be between 1 and 1000.")
    if max_products is not None and max_products <= 0:
        raise ValueError("Maximum product count must be positive when provided.")
    catalogue_filter = build_catalogue_filter(start, end_exclusive, bbox)
    products: list[SentinelProduct] = []
    skip = 0
    while max_products is None or len(products) < max_products:
        remaining = None if max_products is None else max_products - len(products)
        top = page_size if remaining is None else min(page_size, remaining)
        query = urlencode(
            {
                "$filter": catalogue_filter,
                "$select": ("Id,Name,ContentLength,ContentDate,S3Path,Checksum"),
                "$orderby": "ContentDate/Start asc,Name asc",
                "$top": str(top),
                "$skip": str(skip),
            }
        )
        request = Request(
            f"{CATALOGUE_URL}?{query}",
            headers={"User-Agent": USER_AGENT},
            method="GET",
        )
        with urlopen(request, timeout=120) as response:
            payload = json.load(response)
        values = payload.get("value")
        if not isinstance(values, list):
            raise ValueError("Catalogue response does not contain a product list.")
        page = [SentinelProduct.from_mapping(item) for item in values]
        products.extend(page)
        if len(page) < top:
            break
        skip += len(page)

    product_ids = [product.id for product in products]
    product_names = [product.name for product in products]
    if len(set(product_ids)) != len(product_ids):
        raise ValueError("Catalogue response contains duplicate product UUIDs.")
    if len(set(product_names)) != len(product_names):
        raise ValueError("Catalogue response contains duplicate product names.")
    return catalogue_filter, products


def build_inventory(
    start: date = TARGET_START,
    end_exclusive: date = TARGET_END_EXCLUSIVE,
    bbox: tuple[float, float, float, float] = LUXEMBOURG_BBOX,
    *,
    page_size: int = 1000,
    max_products: int | None = None,
) -> dict[str, object]:
    catalogue_filter, products = query_catalogue(
        start,
        end_exclusive,
        bbox,
        page_size=page_size,
        max_products=max_products,
    )
    return {
        "schema_version": 1,
        "status": "catalogue_inventory_not_downloaded_no2_products",
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "source": {
            "provider": "Copernicus Data Space Ecosystem",
            "catalogue_url": CATALOGUE_URL,
            "collection": COLLECTION_NAME,
            "product_name_token": PRODUCT_NAME_TOKEN,
        },
        "query": {
            "start_inclusive": start.isoformat(),
            "end_exclusive": end_exclusive.isoformat(),
            "bbox_wgs84": list(bbox),
            "odata_filter": catalogue_filter,
        },
        "product_count": len(products),
        "total_content_bytes": sum(product.content_length for product in products),
        "products": [asdict(product) for product in products],
    }


def write_inventory(inventory: Mapping[str, object], path: Path) -> Path:
    """Write an inventory atomically so interrupted queries do not corrupt it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(inventory, temporary, indent=2, sort_keys=True, allow_nan=False)
            temporary.write("\n")
        temporary_path.replace(path)
        return path
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_inventory(path: Path) -> tuple[dict[str, object], list[SentinelProduct]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_products = payload.get("products")
    if not isinstance(raw_products, list):
        raise ValueError("Sentinel inventory does not contain a product list.")
    products = [SentinelProduct(**item) for item in raw_products]
    for product in products:
        validate_product(product)
    return payload, products


def _file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_BUFFER_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest().lower()


def verify_product_file(path: Path, product: SentinelProduct) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing Sentinel product: {path}")
    actual_size = path.stat().st_size
    if actual_size != product.content_length:
        raise ValueError(
            f"Size mismatch for {product.name!r}: expected {product.content_length}, "
            f"got {actual_size}."
        )
    actual_md5 = _file_digest(path, "md5")
    expected_md5 = product.checksums["MD5"].lower()
    if actual_md5 != expected_md5:
        raise ValueError(
            f"Checksum mismatch for {product.name!r}: expected MD5 "
            f"{expected_md5}, got {actual_md5}."
        )


def product_destination(product: SentinelProduct, raw_root: Path) -> Path:
    start = datetime.fromisoformat(product.start.replace("Z", "+00:00"))
    return (
        raw_root
        / "sentinel5p"
        / "no2"
        / f"{start.year:04d}"
        / f"{start.month:02d}"
        / f"{start.day:02d}"
        / product.name
    )


def public_s3_url(product: SentinelProduct) -> str:
    """Map an official product name to the documented public Sentinel-5P bucket."""

    parts = product.name.split("_")
    if len(parts) < 2 or parts[1] not in {"RPRO", "OFFL", "NRTI"}:
        raise ValueError(
            f"Cannot derive Sentinel processing mode from {product.name!r}."
        )
    sensing_token = product.name.split("NO2____", maxsplit=1)[1]
    sensing_day = sensing_token[:8]
    if len(sensing_day) != 8 or not sensing_day.isdigit():
        raise ValueError(f"Cannot derive Sentinel sensing day from {product.name!r}.")
    key = (
        f"{parts[1]}/{PRODUCT_NAME_TOKEN}/{sensing_day[:4]}/"
        f"{sensing_day[4:6]}/{sensing_day[6:8]}/{product.name}"
    )
    return f"{PUBLIC_S3_URL}/{quote(key, safe='/')}"


def _download_request(
    product: SentinelProduct,
    source: str,
    access_token: str | None,
) -> Request:
    if source == "public-s3":
        return Request(
            public_s3_url(product),
            headers={"User-Agent": USER_AGENT},
            method="GET",
        )
    if source != "cdse":
        raise ValueError(f"Unknown Sentinel download source: {source!r}")
    token = access_token or os.environ.get(ACCESS_TOKEN_ENV)
    if not token:
        raise ValueError(
            f"Set {ACCESS_TOKEN_ENV} to a current Copernicus Data Space access token."
        )
    return Request(
        f"{DOWNLOAD_URL}({product.id})/$value",
        headers={
            "Authorization": f"Bearer {token}",
            "User-Agent": USER_AGENT,
        },
        method="GET",
    )


def fetch_product(
    product: SentinelProduct,
    raw_root: Path,
    *,
    force: bool = False,
    access_token: str | None = None,
    source: str = "public-s3",
) -> Path:
    """Download one explicitly selected product and verify official metadata."""

    validate_product(product)
    destination = product_destination(product, raw_root)
    if destination.exists() and not force:
        verify_product_file(destination, product)
        return destination

    request = _download_request(product, source, access_token)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            delete=False,
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        ) as temporary:
            temporary_path = Path(temporary.name)
            with urlopen(request, timeout=300) as response:
                shutil.copyfileobj(response, temporary, length=_BUFFER_SIZE)
        verify_product_file(temporary_path, product)
        temporary_path.replace(destination)
        return destination
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def select_products(
    products: Sequence[SentinelProduct],
    identifiers: Sequence[str],
) -> list[SentinelProduct]:
    """Resolve explicit UUIDs or exact product names without implicit bulk fetches."""

    by_identifier = {
        identifier: product
        for product in products
        for identifier in (product.id, product.name)
    }
    unknown = sorted(set(identifiers) - by_identifier.keys())
    if unknown:
        raise ValueError(f"Unknown Sentinel products: {', '.join(unknown)}")
    selected: list[SentinelProduct] = []
    seen: set[str] = set()
    for identifier in identifiers:
        product = by_identifier[identifier]
        if product.id not in seen:
            selected.append(product)
            seen.add(product.id)
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    inventory = commands.add_parser("inventory", help="Query product metadata only.")
    inventory.add_argument("--start", type=date.fromisoformat, default=TARGET_START)
    inventory.add_argument(
        "--end-exclusive",
        type=date.fromisoformat,
        default=TARGET_END_EXCLUSIVE,
    )
    inventory.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        default=LUXEMBOURG_BBOX,
    )
    inventory.add_argument("--page-size", type=int, default=1000)
    inventory.add_argument("--max-products", type=int)
    inventory.add_argument("--output", type=Path, default=default_inventory_path())

    download = commands.add_parser(
        "download",
        help="Download only explicitly selected inventory products.",
    )
    download.add_argument("--inventory", type=Path, default=default_inventory_path())
    download.add_argument("--ids", nargs="+", required=True, metavar="UUID_OR_NAME")
    download.add_argument("--raw-root", type=Path, default=default_raw_root())
    download.add_argument(
        "--source",
        choices=("public-s3", "cdse"),
        default="public-s3",
        help="Public bucket needs no credentials; CDSE requires CDSE_ACCESS_TOKEN.",
    )
    download.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inventory":
        bbox = tuple(args.bbox)
        inventory = build_inventory(
            args.start,
            args.end_exclusive,
            bbox,
            page_size=args.page_size,
            max_products=args.max_products,
        )
        path = write_inventory(inventory, args.output)
        print(path)
        print(f"products={inventory['product_count']}")
        print(f"content_bytes={inventory['total_content_bytes']}")
        return 0

    _, products = load_inventory(args.inventory)
    selected = select_products(products, args.ids)
    for product in selected:
        print(
            fetch_product(
                product,
                args.raw_root,
                force=args.force,
                source=args.source,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
