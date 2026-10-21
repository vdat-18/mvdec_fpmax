"""Deterministic source manifest and fetcher for air-pollution reconstruction."""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

USER_AGENT = "mvdec-fpmax-airpollution-reproduction/0.1"
_BUFFER_SIZE = 1024 * 1024
_SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}")


@dataclass(frozen=True)
class AirPollutionSource:
    """One immutable raw input required by the clean-room reconstruction."""

    name: str
    license: str
    source_date: str
    destination: Path
    expected_size: int
    sha256: str
    url: str
    overpass_query: str | None = None
    historical_query_date: str | None = None


SOURCES: tuple[AirPollutionSource, ...] = (
    AirPollutionSource(
        name="traffic-2018",
        license="CC0-1.0",
        source_date="2018",
        destination=Path("traffic/donneestrafic-2018.xlsx"),
        expected_size=81_722_130,
        sha256="DD4175236A5DDDFF75F8D671CCDBA06AB3B8BE588384C0C8059EE72F98E44BDB",
        url=(
            "https://download.data.public.lu/resources/pch-comptage-trafic/"
            "20190618-154117/donneestrafic-2018.xlsx"
        ),
    ),
    AirPollutionSource(
        name="boundary-communes-2023",
        license="CC0-1.0",
        source_date="2023-11-23",
        destination=Path("boundary/communes4326.geojson"),
        expected_size=1_040_927,
        sha256="48FA4E6F293740017ED1AB941A9A2F3024A7B326F0E21F5B3689329785C515C9",
        url=(
            "https://download.data.public.lu/resources/"
            "limites-administratives-du-grand-duche-de-luxembourg/"
            "20231123-101528/communes4326.geojson"
        ),
    ),
    AirPollutionSource(
        name="osm-roads-2021-06-30",
        license="ODbL-1.0",
        source_date="2021-06-30T23:59:59Z",
        destination=Path("osm/luxembourg_roads_2021-06-30.json"),
        expected_size=6_790_602,
        sha256="C9E8F4742B7142EC942980A79FFF3A80A4D311771361053907AC1643F48C033C",
        url="https://overpass-api.de/api/interpreter",
        overpass_query=(
            '[out:json][timeout:900][date:"2021-06-30T23:59:59Z"];\n'
            "(\n"
            '  way["highway"~"^(motorway|trunk|primary)$"]'
            "(49.44,5.73,50.19,6.54);\n"
            ");\n"
            "out geom;"
        ),
        historical_query_date="2021-06-30T23:59:59Z",
    ),
    AirPollutionSource(
        name="osm-landuse-ways-2021-06-30",
        license="ODbL-1.0",
        source_date="2021-06-30T23:59:59Z",
        destination=Path("osm/luxembourg_landuse_ways_2021-06-30.json"),
        expected_size=70_264_996,
        sha256="56747AF8201CA35681F75D0151EF789151284BFFFF042A24BDD202B1C1551211",
        url="https://overpass-api.de/api/interpreter",
        overpass_query=(
            '[out:json][timeout:1800][date:"2021-06-30T23:59:59Z"];\n'
            "(\n"
            '  way["landuse"~"^(industrial|residential|forest)$"]'
            "(49.44,5.73,50.19,6.54);\n"
            '  way["natural"="wood"](49.44,5.73,50.19,6.54);\n'
            ");\n"
            "out geom;"
        ),
        historical_query_date="2021-06-30T23:59:59Z",
    ),
    AirPollutionSource(
        name="geofabrik-luxembourg-2021-01-01",
        license="ODbL-1.0",
        source_date="2021-01-01",
        destination=Path("osm/luxembourg-210101-free.shp.zip"),
        expected_size=52_856_901,
        sha256="F7A5F48C514E824A2C1C8B56A3D5EC7AFE4FC336AA0CF235A1911A6737A98A0D",
        url="https://download.geofabrik.de/europe/luxembourg-210101-free.shp.zip",
    ),
)


def default_raw_root() -> Path:
    """Return the ignored workspace directory used for raw reconstruction inputs."""

    return (
        Path(__file__).resolve().parents[2]
        / "data"
        / "public"
        / "raw"
        / "airpollution_reconstruction"
    )


def validate_manifest(sources: Sequence[AirPollutionSource]) -> None:
    """Reject ambiguous or unsafe source declarations before any network access."""

    names: set[str] = set()
    destinations: set[str] = set()
    for source in sources:
        if not source.name or source.name in names:
            raise ValueError(f"Duplicate or empty source name: {source.name!r}")
        names.add(source.name)

        destination = source.destination
        if (
            destination.is_absolute()
            or destination.drive
            or ".." in destination.parts
            or destination == Path(".")
        ):
            raise ValueError(
                f"Source {source.name!r} has unsafe destination: {destination}"
            )
        destination_key = destination.as_posix().casefold()
        if destination_key in destinations:
            raise ValueError(f"Duplicate source destination: {destination}")
        destinations.add(destination_key)

        if not isinstance(source.expected_size, int) or source.expected_size <= 0:
            raise ValueError(f"Source {source.name!r} must have a positive size.")
        if _SHA256_PATTERN.fullmatch(source.sha256) is None:
            raise ValueError(f"Source {source.name!r} has an invalid SHA-256 digest.")

        parsed_url = urlparse(source.url)
        if parsed_url.scheme != "https" or not parsed_url.netloc:
            raise ValueError(f"Source {source.name!r} must use an HTTPS URL.")

        has_query = source.overpass_query is not None
        has_historical_date = source.historical_query_date is not None
        if has_query != has_historical_date:
            raise ValueError(
                f"Source {source.name!r} must define both Overpass query fields."
            )
        if has_query and (
            not source.overpass_query or not source.historical_query_date
        ):
            raise ValueError(f"Source {source.name!r} has empty Overpass metadata.")


def file_sha256(path: Path) -> str:
    """Calculate an uppercase SHA-256 digest without loading the file into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_BUFFER_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def verify_source(path: Path, source: AirPollutionSource) -> None:
    """Verify one local file against its immutable manifest declaration."""

    if not path.is_file():
        raise FileNotFoundError(f"Missing source {source.name!r}: {path}")
    actual_size = path.stat().st_size
    if actual_size != source.expected_size:
        raise ValueError(
            f"Size mismatch for {source.name!r}: expected {source.expected_size}, "
            f"got {actual_size}."
        )
    actual_digest = file_sha256(path)
    if actual_digest != source.sha256.upper():
        raise ValueError(
            f"Checksum mismatch for {source.name!r}: expected "
            f"{source.sha256.upper()}, got {actual_digest}."
        )


def _request_for(source: AirPollutionSource) -> Request:
    if source.overpass_query is None:
        return Request(
            source.url,
            headers={"User-Agent": USER_AGENT},
            method="GET",
        )
    body = urlencode({"data": source.overpass_query}).encode("ascii")
    return Request(
        source.url,
        data=body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": USER_AGENT,
        },
        method="POST",
    )


def fetch_source(
    source: AirPollutionSource,
    raw_root: Path,
    force: bool = False,
) -> Path:
    """Fetch one source atomically, preserving any existing file on failure."""

    validate_manifest((source,))
    destination = raw_root / source.destination
    if destination.exists() and not force:
        verify_source(destination, source)
        return destination

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
            with urlopen(_request_for(source)) as response:
                shutil.copyfileobj(response, temporary, length=_BUFFER_SIZE)
        verify_source(temporary_path, source)
        temporary_path.replace(destination)
        return destination
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def fetch_selected(
    names: Sequence[str],
    raw_root: Path,
    force: bool = False,
) -> list[Path]:
    """Fetch only explicitly named manifest entries, in the requested order."""

    validate_manifest(SOURCES)
    sources_by_name = {source.name: source for source in SOURCES}
    unknown = sorted(set(names) - sources_by_name.keys())
    if unknown:
        raise ValueError(f"Unknown air-pollution sources: {', '.join(unknown)}")
    return [fetch_source(sources_by_name[name], raw_root, force) for name in names]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch verified raw sources for air-pollution reconstruction."
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--sources",
        nargs="+",
        metavar="NAME",
        help="Explicit manifest source names to fetch or verify.",
    )
    action.add_argument(
        "--list",
        action="store_true",
        help="List available source names without downloading.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=default_raw_root(),
        help="Destination root for ignored raw inputs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download and verify a replacement even when a cache exists.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_manifest(SOURCES)
    if args.list:
        for source in SOURCES:
            print(source.name)
        return 0

    paths = fetch_selected(args.sources, args.raw_root, args.force)
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
