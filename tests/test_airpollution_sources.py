from __future__ import annotations

import hashlib
import io
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlencode

import pytest

from pipeline import airpollution_sources as sources


def make_source(
    payload: bytes = b"verified payload",
    *,
    name: str = "synthetic",
    destination: str = "synthetic/source.bin",
    overpass_query: str | None = None,
    historical_query_date: str | None = None,
) -> sources.AirPollutionSource:
    return sources.AirPollutionSource(
        name=name,
        license="CC0-1.0",
        source_date="2021-01-01",
        destination=Path(destination),
        expected_size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        url="https://example.test/source",
        overpass_query=overpass_query,
        historical_query_date=historical_query_date,
    )


def test_manifest_contains_only_five_raw_public_sources() -> None:
    sources.validate_manifest(sources.SOURCES)

    assert [source.name for source in sources.SOURCES] == [
        "traffic-2018",
        "boundary-communes-2023",
        "osm-roads-2021-06-30",
        "osm-landuse-ways-2021-06-30",
        "geofabrik-luxembourg-2021-01-01",
    ]
    assert all(
        "data_demvk" not in source.destination.as_posix() for source in sources.SOURCES
    )


def test_valid_cache_does_not_access_network(tmp_path: Path, monkeypatch) -> None:
    payload = b"valid cache"
    source = make_source(payload)
    destination = tmp_path / source.destination
    destination.parent.mkdir(parents=True)
    destination.write_bytes(payload)

    def fail_urlopen(_request):
        raise AssertionError("A valid cache must not access the network.")

    monkeypatch.setattr(sources, "urlopen", fail_urlopen)

    assert sources.fetch_source(source, tmp_path) == destination


def test_tampered_cache_is_rejected_without_network(
    tmp_path: Path, monkeypatch
) -> None:
    source = make_source(b"expected")
    destination = tmp_path / source.destination
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"tampered")

    def fail_urlopen(_request):
        raise AssertionError("A tampered cache must fail before network access.")

    monkeypatch.setattr(sources, "urlopen", fail_urlopen)

    with pytest.raises(ValueError, match="Size mismatch|Checksum mismatch"):
        sources.fetch_source(source, tmp_path)


def test_failed_forced_replacement_preserves_existing_file(
    tmp_path: Path, monkeypatch
) -> None:
    payload = b"original valid content"
    source = make_source(payload)
    destination = tmp_path / source.destination
    destination.parent.mkdir(parents=True)
    destination.write_bytes(payload)
    replacement = b"x" * len(payload)
    monkeypatch.setattr(sources, "urlopen", lambda _request: io.BytesIO(replacement))

    with pytest.raises(ValueError, match="Checksum mismatch"):
        sources.fetch_source(source, tmp_path, force=True)

    assert destination.read_bytes() == payload
    assert list(destination.parent.glob("*.tmp")) == []


@pytest.mark.parametrize(
    "invalid_source, message",
    [
        (replace(make_source(), destination=Path("../escape.bin")), "unsafe"),
        (replace(make_source(), destination=Path("C:/escape.bin")), "unsafe"),
        (replace(make_source(), sha256="not-a-digest"), "SHA-256"),
        (replace(make_source(), url="http://example.test/source"), "HTTPS"),
        (replace(make_source(), expected_size=0), "positive size"),
        (
            replace(make_source(), overpass_query="out;"),
            "both Overpass query fields",
        ),
        (
            replace(make_source(), historical_query_date="2021-01-01T00:00:00Z"),
            "both Overpass query fields",
        ),
    ],
)
def test_manifest_validation_rejects_malformed_entries(
    invalid_source: sources.AirPollutionSource, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        sources.validate_manifest((invalid_source,))


def test_manifest_validation_rejects_duplicate_names_and_destinations() -> None:
    first = make_source(name="first", destination="one.bin")
    duplicate_name = make_source(name="first", destination="two.bin")
    duplicate_destination = make_source(name="second", destination="ONE.bin")

    with pytest.raises(ValueError, match="source name"):
        sources.validate_manifest((first, duplicate_name))
    with pytest.raises(ValueError, match="destination"):
        sources.validate_manifest((first, duplicate_destination))


def test_overpass_request_uses_exact_deterministic_body(
    tmp_path: Path, monkeypatch
) -> None:
    payload = b"overpass response"
    query = '[out:json][date:"2021-01-01T00:00:00Z"];out;'
    source = make_source(
        payload,
        overpass_query=query,
        historical_query_date="2021-01-01T00:00:00Z",
    )
    captured = []

    def fake_urlopen(request):
        captured.append(request)
        return io.BytesIO(payload)

    monkeypatch.setattr(sources, "urlopen", fake_urlopen)

    sources.fetch_source(source, tmp_path)

    assert len(captured) == 1
    request = captured[0]
    assert request.get_method() == "POST"
    assert request.data == urlencode({"data": query}).encode("ascii")
    assert request.get_header("Content-type") == "application/x-www-form-urlencoded"
    assert request.get_header("User-agent") == sources.USER_AGENT


def test_ordinary_request_uses_explicit_get(tmp_path: Path, monkeypatch) -> None:
    payload = b"ordinary response"
    source = make_source(payload)
    captured = []

    def fake_urlopen(request):
        captured.append(request)
        return io.BytesIO(payload)

    monkeypatch.setattr(sources, "urlopen", fake_urlopen)

    sources.fetch_source(source, tmp_path)

    assert captured[0].get_method() == "GET"
    assert captured[0].data is None
    assert captured[0].get_header("User-agent") == sources.USER_AGENT


def test_fetch_selected_fetches_only_requested_sources(
    tmp_path: Path, monkeypatch
) -> None:
    first = make_source(name="first", destination="first.bin")
    second = make_source(name="second", destination="second.bin")
    monkeypatch.setattr(sources, "SOURCES", (first, second))
    fetched = []

    def fake_fetch(source, raw_root, force):
        fetched.append((source.name, raw_root, force))
        return raw_root / source.destination

    monkeypatch.setattr(sources, "fetch_source", fake_fetch)

    result = sources.fetch_selected(("second",), tmp_path, force=True)

    assert fetched == [("second", tmp_path, True)]
    assert result == [tmp_path / second.destination]


def test_fetch_selected_rejects_unknown_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown air-pollution sources: missing"):
        sources.fetch_selected(("missing",), tmp_path)


def test_verify_source_reports_size_mismatch(tmp_path: Path) -> None:
    source = make_source(b"expected")
    path = tmp_path / "source.bin"
    path.write_bytes(b"wrong")

    with pytest.raises(ValueError, match="Size mismatch"):
        sources.verify_source(path, source)


def test_cli_requires_explicit_sources_or_list() -> None:
    with pytest.raises(SystemExit) as error:
        sources.main([])

    assert error.value.code == 2


def test_cli_list_does_not_fetch(monkeypatch, capsys) -> None:
    def fail_fetch(*_args, **_kwargs):
        raise AssertionError("--list must not fetch files.")

    monkeypatch.setattr(sources, "fetch_selected", fail_fetch)

    assert sources.main(["--list"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        source.name for source in sources.SOURCES
    ]
