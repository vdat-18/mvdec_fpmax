"""Download the exact public data files linked by the DEKM 2021 release."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Final

from config import DEKM_DATASET_DIR

SOURCE_FOLDER_ID: Final = "1raiYP1joy8gtsHXYcW5EuNtECSPRu37v"
REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[1]
DATASET_FILES: Final = {
    "REUTERS": {
        "10k_feature.npy": "1SYqb-FLU4XO67RT29Z3-7fC3uXfCXpYp",
        "10k_target.npy": "1aliYFqBjJyYcMhx8myka5QO9f0OFr8nR",
    },
    "20NEWS": {
        "test_data.npz": "1nX5iESm4O4qfL2-wjxU6KIhFoZMJt-gC",
        "test_label.npz": "1vi12IrF-hhKlmkx0r4jpT77ANWrDdjTb",
        "train_data.npz": "15eZ9k1aEZR1LtUGpkM6MxxNFmIxIRGJn",
        "train_label.npz": "1IJAPTf9STJr74rUD3bOs5hjuoO_XxUjl",
    },
    "RCV1": {
        "test": "1WzfCSj8yAgnHsb5tiIQRmw6uwz03td5K",
        "validation": "1d-ZJ5SxneLkTYiRUAQ5y1Ov0ZGuAyLvQ",
    },
}
CANONICAL_SHA256: Final = {
    ("RCV1", "test"): (
        "26c22ad47eda2c283a5481c2f705da61f862c0b7b457941a1564b07f605133bf"
    ),
    ("RCV1", "validation"): (
        "a1a689eb7f808e80d57dab694e57d98fe92a9ba3ef2b787e7f33547a03e9cf92"
    ),
}
BUNDLED_FALLBACK_ROOT: Final = REPOSITORY_ROOT / "data/public/dekm_release"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASET_FILES),
        default=list(DATASET_FILES),
    )
    parser.add_argument("--output-dir", type=Path, default=DEKM_DATASET_DIR)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def import_gdown():
    try:
        import gdown
    except ImportError as error:
        raise RuntimeError(
            "gdown is required. Install the DEKM tools with "
            "`uv sync --group dekm` or `pip install gdown`."
        ) from error
    return gdown


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(
    gdown,
    file_id: str,
    destination: Path,
    force: bool,
    *,
    fallback_path: Path | None = None,
    expected_sha256: str | None = None,
) -> str:
    if destination.exists() and destination.stat().st_size > 0 and not force:
        if expected_sha256 is not None and sha256(destination) != expected_sha256:
            raise RuntimeError(f"Hash mismatch for existing DEKM file {destination}.")
        return "existing"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.download")
    if temporary.exists():
        temporary.unlink()
    try:
        result = gdown.download(id=file_id, output=str(temporary), quiet=False)
    except Exception as error:  # gdown exposes multiple transport exception types.
        print(f"Google Drive download failed for {destination.name}: {error}")
        result = None
    if result is None or not temporary.exists() or temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        if fallback_path is None or not fallback_path.is_file():
            raise RuntimeError(f"Failed to download DEKM file {destination.name}.")
        if expected_sha256 is None or sha256(fallback_path) != expected_sha256:
            raise RuntimeError(
                f"Bundled fallback failed verification for {destination.name}."
            )
        shutil.copyfile(fallback_path, temporary)
        retrieval = "bundled_canonical_copy"
    else:
        retrieval = "google_drive"
    if expected_sha256 is not None and sha256(temporary) != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Hash mismatch for downloaded DEKM file {destination.name}."
        )
    temporary.replace(destination)
    return retrieval


def main() -> None:
    args = parse_args()
    gdown = import_gdown()
    root = args.output_dir.resolve()
    metadata: dict[str, object] = {
        "source": "DEKM 2021 official Google Drive folder",
        "source_folder_id": SOURCE_FOLDER_ID,
        "datasets": {},
    }
    for dataset in args.datasets:
        records = {}
        for filename, file_id in DATASET_FILES[dataset].items():
            destination = root / dataset / filename
            expected_sha256 = CANONICAL_SHA256.get((dataset, filename))
            fallback_path = BUNDLED_FALLBACK_ROOT / dataset / filename
            retrieval = download_file(
                gdown,
                file_id,
                destination,
                args.force,
                fallback_path=fallback_path,
                expected_sha256=expected_sha256,
            )
            records[filename] = {
                "google_drive_file_id": file_id,
                "retrieval": retrieval,
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }
        metadata["datasets"][dataset] = records
        print(f"ready dataset={dataset} path={root / dataset}")

    metadata_path = root / "download_manifest.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if "RCV1" in args.datasets:
        print(
            "RCV1 note: the release contains split indices only; the scikit-learn "
            "RCV1 cache is also required."
        )
    print(f"manifest={metadata_path}")


if __name__ == "__main__":
    main()
