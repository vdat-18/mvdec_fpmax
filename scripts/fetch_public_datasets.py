"""Fetch public datasets and export clean numeric clustering inputs."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
from loguru import logger

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_DIR / "configs" / "public_datasets.json"
USER_AGENT = "mvdec-fpmax-public-dataset-fetcher/1.0"


@dataclass(frozen=True)
class ProcessedDataset:
    """Standardized dataset payload ready to persist."""

    X: pd.DataFrame
    y: pd.DataFrame | None
    views: dict[str, list[str]]
    notes: list[str]


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(
        description="Download and standardize public datasets for mvdec-fpmax."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Dataset registry JSON path.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset names to fetch. Defaults to every registry entry.",
    )
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="Download and extract raw files without writing processed X/y files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload and reprocess datasets even when local files already exist.",
    )
    return parser.parse_args()


def load_registry(config_path: Path) -> dict:
    """Load the public dataset registry."""

    with config_path.open(encoding="utf-8") as file:
        return json.load(file)


def safe_rmtree(path: Path, allowed_root: Path) -> None:
    """Remove a directory only when it is under an allowed root."""

    resolved = path.resolve()
    allowed = allowed_root.resolve()
    if resolved == allowed or allowed not in resolved.parents:
        msg = f"Refusing to remove path outside {allowed}: {resolved}"
        raise ValueError(msg)
    if path.exists():
        shutil.rmtree(path)


def download_file(url: str, output_path: Path, force: bool) -> None:
    """Download a URL to a local file with an atomic replace."""

    if output_path.exists() and not force:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": USER_AGENT})
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    with urlopen(request, timeout=180) as response, tmp_path.open("wb") as file:
        shutil.copyfileobj(response, file)
    tmp_path.replace(output_path)


def extract_zip(archive_path: Path, extract_dir: Path, force: bool) -> None:
    """Extract a zip archive after validating member paths."""

    if extract_dir.exists() and any(extract_dir.iterdir()) and not force:
        return
    if extract_dir.exists():
        safe_rmtree(extract_dir, extract_dir.parent)
    extract_dir.mkdir(parents=True, exist_ok=True)
    resolved_extract_dir = extract_dir.resolve()
    with ZipFile(archive_path) as zip_file:
        for member in zip_file.infolist():
            target = (extract_dir / member.filename).resolve()
            if (
                resolved_extract_dir not in target.parents
                and target != resolved_extract_dir
            ):
                msg = f"Unsafe zip member path: {member.filename}"
                raise ValueError(msg)
        zip_file.extractall(extract_dir)


def read_text(path: Path, encoding: str = "latin1") -> str:
    """Read a text file with replacement for invalid bytes."""

    return path.read_text(encoding=encoding, errors="replace")


def read_csv_no_header(path: Path, **kwargs) -> pd.DataFrame:
    """Read a headerless CSV-like file."""

    return pd.read_csv(path, header=None, **kwargs)


def make_columns(prefix: str, count: int) -> list[str]:
    """Create deterministic numbered feature names."""

    return [f"{prefix}_{index}" for index in range(1, count + 1)]


def make_auto_views(columns: list[str], n_views: int = 3) -> dict[str, list[str]]:
    """Split columns into simple constructed views."""

    if not columns:
        return {}
    splits = np.array_split(np.array(columns, dtype=object), min(n_views, len(columns)))
    return {
        f"view_{index}": [str(column) for column in split.tolist()]
        for index, split in enumerate(splits, start=1)
        if len(split) > 0
    }


def dedupe_columns(columns: list[str]) -> list[str]:
    """Make duplicate column names unique while preserving order."""

    counts: dict[str, int] = {}
    deduped: list[str] = []
    for column in columns:
        base = str(column).strip().replace(" ", "_")
        counts[base] = counts.get(base, 0) + 1
        suffix = f"_{counts[base]}" if counts[base] > 1 else ""
        deduped.append(f"{base}{suffix}")
    return deduped


def ensure_valid_numeric_X(X: pd.DataFrame, dataset_name: str) -> None:
    """Validate that X is non-empty, numeric, and complete."""

    if X.empty:
        msg = f"{dataset_name}: X is empty."
        raise ValueError(msg)

    invalid_columns = [
        column for column in X.columns if not pd.api.types.is_numeric_dtype(X[column])
    ]
    if invalid_columns:
        msg = f"{dataset_name}: non-numeric columns in X: {invalid_columns}"
        raise ValueError(msg)

    if X.isna().to_numpy().any():
        msg = f"{dataset_name}: X still contains missing values after cleaning."
        raise ValueError(msg)


def align_y(y: pd.DataFrame | None, kept_index: pd.Index) -> pd.DataFrame | None:
    """Align labels or metadata to retained X rows."""

    if y is None:
        return None
    return y.loc[kept_index].reset_index(drop=True)


def clean_X_y(
    X: pd.DataFrame,
    y: pd.DataFrame | None,
    dataset_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame | None, list[str]]:
    """Coerce X to numeric values and drop invalid rows/columns."""

    notes: list[str] = []
    X_numeric = X.copy()
    for column in X_numeric.columns:
        X_numeric[column] = pd.to_numeric(X_numeric[column], errors="coerce")
    X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan)

    constant_columns = [
        column
        for column in X_numeric.columns
        if X_numeric[column].nunique(dropna=True) <= 1
    ]
    if constant_columns:
        X_numeric = X_numeric.drop(columns=constant_columns)
        notes.append(f"Dropped constant columns: {constant_columns}.")

    kept_mask = ~X_numeric.isna().any(axis=1)
    dropped_rows = int((~kept_mask).sum())
    if dropped_rows:
        notes.append(f"Dropped {dropped_rows} rows with missing/non-finite X values.")

    cleaned_X = X_numeric.loc[kept_mask].reset_index(drop=True)
    cleaned_X = cleaned_X.astype(float)
    ensure_valid_numeric_X(cleaned_X, dataset_name)
    cleaned_y = align_y(y, X_numeric.index[kept_mask])
    return cleaned_X, cleaned_y, notes


def parse_mfeat(extract_dir: Path) -> ProcessedDataset:
    """Parse UCI Multiple Features into six natural views."""

    view_files = {
        "fourier": "mfeat-fou",
        "profile": "mfeat-fac",
        "karhunen_loeve": "mfeat-kar",
        "morphological": "mfeat-mor",
        "pixel_average": "mfeat-pix",
        "zernike": "mfeat-zer",
    }
    frames: list[pd.DataFrame] = []
    views: dict[str, list[str]] = {}
    for view_name, filename in view_files.items():
        frame = pd.read_csv(
            extract_dir / filename,
            sep=r"\s+",
            header=None,
            engine="python",
        )
        frame.columns = make_columns(view_name, frame.shape[1])
        frames.append(frame)
        views[view_name] = frame.columns.tolist()

    X = pd.concat(frames, axis=1)
    y = pd.DataFrame({"label": np.repeat(np.arange(10), 200)})
    X, y, notes = clean_X_y(X, y, "uci_mfeat")
    return ProcessedDataset(X=X, y=y, views=views, notes=notes)


def parse_dry_bean(extract_dir: Path) -> ProcessedDataset:
    """Parse UCI Dry Bean from the Excel source file."""

    df = pd.read_excel(extract_dir / "DryBeanDataset" / "Dry_Bean_Dataset.xlsx")
    y = df[["Class"]].rename(columns={"Class": "label"})
    X = df.drop(columns=["Class"])
    X, y, notes = clean_X_y(X, y, "uci_dry_bean")
    return ProcessedDataset(
        X=X, y=y, views=make_auto_views(X.columns.tolist()), notes=notes
    )


def parse_gas_sensor_drift(extract_dir: Path) -> ProcessedDataset:
    """Parse UCI gas sensor drift batch files."""

    rows: list[dict[str, float]] = []
    labels: list[dict[str, object]] = []
    batch_files = sorted(
        extract_dir.glob("batch*.dat"),
        key=lambda path: int(path.stem.replace("batch", "")),
    )
    for batch_file in batch_files:
        batch = int(batch_file.stem.replace("batch", ""))
        for line in read_text(batch_file).splitlines():
            if not line.strip():
                continue
            head, *feature_tokens = line.strip().split()
            gas_label, concentration = head.split(";", maxsplit=1)
            row: dict[str, float] = {}
            for token in feature_tokens:
                index, value = token.split(":", maxsplit=1)
                row[f"sensor_{int(index):03d}"] = float(value)
            rows.append(row)
            labels.append(
                {
                    "label": int(gas_label),
                    "concentration": float(concentration),
                    "batch": batch,
                }
            )
    X = pd.DataFrame(rows).reindex(
        columns=[f"sensor_{index:03d}" for index in range(1, 129)]
    )
    y = pd.DataFrame(labels)
    X, y, notes = clean_X_y(X, y, "uci_gas_sensor_drift")
    return ProcessedDataset(
        X=X, y=y, views=make_auto_views(X.columns.tolist(), 8), notes=notes
    )


def parse_sensorless_drive(extract_dir: Path) -> ProcessedDataset:
    """Parse UCI Sensorless Drive Diagnosis."""

    df = pd.read_csv(
        extract_dir / "Sensorless_drive_diagnosis.txt",
        sep=r"\s+",
        header=None,
        engine="python",
    )
    X = df.iloc[:, :-1]
    X.columns = make_columns("drive", X.shape[1])
    y = pd.DataFrame({"label": df.iloc[:, -1].astype(int)})
    X, y, notes = clean_X_y(X, y, "uci_sensorless_drive")
    return ProcessedDataset(
        X=X, y=y, views=make_auto_views(X.columns.tolist(), 6), notes=notes
    )


def parse_har(extract_dir: Path) -> ProcessedDataset:
    """Parse UCI HAR train/test feature matrices."""

    nested_zip = extract_dir / "UCI HAR Dataset.zip"
    with ZipFile(nested_zip) as zip_file:
        features_text = zip_file.read("UCI HAR Dataset/features.txt").decode("utf-8")
        feature_names = [
            line.split(maxsplit=1)[1] for line in features_text.splitlines()
        ]
        feature_names = dedupe_columns(feature_names)

        def read_member(member: str) -> pd.DataFrame:
            """Read one whitespace-separated member from the nested HAR zip."""

            text = zip_file.read(member).decode("utf-8")
            return pd.read_csv(StringIO(text), sep=r"\s+", header=None, engine="python")

        X_train = read_member("UCI HAR Dataset/train/X_train.txt")
        X_test = read_member("UCI HAR Dataset/test/X_test.txt")
        y_train = read_member("UCI HAR Dataset/train/y_train.txt")
        y_test = read_member("UCI HAR Dataset/test/y_test.txt")

    X = pd.concat([X_train, X_test], ignore_index=True)
    X.columns = feature_names
    y = pd.DataFrame(
        {"label": pd.concat([y_train.iloc[:, 0], y_test.iloc[:, 0]], ignore_index=True)}
    )
    X, y, notes = clean_X_y(X, y, "uci_har")
    views = {
        "body_acceleration": [column for column in X.columns if "BodyAcc" in column],
        "gravity_acceleration": [
            column for column in X.columns if "GravityAcc" in column
        ],
        "body_gyroscope": [column for column in X.columns if "BodyGyro" in column],
        "other": [
            column
            for column in X.columns
            if all(
                token not in column for token in ("BodyAcc", "GravityAcc", "BodyGyro")
            )
        ],
    }
    views = {name: columns for name, columns in views.items() if columns}
    return ProcessedDataset(X=X, y=y, views=views, notes=notes)


def parse_landsat(extract_dir: Path) -> ProcessedDataset:
    """Parse UCI Statlog Landsat train/test files."""

    frames = []
    for filename in ("sat.trn", "sat.tst"):
        frames.append(
            pd.read_csv(
                extract_dir / filename, sep=r"\s+", header=None, engine="python"
            )
        )
    df = pd.concat(frames, ignore_index=True)
    X = df.iloc[:, :-1]
    X.columns = make_columns("satellite", X.shape[1])
    y = pd.DataFrame({"label": df.iloc[:, -1].astype(int)})
    X, y, notes = clean_X_y(X, y, "uci_landsat")
    views = {
        f"band_{band}": [
            column for idx, column in enumerate(X.columns) if idx % 4 == band - 1
        ]
        for band in range(1, 5)
    }
    return ProcessedDataset(X=X, y=y, views=views, notes=notes)


def read_segmentation_file(path: Path) -> pd.DataFrame:
    """Read one UCI image segmentation split."""

    lines = [
        line
        for line in read_text(path).splitlines()
        if line.strip() and not line.startswith(";;;")
    ]
    header = lines[0].split(",")
    rows = [line.split(",") for line in lines[1:]]
    return pd.DataFrame(rows, columns=["label", *header])


def parse_image_segmentation(extract_dir: Path) -> ProcessedDataset:
    """Parse UCI Image Segmentation train/test files."""

    df = pd.concat(
        [
            read_segmentation_file(extract_dir / "segmentation.data"),
            read_segmentation_file(extract_dir / "segmentation.test"),
        ],
        ignore_index=True,
    )
    y = df[["label"]]
    X = df.drop(columns=["label"])
    X, y, notes = clean_X_y(X, y, "uci_image_segmentation")
    return ProcessedDataset(
        X=X, y=y, views=make_auto_views(X.columns.tolist()), notes=notes
    )


def parse_wine_quality(extract_dir: Path) -> ProcessedDataset:
    """Parse UCI red and white Wine Quality files."""

    frames = []
    labels = []
    for wine_type in ("red", "white"):
        df = pd.read_csv(extract_dir / f"winequality-{wine_type}.csv", sep=";")
        frames.append(df.drop(columns=["quality"]))
        labels.append(
            pd.DataFrame({"label": df["quality"].astype(int), "wine_type": wine_type})
        )
    X = pd.concat(frames, ignore_index=True)
    y = pd.concat(labels, ignore_index=True)
    X, y, notes = clean_X_y(X, y, "uci_wine_quality")
    return ProcessedDataset(
        X=X, y=y, views=make_auto_views(X.columns.tolist()), notes=notes
    )


def parse_air_quality(extract_dir: Path) -> ProcessedDataset:
    """Parse UCI Air Quality and remove non-feature columns."""

    df = pd.read_csv(extract_dir / "AirQualityUCI.csv", sep=";", decimal=",")
    df = df.dropna(axis=1, how="all")
    drop_columns = [column for column in ("Date", "Time") if column in df.columns]
    X = df.drop(columns=drop_columns).replace(-200, np.nan)
    missing_ratio = X.isna().mean()
    high_missing = missing_ratio[missing_ratio > 0.5].index.tolist()
    if high_missing:
        X = X.drop(columns=high_missing)
    X, _, notes = clean_X_y(X, None, "uci_air_quality")
    if high_missing:
        notes.append(f"Dropped high-missing columns (>50%): {high_missing}.")
    return ProcessedDataset(
        X=X, y=None, views=make_auto_views(X.columns.tolist()), notes=notes
    )


def parse_beijing_air_quality(extract_dir: Path) -> ProcessedDataset:
    """Parse UCI Beijing Air Quality station files."""

    nested_zip = extract_dir / "PRSA2017_Data_20130301-20170228.zip"
    frames = []
    with ZipFile(nested_zip) as zip_file:
        for name in zip_file.namelist():
            if name.endswith(".csv"):
                with zip_file.open(name) as file:
                    frames.append(pd.read_csv(file))
    df = pd.concat(frames, ignore_index=True)
    y_columns = [
        column for column in ("station", "year", "month", "day", "hour") if column in df
    ]
    y = df[y_columns].copy() if y_columns else None
    drop_columns = [
        column
        for column in ("No", "year", "month", "day", "hour", "wd", "station")
        if column in df
    ]
    X = df.drop(columns=drop_columns)
    X, y, notes = clean_X_y(X, y, "uci_beijing_air_quality")
    notes.append("Dropped temporal identifiers, station, and wind direction from X.")
    return ProcessedDataset(
        X=X, y=y, views=make_auto_views(X.columns.tolist()), notes=notes
    )


def parse_seeds(extract_dir: Path) -> ProcessedDataset:
    """Parse UCI Seeds."""

    df = pd.read_csv(
        extract_dir / "seeds_dataset.txt", sep=r"\s+", header=None, engine="python"
    )
    X = df.iloc[:, :7]
    X.columns = make_columns("seed", X.shape[1])
    y = pd.DataFrame({"label": df.iloc[:, 7].astype(int)})
    X, y, notes = clean_X_y(X, y, "uci_seeds")
    return ProcessedDataset(
        X=X, y=y, views=make_auto_views(X.columns.tolist()), notes=notes
    )


def parse_wine(extract_dir: Path) -> ProcessedDataset:
    """Parse UCI Wine."""

    df = read_csv_no_header(extract_dir / "wine.data")
    y = pd.DataFrame({"label": df.iloc[:, 0].astype(int)})
    X = df.iloc[:, 1:]
    X.columns = make_columns("wine", X.shape[1])
    X, y, notes = clean_X_y(X, y, "uci_wine")
    return ProcessedDataset(
        X=X, y=y, views=make_auto_views(X.columns.tolist()), notes=notes
    )


def parse_wdbc(extract_dir: Path) -> ProcessedDataset:
    """Parse UCI Breast Cancer Wisconsin Diagnostic."""

    df = read_csv_no_header(extract_dir / "wdbc.data")
    y = pd.DataFrame({"label": df.iloc[:, 1], "id": df.iloc[:, 0]})
    X = df.iloc[:, 2:]
    X.columns = make_columns("wdbc", X.shape[1])
    X, y, notes = clean_X_y(X, y, "uci_wdbc")
    return ProcessedDataset(
        X=X, y=y, views=make_auto_views(X.columns.tolist()), notes=notes
    )


def parse_ionosphere(extract_dir: Path) -> ProcessedDataset:
    """Parse UCI Ionosphere."""

    df = read_csv_no_header(extract_dir / "ionosphere.data")
    y = pd.DataFrame({"label": df.iloc[:, -1]})
    X = df.iloc[:, :-1]
    X.columns = make_columns("ionosphere", X.shape[1])
    X, y, notes = clean_X_y(X, y, "uci_ionosphere")
    return ProcessedDataset(
        X=X, y=y, views=make_auto_views(X.columns.tolist()), notes=notes
    )


def parse_sonar(extract_dir: Path) -> ProcessedDataset:
    """Parse UCI Sonar."""

    df = read_csv_no_header(extract_dir / "sonar.all-data")
    y = pd.DataFrame({"label": df.iloc[:, -1]})
    X = df.iloc[:, :-1]
    X.columns = make_columns("sonar", X.shape[1])
    X, y, notes = clean_X_y(X, y, "uci_sonar")
    return ProcessedDataset(
        X=X, y=y, views=make_auto_views(X.columns.tolist(), 6), notes=notes
    )


def parse_parkinsons(extract_dir: Path) -> ProcessedDataset:
    """Parse UCI Parkinsons voice measurements."""

    df = pd.read_csv(extract_dir / "parkinsons.data")
    y = pd.DataFrame({"label": df["status"], "name": df["name"]})
    X = df.drop(columns=["name", "status"])
    X, y, notes = clean_X_y(X, y, "uci_parkinsons")
    return ProcessedDataset(
        X=X, y=y, views=make_auto_views(X.columns.tolist()), notes=notes
    )


def parse_glass(extract_dir: Path) -> ProcessedDataset:
    """Parse UCI Glass Identification."""

    df = read_csv_no_header(extract_dir / "glass.data")
    y = pd.DataFrame({"label": df.iloc[:, -1].astype(int), "id": df.iloc[:, 0]})
    X = df.iloc[:, 1:-1]
    X.columns = make_columns("glass", X.shape[1])
    X, y, notes = clean_X_y(X, y, "uci_glass")
    return ProcessedDataset(
        X=X, y=y, views=make_auto_views(X.columns.tolist()), notes=notes
    )


def parse_libras(extract_dir: Path) -> ProcessedDataset:
    """Parse UCI Libras Movement."""

    df = read_csv_no_header(extract_dir / "movement_libras.data")
    y = pd.DataFrame({"label": df.iloc[:, -1].astype(int)})
    X = df.iloc[:, :-1]
    X.columns = make_columns("libras", X.shape[1])
    X, y, notes = clean_X_y(X, y, "uci_libras")
    return ProcessedDataset(
        X=X, y=y, views=make_auto_views(X.columns.tolist(), 6), notes=notes
    )


def parse_waveform_raw_only(extract_dir: Path) -> ProcessedDataset:
    """Mark UCI Waveform as raw-only because data are legacy .Z files."""

    del extract_dir
    raise NotImplementedError(
        "UCI Waveform ships data as legacy .Z files. Raw files are downloaded; "
        "processed conversion is intentionally skipped until a pinned .Z decoder "
        "or generated-data policy is added."
    )


PARSERS = {
    "mfeat": parse_mfeat,
    "dry_bean": parse_dry_bean,
    "gas_sensor_drift": parse_gas_sensor_drift,
    "sensorless_drive": parse_sensorless_drive,
    "har": parse_har,
    "landsat": parse_landsat,
    "image_segmentation": parse_image_segmentation,
    "wine_quality": parse_wine_quality,
    "air_quality": parse_air_quality,
    "beijing_air_quality": parse_beijing_air_quality,
    "seeds": parse_seeds,
    "wine": parse_wine,
    "wdbc": parse_wdbc,
    "ionosphere": parse_ionosphere,
    "sonar": parse_sonar,
    "parkinsons": parse_parkinsons,
    "glass": parse_glass,
    "libras": parse_libras,
    "waveform_raw_only": parse_waveform_raw_only,
}


def write_json(path: Path, payload: dict) -> None:
    """Write a JSON file with stable formatting."""

    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_processed_dataset(
    dataset: dict,
    processed: ProcessedDataset,
    processed_dir: Path,
) -> None:
    """Write X, y, views, and metadata for one dataset."""

    processed_dir.mkdir(parents=True, exist_ok=True)
    processed.X.to_csv(processed_dir / "X.csv", index=False)
    if processed.y is not None:
        processed.y.to_csv(processed_dir / "y.csv", index=False)

    write_json(processed_dir / "views.json", processed.views)
    metadata = {
        "name": dataset["name"],
        "title": dataset["title"],
        "source": dataset["source"],
        "source_url": dataset["url"],
        "parser": dataset["parser"],
        "priority": dataset["priority"],
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "n_samples": int(processed.X.shape[0]),
        "n_features": int(processed.X.shape[1]),
        "n_clusters": dataset["n_clusters"],
        "label_available": processed.y is not None,
        "x_file": "X.csv",
        "y_file": "y.csv" if processed.y is not None else None,
        "views_file": "views.json",
        "all_features_numeric": True,
        "notes": processed.notes,
    }
    write_json(processed_dir / "metadata.json", metadata)


def write_raw_metadata(
    dataset: dict, raw_dir: Path, status: str, message: str | None
) -> None:
    """Write source download metadata for one raw dataset."""

    payload = {
        "name": dataset["name"],
        "title": dataset["title"],
        "source": dataset["source"],
        "source_url": dataset["url"],
        "parser": dataset["parser"],
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "message": message,
    }
    write_json(raw_dir / "raw_metadata.json", payload)


def process_dataset(
    dataset: dict,
    raw_root: Path,
    processed_root: Path,
    raw_only: bool,
    force: bool,
) -> dict[str, object]:
    """Download, extract, and optionally process one registry entry."""

    name = dataset["name"]
    raw_dir = raw_root / name
    extract_dir = raw_dir / "extracted"
    processed_dir = processed_root / name
    archive_path = raw_dir / "source.zip"

    logger.info("Fetching {}", name)
    download_file(dataset["url"], archive_path, force=force)
    extract_zip(archive_path, extract_dir, force=force)
    write_raw_metadata(dataset, raw_dir, status="downloaded", message=None)

    if raw_only:
        return {
            "name": name,
            "status": "raw_only",
            "n_samples": None,
            "n_features": None,
        }

    if processed_dir.exists() and not force and (processed_dir / "X.csv").exists():
        metadata_path = processed_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return {
            "name": name,
            "status": "exists",
            "n_samples": metadata["n_samples"],
            "n_features": metadata["n_features"],
        }

    if processed_dir.exists():
        safe_rmtree(processed_dir, processed_root)

    parser = PARSERS[dataset["parser"]]
    try:
        processed = parser(extract_dir)
    except NotImplementedError as error:
        write_raw_metadata(dataset, raw_dir, status="raw_only", message=str(error))
        return {
            "name": name,
            "status": "raw_only",
            "n_samples": None,
            "n_features": None,
            "message": str(error),
        }

    write_processed_dataset(dataset, processed, processed_dir)
    return {
        "name": name,
        "status": "processed",
        "n_samples": int(processed.X.shape[0]),
        "n_features": int(processed.X.shape[1]),
    }


def main() -> None:
    """Run the public dataset fetch workflow."""

    args = parse_args()
    registry = load_registry(args.config)
    raw_root = (PROJECT_DIR / registry["raw_root"]).resolve()
    processed_root = (PROJECT_DIR / registry["processed_root"]).resolve()

    selected = set(args.datasets) if args.datasets else None
    datasets = [
        dataset
        for dataset in registry["datasets"]
        if selected is None or dataset["name"] in selected
    ]
    missing = (
        selected - {dataset["name"] for dataset in datasets} if selected else set()
    )
    if missing:
        msg = f"Unknown dataset names: {sorted(missing)}"
        raise ValueError(msg)

    summaries = []
    for dataset in datasets:
        summary = process_dataset(
            dataset=dataset,
            raw_root=raw_root,
            processed_root=processed_root,
            raw_only=args.raw_only,
            force=args.force,
        )
        summaries.append(summary)
        size = ""
        if summary["n_samples"] is not None:
            size = f" ({summary['n_samples']} x {summary['n_features']})"
        logger.info("{}{}", summary["status"], size)

    summary_path = processed_root / "public_datasets_summary.csv"
    processed_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    logger.info("Saved summary: {}", summary_path)


if __name__ == "__main__":
    main()
