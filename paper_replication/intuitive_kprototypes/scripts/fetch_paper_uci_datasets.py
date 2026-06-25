"""Fetch and preprocess the UCI datasets used by the paper.

Run from the repository root:

    uv run --link-mode=copy --with ucimlrepo python \
        paper_replication/intuitive_kprototypes/scripts/fetch_paper_uci_datasets.py

The script keeps this replication sandbox independent from the main
``mvdec_fpmax`` pipeline. It writes raw and processed CSV files under
``paper_replication/intuitive_kprototypes/data/uci``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

try:
    from ucimlrepo import fetch_ucirepo
except ImportError as exc:  # pragma: no cover - exercised only by users locally.
    msg = (
        "Missing optional dependency 'ucimlrepo'. Run this script with:\n"
        "uv run --link-mode=copy --with ucimlrepo python "
        "paper_replication/intuitive_kprototypes/scripts/"
        "fetch_paper_uci_datasets.py"
    )
    raise SystemExit(msg) from exc


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "uci"


@dataclass(frozen=True)
class ProcessedDataset:
    key: str
    source_name: str
    processed_csv: str
    raw_csv: str
    n_samples: int
    n_clusters: int
    numeric_cols: list[str]
    categorical_cols: list[str]
    target_col: str
    target_counts: dict[str, int]
    notes: list[str]


def _fetch_frames(name: str) -> tuple[pd.DataFrame, pd.Series]:
    dataset = fetch_ucirepo(name=name)
    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()
    if targets is None or targets.shape[1] != 1:
        msg = f"Expected one target column for {name!r}."
        raise ValueError(msg)
    return features, targets.iloc[:, 0].copy()


def _write_raw(
    raw_dir: Path,
    key: str,
    features: pd.DataFrame,
    target: pd.Series,
) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw = features.copy()
    raw["target"] = target.to_numpy()
    path = raw_dir / f"{key}.csv"
    raw.to_csv(path, index=False)
    return path


def _fill_missing_with_mode(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for col in result.columns:
        if not result[col].isna().any():
            continue
        mode = result[col].mode(dropna=True)
        if mode.empty:
            msg = f"Column {col!r} contains only missing values."
            raise ValueError(msg)
        result[col] = result[col].fillna(mode.iloc[0])
    return result


def _normalise_numeric(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    result = df.copy()
    if numeric_cols:
        result[numeric_cols] = result[numeric_cols].apply(pd.to_numeric)
        result[numeric_cols] = MinMaxScaler().fit_transform(result[numeric_cols])
    return result


def _stringify_categorical(
    df: pd.DataFrame,
    categorical_cols: list[str],
) -> pd.DataFrame:
    result = df.copy()
    for col in categorical_cols:
        result[col] = result[col].astype(str)
    return result


def _save_processed(
    *,
    output_dir: Path,
    key: str,
    source_name: str,
    features: pd.DataFrame,
    target: pd.Series,
    numeric_cols: list[str],
    categorical_cols: list[str],
    n_clusters: int,
    notes: list[str] | None = None,
) -> ProcessedDataset:
    notes = list(notes or [])
    raw_path = _write_raw(output_dir / "raw", key, features, target)

    used_cols = numeric_cols + categorical_cols
    processed = features[used_cols].copy()
    processed = _fill_missing_with_mode(processed)
    processed = _normalise_numeric(processed, numeric_cols)
    processed = _stringify_categorical(processed, categorical_cols)
    processed["target"] = target.to_numpy()

    processed_dir = output_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_path = processed_dir / f"{key}.csv"
    processed.to_csv(processed_path, index=False)

    target_counts = {
        str(label): int(count)
        for label, count in processed["target"].value_counts(dropna=False).items()
    }
    meta = ProcessedDataset(
        key=key,
        source_name=source_name,
        processed_csv=str(processed_path.relative_to(output_dir)),
        raw_csv=str(raw_path.relative_to(output_dir)),
        n_samples=int(processed.shape[0]),
        n_clusters=n_clusters,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        target_col="target",
        target_counts=target_counts,
        notes=notes,
    )
    with (processed_dir / f"{key}.meta.json").open("w", encoding="utf-8") as file:
        json.dump(asdict(meta), file, indent=2, sort_keys=True)
        file.write("\n")
    return meta


def _constant_columns(df: pd.DataFrame) -> list[str]:
    counts = df.nunique(dropna=False)
    return counts[counts <= 1].index.tolist()


def prepare_iris(output_dir: Path) -> ProcessedDataset:
    source = "Iris"
    features, target = _fetch_frames(source)
    return _save_processed(
        output_dir=output_dir,
        key="iris",
        source_name=source,
        features=features,
        target=target,
        numeric_cols=features.columns.tolist(),
        categorical_cols=[],
        n_clusters=3,
        notes=["Matches Table 10: 4 numerical, 0 categorical, 150 objects."],
    )


def prepare_ionosphere(output_dir: Path) -> ProcessedDataset:
    source = "Ionosphere"
    features, target = _fetch_frames(source)
    constants = _constant_columns(features)
    cleaned = features.drop(columns=constants)
    return _save_processed(
        output_dir=output_dir,
        key="ionosphere",
        source_name=source,
        features=cleaned,
        target=target,
        numeric_cols=cleaned.columns.tolist(),
        categorical_cols=[],
        n_clusters=2,
        notes=[
            "Attribute2 is constant in the UCI frame and is removed to match "
            "Table 10: 33 numerical attributes.",
        ],
    )


def prepare_bcw(output_dir: Path) -> ProcessedDataset:
    source = "Breast Cancer Wisconsin (Original)"
    features, target = _fetch_frames(source)
    return _save_processed(
        output_dir=output_dir,
        key="bcw",
        source_name=source,
        features=features,
        target=target,
        numeric_cols=[],
        categorical_cols=features.columns.tolist(),
        n_clusters=2,
        notes=[
            "Integer-valued attributes are treated as categorical to match "
            "Table 10: 0 numerical, 9 categorical.",
            "Missing Bare_nuclei values are filled with the mode, as described "
            "in the paper.",
        ],
    )


def prepare_soybean(output_dir: Path) -> list[ProcessedDataset]:
    source = "Soybean (Small)"
    features, target = _fetch_frames(source)
    constants = _constant_columns(features)
    table_variant = _save_processed(
        output_dir=output_dir,
        key="soybean_small",
        source_name=source,
        features=features,
        target=target,
        numeric_cols=[],
        categorical_cols=features.columns.tolist(),
        n_clusters=4,
        notes=[
            "Kept all 35 categorical attributes to match Table 10.",
            "The current UCI frame contains constant columns: "
            + ", ".join(constants)
            + ". The paper also says single-value attributes are removed, so "
            "this is a documented paper/table inconsistency to check later.",
        ],
    )
    cleaned = features.drop(columns=constants)
    no_constants_variant = _save_processed(
        output_dir=output_dir,
        key="soybean_small_no_constants",
        source_name=source,
        features=cleaned,
        target=target,
        numeric_cols=[],
        categorical_cols=cleaned.columns.tolist(),
        n_clusters=4,
        notes=[
            "Removes constant columns from the current UCI frame, following "
            "the preprocessing sentence in the paper.",
            "This does not match Table 10's printed count of 35 categorical "
            "attributes; it keeps 21 categorical attributes.",
        ],
    )
    return [table_variant, no_constants_variant]


def prepare_australian(output_dir: Path) -> ProcessedDataset:
    source = "Statlog (Australian Credit Approval)"
    features, target = _fetch_frames(source)
    numeric_cols = ["A2", "A3", "A7", "A10", "A13", "A14"]
    categorical_cols = ["A1", "A4", "A5", "A6", "A8", "A9", "A11", "A12"]
    return _save_processed(
        output_dir=output_dir,
        key="australian_credit_approval",
        source_name=source,
        features=features,
        target=target,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        n_clusters=2,
        notes=["Matches Table 10: 6 numerical, 8 categorical, 690 objects."],
    )


def prepare_chess(output_dir: Path) -> ProcessedDataset:
    source = "Chess (King-Rook vs. King)"
    features, target = _fetch_frames(source)
    mask = target.isin(["ten", "fifteen"])
    filtered_features = features.loc[mask].reset_index(drop=True)
    filtered_target = target.loc[mask].reset_index(drop=True)
    numeric_cols = ["white-king-rank", "white-rook-rank", "black-king-rank"]
    categorical_cols = ["white-king-file", "white-rook-file", "black-king-file"]
    return _save_processed(
        output_dir=output_dir,
        key="chess_ten_fifteen",
        source_name=source,
        features=filtered_features,
        target=filtered_target,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        n_clusters=2,
        notes=[
            "Filtered target labels to 'ten' and 'fifteen' as described by "
            "the paper. This yields 4151 objects.",
        ],
    )


def prepare_heart(output_dir: Path) -> list[ProcessedDataset]:
    source = "Heart Disease"
    features, target = _fetch_frames(source)
    numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    categorical_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

    heart_5 = _save_processed(
        output_dir=output_dir,
        key="heart_cleveland_5",
        source_name=source,
        features=features,
        target=target,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        n_clusters=5,
        notes=[
            "Uses original Cleveland target labels 0..4.",
            "The 'ca' attribute is treated as categorical so the feature split "
            "matches Table 10: 5 numerical and 8 categorical.",
        ],
    )

    binary_target = target.apply(lambda value: 0 if int(value) == 0 else 1)
    heart_2 = _save_processed(
        output_dir=output_dir,
        key="heart_cleveland_2",
        source_name=source,
        features=features,
        target=binary_target,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        n_clusters=2,
        notes=[
            "Converts Cleveland target to absence/presence: 0 versus 1..4.",
            "The 'ca' attribute is treated as categorical so the feature split "
            "matches Table 10: 5 numerical and 8 categorical.",
        ],
    )
    return [heart_2, heart_5]


PREPARERS: dict[str, Any] = {
    "iris": prepare_iris,
    "ionosphere": prepare_ionosphere,
    "bcw": prepare_bcw,
    "soybean_small": prepare_soybean,
    "australian_credit_approval": prepare_australian,
    "chess_ten_fifteen": prepare_chess,
    "heart": prepare_heart,
}


def _normalise_result(
    result: ProcessedDataset | list[ProcessedDataset],
) -> list[ProcessedDataset]:
    if isinstance(result, list):
        return result
    return [result]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for raw, processed, and manifest files.",
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(PREPARERS),
        action="append",
        help=(
            "Dataset key to fetch. May be repeated. Defaults to all Table 10 "
            "datasets."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = args.dataset or list(PREPARERS)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    for key in selected:
        for item in _normalise_result(PREPARERS[key](output_dir)):
            manifest.append(asdict(item))
            print(
                f"{item.key}: {item.n_samples} rows, "
                f"{len(item.numeric_cols)} numeric, "
                f"{len(item.categorical_cols)} categorical -> "
                f"{item.processed_csv}"
            )

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
        file.write("\n")
    print(f"Wrote manifest: {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
