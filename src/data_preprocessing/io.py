"""Input and output helpers for Tiki preprocessing."""

from pathlib import Path

import pandas as pd


def read_raw_tiki_data(path: Path) -> pd.DataFrame:
    """Read the raw Tiki seller dataset from Excel or CSV."""

    if not path.exists():
        msg = f"Raw data file does not exist: {path}"
        raise FileNotFoundError(msg)

    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    msg = f"Unsupported raw data format: {path.suffix}"
    raise ValueError(msg)


def save_preprocessed_data(df: pd.DataFrame, output_path: Path) -> None:
    """Save preprocessed data to CSV or Excel based on the file extension."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    if suffix == ".csv":
        df.to_csv(output_path, index=False, encoding="utf-8")
        return

    if suffix in {".xlsx", ".xls"}:
        df.to_excel(output_path, index=False)
        return

    msg = f"Unsupported output format: {output_path.suffix}"
    raise ValueError(msg)
