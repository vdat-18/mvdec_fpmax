"""Read and write CSV experiment outputs."""

from pathlib import Path
from time import sleep
from typing import Any

import pandas as pd
from loguru import logger


def result_csv_path(save_path: Path) -> Path:
    """Return the CSV path used for one experiment output."""

    return save_path if save_path.suffix == ".csv" else save_path.with_suffix(".csv")


def empty_results(columns: list[str]) -> pd.DataFrame:
    """Create an empty result dataframe with expected columns."""

    return pd.DataFrame(columns=columns)


def load_results(save_path: Path, columns: list[str]) -> pd.DataFrame:
    """Load existing CSV output when resuming an experiment."""

    csv_path = result_csv_path(save_path)
    if csv_path.exists():
        if csv_path.stat().st_size == 0:
            logger.warning("Ignoring empty output CSV while resuming: {}", csv_path)
            return empty_results(columns)
        return pd.read_csv(csv_path)

    return empty_results(columns)


def is_job_done(existing_df: pd.DataFrame, job_index: int) -> bool:
    """Return whether a job index is already present in existing results."""

    if existing_df.empty or "job_index" not in existing_df.columns:
        return False

    return bool((existing_df["job_index"].astype(int) == job_index).any())


def replace_with_retry(tmp_path: Path, csv_path: Path) -> None:
    """Replace a CSV path, retrying transient Windows file locks."""

    attempts = 5
    for attempt in range(attempts):
        try:
            tmp_path.replace(csv_path)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            sleep(0.25 * (attempt + 1))


def save_results(
    existing_df: pd.DataFrame,
    records: dict[int, Any],
    save_path: Path,
    label: str,
    columns: list[str],
    dedupe_columns: str | list[str] = 'job_index',
) -> pd.DataFrame:
    """Save experiment records to CSV using the requested schema."""

    new_df = pd.DataFrame(
        [
            {column: getattr(record, column, None) for column in columns}
            for record in records.values()
        ]
    )
    if not new_df.empty:
        new_df = new_df.reindex(columns=columns)

    result_df = pd.concat([existing_df, new_df], ignore_index=True)
    if not result_df.empty:
        result_df = result_df.drop_duplicates(dedupe_columns, keep="last")
        result_df = result_df.sort_values(dedupe_columns).reset_index(drop=True)
        result_df = result_df.reindex(columns=columns)

    csv_path = result_csv_path(save_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = csv_path.with_name(f"{csv_path.name}.tmp")
    result_df.to_csv(tmp_path, index=False)
    replace_with_retry(tmp_path, csv_path)
    logger.info("Saved {} results to {}", label, csv_path)

    return result_df
