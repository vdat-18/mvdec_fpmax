"""Read and write experiment outputs."""

from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger


def result_paths(save_path: Path) -> tuple[Path, Path]:
    """Return CSV and JSONL paths for an experiment result stem."""

    return save_path.with_suffix(".csv"), save_path.with_suffix(".jsonl")


def empty_results(columns: list[str]) -> pd.DataFrame:
    """Create an empty result dataframe with expected columns."""

    return pd.DataFrame(columns=columns)


def load_results(save_path: Path, columns: list[str]) -> pd.DataFrame:
    """Load existing CSV output when resuming an experiment."""

    csv_path, _ = result_paths(save_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)

    return empty_results(columns)


def is_job_done(existing_df: pd.DataFrame, job_index: int) -> bool:
    """Return whether a job index is already present in existing results."""

    if existing_df.empty or "job_index" not in existing_df.columns:
        return False

    return bool((existing_df["job_index"].astype(int) == job_index).any())


def save_results(
    existing_df: pd.DataFrame,
    records: dict[int, Any],
    save_path: Path,
    label: str,
) -> pd.DataFrame:
    """Save experiment records to CSV and JSONL."""

    result_df = pd.concat(
        [existing_df, pd.DataFrame([asdict(record) for record in records.values()])],
        ignore_index=True,
    )
    if not result_df.empty:
        result_df = result_df.drop_duplicates("job_index", keep="last")
        result_df = result_df.sort_values("job_index").reset_index(drop=True)

    csv_path, jsonl_path = result_paths(save_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(csv_path, index=False)
    result_df.to_json(jsonl_path, orient="records", lines=True)
    logger.info("Saved {} results to {} and {}", label, csv_path, jsonl_path)

    return result_df
