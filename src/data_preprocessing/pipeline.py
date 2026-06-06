"""End-to-end Tiki preprocessing pipeline."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from config import DATA_DIR, PREPROCESSED_DATA_DIR
from data_preprocessing.features import (
    add_wilson_features,
    add_years_joined,
    rename_to_model_features,
    select_preprocessing_features,
)
from data_preprocessing.io import read_raw_tiki_data, save_preprocessed_data
from data_preprocessing.transforms import (
    DbscanResult,
    remove_dbscan_noise,
    standardize_features,
    winsorize_preprocessed_features,
)

DEFAULT_RAW_DATA_PATH = DATA_DIR / "raw_data" / "tiki_raw_data.xlsx"
DEFAULT_OUTPUT_PATH = PREPROCESSED_DATA_DIR / "tiki_preprocessed.csv"


@dataclass(frozen=True)
class PreprocessingResult:
    """Intermediate and final outputs from Tiki preprocessing."""

    raw_df: pd.DataFrame
    engineered_df: pd.DataFrame
    selected_features_df: pd.DataFrame
    scaled_df: pd.DataFrame
    winsorized_df: pd.DataFrame
    dbscan: DbscanResult
    output_df: pd.DataFrame


def preprocess_tiki_dataframe(
    raw_df: pd.DataFrame,
    reference_year: int = 2026,
    dbscan_eps: float = 1.05,
    dbscan_min_samples: int = 14,
) -> PreprocessingResult:
    """Run the notebook preprocessing steps on an in-memory dataframe."""

    with_years = add_years_joined(raw_df, reference_year=reference_year)
    engineered_df = add_wilson_features(with_years)
    selected_features_df = select_preprocessing_features(engineered_df)
    scaled_df = standardize_features(selected_features_df)
    winsorized_df = winsorize_preprocessed_features(scaled_df)
    winsorized_df = winsorized_df.dropna()
    dbscan = remove_dbscan_noise(
        winsorized_df,
        eps=dbscan_eps,
        min_samples=dbscan_min_samples,
    )

    output_df = rename_to_model_features(dbscan.non_noise_df).reset_index(drop=True)
    return PreprocessingResult(
        raw_df=raw_df,
        engineered_df=engineered_df,
        selected_features_df=selected_features_df,
        scaled_df=scaled_df,
        winsorized_df=winsorized_df,
        dbscan=dbscan,
        output_df=output_df,
    )


def preprocess_tiki_data(
    input_path: Path = DEFAULT_RAW_DATA_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    reference_year: int = 2026,
    dbscan_eps: float = 1.05,
    dbscan_min_samples: int = 14,
) -> PreprocessingResult:
    """Read raw Tiki data, preprocess it, and save the final dataset."""

    raw_df = read_raw_tiki_data(input_path)
    result = preprocess_tiki_dataframe(
        raw_df=raw_df,
        reference_year=reference_year,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
    )
    save_preprocessed_data(result.output_df, output_path)
    return result
