"""Command-line entrypoint for Tiki preprocessing."""

import argparse
from pathlib import Path

from loguru import logger

from data_preprocessing.pipeline import (
    DEFAULT_OUTPUT_PATH,
    DEFAULT_RAW_DATA_PATH,
    preprocess_tiki_data,
)


def parse_args() -> argparse.Namespace:
    """Parse preprocessing command-line arguments."""

    parser = argparse.ArgumentParser(description="Preprocess raw Tiki seller data.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_RAW_DATA_PATH,
        help="Raw Tiki data path (.xlsx, .xls, or .csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Preprocessed output path (.csv or .xlsx).",
    )
    parser.add_argument(
        "--reference-year",
        type=int,
        default=2026,
        help="Reference year used to compute Years Joined.",
    )
    parser.add_argument(
        "--dbscan-eps",
        type=float,
        default=1.05,
        help="DBSCAN eps parameter from the legacy notebook.",
    )
    parser.add_argument(
        "--dbscan-min-samples",
        type=int,
        default=14,
        help="DBSCAN min_samples parameter from the legacy notebook.",
    )
    return parser.parse_args()


def main() -> None:
    """Run preprocessing and log a compact summary."""

    args = parse_args()
    try:
        result = preprocess_tiki_data(
            input_path=args.input,
            output_path=args.output,
            reference_year=args.reference_year,
            dbscan_eps=args.dbscan_eps,
            dbscan_min_samples=args.dbscan_min_samples,
        )
    except (FileNotFoundError, ValueError) as error:
        logger.error("{}", error)
        raise SystemExit(1) from error

    logger.info("Raw shape: {}", result.raw_df.shape)
    logger.info("Selected feature shape: {}", result.selected_features_df.shape)
    logger.info("DBSCAN clusters: {}", result.dbscan.clusters_info)
    logger.info("Saved preprocessed data: {}", args.output)
    logger.info("Output shape: {}", result.output_df.shape)


if __name__ == "__main__":
    main()
