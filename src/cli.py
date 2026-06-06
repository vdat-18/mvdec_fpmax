"""Command-line entrypoint for smoke tests and MiMvDEC experiments."""

import argparse
import warnings

import numpy as np
from loguru import logger

from config import RANDOM_STATE
from pipeline.clustering import run_kprototypes
from pipeline.data import load_mvdec_result, load_preprocessed_dataset
from pipeline.experiments import run_ffs, run_without_ffs
from pipeline.forward_selection import run_forward_selection
from pipeline.fpmax import extract_fpmax_features


def run_smoke_test() -> None:
    """Run the data loading smoke test."""

    dataset = load_preprocessed_dataset()
    mvdec_result = load_mvdec_result()

    logger.info("Data shape: {}", dataset.X.shape)
    logger.info("Input dimension: {}", dataset.input_dim)
    logger.info("Loaded MvDEC result")
    logger.info("h_fused dataframe shape: {}", mvdec_result.h_fused_df.shape)
    logger.info("Iteration: {}", mvdec_result.iteration)
    logger.info("Init method: {}", mvdec_result.init)
    logger.info("Silhouette (saved): {:.4f}", mvdec_result.score)
    logger.info("Silhouette (recomputed): {:.4f}", mvdec_result.score_check)


def run_without_ffs_smoke_test() -> None:
    """Run one without-FFS configuration for a quick correctness check."""

    mvdec_result = load_mvdec_result()
    fpmax_features = extract_fpmax_features(
        df=mvdec_result.h_fused_df,
        n_bins=7,
        strategy="kmeans",
        min_support=0.2,
    )
    clustering = run_kprototypes(
        continuous_df=mvdec_result.h_fused_df,
        binary_df=fpmax_features.features,
    )

    logger.info("Itemsets found: {}", len(fpmax_features.itemsets))
    logger.info("Binary features used: {}", len(clustering.binary_feature_names))
    logger.info("Best init: {}", clustering.init)
    logger.info("Final silhouette: {:.4f}", clustering.score)


def run_ffs_smoke_test() -> None:
    """Run one FFS configuration for a quick correctness check."""

    mvdec_result = load_mvdec_result()
    fpmax_features = extract_fpmax_features(
        df=mvdec_result.h_fused_df,
        n_bins=7,
        strategy="kmeans",
        min_support=0.35,
    )
    selected = run_forward_selection(
        continuous_df=mvdec_result.h_fused_df,
        binary_df=fpmax_features.features,
        baseline_score=mvdec_result.score,
    )

    logger.info("Itemsets found: {}", len(fpmax_features.itemsets))
    logger.info("Selected features: {}", len(selected.selected_feature_names))
    logger.info("Best init: {}", selected.init)
    logger.info("Final silhouette: {:.4f}", selected.score)
    logger.info("Best observed silhouette: {:.4f}", selected.best_observed_score)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=[
            "smoke",
            "without-ffs-smoke",
            "ffs-smoke",
            "without-ffs",
            "ffs",
        ],
        nargs="?",
        default="smoke",
        help="Which workflow to run.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Number of parallel worker processes for grid modes.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N jobs for quick checks.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing output files and rerun selected jobs.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the selected command-line workflow."""

    warnings.filterwarnings("ignore")
    np.random.seed(RANDOM_STATE)

    args = parse_args()
    mvdec_result = None

    if args.mode == "smoke":
        run_smoke_test()
    elif args.mode == "without-ffs-smoke":
        run_without_ffs_smoke_test()
    elif args.mode == "ffs-smoke":
        run_ffs_smoke_test()
    elif args.mode == "without-ffs":
        mvdec_result = load_mvdec_result()
        run_without_ffs(
            h_fused_df=mvdec_result.h_fused_df,
            baseline_score=mvdec_result.score,
            workers=args.workers,
            limit=args.limit,
            resume=not args.no_resume,
        )
    elif args.mode == "ffs":
        mvdec_result = load_mvdec_result()
        run_ffs(
            h_fused_df=mvdec_result.h_fused_df,
            baseline_score=mvdec_result.score,
            workers=args.workers,
            limit=args.limit,
            resume=not args.no_resume,
        )


if __name__ == "__main__":
    main()
