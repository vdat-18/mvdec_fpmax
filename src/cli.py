"""Command-line entrypoint for smoke tests and MiMvDEC experiments."""

import argparse
import warnings
from pathlib import Path

import numpy as np
from loguru import logger

from config import FUSED_REPRESENTATION_PATH, PREPROCESSED_DATA_PATH, RANDOM_STATE
from pipeline.clustering import (
    ClusteringBackend,
    run_intuitive_kprototypes,
    run_kprototypes,
)
from pipeline.data import load_mvdec_result, load_preprocessed_dataset
from pipeline.experiments import (
    run_ffs,
    run_ffs_intuitive_exhaustive_native,
    run_ffs_intuitive_native,
    run_ffs_intuitive_paper_exhaustive_native,
    run_ffs_intuitive_paper_native,
    run_ffs_intuitive_view_weighted_exhaustive_native,
    run_ffs_intuitive_view_weighted_native,
    run_ffs_intuitive_view_weighted_paper_exhaustive_native,
    run_ffs_intuitive_view_weighted_paper_native,
    run_post_ffs_intuitive,
    run_post_ffs_intuitive_best,
    run_post_ffs_intuitive_exhaustive_best,
    run_post_ffs_intuitive_view_weighted,
    run_post_ffs_intuitive_view_weighted_best,
    run_post_ffs_intuitive_view_weighted_exhaustive_best,
    run_post_without_ffs_intuitive,
    run_post_without_ffs_intuitive_best,
    run_post_without_ffs_intuitive_exhaustive_best,
    run_post_without_ffs_intuitive_view_weighted,
    run_post_without_ffs_intuitive_view_weighted_best,
    run_post_without_ffs_intuitive_view_weighted_exhaustive_best,
    run_without_ffs,
    run_without_ffs_intuitive_exhaustive_native,
    run_without_ffs_intuitive_native,
    run_without_ffs_intuitive_paper_exhaustive_native,
    run_without_ffs_intuitive_paper_native,
    run_without_ffs_intuitive_view_weighted_exhaustive_native,
    run_without_ffs_intuitive_view_weighted_native,
    run_without_ffs_intuitive_view_weighted_paper_exhaustive_native,
    run_without_ffs_intuitive_view_weighted_paper_native,
)
from pipeline.forward_selection import run_forward_selection
from pipeline.fpmax import extract_fpmax_features


def run_smoke_test(data_path, representation_path) -> None:
    """Run the data loading smoke test."""

    dataset = load_preprocessed_dataset(data_path)
    mvdec_result = load_mvdec_result(representation_path, data_path)

    logger.info("Data shape: {}", dataset.X.shape)
    logger.info("Input dimension: {}", dataset.input_dim)
    logger.info("Loaded MvDEC result")
    logger.info("h_fused dataframe shape: {}", mvdec_result.h_fused_df.shape)
    logger.info("Iteration: {}", mvdec_result.iteration)
    logger.info("Init method: {}", mvdec_result.init)
    logger.info("Silhouette (saved): {:.4f}", mvdec_result.score)
    logger.info("Silhouette (recomputed): {:.4f}", mvdec_result.score_check)


def run_without_ffs_smoke_test(
    backend: ClusteringBackend,
    data_path,
    representation_path,
) -> None:
    """Run one without-FFS configuration for a quick correctness check."""

    mvdec_result = load_mvdec_result(representation_path, data_path)
    fpmax_features = extract_fpmax_features(
        df=mvdec_result.h_fused_df,
        n_bins=7,
        strategy="kmeans",
        min_support=0.2,
    )
    if backend == "kprototypes":
        clustering = run_kprototypes(
            continuous_df=mvdec_result.h_fused_df,
            binary_df=fpmax_features.features,
        )
    elif backend == "intuitive":
        clustering = run_intuitive_kprototypes(
            continuous_df=mvdec_result.h_fused_df,
            binary_df=fpmax_features.features,
        )
    else:
        msg = f"Unsupported backend: {backend!r}."
        raise ValueError(msg)

    logger.info("Itemsets found: {}", len(fpmax_features.itemsets))
    logger.info("Backend: {}", backend)
    logger.info("Binary features used: {}", len(clustering.binary_feature_names))
    logger.info("Best init: {}", clustering.init)
    logger.info("Fit time: {:.3f}s", clustering.fit_time_seconds)
    logger.info("Cluster sizes: {}", clustering.cluster_sizes)
    if backend == "intuitive":
        logger.info("Converged: {}", clustering.converged)
        logger.info("Iterations: {}", clustering.n_iter)
    logger.info("Final silhouette: {:.4f}", clustering.score)


def run_ffs_smoke_test(data_path, representation_path) -> None:
    """Run one FFS configuration for a quick correctness check."""

    mvdec_result = load_mvdec_result(representation_path, data_path)
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
            "without-ffs-kprototypes",
            "without-ffs-intuitive",
            "without-ffs-intuitive-best",
            "without-ffs-intuitive-exhaustive-best",
            "without-ffs-intuitive-view-weighted",
            "without-ffs-intuitive-view-weighted-best",
            "without-ffs-intuitive-view-weighted-exhaustive-best",
            "without-ffs-intuitive-native",
            "without-ffs-intuitive-paper-native",
            "without-ffs-intuitive-exhaustive-native",
            "without-ffs-intuitive-paper-exhaustive-native",
            "without-ffs-intuitive-view-weighted-native",
            "without-ffs-intuitive-view-weighted-paper-native",
            "without-ffs-intuitive-view-weighted-exhaustive-native",
            "without-ffs-intuitive-view-weighted-paper-exhaustive-native",
            "ffs-kprototypes",
            "ffs-intuitive",
            "ffs-intuitive-best",
            "ffs-intuitive-exhaustive-best",
            "ffs-intuitive-view-weighted",
            "ffs-intuitive-view-weighted-best",
            "ffs-intuitive-view-weighted-exhaustive-best",
            "ffs-intuitive-native",
            "ffs-intuitive-paper-native",
            "ffs-intuitive-exhaustive-native",
            "ffs-intuitive-paper-exhaustive-native",
            "ffs-intuitive-view-weighted-native",
            "ffs-intuitive-view-weighted-paper-native",
            "ffs-intuitive-view-weighted-exhaustive-native",
            "ffs-intuitive-view-weighted-paper-exhaustive-native",
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
        "--param-workers",
        type=int,
        default=1,
        help=(
            "Number of inner worker processes for Intuitive parameter configs. "
            "Applies to Intuitive grid modes."
        ),
    )
    parser.add_argument(
        "--candidate-workers",
        type=int,
        default=1,
        help=(
            "Number of inner worker processes for FFS candidate feature sets. "
            "Use with FFS K-Prototypes and FFS native Intuitive modes. "
            "For native Intuitive, keep --param-workers at 1 when this is above 1."
        ),
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
    parser.add_argument(
        "--backend",
        choices=["kprototypes", "intuitive"],
        default="kprototypes",
        help="Clustering backend for the without-FFS smoke test.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=PREPROCESSED_DATA_PATH,
        help="Preprocessed CSV used to validate representation row count.",
    )
    parser.add_argument(
        "--representation-path",
        type=Path,
        default=FUSED_REPRESENTATION_PATH,
        help="Fused representation pickle produced by the GPU stage.",
    )
    return parser.parse_args()


def load_selected_mvdec_result(args: argparse.Namespace):
    """Load the fused representation selected by CLI paths."""

    return load_mvdec_result(args.representation_path, args.data_path)


def main() -> None:
    """Run the selected command-line workflow."""

    warnings.filterwarnings("ignore")
    np.random.seed(RANDOM_STATE)

    args = parse_args()

    if args.mode == "smoke":
        run_smoke_test(args.data_path, args.representation_path)
    elif args.mode == "without-ffs-smoke":
        run_without_ffs_smoke_test(
            args.backend,
            args.data_path,
            args.representation_path,
        )
    elif args.mode == "ffs-smoke":
        run_ffs_smoke_test(args.data_path, args.representation_path)
    elif args.mode == "without-ffs-kprototypes":
        mvdec_result = load_selected_mvdec_result(args)
        run_without_ffs(
            h_fused_df=mvdec_result.h_fused_df,
            baseline_score=mvdec_result.score,
            workers=args.workers,
            limit=args.limit,
            resume=not args.no_resume,
        )
    elif args.mode == "without-ffs-intuitive":
        mvdec_result = load_selected_mvdec_result(args)
        run_post_without_ffs_intuitive(
            h_fused_df=mvdec_result.h_fused_df,
            param_workers=args.param_workers,
            resume=not args.no_resume,
        )
    elif args.mode == "without-ffs-intuitive-best":
        mvdec_result = load_selected_mvdec_result(args)
        run_post_without_ffs_intuitive_best(
            h_fused_df=mvdec_result.h_fused_df,
            param_workers=args.param_workers,
            resume=not args.no_resume,
        )
    elif args.mode == "without-ffs-intuitive-exhaustive-best":
        mvdec_result = load_selected_mvdec_result(args)
        run_post_without_ffs_intuitive_exhaustive_best(
            h_fused_df=mvdec_result.h_fused_df,
            param_workers=args.param_workers,
            resume=not args.no_resume,
        )
    elif args.mode == "without-ffs-intuitive-view-weighted":
        mvdec_result = load_selected_mvdec_result(args)
        run_post_without_ffs_intuitive_view_weighted(
            h_fused_df=mvdec_result.h_fused_df,
            param_workers=args.param_workers,
            resume=not args.no_resume,
        )
    elif args.mode == "without-ffs-intuitive-view-weighted-best":
        mvdec_result = load_selected_mvdec_result(args)
        run_post_without_ffs_intuitive_view_weighted_best(
            h_fused_df=mvdec_result.h_fused_df,
            param_workers=args.param_workers,
            resume=not args.no_resume,
        )
    elif args.mode == "without-ffs-intuitive-view-weighted-exhaustive-best":
        mvdec_result = load_selected_mvdec_result(args)
        run_post_without_ffs_intuitive_view_weighted_exhaustive_best(
            h_fused_df=mvdec_result.h_fused_df,
            param_workers=args.param_workers,
            resume=not args.no_resume,
        )
    elif args.mode == "without-ffs-intuitive-native":
        mvdec_result = load_selected_mvdec_result(args)
        run_without_ffs_intuitive_native(
            h_fused_df=mvdec_result.h_fused_df,
            baseline_score=mvdec_result.score,
            workers=args.workers,
            param_workers=args.param_workers,
            limit=args.limit,
            resume=not args.no_resume,
        )
    elif args.mode == "without-ffs-intuitive-paper-native":
        mvdec_result = load_selected_mvdec_result(args)
        run_without_ffs_intuitive_paper_native(
            h_fused_df=mvdec_result.h_fused_df,
            baseline_score=mvdec_result.score,
            workers=args.workers,
            param_workers=args.param_workers,
            limit=args.limit,
            resume=not args.no_resume,
        )
    elif args.mode == "without-ffs-intuitive-exhaustive-native":
        mvdec_result = load_selected_mvdec_result(args)
        run_without_ffs_intuitive_exhaustive_native(
            h_fused_df=mvdec_result.h_fused_df,
            baseline_score=mvdec_result.score,
            workers=args.workers,
            param_workers=args.param_workers,
            limit=args.limit,
            resume=not args.no_resume,
        )
    elif args.mode == "without-ffs-intuitive-paper-exhaustive-native":
        mvdec_result = load_selected_mvdec_result(args)
        run_without_ffs_intuitive_paper_exhaustive_native(
            h_fused_df=mvdec_result.h_fused_df,
            baseline_score=mvdec_result.score,
            workers=args.workers,
            param_workers=args.param_workers,
            limit=args.limit,
            resume=not args.no_resume,
        )
    elif args.mode == "without-ffs-intuitive-view-weighted-native":
        mvdec_result = load_selected_mvdec_result(args)
        run_without_ffs_intuitive_view_weighted_native(
            h_fused_df=mvdec_result.h_fused_df,
            baseline_score=mvdec_result.score,
            workers=args.workers,
            param_workers=args.param_workers,
            limit=args.limit,
            resume=not args.no_resume,
        )
    elif args.mode == "without-ffs-intuitive-view-weighted-paper-native":
        mvdec_result = load_selected_mvdec_result(args)
        run_without_ffs_intuitive_view_weighted_paper_native(
            h_fused_df=mvdec_result.h_fused_df,
            baseline_score=mvdec_result.score,
            workers=args.workers,
            param_workers=args.param_workers,
            limit=args.limit,
            resume=not args.no_resume,
        )
    elif args.mode == "without-ffs-intuitive-view-weighted-exhaustive-native":
        mvdec_result = load_selected_mvdec_result(args)
        run_without_ffs_intuitive_view_weighted_exhaustive_native(
            h_fused_df=mvdec_result.h_fused_df,
            baseline_score=mvdec_result.score,
            workers=args.workers,
            param_workers=args.param_workers,
            limit=args.limit,
            resume=not args.no_resume,
        )
    elif args.mode == "without-ffs-intuitive-view-weighted-paper-exhaustive-native":
        mvdec_result = load_selected_mvdec_result(args)
        run_without_ffs_intuitive_view_weighted_paper_exhaustive_native(
            h_fused_df=mvdec_result.h_fused_df,
            baseline_score=mvdec_result.score,
            workers=args.workers,
            param_workers=args.param_workers,
            limit=args.limit,
            resume=not args.no_resume,
        )
    elif args.mode == "ffs-kprototypes":
        mvdec_result = load_selected_mvdec_result(args)
        run_ffs(
            h_fused_df=mvdec_result.h_fused_df,
            baseline_score=mvdec_result.score,
            workers=args.workers,
            candidate_workers=args.candidate_workers,
            limit=args.limit,
            resume=not args.no_resume,
        )
    elif args.mode == "ffs-intuitive":
        mvdec_result = load_selected_mvdec_result(args)
        run_post_ffs_intuitive(
            h_fused_df=mvdec_result.h_fused_df,
            param_workers=args.param_workers,
            resume=not args.no_resume,
        )
    elif args.mode == "ffs-intuitive-best":
        mvdec_result = load_selected_mvdec_result(args)
        run_post_ffs_intuitive_best(
            h_fused_df=mvdec_result.h_fused_df,
            param_workers=args.param_workers,
            resume=not args.no_resume,
        )
    elif args.mode == "ffs-intuitive-exhaustive-best":
        mvdec_result = load_selected_mvdec_result(args)
        run_post_ffs_intuitive_exhaustive_best(
            h_fused_df=mvdec_result.h_fused_df,
            param_workers=args.param_workers,
            resume=not args.no_resume,
        )
    elif args.mode == "ffs-intuitive-view-weighted":
        mvdec_result = load_selected_mvdec_result(args)
        run_post_ffs_intuitive_view_weighted(
            h_fused_df=mvdec_result.h_fused_df,
            param_workers=args.param_workers,
            resume=not args.no_resume,
        )
    elif args.mode == "ffs-intuitive-view-weighted-best":
        mvdec_result = load_selected_mvdec_result(args)
        run_post_ffs_intuitive_view_weighted_best(
            h_fused_df=mvdec_result.h_fused_df,
            param_workers=args.param_workers,
            resume=not args.no_resume,
        )
    elif args.mode == "ffs-intuitive-view-weighted-exhaustive-best":
        mvdec_result = load_selected_mvdec_result(args)
        run_post_ffs_intuitive_view_weighted_exhaustive_best(
            h_fused_df=mvdec_result.h_fused_df,
            param_workers=args.param_workers,
            resume=not args.no_resume,
        )
    elif args.mode == "ffs-intuitive-native":
        mvdec_result = load_selected_mvdec_result(args)
        run_ffs_intuitive_native(
            h_fused_df=mvdec_result.h_fused_df,
            baseline_score=mvdec_result.score,
            workers=args.workers,
            param_workers=args.param_workers,
            candidate_workers=args.candidate_workers,
            limit=args.limit,
            resume=not args.no_resume,
        )
    elif args.mode == "ffs-intuitive-paper-native":
        mvdec_result = load_selected_mvdec_result(args)
        run_ffs_intuitive_paper_native(
            h_fused_df=mvdec_result.h_fused_df,
            baseline_score=mvdec_result.score,
            workers=args.workers,
            param_workers=args.param_workers,
            candidate_workers=args.candidate_workers,
            limit=args.limit,
            resume=not args.no_resume,
        )
    elif args.mode == "ffs-intuitive-exhaustive-native":
        mvdec_result = load_selected_mvdec_result(args)
        run_ffs_intuitive_exhaustive_native(
            h_fused_df=mvdec_result.h_fused_df,
            baseline_score=mvdec_result.score,
            workers=args.workers,
            param_workers=args.param_workers,
            candidate_workers=args.candidate_workers,
            limit=args.limit,
            resume=not args.no_resume,
        )
    elif args.mode == "ffs-intuitive-paper-exhaustive-native":
        mvdec_result = load_selected_mvdec_result(args)
        run_ffs_intuitive_paper_exhaustive_native(
            h_fused_df=mvdec_result.h_fused_df,
            baseline_score=mvdec_result.score,
            workers=args.workers,
            param_workers=args.param_workers,
            candidate_workers=args.candidate_workers,
            limit=args.limit,
            resume=not args.no_resume,
        )
    elif args.mode == "ffs-intuitive-view-weighted-native":
        mvdec_result = load_selected_mvdec_result(args)
        run_ffs_intuitive_view_weighted_native(
            h_fused_df=mvdec_result.h_fused_df,
            baseline_score=mvdec_result.score,
            workers=args.workers,
            param_workers=args.param_workers,
            candidate_workers=args.candidate_workers,
            limit=args.limit,
            resume=not args.no_resume,
        )
    elif args.mode == "ffs-intuitive-view-weighted-paper-native":
        mvdec_result = load_selected_mvdec_result(args)
        run_ffs_intuitive_view_weighted_paper_native(
            h_fused_df=mvdec_result.h_fused_df,
            baseline_score=mvdec_result.score,
            workers=args.workers,
            param_workers=args.param_workers,
            candidate_workers=args.candidate_workers,
            limit=args.limit,
            resume=not args.no_resume,
        )
    elif args.mode == "ffs-intuitive-view-weighted-exhaustive-native":
        mvdec_result = load_selected_mvdec_result(args)
        run_ffs_intuitive_view_weighted_exhaustive_native(
            h_fused_df=mvdec_result.h_fused_df,
            baseline_score=mvdec_result.score,
            workers=args.workers,
            param_workers=args.param_workers,
            candidate_workers=args.candidate_workers,
            limit=args.limit,
            resume=not args.no_resume,
        )
    elif args.mode == "ffs-intuitive-view-weighted-paper-exhaustive-native":
        mvdec_result = load_selected_mvdec_result(args)
        run_ffs_intuitive_view_weighted_paper_exhaustive_native(
            h_fused_df=mvdec_result.h_fused_df,
            baseline_score=mvdec_result.score,
            workers=args.workers,
            param_workers=args.param_workers,
            candidate_workers=args.candidate_workers,
            limit=args.limit,
            resume=not args.no_resume,
        )


if __name__ == "__main__":
    main()

