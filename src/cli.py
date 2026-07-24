"""Command-line entrypoint for smoke tests and MiMvDEC experiments."""

import argparse
import warnings
from pathlib import Path

import numpy as np
from loguru import logger

import config as project_config
from config import (
    PREPROCESSED_DATA_PATH,
    RANDOM_STATE,
)
from pipeline.clustering import (
    ClusteringBackend,
    run_intuitive_kprototypes,
    run_kprototypes,
)
from pipeline.data import load_mvdec_result, load_preprocessed_dataset
from pipeline.experiment_context import (
    artifact_n_clusters,
    build_experiment_context,
)
from pipeline.experiments import (
    run_ffs,
    run_without_ffs,
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
    logger.info("Silhouette (recomputed Euclidean): {:.4f}", mvdec_result.score_check)
    logger.info(
        "Silhouette (common Gower baseline): {:.4f}",
        mvdec_result.evaluation_score,
    )


def run_without_ffs_smoke_test(
    backend: ClusteringBackend,
    data_path,
    representation_path,
) -> None:
    """Run one without-FFS configuration for a quick correctness check."""

    mvdec_result = load_mvdec_result(representation_path, data_path)
    n_clusters = artifact_n_clusters(mvdec_result.raw)
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
            n_clusters=n_clusters,
        )
    elif backend == "intuitive":
        clustering = run_intuitive_kprototypes(
            continuous_df=mvdec_result.h_fused_df,
            binary_df=fpmax_features.features,
            n_clusters=n_clusters,
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
    n_clusters = artifact_n_clusters(mvdec_result.raw)
    fpmax_features = extract_fpmax_features(
        df=mvdec_result.h_fused_df,
        n_bins=7,
        strategy="kmeans",
        min_support=0.35,
    )
    selected = run_forward_selection(
        continuous_df=mvdec_result.h_fused_df,
        binary_df=fpmax_features.features,
        n_clusters=n_clusters,
        baseline_score=mvdec_result.evaluation_score,
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
            "ffs-kprototypes",
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
        "--candidate-workers",
        type=int,
        default=1,
        help=(
            "Number of inner worker processes for FFS candidate feature sets. "
            "Use with the FFS K-Prototypes mode."
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
        required=True,
        help="Fused representation pickle produced by the GPU stage.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for all result CSV files. Defaults to output/<dataset>; "
            "a manifest prevents resume across different data or artifacts."
        ),
    )
    return parser.parse_args()


def load_selected_mvdec_result(args: argparse.Namespace):
    """Load the fused representation selected by CLI paths."""

    return load_mvdec_result(args.representation_path, args.data_path)


def load_experiment_context(args: argparse.Namespace):
    """Load the selected artifact and prepare its isolated output context."""

    mvdec_result = load_selected_mvdec_result(args)
    context = build_experiment_context(
        artifact=mvdec_result.raw,
        data_path=args.data_path,
        representation_path=args.representation_path,
        requested_output_dir=args.output_dir,
    )
    return mvdec_result, context


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
        mvdec_result, experiment = load_experiment_context(args)
        run_without_ffs(
            h_fused_df=mvdec_result.h_fused_df,
            save_path=experiment.result_path(
                project_config.WITHOUT_FFS_KPROTOTYPES_RESULTS_PATH
            ),
            n_clusters=experiment.n_clusters,
            baseline_score=mvdec_result.evaluation_score,
            workers=args.workers,
            limit=args.limit,
            resume=not args.no_resume,
        )
    elif args.mode == "ffs-kprototypes":
        mvdec_result, experiment = load_experiment_context(args)
        run_ffs(
            h_fused_df=mvdec_result.h_fused_df,
            save_path=experiment.result_path(project_config.FFS_RESULTS_PATH),
            n_clusters=experiment.n_clusters,
            baseline_score=mvdec_result.evaluation_score,
            workers=args.workers,
            candidate_workers=args.candidate_workers,
            limit=args.limit,
            resume=not args.no_resume,
        )


if __name__ == "__main__":
    main()
