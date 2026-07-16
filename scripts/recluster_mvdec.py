"""Recluster a trusted MvDEC embedding and compare it with saved assignments."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from loguru import logger

from config import PREPROCESSED_DATA_PATH
from pipeline.data import load_mvdec_result, load_preprocessed_dataset
from pipeline.reclustering import (
    artifact_kmeans_config,
    build_assignment_frame,
    cluster_sizes,
    recluster_embedding,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Re-run K-Means on a trusted MvDEC pickle, compare assignments, "
            "and export rows for cluster interpretation."
        )
    )
    parser.add_argument(
        "--representation-path",
        type=Path,
        required=True,
        help="Path to a trusted MvDEC pickle artifact.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=PREPROCESSED_DATA_PATH,
        help="Preprocessed CSV whose row order matches the MvDEC artifact.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination CSV for features, embeddings, and cluster assignments.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output CSV if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    """Recluster the fused representation, report agreement, and export rows."""

    args = parse_args()
    if args.output.exists() and not args.force:
        msg = f"Output already exists: {args.output}. Use --force to overwrite it."
        raise FileExistsError(msg)

    mvdec = load_mvdec_result(
        result_path=args.representation_path,
        data_path=args.data_path,
    )
    source = load_preprocessed_dataset(args.data_path)
    config = artifact_kmeans_config(mvdec.raw)
    result = recluster_embedding(mvdec.h_fused, mvdec.labels, config)
    assignments = build_assignment_frame(
        source.df,
        mvdec.h_fused_df,
        mvdec.labels,
        result,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(args.output, index=False)

    changed_percentage = 100.0 * result.changed_count / len(mvdec.labels)
    logger.info(
        "K-Means config: K={} n_init={} random_state={}",
        config.n_clusters,
        config.n_init,
        config.random_state,
    )
    logger.info(
        "Silhouette: artifact_saved={:.6f} artifact_recomputed={:.6f} "
        "reclustered={:.6f}",
        mvdec.score,
        mvdec.score_check,
        result.silhouette,
    )
    logger.info("Agreement: ARI={:.6f} NMI={:.6f}", result.ari, result.nmi)
    logger.info(
        "Changed after label alignment: {}/{} ({:.4f}%)",
        result.changed_count,
        len(mvdec.labels),
        changed_percentage,
    )
    logger.info(
        "Cluster sizes: artifact={} reclustered_aligned={}",
        cluster_sizes(mvdec.labels),
        cluster_sizes(result.labels_aligned),
    )
    logger.info("K-Means inertia={:.6f} iterations={}", result.inertia, result.n_iter)
    scores_match = np.isclose(mvdec.score, mvdec.score_check) and np.isclose(
        mvdec.score_check,
        result.silhouette,
    )
    if result.changed_count == 0 and scores_match:
        logger.success("VERDICT: exact MvDEC clustering reproduction.")
    else:
        logger.warning(
            "VERDICT: clustering was not reproduced exactly; inspect the metrics."
        )
    logger.info("Saved cluster interpretation CSV to {}", args.output)


if __name__ == "__main__":
    main()
