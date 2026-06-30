"""Run the 2025 MvDEC paper baseline on selected public datasets."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from loguru import logger

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


DEFAULT_CONFIG_PATH = PROJECT_DIR / "configs" / "public_datasets.json"
DEFAULT_OUTPUT_PATH = (
    PROJECT_DIR / "output" / "public_baselines" / "mvdec_paper_baseline.xlsx"
)
PAPER_TEXT_DATASETS = ("reuters10k", "20news", "rcv1_10k")


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(
        description="Run the two-view MvDEC paper baseline on public datasets."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Public dataset registry JSON path.",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=None,
        help="Override processed dataset root. Defaults to the registry value.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Excel workbook output path.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset names to run. Defaults to every registry entry.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing workbook.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately when one dataset fails.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device: auto, cpu, cuda, or cuda:0.",
    )
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=None,
        help="Override autoencoder pretraining epochs.",
    )
    parser.add_argument(
        "--joint-epochs",
        type=int,
        default=None,
        help="Override joint clustering epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override training batch size.",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=None,
        help="Override fused embedding dimension.",
    )
    parser.add_argument(
        "--scaler",
        choices=("minmax", "standard"),
        default=None,
        help="Override feature scaler.",
    )
    parser.add_argument(
        "--reconstruction-weight",
        type=float,
        default=None,
        help="Override reconstruction loss weight.",
    )
    parser.add_argument(
        "--kmeans-weight",
        type=float,
        default=None,
        help="Override K-Means loss weight.",
    )
    parser.add_argument(
        "--orthonormal-weight",
        type=float,
        default=None,
        help="Override orthonormal loss weight.",
    )
    parser.add_argument(
        "--greedy-weight",
        type=float,
        default=None,
        help="Override greedy adjustment loss weight.",
    )
    parser.add_argument(
        "--embedding-variance-weight",
        type=float,
        default=None,
        help="Override anti-collapse embedding variance loss weight.",
    )
    parser.add_argument(
        "--embedding-variance-target",
        type=float,
        default=None,
        help="Override anti-collapse embedding target standard deviation.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a short smoke configuration for code-path validation.",
    )
    parser.add_argument(
        "--paper-strict",
        action="store_true",
        help=(
            "Use the paper-style text benchmark defaults and, when --datasets "
            "is omitted, run REUTERS-10K, 20NEWS, and RCV1-10K."
        ),
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace, config_cls):
    """Build the shared MvDEC paper configuration from CLI options."""

    config = config_cls(device=args.device, paper_strict=args.paper_strict)
    if not args.paper_strict:
        config = replace(
            config,
            scaler="standard",
            kmeans_weight=0.1,
            orthonormal_weight=0.0,
            greedy_weight=0.0,
            embedding_variance_weight=0.1,
        )
    if args.paper_strict:
        config = replace(
            config,
            scaler="minmax",
            hidden_dims=(500, 500, 2000),
            latent_dim=10,
            kmeans_n_init=20,
            reconstruction_weight=1.0,
            kmeans_weight=1.0,
            orthonormal_weight=1.0,
            greedy_weight=1.0,
            embedding_variance_weight=0.0,
        )
    if args.pretrain_epochs is not None:
        config = replace(config, pretrain_epochs=args.pretrain_epochs)
    if args.joint_epochs is not None:
        config = replace(config, joint_epochs=args.joint_epochs)
    if args.batch_size is not None:
        config = replace(config, batch_size=args.batch_size)
    if args.latent_dim is not None:
        config = replace(config, latent_dim=args.latent_dim)
    if args.scaler is not None:
        config = replace(config, scaler=args.scaler)
    if args.reconstruction_weight is not None:
        config = replace(config, reconstruction_weight=args.reconstruction_weight)
    if args.kmeans_weight is not None:
        config = replace(config, kmeans_weight=args.kmeans_weight)
    if args.orthonormal_weight is not None:
        config = replace(config, orthonormal_weight=args.orthonormal_weight)
    if args.greedy_weight is not None:
        config = replace(config, greedy_weight=args.greedy_weight)
    if args.embedding_variance_weight is not None:
        config = replace(
            config, embedding_variance_weight=args.embedding_variance_weight
        )
    if args.embedding_variance_target is not None:
        config = replace(
            config, embedding_variance_target=args.embedding_variance_target
        )
    if args.smoke:
        config = replace(
            config,
            hidden_dims=(64, 64, 128),
            latent_dim=args.latent_dim or 10,
            pretrain_epochs=args.pretrain_epochs or 2,
            joint_epochs=args.joint_epochs or 2,
            kmeans_n_init=2,
            early_stopping_patience=2,
        )
    return config


def main() -> None:
    """Run MvDEC paper baselines and save one Excel workbook."""

    args = parse_args()
    if args.output.exists() and not args.force:
        msg = f"Output already exists: {args.output}. Use --force to overwrite."
        raise FileExistsError(msg)

    from public_baselines.mvdec_paper import (  # noqa: PLC0415
        MvDECPaperConfig,
        MvDECPaperResult,
        load_dataset,
        load_registry,
        log_device,
        metadata_frame,
        resolve_processed_root,
        run_mvdec_paper_baseline,
        selected_datasets,
        write_workbook,
    )

    registry = load_registry(args.config)
    dataset_names = (
        list(PAPER_TEXT_DATASETS)
        if args.paper_strict and args.datasets is None
        else args.datasets
    )
    datasets = selected_datasets(registry, dataset_names)
    processed_root = resolve_processed_root(PROJECT_DIR, registry, args.processed_root)
    config = build_config(args, MvDECPaperConfig)
    log_device(config)
    metadata = metadata_frame(processed_root, datasets)

    results = []
    label_frames = []
    history_rows = []
    for dataset in datasets:
        logger.info("Running MvDEC paper baseline for {}", dataset["name"])
        try:
            inputs = load_dataset(dataset, processed_root)
            result, label_frame, history = run_mvdec_paper_baseline(inputs, config)
            label_frames.append(label_frame)
            history_rows.extend(history)
            logger.info(
                "{} acc={} nmi={} ari={} silhouette={:.4f}",
                result.dataset,
                None if result.acc is None else round(result.acc, 4),
                None if result.nmi is None else round(result.nmi, 4),
                None if result.ari is None else round(result.ari, 4),
                result.silhouette,
            )
        except Exception as error:  # noqa: BLE001 - keep long Colab runs alive.
            logger.exception("{} failed", dataset["name"])
            if args.fail_fast:
                raise
            result = MvDECPaperResult(
                dataset=dataset["name"],
                acc=None,
                nmi=None,
                ari=None,
                silhouette=float("nan"),
                cluster_sizes=[],
                inertia=float("nan"),
                best_epoch=0,
                converged=False,
                fit_time_seconds=0.0,
                device=config.device,
                status="failed",
                error_message=str(error),
            )
        results.append(result)
        write_workbook(
            output_path=args.output,
            results=results,
            label_frames=label_frames,
            history_rows=history_rows,
            metadata=metadata,
            config=config,
        )
        logger.info("Saved partial MvDEC workbook to {}", args.output)

    logger.info("Saved MvDEC paper baseline workbook to {}", args.output)


if __name__ == "__main__":
    main()
