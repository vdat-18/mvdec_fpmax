"""Run MvDEC representation learning and save the fused artifact."""

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

from config import (  # noqa: E402
    FUSED_REPRESENTATION_PATH,
    H_FUSED_COLUMNS,
    PREPROCESSED_DATA_PATH,
)
from representation_learning.mvdec_representation import (  # noqa: E402
    MvDECRepresentationConfig,
    load_feature_matrix,
    save_representation,
    tensorflow_gpu_names,
    train_mvdec_representation,
)

DEFAULT_HISTORY_PATH = (
    FUSED_REPRESENTATION_PATH.parent / "mvdec_representation_history.csv"
)


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(
        description=(
            "Train MvDEC representation learning and save fused_representation.pkl."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PREPROCESSED_DATA_PATH,
        help="Preprocessed continuous feature CSV path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=FUSED_REPRESENTATION_PATH,
        help="Output fused representation pickle path.",
    )
    parser.add_argument(
        "--history-output",
        type=Path,
        default=DEFAULT_HISTORY_PATH,
        help="CSV path for per-iteration K-Means silhouette history.",
    )
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--clusters", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow running without a TensorFlow-visible GPU.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny configuration for Colab/runtime validation.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> MvDECRepresentationConfig:
    """Build representation-learning config from CLI options."""

    config = MvDECRepresentationConfig()
    if args.iterations is not None:
        config = replace(config, n_iterations=args.iterations)
    if args.clusters is not None:
        config = replace(config, n_clusters=args.clusters)
    if args.batch_size is not None:
        config = replace(config, batch_size=args.batch_size)
    if args.epochs is not None:
        config = replace(config, epochs=args.epochs)
    if args.learning_rate is not None:
        config = replace(config, learning_rate=args.learning_rate)
    if args.seed is not None:
        config = replace(config, seed=args.seed)
    if args.smoke:
        config = replace(
            config,
            n_iterations=args.iterations or 1,
            epochs=args.epochs or 2,
            kmeans_n_init=2,
            early_stopping_patience=1,
        )
    return config


def ensure_output_paths(args: argparse.Namespace) -> None:
    """Prevent accidental overwrites unless explicitly requested."""

    existing = [path for path in (args.output, args.history_output) if path.exists()]
    if existing and not args.force:
        joined = ", ".join(str(path) for path in existing)
        msg = f"Output already exists: {joined}. Use --force to overwrite."
        raise FileExistsError(msg)


def main() -> None:
    """Train and save the fused MvDEC representation."""

    args = parse_args()
    ensure_output_paths(args)

    gpu_names = tensorflow_gpu_names()
    if not gpu_names and not args.allow_cpu:
        msg = (
            "No TensorFlow GPU detected. In Colab, select Runtime -> Change "
            "runtime type -> GPU, or pass --allow-cpu for a slow local run."
        )
        raise RuntimeError(msg)
    logger.info("TensorFlow GPUs: {}", gpu_names or ["none"])

    df, X = load_feature_matrix(args.input)
    config = build_config(args)
    logger.info("Input shape: {}", X.shape)
    logger.info("Config: {}", config)

    result, history = train_mvdec_representation(
        X=X,
        config=config,
        expected_fused_dim=len(H_FUSED_COLUMNS),
    )
    result["input_path"] = str(args.input)
    result["n_samples"] = int(X.shape[0])
    result["input_dim"] = int(X.shape[1])
    result["feature_columns"] = df.columns.tolist()

    save_representation(result, args.output)
    args.history_output.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(args.history_output, index=False)

    logger.info("Saved fused representation to {}", args.output)
    logger.info("Saved representation history to {}", args.history_output)
    logger.info(
        "Best init={} score={:.6f} iteration={}",
        result["init"],
        result["score"],
        result["iteration"],
    )


if __name__ == "__main__":
    main()
