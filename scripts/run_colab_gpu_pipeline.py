"""Run the GPU-side Colab pipeline from one entrypoint."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_DIR / "configs" / "public_datasets.json"
DEFAULT_REPRESENTATION_OUTPUT_DIR = PROJECT_DIR / "data" / "preprocessed_data"
DEFAULT_BASELINE_OUTPUT_PATH = (
    PROJECT_DIR / "output" / "public_baselines" / "mvdec_gpu_pipeline.xlsx"
)
DEFAULT_GPU_MODES = (
    "mvdec",
    "mimvdec",
    "fused-kmeans",
    "legacy-fused-kmeans",
)

def parse_args() -> argparse.Namespace:
    """Parse Colab GPU pipeline options."""

    parser = argparse.ArgumentParser(
        description=(
            "Run dataset fetch, MvDEC representation learning, and paper-style "
            "MvDEC GPU baselines from one Colab-friendly command."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Dataset registry JSON path.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=("tiki",),
        help="Dataset names from configs/public_datasets.json.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=("all",),
        choices=("all", "representation", "paper-baseline"),
        help="GPU tasks to run. Use 'all' for representation and baselines.",
    )
    parser.add_argument(
        "--representation-output-dir",
        type=Path,
        default=DEFAULT_REPRESENTATION_OUTPUT_DIR,
        help="Directory for dataset-specific fused representation artifacts.",
    )
    parser.add_argument(
        "--baseline-output",
        type=Path,
        default=DEFAULT_BASELINE_OUTPUT_PATH,
        help="Excel workbook path for paper-style GPU baselines.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=DEFAULT_GPU_MODES,
        help="Method modes passed to run_public_mvdec_paper_baseline.py.",
    )
    parser.add_argument(
        "--kmeans-inits",
        nargs="+",
        default=("k-means++", "random"),
        choices=("k-means++", "random"),
        help="K-Means initializations passed to the paper-style baseline.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for paper-style GPU baselines.",
    )
    parser.add_argument(
        "--representation-scaler",
        choices=("none", "minmax", "standard"),
        default="none",
        help="Scaler passed to run_mvdec_representation.py.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run short smoke settings for both GPU tasks.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow representation learning without a TensorFlow-visible GPU.",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Do not rebuild data/public/processed before paper baselines.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing GPU outputs.",
    )
    return parser.parse_args()


def normalized_tasks(tasks: tuple[str, ...] | list[str]) -> set[str]:
    """Expand the task shortcut into concrete task names."""

    if "all" in tasks:
        return {"representation", "paper-baseline"}
    return set(tasks)


def load_registry(config_path: Path) -> dict:
    """Load the dataset registry."""

    with config_path.open(encoding="utf-8") as file:
        return json.load(file)


def selected_registry_entries(registry: dict, dataset_names: list[str]) -> list[dict]:
    """Return registry entries for the requested dataset names."""

    by_name = {dataset["name"]: dataset for dataset in registry["datasets"]}
    missing = sorted(set(dataset_names) - set(by_name))
    if missing:
        msg = f"Unknown dataset names: {missing}"
        raise ValueError(msg)
    return [by_name[name] for name in dataset_names]


def local_dataset_source(dataset: dict, processed_root: Path) -> Path:
    """Resolve the numeric matrix used by representation learning."""

    url_path = PROJECT_DIR / dataset["url"]
    if url_path.exists():
        return url_path

    processed_x = processed_root / dataset["name"] / "X.csv"
    if processed_x.exists():
        return processed_x

    msg = (
        f"Could not find an input matrix for {dataset['name']}. "
        f"Expected {url_path} or {processed_x}."
    )
    raise FileNotFoundError(msg)


def run_command(command: list[str]) -> None:
    """Run one subprocess command from the project root."""

    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_DIR, check=True)


def run_fetch(args: argparse.Namespace) -> None:
    """Build processed dataset files required by paper baselines."""

    if args.skip_fetch:
        return
    command = [
        sys.executable,
        "scripts/fetch_public_datasets.py",
        "--config",
        str(args.config),
        "--datasets",
        *args.datasets,
    ]
    if args.force:
        command.append("--force")
    run_command(command)


def run_representation_task(args: argparse.Namespace, registry: dict) -> None:
    """Train dataset-specific fused representations."""

    processed_root = PROJECT_DIR / registry["processed_root"]
    args.representation_output_dir.mkdir(parents=True, exist_ok=True)
    for dataset in selected_registry_entries(registry, list(args.datasets)):
        input_path = local_dataset_source(dataset, processed_root)
        stem = dataset["name"]
        output_path = (
            args.representation_output_dir / f"{stem}_fused_representation.pkl"
        )
        history_path = args.representation_output_dir / f"{stem}_mvdec_history.csv"
        command = [
            sys.executable,
            "scripts/run_mvdec_representation.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--history-output",
            str(history_path),
            "--clusters",
            str(dataset["n_clusters"]),
            "--scaler",
            args.representation_scaler,
            "--allow-dynamic-fused-dim",
        ]
        if args.smoke:
            command.append("--smoke")
        if args.allow_cpu:
            command.append("--allow-cpu")
        if args.force:
            command.append("--force")
        run_command(command)


def run_paper_baseline_task(args: argparse.Namespace) -> None:
    """Run paper-style GPU baselines for all requested datasets."""

    command = [
        sys.executable,
        "scripts/run_public_mvdec_paper_baseline.py",
        "--config",
        str(args.config),
        "--datasets",
        *args.datasets,
        "--device",
        args.device,
        "--output",
        str(args.baseline_output),
        "--modes",
        *args.modes,
        "--kmeans-inits",
        *args.kmeans_inits,
    ]
    if args.smoke:
        command.append("--smoke")
    if args.force:
        command.append("--force")
    run_command(command)


def main() -> None:
    """Run the selected Colab GPU tasks."""

    args = parse_args()
    tasks = normalized_tasks(args.tasks)
    registry = load_registry(args.config)
    selected_registry_entries(registry, list(args.datasets))

    if "representation" in tasks:
        run_representation_task(args, registry)
    if "paper-baseline" in tasks:
        run_fetch(args)
        run_paper_baseline_task(args)


if __name__ == "__main__":
    main()
