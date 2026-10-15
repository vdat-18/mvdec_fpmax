"""Run bounded post-training diagnostics for MvDEC paper-figure fusion variants."""

import argparse
import importlib
import json
import sys
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.cluster import KMeans

from config import DEKM_DATASET_DIR
from pipeline.external_metrics import compute_external_metrics
from pipeline.mvdec_audit import validate_numerical_runtime
from pipeline.mvdec_contract import (
    FUSION_ENCODER_AVERAGE,
    FUSION_ENCODER_L2_NORMALIZED_AVERAGE,
    FUSION_FIGURE1_FULL_OUTPUT_AVERAGE,
    FUSION_FIGURE2_FULL_OUTPUT_CONCATENATION,
    VIEW_INPUT_CONTRACT,
    paper_figure_fusion_metadata,
)
from pipeline.mvdec_fusion_audit import fused_representation
from pipeline.mvdec_runs import (
    file_sha256,
    load_run_manifest,
    resolve_manifest_output_path,
    validate_manifest_identity,
    validate_manifest_outputs,
)
from pipeline.public_artifacts import write_public_frame
from pipeline.public_data import (
    TWENTY_NEWS_RELEASE_NPZ,
    PublicDataset,
    load_dekm_public_dataset,
)

DIAGNOSTIC_SCHEMA_VERSION = 1
DIAGNOSTIC_CLAIM_SCOPE = "post_training_diagnostic_not_exact_paper_truth"
EQ4_REPRESENTATION = "eq4_latent_average"
FIGURE1_REPRESENTATION = "figure1_full_output_average"
FIGURE2_REPRESENTATION = "figure2_full_output_concatenation"
MAX_INFERENCE_BATCH_SIZE = 4096
MAX_DIAGNOSTIC_SAMPLES = 25_000
MAX_VIEW_OUTPUT_WIDTH = 4096
KMEANS_N_INIT = 100

ModelLoader = Callable[
    [dict[str, object], Path, Path],
    tuple[Callable[[np.ndarray], object], Callable[[np.ndarray], object]],
]
DatasetLoader = Callable[[dict[str, object], Path], PublicDataset]


def _positive_int(config: dict[str, object], name: str) -> int:
    value = config.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"MvDEC diagnostic requires positive integer {name!r}.")
    return value


def _validate_diagnostic_config(
    manifest: dict[str, object],
    inference_batch_size: int,
    expected_seed: int,
) -> dict[str, object]:
    if not 1 <= inference_batch_size <= MAX_INFERENCE_BATCH_SIZE:
        raise ValueError(
            f"inference_batch_size must be between 1 and {MAX_INFERENCE_BATCH_SIZE}."
        )
    if manifest.get("seed") != expected_seed:
        raise ValueError(
            f"MvDEC diagnostic expected seed {expected_seed}, found "
            f"{manifest.get('seed')!r}."
        )
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError("MvDEC diagnostic manifest is missing its config.")

    equality_fields = (
        "dataset",
        "source_id",
        "source_sha256",
        "training_commit",
        "view_input_contract",
    )
    for field in equality_fields:
        if config.get(field) != manifest.get(field):
            raise ValueError(f"MvDEC manifest/config {field} does not match.")
    if manifest.get("view_input_contract") != VIEW_INPUT_CONTRACT:
        raise ValueError("Unsupported MvDEC view-input contract.")

    n_samples = _positive_int(config, "n_samples")
    input_width = _positive_int(config, "input_shape")
    hidden_width = _positive_int(config, "hidden_units")
    if n_samples > MAX_DIAGNOSTIC_SAMPLES:
        raise ValueError(
            f"MvDEC diagnostic sample count exceeds {MAX_DIAGNOSTIC_SAMPLES}."
        )
    if input_width != _positive_int(config, "n_features"):
        raise ValueError("MvDEC input_shape must match n_features.")
    output_width = input_width + hidden_width
    if output_width > MAX_VIEW_OUTPUT_WIDTH:
        raise ValueError(f"MvDEC view output width exceeds {MAX_VIEW_OUTPUT_WIDTH}.")
    if config.get("kmeans_n_init") != KMEANS_N_INIT:
        raise ValueError("MvDEC diagnostic requires K-Means n_init=100.")

    filters = config.get("view1_filters")
    if (
        not isinstance(filters, list)
        or not filters
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in filters
        )
    ):
        raise ValueError("MvDEC diagnostic has invalid View1 filters.")
    if _positive_int(config, "view2_base_units") < 2:
        raise ValueError("MvDEC View2 base width must be at least 2.")

    contract = config.get("protocol_contract")
    architecture = contract.get("architecture") if isinstance(contract, dict) else None
    if not isinstance(architecture, dict):
        raise ValueError("MvDEC diagnostic is missing its architecture contract.")
    architecture_id = config.get("view2_architecture_id")
    if (
        not isinstance(architecture_id, str)
        or architecture.get("view2") != architecture_id
    ):
        raise ValueError("MvDEC View2 architecture identity does not match.")
    if architecture.get("view_input_contract") != VIEW_INPUT_CONTRACT:
        raise ValueError("MvDEC protocol has an unsupported view-input contract.")
    return config


def _load_canonical_dataset(
    manifest: dict[str, object],
    dataset_root: Path,
) -> PublicDataset:
    config = manifest["config"]
    contract = config["protocol_contract"]
    preprocessing = contract.get("preprocessing")
    normalization = (
        preprocessing.get("input_normalization")
        if isinstance(preprocessing, dict)
        else None
    )
    if not isinstance(normalization, str):
        raise ValueError("MvDEC protocol is missing input normalization.")
    source_id = manifest["source_id"]
    twenty_news_source = (
        source_id if manifest["dataset"] == "20NEWS" else TWENTY_NEWS_RELEASE_NPZ
    )
    dataset = load_dekm_public_dataset(
        str(manifest["dataset"]),
        dataset_root,
        normalization=normalization,
        twenty_news_source=twenty_news_source,
    )
    expected = (
        ("source_id", dataset.source_id),
        ("source_sha256", dataset.source_sha256),
        ("n_samples", len(dataset.X)),
        ("n_features", dataset.X.shape[1]),
        ("n_classes", dataset.spec.n_clusters),
    )
    for field, actual in expected:
        if manifest.get(field) != actual:
            raise ValueError(f"Canonical dataset {field} does not match the manifest.")
    return dataset


def _default_model_loader(
    config: dict[str, object],
    view1_weights: Path,
    view2_weights: Path,
) -> tuple[Callable[[np.ndarray], object], Callable[[np.ndarray], object]]:
    """Build the recorded architecture while keeping TensorFlow optional locally."""

    representation_dir = Path(__file__).resolve().parents[1] / "representation_learning"
    representation_path = str(representation_dir)
    if representation_path not in sys.path:
        sys.path.insert(0, representation_path)
    mvdec = importlib.import_module("MVDEC_dense")
    mvdec.input_shape = int(config["input_shape"])
    mvdec.hidden_units = int(config["hidden_units"])
    mvdec.view1_filters = list(config["view1_filters"])
    mvdec.view2_base_units = int(config["view2_base_units"])

    mvdec.tf.keras.backend.clear_session()
    view1 = mvdec.model_view1(load_weights=False)
    view2 = mvdec.model_view2(
        load_weights=False,
        architecture_id=str(config["view2_architecture_id"]),
    )
    view1.load_weights(view1_weights)
    view2.load_weights(view2_weights)
    expected_width = int(config["hidden_units"]) + int(config["input_shape"])
    if view1.output_shape != (None, expected_width):
        raise ValueError(
            "Reconstructed View1 output shape does not match the manifest."
        )
    if view2.output_shape != (None, expected_width):
        raise ValueError(
            "Reconstructed View2 output shape does not match the manifest."
        )
    return view1, view2


def _numpy_output(value: object) -> np.ndarray:
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _write_model_outputs(
    X: np.ndarray,
    view1: Callable[[np.ndarray], object],
    view2: Callable[[np.ndarray], object],
    output_width: int,
    batch_size: int,
    directory: Path,
) -> tuple[np.memmap, np.memmap]:
    shape = (len(X), output_width)
    outputs = (
        np.memmap(directory / "view1.dat", dtype=np.float32, mode="w+", shape=shape),
        np.memmap(directory / "view2.dat", dtype=np.float32, mode="w+", shape=shape),
    )
    for start in range(0, len(X), batch_size):
        stop = min(start + batch_size, len(X))
        batch = X[start:stop]
        for model, destination, name in zip(
            (view1, view2), outputs, ("View1", "View2"), strict=True
        ):
            values = _numpy_output(model(batch))
            if values.shape != (stop - start, output_width):
                raise ValueError(f"{name} output shape does not match the manifest.")
            if not np.isfinite(values).all():
                raise ValueError(f"{name} produced non-finite outputs.")
            destination[start:stop] = values.astype(np.float32, copy=False)
    for output in outputs:
        output.flush()
    return outputs


def _materialize_representations(
    view1: np.memmap,
    view2: np.memmap,
    config: dict[str, object],
    batch_size: int,
    directory: Path,
) -> dict[str, tuple[np.memmap, dict[str, object]]]:
    n_samples = view1.shape[0]
    hidden_width = int(config["hidden_units"])
    output_width = view1.shape[1]
    fusion_contract = config.get("fusion_contract", FUSION_ENCODER_AVERAGE)
    if fusion_contract not in {
        FUSION_ENCODER_AVERAGE,
        FUSION_ENCODER_L2_NORMALIZED_AVERAGE,
    }:
        raise ValueError(f"Unsupported declared latent fusion: {fusion_contract!r}.")

    specifications = {
        EQ4_REPRESENTATION: (
            hidden_width,
            {
                "paper_source": "MvDEC 2025 Equation 4",
                "operation": (
                    "elementwise_average"
                    if fusion_contract == FUSION_ENCODER_AVERAGE
                    else "row_l2_normalized_elementwise_average"
                ),
                "fusion_contract": fusion_contract,
                "input_width_per_view": hidden_width,
                "output_width": hidden_width,
                "claim_scope": DIAGNOSTIC_CLAIM_SCOPE,
            },
        ),
        FIGURE1_REPRESENTATION: (
            output_width,
            paper_figure_fusion_metadata(
                FUSION_FIGURE1_FULL_OUTPUT_AVERAGE, output_width
            ),
        ),
        FIGURE2_REPRESENTATION: (
            2 * output_width,
            paper_figure_fusion_metadata(
                FUSION_FIGURE2_FULL_OUTPUT_CONCATENATION, output_width
            ),
        ),
    }
    representations = {
        name: (
            np.memmap(
                directory / f"{name}.dat",
                dtype=np.float32,
                mode="w+",
                shape=(n_samples, width),
            ),
            metadata,
        )
        for name, (width, metadata) in specifications.items()
    }
    for start in range(0, n_samples, batch_size):
        stop = min(start + batch_size, n_samples)
        output1 = np.asarray(view1[start:stop])
        output2 = np.asarray(view2[start:stop])
        latent1 = output1[:, :hidden_width]
        latent2 = output2[:, :hidden_width]
        representations[EQ4_REPRESENTATION][0][start:stop] = fused_representation(
            latent1, latent2, str(fusion_contract)
        )
        representations[FIGURE1_REPRESENTATION][0][start:stop] = (output1 + output2) / 2
        representations[FIGURE2_REPRESENTATION][0][start:stop, :output_width] = output1
        representations[FIGURE2_REPRESENTATION][0][start:stop, output_width:] = output2
    for representation, _ in representations.values():
        representation.flush()
    return representations


def _assignment_frame(
    manifest: dict[str, object],
    representation: str,
    labels: np.ndarray,
    true_labels: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": manifest["dataset"],
            "protocol_id": manifest["protocol_id"],
            "run_id": manifest["run_id"],
            "training_commit": manifest["training_commit"],
            "source_id": manifest["source_id"],
            "source_sha256": manifest["source_sha256"],
            "representation": representation,
            "sample_index": np.arange(len(labels), dtype=int),
            "random_seed": manifest["seed"],
            "cluster": labels,
            "true_label": true_labels,
        }
    )


def run_paper_figure_diagnostic(
    manifest_path: Path,
    dataset_root: Path,
    output_dir: Path,
    *,
    inference_batch_size: int = 256,
    expected_seed: int = 42,
    model_loader: ModelLoader | None = None,
    dataset_loader: DatasetLoader | None = None,
) -> pd.DataFrame:
    """Audit one complete run and evaluate Eq. 4 plus both figure interpretations."""

    manifest = load_run_manifest(manifest_path)
    validate_manifest_identity(manifest)
    validate_manifest_outputs(manifest)
    validate_numerical_runtime(manifest)
    config = _validate_diagnostic_config(manifest, inference_batch_size, expected_seed)
    dataset = (dataset_loader or _load_canonical_dataset)(manifest, dataset_root)
    dataset_checks = (
        ("source_id", dataset.source_id),
        ("source_sha256", dataset.source_sha256),
        ("n_samples", len(dataset.X)),
        ("n_features", dataset.X.shape[1]),
        ("n_classes", dataset.spec.n_clusters),
    )
    for field, actual in dataset_checks:
        if manifest.get(field) != actual:
            raise ValueError(f"Canonical dataset {field} does not match the manifest.")
    if dataset.X.shape != (int(config["n_samples"]), int(config["n_features"])):
        raise ValueError("Canonical feature matrix shape does not match the config.")
    if len(dataset.y) != len(dataset.X):
        raise ValueError("Canonical labels are not row-aligned with features.")
    if not np.isfinite(dataset.X).all():
        raise ValueError("Canonical features contain non-finite values.")
    if len(np.unique(dataset.y)) != int(config["n_clusters"]):
        raise ValueError("Canonical label count does not match n_clusters.")

    view1_weights = resolve_manifest_output_path(manifest, "final_view1")
    view2_weights = resolve_manifest_output_path(manifest, "final_view2")
    view1, view2 = (model_loader or _default_model_loader)(
        config, view1_weights, view2_weights
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_width = int(config["hidden_units"]) + int(config["input_shape"])

    records = []
    output_files: dict[str, Path] = {}
    representation_metadata: dict[str, dict[str, object]] = {}
    with tempfile.TemporaryDirectory(prefix="mvdec-paper-figures-") as temporary:
        temporary_dir = Path(temporary)
        model_outputs: tuple[np.memmap, ...] = ()
        representations: dict[str, tuple[np.memmap, dict[str, object]]] = {}
        try:
            model_outputs = _write_model_outputs(
                dataset.X,
                view1,
                view2,
                output_width,
                inference_batch_size,
                temporary_dir,
            )
            representations = _materialize_representations(
                *model_outputs,
                config,
                inference_batch_size,
                temporary_dir,
            )
            for name, (representation, metadata) in representations.items():
                labels = KMeans(
                    n_clusters=int(config["n_clusters"]),
                    n_init=KMEANS_N_INIT,
                    random_state=int(manifest["seed"]),
                    copy_x=False,
                    algorithm="lloyd",
                ).fit_predict(representation)
                metrics = compute_external_metrics(
                    dataset.y,
                    labels,
                    n_clusters=int(config["n_clusters"]),
                )
                records.append(
                    {
                        "dataset": manifest["dataset"],
                        "protocol_id": manifest["protocol_id"],
                        "run_id": manifest["run_id"],
                        "training_commit": manifest["training_commit"],
                        "source_id": manifest["source_id"],
                        "source_sha256": manifest["source_sha256"],
                        "representation": name,
                        "paper_source": metadata["paper_source"],
                        "operation": metadata["operation"],
                        "fusion_contract": metadata["fusion_contract"],
                        "claim_scope": metadata["claim_scope"],
                        "n_samples": len(dataset.y),
                        "input_width_per_view": metadata["input_width_per_view"],
                        "representation_width": metadata["output_width"],
                        "seed": manifest["seed"],
                        "kmeans_n_init": KMEANS_N_INIT,
                        "acc": metrics.acc,
                        "nmi": metrics.nmi,
                    }
                )
                assignment_path = output_dir / f"{name}_assignments.csv"
                write_public_frame(
                    _assignment_frame(manifest, name, labels, dataset.y),
                    assignment_path,
                )
                output_files[f"{name}_assignments"] = assignment_path
                representation_metadata[name] = dict(metadata)
        finally:
            for representation, _ in representations.values():
                representation._mmap.close()
            for model_output in model_outputs:
                model_output._mmap.close()

    metrics = pd.DataFrame.from_records(records)
    metrics_path = output_dir / "paper_figure_diagnostic_metrics.csv"
    write_public_frame(metrics, metrics_path)
    output_files["metrics"] = metrics_path

    metadata_path = output_dir / "paper_figure_diagnostic_metadata.json"
    metadata_path.write_text(
        json.dumps(representation_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_files["metadata"] = metadata_path
    diagnostic_manifest = {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "claim_scope": DIAGNOSTIC_CLAIM_SCOPE,
        "source_run": {
            "dataset": manifest["dataset"],
            "protocol_id": manifest["protocol_id"],
            "run_id": manifest["run_id"],
            "training_commit": manifest["training_commit"],
            "config_hash": manifest["config_hash"],
            "seed": manifest["seed"],
            "source_id": manifest["source_id"],
            "source_sha256": manifest["source_sha256"],
            "final_view1_sha256": manifest["output_sha256"]["final_view1"],
            "final_view2_sha256": manifest["output_sha256"]["final_view2"],
        },
        "runtime": {
            "numpy_version": np.__version__,
            "scikit_learn_version": sklearn_version,
            "inference_batch_size": inference_batch_size,
            "kmeans_n_init": KMEANS_N_INIT,
        },
        "representations": representation_metadata,
        "outputs": {
            name: {
                "path": path.name,
                "sha256": file_sha256(path),
            }
            for name, path in sorted(output_files.items())
        },
    }
    (output_dir / "paper_figure_diagnostic_manifest.json").write_text(
        json.dumps(diagnostic_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEKM_DATASET_DIR,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inference-batch-size", type=int, default=256)
    parser.add_argument("--expected-seed", type=int, default=42)
    args = parser.parse_args()
    frame = run_paper_figure_diagnostic(
        args.manifest,
        args.dataset_root,
        args.output_dir,
        inference_batch_size=args.inference_batch_size,
        expected_seed=args.expected_seed,
    )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
