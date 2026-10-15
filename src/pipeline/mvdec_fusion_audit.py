"""Audit saved MvDEC latent spaces without rerunning neural-network training."""

import argparse
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from pipeline.external_metrics import compute_external_metrics
from pipeline.mvdec_audit import validate_expected_run_set
from pipeline.mvdec_contract import (
    FUSION_ENCODER_AVERAGE,
    FUSION_ENCODER_L2_NORMALIZED_AVERAGE,
)
from pipeline.mvdec_runs import (
    RUN_MANIFEST_FILENAME,
    load_run_manifest,
    resolve_manifest_output_path,
    validate_manifest_identity,
    validate_manifest_outputs,
)
from pipeline.public_artifacts import write_public_frame


@dataclass(frozen=True)
class AlignmentDiagnostics:
    """Label-free diagnostics for whether two latent spaces can be averaged."""

    view1_mean_row_norm: float
    view2_mean_row_norm: float
    view2_to_view1_norm_ratio: float
    paired_cosine_mean: float
    paired_cosine_std: float
    paired_cosine_q05: float
    paired_cosine_median: float
    paired_cosine_q95: float
    mean_vector_cosine: float
    centered_paired_cosine_mean: float
    centered_paired_cosine_std: float
    fusion_retained_norm_mean: float
    fusion_retained_norm_q05: float
    fusion_retained_norm_median: float
    centered_fusion_retained_norm_mean: float
    same_coordinate_correlation_mean: float
    strongest_cross_coordinate_abs_correlation_mean: float


def _validated_latents(artifact: dict[str, object]) -> tuple[np.ndarray, ...]:
    arrays = tuple(
        np.asarray(artifact[name]) for name in ("h_view1", "h_view2", "h_fused")
    )
    if any(values.ndim != 2 or not np.isfinite(values).all() for values in arrays):
        raise ValueError("MvDEC latent arrays must be finite two-dimensional arrays.")
    if len({values.shape for values in arrays}) != 1:
        raise ValueError("MvDEC View1, View2, and fused arrays must share one shape.")
    if arrays[0].shape[0] < 2 or arrays[0].shape[1] < 1:
        raise ValueError(
            "MvDEC latent arrays require at least two rows and one column."
        )
    if np.any(np.linalg.norm(arrays[0], axis=1) == 0):
        raise ValueError("MvDEC View1 contains zero-norm latent rows.")
    if np.any(np.linalg.norm(arrays[1], axis=1) == 0):
        raise ValueError("MvDEC View2 contains zero-norm latent rows.")
    return arrays


def _row_l2_normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)


def _paired_cosine(view1: np.ndarray, view2: np.ndarray) -> np.ndarray:
    norm1 = np.linalg.norm(view1, axis=1)
    norm2 = np.linalg.norm(view2, axis=1)
    denominator = norm1 * norm2
    return np.divide(
        np.sum(view1 * view2, axis=1),
        denominator,
        out=np.zeros(len(view1), dtype=np.float64),
        where=denominator > 0,
    )


def _retained_norm(view1: np.ndarray, view2: np.ndarray) -> np.ndarray:
    norm1 = np.linalg.norm(view1, axis=1)
    norm2 = np.linalg.norm(view2, axis=1)
    component_norm_mean = (norm1 + norm2) / 2
    return np.divide(
        np.linalg.norm((view1 + view2) / 2, axis=1),
        component_norm_mean,
        out=np.zeros(len(view1), dtype=np.float64),
        where=component_norm_mean > 0,
    )


def _vector_cosine(view1: np.ndarray, view2: np.ndarray) -> float:
    denominator = np.linalg.norm(view1) * np.linalg.norm(view2)
    return float(np.dot(view1, view2) / denominator) if denominator > 0 else 0.0


def alignment_diagnostics(
    view1: np.ndarray,
    view2: np.ndarray,
) -> AlignmentDiagnostics:
    """Measure scale, orientation, and destructive cancellation without labels."""

    view1 = np.asarray(view1, dtype=np.float64)
    view2 = np.asarray(view2, dtype=np.float64)
    norm1 = np.linalg.norm(view1, axis=1)
    norm2 = np.linalg.norm(view2, axis=1)
    cosine = _paired_cosine(view1, view2)
    retained_norm = _retained_norm(view1, view2)
    centered1 = view1 - view1.mean(axis=0)
    centered2 = view2 - view2.mean(axis=0)
    centered_cosine = _paired_cosine(centered1, centered2)
    centered_retained_norm = _retained_norm(centered1, centered2)
    with np.errstate(invalid="ignore", divide="ignore"):
        correlations = np.corrcoef(centered1.T, centered2.T)
    width = view1.shape[1]
    cross_correlations = np.nan_to_num(
        correlations[:width, width:],
        nan=0.0,
    )
    return AlignmentDiagnostics(
        view1_mean_row_norm=float(norm1.mean()),
        view2_mean_row_norm=float(norm2.mean()),
        view2_to_view1_norm_ratio=float(norm2.mean() / norm1.mean()),
        paired_cosine_mean=float(cosine.mean()),
        paired_cosine_std=float(cosine.std()),
        paired_cosine_q05=float(np.quantile(cosine, 0.05)),
        paired_cosine_median=float(np.median(cosine)),
        paired_cosine_q95=float(np.quantile(cosine, 0.95)),
        mean_vector_cosine=_vector_cosine(view1.mean(axis=0), view2.mean(axis=0)),
        centered_paired_cosine_mean=float(centered_cosine.mean()),
        centered_paired_cosine_std=float(centered_cosine.std()),
        fusion_retained_norm_mean=float(retained_norm.mean()),
        fusion_retained_norm_q05=float(np.quantile(retained_norm, 0.05)),
        fusion_retained_norm_median=float(np.median(retained_norm)),
        centered_fusion_retained_norm_mean=float(centered_retained_norm.mean()),
        same_coordinate_correlation_mean=float(np.diag(cross_correlations).mean()),
        strongest_cross_coordinate_abs_correlation_mean=float(
            np.max(np.abs(cross_correlations), axis=1).mean()
        ),
    )


def fused_representation(
    view1: np.ndarray,
    view2: np.ndarray,
    fusion_contract: str,
) -> np.ndarray:
    """Recompute the latent fusion declared by one immutable protocol."""

    if fusion_contract == FUSION_ENCODER_AVERAGE:
        return (view1 + view2) / 2
    if fusion_contract == FUSION_ENCODER_L2_NORMALIZED_AVERAGE:
        return (_row_l2_normalize(view1) + _row_l2_normalize(view2)) / 2
    raise ValueError(f"Unsupported MvDEC fusion contract: {fusion_contract!r}.")


def diagnostic_representations(
    view1: np.ndarray,
    view2: np.ndarray,
    fusion_contract: str = FUSION_ENCODER_AVERAGE,
) -> dict[str, np.ndarray]:
    """Return declared fusion plus label-free controls for alignment faults."""

    centered1 = view1 - view1.mean(axis=0)
    centered2 = view2 - view2.mean(axis=0)
    rotation, _ = orthogonal_procrustes(centered2, centered1)
    aligned2 = centered2 @ rotation + view1.mean(axis=0)
    return {
        "view1": view1,
        "view2": view2,
        "fused": fused_representation(view1, view2, fusion_contract),
        "row_l2_normalized_average": (
            _row_l2_normalize(view1) + _row_l2_normalize(view2)
        )
        / 2,
        "zscore_average": (
            StandardScaler().fit_transform(view1)
            + StandardScaler().fit_transform(view2)
        )
        / 2,
        "procrustes_aligned_average": (view1 + aligned2) / 2,
        "concatenation_control": np.concatenate((view1, view2), axis=1),
    }


def _assignment_frame(
    artifact: dict[str, object],
    representation: str,
    labels: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": artifact["dataset"],
            "protocol_id": artifact["protocol_id"],
            "run_id": artifact["run_id"],
            "training_commit": artifact["training_commit"],
            "source_id": artifact["source_id"],
            "source_sha256": artifact["source_sha256"],
            "representation": representation,
            "sample_index": np.arange(len(labels)),
            "random_seed": artifact["config"]["seed"],
            "cluster": labels,
            "true_label": np.asarray(artifact["true_labels"]),
        }
    )


def audit_artifact(
    artifact_path: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, AlignmentDiagnostics]:
    """Audit one trusted local artifact and export independent assignments."""

    with artifact_path.open("rb") as file:
        artifact = pickle.load(file)  # noqa: S301 - trusted local CLI input.
    if not isinstance(artifact, dict):
        raise ValueError(f"MvDEC artifact is not a mapping: {artifact_path}")
    view1, view2, fused = _validated_latents(artifact)
    fusion_contract = artifact.get("fusion_contract")
    if not isinstance(fusion_contract, str):
        raise ValueError("MvDEC artifact is missing its fusion contract.")
    expected_fused = fused_representation(view1, view2, fusion_contract)
    if not np.allclose(fused, expected_fused, rtol=1e-6, atol=1e-6):
        raise ValueError("Saved fused latent does not implement its declared contract.")

    true_labels = np.asarray(artifact["true_labels"])
    if true_labels.ndim != 1 or len(true_labels) != len(view1):
        raise ValueError("MvDEC true labels must be one-dimensional and row-aligned.")
    n_clusters = int(artifact["n_clusters"])
    if not 2 <= n_clusters <= len(true_labels):
        raise ValueError("MvDEC cluster count must be between 2 and the sample count.")
    config = artifact.get("config")
    seed_value = config.get("seed") if isinstance(config, dict) else None
    if isinstance(seed_value, bool) or not isinstance(seed_value, (int, np.integer)):
        raise ValueError("MvDEC fusion audit requires an explicit integer seed.")
    seed = int(seed_value)
    n_init = int(artifact["kmeans_n_init"])
    if n_init != 100:
        raise ValueError("Independent MvDEC fusion audit requires K-Means n_init=100.")
    dataset_output = output_dir / str(artifact["dataset"]).lower()
    records = []
    for name, representation in diagnostic_representations(
        view1, view2, fusion_contract
    ).items():
        labels = KMeans(
            n_clusters=n_clusters,
            n_init=n_init,
            random_state=seed,
        ).fit_predict(representation)
        metrics = compute_external_metrics(
            true_labels,
            labels,
            n_clusters=n_clusters,
        )
        records.append(
            {
                "dataset": artifact["dataset"],
                "protocol_id": artifact["protocol_id"],
                "run_id": artifact["run_id"],
                "training_commit": artifact["training_commit"],
                "source_id": artifact["source_id"],
                "source_sha256": artifact["source_sha256"],
                "fusion_contract": fusion_contract,
                "representation": name,
                "acc": metrics.acc,
                "nmi": metrics.nmi,
                "n_samples": len(true_labels),
                "latent_width": representation.shape[1],
                "seed": seed,
                "kmeans_n_init": n_init,
            }
        )
        write_public_frame(
            _assignment_frame(artifact, name, labels),
            dataset_output / f"{name}_assignments.csv",
        )

    diagnostics = alignment_diagnostics(view1, view2)
    dataset_output.mkdir(parents=True, exist_ok=True)
    (dataset_output / "alignment_diagnostics.json").write_text(
        json.dumps(asdict(diagnostics), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metrics_frame = pd.DataFrame.from_records(records)
    write_public_frame(metrics_frame, dataset_output / "representation_metrics.csv")
    return metrics_frame, diagnostics


def audit_root(
    root: Path,
    output_dir: Path,
    expected_datasets: tuple[str, ...] | None = None,
    expected_protocol_id: str | None = None,
    expected_training_commit: str | None = None,
    expected_seed: int | None = None,
) -> pd.DataFrame:
    """Audit exactly the complete, hash-verified artifacts owned by manifests."""

    manifest_paths = sorted(root.rglob(RUN_MANIFEST_FILENAME))
    if not manifest_paths:
        raise ValueError(f"No MvDEC run manifests found below: {root}")
    manifests = [load_run_manifest(path) for path in manifest_paths]
    for manifest in manifests:
        validate_manifest_identity(manifest)
        validate_manifest_outputs(manifest)
    validate_expected_run_set(
        manifests,
        expected_datasets=expected_datasets,
        expected_protocol_id=expected_protocol_id,
        expected_training_commit=expected_training_commit,
        expected_seed=expected_seed,
        require_all_complete=True,
    )
    frames = [
        audit_artifact(
            resolve_manifest_output_path(manifest, "artifact"),
            output_dir,
        )[0]
        for manifest in manifests
    ]
    combined = pd.concat(frames, ignore_index=True)
    write_public_frame(combined, output_dir / "mvdec_fusion_audit_metrics.csv")
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-datasets", nargs="+")
    parser.add_argument("--expected-protocol-id")
    parser.add_argument("--expected-training-commit")
    parser.add_argument("--expected-seed", type=int)
    parser.add_argument(
        "--trust-pickle-artifacts",
        action="store_true",
        help="Required acknowledgement because pickle files can execute code.",
    )
    args = parser.parse_args()
    if not args.trust_pickle_artifacts:
        raise ValueError("Pass --trust-pickle-artifacts for trusted local outputs.")
    print(
        audit_root(
            args.root,
            args.output_dir,
            expected_datasets=(
                tuple(args.expected_datasets) if args.expected_datasets else None
            ),
            expected_protocol_id=args.expected_protocol_id,
            expected_training_commit=args.expected_training_commit,
            expected_seed=args.expected_seed,
        ).to_string(index=False)
    )


if __name__ == "__main__":
    main()
