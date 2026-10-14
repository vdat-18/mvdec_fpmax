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
    fusion_retained_norm_mean: float
    fusion_retained_norm_q05: float
    fusion_retained_norm_median: float
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
    return arrays


def _row_l2_normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)


def alignment_diagnostics(
    view1: np.ndarray,
    view2: np.ndarray,
) -> AlignmentDiagnostics:
    """Measure scale, orientation, and destructive cancellation without labels."""

    view1 = np.asarray(view1, dtype=np.float64)
    view2 = np.asarray(view2, dtype=np.float64)
    norm1 = np.linalg.norm(view1, axis=1)
    norm2 = np.linalg.norm(view2, axis=1)
    denominator = norm1 * norm2
    cosine = np.divide(
        np.sum(view1 * view2, axis=1),
        denominator,
        out=np.zeros(len(view1), dtype=np.float64),
        where=denominator > 0,
    )
    average_norm = np.linalg.norm((view1 + view2) / 2, axis=1)
    component_norm_mean = (norm1 + norm2) / 2
    retained_norm = np.divide(
        average_norm,
        component_norm_mean,
        out=np.zeros(len(view1), dtype=np.float64),
        where=component_norm_mean > 0,
    )
    centered1 = view1 - view1.mean(axis=0)
    centered2 = view2 - view2.mean(axis=0)
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
        fusion_retained_norm_mean=float(retained_norm.mean()),
        fusion_retained_norm_q05=float(np.quantile(retained_norm, 0.05)),
        fusion_retained_norm_median=float(np.median(retained_norm)),
        same_coordinate_correlation_mean=float(np.diag(cross_correlations).mean()),
        strongest_cross_coordinate_abs_correlation_mean=float(
            np.max(np.abs(cross_correlations), axis=1).mean()
        ),
    )


def diagnostic_representations(
    view1: np.ndarray,
    view2: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return paper fusion plus label-free controls that isolate alignment faults."""

    centered1 = view1 - view1.mean(axis=0)
    centered2 = view2 - view2.mean(axis=0)
    rotation, _ = orthogonal_procrustes(centered2, centered1)
    aligned2 = centered2 @ rotation + view1.mean(axis=0)
    return {
        "view1": view1,
        "view2": view2,
        "fused": (view1 + view2) / 2,
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
    expected_fused = (view1 + view2) / 2
    if not np.allclose(fused, expected_fused, rtol=1e-6, atol=1e-6):
        raise ValueError("Saved fused latent does not implement the declared average.")

    true_labels = np.asarray(artifact["true_labels"])
    n_clusters = int(artifact["n_clusters"])
    seed = int(artifact["config"]["seed"])
    n_init = int(artifact["kmeans_n_init"])
    dataset_output = output_dir / str(artifact["dataset"]).lower()
    records = []
    for name, representation in diagnostic_representations(view1, view2).items():
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


def audit_root(root: Path, output_dir: Path) -> pd.DataFrame:
    """Audit every artifact below a downloaded Kaggle output root."""

    artifacts = sorted(root.rglob("artifact.pkl"))
    if not artifacts:
        raise ValueError(f"No artifact.pkl files found below: {root}")
    frames = [audit_artifact(path, output_dir)[0] for path in artifacts]
    combined = pd.concat(frames, ignore_index=True)
    write_public_frame(combined, output_dir / "mvdec_fusion_audit_metrics.csv")
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--trust-pickle-artifacts",
        action="store_true",
        help="Required acknowledgement because pickle files can execute code.",
    )
    args = parser.parse_args()
    if not args.trust_pickle_artifacts:
        raise ValueError("Pass --trust-pickle-artifacts for trusted local outputs.")
    print(audit_root(args.root, args.output_dir).to_string(index=False))


if __name__ == "__main__":
    main()
