"""Audit saved MvDEC latent spaces without rerunning neural-network training."""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from sklearn import __version__ as sklearn_version
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from pipeline.data import validate_mvdec_artifact
from pipeline.external_metrics import compute_external_metrics
from pipeline.mvdec_audit import (
    validate_expected_run_set,
    validate_numerical_runtime,
)
from pipeline.mvdec_calibration import (
    CenteredRmsCalibration,
    center_latent,
    global_rms,
)
from pipeline.mvdec_contract import (
    FUSION_ENCODER_AVERAGE,
    FUSION_ENCODER_CENTERED_RMS_CALIBRATED_AVERAGE,
    FUSION_ENCODER_L2_NORMALIZED_AVERAGE,
    FUSION_FIGURE1_FULL_OUTPUT_AVERAGE,
    FUSION_FIGURE2_FULL_OUTPUT_CONCATENATION,
    PAPER_FIGURE_FUSION_CONTRACTS,
    paper_figure_fusion_metadata,
)
from pipeline.mvdec_runs import (
    RUN_MANIFEST_FILENAME,
    load_run_manifest,
    resolve_manifest_output_path,
    validate_manifest_identity,
    validate_manifest_outputs,
)
from pipeline.public_artifacts import load_public_artifact, write_public_frame

INDEPENDENT_KMEANS_METRIC_ATOL = 0.01
POST_TRAINING_DIAGNOSTIC_CLAIM_SCOPE = "post_training_diagnostic_not_exact_paper_truth"


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


@dataclass(frozen=True)
class ScaleAlignmentDiagnostics:
    """Label-free scale and orthogonal-alignment calibration metadata."""

    view1_centered_global_rms: float
    view2_centered_global_rms: float
    view2_to_view1_centered_rms_ratio: float
    view2_centered_rms_scale_factor: float
    procrustes_rotation_determinant: float
    procrustes_orthogonality_max_abs_error: float
    operation_order_max_abs_delta: float


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


def _center(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float64)
    return center_latent(array), array.mean(axis=0)


def _global_rms(values: np.ndarray) -> float:
    return global_rms(values)


def _centered_rms_scale_factor(
    reference_centered: np.ndarray,
    target_centered: np.ndarray,
) -> float:
    reference_rms = _global_rms(reference_centered)
    target_rms = _global_rms(target_centered)
    if reference_rms == 0.0 or target_rms == 0.0:
        raise ValueError(
            "Centered RMS calibration requires non-constant latent spaces."
        )
    return reference_rms / target_rms


def _orthogonal_alignment(
    reference_centered: np.ndarray,
    target_centered: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rotation, _ = orthogonal_procrustes(target_centered, reference_centered)
    return target_centered @ rotation, rotation


def scale_alignment_representations(
    view1: np.ndarray,
    view2: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return label-free scale/alignment controls using paired sample rows."""

    centered1, mean1 = _center(np.asarray(view1, dtype=np.float64))
    centered2, _ = _center(np.asarray(view2, dtype=np.float64))
    scale = _centered_rms_scale_factor(centered1, centered2)
    rms_matched2 = centered2 * scale
    procrustes2, _ = _orthogonal_alignment(centered1, centered2)
    scaled_then_aligned2, _ = _orthogonal_alignment(centered1, rms_matched2)
    aligned_then_scaled2 = procrustes2 * _centered_rms_scale_factor(
        centered1,
        procrustes2,
    )
    return {
        "centered_global_rms_matched_average": (view1 + rms_matched2 + mean1) / 2,
        "centered_rms_then_procrustes_average": (view1 + scaled_then_aligned2 + mean1)
        / 2,
        "centered_procrustes_then_rms_average": (view1 + aligned_then_scaled2 + mean1)
        / 2,
    }


def scale_alignment_diagnostics(
    view1: np.ndarray,
    view2: np.ndarray,
) -> ScaleAlignmentDiagnostics:
    """Describe label-free calibration and verify operation-order equivalence."""

    centered1, _ = _center(np.asarray(view1, dtype=np.float64))
    centered2, _ = _center(np.asarray(view2, dtype=np.float64))
    view1_rms = _global_rms(centered1)
    view2_rms = _global_rms(centered2)
    scale = _centered_rms_scale_factor(centered1, centered2)
    aligned2, rotation = _orthogonal_alignment(centered1, centered2)
    scaled_then_aligned2, _ = _orthogonal_alignment(
        centered1,
        centered2 * scale,
    )
    aligned_then_scaled2 = aligned2 * _centered_rms_scale_factor(
        centered1,
        aligned2,
    )
    identity = np.eye(rotation.shape[0], dtype=np.float64)
    return ScaleAlignmentDiagnostics(
        view1_centered_global_rms=view1_rms,
        view2_centered_global_rms=view2_rms,
        view2_to_view1_centered_rms_ratio=view2_rms / view1_rms,
        view2_centered_rms_scale_factor=scale,
        procrustes_rotation_determinant=float(np.linalg.det(rotation)),
        procrustes_orthogonality_max_abs_error=float(
            np.max(np.abs(rotation.T @ rotation - identity))
        ),
        operation_order_max_abs_delta=float(
            np.max(np.abs(scaled_then_aligned2 - aligned_then_scaled2))
        ),
    )


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
    calibration: CenteredRmsCalibration | dict[str, object] | None = None,
) -> np.ndarray:
    """Recompute the latent fusion declared by one immutable protocol."""

    if fusion_contract == FUSION_ENCODER_AVERAGE:
        return (view1 + view2) / 2
    if fusion_contract == FUSION_ENCODER_L2_NORMALIZED_AVERAGE:
        return (_row_l2_normalize(view1) + _row_l2_normalize(view2)) / 2
    if fusion_contract == FUSION_ENCODER_CENTERED_RMS_CALIBRATED_AVERAGE:
        resolved = (
            calibration
            if isinstance(calibration, CenteredRmsCalibration)
            else CenteredRmsCalibration.from_metadata(calibration)
        )
        resolved.validate()
        if view1.shape != (resolved.n_samples, resolved.latent_width):
            raise ValueError("Calibration dimensions do not match audited latents.")
        return (view1 + resolved.view2_scale_factor * view2) / 2
    raise ValueError(f"Unsupported MvDEC fusion contract: {fusion_contract!r}.")


def paper_figure_fused_representation(
    view1_output: np.ndarray,
    view2_output: np.ndarray,
    fusion_contract: str,
) -> tuple[np.ndarray, dict[str, object]]:
    """Apply one figure-derived full-output fusion as a post-training diagnostic."""

    view1_output = np.asarray(view1_output)
    view2_output = np.asarray(view2_output)
    if (
        view1_output.ndim != 2
        or view2_output.ndim != 2
        or view1_output.shape != view2_output.shape
    ):
        raise ValueError(
            "MvDEC paper-figure fusion requires equal two-dimensional view outputs."
        )
    if not np.isfinite(view1_output).all() or not np.isfinite(view2_output).all():
        raise ValueError("MvDEC paper-figure view outputs must be finite.")

    metadata = paper_figure_fusion_metadata(
        fusion_contract,
        view_output_width=view1_output.shape[1],
    )
    if fusion_contract == FUSION_FIGURE1_FULL_OUTPUT_AVERAGE:
        fused = (view1_output + view2_output) / 2
    elif fusion_contract == FUSION_FIGURE2_FULL_OUTPUT_CONCATENATION:
        fused = np.concatenate((view1_output, view2_output), axis=1)
    else:  # pragma: no cover - metadata validation owns this branch.
        raise AssertionError("Unhandled paper-figure fusion contract.")
    if fused.shape[1] != metadata["output_width"]:
        raise AssertionError("Paper-figure fusion width does not match its contract.")
    return fused, metadata


def paper_figure_diagnostic_representations(
    view1_output: np.ndarray,
    view2_output: np.ndarray,
) -> dict[str, tuple[np.ndarray, dict[str, object]]]:
    """Return both conflicting figure interpretations under stable names."""

    return {
        fusion_contract: paper_figure_fused_representation(
            view1_output,
            view2_output,
            fusion_contract,
        )
        for fusion_contract in PAPER_FIGURE_FUSION_CONTRACTS
    }


def diagnostic_representations(
    view1: np.ndarray,
    view2: np.ndarray,
    fusion_contract: str = FUSION_ENCODER_AVERAGE,
    calibration: CenteredRmsCalibration | dict[str, object] | None = None,
) -> dict[str, np.ndarray]:
    """Return declared fusion plus label-free controls for alignment faults."""

    centered1 = view1 - view1.mean(axis=0)
    centered2 = view2 - view2.mean(axis=0)
    rotation, _ = orthogonal_procrustes(centered2, centered1)
    aligned2 = centered2 @ rotation + view1.mean(axis=0)
    return {
        "view1": view1,
        "view2": view2,
        "fused": fused_representation(view1, view2, fusion_contract, calibration),
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
        **scale_alignment_representations(view1, view2),
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
    """Audit one restricted artifact and export independent assignments."""

    artifact = load_public_artifact(artifact_path)
    if not isinstance(artifact, dict):
        raise ValueError(f"MvDEC artifact is not a mapping: {artifact_path}")
    view1, view2, fused = _validated_latents(artifact)
    fusion_contract = artifact.get("fusion_contract")
    if not isinstance(fusion_contract, str):
        raise ValueError("MvDEC artifact is missing its fusion contract.")
    calibration_metadata = artifact.get("calibration")
    expected_fused = fused_representation(
        view1,
        view2,
        fusion_contract,
        calibration_metadata,
    )
    fusion_matches = (
        np.array_equal(fused, expected_fused)
        if fusion_contract == FUSION_ENCODER_CENTERED_RMS_CALIBRATED_AVERAGE
        else np.allclose(fused, expected_fused, rtol=1e-6, atol=1e-6)
    )
    if not fusion_matches:
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
        view1,
        view2,
        fusion_contract,
        calibration_metadata,
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
        recorded_acc = None
        recorded_nmi = None
        acc_delta = None
        nmi_delta = None
        if name in {"view1", "view2", "fused"}:
            recorded_acc = artifact.get(f"{name}_acc")
            recorded_nmi = artifact.get(f"{name}_nmi")
            if (recorded_acc is None) != (recorded_nmi is None):
                raise ValueError(f"Recorded MvDEC {name} metrics are incomplete.")
            if recorded_acc is not None:
                recorded_acc = float(recorded_acc)
                recorded_nmi = float(recorded_nmi)
                if not all(
                    np.isfinite(value) and 0.0 <= value <= 1.0
                    for value in (recorded_acc, recorded_nmi)
                ):
                    raise ValueError(f"Recorded MvDEC {name} metrics are invalid.")
                acc_delta = metrics.acc - recorded_acc
                nmi_delta = metrics.nmi - recorded_nmi
                if max(abs(acc_delta), abs(nmi_delta)) > (
                    INDEPENDENT_KMEANS_METRIC_ATOL
                ):
                    raise ValueError(
                        f"Recorded MvDEC {name} metrics diverge materially from "
                        "independent K-Means."
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
                "claim_scope": (
                    "saved_training_representation"
                    if name in {"view1", "view2", "fused"}
                    else POST_TRAINING_DIAGNOSTIC_CLAIM_SCOPE
                ),
                "acc": metrics.acc,
                "nmi": metrics.nmi,
                "recorded_acc": recorded_acc,
                "recorded_nmi": recorded_nmi,
                "acc_delta": acc_delta,
                "nmi_delta": nmi_delta,
                "independent_metric_atol": INDEPENDENT_KMEANS_METRIC_ATOL,
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
    scale_diagnostics = scale_alignment_diagnostics(view1, view2)
    (dataset_output / "scale_alignment_diagnostics.json").write_text(
        json.dumps(asdict(scale_diagnostics), indent=2, sort_keys=True) + "\n",
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
    frames = []
    for manifest in manifests:
        validate_numerical_runtime(manifest)
        artifact_path = resolve_manifest_output_path(manifest, "artifact")
        validate_mvdec_artifact(artifact_path)
        frame = audit_artifact(artifact_path, output_dir)[0]
        runtime = manifest["runtime"]
        frame["training_numpy_version"] = runtime["numpy_version"]
        frame["training_scikit_learn_version"] = runtime["scikit_learn_version"]
        frame["audit_numpy_version"] = np.__version__
        frame["audit_scikit_learn_version"] = sklearn_version
        frames.append(frame)
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
        help=(
            "Deprecated compatibility flag; artifacts are always loaded with a "
            "restricted NumPy-only unpickler."
        ),
    )
    args = parser.parse_args()
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
