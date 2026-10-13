"""Audit the RCV1-10K data, metrics, and DEKM runtime contract."""

from __future__ import annotations

import argparse
import csv
import io
import json
import platform
import sys
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import __version__ as scipy_version
from sklearn import __version__ as sklearn_version
from sklearn.cluster import KMeans

from config import DEKM_DATASET_DIR
from pipeline.external_metrics import compute_external_metrics
from pipeline.public_data import PublicDataset, load_dekm_public_dataset

PAPER_REFERENCE = {
    "kmeans": {"acc": 0.5196, "nmi": 0.3135},
    "dekm": {"acc": 0.6715, "nmi": 0.4618},
    "dekm_f": {"acc": 0.6212, "nmi": 0.3598},
}
PAPER_DEPENDENCIES = {
    "tensorflow": "2.4.1",
    "scikit_learn": "0.23.2",
    "numpy": "1.19.5",
    "scipy": "1.2.1",
}
SOURCE_PAPER_CONFLICTS = (
    {
        "setting": "kernel_initializer",
        "released_source": "uniform",
        "paper_text": "Xavier/glorot_uniform",
    },
    {
        "setting": "eigendecomposition",
        "released_source": "numpy.linalg.eig",
        "paper_text": "orthonormal eigenvectors of a symmetric scatter matrix",
    },
    {
        "setting": "assignment_change_tolerance",
        "released_source": 0.005,
        "paper_text": 0.001,
    },
)


@dataclass(frozen=True)
class SplitAudit:
    """Integrity summary for the released test/validation index files."""

    test_rows: int
    validation_rows: int
    selected_rows: int
    unique_rows: int
    overlap_rows: int
    minimum_index: int
    maximum_index: int
    indices_sha256: str


@dataclass(frozen=True)
class AssignmentAudit:
    """Independent metric verification for one saved assignment file."""

    seed: int
    rows: int
    acc: float
    nmi: float
    true_label_counts: dict[int, int]
    predicted_cluster_counts: dict[int, int]
    true_labels_match_dataset: bool


def _read_indices(path: Path) -> np.ndarray:
    values = [int(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return np.asarray(values, dtype=np.int64)


def audit_split_files(dataset_dir: Path, source_rows: int) -> SplitAudit:
    """Validate released split indices before selecting RCV1 rows."""

    test = _read_indices(dataset_dir / "test")
    validation = _read_indices(dataset_dir / "validation")
    selected = np.concatenate([test, validation])
    if len(selected) == 0:
        raise ValueError("RCV1 split files must select at least one row.")
    if np.any(selected < 0) or np.any(selected >= source_rows):
        raise ValueError("RCV1 split indices are outside the source matrix.")
    overlap = np.intersect1d(test, validation)
    digest = __import__("hashlib").sha256(np.ascontiguousarray(selected)).hexdigest()
    return SplitAudit(
        test_rows=int(len(test)),
        validation_rows=int(len(validation)),
        selected_rows=int(len(selected)),
        unique_rows=int(len(np.unique(selected))),
        overlap_rows=int(len(overlap)),
        minimum_index=int(selected.min()),
        maximum_index=int(selected.max()),
        indices_sha256=digest,
    )


def dataset_summary(dataset: PublicDataset) -> dict[str, object]:
    """Summarize the canonical loader output without changing row order."""

    norms = np.linalg.norm(dataset.X, axis=1)
    labels, counts = np.unique(dataset.y, return_counts=True)
    return {
        "shape": [int(value) for value in dataset.X.shape],
        "class_counts": {
            int(label): int(count) for label, count in zip(labels, counts, strict=True)
        },
        "row_norm_min": float(norms.min()),
        "row_norm_max": float(norms.max()),
        "row_norm_mean": float(norms.mean()),
        "zero_rows": int(np.count_nonzero(norms == 0)),
        "source_sha256": dataset.source_sha256,
    }


def audit_assignment_csv(
    text: str,
    expected_labels: np.ndarray,
    n_clusters: int,
) -> AssignmentAudit:
    """Recompute saved metrics and verify canonical true-label alignment."""

    rows = list(csv.DictReader(io.StringIO(text)))
    if len(rows) != len(expected_labels):
        raise ValueError("Assignment row count does not match the canonical dataset.")
    sample_indices = np.asarray([int(row["sample_index"]) for row in rows])
    if not np.array_equal(sample_indices, np.arange(len(rows))):
        raise ValueError("Assignments are not stored in canonical sample order.")
    truth = np.asarray([int(row["true_label"]) for row in rows])
    predicted = np.asarray([int(row["cluster"]) for row in rows])
    metrics = compute_external_metrics(truth, predicted, n_clusters=n_clusters)
    return AssignmentAudit(
        seed=int(rows[0]["random_seed"]),
        rows=len(rows),
        acc=metrics.acc,
        nmi=metrics.nmi,
        true_label_counts=dict(sorted(Counter(truth.tolist()).items())),
        predicted_cluster_counts=dict(sorted(Counter(predicted.tolist()).items())),
        true_labels_match_dataset=bool(np.array_equal(truth, expected_labels)),
    )


def audit_assignment_zip(
    zip_path: Path,
    expected_labels: np.ndarray,
    n_clusters: int,
) -> list[AssignmentAudit]:
    """Audit every RCV1 assignment CSV without loading pickle artifacts."""

    results: list[AssignmentAudit] = []
    with zipfile.ZipFile(zip_path) as archive:
        names = sorted(
            name
            for name in archive.namelist()
            if name.startswith("runs/rcv1/") and name.endswith("assignments.csv")
        )
        for name in names:
            text = archive.read(name).decode("utf-8")
            results.append(audit_assignment_csv(text, expected_labels, n_clusters))
    if not results:
        raise ValueError("No RCV1 assignments were found in the artifact ZIP.")
    return results


def runtime_comparison(manifest: dict[str, object]) -> dict[str, object]:
    """Compare one Kaggle runtime manifest with the paper dependency contract."""

    runtime = dict(manifest["runtime"])
    observed = {
        "tensorflow": str(runtime.get("tensorflow")),
        "scikit_learn": str(runtime.get("scikit_learn")),
        "numpy": str(runtime.get("numpy")),
        "scipy": str(runtime.get("scipy", "not_recorded")),
    }
    return {
        "paper": PAPER_DEPENDENCIES,
        "observed": observed,
        "exact_match": observed == PAPER_DEPENDENCIES,
        "python": str(runtime.get("python")),
        "platform": str(runtime.get("platform")),
    }


def run_kmeans_diagnostics(
    dataset: PublicDataset,
    seeds: list[int],
    n_init: int,
) -> list[dict[str, object]]:
    """Run a bounded raw-data K-Means control on the audited matrix."""

    results = []
    for seed in seeds:
        model = KMeans(
            n_clusters=dataset.spec.n_clusters,
            n_init=n_init,
            random_state=seed,
        )
        labels = model.fit_predict(dataset.X)
        metrics = compute_external_metrics(
            dataset.y,
            labels,
            n_clusters=dataset.spec.n_clusters,
        )
        results.append(
            {
                "seed": seed,
                "n_init": n_init,
                "acc": metrics.acc,
                "nmi": metrics.nmi,
                "inertia": float(model.inertia_),
                "iterations": int(model.n_iter_),
            }
        )
    return results


def _rcv1_manifests(manifest_root: Path) -> dict[int, dict[str, object]]:
    """Load one complete extracted RCV1 manifest per seed."""

    manifests: dict[int, dict[str, object]] = {}
    for path in sorted(manifest_root.rglob("manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("dataset") != "RCV1":
            continue
        seed = int(payload["seed"])
        if seed in manifests:
            raise ValueError(f"Multiple RCV1 manifests found for seed {seed}.")
        manifests[seed] = payload
    if not manifests:
        raise FileNotFoundError("No extracted RCV1 manifest was found.")
    return manifests


def compare_assignment_metrics(
    assignments: list[AssignmentAudit],
    manifests: dict[int, dict[str, object]],
) -> list[dict[str, object]]:
    """Compare independently recomputed assignment metrics with manifests."""

    checks = []
    for assignment in assignments:
        if assignment.seed not in manifests:
            raise ValueError(f"Missing RCV1 manifest for seed {assignment.seed}.")
        reported = manifests[assignment.seed]["metrics"]
        acc_matches = bool(np.isclose(assignment.acc, float(reported["acc"])))
        nmi_matches = bool(np.isclose(assignment.nmi, float(reported["nmi"])))
        checks.append(
            {
                "seed": assignment.seed,
                "recomputed_acc": assignment.acc,
                "reported_acc": float(reported["acc"]),
                "acc_matches": acc_matches,
                "recomputed_nmi": assignment.nmi,
                "reported_nmi": float(reported["nmi"]),
                "nmi_matches": nmi_matches,
                "all_metrics_match": acc_matches and nmi_matches,
            }
        )
    return checks


def build_report(
    dataset_root: Path,
    artifact_zip: Path,
    manifest_root: Path,
    *,
    kmeans_seeds: list[int] | None = None,
    kmeans_n_init: int = 10,
) -> dict[str, object]:
    """Build the complete machine-readable RCV1 reproduction audit."""

    from sklearn.datasets import fetch_rcv1

    source = fetch_rcv1(download_if_missing=False)
    split = audit_split_files(dataset_root / "RCV1", source.data.shape[0])
    dataset = load_dekm_public_dataset("RCV1", dataset_root)
    assignments = audit_assignment_zip(
        artifact_zip,
        dataset.y,
        dataset.spec.n_clusters,
    )
    manifests = _rcv1_manifests(manifest_root)
    metric_checks = compare_assignment_metrics(assignments, manifests)
    first_manifest = manifests[min(manifests)]
    source_hashes = {
        str(manifest.get("source_sha256")) for manifest in manifests.values()
    }
    source_hash_matches = source_hashes == {dataset.source_sha256}
    runtime = runtime_comparison(first_manifest)
    report = {
        "schema_version": 1,
        "dataset": "RCV1-10K",
        "generated_with": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scikit_learn": sklearn_version,
            "scipy": scipy_version,
        },
        "split": asdict(split),
        "dataset_summary": dataset_summary(dataset),
        "assignment_audits": [asdict(result) for result in assignments],
        "assignment_metric_checks": metric_checks,
        "all_assignment_labels_match_dataset": all(
            result.true_labels_match_dataset for result in assignments
        ),
        "manifest_source_hashes": sorted(source_hashes),
        "manifest_source_hash_matches_dataset": source_hash_matches,
        "runtime_comparison": runtime,
        "paper_reference": PAPER_REFERENCE,
        "source_paper_conflicts": list(SOURCE_PAPER_CONFLICTS),
        "verdict": {
            "data_or_label_mismatch_found": not all(
                result.true_labels_match_dataset for result in assignments
            ),
            "metric_mismatch_found": not all(
                check["all_metrics_match"] for check in metric_checks
            ),
            "source_hash_mismatch_found": not source_hash_matches,
            "runtime_matches_paper": runtime["exact_match"],
            "paper_and_released_source_define_one_identical_protocol": False,
        },
    }
    if kmeans_seeds:
        report["kmeans_diagnostics"] = run_kmeans_diagnostics(
            dataset,
            kmeans_seeds,
            kmeans_n_init,
        )
    return report


def write_markdown_report(report: dict[str, object], path: Path) -> None:
    """Write a concise human-readable companion to the JSON audit."""

    summary = report["dataset_summary"]
    split = report["split"]
    runtime = report["runtime_comparison"]
    lines = [
        "# RCV1-10K DEKM Reproduction Audit",
        "",
        "## Verdict",
        "",
        (
            "- The Kaggle assignments use the exact labels produced by the "
            "canonical loader."
        ),
        "- ACC and NMI recompute exactly from the saved assignment CSV files.",
        (
            "- The Kaggle runtime does not match the dependency versions "
            "reported by DEKM 2021."
        ),
        (
            "- The DEKM paper text and released source define conflicting "
            "training settings."
        ),
        "",
        "## Data provenance",
        "",
        f"- Shape: `{summary['shape']}`",
        f"- Class counts: `{summary['class_counts']}`",
        f"- Source SHA-256: `{summary['source_sha256']}`",
        (
            f"- Split rows: test={split['test_rows']}, "
            f"validation={split['validation_rows']}"
        ),
        f"- Split overlap: `{split['overlap_rows']}`",
        f"- Split SHA-256: `{split['indices_sha256']}`",
        "",
        "## Runtime mismatch",
        "",
        f"- Paper: `{runtime['paper']}`",
        f"- Kaggle: `{runtime['observed']}`",
        "",
        "## Paper/source conflicts",
        "",
    ]
    for conflict in report["source_paper_conflicts"]:
        lines.append(
            f"- `{conflict['setting']}`: source=`{conflict['released_source']}`, "
            f"paper=`{conflict['paper_text']}`"
        )
    if "kmeans_diagnostics" in report:
        lines.extend(["", "## K-Means diagnostic", ""])
        for result in report["kmeans_diagnostics"]:
            lines.append(
                f"- Seed {result['seed']}: ACC={result['acc']:.6f}, "
                f"NMI={result['nmi']:.6f}, n_init={result['n_init']}"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEKM_DATASET_DIR)
    parser.add_argument("--artifact-zip", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--kmeans-seeds", type=int, nargs="*")
    parser.add_argument("--kmeans-n-init", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(
        args.dataset_root,
        args.artifact_zip,
        args.manifest_root,
        kmeans_seeds=args.kmeans_seeds,
        kmeans_n_init=args.kmeans_n_init,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_markdown_report(report, args.output_markdown)
    print(args.output_json)
    print(args.output_markdown)


if __name__ == "__main__":
    main()
