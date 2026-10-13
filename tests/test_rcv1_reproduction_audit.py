"""Tests for the RCV1 DEKM reproduction audit."""

import csv
import io
from pathlib import Path

import numpy as np
import pytest

from pipeline.rcv1_reproduction_audit import (
    audit_assignment_csv,
    audit_split_files,
    compare_assignment_metrics,
    runtime_comparison,
)


def test_split_audit_records_disjoint_release_indices(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "RCV1"
    dataset_dir.mkdir()
    (dataset_dir / "test").write_text("0\n2\n", encoding="utf-8")
    (dataset_dir / "validation").write_text("1\n3\n", encoding="utf-8")

    audit = audit_split_files(dataset_dir, source_rows=4)

    assert audit.test_rows == 2
    assert audit.validation_rows == 2
    assert audit.selected_rows == 4
    assert audit.unique_rows == 4
    assert audit.overlap_rows == 0
    assert len(audit.indices_sha256) == 64


def test_split_audit_rejects_out_of_bounds_indices(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "RCV1"
    dataset_dir.mkdir()
    (dataset_dir / "test").write_text("0\n4\n", encoding="utf-8")
    (dataset_dir / "validation").write_text("1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="outside"):
        audit_split_files(dataset_dir, source_rows=4)


def test_assignment_audit_recomputes_metrics_and_checks_labels() -> None:
    stream = io.StringIO()
    writer = csv.DictWriter(
        stream,
        fieldnames=(
            "dataset",
            "method",
            "sample_index",
            "random_seed",
            "cluster",
            "true_label",
        ),
    )
    writer.writeheader()
    truth = np.array([0, 0, 1, 1])
    for index, (cluster, label) in enumerate(zip([1, 1, 0, 0], truth, strict=True)):
        writer.writerow(
            {
                "dataset": "RCV1",
                "method": "DEKM_F",
                "sample_index": index,
                "random_seed": 42,
                "cluster": cluster,
                "true_label": label,
            }
        )

    audit = audit_assignment_csv(stream.getvalue(), truth, n_clusters=2)

    assert audit.seed == 42
    assert audit.acc == 1.0
    assert audit.nmi == 1.0
    assert audit.true_labels_match_dataset is True


def test_assignment_metrics_are_compared_with_matching_seed_manifest() -> None:
    stream = io.StringIO()
    writer = csv.DictWriter(
        stream,
        fieldnames=(
            "dataset",
            "method",
            "sample_index",
            "random_seed",
            "cluster",
            "true_label",
        ),
    )
    writer.writeheader()
    for index, label in enumerate([0, 0, 1, 1]):
        writer.writerow(
            {
                "dataset": "RCV1",
                "method": "DEKM_F",
                "sample_index": index,
                "random_seed": 42,
                "cluster": label,
                "true_label": label,
            }
        )
    assignment = audit_assignment_csv(
        stream.getvalue(),
        np.array([0, 0, 1, 1]),
        n_clusters=2,
    )

    checks = compare_assignment_metrics(
        [assignment],
        {42: {"metrics": {"acc": 1.0, "nmi": 1.0}}},
    )

    assert checks == [
        {
            "seed": 42,
            "recomputed_acc": 1.0,
            "reported_acc": 1.0,
            "acc_matches": True,
            "recomputed_nmi": 1.0,
            "reported_nmi": 1.0,
            "nmi_matches": True,
            "all_metrics_match": True,
        }
    ]


def test_runtime_comparison_exposes_dependency_mismatch() -> None:
    comparison = runtime_comparison(
        {
            "runtime": {
                "python": "3.13",
                "platform": "Linux",
                "tensorflow": "2.20.0",
                "scikit_learn": "1.6.1",
                "numpy": "2.1.3",
            }
        }
    )

    assert comparison["exact_match"] is False
    assert comparison["paper"]["tensorflow"] == "2.4.1"
    assert comparison["observed"]["scipy"] == "not_recorded"
