"""Tests for canonical public benchmark artifacts."""

import os
import pickle

import numpy as np
import pytest

from pipeline.external_metrics import compute_external_metrics
from pipeline.public_artifacts import (
    build_public_artifact,
    load_public_artifact,
    public_assignment_frame,
    restore_original_order,
    write_public_artifact,
    write_public_assignments,
)


class _ExecutablePayload:
    def __init__(self, marker_path) -> None:
        self.marker_path = marker_path

    def __reduce__(self):
        command = f'echo pickle-executed > "{self.marker_path}"'
        return os.system, (command,)


def test_public_artifact_and_assignments_use_canonical_row_order(tmp_path) -> None:
    """Shuffled deep-model outputs must be restored before persistence."""

    shuffled_indices = np.array([2, 0, 3, 1])
    shuffled_labels = np.array([1, 0, 1, 0])
    labels = restore_original_order(shuffled_labels, shuffled_indices)
    truth = np.array([0, 0, 1, 1])
    metrics = compute_external_metrics(truth, labels, n_clusters=2)
    artifact = build_public_artifact(
        dataset="TEST",
        method="MvDEC",
        n_clusters=2,
        random_seed=42,
        source_sha256="a" * 64,
        labels=labels,
        true_labels=truth,
        metrics=metrics,
        representation=np.arange(8).reshape(4, 2),
        representation_key="h_fused",
    )
    artifact_path = tmp_path / "run.pkl"
    assignments_path = tmp_path / "assignments.csv"
    write_public_artifact(artifact, artifact_path)
    write_public_assignments(
        public_assignment_frame(
            dataset="TEST",
            method="MvDEC",
            labels=labels,
            true_labels=truth,
            random_seed=42,
        ),
        assignments_path,
    )

    with artifact_path.open("rb") as file:
        saved = pickle.load(file)
    assert saved["row_indices"].tolist() == [0, 1, 2, 3]
    assert saved["labels"].tolist() == labels.tolist()
    assert assignments_path.read_text(encoding="utf-8").splitlines()[0] == (
        "dataset,method,sample_index,random_seed,cluster,true_label"
    )


def test_restricted_public_artifact_loader_accepts_numpy_payload(tmp_path) -> None:
    artifact_path = tmp_path / "artifact.pkl"
    expected = {
        "array": np.array([[1.0, 2.0]], dtype=np.float32),
        "scalar": np.float32(3.0),
    }
    artifact_path.write_bytes(pickle.dumps(expected))

    loaded = load_public_artifact(artifact_path)

    np.testing.assert_array_equal(loaded["array"], expected["array"])
    assert loaded["scalar"] == expected["scalar"]


def test_restricted_public_artifact_loader_rejects_executable_global(
    tmp_path,
) -> None:
    artifact_path = tmp_path / "malicious.pkl"
    marker_path = tmp_path / "executed.txt"
    artifact_path.write_bytes(pickle.dumps(_ExecutablePayload(marker_path)))

    with pytest.raises(pickle.UnpicklingError, match="forbidden"):
        load_public_artifact(artifact_path)

    assert not marker_path.exists()


def test_restricted_public_artifact_loader_rejects_trailing_bytes(tmp_path) -> None:
    artifact_path = tmp_path / "trailing.pkl"
    artifact_path.write_bytes(pickle.dumps({"value": 1}) + b"unbound payload")

    with pytest.raises(pickle.UnpicklingError, match="trailing bytes"):
        load_public_artifact(artifact_path)
