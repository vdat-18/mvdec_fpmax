"""Tests for the authorized DEKM_F View1 checkpoint boundary."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from pipeline import dekm_checkpoint
from pipeline.dekm_checkpoint import (
    CHECKPOINT_PROVENANCE_SHA_FIELDS,
    DEKM_F_CONFIG_HASH,
    DEKM_F_MANIFEST_SHA256,
    DEKM_F_METHOD,
    DEKM_F_PROTOCOL_ID,
    DEKM_F_REUTERS_ACC,
    DEKM_F_REUTERS_NMI,
    DEKM_F_REUTERS_SOURCE_SHA256,
    HDF5_SIGNATURE,
    copy_validated_checkpoint,
    validate_dekm_f_checkpoint_provenance,
    validate_dekm_f_view1_checkpoint,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_authorized_manifest_sha_is_pinned() -> None:
    assert DEKM_F_MANIFEST_SHA256 == (
        "71e3c2d423d70bbde1b7423c91c176c1d33a4fb473e7cb6febec187fff8f0607"
    )


@pytest.fixture
def checkpoint_fixture(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    weights = run_dir / "final.weights.h5"
    weights_bytes = HDF5_SIGNATURE + b"authorized-test-checkpoint"
    weights.write_bytes(weights_bytes)
    weights_sha256 = _sha256(weights_bytes)
    monkeypatch.setattr(dekm_checkpoint, "DEKM_F_WEIGHT_SHA256", weights_sha256)

    output_names = {
        "artifact": "artifact.pkl",
        "assignments": "assignments.csv",
        "final_weights": weights.name,
        "pretrain_weights": "pretrain.weights.h5",
        "training_log": "training_log.csv",
    }
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "method": DEKM_F_METHOD,
        "protocol_id": DEKM_F_PROTOCOL_ID,
        "config_hash": DEKM_F_CONFIG_HASH,
        "dataset": "REUTERS",
        "seed": 42,
        "source_sha256": DEKM_F_REUTERS_SOURCE_SHA256,
        "n_samples": 10_000,
        "n_features": 2_000,
        "n_clusters": 4,
        "metrics": {"acc": DEKM_F_REUTERS_ACC, "nmi": DEKM_F_REUTERS_NMI},
        "config": deepcopy(dekm_checkpoint._EXPECTED_CONFIG),
        "paths": output_names,
        "output_sha256": {
            **{name: "0" * 64 for name in output_names},
            "final_weights": weights_sha256,
        },
        "fit_time_seconds": 1.0,
        "official_dense_sha256": "1" * 64,
        "official_repository": "official/repository",
        "official_source_revision": "revision",
        "refreshes_completed": 1,
        "runtime": {},
        "steps_completed": 1,
        "stop_reason": "converged_assignment",
    }
    manifest_path = run_dir / "manifest.json"

    def write(payload=manifest, *, authorize=True):
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        if authorize:
            monkeypatch.setattr(
                dekm_checkpoint,
                "DEKM_F_MANIFEST_SHA256",
                _sha256(manifest_path.read_bytes()),
            )
        return manifest_path

    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "weights": weights,
        "weights_bytes": weights_bytes,
        "weights_sha256": weights_sha256,
        "write": write,
    }


def _validate(path: Path):
    return validate_dekm_f_view1_checkpoint(
        path,
        expected_dataset="REUTERS",
        expected_seed=42,
        expected_source_sha256=DEKM_F_REUTERS_SOURCE_SHA256,
    )


def test_valid_manifest_and_checkpoint(checkpoint_fixture) -> None:
    checkpoint = _validate(checkpoint_fixture["write"]())

    assert checkpoint.weights_path == checkpoint_fixture["weights"].resolve()
    assert checkpoint.weights_sha256 == checkpoint_fixture["weights_sha256"]
    assert checkpoint.provenance()["claim_scope"] == (
        "controlled_diagnostic_not_exact_paper_truth"
    )
    assert not any("path" in key for key in checkpoint.provenance())


def test_manifest_content_is_pinned_after_authorization(checkpoint_fixture) -> None:
    manifest_path = checkpoint_fixture["write"]()
    mutated = deepcopy(checkpoint_fixture["manifest"])
    mutated["fit_time_seconds"] = 2.0
    checkpoint_fixture["write"](mutated, authorize=False)

    with pytest.raises(ValueError, match="authorized manifest"):
        _validate(manifest_path)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("status", "running"),
        ("method", "DEKM"),
        ("protocol_id", "wrong"),
        ("dataset", "RCV1"),
        ("seed", 43),
        ("source_sha256", "f" * 64),
    ],
)
def test_manifest_identity_mismatches_fail_closed(
    checkpoint_fixture,
    field,
    invalid,
) -> None:
    manifest = deepcopy(checkpoint_fixture["manifest"])
    manifest[field] = invalid

    with pytest.raises(ValueError, match=field):
        _validate(checkpoint_fixture["write"](manifest))


def test_manifest_schema_mismatch_fails_closed(checkpoint_fixture) -> None:
    manifest = deepcopy(checkpoint_fixture["manifest"])
    manifest["unexpected"] = True

    with pytest.raises(ValueError, match="schema"):
        _validate(checkpoint_fixture["write"](manifest))


@pytest.mark.parametrize("field", ["paths", "output_sha256"])
def test_nested_output_schema_mismatch_fails_closed(
    checkpoint_fixture,
    field,
) -> None:
    manifest = deepcopy(checkpoint_fixture["manifest"])
    manifest[field].pop("artifact")

    with pytest.raises(ValueError, match=field):
        _validate(checkpoint_fixture["write"](manifest))


def test_config_mismatch_fails_closed(checkpoint_fixture) -> None:
    manifest = deepcopy(checkpoint_fixture["manifest"])
    manifest["config"]["update_interval"] = 9

    with pytest.raises(ValueError, match="config"):
        _validate(checkpoint_fixture["write"](manifest))


def test_config_hash_mismatch_fails_closed(checkpoint_fixture) -> None:
    manifest = deepcopy(checkpoint_fixture["manifest"])
    manifest["config_hash"] = "f" * 64

    with pytest.raises(ValueError, match="config hash"):
        _validate(checkpoint_fixture["write"](manifest))


@pytest.mark.parametrize("metric", ["acc", "nmi"])
def test_metric_mismatch_fails_closed(checkpoint_fixture, metric) -> None:
    manifest = deepcopy(checkpoint_fixture["manifest"])
    manifest["metrics"][metric] = -1.0

    with pytest.raises(ValueError, match="metrics"):
        _validate(checkpoint_fixture["write"](manifest))


@pytest.mark.parametrize(
    "relative_path",
    [
        str(Path("C:/outside/final.weights.h5")),
        "../outside.weights.h5",
    ],
)
def test_unsafe_weight_paths_fail_closed(checkpoint_fixture, relative_path) -> None:
    manifest = deepcopy(checkpoint_fixture["manifest"])
    manifest["paths"]["final_weights"] = relative_path

    with pytest.raises(ValueError, match="safe manifest-relative"):
        _validate(checkpoint_fixture["write"](manifest))


def test_symlink_escape_fails_closed(checkpoint_fixture, tmp_path) -> None:
    outside = tmp_path / "outside.weights.h5"
    outside.write_bytes(checkpoint_fixture["weights_bytes"])
    link = checkpoint_fixture["manifest_path"].parent / "linked.weights.h5"
    try:
        link.symlink_to(outside)
    except (NotImplementedError, OSError):
        pytest.skip("Symlinks are unavailable on this platform.")
    manifest = deepcopy(checkpoint_fixture["manifest"])
    manifest["paths"]["final_weights"] = link.name

    with pytest.raises(ValueError, match="escapes|symlink"):
        _validate(checkpoint_fixture["write"](manifest))


def test_unauthorized_weight_hash_fails_closed(checkpoint_fixture) -> None:
    manifest = deepcopy(checkpoint_fixture["manifest"])
    manifest["output_sha256"]["final_weights"] = "f" * 64

    with pytest.raises(ValueError, match="authorized"):
        _validate(checkpoint_fixture["write"](manifest))


def test_invalid_hdf5_signature_fails_closed(checkpoint_fixture, monkeypatch) -> None:
    invalid = b"not-an-hdf5-container"
    checkpoint_fixture["weights"].write_bytes(invalid)
    invalid_hash = _sha256(invalid)
    monkeypatch.setattr(dekm_checkpoint, "DEKM_F_WEIGHT_SHA256", invalid_hash)
    manifest = deepcopy(checkpoint_fixture["manifest"])
    manifest["output_sha256"]["final_weights"] = invalid_hash

    with pytest.raises(ValueError, match="HDF5"):
        _validate(checkpoint_fixture["write"](manifest))


def test_provenance_is_path_free_and_all_sha_fields_are_validated(
    checkpoint_fixture,
) -> None:
    checkpoint = _validate(checkpoint_fixture["write"]())
    provenance = checkpoint.provenance()

    validated = validate_dekm_f_checkpoint_provenance(
        provenance,
        expected_source_sha256=DEKM_F_REUTERS_SOURCE_SHA256,
    )
    assert validated == provenance
    assert set(validated) == {
        "method",
        "protocol_id",
        "config_hash",
        "dataset",
        "seed",
        "source_sha256",
        "manifest_sha256",
        "weight_sha256",
        "input_normalization",
        "refinement_batching",
        "claim_scope",
    }

    for field in CHECKPOINT_PROVENANCE_SHA_FIELDS:
        invalid = dict(provenance)
        invalid[field] = "not-a-sha"
        with pytest.raises(ValueError, match="SHA-256|identity"):
            validate_dekm_f_checkpoint_provenance(
                invalid,
                expected_source_sha256=DEKM_F_REUTERS_SOURCE_SHA256,
            )


def test_provenance_rejects_paths_and_wrong_expected_source(
    checkpoint_fixture,
) -> None:
    provenance = _validate(checkpoint_fixture["write"]()).provenance()
    provenance["weights_path"] = "C:/private/checkpoint.h5"

    with pytest.raises(ValueError, match="schema"):
        validate_dekm_f_checkpoint_provenance(
            provenance,
            expected_source_sha256=DEKM_F_REUTERS_SOURCE_SHA256,
        )
    with pytest.raises(ValueError, match="source"):
        validate_dekm_f_checkpoint_provenance(
            _validate(checkpoint_fixture["manifest_path"]).provenance(),
            expected_source_sha256="f" * 64,
        )


def test_copy_preserves_hash_and_hdf5_signature(checkpoint_fixture, tmp_path) -> None:
    checkpoint = _validate(checkpoint_fixture["write"]())
    destination = tmp_path / "run" / "pretrain_view1.weights.h5"

    copy_validated_checkpoint(checkpoint, destination)

    assert destination.read_bytes() == checkpoint_fixture["weights_bytes"]
    assert _sha256(destination.read_bytes()) == checkpoint.weights_sha256
    assert destination.read_bytes().startswith(HDF5_SIGNATURE)


def test_copy_rejects_symlink_destination(checkpoint_fixture, tmp_path) -> None:
    checkpoint = _validate(checkpoint_fixture["write"]())
    target = tmp_path / "target.weights.h5"
    target.write_bytes(b"untouched")
    destination = tmp_path / "pretrain_view1.weights.h5"
    try:
        destination.symlink_to(target)
    except (NotImplementedError, OSError):
        pytest.skip("Symlinks are unavailable on this platform.")

    with pytest.raises(ValueError, match="destination.*symlink"):
        copy_validated_checkpoint(checkpoint, destination)
    assert target.read_bytes() == b"untouched"
