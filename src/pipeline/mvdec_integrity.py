"""Deterministic NumPy hashing for MvDEC frozen-model proofs."""

import hashlib
import json
import re
from collections.abc import Sequence

import numpy as np


def array_sha256(values: np.ndarray) -> str:
    """Hash dtype, shape, C-order identity, and bytes for one array."""

    array = np.asarray(values)
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    header = {
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
        "order": "C",
    }
    digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def array_sequence_sha256(values: Sequence[np.ndarray]) -> str:
    """Hash an ordered collection of arrays without concatenating them."""

    digests = [array_sha256(value) for value in values]
    payload = json.dumps(digests, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


FROZEN_VIEW1_PROOF_FIELDS = frozenset(
    {
        "schema_version",
        "weight_count",
        "weights_before_sha256",
        "weights_after_sha256",
        "weights_bitwise_equal",
        "output_before_sha256",
        "output_after_sha256",
        "output_bitwise_equal",
        "latent_before_sha256",
        "latent_after_sha256",
        "latent_bitwise_equal",
        "h_view1_sha256",
    }
)


def validate_frozen_view1_proof(
    proof: object,
    h_view1: np.ndarray,
) -> dict[str, object]:
    """Validate the exact persisted proof that imported View1 stayed frozen."""

    if not isinstance(proof, dict) or set(proof) != FROZEN_VIEW1_PROOF_FIELDS:
        raise ValueError("Frozen View1 integrity proof fields are invalid.")
    if proof["schema_version"] != 1:
        raise ValueError("Frozen View1 integrity proof schema is unsupported.")
    weight_count = proof["weight_count"]
    if isinstance(weight_count, bool) or not isinstance(weight_count, int):
        raise ValueError("Frozen View1 integrity proof weight_count is invalid.")
    if weight_count < 1:
        raise ValueError("Frozen View1 integrity proof requires model weights.")

    equality_fields = (
        "weights_bitwise_equal",
        "output_bitwise_equal",
        "latent_bitwise_equal",
    )
    if any(proof[field] is not True for field in equality_fields):
        raise ValueError("Frozen View1 integrity proof must be bitwise exact.")

    hash_fields = FROZEN_VIEW1_PROOF_FIELDS - {
        "schema_version",
        "weight_count",
        *equality_fields,
    }
    if any(
        not isinstance(proof[field], str)
        or re.fullmatch(r"[0-9a-f]{64}", proof[field]) is None
        for field in hash_fields
    ):
        raise ValueError("Frozen View1 integrity proof contains an invalid SHA-256.")
    if any(
        (
            proof["weights_before_sha256"] != proof["weights_after_sha256"],
            proof["output_before_sha256"] != proof["output_after_sha256"],
            proof["latent_before_sha256"] != proof["latent_after_sha256"],
            proof["latent_after_sha256"] != proof["h_view1_sha256"],
            proof["h_view1_sha256"] != array_sha256(h_view1),
        )
    ):
        raise ValueError("Frozen View1 integrity proof hashes are inconsistent.")
    return dict(proof)


VIEW1_REFINEMENT_TRANSITION_FIELDS = frozenset(
    {
        "schema_version",
        "weight_count",
        "initial_weights_sha256",
        "final_weights_sha256",
        "initial_latent_sha256",
        "final_latent_sha256",
        "initial_view1_acc",
        "initial_view1_nmi",
        "final_view1_acc",
        "final_view1_nmi",
        "view1_nonzero_gradient_observed",
        "view1_max_abs_gradient",
        "weights_changed",
        "latents_changed",
        "h_view1_sha256",
    }
)


def validate_view1_refinement_transition(
    proof: object,
    h_view1: np.ndarray,
) -> dict[str, object]:
    """Validate evidence that imported View1 participated in joint refinement."""

    if not isinstance(proof, dict) or set(proof) != VIEW1_REFINEMENT_TRANSITION_FIELDS:
        raise ValueError("View1 refinement transition fields are invalid.")
    if proof["schema_version"] != 1:
        raise ValueError("View1 refinement transition schema is unsupported.")
    weight_count = proof["weight_count"]
    if isinstance(weight_count, bool) or not isinstance(weight_count, int):
        raise ValueError("View1 refinement transition weight_count is invalid.")
    if weight_count < 1:
        raise ValueError("View1 refinement transition requires model weights.")

    hash_fields = {
        "initial_weights_sha256",
        "final_weights_sha256",
        "initial_latent_sha256",
        "final_latent_sha256",
        "h_view1_sha256",
    }
    if any(
        not isinstance(proof[field], str)
        or re.fullmatch(r"[0-9a-f]{64}", proof[field]) is None
        for field in hash_fields
    ):
        raise ValueError("View1 refinement transition contains an invalid SHA-256.")

    metric_fields = (
        "initial_view1_acc",
        "initial_view1_nmi",
        "final_view1_acc",
        "final_view1_nmi",
    )
    for field in metric_fields:
        value = proof[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("View1 refinement transition metrics are invalid.")
        if not np.isfinite(value) or not 0.0 <= float(value) <= 1.0:
            raise ValueError("View1 refinement transition metrics are invalid.")

    max_gradient = proof["view1_max_abs_gradient"]
    if isinstance(max_gradient, bool) or not isinstance(max_gradient, (int, float)):
        raise ValueError(
            "View1 refinement transition gradient evidence is invalid."
        )
    if not np.isfinite(max_gradient) or float(max_gradient) <= 0.0:
        raise ValueError(
            "View1 refinement transition requires a finite nonzero gradient."
        )
    if proof["view1_nonzero_gradient_observed"] is not True:
        raise ValueError("View1 refinement transition requires a nonzero gradient.")
    if proof["weights_changed"] is not True or proof["latents_changed"] is not True:
        raise ValueError(
            "View1 refinement transition requires changed weights and latents."
        )
    if proof["initial_weights_sha256"] == proof["final_weights_sha256"]:
        raise ValueError("View1 refinement transition weights did not change.")
    if proof["initial_latent_sha256"] == proof["final_latent_sha256"]:
        raise ValueError("View1 refinement transition latents did not change.")
    expected_latent_hash = array_sha256(h_view1)
    if any(
        (
            proof["final_latent_sha256"] != proof["h_view1_sha256"],
            proof["h_view1_sha256"] != expected_latent_hash,
        )
    ):
        raise ValueError(
            "View1 refinement transition final latent hash is inconsistent."
        )
    return dict(proof)
