"""Restricted deserialization for NumPy-only artifact containers."""

import pickle
from collections.abc import Callable
from typing import BinaryIO

import numpy as np


def _safe_legacy_encode(
    value: str,
    encoding: str = "latin1",
    errors: str = "strict",
) -> bytes:
    """Support NumPy's Python-2 pickle compatibility without codec lookup."""

    normalized_encoding = encoding.lower().replace("_", "-")
    if normalized_encoding not in {"latin1", "latin-1", "iso-8859-1"}:
        msg = f"Legacy pickle codec {encoding!r} is forbidden."
        raise pickle.UnpicklingError(msg)
    if errors != "strict" or not isinstance(value, str):
        msg = "Legacy pickle encoding requires a string and strict errors."
        raise pickle.UnpicklingError(msg)
    return value.encode("latin1")


_ALLOWED_PICKLE_GLOBALS: dict[tuple[str, str], Callable | type] = {
    ("_codecs", "encode"): _safe_legacy_encode,
    ("numpy", "dtype"): np.dtype,
    ("numpy", "ndarray"): np.ndarray,
    ("numpy._core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
    ("numpy._core.multiarray", "scalar"): np._core.multiarray.scalar,
    # NumPy 1.x used this module path. NumPy 2 keeps it load-compatible.
    ("numpy.core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
    ("numpy.core.multiarray", "scalar"): np._core.multiarray.scalar,
}


class _RestrictedNumPyUnpickler(pickle.Unpickler):
    """Deserialize containers without importing arbitrary globals."""

    def find_class(self, module: str, name: str) -> object:
        allowed = _ALLOWED_PICKLE_GLOBALS.get((module, name))
        if allowed is None:
            msg = f"Pickle global {module}.{name} is forbidden in NumPy artifacts."
            raise pickle.UnpicklingError(msg)
        return allowed


def load_restricted_pickle(file: BinaryIO) -> object:
    """Load exactly one NumPy-only pickle payload with no trailing bytes."""

    payload = _RestrictedNumPyUnpickler(file).load()
    if file.read(1):
        msg = "NumPy artifact contains trailing bytes after its pickle payload."
        raise pickle.UnpicklingError(msg)
    return payload
