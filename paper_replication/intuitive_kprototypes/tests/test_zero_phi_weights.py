"""Regression tests for the paper's zero-phi weight branch."""

import numpy as np

from intuitive_kprototypes.model import _inverse_phi_weights


def test_zero_phi_is_excluded_from_weight_normalization() -> None:
    """Eq. (16)/(22) assigns zero weight when phi is exactly zero."""

    weights = _inverse_phi_weights(np.array([0.0, 0.5, 1.0]), beta=2.0)

    np.testing.assert_allclose(weights, [0.0, 2.0 / 3.0, 1.0 / 3.0])
