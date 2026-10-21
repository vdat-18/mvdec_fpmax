"""Tests for one-shot centered-RMS MvDEC latent calibration."""

from dataclasses import replace

import numpy as np
import pytest

from pipeline.mvdec_calibration import (
    CENTERED_RMS_CALIBRATION_CLAIM_SCOPE,
    CENTERED_RMS_CALIBRATION_METHOD,
    CENTERED_RMS_CALIBRATION_SOURCE_PHASE,
    CenteredRmsCalibration,
    calibrated_average,
    center_latent,
    global_rms,
)


def _calibration() -> CenteredRmsCalibration:
    view1 = np.array([[1.0, 10.0], [3.0, 14.0], [5.0, 18.0]])
    view2 = np.array([[100.0, 2.0], [101.0, 4.0], [102.0, 6.0]])
    return CenteredRmsCalibration.fit(view1, view2)


def test_centered_rms_calibration_matches_preregistered_formula() -> None:
    view1 = np.array([[1.0, 10.0], [3.0, 14.0], [5.0, 18.0]])
    view2 = np.array([[100.0, 2.0], [101.0, 4.0], [102.0, 6.0]])

    calibration = CenteredRmsCalibration.fit(view1, view2)
    expected_rms1 = np.sqrt(np.mean((view1 - view1.mean(axis=0)) ** 2))
    expected_rms2 = np.sqrt(np.mean((view2 - view2.mean(axis=0)) ** 2))

    np.testing.assert_allclose(center_latent(view1).mean(axis=0), 0.0, atol=1e-15)
    assert global_rms(center_latent(view1)) == pytest.approx(expected_rms1)
    assert calibration.view1_centered_global_rms == pytest.approx(expected_rms1)
    assert calibration.view2_centered_global_rms == pytest.approx(expected_rms2)
    assert calibration.view2_scale_factor == pytest.approx(
        expected_rms1 / expected_rms2
    )
    assert calibration.n_samples == 3
    assert calibration.latent_width == 2
    assert calibration.method == CENTERED_RMS_CALIBRATION_METHOD
    assert calibration.source_phase == CENTERED_RMS_CALIBRATION_SOURCE_PHASE
    assert calibration.claim_scope == CENTERED_RMS_CALIBRATION_CLAIM_SCOPE


def test_calibrated_average_fuses_raw_not_centered_latents() -> None:
    view1 = np.array([[11.0, 20.0], [13.0, 24.0], [15.0, 28.0]])
    view2 = np.array([[101.0, 2.0], [102.0, 4.0], [103.0, 6.0]])
    calibration = CenteredRmsCalibration.fit(view1, view2)

    fused = calibrated_average(view1, view2, calibration)
    raw_expected = (view1 + calibration.view2_scale_factor * view2) / 2
    centered_wrong = (
        center_latent(view1) + calibration.view2_scale_factor * center_latent(view2)
    ) / 2

    np.testing.assert_array_equal(fused, raw_expected)
    assert not np.array_equal(fused, centered_wrong)


@pytest.mark.parametrize(
    ("view1", "view2"),
    [
        (np.ones((2, 2)), np.ones((2, 3))),
        (np.ones((2, 2, 1)), np.ones((2, 2, 1))),
        (np.empty((0, 2)), np.empty((0, 2))),
        (np.empty((2, 0)), np.empty((2, 0))),
        (np.array([[np.nan, 1.0]]), np.ones((1, 2))),
        (np.ones((1, 2)), np.array([[np.inf, 1.0]])),
    ],
)
def test_calibration_rejects_invalid_latent_pairs(view1, view2) -> None:
    with pytest.raises(ValueError):
        CenteredRmsCalibration.fit(view1, view2)


@pytest.mark.parametrize(
    ("view1", "view2"),
    [
        (np.ones((3, 2)), np.arange(6.0).reshape(3, 2)),
        (np.arange(6.0).reshape(3, 2), np.ones((3, 2))),
    ],
)
def test_calibration_rejects_zero_centered_variance(view1, view2) -> None:
    with pytest.raises(ValueError, match="non-constant"):
        CenteredRmsCalibration.fit(view1, view2)


def test_calibration_metadata_round_trip_is_exact() -> None:
    calibration = _calibration()

    restored = CenteredRmsCalibration.from_metadata(calibration.to_metadata())

    assert restored == calibration


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("method", "wrong"),
        ("view1_centered_global_rms", np.nan),
        ("view2_centered_global_rms", 0.0),
        ("view2_scale_factor", np.inf),
        ("n_samples", 0),
        ("latent_width", 1.5),
        ("label_free", False),
        ("source_phase", "during_refinement"),
        ("calibration_count", 2),
        ("recomputed_during_refinement", True),
        ("claim_scope", "exact_paper_truth"),
    ],
)
def test_calibration_metadata_rejects_tampering(field, value) -> None:
    metadata = _calibration().to_metadata()
    metadata[field] = value

    with pytest.raises(ValueError):
        CenteredRmsCalibration.from_metadata(metadata)


def test_calibration_metadata_rejects_inconsistent_ratio() -> None:
    metadata = _calibration().to_metadata()
    metadata["view2_scale_factor"] *= 1.01

    with pytest.raises(ValueError, match="RMS ratio"):
        CenteredRmsCalibration.from_metadata(metadata)


def test_calibration_metadata_rejects_missing_or_extra_fields() -> None:
    metadata = _calibration().to_metadata()
    missing = dict(metadata)
    missing.pop("method")
    extra = {**metadata, "unexpected": True}

    with pytest.raises(ValueError, match="fields"):
        CenteredRmsCalibration.from_metadata(missing)
    with pytest.raises(ValueError, match="fields"):
        CenteredRmsCalibration.from_metadata(extra)


def test_calibrated_average_rejects_shape_different_from_fitted_metadata() -> None:
    calibration = replace(_calibration(), n_samples=4)

    with pytest.raises(ValueError, match="shape"):
        calibrated_average(np.ones((3, 2)), np.ones((3, 2)), calibration)
