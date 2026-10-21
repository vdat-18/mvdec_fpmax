"""Label-free fixed calibration for MvDEC latent fusion."""

from dataclasses import asdict, dataclass
from numbers import Integral, Real

import numpy as np

CENTERED_RMS_CALIBRATION_METHOD = "centered_global_rms_ratio_v1"
CENTERED_RMS_CALIBRATION_SOURCE_PHASE = (
    "after_independent_pretraining_before_refinement"
)
CENTERED_RMS_CALIBRATION_CLAIM_SCOPE = "controlled_diagnostic_not_exact_paper_truth"


def _validated_pair(
    view1: np.ndarray,
    view2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    first = np.asarray(view1)
    second = np.asarray(view2)
    if first.ndim != 2 or second.ndim != 2:
        raise ValueError("Centered-RMS calibration requires two-dimensional latents.")
    if first.shape != second.shape:
        raise ValueError("Centered-RMS calibration requires equal latent shapes.")
    if first.shape[0] < 1 or first.shape[1] < 1:
        raise ValueError("Centered-RMS calibration requires non-empty latents.")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Centered-RMS calibration requires finite latents.")
    return first, second


def center_latent(values: np.ndarray) -> np.ndarray:
    """Center each latent coordinate over samples."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError("Latent values must be a non-empty two-dimensional array.")
    if not np.isfinite(array).all():
        raise ValueError("Latent values must be finite.")
    return array - array.mean(axis=0)


def global_rms(values: np.ndarray) -> float:
    """Return RMS over every cell in a finite non-empty array."""

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Global RMS requires finite non-empty values.")
    return float(np.sqrt(np.mean(np.square(array))))


@dataclass(frozen=True)
class CenteredRmsCalibration:
    """Immutable scale fitted once before calibrated fused refinement."""

    method: str
    view1_centered_global_rms: float
    view2_centered_global_rms: float
    view2_scale_factor: float
    n_samples: int
    latent_width: int
    label_free: bool = True
    source_phase: str = CENTERED_RMS_CALIBRATION_SOURCE_PHASE
    calibration_count: int = 1
    recomputed_during_refinement: bool = False
    claim_scope: str = CENTERED_RMS_CALIBRATION_CLAIM_SCOPE

    @classmethod
    def fit(
        cls,
        view1: np.ndarray,
        view2: np.ndarray,
    ) -> "CenteredRmsCalibration":
        """Fit one global scale from centered full-dataset latents."""

        first, second = _validated_pair(view1, view2)
        first_rms = global_rms(center_latent(first))
        second_rms = global_rms(center_latent(second))
        if first_rms <= 0.0 or second_rms <= 0.0:
            raise ValueError(
                "Centered-RMS calibration requires non-constant latent spaces."
            )
        return cls(
            method=CENTERED_RMS_CALIBRATION_METHOD,
            view1_centered_global_rms=first_rms,
            view2_centered_global_rms=second_rms,
            view2_scale_factor=first_rms / second_rms,
            n_samples=int(first.shape[0]),
            latent_width=int(first.shape[1]),
        )

    @classmethod
    def from_metadata(cls, metadata: object) -> "CenteredRmsCalibration":
        """Parse and fail closed on persisted calibration metadata."""

        if not isinstance(metadata, dict):
            raise ValueError("MvDEC calibrated fusion metadata must be a dictionary.")
        expected_fields = set(cls.__dataclass_fields__)
        if set(metadata) != expected_fields:
            raise ValueError("MvDEC calibrated fusion metadata fields are invalid.")
        try:
            calibration = cls(**metadata)
        except (TypeError, ValueError) as error:
            raise ValueError("MvDEC calibrated fusion metadata is invalid.") from error
        calibration.validate()
        return calibration

    def validate(self) -> None:
        """Validate immutable identity, dimensions, and exact scale relation."""

        numeric = (
            self.view1_centered_global_rms,
            self.view2_centered_global_rms,
            self.view2_scale_factor,
        )
        if self.method != CENTERED_RMS_CALIBRATION_METHOD:
            raise ValueError("Unexpected centered-RMS calibration method.")
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not np.isfinite(value)
            for value in numeric
        ):
            raise ValueError("Centered-RMS calibration values must be finite numbers.")
        if any(value <= 0.0 for value in numeric):
            raise ValueError("Centered-RMS calibration values must be positive.")
        dimensions = (self.n_samples, self.latent_width)
        if any(
            isinstance(value, bool) or not isinstance(value, Integral) or value < 1
            for value in dimensions
        ):
            raise ValueError(
                "Centered-RMS calibration dimensions must be positive integers."
            )
        if self.label_free is not True:
            raise ValueError("Centered-RMS calibration must be label-free.")
        if self.source_phase != CENTERED_RMS_CALIBRATION_SOURCE_PHASE:
            raise ValueError("Centered-RMS calibration source phase is invalid.")
        if isinstance(self.calibration_count, bool) or self.calibration_count != 1:
            raise ValueError("Centered-RMS calibration must be computed exactly once.")
        if self.recomputed_during_refinement is not False:
            raise ValueError(
                "Centered-RMS calibration must remain fixed in refinement."
            )
        if self.claim_scope != CENTERED_RMS_CALIBRATION_CLAIM_SCOPE:
            raise ValueError("Centered-RMS calibration claim scope is invalid.")
        expected_scale = self.view1_centered_global_rms / self.view2_centered_global_rms
        if not np.isclose(
            self.view2_scale_factor,
            expected_scale,
            rtol=1e-12,
            atol=0.0,
        ):
            raise ValueError(
                "Centered-RMS scale does not match the persisted RMS ratio."
            )

    def to_metadata(self) -> dict[str, object]:
        """Return JSON-safe immutable runtime metadata."""

        self.validate()
        return asdict(self)


def calibrated_average(
    view1: np.ndarray,
    view2: np.ndarray,
    calibration: CenteredRmsCalibration,
) -> np.ndarray:
    """Fuse raw, uncentered latents using one fixed View2 scale."""

    first, second = _validated_pair(view1, view2)
    calibration.validate()
    if first.shape != (calibration.n_samples, calibration.latent_width):
        raise ValueError(
            "Latent shape does not match centered-RMS calibration metadata."
        )
    return (first + calibration.view2_scale_factor * second) / 2
