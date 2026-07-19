"""External clustering metrics shared by labeled benchmark methods."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score


@dataclass(frozen=True)
class ExternalMetrics:
    """ACC and NMI for one clustering assignment."""

    acc: float
    nmi: float


def _validated_labels(values: object, name: str) -> np.ndarray:
    """Return one non-empty finite one-dimensional label vector."""

    labels = np.asarray(values)
    if labels.ndim != 1 or len(labels) == 0:
        msg = f"{name} must be a non-empty one-dimensional array."
        raise ValueError(msg)
    if pd.isna(labels).any():
        msg = f"{name} contains missing values."
        raise ValueError(msg)
    return labels


def compute_external_metrics(
    true_labels: object,
    predicted_labels: object,
    *,
    n_clusters: int | None = None,
) -> ExternalMetrics:
    """Compute Hungarian-matched clustering ACC and NMI."""

    truth = _validated_labels(true_labels, "true_labels")
    predicted = _validated_labels(predicted_labels, "predicted_labels")
    if len(truth) != len(predicted):
        msg = (
            "true_labels and predicted_labels must contain the same number "
            f"of rows: {len(truth)} != {len(predicted)}."
        )
        raise ValueError(msg)

    true_values, true_inverse = np.unique(truth, return_inverse=True)
    predicted_values, predicted_inverse = np.unique(predicted, return_inverse=True)
    if n_clusters is not None:
        if n_clusters < 2:
            msg = "n_clusters must be at least 2."
            raise ValueError(msg)
        if len(true_values) != n_clusters or len(predicted_values) != n_clusters:
            msg = (
                "True and predicted labels must both use exactly n_clusters "
                f"classes: true={len(true_values)}, predicted={len(predicted_values)}, "
                f"expected={n_clusters}."
            )
            raise ValueError(msg)

    contingency = np.zeros(
        (len(predicted_values), len(true_values)),
        dtype=np.int64,
    )
    np.add.at(contingency, (predicted_inverse, true_inverse), 1)
    rows, columns = linear_sum_assignment(-contingency)
    accuracy = float(contingency[rows, columns].sum() / len(truth))
    nmi = float(normalized_mutual_info_score(truth, predicted))
    return ExternalMetrics(acc=accuracy, nmi=nmi)
