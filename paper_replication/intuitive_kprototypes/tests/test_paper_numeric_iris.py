"""Regression tests against the paper's Iris numerical examples."""

import numpy as np
from intuitive_kprototypes.model import (
    numeric_complexity_details,
    numeric_similarity_details,
)
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler


def normalized_iris():
    X, labels = load_iris(return_X_y=True)
    return MinMaxScaler().fit_transform(X), labels


def test_numeric_complexity_details_match_tables_3_and_4():
    X, labels = normalized_iris()

    details = numeric_complexity_details(X, labels, n_clusters=3)

    expected_cv_table_3 = np.asarray(
        [
            [0.494, 0.266, 0.370, 0.737],
            [0.312, 0.403, 0.143, 0.160],
            [0.275, 0.328, 0.120, 0.141],
        ]
    )
    expected_norm_table_4 = np.asarray(
        [
            [1.000, 0.660, 1.000, 1.000],
            [0.632, 1.000, 0.386, 0.217],
            [0.557, 0.814, 0.324, 0.191],
        ]
    )

    # Eq. (5) uses population variance, and this reproduces nearly all of
    # Table 3 with sklearn's Iris plus min-max normalization. The Setosa petal
    # width cell is slightly different in the paper table, likely from a
    # dataset/normalization detail not specified by the paper.
    np.testing.assert_allclose(
        details.coefficients,
        expected_cv_table_3,
        atol=0.025,
    )
    np.testing.assert_allclose(
        details.normalized_coefficients,
        expected_norm_table_4,
        atol=0.035,
    )


def test_numeric_similarity_details_match_table_5():
    X, labels = normalized_iris()

    details = numeric_similarity_details(X, labels, n_clusters=3)

    assert details.pair_indices == ((0, 1), (0, 2), (1, 2))
    expected_table_5 = np.asarray(
        [
            [0.125, 0.154, 0.011, 0.011],
            [0.056, 0.253, 0.005, 0.011],
            [0.368, 0.690, 0.087, 0.080],
        ]
    )
    np.testing.assert_allclose(details.pair_similarity, expected_table_5, atol=0.01)

    # Cluster sizes are equal in Iris, so Eq. (14) is the mean of Table 5 rows.
    np.testing.assert_allclose(
        details.similarity,
        expected_table_5.mean(axis=0),
        atol=0.01,
    )
