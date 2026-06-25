"""Regression tests against the paper's artificial categorical example."""

import numpy as np
from intuitive_kprototypes import (
    categorical_attribute_distances,
    categorical_complexity,
    categorical_similarity,
    categorical_weights,
    compute_intuitionistic_centroids,
)


def paper_table_1():
    cluster_1 = [
        ("a", "d", "g"),
        ("a", "d", "g"),
        ("a", "d", "h"),
        ("a", "e", "h"),
        ("a", "e", "h"),
        ("b", "e", "h"),
        ("b", "e", "g"),
        ("b", "e", "h"),
        ("b", "e", "h"),
        ("b", "e", "g"),
    ]
    cluster_2 = [
        ("a", "d", "g"),
        ("a", "d", "g"),
        ("a", "e", "g"),
        ("a", "f", "g"),
        ("a", "f", "g"),
        ("c", "f", "h"),
        ("c", "f", "h"),
        ("c", "f", "h"),
        ("c", "f", "h"),
        ("c", "f", "g"),
    ]
    X_cat = np.asarray(cluster_1 + cluster_2, dtype=object)
    labels = np.asarray([0] * 10 + [1] * 10, dtype=int)
    return X_cat, labels


def test_idc_matches_table_2_key_values():
    X_cat, labels = paper_table_1()
    idc = compute_intuitionistic_centroids(
        X_cat,
        labels,
        n_clusters=2,
        non_membership="table",
    )

    attr_1 = idc.value_to_index[0]
    a = attr_1["a"]
    b = attr_1["b"]
    c = attr_1["c"]

    assert idc.mu[0][0, a] == 1 / 4
    assert idc.nu[0][0, a] == 1 / 4
    assert idc.mu[0][0, b] == 1 / 2
    assert idc.nu[0][0, b] == 0
    assert idc.mu[0][0, c] == 0
    assert idc.nu[0][0, c] == 1

    assert idc.mu[0][1, a] == 1 / 4
    assert idc.nu[0][1, a] == 1 / 4
    assert idc.mu[0][1, b] == 0
    assert idc.nu[0][1, b] == 1
    assert idc.mu[0][1, c] == 1 / 2
    assert idc.nu[0][1, c] == 0

    for j in range(3):
        assert np.all(idc.mu[j] >= 0)
        assert np.all(idc.nu[j] >= 0)
        assert np.all(idc.mu[j] + idc.nu[j] <= 1 + 1e-12)


def test_categorical_complexity_matches_table_6():
    X_cat, labels = paper_table_1()
    idc = compute_intuitionistic_centroids(
        X_cat,
        labels,
        n_clusters=2,
        non_membership="table",
    )

    complexity = categorical_complexity(idc, labels, n_clusters=2)

    expected_attr_1 = 1 / 7
    expected_attr_2 = 0.5 * (20 / 717) + 0.5 * (37 / 512)
    expected_attr_3 = 4 / 9
    np.testing.assert_allclose(
        complexity,
        [expected_attr_1, expected_attr_2, expected_attr_3],
        rtol=1e-12,
        atol=1e-12,
    )


def test_categorical_similarity_matches_table_7():
    X_cat, labels = paper_table_1()
    idc = compute_intuitionistic_centroids(
        X_cat,
        labels,
        n_clusters=2,
        non_membership="table",
    )

    similarity = categorical_similarity(idc, labels, n_clusters=2)

    expected_attr_1 = 0.5
    expected_attr_2 = 1 - ((3 / 20) + (27 / 25) + (17 / 10)) / 6
    expected_attr_3 = 1 - ((2 / 5) + (2 / 5)) / 4
    np.testing.assert_allclose(
        similarity,
        [expected_attr_1, expected_attr_2, expected_attr_3],
        rtol=1e-12,
        atol=1e-12,
    )


def test_categorical_weights_match_table_8():
    X_cat, labels = paper_table_1()
    idc = compute_intuitionistic_centroids(
        X_cat,
        labels,
        n_clusters=2,
        non_membership="table",
    )

    weights, phi, complexity, similarity = categorical_weights(
        idc,
        labels,
        n_clusters=2,
        gamma=0.5,
        beta=2.0,
    )
    scaled = weights * 3

    np.testing.assert_allclose(
        phi,
        0.5 * complexity + 0.5 * similarity,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(phi, [0.32142857, 0.28087324, 0.62222222], rtol=1e-6)

    # Table 8 appears to contain a small typo for Attribute2: the printed
    # weights do not sum to 1, and the printed scaled weights do not sum to 3.
    # These expectations follow Eq. (22) from the printed phi values.
    np.testing.assert_allclose(weights, [0.375803, 0.430065, 0.194133], rtol=1e-5)
    np.testing.assert_allclose(scaled, [1.127408, 1.290194, 0.582398], rtol=1e-5)


def test_categorical_distance_matches_table_9():
    X_cat, labels = paper_table_1()
    idc = compute_intuitionistic_centroids(
        X_cat,
        labels,
        n_clusters=2,
        non_membership="table",
    )

    x0 = np.asarray([["b", "e", "h"]], dtype=object)
    distances = categorical_attribute_distances(x0, idc, mu_param=0.5)

    expected_cluster_1 = [0.125, 0.095, 0.160]
    expected_cluster_2 = [0.875, 0.635, 0.360]
    np.testing.assert_allclose(distances[0, 0], expected_cluster_1, atol=1e-12)
    np.testing.assert_allclose(distances[0, 1], expected_cluster_2, atol=1e-12)
