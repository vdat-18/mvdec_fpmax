"""Tests for the init_strategy and mu_convention options."""

import numpy as np
import pytest
from intuitive_kprototypes import IntuitiveKPrototypes
from intuitive_kprototypes.model import (
    INIT_STRATEGIES,
    mixed_initial_distance_matrix,
    select_initial_prototypes,
)


def _blobs(seed: int = 0, n_per_cluster: int = 20, k: int = 4):
    rng = np.random.default_rng(seed)
    # Positive offsets: Eq. (5) divides by the cluster mean.
    X_num = np.vstack(
        [
            rng.normal(loc=10.0 + 4.0 * c, scale=0.6, size=(n_per_cluster, 3))
            for c in range(k)
        ]
    )
    levels = np.array(list("abcd"), dtype=object)
    X_cat = np.vstack(
        [
            rng.choice(levels[[c, (c + 1) % k]], size=(n_per_cluster, 2))
            for c in range(k)
        ]
    ).astype(object)
    return X_num, X_cat


def _reference_first_admissible(X_num, X_cat, k, lambda_init, seed):
    """Algorithm 1 as shipped before the options: Python loop, first admissible."""

    n = X_num.shape[0]
    rng = np.random.default_rng(seed)
    idx0 = int(rng.integers(0, n))
    d0 = mixed_initial_distance_matrix(X_num, X_cat, X_num[[idx0]], X_cat[[idx0]])
    idx1 = int(np.argmax(d0[:, 0]))
    d1 = mixed_initial_distance_matrix(X_num, X_cat, X_num[[idx1]], X_cat[[idx1]])
    idx2 = int(np.argmax(d1[:, 0]))
    d_lambda = lambda_init * float(d1[idx2, 0]) / k
    selected = [idx1] + ([idx2] if idx2 != idx1 else [])
    while len(selected) < k:
        for idx in range(n):
            if idx in selected:
                continue
            dist = mixed_initial_distance_matrix(
                X_num[[idx]], X_cat[[idx]], X_num[selected], X_cat[selected]
            )[0]
            if np.all(dist >= d_lambda):
                selected.append(idx)
                break
        else:
            msg = "reference found no admissible row"
            raise AssertionError(msg)
    return np.asarray(selected, dtype=int)


@pytest.mark.parametrize("k", [2, 3, 4])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_first_strategy_matches_previous_algorithm_1(k, seed):
    X_num, X_cat = _blobs(seed=3)
    expected = _reference_first_admissible(X_num, X_cat, k, 0.3, seed)
    actual = select_initial_prototypes(
        X_num, X_cat, k, lambda_init=0.3, random_state=seed, strategy="first"
    )
    assert actual.tolist() == expected.tolist()


@pytest.mark.parametrize("strategy", INIT_STRATEGIES)
def test_every_strategy_returns_distinct_rows_and_is_deterministic(strategy):
    X_num, X_cat = _blobs()
    first = select_initial_prototypes(
        X_num, X_cat, 4, lambda_init=0.3, random_state=7, strategy=strategy
    )
    again = select_initial_prototypes(
        X_num, X_cat, 4, lambda_init=0.3, random_state=7, strategy=strategy
    )
    assert first.tolist() == again.tolist()
    assert len(set(first.tolist())) == 4


def test_random_candidate_equals_first_when_there_is_no_free_choice():
    X_num, X_cat = _blobs(k=2)
    for seed in range(5):
        first = select_initial_prototypes(
            X_num, X_cat, 2, lambda_init=0.3, random_state=seed, strategy="first"
        )
        random_candidate = select_initial_prototypes(
            X_num,
            X_cat,
            2,
            lambda_init=0.3,
            random_state=seed,
            strategy="random_candidate",
        )
        assert first.tolist() == random_candidate.tolist()


def test_random_candidate_picks_admissible_rows_beyond_the_first_one():
    X_num, X_cat = _blobs()
    chosen = {
        tuple(
            select_initial_prototypes(
                X_num,
                X_cat,
                4,
                lambda_init=0.3,
                random_state=seed,
                strategy="random_candidate",
            ).tolist()
        )
        for seed in range(12)
    }
    assert len(chosen) > 1


def test_unknown_init_strategy_is_rejected():
    X_num, X_cat = _blobs()
    with pytest.raises(ValueError, match="strategy"):
        select_initial_prototypes(X_num, X_cat, 4, strategy="bogus")
    with pytest.raises(ValueError, match="init_strategy"):
        IntuitiveKPrototypes(n_clusters=4, init_strategy="bogus").fit(
            X_num=X_num, X_cat=X_cat
        )


def _fit(mu, convention):
    X_num, X_cat = _blobs(seed=5)
    model = IntuitiveKPrototypes(
        n_clusters=4,
        lambda_init=0.3,
        mu_param=mu,
        gamma=0.5,
        max_iter=15,
        random_state=1,
        strict_init=False,
        empty_cluster_policy="farthest",
        mu_convention=convention,
    )
    return model.fit(X_num=X_num, X_cat=X_cat).result_


def test_swapped_convention_equals_eq26_at_one_minus_mu():
    swapped = _fit(0.2, "swapped")
    eq26 = _fit(0.8, "eq26")
    np.testing.assert_allclose(swapped.distances, eq26.distances)
    assert swapped.labels.tolist() == eq26.labels.tolist()


def test_conventions_coincide_at_mu_one_half():
    np.testing.assert_allclose(
        _fit(0.5, "swapped").distances, _fit(0.5, "eq26").distances
    )


def test_unknown_mu_convention_is_rejected():
    X_num, X_cat = _blobs()
    with pytest.raises(ValueError, match="mu_convention"):
        IntuitiveKPrototypes(n_clusters=4, mu_convention="bogus").fit(
            X_num=X_num, X_cat=X_cat
        )
