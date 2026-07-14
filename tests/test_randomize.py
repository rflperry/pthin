import numpy as np
import pytest
from scipy import stats

from pthin.randomize import pthin


def test_output_shape_matches_input():
    p = np.random.default_rng(0).uniform(0, 1, size=100)
    p1, p2 = pthin(p, rng=np.random.default_rng(1))
    assert p1.shape == p.shape
    assert p2.shape == p.shape


def test_scalar_input():
    p1, p2 = pthin(0.5, rng=np.random.default_rng(0))
    assert p1.shape == ()
    assert p2.shape == ()


def test_list_input():
    p1, p2 = pthin([0.1, 0.5, 0.9], rng=np.random.default_rng(0))
    assert p1.shape == (3,)
    assert p2.shape == (3,)


def test_empty_input():
    p1, p2 = pthin(np.array([]), rng=np.random.default_rng(0))
    assert p1.shape == (0,)
    assert p2.shape == (0,)


def test_outputs_bounded_in_unit_interval():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, size=10_000)
    p1, p2 = pthin(p, epsilon=0.3, rng=rng)
    assert np.all((p1 >= 0) & (p1 <= 1))
    assert np.all((p2 >= 0) & (p2 <= 1))


def test_epsilon_one_returns_p_unchanged_as_p1():
    p = np.array([0.01, 0.2, 0.5, 0.9])
    p1, p2 = pthin(p, epsilon=1.0, rng=np.random.default_rng(0))
    np.testing.assert_allclose(p1, p)
    assert np.all((p2 >= 0) & (p2 <= 1))


def test_epsilon_zero_returns_p_unchanged_as_p2():
    p = np.array([0.01, 0.2, 0.5, 0.9])
    p1, p2 = pthin(p, epsilon=0.0, rng=np.random.default_rng(0))
    np.testing.assert_allclose(p2, p)
    assert np.all((p1 >= 0) & (p1 <= 1))


def test_invalid_epsilon_raises():
    with pytest.raises(ValueError):
        pthin(np.array([0.5]), epsilon=-0.1)
    with pytest.raises(ValueError):
        pthin(np.array([0.5]), epsilon=1.1)


def test_invalid_p_raises():
    with pytest.raises(ValueError):
        pthin(np.array([-0.1, 0.5]))
    with pytest.raises(ValueError):
        pthin(np.array([0.5, 1.1]))


def test_deterministic_with_seeded_generator():
    p = np.array([0.01, 0.2, 0.5, 0.9])
    p1_a, p2_a = pthin(p, epsilon=0.5, rng=np.random.default_rng(42))
    p1_b, p2_b = pthin(p, epsilon=0.5, rng=np.random.default_rng(42))
    np.testing.assert_array_equal(p1_a, p1_b)
    np.testing.assert_array_equal(p2_a, p2_b)


@pytest.mark.parametrize("epsilon", [0.1, 0.5, 0.9])
def test_thinned_pvalues_are_marginally_uniform_under_null(epsilon):
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, size=200_000)
    p1, p2 = pthin(p, epsilon=epsilon, rng=rng)

    # Loose alpha to avoid flakiness; this checks marginal (super-)uniformity,
    # a required property for p1/p2 to themselves be valid null p-values.
    assert stats.kstest(p1, "uniform").pvalue > 0.001
    assert stats.kstest(p2, "uniform").pvalue > 0.001


def test_thinned_pvalues_are_approximately_independent_under_null():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, size=200_000)
    p1, p2 = pthin(p, epsilon=0.5, rng=rng)

    corr = np.corrcoef(p1, p2)[0, 1]
    assert abs(corr) < 0.01
