import numpy as np
import pytest
from scipy import stats

from pthin.inference import _r_theta, conditional_confidence_interval


def test_no_truncation_recovers_standard_normal_ci():
    # a -> 0, b -> 1 means "always conduct inference" (no conditioning), so
    # the conditional interval should collapse to the textbook z-interval.
    t_obs, theta0, alpha = 1.3, 0.0, 0.05
    lo, hi = conditional_confidence_interval(
        t_obs, theta0, a=1e-9, b=1 - 1e-9, epsilon=0.5, alpha=alpha, input_type="statistic"
    )
    z = stats.norm.isf(alpha / 2)
    np.testing.assert_allclose([lo, hi], [t_obs - z, t_obs + z], atol=1e-4)


def test_pvalue_and_statistic_inputs_agree():
    theta0, a, b, epsilon, alpha = 0.0, 0.05, 0.4, 0.6, 0.1
    t_obs = 1.4
    p_obs = stats.norm.sf(t_obs, loc=theta0, scale=1.0)

    lo_t, hi_t = conditional_confidence_interval(
        t_obs, theta0, a, b, epsilon=epsilon, alpha=alpha, input_type="statistic"
    )
    lo_p, hi_p = conditional_confidence_interval(
        p_obs, theta0, a, b, epsilon=epsilon, alpha=alpha, input_type="pvalue"
    )
    np.testing.assert_allclose([lo_t, hi_t], [lo_p, hi_p], rtol=1e-6)


def test_ci_lower_below_upper():
    lo, hi = conditional_confidence_interval(
        1.4, theta0=0.0, a=0.05, b=0.4, epsilon=0.6, alpha=0.1, input_type="statistic"
    )
    assert lo < hi


def test_r_theta_is_increasing_in_theta():
    # R_theta(t) plays the role of a conditional CDF of theta given t, so it
    # must be monotonically increasing in theta for the interval inversion
    # {theta : R_theta(t) in [alpha/2, 1-alpha/2]} to be well defined.
    t_obs, theta0, a, b, epsilon = 1.2, 0.0, 0.1, 0.5, 0.5
    thetas = np.linspace(-2, 2, 9)
    r_values = [
        _r_theta(theta, t_obs, theta0, a, b, epsilon, "normal") for theta in thetas
    ]
    assert np.all(np.diff(r_values) > 0)


def test_custom_normal_callable_matches_builtin_normal_string():
    theta0, a, b, epsilon, alpha, t_obs = 0.0, 0.05, 0.4, 0.6, 0.1, 1.4
    lo_builtin, hi_builtin = conditional_confidence_interval(
        t_obs, theta0, a, b, epsilon=epsilon, alpha=alpha, input_type="statistic"
    )
    lo_custom, hi_custom = conditional_confidence_interval(
        t_obs,
        theta0,
        a,
        b,
        epsilon=epsilon,
        alpha=alpha,
        input_type="statistic",
        density=lambda theta: stats.norm(loc=theta, scale=1.0),
    )
    np.testing.assert_allclose(
        [lo_builtin, hi_builtin], [lo_custom, hi_custom], rtol=1e-6
    )


def test_custom_density_r_theta_matches_analytic_reference():
    # With no truncation (a -> 0, b -> 1), R_theta(t) reduces exactly to the
    # family's own upper-tailed p-value 1 - G_theta(t), regardless of family
    # -- so this checks the custom-density code path against a
    # family-agnostic closed form without the cost of full root-finding.
    t_obs, theta0, a, b, epsilon = 1.1, 0.0, 1e-9, 1 - 1e-9, 0.5
    laplace = lambda theta: stats.laplace(loc=theta, scale=1.0)
    for theta in [-0.5, 0.0, 0.7]:
        r_value = _r_theta(theta, t_obs, theta0, a, b, epsilon, laplace)
        reference = stats.laplace.sf(t_obs, loc=theta, scale=1.0)
        assert r_value == pytest.approx(reference, abs=1e-4)


def test_invalid_density_raises():
    with pytest.raises(ValueError):
        conditional_confidence_interval(
            1.0, 0.0, a=0.05, b=0.4, density=42, input_type="statistic"
        )


def test_invalid_selection_interval_raises():
    with pytest.raises(ValueError):
        conditional_confidence_interval(1.0, 0.0, a=0.5, b=0.4, input_type="statistic")


def test_invalid_epsilon_raises():
    with pytest.raises(ValueError):
        conditional_confidence_interval(
            1.0, 0.0, a=0.05, b=0.4, epsilon=1.5, input_type="statistic"
        )


def test_invalid_alpha_raises():
    with pytest.raises(ValueError):
        conditional_confidence_interval(
            1.0, 0.0, a=0.05, b=0.4, alpha=1.5, input_type="statistic"
        )


def test_invalid_input_kind_raises():
    with pytest.raises(ValueError):
        conditional_confidence_interval(
            1.0, 0.0, a=0.05, b=0.4, input_type="not-a-real-option"
        )


def test_pvalue_outside_selection_interval_raises():
    with pytest.raises(ValueError):
        conditional_confidence_interval(0.9, 0.0, a=0.05, b=0.4, input_type="pvalue")
