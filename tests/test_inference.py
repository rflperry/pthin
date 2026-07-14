import numpy as np
import pytest
from scipy import stats

from scipy.integrate import quad
from scipy.optimize import brentq

from pthin.inference import (
    _nu_cdf,
    _nu_density,
    _r_theta,
    _truncgauss_survival,
    pcarve_ci,
    pcarve_threshold,
    truncgauss_ci,
    truncgauss_pvalue,
)


def test_no_truncation_recovers_standard_normal_ci():
    # a -> 0, b -> 1 means "always conduct inference" (no conditioning), so
    # the conditional interval should collapse to the textbook z-interval.
    t_obs, theta0, alpha = 1.3, 0.0, 0.05
    lo, hi = pcarve_ci(
        t_obs, theta0, a=1e-9, b=1 - 1e-9, epsilon=0.5, alpha=alpha, input_type="statistic"
    )
    z = stats.norm.isf(alpha / 2)
    np.testing.assert_allclose([lo, hi], [t_obs - z, t_obs + z], atol=1e-4)


def test_pvalue_and_statistic_inputs_agree():
    theta0, a, b, epsilon, alpha = 0.0, 0.05, 0.4, 0.6, 0.1
    t_obs = 1.4
    p_obs = stats.norm.sf(t_obs, loc=theta0, scale=1.0)

    lo_t, hi_t = pcarve_ci(
        t_obs, theta0, a, b, epsilon=epsilon, alpha=alpha, input_type="statistic"
    )
    lo_p, hi_p = pcarve_ci(
        p_obs, theta0, a, b, epsilon=epsilon, alpha=alpha, input_type="pvalue"
    )
    np.testing.assert_allclose([lo_t, hi_t], [lo_p, hi_p], rtol=1e-6)


def test_ci_lower_below_upper():
    lo, hi = pcarve_ci(
        1.4, theta0=0.0, a=0.05, b=0.4, epsilon=0.6, alpha=0.1, input_type="statistic"
    )
    assert lo < hi


# --- a=0 fast path (argmin-style selection: b is the runner-up's p-value) ---


def _reference_calc_r_mu(mu, x_obs, c, epsilon):
    # Independent re-derivation of the a=0 special case for arbitrary
    # epsilon (not copy-pasted from experiments/normal_ci.ipynb, which
    # hardcodes epsilon=0.5), used as ground truth for _r_theta's a=0 path.
    z_jump = stats.norm.isf(c ** (1 / epsilon))
    weight = (1 - epsilon) * c

    def tail(z):
        return weight * stats.norm.pdf(z - mu) * stats.norm.sf(z) ** (-epsilon)

    lb, ub = mu - 20, min(z_jump, mu + 20)
    den_lower = quad(tail, lb, ub)[0] if lb < ub else 0.0
    den_upper = stats.norm.sf(z_jump - mu)
    den = den_lower + den_upper

    if x_obs < z_jump:
        lb_num = max(x_obs, mu - 20)
        num_lower = quad(tail, lb_num, ub)[0] if lb_num < ub else 0.0
        num = num_lower + den_upper
    else:
        num = stats.norm.sf(x_obs - mu)
    return num / den


@pytest.mark.parametrize("epsilon", [0.3, 0.5, 0.7])
def test_a0_fast_path_matches_independent_reference(epsilon):
    x_obs, c, mu = 1.5, 0.3, 0.7
    fast = _r_theta(mu, x_obs, theta0=0.0, a=0.0, b=c, epsilon=epsilon, density="normal")
    reference = _reference_calc_r_mu(mu, x_obs, c, epsilon)
    assert fast == pytest.approx(reference, abs=1e-6)


@pytest.mark.parametrize("epsilon", [0.3, 0.5, 0.7])
def test_a0_fast_path_matches_general_a_path_in_the_limit(epsilon):
    x_obs, c, mu = 1.5, 0.3, 0.7
    fast = _r_theta(mu, x_obs, theta0=0.0, a=0.0, b=c, epsilon=epsilon, density="normal")
    general = _r_theta(mu, x_obs, theta0=0.0, a=1e-9, b=c, epsilon=epsilon, density="normal")
    assert fast == pytest.approx(general, abs=1e-6)


def test_pcarve_ci_allows_a_equals_zero():
    lo, hi = pcarve_ci(
        1.5, theta0=0.0, a=0.0, b=0.3, epsilon=0.5, alpha=0.05, input_type="statistic"
    )
    lo_ref, hi_ref = pcarve_ci(
        1.5, theta0=0.0, a=1e-9, b=0.3, epsilon=0.5, alpha=0.05, input_type="statistic"
    )
    np.testing.assert_allclose([lo, hi], [lo_ref, hi_ref], atol=1e-4)


def test_pcarve_ci_rejects_negative_a():
    with pytest.raises(ValueError):
        pcarve_ci(1.5, 0.0, a=-0.1, b=0.4, input_type="statistic")


def test_pcarve_ci_custom_tolerance_is_close_to_default():
    # Loose tolerances (as used for simulation-scale calls in
    # experiments/normal_ci.ipynb) should still be reasonably accurate, just
    # faster -- not exact to the same precision as the tight defaults.
    kwargs = dict(theta0=0.0, a=0.0, b=0.3, epsilon=0.5, alpha=0.05, input_type="statistic")
    lo_default, hi_default = pcarve_ci(1.5, **kwargs)
    lo_loose, hi_loose = pcarve_ci(1.5, epsabs=1e-4, epsrel=1e-4, limit=50, **kwargs)
    np.testing.assert_allclose([lo_loose, hi_loose], [lo_default, hi_default], atol=1e-3)


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
    lo_builtin, hi_builtin = pcarve_ci(
        t_obs, theta0, a, b, epsilon=epsilon, alpha=alpha, input_type="statistic"
    )
    lo_custom, hi_custom = pcarve_ci(
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
        pcarve_ci(
            1.0, 0.0, a=0.05, b=0.4, density=42, input_type="statistic"
        )


def test_invalid_selection_interval_raises():
    with pytest.raises(ValueError):
        pcarve_ci(1.0, 0.0, a=0.5, b=0.4, input_type="statistic")


def test_invalid_epsilon_raises():
    with pytest.raises(ValueError):
        pcarve_ci(
            1.0, 0.0, a=0.05, b=0.4, epsilon=1.5, input_type="statistic"
        )


def test_invalid_alpha_raises():
    with pytest.raises(ValueError):
        pcarve_ci(
            1.0, 0.0, a=0.05, b=0.4, alpha=1.5, input_type="statistic"
        )


def test_invalid_input_kind_raises():
    with pytest.raises(ValueError):
        pcarve_ci(
            1.0, 0.0, a=0.05, b=0.4, input_type="not-a-real-option"
        )


def test_raw_pvalue_outside_ab_does_not_raise():
    # a, b describe the selection event on the *thinned* p-value p1(T), not
    # on T's own raw p-value -- pcarve_ci never sees p1, so it can't and
    # shouldn't validate against it. Regression test: this used to
    # incorrectly raise, which (per test_argmin_selection_raw_pvalue_is_
    # usually_outside_ab below) rejected the majority of legitimate calls
    # in an argmin-style selection scenario.
    pcarve_ci(0.9, 0.0, a=0.05, b=0.4, input_type="pvalue")


def test_argmin_selection_raw_pvalue_is_usually_outside_ab():
    # Empirical check backing the above: in the winner's-curse-style
    # selection used by experiments/normal_ci.ipynb (sel = argmin(p1), b =
    # runner-up's p1), the raw p-value of the selected statistic is outside
    # [0, b] most of the time, even though the actual precondition p1(T)
    # <= b holds by construction every time.
    rng = np.random.default_rng(0)
    n_outside = 0
    n_reps = 500
    for _ in range(n_reps):
        x = rng.normal(0, 1, size=10)
        p = stats.norm.sf(x)
        z, c = rng.uniform(0, 1, size=10), rng.integers(0, 2, size=10)
        p1 = np.sqrt(p) * (c + (1 - c) * z)
        sel = np.argmin(p1)
        b = np.sort(p1)[1]
        p_raw = stats.norm.sf(x[sel])
        if not (0 <= p_raw <= b):
            n_outside += 1
    assert n_outside > n_reps / 4


@pytest.mark.parametrize(
    "a,b,epsilon", [(0.1, 0.4, 0.5), (0.01, 0.9, 0.3), (0.2, 0.25, 0.7), (0.001, 0.5, 0.5)]
)
def test_nu_cdf_matches_numerical_integration(a, b, epsilon):
    # _nu_cdf is a hand-derived closed-form antiderivative of -_nu_density
    # (the (b-a)-normalized, sign-corrected version -- see _nu_cdf's
    # docstring); check it against numerically integrating the density.
    for t in [0.005, 0.05, 0.2, 0.5, 0.9, 1.0]:
        numeric, _ = quad(lambda q: -_nu_density(q, a, b, epsilon), 0, t, limit=200)
        assert _nu_cdf(t, a, b, epsilon) == pytest.approx(numeric, abs=1e-8)


@pytest.mark.parametrize(
    "a,b,epsilon", [(0.1, 0.4, 0.5), (0.01, 0.9, 0.3), (0.2, 0.25, 0.7), (0.001, 0.5, 0.5)]
)
def test_nu_cdf_is_a_valid_cdf(a, b, epsilon):
    assert _nu_cdf(0.0, a, b, epsilon) == 0.0
    assert _nu_cdf(1.0, a, b, epsilon) == pytest.approx(1.0, abs=1e-10)
    ts = np.linspace(0, 1, 50)
    values = [_nu_cdf(t, a, b, epsilon) for t in ts]
    assert np.all(np.diff(values) >= 0)


def test_pcarve_threshold_inverts_nu_cdf():
    alpha, a, b, epsilon = 0.05, 0.1, 0.4, 0.5
    t_star = pcarve_threshold(alpha, a, b, epsilon=epsilon)
    assert _nu_cdf(t_star, a, b, epsilon) == pytest.approx(alpha, abs=1e-10)


def test_pcarve_threshold_increasing_in_alpha():
    a, b, epsilon = 0.1, 0.4, 0.5
    alphas = [0.01, 0.05, 0.1, 0.3, 0.5, 0.9]
    thresholds = [
        pcarve_threshold(alpha, a, b, epsilon=epsilon)
        for alpha in alphas
    ]
    assert np.all(np.diff(thresholds) > 0)


def test_pcarve_threshold_invalid_alpha_raises():
    with pytest.raises(ValueError):
        pcarve_threshold(1.5, a=0.1, b=0.4)


def test_pcarve_threshold_invalid_selection_interval_raises():
    with pytest.raises(ValueError):
        pcarve_threshold(0.05, a=0.5, b=0.4)


def test_pcarve_threshold_invalid_epsilon_raises():
    with pytest.raises(ValueError):
        pcarve_threshold(0.05, a=0.1, b=0.4, epsilon=1.5)


# --- truncgauss: conditional selective inference for T ~ N(theta, scale^2) | T > c ---


def _reference_truncgauss_survival(mu, t_obs, c):
    # Independent re-derivation (not copy-pasted from pthin) matching
    # experiments/normal_ci.ipynb's calc_r_cond_mu, used as ground truth.
    log_num = stats.norm.logsf(t_obs - mu)
    log_den = stats.norm.logsf(c - mu)
    return np.exp(log_num - log_den)


@pytest.mark.parametrize(
    "t_obs,c,mu", [(1.5, 0.3, 0.0), (2.5, 1.0, 1.2), (0.8, 0.5, -0.3), (3.0, 2.9, 0.5)]
)
def test_truncgauss_survival_matches_reference(t_obs, c, mu):
    assert _truncgauss_survival(mu, t_obs, c, 1.0) == pytest.approx(
        _reference_truncgauss_survival(mu, t_obs, c), abs=1e-12
    )


def test_truncgauss_pvalue_matches_survival_at_theta0():
    t_obs, theta0, c = 1.5, 0.0, 0.3
    assert truncgauss_pvalue(t_obs, theta0, c) == pytest.approx(
        _truncgauss_survival(theta0, t_obs, c, 1.0), abs=1e-12
    )


def test_truncgauss_pvalue_is_uniform_under_null():
    rng = np.random.default_rng(0)
    theta0, c = 0.0, 0.5
    n = 200_000
    truncated = stats.truncnorm.rvs(
        a=c - theta0, b=np.inf, loc=theta0, scale=1.0, size=n, random_state=rng
    )

    p_values = stats.norm.sf(truncated, loc=theta0, scale=1.0) / stats.norm.sf(
        c, loc=theta0, scale=1.0
    )
    assert stats.kstest(p_values, "uniform").pvalue > 0.001


def test_truncgauss_survival_is_increasing_in_theta():
    t_obs, c = 1.2, 0.3
    thetas = np.linspace(-2, 2, 9)
    values = [_truncgauss_survival(theta, t_obs, c, 1.0) for theta in thetas]
    assert np.all(np.diff(values) > 0)


def test_truncgauss_ci_matches_reference():
    t_obs, c, alpha = 1.5, 0.3, 0.05

    def f(mu, target):
        return _reference_truncgauss_survival(mu, t_obs, c) - target

    lo_ref = brentq(lambda mu: f(mu, alpha / 2), t_obs - 50, t_obs + 50)
    hi_ref = brentq(lambda mu: f(mu, 1 - alpha / 2), t_obs - 50, t_obs + 50)

    lo, hi = truncgauss_ci(t_obs, c, alpha=alpha)
    assert (lo, hi) == pytest.approx((lo_ref, hi_ref), abs=1e-6)


def test_truncgauss_ci_lower_below_upper():
    lo, hi = truncgauss_ci(1.5, c=0.3, alpha=0.1)
    assert lo < hi


def test_truncgauss_ci_invalid_scale_raises():
    with pytest.raises(ValueError):
        truncgauss_ci(1.5, c=0.3, scale=-1.0)


def test_truncgauss_ci_invalid_alpha_raises():
    with pytest.raises(ValueError):
        truncgauss_ci(1.5, c=0.3, alpha=1.5)


def test_truncgauss_ci_below_threshold_raises():
    with pytest.raises(ValueError):
        truncgauss_ci(0.1, c=0.3)


def test_truncgauss_pvalue_below_threshold_raises():
    with pytest.raises(ValueError):
        truncgauss_pvalue(0.1, theta0=0.0, c=0.3)
