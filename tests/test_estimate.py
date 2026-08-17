import numpy as np
import pytest
from scipy import stats

from pthin.estimate import pcarve_estimate, truncgauss_estimate


def test_no_truncation_all_estimators_recover_t_obs():
    # a -> 0, b -> 1 means "always conduct inference" (no conditioning), so
    # the conditional likelihood in theta reduces to the family's own
    # (symmetric) density g_theta(t_obs), whose mode and mean both sit
    # exactly at theta = t_obs.
    theta0, t_obs = 0.0, 1.3
    a, b = 1e-9, 1 - 1e-9
    for estimator in ["mle", "mean", "combined"]:
        value = pcarve_estimate(
            t_obs, theta0, a, b, estimator=estimator, input_type="statistic"
        )
        assert value == pytest.approx(t_obs, abs=1e-3)


def test_combined_is_average_of_mean_and_mle():
    theta0, t_obs, a, b = 0.0, 1.2, 0.1, 0.4
    mle = pcarve_estimate(
        t_obs, theta0, a, b, estimator="mle", input_type="statistic"
    )
    mean = pcarve_estimate(
        t_obs, theta0, a, b, estimator="mean", input_type="statistic"
    )
    combined = pcarve_estimate(
        t_obs, theta0, a, b, estimator="combined", input_type="statistic"
    )
    assert combined == pytest.approx((mean + mle) / 2, abs=1e-6)


def test_pvalue_and_statistic_inputs_agree_for_mle():
    theta0, a, b = 0.0, 0.1, 0.4
    t_obs = 1.2
    p_obs = stats.norm.sf(t_obs, loc=theta0, scale=1.0)
    mle_t = pcarve_estimate(t_obs, theta0, a, b, input_type="statistic")
    mle_p = pcarve_estimate(p_obs, theta0, a, b, input_type="pvalue")
    assert mle_t == pytest.approx(mle_p, abs=1e-6)


def test_invalid_estimator_raises():
    with pytest.raises(ValueError):
        pcarve_estimate(
            1.2, 0.0, a=0.1, b=0.4, estimator="bogus", input_type="statistic"
        )


def test_invalid_selection_interval_raises():
    with pytest.raises(ValueError):
        pcarve_estimate(1.2, 0.0, a=0.5, b=0.4, input_type="statistic")


def test_invalid_epsilon_raises():
    with pytest.raises(ValueError):
        pcarve_estimate(
            1.2, 0.0, a=0.1, b=0.4, epsilon=1.5, input_type="statistic"
        )


def test_invalid_density_raises():
    with pytest.raises(ValueError):
        pcarve_estimate(
            1.2, 0.0, a=0.1, b=0.4, density=42, input_type="statistic"
        )


def test_invalid_input_kind_raises():
    with pytest.raises(ValueError):
        pcarve_estimate(
            1.2, 0.0, a=0.1, b=0.4, input_type="not-a-real-option"
        )


def test_raw_pvalue_outside_ab_does_not_raise():
    # a, b describe the selection event on the thinned p-value p1(T), not
    # on T's own raw p-value -- see the matching regression test in
    # test_inference.py for why this must not raise.
    pcarve_estimate(0.9, 0.0, a=0.05, b=0.4, input_type="pvalue")


def test_a0_mle_and_mean_are_pulled_below_t_obs_and_close_to_general_path():
    # Regression test for the _r_theta_a0 scale-mismatch bug: with a=0 this
    # used to return wildly wrong values (e.g. t_obs + 5, an artifact of
    # the optimizer chasing a corrupted, sign-broken likelihood to a search
    # bracket's edge) instead of a real winner's-curse correction.
    theta0, t_obs, b = 0.0, 2.126, 0.1
    for estimator in ["mle", "mean"]:
        a0 = pcarve_estimate(
            t_obs, theta0, a=0.0, b=b, estimator=estimator, input_type="statistic",
            epsabs=1e-6, epsrel=1e-6,
        )
        general = pcarve_estimate(
            t_obs, theta0, a=1e-9, b=b, estimator=estimator, input_type="statistic",
            epsabs=1e-6, epsrel=1e-6,
        )
        assert a0 < t_obs
        assert a0 == pytest.approx(general, abs=0.05)


# --- theta_min/theta_max: bounded parameter domains (e.g. reflection-symmetric families) ---


def test_theta_bound_mean_matches_mle_ballpark_for_symmetric_family():
    # Regression test: with theta_min=0 against a reflection-symmetric
    # family (g_theta = g_{-theta}), _mean's radius-expansion used to treat
    # hitting the theta_min boundary as "not yet decayed" (since the
    # density right at theta=0 need not be small), so it kept expanding the
    # search radius all the way to the 1e4 cap -- producing a grid so wide
    # relative to n_points that the integral came out as exactly 0 (or
    # NaN), rather than a real magnitude estimate near the MLE.
    density = lambda theta: stats.foldnorm(c=theta, loc=0, scale=1)
    x_obs, b = 3.0, 0.05

    mle = pcarve_estimate(
        x_obs, 0.0, a=0.0, b=b, density=density, estimator="mle",
        input_type="statistic", epsabs=1e-6, epsrel=1e-6, theta_min=0.0,
    )
    mean = pcarve_estimate(
        x_obs, 0.0, a=0.0, b=b, density=density, estimator="mean",
        input_type="statistic", epsabs=1e-6, epsrel=1e-6, n_points=81, theta_min=0.0,
    )
    assert mle > 0
    assert mean > 0
    assert mean == pytest.approx(mle, abs=1.0)


def test_theta_min_zero_gives_nonnegative_estimates():
    density = lambda theta: stats.foldnorm(c=theta, loc=0, scale=1)
    for x_obs in [0.5, 1.5, 3.0, 5.0]:
        for estimator in ["mle", "mean", "combined"]:
            value = pcarve_estimate(
                x_obs, 0.0, a=0.0, b=0.05, density=density, estimator=estimator,
                input_type="statistic", epsabs=1e-6, epsrel=1e-6, n_points=61,
                theta_min=0.0,
            )
            assert value >= 0


def test_theta_bounds_no_truncation_recovers_t_obs_for_symmetric_family():
    # Same closed-form sanity check as the unconstrained no-truncation
    # test, but for a reflection-symmetric family with theta_min=0: with no
    # conditioning, the conditional likelihood is just g_theta(t_obs)
    # again, whose mode/mean (restricted to theta>=0) sit at theta=t_obs
    # for t_obs comfortably away from the theta_min boundary.
    density = lambda theta: stats.foldnorm(c=theta, loc=0, scale=1)
    x_obs = 3.0
    for estimator in ["mle", "mean"]:
        value = pcarve_estimate(
            x_obs, 0.0, a=1e-9, b=1 - 1e-9, density=density, estimator=estimator,
            input_type="statistic", theta_min=0.0,
        )
        assert value == pytest.approx(x_obs, abs=1e-2)


# --- truncgauss: conditional selective inference for T ~ N(theta, scale^2) | T > c ---


def test_truncgauss_far_below_threshold_recovers_t_obs():
    # c far below t_obs means the truncation barely binds, so the
    # conditional likelihood reduces to the family's own (symmetric)
    # density g_theta(t_obs), whose mode and mean both sit at theta = t_obs.
    t_obs, c = 1.3, -50.0
    for estimator in ["mle", "mean"]:
        value = truncgauss_estimate(t_obs, c, estimator=estimator)
        assert value == pytest.approx(t_obs, abs=1e-4)


def test_truncgauss_estimates_pulled_below_t_obs_under_truncation():
    # Selecting because T > c biases the naive t_obs upward (winner's
    # curse); both conditional estimators should correct downward from it.
    t_obs, c = 1.5, 0.3
    mle = truncgauss_estimate(t_obs, c, estimator="mle")
    mean = truncgauss_estimate(t_obs, c, estimator="mean")
    assert mle < t_obs
    assert mean < t_obs


def test_truncgauss_mle_matches_direct_optimization():
    from scipy.optimize import minimize_scalar

    t_obs, c = 1.5, 0.3
    objective = lambda theta: -(
        stats.norm.logpdf(t_obs, loc=theta) - stats.norm.logsf(c, loc=theta)
    )
    reference = minimize_scalar(
        objective, bounds=(t_obs - 50, t_obs + 50), method="bounded"
    ).x

    assert truncgauss_estimate(t_obs, c, estimator="mle") == pytest.approx(
        reference, abs=1e-6
    )


def test_truncgauss_invalid_scale_raises():
    with pytest.raises(ValueError):
        truncgauss_estimate(1.5, c=0.3, scale=-1.0)


def test_truncgauss_invalid_estimator_raises():
    with pytest.raises(ValueError):
        truncgauss_estimate(1.5, c=0.3, estimator="bogus")


def test_truncgauss_below_threshold_raises():
    with pytest.raises(ValueError):
        truncgauss_estimate(0.1, c=0.3)
