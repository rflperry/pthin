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


def test_pvalue_outside_selection_interval_raises():
    with pytest.raises(ValueError):
        pcarve_estimate(0.9, 0.0, a=0.05, b=0.4, input_type="pvalue")


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
