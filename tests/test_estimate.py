import numpy as np
import pytest
from scipy import stats

from pthin.estimate import pcarve_estimate


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
