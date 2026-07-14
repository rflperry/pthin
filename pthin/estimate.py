"""Point estimation of a location parameter after selection on a thinned p-value."""

from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import minimize_scalar

from pthin.inference import DensityFamily, _conditional_likelihood, _p_value, _p_value_inv

__all__ = ["pcarve_estimate"]

_ESTIMATORS = ("mle", "mean", "combined")


def _log_likelihood(theta, t_obs, theta0, a, b, epsilon, density):
    likelihood = _conditional_likelihood(theta, t_obs, theta0, a, b, epsilon, density)
    return -np.inf if likelihood <= 0 else np.log(likelihood)


def _mle(t_obs, theta0, a, b, epsilon, density, search_radii=(5, 20, 60, 200)):
    """Maximize the conditional likelihood over theta via bracketed Brent search."""
    objective = lambda theta: -_log_likelihood(theta, t_obs, theta0, a, b, epsilon, density)
    result = None
    for radius in search_radii:
        lo, hi = t_obs - radius, t_obs + radius
        result = minimize_scalar(objective, bounds=(lo, hi), method="bounded")
        at_boundary = min(result.x - lo, hi - result.x) < 1e-6 * radius
        if not at_boundary:
            return result.x
    return result.x


def _mean(t_obs, theta0, a, b, epsilon, density, n_points=121):
    """Conditional mean via trapezoidal quadrature over a grid in theta.

    Each grid point costs its own numerical integration (see
    ``_conditional_likelihood``), so this evaluates the likelihood on a
    single shared grid spanning most of its mass rather than driving two
    independent adaptive-quadrature calls (numerator and normalizing
    constant) that would each rediscover where that mass lies from scratch.
    """
    peak = _conditional_likelihood(t_obs, t_obs, theta0, a, b, epsilon, density)
    radius = 4.0
    while radius <= 1e4:
        edge = max(
            _conditional_likelihood(t_obs - radius, t_obs, theta0, a, b, epsilon, density),
            _conditional_likelihood(t_obs + radius, t_obs, theta0, a, b, epsilon, density),
        )
        if edge < 1e-6 * max(peak, 1e-300):
            break
        radius *= 2

    thetas = np.linspace(t_obs - radius, t_obs + radius, n_points)
    likelihoods = np.array(
        [
            _conditional_likelihood(theta, t_obs, theta0, a, b, epsilon, density)
            for theta in thetas
        ]
    )
    total = trapezoid(likelihoods, thetas)
    return trapezoid(thetas * likelihoods, thetas) / total


def pcarve_estimate(
    stat: float,
    theta0: float,
    a: float,
    b: float,
    epsilon: float = 0.5,
    density: DensityFamily = "normal",
    input_type: str = "pvalue",
    estimator: str = "mle",
) -> float:
    r"""Point estimate of a location parameter after selection.

    Following :cite:`ghosh_estimating_2008`, this estimates :math:`\theta^*`
    from the conditional likelihood :math:`r_\theta(p_{\theta_0}(t))`
    induced by the same conditional distribution as
    :func:`pcarve_ci`'s :math:`R_\theta(t)`:
    :math:`r_\theta(p_{\theta_0}(t)) = -\frac{d}{dt} R_\theta(t) \big|_{t}`,
    the density (in :math:`t`) of :math:`T` at the observed value, given
    :math:`\theta` and conditional on the selection event
    :math:`p_{\theta_0}(T) \in [a, b]`.

    Three estimators are available via ``estimator``:

    - ``"mle"``: the conditional MLE,
      :math:`\hat\theta^{\mathrm{MLE}} := \arg\max_\theta r_\theta(p_{\theta_0}(t))`.
    - ``"mean"``: the conditional mean,
      :math:`\hat\theta^{\mathrm{mean}} := \int_\Theta \theta\, r_\theta(p_{\theta_0}(t))
      \, d\theta \big/ \int_\Theta r_\theta(p_{\theta_0}(t)) \, d\theta`,
      i.e. :math:`r_\theta(p_{\theta_0}(t))` normalized to a proper density
      over :math:`\theta` before taking its mean.
    - ``"combined"``: :math:`(\hat\theta^{\mathrm{mean}} +
      \hat\theta^{\mathrm{MLE}})/2`, the average of the conditional mean and
      the conditional MLE.

    Because :math:`r_\theta(p_{\theta_0}(t))` requires its own numerical
    integration per evaluation (see :func:`pthin.inference._denominator`),
    ``"mean"`` evaluates it on a grid over :math:`\theta` and integrates
    that via the trapezoidal rule (see :func:`_mean`) rather than with
    adaptive quadrature, so it is accurate only up to grid resolution as
    well as numerical-integration tolerance. ``"mle"`` uses direct
    bracketed optimization and is not subject to grid error; ``"combined"``
    inherits the grid error of its ``"mean"`` half.

    Parameters
    ----------
    stat : float
        Either the p-value :math:`p_{\theta_0}(T)` (default, when
        ``input_type="pvalue"``) or the raw test statistic :math:`T`
        (when ``input_type="statistic"``).
    theta0 : float
        Null value :math:`\theta_0` defining the upper-tailed p-value
        :math:`p_{\theta_0}(t) = 1 - G_{\theta_0}(t)`.
    a, b : float
        Endpoints of the selection interval: inference is conducted only
        given :math:`p_{\theta_0}(T) \in [a, b]`. Must satisfy
        ``0 < a < b < 1``.
    epsilon : float, default=0.5
        Thinning fraction used to construct the p-value used for selection,
        matching the ``epsilon`` of :func:`pthin.randomize.pthin`. Must lie
        in ``(0, 1)``.
    density : "normal" or callable, default="normal"
        The family :math:`\{g_\theta\}`, as in :func:`pcarve_ci`.
    input_type : {"pvalue", "statistic"}, default="pvalue"
        Whether ``stat`` is the p-value :math:`p_{\theta_0}(T)` or the raw
        statistic :math:`T`.
    estimator : {"mle", "mean", "combined"}, default="mle"
        Which point estimator to return.

    Returns
    -------
    theta_hat : float
        The requested point estimate of :math:`\theta^*`.

    Raises
    ------
    ValueError
        If ``a``, ``b``, or ``epsilon`` are out of range, if ``density``,
        ``input_type``, or ``estimator`` is not recognized, or if the
        observed p-value falls outside ``[a, b]``.
    """
    if not 0 < a < b < 1:
        raise ValueError(f"Require 0 < a < b < 1, got a={a}, b={b}")
    if not 0 < epsilon < 1:
        raise ValueError(f"epsilon must lie in (0, 1), got {epsilon}")
    if density != "normal" and not callable(density):
        raise ValueError(
            "density must be the string 'normal' or a callable "
            f"theta -> frozen scipy.stats distribution, got {density!r}"
        )
    if estimator not in _ESTIMATORS:
        raise ValueError(f"estimator must be one of {_ESTIMATORS}, got {estimator!r}")

    if input_type == "pvalue":
        p_obs = float(stat)
        t_obs = _p_value_inv(p_obs, theta0, density)
    elif input_type == "statistic":
        t_obs = float(stat)
        p_obs = _p_value(t_obs, theta0, density)
    else:
        raise ValueError(
            f"input_type must be 'pvalue' or 'statistic', got {input_type!r}"
        )

    if not a <= p_obs <= b:
        raise ValueError(
            f"Observed p-value {p_obs} lies outside the selection interval "
            f"[{a}, {b}]; the conditional estimate is undefined off the "
            "selection event."
        )

    if estimator == "mle":
        return _mle(t_obs, theta0, a, b, epsilon, density)
    if estimator == "mean":
        return _mean(t_obs, theta0, a, b, epsilon, density)
    return (
        _mean(t_obs, theta0, a, b, epsilon, density)
        + _mle(t_obs, theta0, a, b, epsilon, density)
    ) / 2
