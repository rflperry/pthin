"""Point estimation of a location parameter after selection on a thinned p-value."""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.integrate import quad, trapezoid
from scipy.optimize import minimize_scalar

from pthin.inference import DensityFamily, _conditional_likelihood, _p_value_inv

__all__ = ["pcarve_estimate", "truncgauss_estimate"]

_ESTIMATORS = ("mle", "mean", "combined")


def _log_likelihood(theta, t_obs, theta0, a, b, epsilon, density, quad_kwargs):
    likelihood = _conditional_likelihood(
        theta, t_obs, theta0, a, b, epsilon, density, quad_kwargs
    )
    return -np.inf if likelihood <= 0 else np.log(likelihood)


def _mle(t_obs, theta0, a, b, epsilon, density, quad_kwargs, search_radii=(5, 20, 60, 200)):
    """Maximize the conditional likelihood over theta via bracketed Brent search."""
    objective = lambda theta: -_log_likelihood(
        theta, t_obs, theta0, a, b, epsilon, density, quad_kwargs
    )
    result = None
    for radius in search_radii:
        lo, hi = t_obs - radius, t_obs + radius
        result = minimize_scalar(objective, bounds=(lo, hi), method="bounded")
        at_boundary = min(result.x - lo, hi - result.x) < 1e-6 * radius
        if not at_boundary:
            return result.x
    return result.x


def _mean(t_obs, theta0, a, b, epsilon, density, quad_kwargs, n_points):
    """Conditional mean via trapezoidal quadrature over a grid in theta.

    Each grid point costs its own numerical integration (see
    ``_conditional_likelihood``), so this evaluates the likelihood on a
    single shared grid spanning most of its mass rather than driving two
    independent adaptive-quadrature calls (numerator and normalizing
    constant) that would each rediscover where that mass lies from scratch.
    """
    likelihood_at = lambda theta: _conditional_likelihood(
        theta, t_obs, theta0, a, b, epsilon, density, quad_kwargs
    )
    peak = likelihood_at(t_obs)
    radius = 4.0
    while radius <= 1e4:
        edge = max(likelihood_at(t_obs - radius), likelihood_at(t_obs + radius))
        if edge < 1e-6 * max(peak, 1e-300):
            break
        radius *= 2

    thetas = np.linspace(t_obs - radius, t_obs + radius, n_points)
    likelihoods = np.array([likelihood_at(theta) for theta in thetas])
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
    epsabs: float = 1e-12,
    epsrel: float = 1e-9,
    limit: int = 100,
    n_points: int = 121,
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
        Either the raw p-value :math:`p_{\theta_0}(T)` of the tested
        statistic :math:`T` (default, when ``input_type="pvalue"``) or
        :math:`T` itself (when ``input_type="statistic"``) -- this is *not*
        the thinned p-value :math:`p_1(T)` used for selection below, which
        this function never sees directly.
    theta0 : float
        Null value :math:`\theta_0` defining the upper-tailed p-value
        :math:`p_{\theta_0}(t) = 1 - G_{\theta_0}(t)`.
    a, b : float
        Endpoints of the selection interval: inference is conducted only
        given :math:`p_1(T) \in [a, b]`, where :math:`p_1` is the thinned
        p-value from :func:`pthin.randomize.pthin`. Must satisfy ``0 <= a <
        b < 1``. This event is the *caller's* responsibility to have
        actually arranged; it is not, and cannot be, checked from ``stat``
        alone (see :func:`pcarve_ci`). ``a=0`` uses a much faster code path
        than ``a > 0`` -- see :func:`pthin.inference._r_theta_a0`.
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
    epsabs, epsrel, limit : float, float, int
        Tolerance/subdivision-count knobs passed to the underlying
        ``scipy.integrate.quad`` calls, as in :func:`pcarve_ci`. Loosen
        these (e.g. ``epsabs=epsrel=1e-4``) for simulation-scale usage --
        ``"mean"`` in particular costs one such call *per grid point*
        (~120 by default), so tight tolerances make it seconds-per-call
        slow.
    n_points : int, default=121
        Grid resolution used by ``"mean"`` (see :func:`_mean`); irrelevant
        for ``"mle"``. Reducing it (e.g. to 41) trades ``"mean"``/
        ``"combined"`` accuracy for roughly proportionally less runtime.

    Returns
    -------
    theta_hat : float
        The requested point estimate of :math:`\theta^*`.

    Raises
    ------
    ValueError
        If ``a``, ``b``, or ``epsilon`` are out of range, or if
        ``density``, ``input_type``, or ``estimator`` is not recognized.
    """
    if not 0 <= a < b < 1:
        raise ValueError(f"Require 0 <= a < b < 1, got a={a}, b={b}")
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
        t_obs = _p_value_inv(float(stat), theta0, density)
    elif input_type == "statistic":
        t_obs = float(stat)
    else:
        raise ValueError(
            f"input_type must be 'pvalue' or 'statistic', got {input_type!r}"
        )

    quad_kwargs = dict(epsabs=epsabs, epsrel=epsrel, limit=limit)
    if estimator == "mle":
        return _mle(t_obs, theta0, a, b, epsilon, density, quad_kwargs)
    if estimator == "mean":
        return _mean(t_obs, theta0, a, b, epsilon, density, quad_kwargs, n_points)
    return (
        _mean(t_obs, theta0, a, b, epsilon, density, quad_kwargs, n_points)
        + _mle(t_obs, theta0, a, b, epsilon, density, quad_kwargs)
    ) / 2


_TRUNCGAUSS_ESTIMATORS = ("mle", "mean")


def _truncgauss_log_likelihood(theta, t_obs, c, scale):
    return stats.norm.logpdf(t_obs, loc=theta, scale=scale) - stats.norm.logsf(
        c, loc=theta, scale=scale
    )


def _truncgauss_mle(t_obs, c, scale, search_radii=(5, 20, 60, 200)):
    """Maximize the truncated-normal conditional likelihood via bracketed Brent search.

    Unlike ``_mle`` above, the (closed-form) likelihood here costs one
    ``logpdf``/``logsf`` evaluation rather than its own numerical
    integration, so this converges in a fraction of the time.
    """
    objective = lambda theta: -_truncgauss_log_likelihood(theta, t_obs, c, scale)
    result = None
    for radius in search_radii:
        lo, hi = t_obs - radius, t_obs + radius
        result = minimize_scalar(objective, bounds=(lo, hi), method="bounded")
        at_boundary = min(result.x - lo, hi - result.x) < 1e-6 * radius
        if not at_boundary:
            return result.x
    return result.x


def _truncgauss_mean(t_obs, c, scale):
    """Conditional mean via direct adaptive quadrature over theta.

    Unlike ``_mean`` above, the likelihood here is closed-form (no nested
    integration), so ordinary adaptive quadrature is cheap and accurate
    enough directly -- no need for the grid-based workaround.
    """
    likelihood = lambda theta: np.exp(
        _truncgauss_log_likelihood(theta, t_obs, c, scale)
    )
    total, _ = quad(likelihood, -np.inf, np.inf, limit=200)
    numerator, _ = quad(lambda theta: theta * likelihood(theta), -np.inf, np.inf, limit=200)
    return numerator / total


def truncgauss_estimate(
    t_obs: float,
    c: float,
    scale: float = 1.0,
    estimator: str = "mle",
) -> float:
    r"""Point estimate of a normal mean truncated to ``T > c``.

    The classic conditional-selective-inference point estimators (e.g.
    :cite:`lee_exact_2016`, :cite:`ghosh_estimating_2008`), built from the
    same conditional likelihood as :func:`truncgauss_ci`'s
    :math:`R^{TG}_\theta(t)`: the density (in :math:`t`) of :math:`T` at
    the observed value, given :math:`\theta` and conditional on :math:`T >
    c`,

    .. math::

        r^{\mathrm{TG}}_\theta(t) := \frac{g_\theta(t)}{\Pr_\theta(T > c)}
        = \frac{g_\theta(t)}{1 - \Phi((c - \theta)/\text{scale})}.

    Two estimators are available via ``estimator``, matching
    :func:`pcarve_estimate`'s ``"mle"``/``"mean"``:

    - ``"mle"``: :math:`\hat\theta^{\mathrm{MLE}} := \arg\max_\theta
      r^{\mathrm{TG}}_\theta(t)`.
    - ``"mean"``: :math:`\hat\theta^{\mathrm{mean}} := \int_\Theta \theta\,
      r^{\mathrm{TG}}_\theta(t) \, d\theta \big/ \int_\Theta
      r^{\mathrm{TG}}_\theta(t) \, d\theta`.

    Unlike :func:`pcarve_estimate`, no null value :math:`\theta_0` is
    needed (estimation doesn't test a specific hypothesis), and
    :math:`r^{\mathrm{TG}}_\theta` is closed-form rather than requiring its
    own numerical integration per evaluation, so both estimators are exact
    up to root-finding/quadrature tolerance rather than also being subject
    to grid-resolution error.

    Parameters
    ----------
    t_obs : float
        Observed test statistic :math:`T`. Must satisfy ``t_obs >= c``
        (the selection event).
    c : float
        Truncation/selection threshold: inference is conducted only given
        :math:`T > c`.
    scale : float, default=1.0
        Standard deviation of :math:`T`.
    estimator : {"mle", "mean"}, default="mle"
        Which point estimator to return.

    Returns
    -------
    theta_hat : float
        The requested point estimate of :math:`\theta^*`.

    Raises
    ------
    ValueError
        If ``scale <= 0``, ``estimator`` is not recognized, or ``t_obs <
        c``.
    """
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    if estimator not in _TRUNCGAUSS_ESTIMATORS:
        raise ValueError(
            f"estimator must be one of {_TRUNCGAUSS_ESTIMATORS}, got {estimator!r}"
        )
    if t_obs < c:
        raise ValueError(
            f"Observed statistic {t_obs} lies below the selection threshold "
            f"c={c}; the conditional estimate is undefined off the "
            "selection event."
        )

    if estimator == "mle":
        return _truncgauss_mle(t_obs, c, scale)
    return _truncgauss_mean(t_obs, c, scale)
