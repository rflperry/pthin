"""Conditional confidence intervals after selection on a thinned p-value."""

from __future__ import annotations

from typing import Callable, Union

import numpy as np
from scipy import stats
from scipy.integrate import quad
from scipy.optimize import brentq

__all__ = ["conditional_confidence_interval"]

# `object` stands in for scipy's frozen-distribution type, which isn't
# publicly exported under a stable name.
DensityFamily = Union[str, Callable[[float], object]]


def _dist(theta, density: DensityFamily):
    """Resolve ``density`` to a frozen scipy distribution ``g_theta``.

    ``density="normal"`` is the standard normal shifted to location
    ``theta`` (scale fixed at 1). Otherwise ``density`` is treated as a
    callable ``theta -> frozen scipy.stats distribution``, so any family can
    be plugged in as long as it exposes the usual ``.pdf``/``.sf``/``.isf``
    methods; scipy's generic ``rv_continuous`` machinery numerically inverts
    the CDF for families that don't provide a closed-form ``.ppf``/``.isf``.
    """
    if density == "normal":
        return stats.norm(loc=theta, scale=1.0)
    return density(theta)


def _p_value(t, theta0, density):
    """Upper-tailed p-value ``p_theta0(t) = 1 - G_theta0(t)``."""
    return _dist(theta0, density).sf(t)


def _p_value_inv(q, theta0, density):
    """Inverse of ``_p_value``, i.e. ``p_theta0^{-1}(q)``."""
    return _dist(theta0, density).isf(q)


def _nu_density(q, a, b, epsilon):
    """Density of the reference measure ``nu_{a,b}`` from the theorem, at ``q``."""
    q = np.asarray(q, dtype=float)
    indicator = (q >= a ** (1 / epsilon)) & (q <= b ** (1 / epsilon))
    with np.errstate(divide="ignore", invalid="ignore"):
        q_pow = np.where(q > 0, q ** (-epsilon), np.inf)
        tail = np.minimum(1.0, b * q_pow) - np.minimum(1.0, a * q_pow)
    return epsilon / (a - b) * indicator + (1 - epsilon) / (a - b) * tail


def _split_integrate(integrand, lo, hi, breakpoints, **quad_kwargs):
    """``quad`` from ``lo`` to ``hi``, pre-splitting at any interior breakpoints.

    ``nu_density`` is only piecewise smooth, with kinks at the two
    breakpoints below. Handing an unbounded interval containing those kinks
    to a single adaptive-quadrature call forces it to rediscover the
    non-smoothness by trial and error, which is drastically slower than
    integrating each smooth piece separately.
    """
    points = sorted(p for p in breakpoints if lo < p < hi)
    bounds = [lo, *points, hi]
    total = 0.0
    for left, right in zip(bounds[:-1], bounds[1:]):
        value, _ = quad(integrand, left, right, **quad_kwargs)
        total += value
    return total


def _r_theta(theta, t_obs, theta0, a, b, epsilon, density):
    """Evaluate ``R_theta(t)`` (the conditional CDF of the theorem) at ``theta``.

    Integrated over the test-statistic scale ``z`` rather than the p-value
    scale ``q``: substituting ``q = p_theta0(z)`` and ``dq =
    -g_theta0(z) dz`` turns ``f_theta(q) dq`` into ``g_theta(z) dz`` directly
    (the ``g_theta0`` normalizer cancels the likelihood-ratio's denominator),
    which avoids evaluating near the ``q in {0, 1}`` boundary where the
    p-value-scale integrand is numerically unstable. This substitution holds
    for any family, not just location families.

    ``nu_density(p_theta0(z))`` is zero for ``z`` above ``z_qa`` (the ``q <
    a**(1/epsilon)`` region) and has a kink at ``z_qb`` (where ``q =
    b**(1/epsilon)``), so both integrals are truncated/split there.
    """
    z_qa = _p_value_inv(a ** (1 / epsilon), theta0, density)
    z_qb = _p_value_inv(b ** (1 / epsilon), theta0, density)
    g_theta = _dist(theta, density)

    def integrand(z):
        q = _p_value(z, theta0, density)
        return g_theta.pdf(z) * _nu_density(q, a, b, epsilon)

    quad_kwargs = dict(epsabs=1e-12, epsrel=1e-9, limit=100)
    numerator = _split_integrate(integrand, t_obs, z_qa, [z_qb], **quad_kwargs)
    denominator = _split_integrate(integrand, -np.inf, z_qa, [z_qb], **quad_kwargs)
    return numerator / denominator


def _find_root(f, x0, target, search_radii=(5, 20, 60, 200)):
    """Find ``x`` with ``f(x) = target``, expanding a bracket around ``x0``."""
    shifted = lambda x: f(x) - target
    for radius in search_radii:
        lo, hi = x0 - radius, x0 + radius
        f_lo, f_hi = shifted(lo), shifted(hi)
        if (
            not np.isnan(f_lo)
            and not np.isnan(f_hi)
            and np.sign(f_lo) != np.sign(f_hi)
        ):
            return brentq(shifted, lo, hi, xtol=1e-8, rtol=1e-8)
    raise RuntimeError(
        f"Could not bracket a root for target={target} starting from x0={x0}."
    )


def conditional_confidence_interval(
    stat: float,
    theta0: float,
    a: float,
    b: float,
    *,
    epsilon: float = 0.5,
    alpha: float = 0.05,
    density: DensityFamily = "normal",
    input_type: str = "pvalue",
) -> tuple[float, float]:
    r"""Conditional confidence interval for a location parameter after selection.

    Implements the confidence set of Theorem (conditional confidence
    interval under p-value masking): given a test statistic :math:`T \sim
    g_{\theta^*}`, its upper-tailed p-value :math:`p_{\theta_0}(T) = 1 -
    G_{\theta_0}(T)` against a null :math:`\theta_0`, and the decision to
    conduct inference only on the event :math:`p_{\theta_0}(T) \in [a, b]`,
    this returns an interval :math:`\mathrm{CI}^\alpha(T)` satisfying

    .. math::

        \Pr(\theta^* \in \mathrm{CI}^\alpha(T) \mid p_{\theta_0}(T) \in [a, b])
        = 1 - \alpha.

    The interval is :math:`\{\theta : R_\theta(T) \in [\alpha/2, 1-\alpha/2]\}`,
    where :math:`R_\theta(t) = \int_0^{p_{\theta_0}(t)} f_\theta \, d\nu_{a,b}
    \big/ \int_0^1 f_\theta \, d\nu_{a,b}`, :math:`f_\theta(q) =
    g_\theta(p_{\theta_0}^{-1}(q)) / g_{\theta_0}(p_{\theta_0}^{-1}(q))` is a
    likelihood-ratio reweighting, and :math:`\nu_{a,b}` is the reference
    measure induced by the p-value thinning construction (see
    :func:`pthin.randomize.pthin`) with thinning fraction ``epsilon``.
    :math:`R_\theta` is found by numerical integration and the interval
    endpoints by root-finding in :math:`\theta`, so this is exact only up to
    numerical-integration and root-finding tolerance.

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
    alpha : float, default=0.05
        Target miscoverage level; the returned interval targets conditional
        coverage ``1 - alpha``.
    density : "normal" or callable, default="normal"
        The family :math:`\{g_\theta\}`. Either the string ``"normal"``,
        for the standard normal family :math:`g_\theta = N(\theta, 1)`, or a
        callable ``theta -> frozen scipy.stats distribution`` giving
        :math:`g_\theta` for an arbitrary family, e.g. ``lambda theta:
        scipy.stats.gamma(a=2, loc=theta)``. Any such distribution is
        assumed to be numerically invertible, i.e. to support ``.isf``
        (scipy provides this generically via root-finding for custom
        ``rv_continuous`` subclasses even without a closed-form ``.ppf``).
    input_type : {"pvalue", "statistic"}, default="pvalue"
        Whether ``stat`` is the p-value :math:`p_{\theta_0}(T)` or the raw
        statistic :math:`T`.

    Returns
    -------
    ci_lower : float
        Lower endpoint of :math:`\mathrm{CI}^\alpha(T)`.
    ci_upper : float
        Upper endpoint of :math:`\mathrm{CI}^\alpha(T)`.

    Raises
    ------
    ValueError
        If ``a``, ``b``, ``epsilon``, or ``alpha`` are out of range, if
        ``input_type`` is not recognized, or if the observed p-value falls
        outside ``[a, b]`` (the conditional interval is only defined on the
        selection event).
    """
    if not 0 < a < b < 1:
        raise ValueError(f"Require 0 < a < b < 1, got a={a}, b={b}")
    if not 0 < epsilon < 1:
        raise ValueError(f"epsilon must lie in (0, 1), got {epsilon}")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    if density != "normal" and not callable(density):
        raise ValueError(
            "density must be the string 'normal' or a callable "
            f"theta -> frozen scipy.stats distribution, got {density!r}"
        )

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
            f"[{a}, {b}]; the conditional interval is undefined off the "
            "selection event."
        )

    r_of_theta = lambda theta: _r_theta(theta, t_obs, theta0, a, b, epsilon, density)

    theta_lo = _find_root(r_of_theta, t_obs, alpha / 2)
    theta_hi = _find_root(r_of_theta, t_obs, 1 - alpha / 2)

    return min(theta_lo, theta_hi), max(theta_lo, theta_hi)
