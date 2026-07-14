"""Conditional confidence intervals after selection on a thinned p-value."""

from __future__ import annotations

from typing import Callable, Union

import numpy as np
from scipy import stats
from scipy.integrate import quad
from scipy.optimize import brentq

__all__ = ["pcarve_ci", "pcarve_threshold", "truncgauss_pvalue", "truncgauss_ci"]

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


def _nu_cdf(t, a, b, epsilon):
    r"""Closed form of ``N(t) = int_0^t d nu_{a,b}(q)``, the reference CDF.

    This uses a ``1/(b - a)`` normalization rather than the ``1/(a - b)`` in
    ``_nu_density``. In ``_r_theta``, ``_nu_density`` appears in both the
    numerator and denominator of a ratio, so its sign is irrelevant there.
    Used bare, as it is for the hypothesis-testing bound below, the ``(a -
    b)`` convention (with ``a < b``, matching the ``p_1 in [a, b]``
    selection interval elsewhere in this module) gives a *negative*,
    decreasing function of ``t`` -- which cannot bound a probability, since
    ``Pr(p <= t | ...) >= 0`` for every ``t``. The ``(b - a)`` convention
    used here instead makes ``N`` non-negative, non-decreasing, and
    satisfies ``N(1) = 1`` for every ``0 < a < b < 1`` and ``0 < epsilon <
    1``, i.e. a genuine CDF on ``[0, 1]``.

    Obtained by direct antiderivatives of ``_nu_density``'s three pieces
    (zero below ``a**(1/epsilon)``, a linear-minus-power piece up to
    ``b**(1/epsilon)``, then a pure power-law tail), verified against
    numerical integration of ``-_nu_density`` to ~1e-12.
    """
    q_a = a ** (1 / epsilon)
    q_b = b ** (1 / epsilon)
    if t <= q_a:
        return 0.0
    if t <= q_b:
        return (t - a * t ** (1 - epsilon)) / (b - a)
    n_qb = (q_b - a * q_b ** (1 - epsilon)) / (b - a)
    return n_qb + (t ** (1 - epsilon) - q_b ** (1 - epsilon))


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


def _integrand_and_breakpoints(theta, theta0, a, b, epsilon, density):
    """Shared building block of ``R_theta`` and the conditional likelihood.

    Returns the integrand ``g_theta(z) * nu_density(p_theta0(z))`` together
    with the two ``z``-breakpoints where ``nu_density(p_theta0(z))`` is zero
    (above ``z_qa``) or has a kink (at ``z_qb``); see ``_r_theta``.

    Both ``g_theta`` and ``g_theta0`` are frozen once, outside the
    integrand: re-freezing a scipy distribution (as ``_p_value`` does)
    rebuilds its docstring every call, which dominates runtime by ~2 orders
    of magnitude when the integrand is evaluated thousands of times across
    the nested integrals ``pthin.estimate`` needs for the mean/median.
    """
    g_theta0 = _dist(theta0, density)
    z_qa = g_theta0.isf(a ** (1 / epsilon))
    z_qb = g_theta0.isf(b ** (1 / epsilon))
    g_theta = _dist(theta, density)

    def integrand(z):
        q = g_theta0.sf(z)
        return g_theta.pdf(z) * _nu_density(q, a, b, epsilon)

    return integrand, z_qa, z_qb


_QUAD_KWARGS = dict(epsabs=1e-12, epsrel=1e-9, limit=100)


def _a0_tail_weight(q, epsilon, b):
    r"""``a=0`` special case of ``nu_density`` for ``q > b**(1/epsilon)``.

    When ``a=0``, ``nu_density(q, 0, b, epsilon)`` collapses to a constant
    (``-1/b``, up to an overall scale -- see ``_r_theta_a0``) for ``q <=
    b**(1/epsilon)`` and to ``-(1-epsilon) * q**(-epsilon)`` above it. This
    is that second (tail) piece, rescaled by ``-b`` (a constant that cancels
    in the ``R_theta``/``_conditional_likelihood`` ratio, matching the
    convention used in ``experiments/normal_ci.ipynb``'s ``calc_r_mu``,
    which this fast path reproduces): ``(1-epsilon) * b * q**(-epsilon)``.

    ``epsilon=0.5`` is special-cased to ``1/sqrt(q)`` (a faster op than a
    general non-half-integer ``**`` power) since it is the common case in
    practice (:func:`pthin.randomize.pthin`'s default thinning fraction).
    """
    if epsilon == 0.5:
        return 0.5 * b / np.sqrt(q)
    return (1 - epsilon) * b * q ** (-epsilon)


def _r_theta_a0(theta, t_obs, theta0, b, epsilon, density, quad_kwargs):
    r"""``a=0`` fast path shared by ``_denominator`` and ``_r_theta``.

    Mirrors ``experiments/normal_ci.ipynb``'s ``calc_r_mu`` (the ``a=0``,
    ``epsilon=0.5`` closed-form special case this generalizes to arbitrary
    ``epsilon``): with ``a=0``, the reference measure's density is constant
    for ``q <= b**(1/epsilon)`` (i.e. ``z >= z_qb``) and a pure power law
    above it, so the constant-density piece integrates in closed form
    (``g_theta.sf``) and only the power-law piece needs ``quad`` -- instead
    of the general ``0 < a`` case's up to two ``quad`` calls each for
    numerator and denominator (see ``_integrand_and_breakpoints``).

    Returns ``(numerator, denominator)``, i.e. ``R_theta(t_obs) =
    numerator / denominator``.
    """
    g_theta0 = _dist(theta0, density)
    z_qb = g_theta0.isf(b ** (1 / epsilon))
    g_theta = _dist(theta, density)

    def tail_integrand(z):
        q = g_theta0.sf(z)
        return g_theta.pdf(z) * _a0_tail_weight(q, epsilon, b)

    def integral_from(lo):
        constant_region = g_theta.sf(z_qb)
        if lo >= z_qb:
            return g_theta.sf(lo)
        tail, _ = quad(tail_integrand, lo, z_qb, **quad_kwargs)
        return tail + constant_region

    return integral_from(t_obs), integral_from(-np.inf)


def _denominator(theta, theta0, a, b, epsilon, density, quad_kwargs=None):
    """``int_{-inf}^{inf} g_theta(z) * nu_density(p_theta0(z)) dz``.

    The (theta-dependent) normalizing constant shared by ``_r_theta`` and
    ``_conditional_likelihood``.
    """
    quad_kwargs = _QUAD_KWARGS if quad_kwargs is None else quad_kwargs
    if a == 0:
        _, denominator = _r_theta_a0(
            theta, -np.inf, theta0, b, epsilon, density, quad_kwargs
        )
        return denominator
    integrand, z_qa, z_qb = _integrand_and_breakpoints(
        theta, theta0, a, b, epsilon, density
    )
    return _split_integrate(integrand, -np.inf, z_qa, [z_qb], **quad_kwargs)


def _r_theta(theta, t_obs, theta0, a, b, epsilon, density, quad_kwargs=None):
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
    b**(1/epsilon)``), so both integrals are truncated/split there. When
    ``a=0`` (``z_qa = +inf``), :func:`_r_theta_a0` computes the same ratio
    much faster -- see its docstring.
    """
    quad_kwargs = _QUAD_KWARGS if quad_kwargs is None else quad_kwargs
    if a == 0:
        numerator, denominator = _r_theta_a0(
            theta, t_obs, theta0, b, epsilon, density, quad_kwargs
        )
        return numerator / denominator
    integrand, z_qa, z_qb = _integrand_and_breakpoints(
        theta, theta0, a, b, epsilon, density
    )
    numerator = _split_integrate(integrand, t_obs, z_qa, [z_qb], **quad_kwargs)
    denominator = _split_integrate(integrand, -np.inf, z_qa, [z_qb], **quad_kwargs)
    return numerator / denominator


def _conditional_likelihood(theta, t_obs, theta0, a, b, epsilon, density):
    r"""Conditional likelihood ``r_theta(p_theta0(t))`` at the observed ``t_obs``.

    This is the density (in ``t``) of ``T`` at the observed value, given
    ``theta`` and conditional on the selection event ``p_theta0(T) in [a,
    b]`` -- equivalently ``-d/dt R_theta(t)`` evaluated at ``t = t_obs``
    (verified by finite differences against ``_r_theta``). As a function of
    ``theta`` for fixed ``t_obs``, this is the conditional likelihood used
    by the point estimators in :mod:`pthin.estimate`.
    """
    q_obs = _p_value(t_obs, theta0, density)
    g_theta_t = _dist(theta, density).pdf(t_obs)
    w_t = _nu_density(q_obs, a, b, epsilon)
    return g_theta_t * w_t / _denominator(theta, theta0, a, b, epsilon, density)


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


def pcarve_ci(
    stat: float,
    theta0: float,
    a: float,
    b: float,
    epsilon: float = 0.5,
    alpha: float = 0.05,
    density: DensityFamily = "normal",
    input_type: str = "pvalue",
    epsabs: float = 1e-12,
    epsrel: float = 1e-9,
    limit: int = 100,
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
        actually arranged (e.g. by selecting on :math:`p_1 \le b`); it is
        not, and cannot be, checked from ``stat`` alone, since :math:`p_1`
        is an independent random quantity not derivable from :math:`T` or
        its raw p-value. ``a=0`` (e.g. an "arg min :math:`p_1`" selection
        rule, where ``b`` is the runner-up's :math:`p_1`) uses a much
        faster code path than ``a > 0`` -- see :func:`_r_theta_a0`.
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
    epsabs, epsrel, limit : float, float, int
        Tolerance/subdivision-count knobs passed to the underlying
        ``scipy.integrate.quad`` calls. Defaults are tight enough for
        general use; loosen them (e.g. ``epsabs=epsrel=1e-4``) for
        simulation-scale usage where many calls are made and the added
        numerical-integration error is acceptable relative to Monte Carlo
        noise -- see ``experiments/normal_ci.ipynb``.

    Returns
    -------
    ci_lower : float
        Lower endpoint of :math:`\mathrm{CI}^\alpha(T)`.
    ci_upper : float
        Upper endpoint of :math:`\mathrm{CI}^\alpha(T)`.

    Raises
    ------
    ValueError
        If ``a``, ``b``, ``epsilon``, or ``alpha`` are out of range, or if
        ``input_type`` is not recognized.
    """
    if not 0 <= a < b < 1:
        raise ValueError(f"Require 0 <= a < b < 1, got a={a}, b={b}")
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
        t_obs = _p_value_inv(float(stat), theta0, density)
    elif input_type == "statistic":
        t_obs = float(stat)
    else:
        raise ValueError(
            f"input_type must be 'pvalue' or 'statistic', got {input_type!r}"
        )

    quad_kwargs = dict(epsabs=epsabs, epsrel=epsrel, limit=limit)
    r_of_theta = lambda theta: _r_theta(
        theta, t_obs, theta0, a, b, epsilon, density, quad_kwargs
    )

    theta_lo = _find_root(r_of_theta, t_obs, alpha / 2)
    theta_hi = _find_root(r_of_theta, t_obs, 1 - alpha / 2)

    return min(theta_lo, theta_hi), max(theta_lo, theta_hi)


def pcarve_threshold(
    alpha: float,
    a: float,
    b: float,
    epsilon: float = 0.5,
) -> float:
    r"""Rejection threshold for a p-value after selection on its thinned mask.

    Implements the Lemma bounding the conditional false-positive rate of
    testing the original p-value :math:`p` after selecting on the thinned
    p-value :math:`p_1` (see :func:`pthin.randomize.pthin`) landing in
    :math:`[a, b]`: for any :math:`t \in (0, 1)`,

    .. math::

        \Pr(p \le t \mid p_1 \in [a, b]) \le \int_0^t d\nu_{a,b}(q).

    This function returns that
    :math:`t^\star(\alpha)`: rejecting whenever :math:`p \le t^\star(\alpha)`
    controls the conditional false-positive rate at level :math:`\alpha`,

    .. math::

        \Pr(p \le t^\star(\alpha) \mid p_1 \in [a, b]) \le \alpha.

    Parameters
    ----------
    alpha : float
        Target conditional false-positive rate. Must lie in ``(0, 1)``.
    a, b : float
        Endpoints of the selection interval: inference is conducted only
        given :math:`p_1 \in [a, b]`. Must satisfy ``0 < a < b < 1``.
    epsilon : float, default=0.5
        Thinning fraction used to construct :math:`p_1`, matching the
        ``epsilon`` of :func:`pthin.randomize.pthin`. Must lie in ``(0, 1)``.

    Returns
    -------
    threshold : float
        The rejection threshold :math:`t^\star(\alpha) \in (0, 1)`.

    Raises
    ------
    ValueError
        If ``alpha``, ``a``, ``b``, or ``epsilon`` are out of range.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    if not 0 < a < b < 1:
        raise ValueError(f"Require 0 < a < b < 1, got a={a}, b={b}")
    if not 0 < epsilon < 1:
        raise ValueError(f"epsilon must lie in (0, 1), got {epsilon}")

    return brentq(
        lambda t: _nu_cdf(t, a, b, epsilon) - alpha,
        0.0,
        1.0,
        xtol=1e-14,
        rtol=1e-14,
    )


def _truncgauss_survival(theta, t, c, scale):
    r"""``R^{TG}_theta(t) := Pr_theta(T >= t | T > c)`` for ``T ~ N(theta, scale**2)``.

    A closed form (no numerical integration): the survival function of a
    normal truncated to ``{T > c}``, evaluated in log-space for stability
    (``sf(t - theta)`` and ``sf(c - theta)`` can each individually underflow
    to 0 while their ratio stays well within range).
    """
    log_num = stats.norm.logsf(t, loc=theta, scale=scale)
    log_den = stats.norm.logsf(c, loc=theta, scale=scale)
    return np.exp(log_num - log_den)


def truncgauss_pvalue(t_obs: float, theta0: float, c: float, scale: float = 1.0) -> float:
    r"""Exact conditional p-value for a normal mean truncated to ``T > c``.

    The classic conditional-selective-inference construction (e.g.
    :cite:`lee_exact_2016`): given :math:`T \sim N(\theta_0, \text{scale}^2)`
    under the null and conditioning on the selection event :math:`T > c`,

    .. math::

        p^{\mathrm{TG}} := \Pr_{\theta_0}(T \ge t \mid T > c)
        = \frac{1 - \Phi((t - \theta_0)/\text{scale})}
               {1 - \Phi((c - \theta_0)/\text{scale})}

    is exactly (not merely boundedly) uniform on ``(0, 1)`` under the null,
    conditional on ``T > c``, so rejecting :math:`H_0: \theta = \theta_0`
    when :math:`p^{\mathrm{TG}} \le \alpha` controls the conditional
    false-positive rate at exactly :math:`\alpha`. Contrast
    :func:`pcarve_threshold`, whose bound is only exact when the selection
    variable and the tested statistic coincide, as they do here.

    Parameters
    ----------
    t_obs : float
        Observed test statistic :math:`T`. Must satisfy ``t_obs >= c``
        (the selection event).
    theta0 : float
        Null value :math:`\theta_0`.
    c : float
        Truncation/selection threshold: inference is conducted only given
        :math:`T > c`.
    scale : float, default=1.0
        Standard deviation of :math:`T`.

    Returns
    -------
    p_value : float
        The exact conditional p-value :math:`p^{\mathrm{TG}} \in (0, 1)`.

    Raises
    ------
    ValueError
        If ``scale <= 0`` or ``t_obs < c``.
    """
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    if t_obs < c:
        raise ValueError(
            f"Observed statistic {t_obs} lies below the selection threshold "
            f"c={c}; the conditional p-value is undefined off the selection "
            "event."
        )
    return _truncgauss_survival(theta0, t_obs, c, scale)


def truncgauss_ci(
    t_obs: float,
    c: float,
    alpha: float = 0.05,
    scale: float = 1.0,
) -> tuple[float, float]:
    r"""Confidence interval for a normal mean truncated to ``T > c``.

    The classic conditional-selective-inference CI (e.g.
    :cite:`lee_exact_2016`): inverting :func:`truncgauss_pvalue`-style
    tails at :math:`\alpha/2` and :math:`1 - \alpha/2` gives an interval
    :math:`\mathrm{CI}^\alpha(T)` with *exact* conditional coverage,

    .. math::

        \Pr(\theta^* \in \mathrm{CI}^\alpha(T) \mid T > c) = 1 - \alpha,

    since :math:`R^{TG}_\theta(t) := \Pr_\theta(T \ge t \mid T > c)` is
    available in closed form (see :func:`_truncgauss_survival`) and,
    exactly as :math:`R_\theta` in :func:`pcarve_ci`, is monotonically
    increasing in :math:`\theta`; unlike :func:`pcarve_ci`, no numerical
    integration is needed here, only the root-finding. Unlike
    :func:`pcarve_ci`/:func:`truncgauss_pvalue`, no null value
    :math:`\theta_0` is needed: a confidence interval doesn't test a
    specific hypothesis, so there's nothing to condition the p-value
    calculation on.

    Parameters
    ----------
    t_obs : float
        Observed test statistic :math:`T`. Must satisfy ``t_obs >= c``
        (the selection event).
    c : float
        Truncation/selection threshold: inference is conducted only given
        :math:`T > c`.
    alpha : float, default=0.05
        Target miscoverage level; the returned interval targets coverage
        ``1 - alpha``.
    scale : float, default=1.0
        Standard deviation of :math:`T`.

    Returns
    -------
    ci_lower : float
        Lower endpoint of :math:`\mathrm{CI}^\alpha(T)`.
    ci_upper : float
        Upper endpoint of :math:`\mathrm{CI}^\alpha(T)`.

    Raises
    ------
    ValueError
        If ``scale <= 0``, ``alpha`` is out of ``(0, 1)``, or ``t_obs < c``.
    """
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    if t_obs < c:
        raise ValueError(
            f"Observed statistic {t_obs} lies below the selection threshold "
            f"c={c}; the conditional interval is undefined off the "
            "selection event."
        )

    r_of_theta = lambda theta: _truncgauss_survival(theta, t_obs, c, scale)

    theta_lo = _find_root(r_of_theta, t_obs, alpha / 2)
    theta_hi = _find_root(r_of_theta, t_obs, 1 - alpha / 2)

    return min(theta_lo, theta_hi), max(theta_lo, theta_hi)
