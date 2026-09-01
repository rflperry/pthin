"""Point estimation of a location parameter after selection on a thinned p-value."""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.integrate import quad, trapezoid
from scipy.optimize import minimize_scalar

from pthin.inference import DensityFamily, _conditional_likelihood, _p_value_inv

__all__ = ["pcarve_estimate", "truncgauss_estimate", "normal_carving_estimate"]

_ESTIMATORS = ("mle", "mean", "combined")


def _log_likelihood(theta, t_obs, theta0, a, b, epsilon, density, quad_kwargs):
    likelihood = _conditional_likelihood(
        theta, t_obs, theta0, a, b, epsilon, density, quad_kwargs
    )
    return -np.inf if likelihood <= 0 else np.log(likelihood)


def _mle(
    t_obs, theta0, a, b, epsilon, density, quad_kwargs,
    search_radii=(5, 20, 60, 200), theta_min=-np.inf, theta_max=np.inf,
):
    """Maximize the conditional likelihood over theta via bracketed Brent search.

    ``theta_min``/``theta_max`` constrain the search (e.g. ``theta_min=0``
    for a magnitude parameter under a reflection-symmetric family, where
    ``theta`` is otherwise only identifiable up to sign). A result sitting
    at one of these hard domain bounds is accepted immediately rather than
    triggering a wider-bracket retry, since expanding the search radius
    can't move it off a genuine constraint boundary.
    """
    objective = lambda theta: -_log_likelihood(
        theta, t_obs, theta0, a, b, epsilon, density, quad_kwargs
    )
    result = None
    for radius in search_radii:
        lo = max(theta_min, t_obs - radius)
        hi = min(theta_max, t_obs + radius)
        result = minimize_scalar(objective, bounds=(lo, hi), method="bounded")
        stuck_at_lo = abs(result.x - lo) < 1e-6 * radius and lo > theta_min
        stuck_at_hi = abs(result.x - hi) < 1e-6 * radius and hi < theta_max
        if not (stuck_at_lo or stuck_at_hi):
            return result.x
    return result.x


def _mean(
    t_obs, theta0, a, b, epsilon, density, quad_kwargs, n_points,
    theta_min=-np.inf, theta_max=np.inf,
):
    """Conditional mean via trapezoidal quadrature over a grid in theta.

    Each grid point costs its own numerical integration (see
    ``_conditional_likelihood``), so this evaluates the likelihood on a
    single shared grid spanning most of its mass rather than driving two
    independent adaptive-quadrature calls (numerator and normalizing
    constant) that would each rediscover where that mass lies from scratch.

    ``theta_min``/``theta_max`` constrain the grid to a bounded parameter
    domain -- see ``_mle``.
    """
    likelihood_at = lambda theta: _conditional_likelihood(
        theta, t_obs, theta0, a, b, epsilon, density, quad_kwargs
    )
    center = min(max(t_obs, theta_min), theta_max)
    peak = likelihood_at(center)
    tiny = 1e-6 * max(peak, 1e-300)
    radius = 4.0
    while radius <= 1e4:
        lo = max(theta_min, center - radius)
        hi = min(theta_max, center + radius)
        # A side pinned at a hard theta_min/theta_max boundary is "done"
        # regardless of the density there (nowhere further to expand into);
        # only an *unclamped* side still needs to show density decay before
        # the grid is wide enough to capture (near enough) all the mass.
        lo_done = lo <= theta_min or likelihood_at(lo) < tiny
        hi_done = hi >= theta_max or likelihood_at(hi) < tiny
        if lo_done and hi_done:
            break
        radius *= 2

    lo = max(theta_min, center - radius)
    hi = min(theta_max, center + radius)
    thetas = np.linspace(lo, hi, n_points)
    likelihoods = np.array([likelihood_at(theta) for theta in thetas])
    total = trapezoid(likelihoods, thetas)
    return trapezoid(thetas * likelihoods, thetas) / total


def pcarve_estimate(
    stat: float,
    *,
    theta0: float = 0.0,
    a: float = 0.0,
    b: float,
    epsilon: float = 0.5,
    density: DensityFamily = "normal",
    input_type: str = "pvalue",
    estimator: str = "mle",
    epsabs: float = 1e-12,
    epsrel: float = 1e-9,
    limit: int = 100,
    n_points: int = 121,
    theta_min: float = -np.inf,
    theta_max: float = np.inf,
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

    All parameters other than ``stat`` are keyword-only.

    Parameters
    ----------
    stat : float
        Either the raw p-value :math:`p_{\theta_0}(T)` of the tested
        statistic :math:`T` (default, when ``input_type="pvalue"``) or
        :math:`T` itself (when ``input_type="statistic"``) -- this is *not*
        the thinned p-value :math:`p_1(T)` used for selection below, which
        this function never sees directly.
    theta0 : float, default=0.0
        Null value :math:`\theta_0` defining the upper-tailed p-value
        :math:`p_{\theta_0}(t) = 1 - G_{\theta_0}(t)`.
    a : float, default=0.0
    b : float
        Endpoints of the selection interval: inference is conducted only
        given :math:`p_1(T) \in [a, b]`, where :math:`p_1` is the thinned
        p-value from :func:`pthin.randomize.pthin`. Must satisfy ``0 <= a <
        b <= 1``. ``b`` has no default.
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
    theta_min, theta_max : float, default=-inf, inf
        Bounds constraining the search/integration domain for ``theta``.
        Useful when ``theta`` is only identifiable up to some symmetry of
        ``density`` (e.g. a reflection-symmetric family like
        ``scipy.stats.foldnorm``, where :math:`g_\theta = g_{-\theta}`) and
        you want a magnitude estimate specifically: without a bound,
        ``"mean"`` would integrate a ``theta``-symmetric likelihood over a
        ``theta``-symmetric domain and collapse toward 0, and ``"mle"``
        would return an arbitrary sign.

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
    if not 0 <= a < b <= 1:
        raise ValueError(f"Require 0 <= a < b <= 1, got a={a}, b={b}")
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
    mle_kwargs = dict(theta_min=theta_min, theta_max=theta_max)
    mean_kwargs = dict(theta_min=theta_min, theta_max=theta_max)
    if estimator == "mle":
        return _mle(t_obs, theta0, a, b, epsilon, density, quad_kwargs, **mle_kwargs)
    if estimator == "mean":
        return _mean(
            t_obs, theta0, a, b, epsilon, density, quad_kwargs, n_points, **mean_kwargs
        )
    return (
        _mean(t_obs, theta0, a, b, epsilon, density, quad_kwargs, n_points, **mean_kwargs)
        + _mle(t_obs, theta0, a, b, epsilon, density, quad_kwargs, **mle_kwargs)
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

    The classic conditional-selective-inference point estimators (e.g.,
    :cite:`ghosh_estimating_2008`), built from the
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


_NORMAL_CARVING_ESTIMATORS = ("mle", "mean")


def _normal_carving_log_likelihood(mu, x_obs, c, sigma_x, sigma_y, rho):
    r"""``log r^{NC}_mu(x_obs) := log Pr_mu(X = x_obs, Y > c) - log Pr_mu(Y > c)``.

    ``Y | X = x_obs ~ N(mu + rho*(sigma_y/sigma_x)*(x_obs - mu), sigma_y**2*(1
    - rho**2))`` (the usual bivariate-normal conditional), so the numerator
    factors as ``g_X(x_obs) * Pr(Y > c | X = x_obs)``. Verified by finite
    differences against ``-d/dx Pr_mu(X >= x | Y > c)``
    (:func:`pthin.inference._normal_carving_survival`) at ``x = x_obs``, to
    ~1e-12.
    """
    cond_mean = mu + rho * (sigma_y / sigma_x) * (x_obs - mu)
    cond_std = sigma_y * np.sqrt(1 - rho**2)
    log_num = stats.norm.logpdf(x_obs, loc=mu, scale=sigma_x) + stats.norm.logsf(
        c, loc=cond_mean, scale=cond_std
    )
    log_den = stats.norm.logsf(c, loc=mu, scale=sigma_y)
    return log_num - log_den


def _normal_carving_mle(x_obs, c, sigma_x, sigma_y, rho, search_radii=(5, 20, 60, 200)):
    """Maximize the data-carving conditional likelihood via bracketed Brent search.

    Mirrors ``_truncgauss_mle``: the (closed-form) likelihood here costs one
    ``logpdf``/``logsf`` pair rather than its own numerical integration.
    """
    objective = lambda mu: -_normal_carving_log_likelihood(
        mu, x_obs, c, sigma_x, sigma_y, rho
    )
    result = None
    for radius in search_radii:
        lo, hi = x_obs - radius, x_obs + radius
        result = minimize_scalar(objective, bounds=(lo, hi), method="bounded")
        at_boundary = min(result.x - lo, hi - result.x) < 1e-6 * radius
        if not at_boundary:
            return result.x
    return result.x


def _normal_carving_mean(x_obs, c, sigma_x, sigma_y, rho):
    """Conditional mean via direct adaptive quadrature over mu.

    Mirrors ``_truncgauss_mean``: the likelihood here is closed-form (no
    nested integration), so ordinary adaptive quadrature is cheap and
    accurate enough directly.
    """
    likelihood = lambda mu: np.exp(
        _normal_carving_log_likelihood(mu, x_obs, c, sigma_x, sigma_y, rho)
    )
    total, _ = quad(likelihood, -np.inf, np.inf, limit=200)
    numerator, _ = quad(lambda mu: mu * likelihood(mu), -np.inf, np.inf, limit=200)
    return numerator / total


def normal_carving_estimate(
    x_obs: float,
    c: float,
    sigma_x: float = 1.0,
    sigma_y: float = 1.0,
    rho: float = 0.0,
    estimator: str = "mle",
) -> float:
    r"""Point estimate of a shared mean after data carving with bivariate normal data.

    Given :math:`(X, Y) \sim N\left(\binom{\mu}{\mu}, \begin{psmallmatrix}
    \sigma_X^2 & \rho\sigma_X\sigma_Y \\ \rho\sigma_X\sigma_Y & \sigma_Y^2
    \end{psmallmatrix}\right)` and the decision to conduct inference only on
    the selection event :math:`Y > c`, the conditional likelihood of
    :math:`\mu` given the observed :math:`X = x_{\mathrm{obs}}` is

    .. math::

        r^{\mathrm{NC}}_\mu(x_{\mathrm{obs}})
        := \frac{g_\mu(x_{\mathrm{obs}}) \cdot
        \Pr_\mu(Y > c \mid X = x_{\mathrm{obs}})}{\Pr_\mu(Y > c)},

    matching :func:`normal_carving_pvalue`/:func:`normal_carving_ci`'s
    :math:`R^{\mathrm{NC}}_\mu(x) := \Pr_\mu(X \ge x \mid Y > c)` via
    :math:`r^{\mathrm{NC}}_\mu(x) = -\frac{d}{dx} R^{\mathrm{NC}}_\mu(x)`
    (verified numerically). Two estimators are available via ``estimator``,
    matching :func:`truncgauss_estimate`:

    - ``"mle"``: :math:`\hat\mu^{\mathrm{MLE}} := \arg\max_\mu
      r^{\mathrm{NC}}_\mu(x_{\mathrm{obs}})`.
    - ``"mean"``: :math:`\hat\mu^{\mathrm{mean}} := \int \mu\,
      r^{\mathrm{NC}}_\mu(x_{\mathrm{obs}}) \, d\mu \big/ \int
      r^{\mathrm{NC}}_\mu(x_{\mathrm{obs}}) \, d\mu`.

    As in :func:`truncgauss_estimate`, :math:`r^{\mathrm{NC}}_\mu` is
    closed-form (the conditional density of a bivariate normal), so both
    estimators are exact up to root-finding/quadrature tolerance, with no
    grid-resolution error. Setting :math:`\rho = 0` recovers the ordinary
    (unconditional) sample-mean-style MLE :math:`\hat\mu = x_{\mathrm{obs}}`
    (selection carries no information about :math:`X`); :math:`\rho \to 1`
    with :math:`\sigma_X = \sigma_Y` recovers :func:`truncgauss_estimate`.

    Parameters
    ----------
    x_obs : float
        Observed value of :math:`X`.
    c : float
        Selection threshold: inference is conducted only given :math:`Y >
        c`.
    sigma_x, sigma_y : float, default=1.0
        Standard deviations of :math:`X` and :math:`Y`.
    rho : float, default=0.0
        Correlation between :math:`X` and :math:`Y`. Must lie in ``(-1,
        1)`` (a non-degenerate bivariate normal).
    estimator : {"mle", "mean"}, default="mle"
        Which point estimator to return.

    Returns
    -------
    mu_hat : float
        The requested point estimate of :math:`\mu^*`.

    Raises
    ------
    ValueError
        If ``sigma_x``/``sigma_y`` is not positive, ``rho`` is not in
        ``(-1, 1)``, or ``estimator`` is not recognized.
    """
    if sigma_x <= 0 or sigma_y <= 0:
        raise ValueError(
            f"sigma_x and sigma_y must be positive, got sigma_x={sigma_x}, "
            f"sigma_y={sigma_y}"
        )
    if not -1 < rho < 1:
        raise ValueError(f"rho must lie in (-1, 1), got {rho}")
    if estimator not in _NORMAL_CARVING_ESTIMATORS:
        raise ValueError(
            f"estimator must be one of {_NORMAL_CARVING_ESTIMATORS}, got {estimator!r}"
        )

    if estimator == "mle":
        return _normal_carving_mle(x_obs, c, sigma_x, sigma_y, rho)
    return _normal_carving_mean(x_obs, c, sigma_x, sigma_y, rho)
