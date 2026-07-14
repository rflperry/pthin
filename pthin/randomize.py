"""Randomized thinning of p-values."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def pthin(
    p: ArrayLike,
    epsilon: float = 0.5,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Split p-values into two randomized, marginally valid p-values.

    Under the null hypothesis, a p-value ``p`` is (super-)uniform on
    ``[0, 1]``. ``pthin`` uses independent auxiliary randomness to derive a
    pair ``(p1, p2)`` from ``p`` such that, under the null, ``p1`` and ``p2``
    are each *also* (super-)uniform on ``[0, 1]`` and approximately
    independent of one another. This lets one thinned p-value be used to
    select a hypothesis or threshold and the other for inference, avoiding
    the double-dipping that would come from reusing ``p`` for both.

    For each entry of ``p``, draw ``Z ~ Uniform(0, 1)`` and
    ``C ~ Bernoulli(epsilon)`` independently, and set:

    - if ``C = 1``: ``p1 = p ** epsilon``, ``p2 = p ** (1 - epsilon) * Z``
    - if ``C = 0``: ``p1 = p ** epsilon * Z``, ``p2 = p ** (1 - epsilon)``

    so that ``p1 * p2 = p * Z`` in either case. ``epsilon`` controls how the
    signal in ``p`` is divided: ``epsilon=1`` returns ``p1 = p`` with ``p2``
    pure noise, ``epsilon=0`` does the reverse, and ``epsilon=0.5`` splits
    the signal evenly between ``p1`` and ``p2``.

    Parameters
    ----------
    p : array_like
        P-value(s) to thin. Every entry must lie in ``[0, 1]``.
    epsilon : float, default=0.5
        Fraction of the signal in ``p`` allocated to ``p1`` (the remainder,
        ``1 - epsilon``, goes to ``p2``). Must lie in ``[0, 1]``.
    rng : numpy.random.Generator, optional
        Random number generator to use. Defaults to a fresh
        ``numpy.random.default_rng()`` when not given; pass a seeded
        generator for reproducible output.

    Returns
    -------
    p1 : numpy.ndarray
        First thinned p-value, same shape as ``p``.
    p2 : numpy.ndarray
        Second thinned p-value, same shape as ``p``.

    Raises
    ------
    ValueError
        If ``epsilon`` is not in ``[0, 1]``, or ``p`` contains values outside
        ``[0, 1]``.

    Examples
    --------
    >>> import numpy as np
    >>> p = np.array([0.01, 0.5, 0.9])
    >>> p1, p2 = pthin(p, epsilon=0.5)
    """
    if not 0 <= epsilon <= 1:
        raise ValueError(f"epsilon must lie in [0, 1], got {epsilon}")

    p = np.asarray(p, dtype=float)
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p must contain values in [0, 1]")

    rng = np.random.default_rng() if rng is None else rng

    z = rng.uniform(0, 1, size=p.shape)
    c = rng.binomial(1, epsilon, size=p.shape)

    p1 = p**epsilon * (c + (1 - c) * z)
    p2 = p ** (1 - epsilon) * ((1 - c) + c * z)

    return p1, p2
