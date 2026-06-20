"""Probability calibration for the 1X2 forecasts.

The raw model is mis-calibrated in a *sign-flipping* way: it was over-confident on
elite mismatches (rating a top team far above the market in a blow-out) yet
slightly under-confident on mid-tier favourites, where it leaks probability onto
draws and underdog wins. Because the error changes sign across the spread, a
single global temperature cannot fix it — the structural fix lives upstream in the
non-linear spread compression of :mod:`wcpredictor.poisson` (``Params.spread_slope``).

This module provides the *secondary*, market-aware adjustments layered on top:

  * ``sharpen`` — a single-parameter temperature ``q_i proportional to p_i ** gamma``,
    fitted to minimise log-loss on settled games. ``gamma > 1`` sharpens (mass onto
    the favourite, away from longshots and draws); ``gamma == 1`` is unchanged. It
    is monotonic, keeps a valid distribution, and adds exactly one degree of freedom.
    With the spread fix in place the fitted gamma now sits close to 1 — the raw
    model needs almost no sharpening.
  * ``blend`` — shrink the model toward the market to filter the small, noisy
    "edges" it still invents on most games.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

Triple = Tuple[float, float, float]


def sharpen(probs: Sequence[float], gamma: float, eps: float = 1e-12) -> Triple:
    """Return ``probs`` re-weighted by exponent ``gamma`` and renormalised."""
    xs = [max(p, eps) ** gamma for p in probs]
    s = sum(xs)
    return (xs[0] / s, xs[1] / s, xs[2] / s)


def blend(model: Sequence[float], market: Sequence[float], w: float) -> Triple:
    """Shrink ``model`` toward ``market`` by weight ``w`` (0=pure model, 1=pure market).

    The market is the best single probability estimate available, so we only act
    on a disagreement that survives trusting the market this much. This filters
    out the small, noisy "edges" an over-eager model invents on every game.
    """
    b = tuple((1.0 - w) * m + w * k for m, k in zip(model, market))
    s = sum(b)
    return (b[0] / s, b[1] / s, b[2] / s)


def _log_loss(probs: Sequence[Triple], outcomes: Sequence[int], eps: float = 1e-12) -> float:
    return sum(-math.log(max(p[o], eps)) for p, o in zip(probs, outcomes)) / len(probs)


def fit_sharpness(
    probs: Sequence[Triple],
    outcomes: Sequence[int],
    lo: float = 0.3,
    hi: float = 3.0,
) -> float:
    """Fit the sharpening exponent ``gamma`` that minimises log-loss.

    Coarse grid then a local refine — no SciPy dependency, plenty accurate for a
    smooth 1-D objective. Returns ``1.0`` (no change) when there is no data.
    """
    if not probs:
        return 1.0

    def loss(g: float) -> float:
        return _log_loss([sharpen(p, g) for p in probs], outcomes)

    best = min((x / 100.0 for x in range(int(lo * 100), int(hi * 100) + 1, 5)), key=loss)
    # Refine around the best grid point.
    step = 0.05
    for _ in range(3):
        step /= 2.0
        for g in (best - step, best + step):
            if lo <= g <= hi and loss(g) < loss(best):
                best = g
    return round(best, 3)
