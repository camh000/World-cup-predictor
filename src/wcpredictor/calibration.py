"""Probability calibration for the 1X2 forecasts.

The raw model leaks too much probability onto unlikely outcomes (underdog wins
and draws), so against fair market odds it "finds value" almost everywhere — a
classic sign of an under-confident, poorly-calibrated model rather than a sharp
one.

The fix here is a single-parameter recalibration ("temperature"/sharpening):

    q_i  proportional to  p_i ** gamma

with ``gamma`` fitted to minimise log-loss on games already played. ``gamma > 1``
sharpens the distribution (mass moves onto the favourite, away from longshots and
draws); ``gamma == 1`` leaves the probabilities unchanged. It is monotonic, keeps
the probabilities a valid distribution, and adds exactly one degree of freedom —
appropriate for the handful of results we can fit on.
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
