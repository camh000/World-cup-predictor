"""Poisson goal model.

Converts an Elo rating difference into expected goals for each side, then either
draws a random scoreline (for simulation) or computes exact outcome
probabilities (for the backtest / single-match prediction).
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from .config import Params


def _spread_transform(elo_diff: float, params: Params) -> float:
    """Compress the *excess* of an Elo gap above ``spread_threshold``.

    A forecast-only, odd-symmetric ("threshold-excess") lens: gaps with
    magnitude up to ``spread_threshold`` pass through unchanged, and only the
    part above it is scaled by ``spread_slope``::

        eff = sign(d) * ( min(|d|, T) + max(|d| - T, 0) * s )

    This deflates the over-confidence the raw eloratings.net scale produces on
    elite mismatches (large gaps) while leaving mid-tier favourites (sub-threshold
    gaps) untouched -- the error the model makes flips sign across the spread, so
    a *non-linear* transform is required (a linear shrink only rescales ``beta``
    and cannot fix a sign-flipping error). ``spread_slope == 1.0`` returns
    ``elo_diff`` exactly for every gap, i.e. an exact no-op.
    """
    mag = abs(elo_diff)
    t = params.spread_threshold
    if mag <= t:
        return elo_diff
    eff = t + (mag - t) * params.spread_slope
    return eff if elo_diff > 0 else -eff


def expected_goals(
    elo_diff: float,
    params: Params,
    attack_home: float = 0.0,
    defense_away: float = 0.0,
    attack_away: float = 0.0,
    defense_home: float = 0.0,
    form_home: float = 1.0,
    form_away: float = 1.0,
) -> Tuple[float, float]:
    """Expected goals ``(lambda_home, lambda_away)`` from an Elo difference.

    ``elo_diff`` should already include any home advantage. It is first passed
    through :func:`_spread_transform` (a forecast-only, non-linear compression of
    the gap's excess above ``params.spread_threshold``; an exact no-op when
    ``params.spread_slope == 1.0``). The optional per-team attack/defense offsets
    are log-scale adjustments that default to 0, so the model degrades gracefully
    to a pure-Elo goal model.

    ``form_home``/``form_away`` are the teams' tournament-form multipliers (1.0 =
    neutral): a team scores in proportion to its own form and concedes in inverse
    proportion to its opponent's form, so an in-form team both scores more and
    concedes less. The ratio is 1.0 when both teams are at neutral form.
    """
    base = params.beta * _spread_transform(elo_diff, params) / params.elo_divisor
    lam_home = params.mu * math.exp(base + attack_home - defense_away) * (form_home / form_away)
    lam_away = params.mu * math.exp(-base + attack_away - defense_home) * (form_away / form_home)
    return lam_home, lam_away


def dixon_coles_matrix(
    lam_home: float, lam_away: float, rho: float = 0.0, max_goals: int = 10
) -> np.ndarray:
    """Normalised joint score distribution ``P[i, j]`` (i home goals, j away).

    With ``rho == 0`` this is just two independent Poissons. The Dixon-Coles
    correction multiplies the four lowest-score cells by a dependency factor so
    that a negative ``rho`` lifts 0-0 and 1-1 (draws) at the expense of 1-0/0-1,
    fixing the independent model's tendency to under-predict draws.
    """
    goals = np.arange(0, max_goals + 1)
    ph = np.exp(-lam_home) * np.power(lam_home, goals) / _factorials(goals)
    pa = np.exp(-lam_away) * np.power(lam_away, goals) / _factorials(goals)
    joint = np.outer(ph, pa)
    if rho != 0.0 and max_goals >= 1:
        joint[0, 0] *= 1.0 - lam_home * lam_away * rho
        joint[0, 1] *= 1.0 + lam_home * rho
        joint[1, 0] *= 1.0 + lam_away * rho
        joint[1, 1] *= 1.0 - rho
        np.clip(joint, 0.0, None, out=joint)  # guard against negative cells
    total = joint.sum()
    if total <= 0:
        joint[:] = 1.0
        total = joint.sum()
    return joint / total


def simulate_scoreline(
    lam_home: float,
    lam_away: float,
    rng: np.random.Generator,
    rho: float = 0.0,
    max_goals: int = 10,
) -> Tuple[int, int]:
    """Draw a single random scoreline.

    Falls back to two independent Poisson draws when ``rho == 0``; otherwise
    samples from the Dixon-Coles-corrected joint distribution so simulated
    scorelines stay consistent with the forecast probabilities.
    """
    if rho == 0.0:
        return int(rng.poisson(lam_home)), int(rng.poisson(lam_away))
    # Exact rejection sampling: propose from two independent Poissons and accept
    # with probability tau(i, j) / M. The Dixon-Coles factor tau differs from 1
    # only on the four lowest-score cells, so this reproduces the corrected
    # distribution exactly without building (and cumsum-ing) a full score grid --
    # about 10x faster than the grid approach, with no change to the statistics.
    t = (
        max(0.0, 1.0 - lam_home * lam_away * rho),  # (0, 0)
        max(0.0, 1.0 + lam_home * rho),             # (0, 1)
        max(0.0, 1.0 + lam_away * rho),             # (1, 0)
        max(0.0, 1.0 - rho),                        # (1, 1)
    )
    m = max(1.0, *t)
    while True:
        i = int(rng.poisson(lam_home))
        j = int(rng.poisson(lam_away))
        tau = t[i * 2 + j] if (i <= 1 and j <= 1) else 1.0
        if rng.random() * m < tau:
            return i, j


def match_probabilities(
    lam_home: float, lam_away: float, max_goals: int = 10, rho: float = 0.0
) -> Tuple[float, float, float]:
    """Exact ``(p_home_win, p_draw, p_away_win)`` over a truncated score grid,
    optionally with the Dixon-Coles low-score correction."""
    joint = dixon_coles_matrix(lam_home, lam_away, rho, max_goals)
    p_home = float(np.tril(joint, -1).sum())  # home > away
    p_draw = float(np.trace(joint))           # home == away
    p_away = float(np.triu(joint, 1).sum())   # away > home
    total = p_home + p_draw + p_away
    if total <= 0:
        return 1 / 3, 1 / 3, 1 / 3
    return p_home / total, p_draw / total, p_away / total


_FACT_CACHE: dict[int, np.ndarray] = {}


def _factorials(goals: np.ndarray) -> np.ndarray:
    n = int(goals[-1])
    cached = _FACT_CACHE.get(n)
    if cached is None:
        cached = np.array([math.factorial(int(g)) for g in goals], dtype=float)
        _FACT_CACHE[n] = cached
    return cached
