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

    ``elo_diff`` should already include any home advantage. The optional
    per-team attack/defense offsets are log-scale adjustments that default to 0,
    so the model degrades gracefully to a pure-Elo goal model.

    ``form_home``/``form_away`` are the teams' tournament-form multipliers (1.0 =
    neutral): a team scores in proportion to its own form and concedes in inverse
    proportion to its opponent's form, so an in-form team both scores more and
    concedes less. The ratio is 1.0 when both teams are at neutral form.
    """
    base = params.beta * elo_diff / params.elo_divisor
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
    joint = dixon_coles_matrix(lam_home, lam_away, rho, max_goals)
    cdf = np.cumsum(joint.ravel())
    idx = int(np.searchsorted(cdf, rng.random() * cdf[-1]))
    i, j = divmod(idx, max_goals + 1)
    return int(i), int(j)


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
