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
) -> Tuple[float, float]:
    """Expected goals ``(lambda_home, lambda_away)`` from an Elo difference.

    ``elo_diff`` should already include any home advantage. The optional
    per-team attack/defense offsets are log-scale adjustments that default to 0,
    so the model degrades gracefully to a pure-Elo goal model.
    """
    base = params.beta * elo_diff / params.elo_divisor
    lam_home = params.mu * math.exp(base + attack_home - defense_away)
    lam_away = params.mu * math.exp(-base + attack_away - defense_home)
    return lam_home, lam_away


def simulate_scoreline(lam_home: float, lam_away: float, rng: np.random.Generator) -> Tuple[int, int]:
    """Draw a single random scoreline from two independent Poisson processes."""
    return int(rng.poisson(lam_home)), int(rng.poisson(lam_away))


def match_probabilities(
    lam_home: float, lam_away: float, max_goals: int = 10
) -> Tuple[float, float, float]:
    """Exact ``(p_home_win, p_draw, p_away_win)`` over a truncated score grid."""
    goals = np.arange(0, max_goals + 1)
    # Poisson pmf vectors for each side.
    ph = np.exp(-lam_home) * np.power(lam_home, goals) / _factorials(goals)
    pa = np.exp(-lam_away) * np.power(lam_away, goals) / _factorials(goals)
    # Joint distribution over (home_goals, away_goals).
    joint = np.outer(ph, pa)
    p_home = float(np.tril(joint, -1).sum())  # home > away
    p_draw = float(np.trace(joint))           # home == away
    p_away = float(np.triu(joint, 1).sum())   # away > home
    total = p_home + p_draw + p_away
    if total <= 0:
        return 1 / 3, 1 / 3, 1 / 3
    # Renormalise to absorb the small mass beyond the truncation grid.
    return p_home / total, p_draw / total, p_away / total


_FACT_CACHE: dict[int, np.ndarray] = {}


def _factorials(goals: np.ndarray) -> np.ndarray:
    n = int(goals[-1])
    cached = _FACT_CACHE.get(n)
    if cached is None:
        cached = np.array([math.factorial(int(g)) for g in goals], dtype=float)
        _FACT_CACHE[n] = cached
    return cached
