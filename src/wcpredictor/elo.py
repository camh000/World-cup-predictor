"""Elo rating mathematics.

The Elo rating is the *primary learned quantity* of the engine: it is updated
online after every real result (see :mod:`wcpredictor.learn`). The functions
here are pure and deterministic so they are easy to unit-test.
"""

from __future__ import annotations

from typing import Tuple


def expected_score(rating_a: float, rating_b: float, divisor: float = 400.0) -> float:
    """Win-expectancy of A vs B under the Elo logistic (draw counts as 0.5)."""
    return 1.0 / (1.0 + 10.0 ** (-(rating_a - rating_b) / divisor))


def mov_multiplier(goal_diff: int, rating_diff_winner: float) -> float:
    """Margin-of-victory multiplier (World Football Elo style).

    Bigger wins move ratings more, but the effect is dampened when the winner
    was already heavily favoured (``rating_diff_winner`` = winner_elo - loser_elo)
    so blowouts by strong teams do not cause runaway ratings.

    For draws (``goal_diff == 0``) and one-goal games the base multiplier is 1.0,
    so draws still update ratings (important for capturing upsets).
    """
    agd = abs(int(goal_diff))
    if agd < 2:
        base = 1.0
    elif agd == 2:
        base = 1.5
    else:
        base = (11.0 + agd) / 8.0
    damp = 2.2 / (0.001 * rating_diff_winner + 2.2)
    return base * damp


def update_elo(
    elo_home: float,
    elo_away: float,
    home_goals: int,
    away_goals: int,
    k_factor: float,
    home_advantage: float = 0.0,
    divisor: float = 400.0,
) -> Tuple[float, float]:
    """Return updated ``(elo_home, elo_away)`` after one result.

    The update is zero-sum: whatever one team gains, the other loses, so the
    total Elo across all teams is conserved.
    """
    d = (elo_home - elo_away) + home_advantage
    e_home = 1.0 / (1.0 + 10.0 ** (-d / divisor))

    if home_goals > away_goals:
        s_home = 1.0
        rating_diff_winner = d
    elif home_goals < away_goals:
        s_home = 0.0
        rating_diff_winner = -d
    else:
        s_home = 0.5
        rating_diff_winner = 0.0

    mult = mov_multiplier(home_goals - away_goals, rating_diff_winner)
    delta = k_factor * mult * (s_home - e_home)
    return elo_home + delta, elo_away - delta
