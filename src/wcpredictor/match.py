"""Single-match simulation: group games (may draw) and knockout games
(resolved via extra time and penalties)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import Params
from .poisson import expected_goals
from .ratings import RatingStore


@dataclass
class MatchResult:
    home: str
    away: str
    home_goals: int
    away_goals: int
    winner: Optional[str]          # None only for a drawn group game
    decided_by: str = "regulation"  # regulation | extra_time | penalties

    @property
    def is_draw(self) -> bool:
        return self.winner is None


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def simulate_match(
    home: str,
    away: str,
    params: Params,
    ratings: RatingStore,
    rng: np.random.Generator,
    *,
    neutral: bool = True,
    knockout: bool = False,
    home_is_host: bool = False,
) -> MatchResult:
    """Simulate one match.

    Group games can end level (``winner is None``). Knockout games always return
    a winner: a level score triggers a 30-minute extra-time Poisson draw, and a
    still-level score is settled by an Elo-weighted penalty shootout.
    """
    rh = ratings[home]
    ra = ratings[away]
    home_adv = 0.0 if (neutral and not home_is_host) else params.home_advantage
    d = (rh.elo - ra.elo) + home_adv

    lam_h, lam_a = expected_goals(
        d, params,
        attack_home=rh.attack, defense_away=ra.defense,
        attack_away=ra.attack, defense_home=rh.defense,
        form_home=rh.form, form_away=ra.form,
    )

    hg = int(rng.poisson(lam_h))
    ag = int(rng.poisson(lam_a))

    if not knockout:
        winner = None if hg == ag else (home if hg > ag else away)
        return MatchResult(home, away, hg, ag, winner)

    if hg != ag:
        winner = home if hg > ag else away
        return MatchResult(home, away, hg, ag, winner)

    # Extra time: 30 minutes => scale expected goals by 30/90.
    hg += int(rng.poisson(lam_h * (30.0 / 90.0)))
    ag += int(rng.poisson(lam_a * (30.0 / 90.0)))
    if hg != ag:
        winner = home if hg > ag else away
        return MatchResult(home, away, hg, ag, winner, decided_by="extra_time")

    # Penalty shootout: near coin-flip, gently tilted by Elo.
    p_home = _clamp(0.5 + params.penalty_elo_weight * (d / params.elo_divisor), 0.05, 0.95)
    if rng.random() < p_home:
        return MatchResult(home, away, hg + 1, ag, home, decided_by="penalties")
    return MatchResult(home, away, hg, ag + 1, away, decided_by="penalties")
