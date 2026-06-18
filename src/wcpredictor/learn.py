"""The self-learning core.

Layer A — *online* Elo update after every real result: the model literally
alters itself after each game (:func:`apply_result`).

Layer B — *periodic* retuning of hyperparameters by a strict walk-forward
backtest over accumulated results, minimising a proper scoring rule
(:func:`backtest`, :func:`retune`). Walk-forward = predict each game using only
information available *before* it, then update — so there is no look-ahead bias.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from .config import Params, TUNABLE_FIELDS
from .data_io import MatchRecord
from .elo import update_elo
from .metrics import SCORERS, outcome_index
from .poisson import expected_goals, match_probabilities
from .ratings import DEFAULT_ELO, Rating, RatingStore


def apply_result(
    ratings: RatingStore,
    params: Params,
    record: MatchRecord,
) -> Tuple[float, float]:
    """Layer A: update the two teams' Elo ratings in place for one result.

    Unknown teams are added at the default rating first, so the engine can learn
    about teams it has never seen. Returns the Elo deltas ``(home_delta, away_delta)``.
    """
    for tid in (record.home_team_id, record.away_team_id):
        if tid not in ratings:
            ratings[tid] = Rating(elo=DEFAULT_ELO)

    home = ratings[record.home_team_id]
    away = ratings[record.away_team_id]
    home_adv = 0.0 if record.neutral else params.home_advantage

    before_home, before_away = home.elo, away.elo
    new_home, new_away = update_elo(
        home.elo, away.elo,
        record.home_goals, record.away_goals,
        k_factor=params.k_factor,
        home_advantage=home_adv,
        divisor=params.elo_divisor,
    )
    home.elo, away.elo = new_home, new_away

    # Update tournament form from the "surprise" relative to the Elo expectation
    # computed from the *pre-update* ratings. Each game first decays form back
    # toward 1.0 (mean reversion), then nudges it by the surprise. Form is a fast
    # overlay; Elo remains the slow baseline.
    if params.form_alpha > 0:
        e_home = 1.0 / (1.0 + 10.0 ** (-((before_home - before_away) + home_adv) / params.elo_divisor))
        if record.home_goals > record.away_goals:
            s_home = 1.0
        elif record.home_goals < record.away_goals:
            s_home = 0.0
        else:
            s_home = 0.5
        surprise = s_home - e_home  # >0 = home over-performed, away under-performed
        home.form = _clip_form(home.form, surprise, params)
        away.form = _clip_form(away.form, -surprise, params)

    return new_home - before_home, new_away - before_away


def _clip_form(form: float, surprise: float, params: Params) -> float:
    updated = 1.0 + (form - 1.0) * params.form_decay + params.form_alpha * surprise
    return max(params.form_min, min(params.form_max, updated))


def _forecast(home: Rating, away: Rating, params: Params, neutral: bool) -> Tuple[float, float, float]:
    home_adv = 0.0 if neutral else params.home_advantage
    d = (home.elo - away.elo) + home_adv
    lam_h, lam_a = expected_goals(
        d, params,
        attack_home=home.attack, defense_away=away.defense,
        attack_away=away.attack, defense_home=home.defense,
        form_home=home.form, form_away=away.form,
    )
    return match_probabilities(lam_h, lam_a, params.max_goals, params.dc_rho)


def backtest(
    matches: Sequence[MatchRecord],
    initial: RatingStore,
    params: Params,
    metric: str = "logloss",
) -> float:
    """Walk-forward score of ``params`` over ``matches`` (chronological order).

    For each match: forecast with the *current* ratings, record the forecast,
    then apply the online update. The accumulated forecasts are scored by the
    chosen proper scoring rule. ``initial`` is left untouched (a copy is used).
    """
    scorer = SCORERS[metric]
    ratings = initial.copy()
    probs: List[Tuple[float, float, float]] = []
    outcomes: List[int] = []

    for m in _chronological(matches):
        for tid in (m.home_team_id, m.away_team_id):
            if tid not in ratings:
                ratings[tid] = Rating(elo=DEFAULT_ELO)
        probs.append(_forecast(ratings[m.home_team_id], ratings[m.away_team_id], params, m.neutral))
        outcomes.append(outcome_index(m.home_goals, m.away_goals))
        apply_result(ratings, params, m)

    if not probs:
        return float("nan")
    return scorer(probs, outcomes)


@dataclass
class RetuneResult:
    params: Params
    score_before: float
    score_after: float
    metric: str
    success: bool


def retune(
    matches: Sequence[MatchRecord],
    initial: RatingStore,
    params: Params,
    metric: str = "logloss",
    method: str = "nelder-mead",
) -> RetuneResult:
    """Layer B: optimise the tunable hyperparameters to minimise ``metric``.

    Only :data:`wcpredictor.config.TUNABLE_FIELDS` are optimised; the rest of
    ``params`` is held fixed. Bounds keep parameters physically sensible.
    """
    score_before = backtest(matches, initial, params, metric)

    x0 = np.array([getattr(params, f) for f in TUNABLE_FIELDS], dtype=float)
    bounds = {
        "home_advantage": (0.0, 200.0),
        "k_factor": (1.0, 120.0),
        "beta": (0.05, 5.0),
        "mu": (0.5, 3.0),
    }
    lo = np.array([bounds[f][0] for f in TUNABLE_FIELDS])
    hi = np.array([bounds[f][1] for f in TUNABLE_FIELDS])

    def objective(x: np.ndarray) -> float:
        xc = np.clip(x, lo, hi)
        trial = params.copy_with(**{f: float(v) for f, v in zip(TUNABLE_FIELDS, xc)})
        score = backtest(matches, initial, trial, metric)
        if not np.isfinite(score):
            return 1e6
        return score

    if method == "grid":
        best = _grid_search(objective, lo, hi)
    else:
        # Build an explicit initial simplex. Nelder-Mead's default step for a
        # component that starts at 0 is microscopic, so a dimension seeded at 0
        # (e.g. home_advantage) would never be explored. Step each dimension by a
        # sensible fraction of its allowed range instead.
        steps = 0.15 * (hi - lo)
        simplex = np.vstack([x0] + [np.clip(x0 + s * e, lo, hi)
                                    for s, e in zip(steps, np.eye(len(x0)))])
        res = minimize(objective, x0, method="Nelder-Mead",
                       options={"xatol": 1e-2, "fatol": 1e-5, "maxiter": 600,
                                "initial_simplex": simplex})
        best = np.clip(res.x, lo, hi)

    tuned = params.copy_with(**{f: float(v) for f, v in zip(TUNABLE_FIELDS, best)})
    score_after = backtest(matches, initial, tuned, metric)

    # Never accept a worse model than we started with.
    if not (np.isfinite(score_after) and score_after <= score_before):
        return RetuneResult(params, score_before, score_before, metric, success=False)
    return RetuneResult(tuned, score_before, score_after, metric, success=True)


def _grid_search(objective, lo: np.ndarray, hi: np.ndarray, steps: int = 5) -> np.ndarray:
    """Coarse exhaustive search — transparent and deterministic."""
    import itertools

    axes = [np.linspace(l, h, steps) for l, h in zip(lo, hi)]
    best_x: Optional[np.ndarray] = None
    best_val = float("inf")
    for combo in itertools.product(*axes):
        x = np.array(combo)
        val = objective(x)
        if val < best_val:
            best_val, best_x = val, x
    return best_x if best_x is not None else (lo + hi) / 2


def _chronological(matches: Sequence[MatchRecord]) -> List[MatchRecord]:
    # Stable sort by date string (ISO dates sort lexicographically); blanks last.
    return sorted(matches, key=lambda m: (m.date == "", m.date))
