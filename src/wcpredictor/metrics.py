"""Proper scoring rules for 3-outcome (home win / draw / away win) forecasts.

Used by the walk-forward backtest in :mod:`wcpredictor.learn` to measure — and
then minimise — prediction error when retuning hyperparameters.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

# Outcome index convention: 0 = home win, 1 = draw, 2 = away win.
OUTCOMES = ("home", "draw", "away")


def outcome_index(home_goals: int, away_goals: int) -> int:
    if home_goals > away_goals:
        return 0
    if home_goals == away_goals:
        return 1
    return 2


def log_loss(probs: Sequence[Tuple[float, float, float]], outcomes: Sequence[int], eps: float = 1e-12) -> float:
    """Mean multiclass log-loss (a.k.a. cross-entropy). Lower is better."""
    if not probs:
        return float("nan")
    total = 0.0
    for p, o in zip(probs, outcomes):
        total += -math.log(max(p[o], eps))
    return total / len(probs)


def brier_score(probs: Sequence[Tuple[float, float, float]], outcomes: Sequence[int]) -> float:
    """Mean multiclass Brier score. Lower is better."""
    if not probs:
        return float("nan")
    total = 0.0
    for p, o in zip(probs, outcomes):
        target = [0.0, 0.0, 0.0]
        target[o] = 1.0
        total += sum((pi - ti) ** 2 for pi, ti in zip(p, target))
    return total / len(probs)


SCORERS = {"logloss": log_loss, "brier": brier_score}
