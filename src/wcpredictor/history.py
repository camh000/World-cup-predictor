"""Prediction history & accuracy tracking.

Every time a real result is recorded, we snapshot — *before* the model learns
from it — what the model predicted (forecast probabilities, expected goals, and
both teams' pre-match Elo), then the actual outcome and the post-update ratings,
plus the per-match accuracy score. This builds an append-only ledger so you can:

  * see how accurate the predictions were (running log-loss / Brier vs a baseline);
  * check calibration (do "70%" calls win ~70% of the time?);
  * spot whether accuracy is drifting, signalling the model needs retuning.

Two artefacts are written, both in ``data/`` so they survive ``reset`` and live
in version control alongside results:

  * ``predictions.csv``     — one human-readable row per match (the ledger).
  * ``ratings_history.csv`` — long-format snapshot of every team's Elo after each
                              match, so each team's rating trajectory is preserved.
"""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

from .config import Params
from .data_io import MatchRecord
from .metrics import OUTCOMES, brier_score, log_loss, outcome_index
from .poisson import expected_goals, match_probabilities
from .ratings import RatingStore


PREDICTIONS_HEADER = [
    "seq", "date", "home_team_id", "away_team_id",
    "home_elo_pre", "away_elo_pre",
    "p_home", "p_draw", "p_away",
    "exp_home_goals", "exp_away_goals",
    "predicted_outcome",
    "home_goals", "away_goals", "actual_outcome",
    "home_elo_post", "away_elo_post", "home_delta", "away_delta",
    "log_loss", "brier",
]

RATINGS_HISTORY_HEADER = ["seq", "date", "home_team_id", "away_team_id", "team_id", "elo"]


@dataclass
class PredictionRecord:
    seq: int
    date: str
    home_team_id: str
    away_team_id: str
    home_elo_pre: float
    away_elo_pre: float
    p_home: float
    p_draw: float
    p_away: float
    exp_home_goals: float
    exp_away_goals: float
    predicted_outcome: str
    home_goals: int
    away_goals: int
    actual_outcome: str
    home_elo_post: float
    away_elo_post: float
    home_delta: float
    away_delta: float
    log_loss: float
    brier: float


def forecast(ratings: RatingStore, params: Params, home: str, away: str, neutral: bool = True):
    """Pre-match forecast: ``((p_home, p_draw, p_away), (lam_home, lam_away))``."""
    rh, ra = ratings[home], ratings[away]
    home_adv = 0.0 if neutral else params.home_advantage
    d = (rh.elo - ra.elo) + home_adv
    lam_h, lam_a = expected_goals(
        d, params,
        attack_home=rh.attack, defense_away=ra.defense,
        attack_away=ra.attack, defense_home=rh.defense,
        form_home=rh.form, form_away=ra.form,
    )
    return match_probabilities(lam_h, lam_a, params.max_goals, params.dc_rho), (lam_h, lam_a)


def build_record(
    seq: int,
    ratings_pre: RatingStore,
    ratings_post: RatingStore,
    params: Params,
    record: MatchRecord,
    deltas,
) -> PredictionRecord:
    """Assemble a full ledger row from the pre/post ratings and the result."""
    (p_home, p_draw, p_away), (lam_h, lam_a) = forecast(
        ratings_pre, params, record.home_team_id, record.away_team_id, record.neutral
    )
    probs = (p_home, p_draw, p_away)
    actual_idx = outcome_index(record.home_goals, record.away_goals)
    pred_idx = max(range(3), key=lambda i: probs[i])
    dh, da = deltas
    return PredictionRecord(
        seq=seq,
        date=record.date,
        home_team_id=record.home_team_id,
        away_team_id=record.away_team_id,
        home_elo_pre=round(ratings_pre.elo(record.home_team_id), 1),
        away_elo_pre=round(ratings_pre.elo(record.away_team_id), 1),
        p_home=round(p_home, 4),
        p_draw=round(p_draw, 4),
        p_away=round(p_away, 4),
        exp_home_goals=round(lam_h, 3),
        exp_away_goals=round(lam_a, 3),
        predicted_outcome=OUTCOMES[pred_idx],
        home_goals=record.home_goals,
        away_goals=record.away_goals,
        actual_outcome=OUTCOMES[actual_idx],
        home_elo_post=round(ratings_post.elo(record.home_team_id), 1),
        away_elo_post=round(ratings_post.elo(record.away_team_id), 1),
        home_delta=round(dh, 1),
        away_delta=round(da, 1),
        log_loss=round(log_loss([probs], [actual_idx]), 4),
        brier=round(brier_score([probs], [actual_idx]), 4),
    )


def next_seq(path: Path) -> int:
    rows = read_predictions(path)
    return (max((int(r["seq"]) for r in rows), default=0)) + 1


def append_prediction(path: Path, record: PredictionRecord) -> None:
    path = Path(path)
    new_file = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        if new_file:
            writer.writerow(PREDICTIONS_HEADER)
        row = asdict(record)
        writer.writerow([row[c] for c in PREDICTIONS_HEADER])


def append_ratings_snapshot(
    path: Path, seq: int, record: MatchRecord, ratings: RatingStore
) -> None:
    """Append a long-format snapshot of every team's Elo after this match."""
    path = Path(path)
    new_file = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        if new_file:
            writer.writerow(RATINGS_HISTORY_HEADER)
        for team_id, rating in sorted(ratings.items()):
            writer.writerow([seq, record.date, record.home_team_id,
                             record.away_team_id, team_id, round(rating.elo, 1)])


def read_predictions(path: Path) -> List[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return [row for row in csv.DictReader(fh) if row.get("seq")]


# --------------------------------------------------------------------------- #
# Accuracy summary
# --------------------------------------------------------------------------- #
UNIFORM_LOGLOSS = math.log(3)  # ~1.0986: a model that always guesses 1/3 each


@dataclass
class AccuracySummary:
    n: int
    log_loss: float
    brier: float
    baseline_log_loss: float
    hit_rate: float            # fraction where the most-likely outcome occurred
    recent_log_loss: Optional[float]
    skill: float               # 1 - log_loss/baseline; >0 means better than guessing


def summarize(rows: List[dict], recent: int = 10) -> Optional[AccuracySummary]:
    """Compute running accuracy over a list of ledger rows."""
    if not rows:
        return None
    ll = [float(r["log_loss"]) for r in rows]
    bs = [float(r["brier"]) for r in rows]
    hits = sum(1 for r in rows if r["predicted_outcome"] == r["actual_outcome"])
    mean_ll = sum(ll) / len(ll)
    recent_ll = sum(ll[-recent:]) / len(ll[-recent:]) if len(ll) >= recent else None
    return AccuracySummary(
        n=len(rows),
        log_loss=mean_ll,
        brier=sum(bs) / len(bs),
        baseline_log_loss=UNIFORM_LOGLOSS,
        hit_rate=hits / len(rows),
        recent_log_loss=recent_ll,
        skill=1.0 - mean_ll / UNIFORM_LOGLOSS,
    )
