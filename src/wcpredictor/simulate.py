"""Monte Carlo driver: run the tournament many times and aggregate per-team
probabilities into a tidy table."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from .config import Params
from .data_io import Team
from .ratings import RatingStore
from .tournament import simulate_tournament, _STAGE_ORDER


# Probability columns reported, mapped to the minimum stage that counts as "reached".
_PROB_COLUMNS = [
    ("p_advance", "R32"),   # made the knockout stage
    ("p_r16", "R16"),
    ("p_qf", "QF"),
    ("p_sf", "SF"),
    ("p_final", "F"),
    ("p_champion", "champion"),
]

_RANK = {s: i for i, s in enumerate(_STAGE_ORDER)}


def run_simulation(
    teams: List[Team],
    params: Params,
    ratings: RatingStore,
    n_sims: int = 10000,
    seed: int | None = None,
) -> pd.DataFrame:
    """Run ``n_sims`` tournaments and return a DataFrame of probabilities.

    Reproducible: the same ``(ratings, params, seed)`` always yields the same
    table. Per-simulation RNGs are spawned from the root seed so individual
    tournaments are independent and the work could be parallelised later.
    """
    team_ids = [t.team_id for t in teams]
    counts = {tid: {col: 0 for col, _ in _PROB_COLUMNS} for tid in team_ids}
    group_winner_counts = {tid: 0 for tid in team_ids}

    root = np.random.default_rng(seed)
    child_seeds = root.spawn(n_sims)

    for sim_rng in child_seeds:
        result = simulate_tournament(teams, params, ratings, sim_rng)
        for tid, stage in result.reached.items():
            reached_rank = _RANK[stage]
            for col, threshold in _PROB_COLUMNS:
                if reached_rank >= _RANK[threshold]:
                    counts[tid][col] += 1
        for winner in result.group_winners.values():
            group_winner_counts[winner] += 1

    rows = []
    name_by_id = {t.team_id: t.name for t in teams}
    group_by_id = {t.team_id: t.group for t in teams}
    for tid in team_ids:
        row = {
            "team_id": tid,
            "team": name_by_id[tid],
            "group": group_by_id[tid],
            "elo": round(ratings.elo(tid), 1),
            "p_group_winner": group_winner_counts[tid] / n_sims,
        }
        for col, _ in _PROB_COLUMNS:
            row[col] = counts[tid][col] / n_sims
        rows.append(row)

    df = pd.DataFrame(rows)
    return df.sort_values("p_champion", ascending=False).reset_index(drop=True)
