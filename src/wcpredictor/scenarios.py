"""Group qualification scenarios.

Unlike ``simulate`` (which replays the whole tournament from scratch), this
conditions on the group games *already played* — taking current points/goals as
fixed — and Monte-Carlo simulates only the remaining group fixtures to estimate
each team's chance of advancing (top two of the group, or one of the eight best
third-placed teams across all groups).
"""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np

from .config import Params
from .data_io import MatchRecord, Team
from .match import simulate_match
from .ratings import RatingStore
from .tournament import N_BEST_THIRDS, TeamStanding, best_third_placed, teams_by_group


def current_table(gteams: List[Team], matches: List[MatchRecord], ratings: RatingStore) -> Dict[str, TeamStanding]:
    """Standings from the group games already played among ``gteams``."""
    st = {t.team_id: TeamStanding(t.team_id, elo=ratings.elo(t.team_id), group=t.group) for t in gteams}
    ids = set(st)
    for m in matches:
        if m.home_team_id in ids and m.away_team_id in ids:
            h, a = st[m.home_team_id], st[m.away_team_id]
            h.played += 1; a.played += 1
            h.gf += m.home_goals; h.ga += m.away_goals
            a.gf += m.away_goals; a.ga += m.home_goals
            if m.home_goals > m.away_goals:
                h.points += 3
            elif m.home_goals < m.away_goals:
                a.points += 3
            else:
                h.points += 1; a.points += 1
    return st


def remaining_fixtures(gteams: List[Team], matches: List[MatchRecord]) -> List[Tuple[str, str]]:
    """Round-robin pairs in this group not yet played."""
    ids = {t.team_id for t in gteams}
    played = {frozenset((m.home_team_id, m.away_team_id))
              for m in matches if m.home_team_id in ids and m.away_team_id in ids}
    return [(a, b) for a, b in combinations([t.team_id for t in gteams], 2)
            if frozenset((a, b)) not in played]


def qualification(
    teams: List[Team],
    matches: List[MatchRecord],
    params: Params,
    ratings: RatingStore,
    n_sims: int = 20000,
    seed: int = 42,
):
    """Return ``(base_tables, advance_prob, win_group_prob)`` conditioned on
    results so far. ``base_tables`` maps group -> current standings dict."""
    groups = teams_by_group(teams)
    base = {g: current_table(gt, matches, ratings) for g, gt in groups.items()}
    rem = {g: remaining_fixtures(gt, matches) for g, gt in groups.items()}

    advance = {t.team_id: 0 for t in teams}
    win_group = {t.team_id: 0 for t in teams}

    for rng in np.random.default_rng(seed).spawn(n_sims):
        thirds: List[TeamStanding] = []
        for g in groups:
            st = {tid: replace(s) for tid, s in base[g].items()}
            for home, away in rem[g]:
                res = simulate_match(home, away, params, ratings, rng, neutral=True, knockout=False)
                h, a = st[home], st[away]
                h.played += 1; a.played += 1
                h.gf += res.home_goals; h.ga += res.away_goals
                a.gf += res.away_goals; a.ga += res.home_goals
                if res.winner is None:
                    h.points += params.points_draw; a.points += params.points_draw
                else:
                    st[res.winner].points += params.points_win
            table = sorted(st.values(), key=lambda s: s.sort_key())
            advance[table[0].team_id] += 1
            advance[table[1].team_id] += 1
            win_group[table[0].team_id] += 1
            thirds.append(table[2])
        for s in best_third_placed(thirds, N_BEST_THIRDS):
            advance[s.team_id] += 1

    return (base,
            {tid: advance[tid] / n_sims for tid in advance},
            {tid: win_group[tid] / n_sims for tid in win_group})
