"""Tournament structure: group stage, standings/tie-breakers, the 2026
third-placed-team rule, and the single-elimination knockout bracket.

The structure is data-driven via ``teams.csv`` (each team has a ``group``), so
the same engine handles any tournament with 12 groups of 4. Group fixtures are
generated as a round-robin; the knockout bracket uses a fixed, self-consistent
2026-style template (12 winners + 12 runners-up + 8 best third-placed teams).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import Params
from .data_io import Team
from .match import simulate_match
from .ratings import RatingStore


# Knockout rounds, in order, with the number of matches in each.
KNOCKOUT_ROUNDS = [("R32", 16), ("R16", 8), ("QF", 4), ("SF", 2), ("F", 1)]

# Round-of-32 seeding template. Slots resolve after the group stage:
#   ("W", "A")  -> winner of group A
#   ("RU", "A") -> runner-up of group A
#   ("T", n)    -> the (n+1)-th best third-placed team (0-indexed, 0 = best)
# This consumes exactly 12 winners, 12 runners-up and 8 thirds = 32 teams.
R32_TEMPLATE: List[Tuple[Tuple, Tuple]] = [
    (("W", "A"), ("T", 0)),
    (("W", "I"), ("RU", "A")),
    (("W", "B"), ("T", 1)),
    (("RU", "E"), ("RU", "F")),
    (("W", "C"), ("T", 2)),
    (("W", "J"), ("RU", "B")),
    (("W", "D"), ("T", 3)),
    (("RU", "G"), ("RU", "H")),
    (("W", "E"), ("T", 4)),
    (("W", "K"), ("RU", "C")),
    (("W", "F"), ("T", 5)),
    (("RU", "I"), ("RU", "J")),
    (("W", "G"), ("T", 6)),
    (("W", "L"), ("RU", "D")),
    (("W", "H"), ("T", 7)),
    (("RU", "K"), ("RU", "L")),
]

N_BEST_THIRDS = 8


@dataclass
class TeamStanding:
    team_id: str
    played: int = 0
    points: int = 0
    gf: int = 0
    ga: int = 0
    elo: float = 0.0  # deterministic final tie-breaker

    @property
    def gd(self) -> int:
        return self.gf - self.ga

    def sort_key(self) -> Tuple:
        # Higher is better; negate so ascending sort puts best first.
        return (-self.points, -self.gd, -self.gf, -self.elo, self.team_id)


@dataclass
class TournamentResult:
    champion: str
    # team_id -> furthest stage reached: "group", "R32", "R16", "QF", "SF", "F", "champion"
    reached: Dict[str, str] = field(default_factory=dict)
    group_winners: Dict[str, str] = field(default_factory=dict)  # group -> team_id


# Ordering of progression labels, used to keep only the *furthest* stage reached.
_STAGE_ORDER = ["group", "R32", "R16", "QF", "SF", "F", "champion"]
_STAGE_RANK = {s: i for i, s in enumerate(_STAGE_ORDER)}


def teams_by_group(teams: List[Team]) -> Dict[str, List[Team]]:
    groups: Dict[str, List[Team]] = {}
    for t in teams:
        groups.setdefault(t.group, []).append(t)
    return groups


def group_fixtures(group_teams: List[Team]) -> List[Tuple[str, str]]:
    """Single round-robin: every pair plays once."""
    return [(a.team_id, b.team_id) for a, b in combinations(group_teams, 2)]


def _simulate_group(
    group_teams: List[Team],
    params: Params,
    ratings: RatingStore,
    rng: np.random.Generator,
) -> List[TeamStanding]:
    standings = {t.team_id: TeamStanding(t.team_id, elo=ratings.elo(t.team_id)) for t in group_teams}
    host_ids = {t.team_id for t in group_teams if t.host}
    for home, away in group_fixtures(group_teams):
        res = simulate_match(
            home, away, params, ratings, rng,
            neutral=True, knockout=False,
            home_is_host=home in host_ids,
        )
        for tid, gf, ga in ((home, res.home_goals, res.away_goals),
                            (away, res.away_goals, res.home_goals)):
            s = standings[tid]
            s.played += 1
            s.gf += gf
            s.ga += ga
        if res.winner is None:
            standings[home].points += params.points_draw
            standings[away].points += params.points_draw
        else:
            standings[res.winner].points += params.points_win
    return sorted(standings.values(), key=lambda s: s.sort_key())


def best_third_placed(thirds: List[TeamStanding], n: int = N_BEST_THIRDS) -> List[TeamStanding]:
    """Rank all third-placed teams and return the top ``n`` that advance."""
    return sorted(thirds, key=lambda s: s.sort_key())[:n]


def _resolve_slot(slot: Tuple, winners, runners, thirds_ranked) -> str:
    kind, key = slot
    if kind == "W":
        return winners[key]
    if kind == "RU":
        return runners[key]
    if kind == "T":
        return thirds_ranked[key]
    raise ValueError(f"unknown slot kind: {kind!r}")


def simulate_tournament(
    teams: List[Team],
    params: Params,
    ratings: RatingStore,
    rng: np.random.Generator,
) -> TournamentResult:
    """Simulate one full tournament and return the champion + furthest stage
    reached by every team."""
    groups = teams_by_group(teams)
    reached: Dict[str, str] = {t.team_id: "group" for t in teams}

    winners: Dict[str, str] = {}
    runners: Dict[str, str] = {}
    thirds: List[TeamStanding] = []
    group_winner_map: Dict[str, str] = {}

    for g, gteams in groups.items():
        table = _simulate_group(gteams, params, ratings, rng)
        winners[g] = table[0].team_id
        runners[g] = table[1].team_id
        group_winner_map[g] = table[0].team_id
        if len(table) >= 3:
            thirds.append(table[2])

    qualifying_thirds = best_third_placed(thirds, N_BEST_THIRDS)
    thirds_ranked = [s.team_id for s in qualifying_thirds]

    # Everyone in the knockout has reached at least "R32".
    knockout_ids = list(winners.values()) + list(runners.values()) + thirds_ranked
    for tid in knockout_ids:
        _record(reached, tid, "R32")

    # Resolve the Round of 32 from the template.
    current = [
        (_resolve_slot(a, winners, runners, thirds_ranked),
         _resolve_slot(b, winners, runners, thirds_ranked))
        for a, b in R32_TEMPLATE
    ]

    champion: Optional[str] = None
    for round_name, n_matches in KNOCKOUT_ROUNDS:
        assert len(current) == n_matches, f"{round_name}: expected {n_matches} matches, got {len(current)}"
        round_winners: List[str] = []
        for home, away in current:
            res = simulate_match(home, away, params, ratings, rng, neutral=True, knockout=True)
            round_winners.append(res.winner)

        if round_name == "F":
            champion = round_winners[0]
            _record(reached, champion, "champion")
            break

        # Winners advance to the next stage label.
        next_label = KNOCKOUT_ROUNDS[KNOCKOUT_ROUNDS.index((round_name, n_matches)) + 1][0]
        for w in round_winners:
            _record(reached, w, next_label)

        # Pair consecutive winners for the next round.
        current = [(round_winners[i], round_winners[i + 1]) for i in range(0, len(round_winners), 2)]

    assert champion is not None
    return TournamentResult(champion=champion, reached=reached, group_winners=group_winner_map)


def _record(reached: Dict[str, str], team_id: str, stage: str) -> None:
    """Keep only the furthest stage a team reached."""
    if _STAGE_RANK[stage] > _STAGE_RANK.get(reached.get(team_id, "group"), 0):
        reached[team_id] = stage
