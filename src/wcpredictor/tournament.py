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


# Official 2026 Round-of-32 bracket, in match-number order (matches 73-88 of the
# FIFA schedule). Codes: 1X = winner of group X, 2X = runner-up X, 3XXXX = a best
# third-placed team from one of the listed groups (FIFA's allocation table). The
# match-number order also defines the knockout tree: adjacent pairs meet in the
# Round of 16, and so on.
_R32_2026_CODES = [
    ("2A", "2B"), ("1E", "3ABCDF"), ("1F", "2C"), ("1C", "2F"),
    ("1I", "3CDFGH"), ("2E", "2I"), ("1A", "3CEFHI"), ("1L", "3EHIJK"),
    ("1D", "3BEFIJ"), ("1G", "3AEHIJ"), ("2K", "2L"), ("1H", "2J"),
    ("1B", "3EFGIJ"), ("1J", "2H"), ("1K", "3DEIJL"), ("2D", "2G"),
]


def _parse_slot_code(code: str):
    head, rest = code[0], code[1:]
    if head == "1":
        return ("W", rest)
    if head == "2":
        return ("RU", rest)
    if head == "3":
        return ("3", frozenset(rest))
    raise ValueError(f"bad slot code: {code!r}")


R32_2026 = [(_parse_slot_code(a), _parse_slot_code(b)) for a, b in _R32_2026_CODES]


def _assign_thirds(slot_allowed: List[frozenset], third_groups: List[str]) -> List[int]:
    """Match each third-placed slot to a qualifying third from an allowed group.

    Bipartite maximum matching (Kuhn's algorithm): a slot like ``3ABCDF`` can only
    take a third that finished 3rd in group A, B, C, D or F. Returns, per slot, the
    index into ``third_groups`` (or -1 if unmatched, handled by the caller).
    """
    match_slot = [-1] * len(slot_allowed)
    match_third = [-1] * len(third_groups)

    def augment(s: int, seen: List[bool]) -> bool:
        for t, g in enumerate(third_groups):
            if g in slot_allowed[s] and not seen[t]:
                seen[t] = True
                if match_third[t] == -1 or augment(match_third[t], seen):
                    match_slot[s] = t
                    match_third[t] = s
                    return True
        return False

    for s in range(len(slot_allowed)):
        augment(s, [False] * len(third_groups))
    return match_slot



@dataclass
class TeamStanding:
    team_id: str
    played: int = 0
    points: int = 0
    gf: int = 0
    ga: int = 0
    elo: float = 0.0  # deterministic final tie-breaker
    group: str = ""

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


def _build_r32(r32_slots, winners, runners, thirds_by_group):
    """Resolve a bracket spec into 16 concrete (home, away) Round-of-32 matchups."""
    ranked = best_third_placed(list(thirds_by_group.values()), N_BEST_THIRDS)
    third_ids = [s.team_id for s in ranked]
    third_groups = [s.group for s in ranked]

    # Constraint-match the 8 qualifying thirds to the 8 "3XXXX" slots, in order.
    slot_allowed = [s[1] for a, b in r32_slots for s in (a, b) if s[0] == "3"]
    assignment = _assign_thirds(slot_allowed, third_groups)
    used = {t for t in assignment if t != -1}
    leftover = iter([third_ids[i] for i in range(len(third_ids)) if i not in used])
    third_queue = iter([third_ids[a] if a != -1 else next(leftover) for a in assignment])

    def resolve(slot):
        kind, key = slot
        if kind == "W":
            return winners[key]
        if kind == "RU":
            return runners[key]
        return next(third_queue)  # "3" slot, consumed in spec order

    return [(resolve(a), resolve(b)) for a, b in r32_slots]


def simulate_tournament(
    teams: List[Team],
    params: Params,
    ratings: RatingStore,
    rng: np.random.Generator,
    r32_slots=R32_2026,
) -> TournamentResult:
    """Simulate one full tournament and return the champion + furthest stage
    reached by every team. ``r32_slots`` is the Round-of-32 bracket spec
    (defaults to the official 2026 mapping)."""
    groups = teams_by_group(teams)
    reached: Dict[str, str] = {t.team_id: "group" for t in teams}

    winners: Dict[str, str] = {}
    runners: Dict[str, str] = {}
    thirds_by_group: Dict[str, TeamStanding] = {}
    group_winner_map: Dict[str, str] = {}

    for g, gteams in groups.items():
        table = _simulate_group(gteams, params, ratings, rng)
        winners[g] = table[0].team_id
        runners[g] = table[1].team_id
        group_winner_map[g] = table[0].team_id
        if len(table) >= 3:
            third = table[2]
            third.group = g
            thirds_by_group[g] = third

    current = _build_r32(r32_slots, winners, runners, thirds_by_group)

    # Everyone in the knockout has reached at least "R32".
    for home, away in current:
        _record(reached, home, "R32")
        _record(reached, away, "R32")

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
