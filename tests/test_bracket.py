"""Tests for the knockout-bracket section (scripts/make_dashboard.py).

scripts/ is not a package, so load the module by path (cf. test_fetch_odds.py).
The bracket only appears once every group game is played.
"""

import importlib.util
from collections import defaultdict
from pathlib import Path

from wcpredictor.config import Params
from wcpredictor.data_io import read_seed_ratings, read_teams
from wcpredictor.ratings import RatingStore
from wcpredictor.tournament import TeamStanding

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "make_dashboard", ROOT / "scripts" / "make_dashboard.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _fixture():
    teams = read_teams(ROOT / "data" / "teams.csv")
    ratings = RatingStore.seed(teams, read_seed_ratings(ROOT / "data" / "seed_ratings.csv"))
    name = {t.team_id: t.name for t in teams}
    groups = defaultdict(list)
    for t in teams:
        groups[t.group].append(t)
    # A fully-decided table per group: distinct points so 1st/2nd/3rd are clear.
    base = {}
    for g, ts in groups.items():
        ordered = sorted(ts, key=lambda x: ratings.elo(x.team_id), reverse=True)
        base[g] = {t.team_id: TeamStanding(t.team_id, played=3, points=9 - 3 * i,
                                           gf=6 - i, ga=i, elo=ratings.elo(t.team_id), group=g)
                   for i, t in enumerate(ordered)}
    return base, ratings, name, teams


def test_no_bracket_until_group_stage_complete():
    md = _load()
    base, ratings, name, _ = _fixture()
    one = next(iter(next(iter(base.values())).values()))
    one.played = 2  # one game outstanding -> not complete
    assert md.group_stage_complete(base) is False
    assert md._knockout_bracket(base, Params(), ratings, name) == ""


def test_bracket_has_16_ties_when_complete():
    md = _load()
    base, ratings, name, _ = _fixture()
    assert md.group_stage_complete(base) is True
    html = md._knockout_bracket(base, Params(), ratings, name)
    assert "ROUND OF 32" in html
    assert html.count("Round of 32</font>") == 16


def test_bracket_uses_32_distinct_qualified_teams():
    md = _load()
    base, ratings, name, _ = _fixture()
    from wcpredictor.tournament import R32_2026, _build_r32
    winners, runners, thirds = {}, {}, {}
    for g, tbl in base.items():
        order = sorted(tbl.values(), key=lambda s: s.sort_key())
        winners[g], runners[g], thirds[g] = order[0].team_id, order[1].team_id, order[2]
    ties = _build_r32(R32_2026, winners, runners, thirds)
    qualified = [t for tie in ties for t in tie]
    assert len(qualified) == 32
    assert len(set(qualified)) == 32  # no team appears twice
