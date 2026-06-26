"""Tests for the football-data.org result mapping (wcpredictor.fetch.parse_matches).

No network: parse_matches() works on already-fetched payload dicts.
"""

from wcpredictor.data_io import Team
from wcpredictor.fetch import parse_matches


def _team(tid, name, host=False):
    return Team(team_id=tid, name=name, confederation="X", group="A", host=host)


TEAMS = [
    _team("MEX", "Mexico", host=True),
    _team("KOR", "South Korea"),
    _team("TUR", "Turkey"),
    _team("CUW", "Curacao"),
    _team("CIV", "Ivory Coast"),
    _team("COD", "DR Congo"),
    _team("CPV", "Cape Verde"),
    _team("GER", "Germany"),
]


def _match(home, away, hg, ag, stage="GROUP_STAGE", date="2026-06-25T19:00:00Z"):
    return {
        "utcDate": date,
        "stage": stage,
        "homeTeam": {"name": home},
        "awayTeam": {"name": away},
        "score": {"fullTime": {"home": hg, "away": ag}},
    }


def test_maps_alias_and_accented_names():
    # Provider spellings differ from teams.csv: Türkiye, Curaçao, Korea Republic,
    # Côte d'Ivoire, DR Congo, Cabo Verde — all must resolve, none skipped.
    payload = [
        _match("Türkiye", "Korea Republic", 3, 2),
        _match("Curaçao", "Côte d'Ivoire", 0, 2),
        _match("DR Congo", "Cabo Verde", 1, 1),
    ]
    recs = parse_matches(payload, TEAMS)
    pairs = {(r.home_team_id, r.away_team_id) for r in recs}
    assert pairs == {("TUR", "KOR"), ("CUW", "CIV"), ("COD", "CPV")}


def test_group_stage_is_normalised_to_group():
    recs = parse_matches([_match("Germany", "Curaçao", 7, 1)], TEAMS)
    assert recs[0].stage == "group"


def test_knockout_stage_labels():
    recs = parse_matches(
        [_match("Germany", "Mexico", 1, 0, stage="QUARTER_FINALS")], TEAMS)
    assert recs[0].stage == "quarter finals"


def test_host_at_home_is_not_neutral():
    recs = parse_matches([_match("Mexico", "South Korea", 1, 0)], TEAMS)
    assert recs[0].neutral is False
    recs = parse_matches([_match("Germany", "Mexico", 2, 2)], TEAMS)
    assert recs[0].neutral is True  # host as away team -> still neutral venue


def test_unfinished_and_unknown_teams_are_skipped():
    payload = [
        {"utcDate": "2026-06-25", "stage": "GROUP_STAGE",
         "homeTeam": {"name": "Germany"}, "awayTeam": {"name": "Mexico"},
         "score": {"fullTime": {"home": None, "away": None}}},          # not finished
        _match("Germany", "Atlantis", 5, 0),                            # unknown team
    ]
    assert parse_matches(payload, TEAMS) == []


def test_scores_and_date_are_extracted():
    recs = parse_matches([_match("Germany", "Curaçao", 7, 1)], TEAMS)
    r = recs[0]
    assert (r.home_goals, r.away_goals, r.date) == (7, 1, "2026-06-25")
