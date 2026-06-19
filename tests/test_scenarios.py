from wcpredictor.config import Params
from wcpredictor.data_io import MatchRecord, Team
from wcpredictor.ratings import Rating, RatingStore
from wcpredictor.scenarios import current_table, qualification, remaining_fixtures


def _group():
    return [
        Team("AAA", "Alpha", "UEFA", "Z"),
        Team("BBB", "Bravo", "UEFA", "Z"),
        Team("CCC", "Charlie", "UEFA", "Z"),
        Team("DDD", "Delta", "UEFA", "Z"),
    ]


def test_current_table_counts_played_results():
    matches = [MatchRecord("2026-06-11", "AAA", "BBB", 2, 0, stage="group"),
               MatchRecord("2026-06-11", "CCC", "DDD", 1, 1, stage="group")]
    st = current_table(_group(), matches, RatingStore({t.team_id: Rating() for t in _group()}))
    assert st["AAA"].points == 3 and st["AAA"].gd == 2
    assert st["BBB"].points == 0
    assert st["CCC"].points == 1 and st["DDD"].points == 1


def test_remaining_fixtures_excludes_played():
    matches = [MatchRecord("2026-06-11", "AAA", "BBB", 2, 0, stage="group")]
    rem = remaining_fixtures(_group(), matches)
    assert len(rem) == 5                       # 6 round-robin pairs minus 1 played
    assert frozenset(("AAA", "BBB")) not in {frozenset(p) for p in rem}


def test_qualification_clinch_and_eliminate():
    # A 4-team group, two rounds played; build a runaway leader and a dead team.
    teams = _group()
    ratings = RatingStore({t.team_id: Rating(elo=1500) for t in teams})
    matches = [
        MatchRecord("2026-06-11", "AAA", "BBB", 5, 0, stage="group"),
        MatchRecord("2026-06-11", "AAA", "CCC", 5, 0, stage="group"),
        MatchRecord("2026-06-12", "BBB", "DDD", 0, 0, stage="group"),
        MatchRecord("2026-06-12", "CCC", "DDD", 0, 0, stage="group"),
    ]
    # Only AAA-DDD and BBB-CCC remain; AAA on 6 with +10 GD is safe.
    _, adv, win = qualification(teams, matches, Params(), ratings, n_sims=2000, seed=1)
    assert adv["AAA"] == 1.0 and win["AAA"] > 0.9
    assert 0.0 <= adv["DDD"] <= 1.0
    assert abs(sum(win.values()) - 1.0) < 1e-9   # exactly one group winner
