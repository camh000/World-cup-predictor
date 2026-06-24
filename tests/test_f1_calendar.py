"""The real 2026 F1 calendar (scripts/fetch_f1_calendar.py) + remaining-sprint scoring."""

import csv
from pathlib import Path

from wcpredictor.f1 import simulate_championship

ROOT = Path(__file__).resolve().parents[1]


def test_calendar_2026_is_the_real_schedule():
    rows = list(csv.DictReader((ROOT / "data" / "f1" / "calendar_2026.csv").open(encoding="utf-8")))
    assert len(rows) == 22                                   # real 2026 length, not the old 24
    assert sum(1 for r in rows if r["sprint"] == "true") == 6
    assert [int(r["round"]) for r in rows] == list(range(1, 23))


def test_remaining_sprints_add_points_and_zero_is_noop():
    drivers = ["a", "b", "c"]
    team_of = {"a": "X", "b": "Y", "c": "Z"}
    elo = {"a": 1600.0, "b": 1500.0, "c": 1400.0}
    dp = {d: 0.0 for d in drivers}
    cp = {t: 0.0 for t in team_of.values()}
    _, _, proj0 = simulate_championship(drivers, team_of, elo, dp, cp, 0,
                                        remaining_sprints=0, n_sims=300, seed=1)
    _, _, proj2 = simulate_championship(drivers, team_of, elo, dp, cp, 0,
                                        remaining_sprints=2, n_sims=300, seed=1)
    assert sum(proj0.values()) == 0.0          # no races, no sprints -> no points (back-compat)
    assert sum(proj2.values()) > 0.0           # sprints award points
