"""Closing-line-value: the betting.clv primitive + the dashboard scoreboard
aggregation (scripts/make_dashboard.py is not a package, so load it by path).
"""

import importlib.util
from pathlib import Path

from wcpredictor.betting import clv
from wcpredictor.config import Paths

ROOT = Path(__file__).resolve().parents[1]


def _load_dashboard():
    spec = importlib.util.spec_from_file_location("make_dashboard", ROOT / "scripts" / "make_dashboard.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_clv_primitive():
    assert abs(clv(3.0, 0.4) - 0.2) < 1e-12     # took 3.0 on a 0.4 fair shot -> +20% EV
    assert clv(2.0, 0.4) < 0                     # took too short a price -> negative CLV


def test_clv_scoreboard_beats_the_close(tmp_path):
    md = _load_dashboard()
    hist = tmp_path / "odds_history.csv"
    # One fixture, two snapshots: the model's home pick shortens 3.0 -> 2.4, so the
    # opening price we'd have taken beat the close.
    hist.write_text(
        "fetched_at,date,home_team_id,away_team_id,odds_home,odds_draw,odds_away\n"
        "2026-06-01T00:00:00+00:00,2026-06-10,AAA,BBB,3.0,3.5,4.0\n"
        "2026-06-09T00:00:00+00:00,2026-06-10,AAA,BBB,2.4,3.5,4.0\n", encoding="utf-8")
    preds = [{"date": "2026-06-10", "home_team_id": "AAA", "away_team_id": "BBB",
              "p_home": "0.7", "p_draw": "0.15", "p_away": "0.15", "actual_outcome": "home"}]

    res = md._clv_scoreboard(Paths(data_dir=tmp_path), preds, 1.0)
    assert res["n_pairs"] == 1 and len(res["bets"]) == 1
    b = res["bets"][0]
    assert b["sel"] == 0 and b["won"] is True
    assert b["clv"] > 0                          # beat the close


def test_clv_scoreboard_needs_two_snapshots(tmp_path):
    md = _load_dashboard()
    hist = tmp_path / "odds_history.csv"
    hist.write_text(
        "fetched_at,date,home_team_id,away_team_id,odds_home,odds_draw,odds_away\n"
        "2026-06-01T00:00:00+00:00,2026-06-10,AAA,BBB,3.0,3.5,4.0\n", encoding="utf-8")
    preds = [{"date": "2026-06-10", "home_team_id": "AAA", "away_team_id": "BBB",
              "p_home": "0.7", "p_draw": "0.15", "p_away": "0.15", "actual_outcome": "home"}]
    res = md._clv_scoreboard(Paths(data_dir=tmp_path), preds, 1.0)
    assert res["n_pairs"] == 0 and res["bets"] == []
