"""Integrity guards for the expanded historical_matches.csv (scripts/fetch_history.py).

The fetcher pulls recent real internationals for retune training, but must NOT
leak the 2026 World Cup results that live in data/results.csv (that would make the
held-out backtest circular). These tests lock that invariant.
"""

from pathlib import Path

from wcpredictor.data_io import read_matches

ROOT = Path(__file__).resolve().parents[1]


def test_historical_does_not_leak_2026_results():
    hist = {(m.date, m.home_team_id, m.away_team_id)
            for m in read_matches(ROOT / "data" / "historical_matches.csv")}
    res = {(m.date, m.home_team_id, m.away_team_id)
           for m in read_matches(ROOT / "data" / "results.csv")}
    assert hist.isdisjoint(res), f"historical leaks 2026 results: {sorted(hist & res)}"


def test_historical_has_recent_internationals():
    # The WC2018/2022 seed history is all pre-2023; the fetcher adds ~397 recent
    # non-WC internationals (2023+). Confirm a healthy number landed.
    ms = read_matches(ROOT / "data" / "historical_matches.csv")
    recent = [m for m in ms if m.date >= "2023-01-01"]
    assert len(recent) >= 200
