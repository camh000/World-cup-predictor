"""Guards for the odds fetcher's security-critical helpers (no network calls).

scripts/ is not a package, so load the module by path.
"""

import importlib.util
import os
import urllib.error
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("fetch_odds", ROOT / "scripts" / "fetch_odds.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_err_never_leaks_the_api_key_url():
    m = _load()
    leaky = "https://api.the-odds-api.com/v4/sports/x/odds?apiKey=SUPERSECRET&regions=uk"
    e = urllib.error.HTTPError(leaky, 401, "Unauthorized", {}, None)
    msg = m._err(e)
    assert msg == "HTTP 401"
    assert "SUPERSECRET" not in msg and "apiKey" not in msg
    ue = urllib.error.URLError(OSError("connection refused"))
    assert "apiKey" not in m._err(ue) and "SUPERSECRET" not in m._err(ue)


def test_append_writes_header_once_and_never_truncates(tmp_path):
    m = _load()
    p = tmp_path / "hist.csv"
    m._append(p, ["a", "b"], [[1, 2]])
    m._append(p, ["a", "b"], [[3, 4]])
    lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln]
    assert lines[0] == "a,b"            # header written exactly once
    assert lines[1:] == ["1,2", "3,4"]  # both snapshots preserved (append-only)


def test_select_keys_ignores_non_football_world_cups():
    # Regression: 'world_cup' alone wrongly matched cricket_t20_world_cup_womens
    # (sorts before soccer), fetching cricket odds that drop to zero 1X2 rows.
    m = _load()
    sports = [
        {"key": "cricket_t20_world_cup_womens", "active": True},
        {"key": "soccer_fifa_world_cup_qualifiers_conmebol", "active": False},
        {"key": "soccer_fifa_world_cup_winner", "active": True},
        {"key": "soccer_fifa_world_cup", "active": True},
    ]
    assert m._select_keys(sports) == ("soccer_fifa_world_cup", "soccer_fifa_world_cup_winner")
    # Nothing active / nothing FIFA -> no keys.
    assert m._select_keys([{"key": "cricket_t20_world_cup_womens", "active": True}]) == (None, None)
    assert m._select_keys([{"key": "soccer_fifa_world_cup", "active": False}]) == (None, None)


def test_load_dotenv_real_env_wins(tmp_path, monkeypatch):
    m = _load()
    f = tmp_path / "envfile"
    f.write_text('ODDS_API_KEY="from-file"\n', encoding="utf-8")
    monkeypatch.setenv("ODDS_API_KEY", "from-env")
    m._load_dotenv(f)
    assert os.environ["ODDS_API_KEY"] == "from-env"   # setdefault: env beats file
