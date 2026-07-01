"""The knockout panel is built from REAL data (played results + priced upcoming
ties), not the model's mis-allocated bracket. scripts/make_dashboard.py isn't a
package, so load it by path.
"""

import importlib.util
from pathlib import Path

from wcpredictor.config import Params, Paths
from wcpredictor.ratings import Rating, RatingStore

ROOT = Path(__file__).resolve().parents[1]

RES_HEADER = "date,home_team_id,away_team_id,home_goals,away_goals,stage,competition,neutral\n"
ODDS_HEADER = "date,home_team_id,away_team_id,odds_home,odds_draw,odds_away\n"


def _load_dashboard():
    spec = importlib.util.spec_from_file_location("make_dashboard", ROOT / "scripts" / "make_dashboard.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _ratings():
    return RatingStore({t: Rating(elo=1500.0) for t in ("GER", "PAR", "FRA", "MEX")})


NAME = {"GER": "Germany", "PAR": "Paraguay", "FRA": "France", "MEX": "Mexico"}


def test_knockout_panel_shows_real_result_and_upcoming(tmp_path):
    md = _load_dashboard()
    (tmp_path / "results.csv").write_text(
        RES_HEADER + "2026-06-29,GER,PAR,5,6,round of 32,WC,true\n", encoding="utf-8")
    (tmp_path / "odds.csv").write_text(
        ODDS_HEADER + "2026-07-02,FRA,MEX,1.5,4.0,6.0\n", encoding="utf-8")
    html = md._knockout_bracket(Paths(data_dir=tmp_path), {}, Params(), _ratings(), NAME)
    assert "KNOCKOUT STAGE" in html
    # the REAL tie + result, with the side that went through highlighted (Paraguay won 6-5)
    assert "Germany" in html and "Paraguay" in html and ">5<" in html and ">6<" in html
    assert "Results so far" in html
    # the actual upcoming tie is shown as a prediction, not a reconstructed matchup
    assert "Coming up" in html and "France" in html and "Mexico" in html


def test_knockout_panel_empty_during_group_stage(tmp_path):
    md = _load_dashboard()
    (tmp_path / "results.csv").write_text(RES_HEADER, encoding="utf-8")   # no knockout games
    (tmp_path / "odds.csv").write_text(ODDS_HEADER, encoding="utf-8")
    # base empty -> group stage not complete -> panel hidden
    assert md._knockout_bracket(Paths(data_dir=tmp_path), {}, Params(), _ratings(), NAME) == ""
