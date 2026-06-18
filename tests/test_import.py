"""Tests for importing the official FIFA 2026 schedule CSV."""

from wcpredictor.config import Paths
from wcpredictor.data_io import (
    build_name_index,
    read_official_schedule,
    read_teams,
    write_results,
    read_matches,
)


SAMPLE = """Match Number,Round Number,Date,Location,Home Team,Away Team,Group,Result
1,1,11/06/2026 19:00,Mexico City Stadium,Mexico,South Africa,Group A,2 - 0
2,1,12/06/2026 02:00,Guadalajara Stadium,Korea Republic,Czechia,Group A,2 - 1
6,1,14/06/2026 04:00,BC Place Vancouver,Australia,Türkiye,Group D,2 - 0
9,1,14/06/2026 23:00,Philadelphia Stadium,Côte d'Ivoire,Ecuador,Group E,1 - 0
14,1,15/06/2026 16:00,Atlanta Stadium,Spain,Cabo Verde,Group H,0 - 0
73,Round of 32,28/06/2026 19:00,Stadium,1A,3ABCDF,,
"""


def _teams():
    return read_teams(Paths().teams_csv)


def test_name_aliases_resolve():
    idx = build_name_index(_teams())
    from wcpredictor.data_io import _norm_name
    assert idx[_norm_name("Korea Republic")] == "KOR"
    assert idx[_norm_name("Türkiye")] == "TUR"
    assert idx[_norm_name("Côte d'Ivoire")] == "CIV"
    assert idx[_norm_name("Cabo Verde")] == "CPV"
    assert idx[_norm_name("Congo DR")] == "COD"
    assert idx[_norm_name("IR Iran")] == "IRN"


def test_read_official_schedule_skips_unplayed_and_maps(tmp_path):
    f = tmp_path / "sched.csv"
    f.write_text(SAMPLE, encoding="utf-8")
    recs = read_official_schedule(f, _teams())
    # 5 played rows; the Round-of-32 row has placeholder teams + no result -> skipped.
    assert len(recs) == 5
    by_home = {r.home_team_id: r for r in recs}
    assert by_home["MEX"].away_team_id == "RSA"
    assert by_home["MEX"].date == "2026-06-11"
    assert by_home["AUS"].away_team_id == "TUR"      # Türkiye alias
    assert by_home["CIV"].away_team_id == "ECU"      # Côte d'Ivoire alias


def test_host_games_marked_non_neutral(tmp_path):
    f = tmp_path / "sched.csv"
    f.write_text(SAMPLE, encoding="utf-8")
    recs = {r.home_team_id: r for r in read_official_schedule(f, _teams())}
    assert recs["MEX"].neutral is False    # host at home
    assert recs["KOR"].neutral is True     # non-host


def test_write_results_roundtrip(tmp_path):
    f = tmp_path / "sched.csv"
    f.write_text(SAMPLE, encoding="utf-8")
    recs = read_official_schedule(f, _teams())
    out = tmp_path / "results.csv"
    write_results(out, recs)
    back = read_matches(out)
    assert len(back) == len(recs)
    assert {r.home_team_id for r in back} == {r.home_team_id for r in recs}
