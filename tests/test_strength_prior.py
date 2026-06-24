"""Guards for the FIFA strength-prior data + fold-in (scripts are not a package,
so load the modules by path). No network in the tests.
"""

import csv
import importlib.util
from pathlib import Path

from wcpredictor.data_io import _norm_name, build_name_index, read_teams

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _teams():
    return read_teams(ROOT / "data" / "teams.csv")


def test_strength_prior_covers_every_team():
    teams = _teams()
    wc_ids = {t.team_id for t in teams}
    rows = list(csv.DictReader((ROOT / "data" / "strength_prior.csv").open(encoding="utf-8")))
    covered = {r["team_id"] for r in rows}
    assert covered == wc_ids, f"missing: {wc_ids - covered}, extra: {covered - wc_ids}"
    for r in rows:
        assert int(r["fifa_rank"]) > 0
        assert float(r["fifa_points"]) > 0


def test_tricky_fifa_spellings_resolve():
    # The fetcher's coverage depends on these FIFA spellings mapping to our ids.
    idx = build_name_index(_teams())
    expect = {"USA": "USA", "Korea Republic": "KOR", "IR Iran": "IRN",
              "Cabo Verde": "CPV", "Congo DR": "COD"}
    for fifa_name, tid in expect.items():
        assert idx.get(_norm_name(fifa_name)) == tid, f"{fifa_name} -> {idx.get(_norm_name(fifa_name))}, want {tid}"


def test_blend_weight_zero_is_noop():
    rs = _load("respread_seeds")
    rows = rs._read_rows()
    before = {r["team_id"]: float(r["elo"]) for r in rows}
    rs.blend_strength(rows, 0.0)
    after = {r["team_id"]: float(r["elo"]) for r in rows}
    assert after == before


def test_blend_weight_one_equals_rescaled_fifa():
    rs = _load("respread_seeds")
    rows = rs._read_rows()
    seed_elo = {r["team_id"]: float(r["elo"]) for r in rows}
    fifa_elo = rs._fifa_elo_map(seed_elo)
    rs.blend_strength(rows, 1.0)
    for r in rows:
        if r["team_id"] in fifa_elo:
            assert float(r["elo"]) == round(fifa_elo[r["team_id"]], 0)


def test_blend_is_monotonic_toward_fifa():
    rs = _load("respread_seeds")
    seed_elo = {r["team_id"]: float(r["elo"]) for r in rs._read_rows()}
    fifa_elo = rs._fifa_elo_map(seed_elo)
    # A team FIFA rates clearly higher than its seed (e.g. Mexico).
    tid = max(fifa_elo, key=lambda t: fifa_elo[t] - seed_elo[t])
    vals = []
    for w in (0.0, 0.25, 0.5, 1.0):
        rows = rs._read_rows()
        rs.blend_strength(rows, w)
        vals.append(next(float(r["elo"]) for r in rows if r["team_id"] == tid))
    assert vals[0] <= vals[1] <= vals[2] <= vals[3]   # rises monotonically toward FIFA
    assert vals[3] > vals[0]
