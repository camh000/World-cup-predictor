"""Once the group stage is complete, every team must resolve to QUALIFIED/OUT
(no lingering '>99%'): top two per group + the eight best third-placed teams.
scripts/make_dashboard.py is not a package, so load it by path.
"""

import importlib.util
from pathlib import Path

from wcpredictor.tournament import TeamStanding

ROOT = Path(__file__).resolve().parents[1]


def _load_dashboard():
    spec = importlib.util.spec_from_file_location("make_dashboard", ROOT / "scripts" / "make_dashboard.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_completed_groups_resolve_to_qualified_or_out():
    md = _load_dashboard()
    # 10 finished groups; each third has points 3 with a distinct GD (= i), so the
    # best-third ranking is unambiguous: the 8 highest-GD thirds go through.
    base = {}
    for i in range(10):
        g = chr(ord("A") + i)
        base[g] = {
            f"{g}1": TeamStanding(f"{g}1", played=3, points=9, gf=6, ga=0, elo=1500, group=g),
            f"{g}2": TeamStanding(f"{g}2", played=3, points=6, gf=5, ga=2, elo=1500, group=g),
            f"{g}3": TeamStanding(f"{g}3", played=3, points=3, gf=3 + i, ga=3, elo=1500, group=g),  # gd=i
            f"{g}4": TeamStanding(f"{g}4", played=3, points=0, gf=0, ga=6, elo=1500, group=g),
        }
    status = md._clinch_status(base, {}, set())

    assert all(v in ("QUALIFIED", "OUT") for v in status.values())   # nothing left probabilistic
    for g in base:
        assert status[f"{g}1"] == "QUALIFIED" and status[f"{g}2"] == "QUALIFIED"
        assert status[f"{g}4"] == "OUT"
    assert sum(1 for g in base if status[f"{g}3"] == "QUALIFIED") == 8   # exactly 8 best thirds
    assert status["A3"] == "OUT" and status["B3"] == "OUT"              # the two lowest-GD thirds


def test_incomplete_group_still_uses_conservative_clinch():
    md = _load_dashboard()
    # A group mid-stage (one game unplayed) must NOT be force-resolved.
    g = "A"
    base = {g: {
        "A1": TeamStanding("A1", played=2, points=6, gf=4, ga=0, elo=1500, group=g),
        "A2": TeamStanding("A2", played=2, points=3, gf=2, ga=2, elo=1500, group=g),
        "A3": TeamStanding("A3", played=2, points=3, gf=2, ga=2, elo=1500, group=g),
        "A4": TeamStanding("A4", played=2, points=0, gf=0, ga=4, elo=1500, group=g),
    }}
    played = {frozenset(("A1", "A2")), frozenset(("A1", "A3")), frozenset(("A2", "A4")),
              frozenset(("A3", "A4")), frozenset(("A1", "A4"))}   # A2-A3 still to play
    status = md._clinch_status(base, {"A1": 1.0, "A2": 0.5, "A3": 0.5, "A4": 0.0}, played)
    assert status["A1"] == "QUALIFIED"        # 6 pts with one game left, unreachable by 2 others
    assert None in status.values()            # the A2/A3 race is not yet decided
