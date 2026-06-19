import numpy as np

from wcpredictor.tournament import (
    N_BEST_THIRDS,
    R32_2026,
    R32_TEMPLATE,
    TeamStanding,
    _assign_thirds,
    best_third_placed,
    simulate_tournament,
)


def test_r32_2026_consumes_all_slots():
    winners, runners, thirds = set(), set(), 0
    for a, b in R32_2026:
        for kind, key in (a, b):
            if kind == "W":
                winners.add(key)
            elif kind == "RU":
                runners.add(key)
            else:
                thirds += 1
    assert len(R32_2026) == 16
    assert winners == set("ABCDEFGHIJKL")   # all 12 group winners
    assert runners == set("ABCDEFGHIJKL")   # all 12 runners-up
    assert thirds == N_BEST_THIRDS          # 8 third-placed slots


def test_third_assignment_respects_allowed_groups():
    slots = [frozenset("ABCDF"), frozenset("CDFGH"), frozenset("CEFHI")]
    third_groups = ["A", "G", "E"]          # exactly one valid slot each
    assign = _assign_thirds(slots, third_groups)
    assert assign[0] == 0   # only A-slot can take group A
    assert third_groups[assign[1]] == "G"   # G only allowed by slot 1
    assert third_groups[assign[2]] == "E"
    assert sorted(assign) == [0, 1, 2]      # a perfect matching


def test_r32_template_consumes_all_slots():
    # Exactly 12 winners, 12 runners-up and 8 thirds = 32 distinct slots.
    winners, runners, thirds = set(), set(), set()
    for a, b in R32_TEMPLATE:
        for kind, key in (a, b):
            {"W": winners, "RU": runners, "T": thirds}[kind].add(key)
    assert len(R32_TEMPLATE) == 16
    assert len(winners) == 12
    assert len(runners) == 12
    assert thirds == set(range(N_BEST_THIRDS))


def test_best_thirds_selects_top_n():
    thirds = [TeamStanding(f"T{i}", points=i, gf=i) for i in range(12)]
    chosen = best_third_placed(thirds, N_BEST_THIRDS)
    assert len(chosen) == N_BEST_THIRDS
    # Highest-points teams chosen; T0 (0 points) excluded.
    chosen_ids = {s.team_id for s in chosen}
    assert "T11" in chosen_ids and "T0" not in chosen_ids


def test_standings_tiebreak_order():
    a = TeamStanding("A", points=6, gf=5, ga=2, elo=1500)  # gd +3
    b = TeamStanding("B", points=6, gf=4, ga=2, elo=1500)  # gd +2
    c = TeamStanding("C", points=6, gf=5, ga=3, elo=1500)  # gd +2, gf 5
    ordered = sorted([b, c, a], key=lambda s: s.sort_key())
    assert [s.team_id for s in ordered] == ["A", "C", "B"]


def test_tournament_produces_single_champion(teams, params, ratings, rng):
    result = simulate_tournament(teams, params, ratings, rng)
    assert result.champion in {t.team_id for t in teams}
    assert result.reached[result.champion] == "champion"


def test_tournament_has_twelve_group_winners(teams, params, ratings, rng):
    result = simulate_tournament(teams, params, ratings, rng)
    assert len(result.group_winners) == 12


def test_exactly_32_teams_reach_knockout(teams, params, ratings, rng):
    result = simulate_tournament(teams, params, ratings, rng)
    knockout_stages = {"R32", "R16", "QF", "SF", "F", "champion"}
    advanced = [tid for tid, stage in result.reached.items() if stage in knockout_stages]
    assert len(advanced) == 32


def test_tournament_reproducible(teams, params, ratings):
    r1 = simulate_tournament(teams, params, ratings, np.random.default_rng(3))
    r2 = simulate_tournament(teams, params, ratings, np.random.default_rng(3))
    assert r1.champion == r2.champion
    assert r1.reached == r2.reached
