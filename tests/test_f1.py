from wcpredictor.f1 import (
    Race, RaceRow, constructor_of, current_grid, next_race_probs,
    rate_drivers, simulate_championship, standings,
)


def _race(track, order, teams):
    rows = [RaceRow(d, teams[d], i + 1, 0.0) for i, d in enumerate(order)]
    return Race(track, rows)


def test_constructor_strips_engine_suffix():
    assert constructor_of("McLaren Mercedes") == "McLaren"
    assert constructor_of("Red Bull Racing Red Bull Ford") == "Red Bull Racing"
    assert constructor_of("Racing Bulls Red Bull Ford") == "Racing Bulls"
    assert constructor_of("Mercedes") == "Mercedes"
    assert constructor_of("Alpine Renault") == constructor_of("Alpine Mercedes") == "Alpine"


def test_rating_orders_consistent_winner_above_loser():
    teams = {"A": "McLaren Mercedes", "B": "Ferrari", "C": "Haas Ferrari"}
    # A always beats B beats C across several races.
    season = [_race(f"R{i}", ["A", "B", "C"], teams) for i in range(6)]
    elo = rate_drivers([season])
    assert elo["A"] > elo["B"] > elo["C"]


def test_simulate_championship_favours_the_stronger_leader():
    teams = {"A": "McLaren Mercedes", "B": "Ferrari"}
    season = [_race(f"R{i}", ["A", "B"], teams) for i in range(8)]
    elo = rate_drivers([season])
    drivers, team_of = current_grid(season)
    dt, ct, proj = simulate_championship(
        drivers, team_of, elo, {"A": 100.0, "B": 50.0}, {"McLaren": 100.0, "Ferrari": 50.0},
        remaining_races=10, n_sims=2000, scale=110.0)
    assert dt["A"] > dt["B"]
    assert abs(sum(dt.values()) - 1.0) < 1e-9
    assert ct["McLaren"] > ct["Ferrari"]


def test_next_race_probs_sum_to_one_for_winner():
    teams = {"A": "McLaren Mercedes", "B": "Ferrari", "C": "Haas Ferrari"}
    season = [_race(f"R{i}", ["A", "B", "C"], teams) for i in range(5)]
    elo = rate_drivers([season])
    drivers, _ = current_grid(season)
    probs = next_race_probs(drivers, elo, n_sims=3000)
    assert abs(sum(p[0] for p in probs.values()) - 1.0) < 1e-9   # one winner per race
    assert probs["A"][0] > probs["C"][0]                          # stronger wins more
