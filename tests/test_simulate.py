import math

from wcpredictor.simulate import run_simulation


def test_champion_probs_sum_to_one(teams, params, ratings):
    df = run_simulation(teams, params, ratings, n_sims=300, seed=1)
    assert math.isclose(df["p_champion"].sum(), 1.0, abs_tol=1e-9)


def test_reproducible_same_seed(teams, params, ratings):
    a = run_simulation(teams, params, ratings, n_sims=200, seed=5)
    b = run_simulation(teams, params, ratings, n_sims=200, seed=5)
    assert a.equals(b)


def test_probabilities_are_monotone_by_stage(teams, params, ratings):
    df = run_simulation(teams, params, ratings, n_sims=300, seed=2)
    # A team can't reach the final more often than it reaches the semis, etc.
    assert (df["p_advance"] >= df["p_r16"] - 1e-9).all()
    assert (df["p_r16"] >= df["p_qf"] - 1e-9).all()
    assert (df["p_sf"] >= df["p_final"] - 1e-9).all()
    assert (df["p_final"] >= df["p_champion"] - 1e-9).all()


def test_stronger_team_wins_more(teams, params, ratings):
    df = run_simulation(teams, params, ratings, n_sims=500, seed=10)
    row_arg = df[df["team_id"] == "ARG"].iloc[0]
    row_nzl = df[df["team_id"] == "NZL"].iloc[0]
    assert row_arg["p_champion"] > row_nzl["p_champion"]
