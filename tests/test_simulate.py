import math

from wcpredictor.config import Params
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


def test_rating_sigma_reproducible(teams, ratings):
    a = run_simulation(teams, Params(rating_sigma=80.0), ratings, n_sims=300, seed=3)
    b = run_simulation(teams, Params(rating_sigma=80.0), ratings, n_sims=300, seed=3)
    assert a.equals(b)


def test_rating_sigma_zero_is_sharper_than_default(teams, ratings):
    # The shipped default applies tournament uncertainty, so disabling it
    # (sigma=0) leaves the favourite MORE concentrated than the default.
    sharp = run_simulation(teams, Params(rating_sigma=0.0), ratings, n_sims=1500, seed=4)
    default = run_simulation(teams, Params(), ratings, n_sims=1500, seed=4)
    fav = sharp.iloc[0]["team_id"]
    assert (default[default["team_id"] == fav]["p_champion"].iloc[0]
            < sharp[sharp["team_id"] == fav]["p_champion"].iloc[0])


def test_rating_sigma_spreads_title_odds(teams, ratings):
    # Tournament rating uncertainty must deflate the favourite's concentration
    # while keeping a valid distribution.
    sharp = run_simulation(teams, Params(rating_sigma=0.0), ratings, n_sims=2000, seed=7)
    fuzzy = run_simulation(teams, Params(rating_sigma=150.0), ratings, n_sims=2000, seed=7)
    fav = sharp.iloc[0]["team_id"]
    p_sharp = sharp[sharp["team_id"] == fav]["p_champion"].iloc[0]
    p_fuzzy = fuzzy[fuzzy["team_id"] == fav]["p_champion"].iloc[0]
    assert p_fuzzy < p_sharp
    assert math.isclose(fuzzy["p_champion"].sum(), 1.0, abs_tol=1e-9)
