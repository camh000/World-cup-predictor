import math

import numpy as np

from wcpredictor.config import Params
from wcpredictor.elo import expected_score
from wcpredictor.poisson import expected_goals, match_probabilities, simulate_scoreline


def test_expected_goals_monotonic_in_rating_diff():
    p = Params()
    lo_h, lo_a = expected_goals(-100, p)
    eq_h, eq_a = expected_goals(0, p)
    hi_h, hi_a = expected_goals(100, p)
    assert lo_h < eq_h < hi_h          # home goals rise with home strength
    assert lo_a > eq_a > hi_a          # away goals fall


def test_equal_strength_symmetric():
    p = Params()
    lam_h, lam_a = expected_goals(0, p)
    assert math.isclose(lam_h, lam_a)
    assert math.isclose(lam_h, p.mu)


def test_probabilities_sum_to_one():
    p = Params()
    probs = match_probabilities(*expected_goals(150, p), p.max_goals)
    assert math.isclose(sum(probs), 1.0, abs_tol=1e-9)


def test_scoreline_reproducible_with_seed():
    p = Params()
    lam_h, lam_a = expected_goals(80, p)
    a = simulate_scoreline(lam_h, lam_a, np.random.default_rng(7))
    b = simulate_scoreline(lam_h, lam_a, np.random.default_rng(7))
    assert a == b


def test_empirical_winrate_matches_elo_logistic():
    # Consistency: the Poisson model's home-win share (excluding draws) should be
    # close to the Elo logistic expectation for a given rating gap.
    p = Params()
    diff = 120.0
    lam_h, lam_a = expected_goals(diff, p)
    p_home, p_draw, p_away = match_probabilities(lam_h, lam_a, p.max_goals)
    poisson_winrate = p_home / (p_home + p_away)
    elo_winrate = expected_score(diff, 0, p.elo_divisor)
    # Loose tolerance — beta is only roughly calibrated by default, exact match
    # is the job of `retune`.
    assert abs(poisson_winrate - elo_winrate) < 0.15
