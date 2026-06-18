import math

import numpy as np

from wcpredictor.config import Params
from wcpredictor.elo import expected_score
from wcpredictor.poisson import (
    dixon_coles_matrix,
    expected_goals,
    match_probabilities,
    simulate_scoreline,
)


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


def test_dixon_coles_matrix_normalised():
    m = dixon_coles_matrix(1.4, 1.1, rho=-0.12, max_goals=10)
    assert math.isclose(m.sum(), 1.0, abs_tol=1e-9)
    assert (m >= 0).all()


def test_rho_zero_matches_independent_poisson():
    # rho=0 must reproduce the plain independent-Poisson probabilities exactly.
    with_default = match_probabilities(1.5, 1.2, 10, rho=0.0)
    independent = match_probabilities(1.5, 1.2, 10)
    assert with_default == independent


def test_negative_rho_increases_draw_probability():
    _, draw_indep, _ = match_probabilities(1.3, 1.2, 10, rho=0.0)
    _, draw_dc, _ = match_probabilities(1.3, 1.2, 10, rho=-0.15)
    assert draw_dc > draw_indep


def test_dc_probabilities_sum_to_one():
    probs = match_probabilities(*expected_goals(150, Params()), Params().max_goals, rho=-0.1)
    assert math.isclose(sum(probs), 1.0, abs_tol=1e-9)


def test_scoreline_with_rho_reproducible():
    lam_h, lam_a = expected_goals(40, Params())
    a = simulate_scoreline(lam_h, lam_a, np.random.default_rng(3), rho=-0.12, max_goals=10)
    b = simulate_scoreline(lam_h, lam_a, np.random.default_rng(3), rho=-0.12, max_goals=10)
    assert a == b


def test_sampled_scorelines_match_exact_distribution():
    # The (fast rejection) sampler must reproduce the exact Dixon-Coles outcome
    # probabilities from match_probabilities, within Monte-Carlo error.
    lam_h, lam_a, rho = 1.4, 1.2, -0.13
    exact = match_probabilities(lam_h, lam_a, 10, rho)
    rng = np.random.default_rng(0)
    n = 60000
    counts = [0, 0, 0]
    for _ in range(n):
        i, j = simulate_scoreline(lam_h, lam_a, rng, rho=rho)
        counts[0 if i > j else 1 if i == j else 2] += 1
    for emp, ex in zip((c / n for c in counts), exact):
        assert abs(emp - ex) < 0.012


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
