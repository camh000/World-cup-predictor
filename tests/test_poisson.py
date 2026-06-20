import math

import numpy as np

from wcpredictor.config import Params
from wcpredictor.elo import expected_score
from wcpredictor.poisson import (
    _spread_transform,
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


def test_spread_transform_noop_at_slope_one():
    # slope == 1.0 must be an EXACT identity at every gap (incl. above the knee),
    # so the default-config model is unchanged.
    p = Params(spread_slope=1.0, spread_threshold=150.0)
    for d in (-303.0, -150.0, -10.0, 0.0, 10.0, 120.0, 150.0, 226.5, 303.0, 500.0):
        assert _spread_transform(d, p) == d


def test_spread_transform_identity_below_knee():
    # At any slope, gaps within +-threshold pass through unchanged.
    p = Params(spread_slope=0.5, spread_threshold=150.0)
    for d in (0.0, 50.0, -120.0, 150.0, -150.0):
        assert _spread_transform(d, p) == d


def test_spread_transform_compresses_excess():
    p = Params(spread_slope=0.5, spread_threshold=150.0)
    assert math.isclose(_spread_transform(303.0, p), 150.0 + (303.0 - 150.0) * 0.5)  # 226.5
    assert math.isclose(_spread_transform(303.0, p), 226.5)
    # Compression shrinks the magnitude but never past the threshold.
    assert 150.0 < _spread_transform(303.0, p) < 303.0


def test_spread_transform_odd_symmetry():
    p = Params(spread_slope=0.5, spread_threshold=150.0)
    for d in (10.0, 150.0, 226.5, 303.0, 480.0):
        assert math.isclose(_spread_transform(-d, p), -_spread_transform(d, p))


def test_spread_transform_monotonic_and_continuous():
    p = Params(spread_slope=0.5, spread_threshold=150.0)
    xs = [0, 50, 100, 149, 150, 151, 200, 303, 500]
    effs = [_spread_transform(float(x), p) for x in xs]
    assert all(b > a for a, b in zip(effs, effs[1:]))      # strictly increasing
    # continuous across the kink at the threshold
    assert math.isclose(_spread_transform(150.0, p), 150.0)
    assert math.isclose(_spread_transform(150.0001, p), 150.0, abs_tol=1e-3)


def test_spread_transform_reduces_favourite_probability():
    # The whole point: compressing the elite tail lowers the favourite's win prob
    # on a big mismatch, while an at-knee gap is untouched.
    noop = Params(spread_slope=1.0, spread_threshold=150.0)
    active = Params(spread_slope=0.5, spread_threshold=150.0)
    ph_noop = match_probabilities(*expected_goals(303.0, noop), noop.max_goals, noop.dc_rho)[0]
    ph_active = match_probabilities(*expected_goals(303.0, active), active.max_goals, active.dc_rho)[0]
    assert ph_active < ph_noop                              # elite favourite compressed
    # A gap at the knee is identical under both.
    knee_noop = match_probabilities(*expected_goals(150.0, noop), noop.max_goals, noop.dc_rho)
    knee_active = match_probabilities(*expected_goals(150.0, active), active.max_goals, active.dc_rho)
    assert knee_noop == knee_active


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
