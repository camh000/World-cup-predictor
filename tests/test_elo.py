import math

from wcpredictor.elo import expected_score, mov_multiplier, update_elo


def test_expected_score_symmetry():
    e_home = expected_score(1800, 1600)
    e_away = expected_score(1600, 1800)
    assert math.isclose(e_home + e_away, 1.0, abs_tol=1e-9)


def test_equal_ratings_give_half():
    assert math.isclose(expected_score(1500, 1500), 0.5)


def test_update_is_zero_sum():
    h, a = update_elo(1800, 1600, 2, 1, k_factor=40)
    assert math.isclose((h - 1800) + (a - 1600), 0.0, abs_tol=1e-9)


def test_favorite_gains_less_on_expected_win():
    # Strong team beating weak team gains little; upset gains a lot.
    strong_h, _ = update_elo(1900, 1500, 1, 0, k_factor=40)
    upset_h, _ = update_elo(1500, 1900, 1, 0, k_factor=40)
    assert (strong_h - 1900) < (upset_h - 1500)


def test_draw_of_equals_no_change():
    h, a = update_elo(1700, 1700, 1, 1, k_factor=40)
    assert math.isclose(h, 1700) and math.isclose(a, 1700)


def test_upset_draw_moves_ratings():
    # Weaker home team drawing a much stronger away team should gain Elo.
    h, a = update_elo(1500, 1900, 0, 0, k_factor=40)
    assert h > 1500 and a < 1900


def test_mov_multiplier_increases_with_goal_diff():
    one = mov_multiplier(1, 0)
    two = mov_multiplier(2, 0)
    five = mov_multiplier(5, 0)
    assert one < two < five


def test_mov_dampens_for_strong_favorite():
    # Same 3-goal win, but a bigger favourite gets a smaller multiplier.
    underdog_win = mov_multiplier(3, -200)
    favorite_win = mov_multiplier(3, 200)
    assert favorite_win < underdog_win
