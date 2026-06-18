import numpy as np

from wcpredictor.match import simulate_match


def test_group_match_can_draw(params, tiny_ratings):
    saw_draw = False
    for s in range(200):
        res = simulate_match("AAA", "BBB", params, tiny_ratings,
                             np.random.default_rng(s), knockout=False)
        if res.is_draw:
            saw_draw = True
            break
    assert saw_draw


def test_knockout_never_draws(params, tiny_ratings):
    for s in range(200):
        res = simulate_match("AAA", "BBB", params, tiny_ratings,
                             np.random.default_rng(s), knockout=True)
        assert res.winner in ("AAA", "BBB")
        assert not res.is_draw


def test_knockout_winner_is_a_participant(params, tiny_ratings, rng):
    res = simulate_match("CCC", "DDD", params, tiny_ratings, rng, knockout=True)
    assert res.winner in ("CCC", "DDD")


def test_match_reproducible_with_seed(params, tiny_ratings):
    a = simulate_match("AAA", "DDD", params, tiny_ratings,
                       np.random.default_rng(99), knockout=True)
    b = simulate_match("AAA", "DDD", params, tiny_ratings,
                       np.random.default_rng(99), knockout=True)
    assert (a.home_goals, a.away_goals, a.winner) == (b.home_goals, b.away_goals, b.winner)


def test_penalties_and_extra_time_labelled(params, tiny_ratings):
    decided = set()
    for s in range(400):
        res = simulate_match("AAA", "BBB", params, tiny_ratings,
                             np.random.default_rng(s), knockout=True)
        decided.add(res.decided_by)
    # Over many seeds we should see at least some games go beyond regulation.
    assert decided & {"extra_time", "penalties"}
