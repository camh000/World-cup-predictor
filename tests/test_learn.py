import numpy as np

from wcpredictor.config import Params
from wcpredictor.data_io import MatchRecord
from wcpredictor.learn import apply_result, backtest, retune
from wcpredictor.ratings import Rating, RatingStore


def _store():
    return RatingStore({"AAA": Rating(elo=1800), "BBB": Rating(elo=1600)})


def test_apply_result_updates_winner_up_loser_down():
    store = _store()
    params = Params()
    rec = MatchRecord("2026-01-01", "AAA", "BBB", 2, 0)
    dh, da = apply_result(store, params, rec)
    assert dh > 0 and da < 0
    assert store.elo("AAA") > 1800 and store.elo("BBB") < 1600


def test_apply_result_adds_unknown_team():
    store = _store()
    rec = MatchRecord("2026-01-01", "AAA", "ZZZ", 1, 1)
    apply_result(store, Params(), rec)
    assert "ZZZ" in store


def test_backtest_returns_finite_score():
    store = _store()
    matches = [
        MatchRecord("2026-01-01", "AAA", "BBB", 2, 0),
        MatchRecord("2026-01-05", "BBB", "AAA", 1, 1),
        MatchRecord("2026-01-09", "AAA", "BBB", 0, 1),
    ]
    score = backtest(matches, store, Params(), metric="logloss")
    assert np.isfinite(score)


def _synthetic_matches(true_mu: float, n: int = 500):
    """Generate neutral games from a known goal model so retune can recover it."""
    rng = np.random.default_rng(0)
    teams = {f"T{i}": Rating(elo=1500 + 60 * i) for i in range(6)}
    store = RatingStore(teams)
    gen_params = Params(mu=true_mu, k_factor=0.0)  # static ratings
    matches = []
    ids = list(teams)
    from wcpredictor.poisson import expected_goals
    for d in range(n):
        h, a = rng.choice(ids, size=2, replace=False)
        diff = store.elo(h) - store.elo(a)
        lam_h, lam_a = expected_goals(diff, gen_params)
        hg, ag = int(rng.poisson(lam_h)), int(rng.poisson(lam_a))
        matches.append(MatchRecord(f"2025-{d % 12 + 1:02d}-01", h, a, hg, ag, neutral=True))
    return store.copy(), matches


def test_retune_improves_or_holds_score():
    initial, matches = _synthetic_matches(true_mu=1.6)
    start = Params(mu=1.0, k_factor=10.0)
    result = retune(matches, initial, start, metric="logloss", method="nelder-mead")
    # Retune must never make the model worse than where it started.
    assert result.score_after <= result.score_before + 1e-9


def test_retune_recovers_goal_level():
    # Data generated with a high goal level (mu=1.9); starting low, retune should
    # move mu upward toward the truth.
    initial, matches = _synthetic_matches(true_mu=1.9)
    start = Params(mu=1.1)
    result = retune(matches, initial, start, metric="logloss", method="nelder-mead")
    assert result.params.mu > start.mu + 0.2
