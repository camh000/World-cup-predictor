import math

from wcpredictor.betting import (
    devig,
    evaluate,
    ev_per_unit,
    kelly_fraction,
    overround,
)


def test_devig_sums_to_one_and_reports_margin():
    odds = (2.0, 4.0, 4.0)  # implied 0.5 + 0.25 + 0.25 = 1.0 -> no margin
    fair = devig(odds)
    assert abs(sum(fair) - 1.0) < 1e-12
    assert abs(overround(odds)) < 1e-12

    odds2 = (1.9, 3.8, 3.8)  # inflated -> positive overround
    assert overround(odds2) > 0
    assert abs(sum(devig(odds2)) - 1.0) < 1e-12


def test_ev_and_kelly():
    # Fair coin priced at 2.10: positive edge.
    assert ev_per_unit(0.5, 2.1) > 0
    assert kelly_fraction(0.5, 2.1) > 0
    # No edge -> no stake.
    assert ev_per_unit(0.5, 1.9) < 0
    assert kelly_fraction(0.5, 1.9) == 0.0
    # Full Kelly for p=0.5, odds=3.0 (b=2): f = (0.5*2 - 0.5)/2 = 0.25
    assert abs(kelly_fraction(0.5, 3.0) - 0.25) < 1e-12


def test_evaluate_wins_when_model_has_real_edge():
    # Model knows home wins 60%; bookie prices home at 2.20 (implied ~45%).
    # Outcomes realise at the true 60% rate -> model should profit.
    probs = (0.6, 0.25, 0.15)
    odds = (2.2, 4.0, 6.0)
    matches = []
    for i in range(100):
        outcome = 0 if i % 5 < 3 else (1 if i % 5 == 3 else 2)  # 60/20/20
        matches.append((probs, odds, outcome))
    res = evaluate(matches)
    assert res.n_bets > 0
    assert res.flat_profit > 0
    assert res.kelly_end > res.kelly_start
    assert res.beats_market  # model closer to the 60/20/20 truth than the de-vigged price


def test_evaluate_empty():
    res = evaluate([])
    assert res.n_matches == 0 and res.n_bets == 0
    assert math.isnan(res.model_log_loss)
