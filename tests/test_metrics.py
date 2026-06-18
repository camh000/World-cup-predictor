import math

from wcpredictor.metrics import brier_score, log_loss, outcome_index


def test_outcome_index():
    assert outcome_index(2, 1) == 0
    assert outcome_index(1, 1) == 1
    assert outcome_index(0, 2) == 2


def test_log_loss_perfect_prediction():
    ll = log_loss([(1.0, 0.0, 0.0)], [0])
    assert math.isclose(ll, 0.0, abs_tol=1e-9)


def test_log_loss_known_value():
    # Uniform forecast => -log(1/3) per sample.
    ll = log_loss([(1 / 3, 1 / 3, 1 / 3)], [1])
    assert math.isclose(ll, math.log(3), abs_tol=1e-9)


def test_brier_known_value():
    # Forecast (0.5,0.3,0.2), outcome home => (0.5-1)^2+0.3^2+0.2^2 = 0.38
    bs = brier_score([(0.5, 0.3, 0.2)], [0])
    assert math.isclose(bs, 0.38, abs_tol=1e-9)


def test_confident_wrong_penalised_more_than_uncertain():
    confident_wrong = log_loss([(0.01, 0.01, 0.98)], [0])
    uncertain = log_loss([(1 / 3, 1 / 3, 1 / 3)], [0])
    assert confident_wrong > uncertain
