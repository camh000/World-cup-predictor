import math

from wcpredictor.config import Params
from wcpredictor.data_io import MatchRecord
from wcpredictor.history import (
    append_prediction,
    append_ratings_snapshot,
    build_record,
    forecast,
    next_seq,
    read_predictions,
    summarize,
)
from wcpredictor.learn import apply_result
from wcpredictor.ratings import Rating, RatingStore


def _store():
    return RatingStore({"AAA": Rating(elo=1900), "BBB": Rating(elo=1600)})


def test_forecast_probs_sum_to_one():
    (probs, lams) = forecast(_store(), Params(), "AAA", "BBB")
    assert math.isclose(sum(probs), 1.0, abs_tol=1e-9)
    assert lams[0] > lams[1]  # stronger home team expected to score more


def test_build_record_captures_pre_and_post():
    params = Params()
    pre = _store()
    post = pre.copy()
    rec = MatchRecord("2026-06-11", "AAA", "BBB", 2, 0)
    deltas = apply_result(post, params, rec)
    pr = build_record(1, pre, post, params, rec, deltas)

    assert pr.home_elo_pre == round(pre.elo("AAA"), 1)
    assert pr.home_elo_post == round(post.elo("AAA"), 1)
    assert pr.actual_outcome == "home"
    assert pr.predicted_outcome == "home"   # strong favourite won
    assert pr.home_delta > 0 and pr.away_delta < 0
    assert pr.log_loss >= 0


def test_append_and_read_roundtrip(tmp_path):
    path = tmp_path / "predictions.csv"
    assert next_seq(path) == 1
    params = Params()
    pre = _store()
    post = pre.copy()
    rec = MatchRecord("2026-06-11", "AAA", "BBB", 0, 1)
    deltas = apply_result(post, params, rec)
    append_prediction(path, build_record(1, pre, post, params, rec, deltas))

    rows = read_predictions(path)
    assert len(rows) == 1
    assert rows[0]["home_team_id"] == "AAA"
    assert rows[0]["actual_outcome"] == "away"
    assert next_seq(path) == 2


def test_ratings_snapshot_has_row_per_team(tmp_path):
    path = tmp_path / "ratings_history.csv"
    store = _store()
    rec = MatchRecord("2026-06-11", "AAA", "BBB", 1, 1)
    append_ratings_snapshot(path, 1, rec, store)
    lines = path.read_text().strip().splitlines()
    assert lines[0].startswith("seq,date")
    assert len(lines) == 1 + len(store)  # header + one row per team


def test_summarize_skill_positive_for_good_calls():
    rows = [
        {"seq": "1", "log_loss": "0.3", "brier": "0.2",
         "predicted_outcome": "home", "actual_outcome": "home"},
        {"seq": "2", "log_loss": "0.4", "brier": "0.25",
         "predicted_outcome": "away", "actual_outcome": "away"},
    ]
    s = summarize(rows, recent=1)
    assert s.n == 2
    assert s.hit_rate == 1.0
    assert s.skill > 0     # well below the log(3) baseline
    assert s.recent_log_loss == 0.4


def test_summarize_empty_returns_none():
    assert summarize([]) is None
