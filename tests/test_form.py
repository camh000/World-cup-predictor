import math

from wcpredictor.config import Params
from wcpredictor.data_io import MatchRecord
from wcpredictor.learn import _clip_form, apply_result
from wcpredictor.poisson import expected_goals
from wcpredictor.ratings import Rating, RatingStore


def _store():
    return RatingStore({"STR": Rating(elo=1900), "WEAK": Rating(elo=1600)})


def test_form_starts_at_one():
    assert Rating().form == 1.0


def test_overperformer_form_rises_underperformer_falls():
    store = _store()
    params = Params()
    # Big upset: weak team (home) beats strong team -> weak over-performs.
    apply_result(store, params, MatchRecord("2026-06-11", "WEAK", "STR", 1, 0))
    assert store["WEAK"].form > 1.0
    assert store["STR"].form < 1.0


def test_form_disabled_when_alpha_zero():
    store = _store()
    params = Params(form_alpha=0.0)
    apply_result(store, params, MatchRecord("2026-06-11", "WEAK", "STR", 3, 0))
    assert store["WEAK"].form == 1.0
    assert store["STR"].form == 1.0


def test_form_mean_reverts_toward_one():
    params = Params()
    # With zero surprise, an elevated form is pulled back toward 1.0.
    assert _clip_form(1.10, 0.0, params) < 1.10
    assert _clip_form(1.10, 0.0, params) > 1.0
    # And a depressed form is pushed back up toward 1.0.
    assert _clip_form(0.90, 0.0, params) > 0.90


def test_form_respects_bounds():
    params = Params()
    assert _clip_form(1.0, 100.0, params) == params.form_max
    assert _clip_form(1.0, -100.0, params) == params.form_min


def test_form_multiplier_neutral_when_equal():
    params = Params()
    base = expected_goals(120.0, params)
    with_form = expected_goals(120.0, params, form_home=1.1, form_away=1.1)
    assert math.isclose(base[0], with_form[0], rel_tol=1e-9)
    assert math.isclose(base[1], with_form[1], rel_tol=1e-9)


def test_form_in_form_team_scores_more_concedes_less():
    params = Params()
    lam_h, lam_a = expected_goals(0.0, params, form_home=1.15, form_away=0.9)
    base_h, base_a = expected_goals(0.0, params)
    assert lam_h > base_h    # in-form home scores more
    assert lam_a < base_a    # ...and concedes less


def test_form_persists_through_save_load(tmp_path):
    store = _store()
    store["STR"].form = 1.07
    path = tmp_path / "ratings.json"
    store.save(path)
    loaded = RatingStore.load(path)
    assert math.isclose(loaded["STR"].form, 1.07)
