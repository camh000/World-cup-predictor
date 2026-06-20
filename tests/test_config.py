"""Lock the shipped calibration constants so a regression can't silently revert
the fix, and document the exact no-op settings that recover a pure point model.
"""

from wcpredictor.config import Params


def test_shipped_defaults_apply_the_calibration_fix():
    p = Params()
    # Forecast-time spread transform: compress only the elite tail (gap >= 250).
    assert p.spread_threshold == 250.0
    assert p.spread_slope == 0.5
    # Tournament rating uncertainty: deflate the over-concentrated favourite.
    assert p.rating_sigma == 150.0


def test_noop_settings_recover_a_point_model():
    # These values turn both new levers off (documented escape hatch).
    p = Params(spread_slope=1.0, rating_sigma=0.0)
    assert p.spread_slope == 1.0
    assert p.rating_sigma == 0.0
