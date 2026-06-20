from wcpredictor.calibration import fit_sharpness, sharpen


def test_sharpen_identity_and_monotone():
    p = (0.5, 0.3, 0.2)
    assert sharpen(p, 1.0) == p
    sharp = sharpen(p, 2.0)
    assert abs(sum(sharp) - 1.0) < 1e-12
    assert sharp[0] > p[0]      # favourite gains
    assert sharp[2] < p[2]      # longshot loses
    flat = sharpen(p, 0.5)
    assert flat[0] < p[0]       # gamma<1 flattens toward uniform


def test_fit_recovers_sharpening_when_model_is_underconfident():
    # Model says 60/25/15 but the favourite actually wins ~85% of the time:
    # the model is under-confident, so the fit should sharpen (gamma > 1).
    probs = [(0.6, 0.25, 0.15)] * 100
    outcomes = [0] * 85 + [1] * 8 + [2] * 7
    gamma = fit_sharpness(probs, outcomes)
    assert gamma > 1.0


def test_fit_empty_is_identity():
    assert fit_sharpness([], []) == 1.0
