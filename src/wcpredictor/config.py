"""Configuration: model hyperparameters and filesystem paths.

``Params`` holds the tunable hyperparameters of the model. They are persisted to
``state/params.json`` and re-loaded on every run; if the file is missing the
baked-in defaults below are used. ``retune`` (see :mod:`wcpredictor.learn`)
optimises a subset of these against accumulated real results.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field, fields
from pathlib import Path
from typing import Dict


# Hyperparameters that ``retune`` is allowed to optimise. Deliberately limited to
# the goal-distribution shape, which match data genuinely informs. home_advantage
# is excluded (tournament data is mostly neutral, so there's no signal) and
# k_factor is excluded (minimising in-sample loss just rewards not adapting, which
# would defeat the online learning) -- both are design choices kept at sane defaults.
TUNABLE_FIELDS = ("beta", "mu", "dc_rho")


@dataclass
class Params:
    """Model hyperparameters.

    Attributes
    ----------
    elo_divisor:
        The classic Elo logistic divisor (400 => a 400-point gap is ~10:1 odds).
    home_advantage:
        Elo points added to a host/home team. Group/knockout games at the World
        Cup are mostly neutral, so this only applies when a team is flagged host.
    k_factor:
        Base Elo update step size. Larger => ratings react faster to results.
    beta:
        "goal_scale" — links the Elo gap to expected goals. Tuned so the Poisson
        model's win probabilities agree with the Elo logistic.
    mu:
        Baseline expected goals per team for an evenly-matched game (~1.35 for
        international football).
    penalty_elo_weight:
        How much Elo difference tilts a penalty shootout away from a coin flip.
    max_goals:
        Truncation for the Poisson probability grid used by ``match_probabilities``.
    points_win / points_draw:
        Group-stage points.
    """

    elo_divisor: float = 400.0
    home_advantage: float = 55.0
    k_factor: float = 40.0
    beta: float = 1.0
    mu: float = 1.35
    penalty_elo_weight: float = 0.5
    max_goals: int = 10
    points_win: int = 3
    points_draw: int = 1
    # Dixon-Coles low-score dependency parameter. The independent-Poisson model
    # under-predicts draws; a negative rho shifts probability mass from 1-0/0-1
    # into 0-0/1-1, correcting that. 0 disables the correction. Default is the
    # literature value (Dixon & Coles 1997, ~-0.13); a sweep on the draw-heavy
    # matchday 1 prefers a larger magnitude, but that overfits one round -- tune
    # via `retune` once more results exist (see TODO.md).
    dc_rho: float = -0.13
    # Tournament form overlay (see Rating.form). form_alpha is the learning rate
    # per game (0 disables form entirely); form_decay pulls form back toward 1.0
    # between games (mean reversion); form_min/max bound the multiplier.
    form_alpha: float = 0.10
    form_decay: float = 0.85
    form_min: float = 0.85
    form_max: float = 1.15
    # Forecast-time spread compression (see poisson._spread_transform). A
    # FORECAST-ONLY, non-linear lens on the effective Elo gap: gaps up to
    # spread_threshold pass through unchanged, and only the *excess* above it is
    # scaled by spread_slope. This deflates the elite over-confidence the raw
    # eloratings.net scale produces (e.g. ARG vs a mid team) without disturbing
    # mid-tier favourites, and -- because tournament title odds are ~p**5 convex
    # in the per-match edge -- pulls top-heavy outright odds toward the market.
    # It never touches the learned Elo (update_elo stays on raw ratings), so it is
    # a reversible lens. spread_slope == 1.0 is an EXACT no-op at any threshold.
    # Shipped values chosen by the prior-calibration grid (scripts/validate_prior.py):
    # T=250 touches only the genuinely over-confident elite mismatches (gap >=250)
    # and leaves every mid-tier gap untouched; s=0.5 under-shoots (closes ~60% of
    # the elite excess). Revert to a pure point model with spread_slope=1.0.
    spread_threshold: float = 250.0
    spread_slope: float = 0.5
    # Tournament-level rating uncertainty (Elo std-dev), used ONLY by the Monte
    # Carlo champion/advance simulation (simulate_tournament). Each simulated
    # tournament draws every team's effective Elo once from N(elo, rating_sigma),
    # representing the irreducible "true strength this tournament" uncertainty
    # (injuries, form, squad depth) that a fixed point-estimate cannot express.
    # Because the title map is ~p**5 convex in the per-match edge, this widens the
    # outcome distribution and deflates an over-concentrated favourite toward the
    # market WITHOUT changing any per-match 1X2 forecast (forecast paths never call
    # the tournament sim). 0.0 is an exact no-op. The shipped 150.0 is calibrated
    # (scripts/decompose_outright.py) so the title-odds spread matches the large
    # irreducible uncertainty the market prices: it pulls a clear #1 from ~31% to
    # ~17% (within the bookmakers' ~10-14% band) while keeping the Elo ordering.
    rating_sigma: float = 150.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "Params":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def copy_with(self, **overrides) -> "Params":
        data = self.to_dict()
        data.update(overrides)
        return Params.from_dict(data)

    def save(self, path: os.PathLike | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: os.PathLike | str) -> "Params":
        path = Path(path)
        if not path.exists():
            return cls()
        with path.open("r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


def _repo_root() -> Path:
    # src/wcpredictor/config.py -> parents[2] == repo root
    return Path(__file__).resolve().parents[2]


@dataclass
class Paths:
    """Resolved locations for canonical input data and evolving model state."""

    data_dir: Path = field(default_factory=lambda: _repo_root() / "data")
    state_dir: Path = field(default_factory=lambda: _repo_root() / "state")

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.state_dir = Path(self.state_dir)

    # --- canonical inputs (committed) ---
    @property
    def teams_csv(self) -> Path:
        return self.data_dir / "teams.csv"

    @property
    def seed_ratings_csv(self) -> Path:
        return self.data_dir / "seed_ratings.csv"

    @property
    def results_csv(self) -> Path:
        return self.data_dir / "results.csv"

    @property
    def historical_csv(self) -> Path:
        return self.data_dir / "historical_matches.csv"

    @property
    def predictions_csv(self) -> Path:
        return self.data_dir / "predictions.csv"

    @property
    def ratings_history_csv(self) -> Path:
        return self.data_dir / "ratings_history.csv"

    # --- evolving state (git-ignored) ---
    @property
    def ratings_json(self) -> Path:
        return self.state_dir / "ratings.json"

    @property
    def params_json(self) -> Path:
        return self.state_dir / "params.json"

    def ensure_state_dir(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)


def _atomic_write_json(path: Path, payload) -> None:
    """Write JSON to a temp file then rename, so a crash never corrupts state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)
