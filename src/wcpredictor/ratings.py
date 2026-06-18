"""Persisted, evolving team ratings.

``RatingStore`` is the mutable model state: a mapping of team id -> Elo (plus
optional attack/defense offsets). It is seeded from ``data/seed_ratings.csv``
(or a flat fallback) and saved to ``state/ratings.json``. The online learner
mutates it after every real game.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, Tuple

from .config import _atomic_write_json


DEFAULT_ELO = 1500.0

# Small confederation offsets used only when no seed ratings exist at all, so a
# cold-start model is not completely uniform. The online learner corrects these.
CONFED_OFFSET = {
    "UEFA": 80.0,
    "CONMEBOL": 90.0,
    "CONCACAF": -10.0,
    "CAF": 10.0,
    "AFC": -20.0,
    "OFC": -60.0,
}


@dataclass
class Rating:
    elo: float = DEFAULT_ELO
    attack: float = 0.0
    defense: float = 0.0
    # Tournament "form": a fast, mean-reverting overlay on the slow Elo baseline.
    # Starts at 1.0 each tournament and is nudged by how much a team over- or
    # under-performs its Elo expectation in this tournament's games. Applied as a
    # multiplier on expected goals. 1.0 == no effect.
    form: float = 1.0


class RatingStore:
    """Dict-like container of :class:`Rating` keyed by team id."""

    def __init__(self, ratings: Dict[str, Rating] | None = None):
        self._r: Dict[str, Rating] = dict(ratings or {})

    # --- mapping helpers ---
    def __getitem__(self, team_id: str) -> Rating:
        return self._r[team_id]

    def __setitem__(self, team_id: str, rating: Rating) -> None:
        self._r[team_id] = rating

    def __contains__(self, team_id: object) -> bool:
        return team_id in self._r

    def __len__(self) -> int:
        return len(self._r)

    def __iter__(self) -> Iterator[str]:
        return iter(self._r)

    def items(self) -> Iterable[Tuple[str, Rating]]:
        return self._r.items()

    def elo(self, team_id: str) -> float:
        return self._r[team_id].elo

    def copy(self) -> "RatingStore":
        return RatingStore({tid: Rating(**asdict(r)) for tid, r in self._r.items()})

    def total_elo(self) -> float:
        return sum(r.elo for r in self._r.values())

    def ranked(self):
        """Teams sorted by descending Elo as ``[(team_id, Rating), ...]``."""
        return sorted(self._r.items(), key=lambda kv: kv[1].elo, reverse=True)

    # --- persistence ---
    def save(self, path: Path) -> None:
        payload = {tid: asdict(r) for tid, r in self._r.items()}
        _atomic_write_json(Path(path), payload)

    @classmethod
    def load(cls, path: Path) -> "RatingStore":
        with Path(path).open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls({tid: Rating(**vals) for tid, vals in data.items()})

    # --- seeding ---
    @classmethod
    def seed(cls, teams, seed_ratings: Dict[str, Rating] | None = None) -> "RatingStore":
        """Build a fresh store for ``teams`` from seed ratings or a fallback.

        ``teams`` is an iterable of objects with ``team_id`` and ``confederation``
        attributes. Any team missing from ``seed_ratings`` falls back to a flat
        baseline plus a small confederation offset.
        """
        seed_ratings = seed_ratings or {}
        out: Dict[str, Rating] = {}
        for t in teams:
            if t.team_id in seed_ratings:
                out[t.team_id] = Rating(**asdict(seed_ratings[t.team_id]))
            else:
                offset = CONFED_OFFSET.get(getattr(t, "confederation", ""), 0.0)
                out[t.team_id] = Rating(elo=DEFAULT_ELO + offset)
        return cls(out)
