"""A small Formula 1 model: driver+car ratings, race & championship simulation.

Mirrors the football side in spirit. We rate each *driver* (the rating absorbs
car performance) with a pairwise Elo over finishing orders, season by season, then
simulate races as a Plackett-Luce process (Gumbel-perturbed ratings give a random
finishing order) to get win/podium probabilities and a Monte-Carlo championship.

Data: the CSVs fetched by scripts/fetch_f1.py (toUpperCase78/formula1-datasets).
Team strings carry the engine ("McLaren Mercedes"), so we normalise to the
constructor for the constructors' championship.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# 2026 points: top 10 finishers; sprints: top 8. (No fastest-lap point since 2025.)
RACE_POINTS = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
SPRINT_POINTS = [8, 7, 6, 5, 4, 3, 2, 1]

# Known 2026 constructors (longest/unique prefixes of the engine-suffixed strings).
CONSTRUCTORS = [
    "Red Bull Racing", "Racing Bulls", "Aston Martin", "McLaren", "Mercedes",
    "Ferrari", "Williams", "Alpine", "Haas", "Cadillac", "Audi", "Kick Sauber",
]


def constructor_of(team: str) -> str:
    """Normalise an engine-suffixed team string to its constructor name."""
    team = team.strip()
    for c in CONSTRUCTORS:
        if team.startswith(c):
            return c
    return team.split(" ")[0] if team else team


@dataclass
class RaceRow:
    driver: str
    team: str
    position: Optional[int]   # None = did not classify (DNF)
    points: float


@dataclass
class Race:
    track: str
    rows: List[RaceRow]


def load_races(path: Path) -> List[Race]:
    """Parse a season's race-results CSV into chronological races."""
    if not path.exists():
        return []
    races: Dict[str, Race] = {}
    order: List[str] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            track = r["Track"].strip()
            if track not in races:
                races[track] = Race(track, [])
                order.append(track)
            pos_raw = r["Position"].strip()
            try:
                pos: Optional[int] = int(pos_raw)
            except ValueError:
                pos = None  # "NC"
            try:
                pts = float(r["Points"])
            except (ValueError, KeyError):
                pts = 0.0
            races[track].rows.append(
                RaceRow(r["Driver"].strip(), r["Team"].strip(), pos, pts))
    return [races[t] for t in order]


def _finishing_order(race: Race) -> List[str]:
    """Drivers best→worst: classified by position, then DNFs at the back."""
    classified = sorted((r for r in race.rows if r.position is not None),
                        key=lambda r: r.position)
    dnf = [r for r in race.rows if r.position is None]
    return [r.driver for r in classified] + [r.driver for r in dnf]


def rate_drivers(seasons: Sequence[List[Race]], *, k: float = 32.0,
                 divisor: float = 400.0, base: float = 1500.0) -> Dict[str, float]:
    """Pairwise-Elo driver ratings over the given seasons (oldest first).

    Each race is a round-robin: every driver who finished ahead of another beats
    them. The per-pair K is divided by (n-1) so a driver's total move per race is
    ~K regardless of field size.
    """
    elo: Dict[str, float] = defaultdict(lambda: base)
    for season in seasons:
        for race in season:
            order = _finishing_order(race)
            n = len(order)
            if n < 2:
                continue
            kk = k / (n - 1)
            deltas: Dict[str, float] = defaultdict(float)
            for i in range(n):
                for j in range(i + 1, n):
                    win, lose = order[i], order[j]
                    exp_win = 1.0 / (1.0 + 10 ** ((elo[lose] - elo[win]) / divisor))
                    deltas[win] += kk * (1.0 - exp_win)
                    deltas[lose] -= kk * (1.0 - exp_win)
            for d, dd in deltas.items():
                elo[d] += dd
    return dict(elo)


def current_grid(races_2026: List[Race]) -> Tuple[List[str], Dict[str, str]]:
    """Active drivers and their latest constructor from the 2026 results."""
    team_of: Dict[str, str] = {}
    for race in races_2026:                      # later races overwrite earlier
        for r in race.rows:
            team_of[r.driver] = constructor_of(r.team)
    return list(team_of), team_of


def standings(seasons_results: Sequence[List[Race]],
              sprint_rows: Sequence[Tuple[str, str, float]] = ()) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Driver and constructor points from race rows (+ optional sprint rows)."""
    drv: Dict[str, float] = defaultdict(float)
    con: Dict[str, float] = defaultdict(float)
    for season in seasons_results:
        for race in season:
            for r in race.rows:
                drv[r.driver] += r.points
                con[constructor_of(r.team)] += r.points
    for driver, team, pts in sprint_rows:
        drv[driver] += pts
        con[constructor_of(team)] += pts
    return dict(drv), dict(con)


def _gumbel(rng) -> float:
    u = rng.random()
    return -math.log(-math.log(max(u, 1e-12)))


def simulate_race(drivers: Sequence[str], elo: Dict[str, float], rng, *,
                  scale: float = 110.0, dnf_prob: float = 0.12, base: float = 1500.0
                  ) -> List[str]:
    """One race: Gumbel-perturbed ratings → finishing order (DNFs at the back)."""
    classified, retired = [], []
    for d in drivers:
        if rng.random() < dnf_prob:
            retired.append(d)
        else:
            classified.append((elo.get(d, base) + scale * _gumbel(rng), d))
    classified.sort(reverse=True)
    return [d for _, d in classified] + retired


def simulate_championship(
    drivers: Sequence[str], team_of: Dict[str, str], elo: Dict[str, float],
    driver_points: Dict[str, float], constructor_points: Dict[str, float],
    remaining_races: int, *, remaining_sprints: int = 0, n_sims: int = 20000,
    seed: int = 42, scale: float = 110.0, dnf_prob: float = 0.12,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Monte-Carlo the remaining races (and any remaining sprints).

    ``remaining_sprints`` adds that many sprint races (top-8 ``SPRINT_POINTS``)
    on top of the ``remaining_races`` grands prix, so the title projection counts
    the sprint points still on offer. 0 = no sprints (back-compatible).

    Returns (driver champion probs, constructor champion probs, mean projected
    driver points).
    """
    import random

    rng = random.Random(seed)
    drv_titles: Dict[str, float] = defaultdict(float)
    con_titles: Dict[str, float] = defaultdict(float)
    proj_points: Dict[str, float] = defaultdict(float)

    for _ in range(n_sims):
        dp = dict(driver_points)
        cp = dict(constructor_points)
        for d in drivers:
            dp.setdefault(d, 0.0)
            cp.setdefault(team_of[d], 0.0)
        for _r in range(remaining_races):
            order = simulate_race(drivers, elo, rng, scale=scale, dnf_prob=dnf_prob)
            for pos, d in enumerate(order[:10]):
                dp[d] += RACE_POINTS[pos]
                cp[team_of[d]] += RACE_POINTS[pos]
        for _s in range(remaining_sprints):
            order = simulate_race(drivers, elo, rng, scale=scale, dnf_prob=dnf_prob)
            for pos, d in enumerate(order[:len(SPRINT_POINTS)]):
                dp[d] += SPRINT_POINTS[pos]
                cp[team_of[d]] += SPRINT_POINTS[pos]
        drv_titles[max(dp, key=dp.get)] += 1
        con_titles[max(cp, key=cp.get)] += 1
        for d in drivers:
            proj_points[d] += dp[d]

    inv = 1.0 / n_sims
    return ({d: drv_titles.get(d, 0.0) * inv for d in drivers},
            {t: con_titles.get(t, 0.0) * inv for t in set(team_of.values())},
            {d: proj_points[d] * inv for d in drivers})


def next_race_probs(drivers: Sequence[str], elo: Dict[str, float], *,
                    n_sims: int = 20000, seed: int = 7, scale: float = 110.0,
                    dnf_prob: float = 0.12) -> Dict[str, Tuple[float, float, float]]:
    """Per-driver (P(win), P(podium), P(points)) for a single upcoming race."""
    import random

    rng = random.Random(seed)
    win: Dict[str, float] = defaultdict(float)
    pod: Dict[str, float] = defaultdict(float)
    pts: Dict[str, float] = defaultdict(float)
    for _ in range(n_sims):
        order = simulate_race(drivers, elo, rng, scale=scale, dnf_prob=dnf_prob)
        if order:
            win[order[0]] += 1
        for d in order[:3]:
            pod[d] += 1
        for d in order[:10]:
            pts[d] += 1
    inv = 1.0 / n_sims
    return {d: (win.get(d, 0.0) * inv, pod.get(d, 0.0) * inv, pts.get(d, 0.0) * inv)
            for d in drivers}
