#!/usr/bin/env python3
"""Does more seasons of history improve the 2026 F1 race backtest?

Pulls full race history from f1db (the maintained Ergast-successor database, via
GitHub releases) and walk-forwards over the 2026 races, rating drivers on varying
depths of prior history (2026-only -> +2025 -> +2024 -> +2023 -> +2022). Scores
the actual winner (and podium) log-loss. If deeper history lowers the loss, the
multi-season upgrade is worth wiring in; if not, the current 2025+2026 base is
fine and we don't bloat the pipeline. Read-only validation.
"""

from __future__ import annotations

import io
import math
import random
import urllib.request
import zipfile
from collections import defaultdict

from wcpredictor.f1 import Race, RaceRow, _finishing_order, rate_drivers, simulate_race

URL = "https://github.com/f1db/f1db/releases/latest/download/f1db-csv.zip"
N_SIMS = 6000
SCALE, DNF = 110.0, 0.18


def load_f1db_by_year():
    import csv
    raw = urllib.request.urlopen(
        urllib.request.Request(URL, headers={"User-Agent": "wcpredictor/1.0"}), timeout=120).read()
    z = zipfile.ZipFile(io.BytesIO(raw))
    rows = list(csv.DictReader(io.TextIOWrapper(z.open("f1db-races-race-results.csv"), "utf-8")))
    by_year = defaultdict(lambda: defaultdict(list))   # year -> round -> [RaceRow]
    for r in rows:
        y, rnd = r["year"], r["round"]
        pos = r["positionNumber"]
        try:
            p = int(pos)
        except ValueError:
            p = None
        try:
            pts = float(r["points"] or 0)
        except ValueError:
            pts = 0.0
        by_year[y][rnd].append(RaceRow(r["driverId"], r["constructorId"], p, pts))
    seasons = {}
    for y, rounds in by_year.items():
        seasons[y] = [Race(f"{y}-{k}", rounds[k]) for k in sorted(rounds, key=int)]
    return seasons


def winner_podium_ll(prior_seasons, races_2026):
    wlls, plls = [], []
    for i, race in enumerate(races_2026):
        order = _finishing_order(race)
        if len(order) < 3:
            continue
        elo = rate_drivers(prior_seasons + [races_2026[:i]])
        drivers = [r.driver for r in race.rows]
        rng = random.Random(100 + i)
        win = dict.fromkeys(drivers, 0); pod = dict.fromkeys(drivers, 0)
        for _ in range(N_SIMS):
            o = simulate_race(drivers, elo, rng, scale=SCALE, dnf_prob=DNF)
            if o:
                win[o[0]] += 1
            for d in o[:3]:
                pod[d] += 1
        wlls.append(-math.log(max(win[order[0]] / N_SIMS, 1e-6)))
        plls += [-math.log(max(pod[d] / N_SIMS, 1e-6)) for d in order[:3]]
    return sum(wlls) / len(wlls), sum(plls) / len(plls)


def main() -> None:
    seasons = load_f1db_by_year()
    r26 = seasons.get("2026", [])
    print(f"f1db seasons present: {sorted(seasons)[-6:]} | 2026 races: {len(r26)}\n")
    depths = [
        ("2026 only", []),
        ("+2025", ["2025"]),
        ("+2024", ["2024", "2025"]),
        ("+2023", ["2023", "2024", "2025"]),
        ("+2022", ["2022", "2023", "2024", "2025"]),
    ]
    print(f"{'history':<12}{'winnerLL':>10}{'podiumLL':>10}")
    print("-" * 32)
    for label, yrs in depths:
        prior = [seasons[y] for y in yrs if y in seasons]
        wll, pll = winner_podium_ll(prior, r26)
        print(f"{label:<12}{wll:>10.4f}{pll:>10.4f}")


if __name__ == "__main__":
    main()
