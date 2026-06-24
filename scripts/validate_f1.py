#!/usr/bin/env python3
"""Calibrate the F1 sim's ``scale`` and ``dnf_prob`` on a walk-forward backtest.

For each 2026 race we rate drivers on prior races only (2025 + the 2026 races
before it), simulate the race many times, and score the model on the actual
result. Primary metric: race-winner log-loss (the proper score for "who wins");
secondary: podium log-loss. Lowest is best. Read-only.

The current sim uses scale=110, dnf_prob=0.12 — but the empirical DNF rate is
11.5% in 2025 and 23.4% in 2026 (new-regs attrition), so the baseline likely
under-weights retirements. We tune both, but keep it conservative: only 7 races,
so prefer values grounded in the empirical rate over a sharp in-sample minimum.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

from wcpredictor.f1 import _finishing_order, load_races, rate_drivers, simulate_race

F1 = Path(__file__).resolve().parents[1] / "data" / "f1"
SCALES = [80.0, 110.0, 140.0, 170.0]
DNFS = [0.12, 0.15, 0.18, 0.23]
N_SIMS = 6000


def _race_probs(drivers, elo, scale, dnf, seed):
    rng = random.Random(seed)
    win = dict.fromkeys(drivers, 0)
    pod = dict.fromkeys(drivers, 0)
    for _ in range(N_SIMS):
        order = simulate_race(drivers, elo, rng, scale=scale, dnf_prob=dnf)
        if order:
            win[order[0]] += 1
        for d in order[:3]:
            pod[d] += 1
    return ({d: win[d] / N_SIMS for d in drivers},
            {d: pod[d] / N_SIMS for d in drivers})


def main() -> None:
    r25 = load_races(F1 / "race_results_2025.csv")
    r26 = load_races(F1 / "race_results_2026.csv")

    # Pre-rate (walk-forward, leak-free) and capture each race's field + result.
    cache = []
    for i, race in enumerate(r26):
        order = _finishing_order(race)
        if len(order) < 3:
            continue
        elo = rate_drivers([r25, r26[:i]])   # only races strictly before race i
        cache.append((i, [r.driver for r in race.rows], elo, order))
    print(f"Walk-forward over {len(cache)} of {len(r26)} 2026 races.\n")

    print(f"{'scale':>6}{'dnf':>6}{'winLL':>9}{'podLL':>9}")
    print("-" * 30)
    best = None
    for scale in SCALES:
        for dnf in DNFS:
            wlls, plls = [], []
            for i, drivers, elo, order in cache:
                pw, pp = _race_probs(drivers, elo, scale, dnf, seed=100 + i)
                wlls.append(-math.log(max(pw.get(order[0], 0.0), 1e-6)))
                plls += [-math.log(max(pp.get(d, 0.0), 1e-6)) for d in order[:3]]
            wll, pll = sum(wlls) / len(wlls), sum(plls) / len(plls)
            tag = ""
            if best is None or wll < best[0]:
                best = (wll, scale, dnf); tag = " <-"
            print(f"{scale:>6.0f}{dnf:>6.2f}{wll:>9.4f}{pll:>9.4f}{tag}")
    print(f"\nbaseline (110, 0.12) vs best winner-LL: see column. Best={best[1:]}.")


if __name__ == "__main__":
    main()
