#!/usr/bin/env python3
"""Correct a handful of implausibly-high minnow seed ratings.

The committed seeds use exact eloratings.net values for the top contenders but
"close approximations" for the rest (see README). A few of those approximations
are just wrong: genuine minnows were seeded ~1650-1740, only ~150 below solid
sides, so the model massively over-rated them (e.g. it gave Curacao ~17% to beat
Ecuador vs the market's ~4%, inventing a +400% "edge").

This applies a SMALL, targeted correction to those specific teams, anchored to
their market-implied Elo (derived from live 1X2 odds) and sanity-checked against
their world standing. It is deliberately NOT a blanket re-spread (the model's
mid-tier is actually too *low*, so a uniform stretch over-corrects) and NOT a
wholesale re-anchor to the market (that would overfit one odds snapshot and turn
the model into a market clone). Only clearly-broken tiny-nation seeds are touched.

Validate with scripts/validate_prior.py: the worst longshot "edges" shrink and the
per-Elo-gap buckets stay near zero (no over-correction into over-rating favourites).

Usage:
    python scripts/respread_seeds.py            # preview
    python scripts/respread_seeds.py --apply     # rewrite data/seed_ratings.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = ROOT / "data" / "seed_ratings.csv"

# team_id -> corrected Elo. Each is a genuine minnow whose approximate seed was
# far above its market-implied strength; the value is its market-implied Elo
# (rounded), which is also consistent with its eloratings.net world standing.
CORRECTIONS = {
    "CUW": 1400,   # Curacao  (was 1660; mkt-implied ~1400)
    "HAI": 1520,   # Haiti    (was 1645; mkt-implied ~1550)
    "NZL": 1600,   # New Zealand (was 1678; mkt-implied ~1620)
    "CPV": 1660,   # Cape Verde  (was 1722; mkt-implied ~1672)
    "IRQ": 1580,   # Iraq     (was 1715; mkt-implied ~1575)
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="rewrite data/seed_ratings.csv")
    args = ap.parse_args()

    rows = list(csv.DictReader(SEEDS.open(encoding="utf-8")))
    print(f"{'team':<6}{'old':>7}{'new':>7}{'delta':>7}")
    n = 0
    for r in rows:
        tid = r["team_id"]
        if tid in CORRECTIONS:
            old = float(r["elo"]); new = float(CORRECTIONS[tid])
            print(f"{tid:<6}{old:>7.0f}{new:>7.0f}{new-old:>+7.0f}")
            r["elo"] = f"{new:.0f}"
            n += 1
    missing = set(CORRECTIONS) - {r["team_id"] for r in rows}
    if missing:
        raise SystemExit(f"error: correction targets not in seeds: {sorted(missing)}")
    print(f"\n{n} minnow seeds corrected; {len(rows)-n} unchanged.")

    if not args.apply:
        print("\nPreview only. Re-run with --apply to write data/seed_ratings.csv.")
        return
    with SEEDS.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["team_id", "elo", "attack", "defense"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in ("team_id", "elo", "attack", "defense")})
    print(f"\nWrote {SEEDS}")


if __name__ == "__main__":
    main()
