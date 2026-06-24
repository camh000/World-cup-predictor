#!/usr/bin/env python3
"""Adjust the committed seed ratings from external priors.

Two independent, opt-in operations (preview by default; ``--apply`` writes
data/seed_ratings.csv):

``--strength [W]`` — fold the FIFA strength prior (data/strength_prior.csv, from
    scripts/fetch_strength_prior.py) into the seeds: rescale FIFA points onto the
    field's seed mean/std, then ``seed' = (1-W)*seed + W*fifa_elo``. The default
    W=0.25 was chosen on the walk-forward backtest AND the prior-vs-market KL
    (scripts/validate_strength_prior.py): a modest blend improves both, peaking
    around 0.2–0.3 and degrading past 0.5, so we under-weight to a conservative
    0.25. FIFA's post-2018 ranking is itself an Elo-style system, so the points
    sit on a sensible strength axis. Don't fabricate — refresh the CSV first.

``--minnows`` — the one-off, already-applied targeted correction of a few
    implausibly-high tiny-nation seeds (Curacao etc.), kept here for provenance
    and reproducibility. The values are absolute market-implied Elos.

Usage:
    python scripts/fetch_strength_prior.py          # refresh data/strength_prior.csv
    python scripts/respread_seeds.py --strength      # preview the 0.25 blend
    python scripts/respread_seeds.py --strength 0.25 --apply
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = ROOT / "data" / "seed_ratings.csv"
STRENGTH = ROOT / "data" / "strength_prior.csv"

DEFAULT_STRENGTH_WEIGHT = 0.25

# Historical one-off minnow correction (already baked into the committed CSV);
# absolute market-implied Elos, kept for provenance. See git history / PR #3.
CORRECTIONS = {"CUW": 1400, "HAI": 1520, "NZL": 1600, "CPV": 1660, "IRQ": 1580}


def _read_rows():
    return list(csv.DictReader(SEEDS.open(encoding="utf-8")))


def _fifa_elo_map(seed_elo: dict) -> dict:
    """Rescale FIFA points onto the seed Elo scale (match mean & std over shared teams)."""
    fifa = {}
    with STRENGTH.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            fifa[r["team_id"]] = float(r["fifa_points"])
    common = [t for t in seed_elo if t in fifa]
    if not common:
        raise SystemExit("error: no overlap between seeds and strength_prior.csv")
    s_mu, s_sd = statistics.mean(seed_elo[t] for t in common), statistics.pstdev([seed_elo[t] for t in common])
    f_mu, f_sd = statistics.mean(fifa[t] for t in common), statistics.pstdev([fifa[t] for t in common])
    f_sd = f_sd or 1.0
    return {t: s_mu + s_sd * (fifa[t] - f_mu) / f_sd for t in common}


def blend_strength(rows, w: float):
    """Blend the FIFA prior into ``rows`` in place; return list of (team, old, new)."""
    seed_elo = {r["team_id"]: float(r["elo"]) for r in rows}
    fifa_elo = _fifa_elo_map(seed_elo)
    changed = []
    for r in rows:
        tid = r["team_id"]
        if tid in fifa_elo:
            old = float(r["elo"])
            new = round((1.0 - w) * old + w * fifa_elo[tid], 0)
            if new != old:
                changed.append((tid, old, new))
            r["elo"] = f"{new:.0f}"
    return changed


def apply_minnows(rows):
    changed = []
    for r in rows:
        if r["team_id"] in CORRECTIONS:
            old, new = float(r["elo"]), float(CORRECTIONS[r["team_id"]])
            if new != old:
                changed.append((r["team_id"], old, new))
            r["elo"] = f"{new:.0f}"
    return changed


def _write(rows):
    with SEEDS.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["team_id", "elo", "attack", "defense"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in ("team_id", "elo", "attack", "defense")})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strength", nargs="?", type=float, const=DEFAULT_STRENGTH_WEIGHT,
                    default=None, metavar="W", help="blend the FIFA prior at weight W (default 0.25)")
    ap.add_argument("--minnows", action="store_true", help="apply the historical minnow correction")
    ap.add_argument("--apply", action="store_true", help="rewrite data/seed_ratings.csv")
    args = ap.parse_args()
    if args.strength is None and not args.minnows:
        ap.error("choose an operation: --strength [W] and/or --minnows")

    rows = _read_rows()
    changed = []
    if args.minnows:
        changed += apply_minnows(rows)
    if args.strength is not None:
        if not (0.0 <= args.strength <= 1.0):
            ap.error("--strength W must be in [0, 1]")
        changed += blend_strength(rows, args.strength)

    changed.sort(key=lambda c: abs(c[2] - c[1]), reverse=True)
    print(f"{'team':<6}{'old':>7}{'new':>7}{'delta':>7}")
    for tid, old, new in changed:
        print(f"{tid:<6}{old:>7.0f}{new:>7.0f}{new-old:>+7.0f}")
    print(f"\n{len(changed)} seeds changed.")
    if not args.apply:
        print("Preview only. Re-run with --apply to write data/seed_ratings.csv.")
        return
    _write(rows)
    print(f"Wrote {SEEDS}")


if __name__ == "__main__":
    main()
