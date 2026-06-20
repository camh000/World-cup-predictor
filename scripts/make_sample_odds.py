#!/usr/bin/env python3
"""Generate a SYNTHETIC bookmaker-odds file for the betting backtest demo.

This does NOT contain real bookmaker prices. It models an *efficient market*
that is exactly as sharp as our own model (fair probs = the model's probs) and
then adds a realistic ~6.5% bookmaker margin.

Why no noise? If we perturbed the model's probs to invent a "market", the model
would have a rigged structural advantage over its own noisy copy and the
backtest would print fake profit on our tiny sample — the classic way people
fool themselves with backtests. Keeping the market exactly as sharp as the model
gives the honest answer: the margin alone locks you out unless you are genuinely
*better* than the market.

To run a REAL test, replace data/odds.csv with genuine closing decimal odds in
the same columns: date,home_team_id,away_team_id,odds_home,odds_draw,odds_away
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "data" / "predictions.csv"
OUT = ROOT / "data" / "odds.csv"

MARGIN = 0.065        # ~6.5% overround, typical for 1X2 markets


def main() -> None:
    rows = []
    with PRED.open("r", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            p = [float(r["p_home"]), float(r["p_draw"]), float(r["p_away"])]
            # Efficient market: fair probs == model probs, then apply the margin.
            odds = [round(1.0 / (pi * (1.0 + MARGIN)), 2) for pi in p]
            rows.append([r["date"], r["home_team_id"], r["away_team_id"], *odds])

    with OUT.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "home_team_id", "away_team_id",
                    "odds_home", "odds_draw", "odds_away"])
        w.writerows(rows)
    print(f"Wrote {OUT} ({len(rows)} synthetic efficient-market prices)")


if __name__ == "__main__":
    main()
