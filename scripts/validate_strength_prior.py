#!/usr/bin/env python3
"""Validate folding the FIFA strength prior into the seeds, BEFORE adopting it.

Builds candidate seeds for a range of blend weights ``w`` (seed_new = (1-w)*seed
+ w*fifa_elo, where FIFA points are rescaled to the WC field's seed mean/std),
then scores each on:

  * the walk-forward backtest over data/results.csv (log-loss / Brier) -- the
    primary "does it predict real games better" test; and
  * mean KL(market || model) over the live data/odds.csv fixtures, forecast from
    the candidate seeds (a leak-free prior-vs-market sanity check).

w=0.0 is the current seeds (the baseline to beat). Read-only; writes nothing.
"""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path

from wcpredictor.betting import devig
from wcpredictor.config import Params, Paths
from wcpredictor.data_io import read_matches, read_seed_ratings, read_teams, HOST_TEAM_IDS
from wcpredictor.history import forecast
from wcpredictor.learn import backtest
from wcpredictor.ratings import Rating, RatingStore

WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]


def fifa_to_elo(seeds, fifa_pts):
    """Rescale FIFA points onto the seed Elo scale (match mean & std over shared teams)."""
    common = [t for t in seeds if t in fifa_pts]
    s_vals = [seeds[t] for t in common]
    f_vals = [fifa_pts[t] for t in common]
    s_mu, s_sd = statistics.mean(s_vals), statistics.pstdev(s_vals)
    f_mu, f_sd = statistics.mean(f_vals), statistics.pstdev(f_vals)
    f_sd = f_sd or 1.0
    return {t: s_mu + s_sd * (fifa_pts[t] - f_mu) / f_sd for t in common}


def blended_store(seeds, fifa_elo, w):
    out = {}
    for tid, elo in seeds.items():
        e = (1.0 - w) * elo + w * fifa_elo[tid] if tid in fifa_elo else elo
        out[tid] = Rating(elo=e)
    return RatingStore(out)


def market_kl(store, params, odds_rows):
    kl = 0.0
    for h, a, odds in odds_rows:
        neutral = h not in HOST_TEAM_IDS
        probs, _ = forecast(store, params, h, a, neutral)
        fair = devig(odds)
        kl += sum(fair[i] * math.log(fair[i] / max(probs[i], 1e-12)) for i in range(3) if fair[i] > 0)
    return kl / len(odds_rows) if odds_rows else float("nan")


def main() -> None:
    paths = Paths()
    teams = read_teams(paths.teams_csv)
    params = Params.load(paths.params_json)   # shipped config (spread/sigma/etc.)
    seeds = {t: r.elo for t, r in read_seed_ratings(paths.seed_ratings_csv).items()}

    fifa_pts = {}
    with (paths.data_dir / "strength_prior.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            fifa_pts[r["team_id"]] = float(r["fifa_points"])
    fifa_elo = fifa_to_elo(seeds, fifa_pts)

    odds_rows = []
    op = paths.data_dir / "odds.csv"
    if op.exists():
        for r in csv.DictReader(op.open(encoding="utf-8")):
            odds_rows.append((r["home_team_id"], r["away_team_id"],
                              (float(r["odds_home"]), float(r["odds_draw"]), float(r["odds_away"]))))

    matches = read_matches(paths.results_csv)
    print(f"Backtest over {len(matches)} results | prior-market over {len(odds_rows)} priced fixtures\n")
    print(f"{'w':>5}{'logloss':>10}{'brier':>9}{'mkt_KL':>9}   notes")
    print("-" * 50)
    base_ll = None
    for w in WEIGHTS:
        store = blended_store(seeds, fifa_elo, w)
        ll = backtest(matches, store, params, "logloss")
        br = backtest(matches, store, params, "brier")
        kl = market_kl(store, params, odds_rows)
        if base_ll is None:
            base_ll = ll
        tag = "current seeds (baseline)" if w == 0.0 else f"{ll-base_ll:+.4f} vs baseline"
        print(f"{w:>5.1f}{ll:>10.4f}{br:>9.4f}{kl:>9.4f}   {tag}")


if __name__ == "__main__":
    main()
