#!/usr/bin/env python3
"""Read-only calibration harness for the forecast-time spread transform.

Selects/validates the spread-transform constants WITHOUT touching any committed
state, and WITHOUT letting the 29 settled 2026 outcomes leak into the choice:

  * the primary gate (GATE 1) is computed from the SEED prior
    (``RatingStore.seed(teams, seed_ratings.csv)`` -- never ``state/ratings.json``,
    which has absorbed the 29 results, and never ``seed(teams, {})``, the flat
    ~1500 footgun) at gamma=1, comparing the model's 1X2 against the de-vigged
    market on the 36 priced fixtures: per-Elo-gap-bucket signed favourite error,
    KL(market||model), and the dashboard's robust surviving-edge count.
  * the corroborating gate (GATE 2) is a leak-free walk-forward log-loss over the
    29 results seeded from the CSV prior, split into favourite-WON vs
    favourite-DREW subsets (the fix must improve the draw subset WITHOUT worsening
    the won subset -- not a mechanically-entailed check).
  * the 128-game historical backtest is reported but DEMOTED: cold-start caps its
    gaps below the T=150 knee, so it is near-constant under the transform and must
    not be used in the flat-optimum argument.

The replay-state outright (champion %, incl. ARG) is DESCRIPTIVE only -- it uses
the replayed ratings (which are identical for every spread value, since the Elo
learner never sees the transform), varying only the forecast lens in the sim.

Run:  python scripts/validate_prior.py
Nothing is written; this only prints tables.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

from wcpredictor.betting import devig, ev_per_unit
from wcpredictor.calibration import blend
from wcpredictor.config import Params, Paths
from wcpredictor.data_io import (
    HOST_TEAM_IDS,
    read_matches,
    read_seed_ratings,
    read_teams,
)
from wcpredictor.history import forecast
from wcpredictor.learn import _chronological, apply_result, backtest
from wcpredictor.metrics import log_loss, outcome_index
from wcpredictor.ratings import RatingStore
from wcpredictor.simulate import run_simulation

GRID_T = [120.0, 150.0, 180.0, 250.0]
GRID_S = [0.4, 0.5, 0.6, 0.7, 1.0]
OUTRIGHT_SIMS = 6000   # modest for the grid; P3 uses the full 20000
OUTRIGHT_SEED = 42


def load_odds(paths: Paths):
    """Return [(home_id, away_id, (oh, od, oa)), ...] from data/odds.csv."""
    rows = []
    with (paths.data_dir / "odds.csv").open("r", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append((r["home_team_id"], r["away_team_id"],
                         (float(r["odds_home"]), float(r["odds_draw"]), float(r["odds_away"]))))
    return rows


def seed_store(teams, paths) -> RatingStore:
    return RatingStore.seed(teams, read_seed_ratings(paths.seed_ratings_csv))


def replayed_store(teams, paths) -> RatingStore:
    """Walk-forward the 29 results from the seed prior (== `wcpredict replay`).

    Independent of the spread params (the Elo/form learner never uses the
    transform), so it is computed once and reused for every grid cell.
    """
    r = seed_store(teams, paths)
    for m in _chronological(read_matches(paths.results_csv)):
        apply_result(r, Params(), m)
    return r


def fixture_metrics(ratings: RatingStore, params: Params, odds_rows):
    """Per-gap-bucket signed favourite error, KL(market||model), robust edges."""
    buckets = {"<120": [], "120-250": [], ">=250": []}
    kl_sum = 0.0
    robust = 0
    for h, a, odds in odds_rows:
        neutral = h not in HOST_TEAM_IDS
        probs, _ = forecast(ratings, params, h, a, neutral)
        fair = devig(odds)
        d = ratings.elo(h) - ratings.elo(a) + (0.0 if neutral else params.home_advantage)
        gap = abs(d)
        fav = 0 if d > 0 else 2                       # favourite = higher-rated side
        err = probs[fav] - fair[fav]                  # >0 model over-rates favourite
        key = "<120" if gap < 120 else ("120-250" if gap < 250 else ">=250")
        buckets[key].append(err)
        kl_sum += sum(fair[i] * math.log(fair[i] / max(probs[i], 1e-12))
                      for i in range(3) if fair[i] > 0)
        pb = blend(probs, fair, 0.75)                 # trust the market 75%
        if max(ev_per_unit(pb[k], odds[k]) for k in range(3)) > 0:
            robust += 1
    bmean = {k: (sum(v) / len(v) if v else float("nan"), len(v)) for k, v in buckets.items()}
    return {"kl": kl_sum / len(odds_rows), "robust": robust, "n": len(odds_rows), "buckets": bmean}


def walkforward_ll(teams, paths, params: Params):
    """Leak-free walk-forward log-loss over the 29 results, seeded from the CSV prior."""
    ratings = seed_store(teams, paths)
    rows = []
    for m in _chronological(read_matches(paths.results_csv)):
        probs, _ = forecast(ratings, params, m.home_team_id, m.away_team_id, m.neutral)
        outcome = outcome_index(m.home_goals, m.away_goals)
        d = ratings.elo(m.home_team_id) - ratings.elo(m.away_team_id) \
            + (0.0 if m.neutral else params.home_advantage)
        fav = 0 if d > 0 else 2
        rows.append((probs, outcome, fav))
        apply_result(ratings, params, m)
    overall = log_loss([r[0] for r in rows], [r[1] for r in rows])
    won = [(p, o) for p, o, f in rows if o == f]
    drew = [(p, o) for p, o, f in rows if o == 1]
    fl = lambda xs: log_loss([p for p, _ in xs], [o for _, o in xs]) if xs else float("nan")
    return {"overall": overall, "fav_won": fl(won), "n_won": len(won),
            "fav_drew": fl(drew), "n_drew": len(drew)}


def historical_ll(teams, paths, params: Params) -> float:
    return backtest(read_matches(paths.historical_csv), seed_store(teams, paths), params, "logloss")


def outright(teams, params: Params, replayed: RatingStore, n_sims: int = OUTRIGHT_SIMS):
    df = run_simulation(teams, params, replayed, n_sims=n_sims, seed=OUTRIGHT_SEED)
    top = df.iloc[0]
    ids = set(df["team_id"])
    arg = float(df[df["team_id"] == "ARG"]["p_champion"].iloc[0]) if "ARG" in ids else float("nan")
    return str(top["team_id"]), float(top["p_champion"]), arg


def main() -> None:
    paths = Paths()
    teams = read_teams(paths.teams_csv)
    seeds = seed_store(teams, paths)
    odds_rows = load_odds(paths)
    replayed = replayed_store(teams, paths)

    configs = [("baseline s=1.0", Params(spread_threshold=150.0, spread_slope=1.0))]
    for t in GRID_T:
        for s in GRID_S:
            if t == 150.0 and s == 1.0:
                continue   # == baseline
            configs.append((f"T={t:.0f} s={s:.2f}", Params(spread_threshold=t, spread_slope=s)))
    configs.append(("PLACEBO s=1.5", Params(spread_threshold=150.0, spread_slope=1.5)))

    print(f"\nPrior fixtures: {len(odds_rows)} priced | walk-forward: "
          f"{len(read_matches(paths.results_csv))} results | historical: "
          f"{len(read_matches(paths.historical_csv))} games\n")
    print("GATE 1 (seed prior, gamma=1) signed favourite error per gap bucket + KL + robust edges;")
    print("GATE 2 (leak-free walk-forward) overall / fav-won / fav-drew log-loss.\n")

    hdr = (f"{'config':<16}{'<120':>9}{'120-250':>9}{'>=250':>9}{'KL':>8}"
           f"{'robust':>8}{'wf_all':>8}{'wf_won':>8}{'wf_drew':>8}{'hist':>7}")
    print(hdr)
    print("-" * len(hdr))
    for label, p in configs:
        fm = fixture_metrics(seeds, p, odds_rows)
        wf = walkforward_ll(teams, paths, p)
        hist = historical_ll(teams, paths, p)
        b = fm["buckets"]
        print(f"{label:<16}"
              f"{b['<120'][0]:>+9.3f}{b['120-250'][0]:>+9.3f}{b['>=250'][0]:>+9.3f}"
              f"{fm['kl']:>8.3f}{fm['robust']:>6d}/{fm['n']:<2d}"
              f"{wf['overall']:>8.3f}{wf['fav_won']:>8.3f}{wf['fav_drew']:>8.3f}{hist:>7.3f}")

    nb = {k: fixture_metrics(seeds, configs[0][1], odds_rows)["buckets"][k][1]
          for k in ("<120", "120-250", ">=250")}
    print(f"\nbucket counts: <120={nb['<120']}  120-250={nb['120-250']}  >=250={nb['>=250']}")

    # Descriptive: replay-state outright (champion + ARG) at baseline, the T=150
    # slope sweep, and the placebo. Uses replayed ratings (ARG on top), varying
    # only the forecast lens in the sim.
    print(f"\nDescriptive replay-state outright ({OUTRIGHT_SIMS} sims, seed {OUTRIGHT_SEED}):")
    print(f"{'config':<16}{'favourite':>12}{'fav %':>8}{'ARG %':>8}")
    print("-" * 44)
    desc = [("baseline s=1.0", Params(spread_threshold=150.0, spread_slope=1.0))]
    desc += [(f"T=150 s={s:.2f}", Params(spread_threshold=150.0, spread_slope=s))
             for s in GRID_S if s != 1.0]
    desc += [("PLACEBO s=1.5", Params(spread_threshold=150.0, spread_slope=1.5))]
    for label, p in desc:
        fav_id, fav_p, arg_p = outright(teams, p, replayed)
        print(f"{label:<16}{fav_id:>12}{fav_p * 100:>7.1f}{arg_p * 100:>7.1f}")


if __name__ == "__main__":
    main()
