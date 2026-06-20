#!/usr/bin/env python3
"""Read-only decomposition of the outright top-heaviness (no model changes).

Attributes the favourite's champion probability to: the seed-prior spread, the
online Elo amplification from replaying the 29 results, the tournament-form
overlay, and the forecast-time spread lens. Prints the top-5 champion table
under several configurations so the partner-lever decision (P4/P5) is data-driven.
"""

from __future__ import annotations

from wcpredictor.config import Params, Paths
from wcpredictor.data_io import read_matches, read_seed_ratings, read_teams
from wcpredictor.learn import _chronological, apply_result
from wcpredictor.ratings import RatingStore
from wcpredictor.simulate import run_simulation

SIMS = 10000
SEED = 42


def seed_store(teams, paths):
    return RatingStore.seed(teams, read_seed_ratings(paths.seed_ratings_csv))


def replay(teams, paths, *, form: bool, k_factor: float | None = None):
    r = seed_store(teams, paths)
    p = Params()
    if not form:
        p = p.copy_with(form_alpha=0.0)
    if k_factor is not None:
        p = p.copy_with(k_factor=k_factor)
    for m in _chronological(read_matches(paths.results_csv)):
        apply_result(r, p, m)
    return r


def top5(teams, params, ratings, label):
    df = run_simulation(teams, params, ratings, n_sims=SIMS, seed=SEED)
    head = df.head(5)
    cells = "  ".join(f"{row.team_id}:{row.p_champion*100:4.1f}" for row in head.itertuples())
    arg = float(df[df["team_id"] == "ARG"]["p_champion"].iloc[0])
    print(f"{label:<34} ARG={arg*100:4.1f}  | top5: {cells}")


def main():
    paths = Paths()
    teams = read_teams(paths.teams_csv)
    noop = Params(spread_slope=1.0, spread_threshold=150.0)
    s250 = Params(spread_slope=0.5, spread_threshold=250.0)
    s150 = Params(spread_slope=0.5, spread_threshold=150.0)
    aggr = Params(spread_slope=0.4, spread_threshold=120.0)

    seeds = seed_store(teams, paths)
    rep_form = replay(teams, paths, form=True)
    rep_noform = replay(teams, paths, form=False)
    rep_k20 = replay(teams, paths, form=True, k_factor=20.0)

    print(f"\nOutright decomposition ({SIMS} sims, seed {SEED})\n")
    print("== prior (seeds only, no replay) ==")
    top5(teams, noop, seeds, "seeds  | no-op")
    top5(teams, s250, seeds, "seeds  | T=250 s=0.5")
    print("\n== replayed (29 results, form ON) ==")
    top5(teams, noop, rep_form, "replay | no-op  (baseline)")
    top5(teams, s250, rep_form, "replay | T=250 s=0.5")
    top5(teams, s150, rep_form, "replay | T=150 s=0.5")
    top5(teams, aggr, rep_form, "replay | T=120 s=0.4 (aggressive)")
    print("\n== replayed, form OFF ==")
    top5(teams, noop, rep_noform, "replay-noform | no-op")
    top5(teams, s250, rep_noform, "replay-noform | T=250 s=0.5")
    print("\n== replayed, k_factor=20 (half), form ON ==")
    top5(teams, noop, rep_k20, "replay-k20 | no-op")
    top5(teams, s250, rep_k20, "replay-k20 | T=250 s=0.5")

    print("\n== rating-uncertainty sweep (replay, T=250 s=0.5) vs market FRA~13 ESP/ENG~13 ARG~10 ==")
    for sigma in (0.0, 40.0, 60.0, 80.0, 100.0, 120.0, 150.0):
        p = Params(spread_threshold=250.0, spread_slope=0.5, rating_sigma=sigma)
        top5(teams, p, rep_form, f"sigma={sigma:.0f}")


if __name__ == "__main__":
    main()
