"""Command-line interface: ``wcpredict <subcommand>``.

Loads the current model state (ratings + tuned params) at the start of every
command and saves it back where appropriate, so the engine carries what it has
learned from one run to the next.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

from .config import Params, Paths
from .data_io import (
    MatchRecord,
    Team,
    append_result,
    read_matches,
    read_official_schedule,
    read_seed_ratings,
    read_teams,
    write_results,
)
from .history import (
    append_prediction,
    append_ratings_snapshot,
    build_record,
    next_seq,
    read_predictions,
    summarize,
)
from .learn import apply_result, retune
from .match import simulate_match
from .poisson import expected_goals, match_probabilities
from .ratings import RatingStore
from .simulate import run_simulation


# --------------------------------------------------------------------------- #
# State loading helpers
# --------------------------------------------------------------------------- #
def _load_teams(paths: Paths) -> List[Team]:
    if not paths.teams_csv.exists():
        sys.exit(f"error: teams file not found at {paths.teams_csv}")
    return read_teams(paths.teams_csv)


def _load_params(paths: Paths) -> Params:
    return Params.load(paths.params_json)


def _load_ratings(paths: Paths, teams: List[Team]) -> RatingStore:
    if paths.ratings_json.exists():
        return RatingStore.load(paths.ratings_json)
    # Cold start: seed from disk (or fallback) without persisting yet.
    seeds = read_seed_ratings(paths.seed_ratings_csv)
    return RatingStore.seed(teams, seeds)


def _resolve_team(teams: List[Team], token: str) -> str:
    token_l = token.strip().lower()
    for t in teams:
        if t.team_id.lower() == token_l or t.name.lower() == token_l:
            return t.team_id
    sys.exit(f"error: unknown team {token!r} (use a team_id from teams.csv)")


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #
def cmd_reset(args, paths: Paths) -> int:
    teams = _load_teams(paths)
    seeds = read_seed_ratings(paths.seed_ratings_csv)
    ratings = RatingStore.seed(teams, seeds)
    paths.ensure_state_dir()
    ratings.save(paths.ratings_json)
    Params().save(paths.params_json)
    print(f"Reset state for {len(ratings)} teams -> {paths.ratings_json}")
    print(f"Wrote default params -> {paths.params_json}")
    return 0


def cmd_simulate(args, paths: Paths) -> int:
    teams = _load_teams(paths)
    params = _load_params(paths)
    ratings = _load_ratings(paths, teams)
    df = run_simulation(teams, params, ratings, n_sims=args.sims, seed=args.seed)

    top = df.head(args.top)
    print(f"\nWorld Cup simulation — {args.sims} runs (seed={args.seed})\n")
    header = f"{'Team':<22}{'Grp':<5}{'Elo':>7}{'Adv%':>8}{'QF%':>8}{'SF%':>8}{'Win%':>8}"
    print(header)
    print("-" * len(header))
    for _, r in top.iterrows():
        print(
            f"{r['team']:<22}{r['group']:<5}{r['elo']:>7.0f}"
            f"{r['p_advance'] * 100:>7.1f} {r['p_qf'] * 100:>7.1f} "
            f"{r['p_sf'] * 100:>7.1f} {r['p_champion'] * 100:>7.1f}"
        )
    fav = df.iloc[0]
    print(f"\nPredicted champion: {fav['team']} ({fav['p_champion'] * 100:.1f}%)")
    return 0


def cmd_predict(args, paths: Paths) -> int:
    teams = _load_teams(paths)
    params = _load_params(paths)
    ratings = _load_ratings(paths, teams)
    home = _resolve_team(teams, args.match[0])
    away = _resolve_team(teams, args.match[1])

    d = ratings.elo(home) - ratings.elo(away)
    rh, ra = ratings[home], ratings[away]
    lam_h, lam_a = expected_goals(d, params, form_home=rh.form, form_away=ra.form)
    p_home, p_draw, p_away = match_probabilities(lam_h, lam_a, params.max_goals, params.dc_rho)

    print(f"\n{home} (Elo {ratings.elo(home):.0f}) vs {away} (Elo {ratings.elo(away):.0f})")
    print(f"Expected goals: {lam_h:.2f} - {lam_a:.2f}")
    if args.knockout:
        # Knockout: redistribute the draw probability by who would likely win it.
        tilt = 0.5 + params.penalty_elo_weight * (d / params.elo_divisor)
        tilt = max(0.05, min(0.95, tilt))
        adv_home = p_home + p_draw * tilt
        adv_away = p_away + p_draw * (1 - tilt)
        print(f"Advance probability: {home} {adv_home * 100:.1f}%  |  {away} {adv_away * 100:.1f}%")
    else:
        print(
            f"Win {home}: {p_home * 100:.1f}%  |  Draw: {p_draw * 100:.1f}%  |  "
            f"Win {away}: {p_away * 100:.1f}%"
        )
    return 0


def cmd_update_result(args, paths: Paths) -> int:
    teams = _load_teams(paths)
    params = _load_params(paths)
    ratings = _load_ratings(paths, teams)

    home = _resolve_team(teams, args.home)
    away = _resolve_team(teams, args.away)
    try:
        hg, ag = (int(x) for x in args.score.lower().split("-"))
    except ValueError:
        sys.exit("error: --score must look like '2-1'")

    record = MatchRecord(
        date=args.date or "",
        home_team_id=home, away_team_id=away,
        home_goals=hg, away_goals=ag,
        stage=args.stage, competition=args.competition,
        neutral=not args.home_advantage,
    )
    append_result(paths.results_csv, record)

    # Snapshot the pre-match model state, learn from the result, then log what we
    # had predicted vs what happened — so accuracy can be reviewed later.
    ratings_pre = ratings.copy()
    seq = next_seq(paths.predictions_csv)
    dh, da = apply_result(ratings, params, record)
    paths.ensure_state_dir()
    ratings.save(paths.ratings_json)

    pred = build_record(seq, ratings_pre, ratings, params, record, (dh, da))
    append_prediction(paths.predictions_csv, pred)
    append_ratings_snapshot(paths.ratings_history_csv, seq, record, ratings)

    print(f"Recorded {home} {hg}-{ag} {away}")
    print(f"Pre-match forecast: {home} {pred.p_home * 100:.0f}%  draw "
          f"{pred.p_draw * 100:.0f}%  {away} {pred.p_away * 100:.0f}%  "
          f"(predicted: {pred.predicted_outcome}, actual: {pred.actual_outcome}, "
          f"log-loss {pred.log_loss:.3f})")
    print(f"Elo update: {home} {dh:+.1f} -> {ratings.elo(home):.0f}   "
          f"{away} {da:+.1f} -> {ratings.elo(away):.0f}")
    print(f"Logged prediction #{seq} -> {paths.predictions_csv}")
    return 0


def cmd_fetch_results(args, paths: Paths) -> int:
    from .fetch import FetchError, fetch_results

    teams = _load_teams(paths)
    params = _load_params(paths)
    ratings = _load_ratings(paths, teams)
    try:
        records = fetch_results(teams, source=args.source, since=args.since)
    except FetchError as exc:
        sys.exit(f"error: {exc}")

    seq = next_seq(paths.predictions_csv)
    for rec in records:
        append_result(paths.results_csv, rec)
        if args.learn:
            ratings_pre = ratings.copy()
            deltas = apply_result(ratings, params, rec)
            pred = build_record(seq, ratings_pre, ratings, params, rec, deltas)
            append_prediction(paths.predictions_csv, pred)
            append_ratings_snapshot(paths.ratings_history_csv, seq, rec, ratings)
            seq += 1
    if args.learn and records:
        paths.ensure_state_dir()
        ratings.save(paths.ratings_json)
    print(f"Fetched and stored {len(records)} result(s)"
          + (" and updated ratings" if args.learn else ""))
    return 0


def cmd_retune(args, paths: Paths) -> int:
    teams = _load_teams(paths)
    params = _load_params(paths)
    seeds = read_seed_ratings(paths.seed_ratings_csv)
    base_ratings = RatingStore.seed(teams, seeds)

    matches = read_matches(paths.historical_csv) + read_matches(paths.results_csv)
    if not matches:
        sys.exit("error: no matches to retune on (need data/historical_matches.csv or results)")

    result = retune(matches, base_ratings, params, metric=args.metric, method=args.method)
    print(f"Retune ({args.metric}, {args.method}) over {len(matches)} matches:")
    print(f"  score before: {result.score_before:.4f}")
    print(f"  score after:  {result.score_after:.4f}")
    if result.success:
        result.params.save(paths.params_json)
        print("  improved -> wrote new params:")
        from .config import TUNABLE_FIELDS
        for f in TUNABLE_FIELDS:
            print(f"    {f:<16}{getattr(params, f):.3f} -> {getattr(result.params, f):.3f}")
    else:
        print("  no improvement found; params unchanged.")
    return 0


def cmd_standings(args, paths: Paths) -> int:
    teams = _load_teams(paths)
    ratings = _load_ratings(paths, teams)
    by_group: dict[str, list[Team]] = {}
    for t in teams:
        if args.group and t.group.upper() != args.group.upper():
            continue
        by_group.setdefault(t.group, []).append(t)
    for g in sorted(by_group):
        print(f"\nGroup {g}")
        ordered = sorted(by_group[g], key=lambda t: ratings.elo(t.team_id), reverse=True)
        for t in ordered:
            print(f"  {t.name:<22}{ratings.elo(t.team_id):>7.0f}")
    return 0


def cmd_ratings(args, paths: Paths) -> int:
    teams = _load_teams(paths)
    ratings = _load_ratings(paths, teams)
    name_by_id = {t.team_id: t.name for t in teams}
    print(f"\n{'#':<4}{'Team':<24}{'Elo':>7}{'Form':>8}")
    print("-" * 43)
    for i, (tid, r) in enumerate(ratings.ranked()[: args.top], start=1):
        print(f"{i:<4}{name_by_id.get(tid, tid):<24}{r.elo:>7.0f}{r.form:>8.2f}")
    return 0


def cmd_import_results(args, paths: Paths) -> int:
    """Import played matches from the official FIFA schedule CSV into results.csv."""
    teams = _load_teams(paths)
    src = Path(args.file)
    if not src.exists():
        sys.exit(f"error: file not found: {src}")
    records = read_official_schedule(src, teams)
    if not records:
        sys.exit("error: no completed matches found in the file")
    write_results(paths.results_csv, records)
    last = records[-1]
    print(f"Imported {len(records)} completed match(es) -> {paths.results_csv}")
    print(f"  date range: {records[0].date} .. {last.date}")
    print(f"  latest: {last.home_team_id} {last.home_goals}-{last.away_goals} {last.away_team_id}")
    print("\nNext: 'wcpredict replay' to validate, or 'wcpredict simulate' to predict.")
    return 0


def cmd_import_historical(args, paths: Paths) -> int:
    """Import past tournaments (FIFA-format CSVs) into historical_matches.csv for retuning."""
    teams = _load_teams(paths)
    all_records: List[MatchRecord] = []
    for f in args.file:
        src = Path(f)
        if not src.exists():
            sys.exit(f"error: file not found: {src}")
        recs = read_official_schedule(
            src, teams,
            allow_unknown=True,      # keep teams not in the 2026 field
            host_ids=set(),          # historical hosts differ; treat WC games as neutral
            competition=src.stem,
        )
        print(f"  {src.name}: {len(recs)} matches")
        all_records.extend(recs)
    all_records.sort(key=lambda m: (m.date, m.home_team_id))
    write_results(paths.historical_csv, all_records)
    print(f"Wrote {len(all_records)} historical matches -> {paths.historical_csv}")
    print("\nNext: 'wcpredict retune' to optimise weights on this data.")
    return 0


def cmd_replay(args, paths: Paths) -> int:
    """Replay already-played matches through the model with the current weights.

    Walk-forward validation: each match is forecast using the ratings as they
    stood *before* it, the forecast is scored, then the model learns from the
    result and moves on. Rebuilds the prediction ledger and ratings history from
    scratch so the run is fully reproducible.
    """
    from .learn import _chronological

    teams = _load_teams(paths)
    params = _load_params(paths)
    if not args.form:
        params = params.copy_with(form_alpha=0.0)

    if args.from_state and paths.ratings_json.exists():
        ratings = RatingStore.load(paths.ratings_json)
        start = "current saved ratings"
    else:
        ratings = RatingStore.seed(teams, read_seed_ratings(paths.seed_ratings_csv))
        start = "seed ratings"

    source = Path(args.source) if args.source else paths.results_csv
    matches = _chronological(read_matches(source))
    if not matches:
        sys.exit(f"error: no matches to replay in {source}")

    # Fresh ledgers — a replay regenerates the full history deterministically.
    for p in (paths.predictions_csv, paths.ratings_history_csv):
        if p.exists():
            p.unlink()

    print(f"Replaying {len(matches)} match(es) from {source}")
    print(f"Starting from {start} with current weights "
          f"(k={params.k_factor:.1f}, home_adv={params.home_advantage:.0f}, "
          f"beta={params.beta:.2f}, mu={params.mu:.2f})\n")

    for seq, m in enumerate(matches, start=1):
        ratings_pre = ratings.copy()
        deltas = apply_result(ratings, params, m)
        pred = build_record(seq, ratings_pre, ratings, params, m, deltas)
        append_prediction(paths.predictions_csv, pred)
        append_ratings_snapshot(paths.ratings_history_csv, seq, m, ratings)
        if not args.quiet:
            mark = "OK " if pred.predicted_outcome == pred.actual_outcome else "miss"
            print(f"  #{seq:<3} {m.home_team_id} {m.home_goals}-{m.away_goals} {m.away_team_id:<4}"
                  f"  pred {pred.p_home * 100:>4.0f}/{pred.p_draw * 100:>3.0f}/{pred.p_away * 100:>4.0f}"
                  f"  ->{pred.predicted_outcome:<5} actual {pred.actual_outcome:<5}[{mark}]"
                  f"  ll={pred.log_loss:.3f}")

    if args.save:
        paths.ensure_state_dir()
        ratings.save(paths.ratings_json)

    summary = summarize(read_predictions(paths.predictions_csv), recent=args.recent)
    print(f"\nAccuracy over {summary.n} played match(es):")
    print(f"  log-loss:       {summary.log_loss:.4f}   (baseline guess {summary.baseline_log_loss:.4f})")
    print(f"  Brier score:    {summary.brier:.4f}")
    print(f"  top-pick hit:   {summary.hit_rate * 100:.1f}%")
    print(f"  skill vs guess: {summary.skill * 100:+.1f}%")
    if summary.skill < 0:
        print("  ! Below baseline — try 'wcpredict retune' or revisit seed ratings.")
    if args.save:
        print(f"\nSaved updated ratings -> {paths.ratings_json}")
    else:
        print("\n(ratings not saved; pass --save to persist, or this was a dry run)")
    return 0


def cmd_accuracy(args, paths: Paths) -> int:
    rows = read_predictions(paths.predictions_csv)
    summary = summarize(rows, recent=args.recent)
    if summary is None:
        print("No predictions logged yet. Record results with 'update-result' first.")
        return 0

    print(f"\nPrediction accuracy over {summary.n} match(es)")
    print(f"  log-loss:        {summary.log_loss:.4f}   (baseline guess {summary.baseline_log_loss:.4f})")
    print(f"  Brier score:     {summary.brier:.4f}")
    print(f"  top-pick hit:    {summary.hit_rate * 100:.1f}%")
    print(f"  skill vs guess:  {summary.skill * 100:+.1f}%  (higher is better)")
    if summary.recent_log_loss is not None:
        trend = summary.recent_log_loss - summary.log_loss
        arrow = "improving" if trend < 0 else "worsening"
        print(f"  last {args.recent} log-loss: {summary.recent_log_loss:.4f}  ({arrow})")

    if summary.skill < 0:
        print("\n  ! Worse than random — consider 'wcpredict retune' or revisiting seed ratings.")
    elif summary.recent_log_loss is not None and summary.recent_log_loss > summary.log_loss * 1.15:
        print("\n  ! Recent accuracy is drifting — a 'wcpredict retune' may help.")

    if args.last:
        print(f"\nLast {args.last} prediction(s):")
        for r in rows[-args.last:]:
            mark = "OK " if r["predicted_outcome"] == r["actual_outcome"] else "miss"
            print(f"  #{r['seq']:<3} {r['home_team_id']} {r['home_goals']}-{r['away_goals']} "
                  f"{r['away_team_id']}  pred={r['predicted_outcome']:<5} "
                  f"actual={r['actual_outcome']:<5} [{mark}] logloss={r['log_loss']}")
    return 0


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wcpredict", description="Self-learning World Cup predictor")
    p.add_argument("--data-dir", default=None, help="override the data/ directory")
    p.add_argument("--state-dir", default=None, help="override the state/ directory")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("simulate", help="Monte Carlo simulate the tournament")
    s.add_argument("--sims", type=int, default=10000)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--top", type=int, default=15)
    s.set_defaults(func=cmd_simulate)

    s = sub.add_parser("predict", help="predict a single match")
    s.add_argument("--match", nargs=2, metavar=("HOME", "AWAY"), required=True)
    s.add_argument("--knockout", action="store_true")
    s.set_defaults(func=cmd_predict)

    s = sub.add_parser("update-result", help="record a real result and learn from it")
    s.add_argument("--home", required=True)
    s.add_argument("--away", required=True)
    s.add_argument("--score", required=True, help="e.g. 2-1")
    s.add_argument("--stage", default="group")
    s.add_argument("--competition", default="WC2026")
    s.add_argument("--date", default=None)
    s.add_argument("--home-advantage", action="store_true",
                   help="treat as a non-neutral (home) game")
    s.set_defaults(func=cmd_update_result)

    s = sub.add_parser("fetch-results", help="fetch real results from an API (optional)")
    s.add_argument("--source", default="football-data")
    s.add_argument("--since", default=None, help="ISO date, e.g. 2026-06-11")
    s.add_argument("--no-learn", dest="learn", action="store_false", default=True)
    s.set_defaults(func=cmd_fetch_results)

    s = sub.add_parser("retune", help="backtest and optimise hyperparameters")
    s.add_argument("--metric", choices=["logloss", "brier"], default="logloss")
    s.add_argument("--method", choices=["nelder-mead", "grid"], default="nelder-mead")
    s.set_defaults(func=cmd_retune)

    s = sub.add_parser("standings", help="show current group standings by Elo")
    s.add_argument("--group", default=None)
    s.set_defaults(func=cmd_standings)

    s = sub.add_parser("ratings", help="show the current Elo table")
    s.add_argument("--top", type=int, default=20)
    s.set_defaults(func=cmd_ratings)

    s = sub.add_parser("import-results", help="import played matches from the official FIFA schedule CSV")
    s.add_argument("--file", required=True, help="path to the official schedule CSV")
    s.set_defaults(func=cmd_import_results)

    s = sub.add_parser("import-historical", help="import past tournaments for retuning")
    s.add_argument("--file", required=True, nargs="+", help="one or more FIFA-format schedule CSVs")
    s.set_defaults(func=cmd_import_historical)

    s = sub.add_parser("replay", help="replay already-played matches to validate the model")
    s.add_argument("--source", default=None, help="results CSV to replay (default: data/results.csv)")
    s.add_argument("--from-state", action="store_true",
                   help="start from current saved ratings instead of seed ratings")
    s.add_argument("--no-save", dest="save", action="store_false", default=True,
                   help="don't persist the resulting ratings (dry run)")
    s.add_argument("--quiet", action="store_true", help="suppress the play-by-play lines")
    s.add_argument("--no-form", dest="form", action="store_false", default=True,
                   help="disable the tournament-form overlay (Elo only)")
    s.add_argument("--recent", type=int, default=10)
    s.set_defaults(func=cmd_replay)

    s = sub.add_parser("accuracy", help="review prediction accuracy over recorded results")
    s.add_argument("--recent", type=int, default=10, help="window for the recent-trend score")
    s.add_argument("--last", type=int, default=10, help="show the last N predictions (0 to hide)")
    s.set_defaults(func=cmd_accuracy)

    s = sub.add_parser("reset", help="re-seed ratings and params from data/")
    s.set_defaults(func=cmd_reset)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    paths = Paths(
        data_dir=args.data_dir or Paths().data_dir,
        state_dir=args.state_dir or Paths().state_dir,
    )
    return args.func(args, paths)


if __name__ == "__main__":
    raise SystemExit(main())
