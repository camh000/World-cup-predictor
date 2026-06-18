# World-cup-predictor

A self-learning Monte Carlo engine that simulates the World Cup to predict the
champion — and **automatically alters itself after every real-life game to
become more accurate**.

## How it works

**Predicting (Monte Carlo).** Each team has an **Elo rating**. For any match the
Elo gap is turned into expected goals via a **Poisson model**, and a scoreline is
drawn at random. The whole tournament — 12 groups of 4, then a 32-team knockout
bracket (with the 8 best third-placed teams advancing) — is simulated tens of
thousands of times. Aggregating the outcomes gives each team's probability of
advancing, reaching each round, and winning the trophy.

**Self-learning (two layers).**

1. **Online updates (after every game).** When a real result comes in, both
   teams' Elo ratings are updated immediately — winners rise, losers fall, by an
   amount scaled to how surprising the result was and its margin of victory. The
   next simulation automatically reflects the new ratings.
2. **Periodic retuning.** `retune` runs a strict *walk-forward* backtest over all
   accumulated results and optimises the model's hyperparameters (home advantage,
   learning rate, goal scaling) to minimise prediction error (log-loss / Brier).

**Tournament form (a two-speed overlay).** International teams rarely play, so a
seed Elo can be stale by kick-off. Alongside the slow Elo baseline, each team
carries a **form** multiplier that starts at **1.0** each tournament and reacts
*fast* to how much a team over- or under-performs its Elo expectation in this
tournament's games. It mean-reverts toward 1.0 between games and is bounded
(0.85–1.15). An in-form team is modelled to score more and concede less. Because
form starts at 1.0 and only changes after a team plays, it has no effect on a
team's first match and grows in influence over matchdays 2–3. Disable it for a
pure-Elo run with `wcpredict replay --no-form` (or `form_alpha = 0`).

Canonical inputs live in `data/` (committed); the evolving learned model lives in
`state/` (git-ignored, rebuildable with `wcpredict reset`). A given
`(state, seed)` pair is fully deterministic.

## Install

```bash
pip install -e .            # core engine (numpy, pandas, scipy)
pip install -e '.[api]'     # optional: live results fetcher (requests)
pip install -e '.[dev]'     # optional: pytest
```

## Usage

```bash
wcpredict reset                                   # seed ratings + default params
wcpredict simulate --sims 20000 --seed 42         # predict the champion
wcpredict predict --match BRA ARG --knockout      # one-off match probabilities
wcpredict update-result --home BRA --away ARG --score 3-0 --stage group
wcpredict simulate --sims 20000 --seed 42         # BRA's odds have now shifted
wcpredict accuracy                                # how good have predictions been?
wcpredict replay                                  # validate model on already-played games
wcpredict retune --metric logloss                 # optimise hyperparameters
wcpredict ratings --top 20                         # current Elo table
wcpredict standings --group C                       # group view
wcpredict fetch-results --since 2026-06-11          # optional, needs API key
```

### Tracking accuracy over time

Every recorded result is added to a prediction ledger **before** the model
learns from it, so you always have a record of what was predicted versus what
happened — and can tell whether the model is improving or needs adjusting:

- `data/predictions.csv` — one row per match: both teams' pre-match Elo, the
  forecast (win/draw/win probabilities + expected goals), the predicted and
  actual outcomes, the post-match Elo and rating changes, and the per-match
  log-loss & Brier score.
- `data/ratings_history.csv` — every team's Elo snapshotted after each match, so
  each team's rating trajectory through the tournament is preserved.

`wcpredict accuracy` summarises the ledger: running log-loss / Brier versus a
naive baseline, top-pick hit rate, a "skill" score, and a recent-form trend. It
flags when accuracy is drifting and a `retune` is worth running.

### Validating against matches already played

`wcpredict replay` (or `scripts/replay_2026.sh`) runs every result in
`data/results.csv` back through the model **with the current weights**, in
chronological order — forecasting each game from the ratings as they stood
*before* it, scoring the forecast, then learning from the result. It rebuilds the
ledger, prints a play-by-play, and reports overall accuracy, so you can confirm
the model is sensible before trusting its forward predictions:

```bash
scripts/replay_2026.sh            # play-by-play + accuracy + updated title odds
wcpredict replay --quiet          # accuracy summary only
wcpredict replay --no-save        # dry run (don't persist the learned ratings)
wcpredict replay --from-state     # start from current ratings instead of seeds
```

> The 24 first-round 2026 results in `data/results.csv` were compiled from public
> match reporting; verify/adjust them as needed. On that draw-heavy opening round
> the default-weight model scores a log-loss of ~1.04 vs a 1.10 naive baseline
> (~+5% skill) — modestly better than guessing, with clear room to improve
> (notably it under-predicts draws), which is exactly the kind of thing this
> replay is meant to reveal.

The live fetcher uses [football-data.org](https://www.football-data.org/); set
`FOOTBALL_DATA_API_KEY` and install the `api` extra. Without it the engine works
fully offline — just record results with `update-result`.

## Data

- `data/teams.csv` — the **real 48 teams and official group draw** for the 2026
  World Cup (drawn 5 Dec 2025; hosts Mexico/Canada/USA in A1/B1/D1).
- `data/seed_ratings.csv` — seed strengths on the **eloratings.net scale**
  (June 2026). Top contenders use the exact published values (Spain 2129,
  Argentina 2115, France 2063, England 2024, Portugal 1989, Brazil 1978); the
  remaining teams are close approximations. The engine refines all of them from
  real results, so exact starting values matter less over time.
- `data/results.csv` — append-only log of real results the engine learns from.
- `data/historical_matches.csv` — optional past matches used by `retune`
  (illustrative sample; extend with a larger dataset for better tuning).

## Project layout

```
src/wcpredictor/   elo, poisson, match, tournament, simulate, learn, history, metrics, cli, ...
data/              teams, seed ratings, results, historical matches, predictions, ratings history
state/             generated: ratings.json, params.json (git-ignored)
tests/             pytest suite
```

## Tests

```bash
pytest
```
