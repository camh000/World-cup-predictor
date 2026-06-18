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
