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
wcpredict retune --metric logloss                 # optimise hyperparameters
wcpredict ratings --top 20                         # current Elo table
wcpredict standings --group C                       # group view
wcpredict fetch-results --since 2026-06-11          # optional, needs API key
```

The live fetcher uses [football-data.org](https://www.football-data.org/); set
`FOOTBALL_DATA_API_KEY` and install the `api` extra. Without it the engine works
fully offline — just record results with `update-result`.

## Data

- `data/teams.csv` — 48 teams, their confederation and group. **The group draw and
  the Elo seeds in `data/seed_ratings.csv` are illustrative and editable** — drop
  in the official draw and ratings when finalised.
- `data/results.csv` — append-only log of real results the engine learns from.
- `data/historical_matches.csv` — optional past matches used by `retune`
  (illustrative sample; extend with a larger dataset for better tuning).

## Project layout

```
src/wcpredictor/   elo, poisson, match, tournament, simulate, learn, metrics, cli, ...
data/              teams, seed ratings, results, historical matches
state/             generated: ratings.json, params.json (git-ignored)
tests/             pytest suite
```

## Tests

```bash
pytest
```
