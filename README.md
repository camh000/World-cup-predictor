# World-cup-predictor

A self-learning Monte Carlo engine that simulates the World Cup to predict the
champion — and **automatically alters itself after every real-life game to
become more accurate**.

## How it works

**Predicting (Monte Carlo).** Each team has an **Elo rating**. For any match the
Elo gap is turned into expected goals via a **Poisson model** with a
**Dixon–Coles** low-score correction (the independent-Poisson model under-predicts
draws; this shifts mass into 0-0/1-1), and a scoreline is drawn at random. The whole tournament — 12 groups of 4, then a 32-team knockout
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

**Calibration to the market (two forecast-only lenses).** The raw eloratings.net
scale is mis-calibrated against bookmaker odds in a *sign-flipping* way — it was
over-confident on elite mismatches yet about right on the mid-tier — and, fed
through a fixed-rating Monte Carlo, it concentrated a silly ~31% on a single
outright favourite. Two lenses fix this without distorting the learned Elo:

1. **Spread compression** (`Params.spread_slope`/`spread_threshold`). A non-linear
   transform on the *effective Elo gap* inside the goal model: gaps up to ~250 pass
   through untouched and only the excess above that is scaled (default `0.5`). It
   removes the elite over-confidence (a blow-out favourite no longer reads ~92%)
   and is an exact no-op at `spread_slope = 1.0`. A *linear* shrink can't do this —
   it only rescales `beta` and so can't fix an error that changes sign.
2. **Tournament rating uncertainty** (`Params.rating_sigma`, Elo std-dev). Used
   *only* by the champion Monte Carlo: each simulated tournament draws every team's
   strength once from `N(elo, rating_sigma)`, representing the irreducible
   uncertainty (injuries, form, squad depth) the market prices. Because title odds
   are ~`p⁵` convex, this deflates an over-concentrated favourite toward the market
   (default `150` pulls a clear #1 from ~31% to ~17%) **without touching any
   per-match 1X2 forecast**. `0.0` is an exact no-op.

Both are *forecast-only* and reversible — they never change `update_elo`, so the
learned ratings and the replay ledger are unaffected. They were chosen on a
leak-free prior-vs-market grid (`scripts/validate_prior.py`,
`scripts/decompose_outright.py`), not fitted to match outcomes. The draw rate and
the Elo learning-rate/form overlay were deliberately left alone: the draw excess
is ~1.6pt and almost entirely structural Poisson (lowering `mu` would *raise* it),
and the outright top-heaviness proved insensitive to `k_factor`/form (it is in the
seeds themselves), so neither was a useful lever. The model still genuinely
*disagrees* with the market on most single games — it is an Elo-only estimator, and
that disagreement is mostly its own error, not value; the dashboard says so.

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
wcpredict import-results --file <official_schedule.csv>  # load real results
wcpredict simulate --sims 20000 --seed 42         # predict the champion
wcpredict predict --match BRA ARG --knockout      # one-off match probabilities
wcpredict update-result --home BRA --away ARG --score 3-0 --stage group
wcpredict simulate --sims 20000 --seed 42         # BRA's odds have now shifted
wcpredict accuracy                                # how good have predictions been?
wcpredict replay                                  # validate model on already-played games
wcpredict import-historical --file wc2018.csv wc2022.csv  # past tournaments for tuning
wcpredict retune --metric logloss                 # tune goal-model weights on real data
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

### Can it beat the bookies?

`src/wcpredictor/betting.py` does the honest test — not "is the model better than
guessing" but "is it better than the **market**, after the bookmaker's margin".
It de-vigs decimal odds into fair probabilities, scores the model against them
with log-loss, and backtests flat-stake and fractional-Kelly bankrolls over every
+EV bet. The dashboard's "Can we beat the bookies?" panel shows the result.

`data/odds.csv` now holds **real** committed bookmaker prices (pulled by
`scripts/fetch_odds.py`; `scripts/make_sample_odds.py` remains a synthetic fallback
for offline/testing). The sobering truth still holds against the real market: the
model's apparent "value" is overwhelmingly its own error, not an edge, and once the
vig is paid there is no durable +EV. The committed `data/odds.csv` uses decimal
odds, one row per match:

```
date,home_team_id,away_team_id,odds_home,odds_draw,odds_away
2026-06-11,MEX,RSA,2.10,3.40,3.60
```

> A backtest that prints profit on a **synthetic** market built from the model's own
> numbers is **circular** and meaningless — the only trustworthy signal is positive
> closing-line value over a real, sizeable sample, which is why `scripts/fetch_odds.py`
> now appends a timestamped snapshot to `data/odds_history.csv` on every run.

**Auto-fetching real odds.** `scripts/fetch_odds.py` pulls live bookmaker prices
from [the-odds-api.com](https://the-odds-api.com/) into `data/odds.csv` (match
1X2) and `data/outright_odds.csv` (tournament winner). It is credit-thrifty: the
free `/sports` lookup (0 credits) auto-detects the live World Cup keys, then one
region + one market makes a full refresh just **2 credits** (1 match + 1 outright)
against the free tier's 500/month. Set the `ODDS_API_KEY` secret and it runs in
the daily `refresh.yml`; without the key it no-ops. It never overwrites a CSV with
an empty result and prints the remaining credit balance each run.

## Formula 1 (the webring)

A sister page, `f1.html`, applies the same idea to the 2026 F1 season — linked to
the football page by a little early-2000s **webring** nav bar. `wcpredictor.f1`
rates each driver with a pairwise Elo over 2025+2026 finishing orders (the rating
absorbs car pace), then simulates races Plackett-Luce style (Gumbel-perturbed
ratings) to get next-race podium odds and a Monte-Carlo drivers'/constructors'
championship. Build it with:

```bash
python scripts/fetch_f1.py          # pull data from toUpperCase78/formula1-datasets
python scripts/make_f1_dashboard.py # -> f1.html
```

Both run in the daily `refresh.yml`. Caveats: the dataset has no 2026 calendar, so
the remaining season is assumed to be a 24-race run, and sprints/track effects
aren't modelled — it's for fun.

## Data

- `data/teams.csv` — the **real 48 teams and official group draw** for the 2026
  World Cup (drawn 5 Dec 2025; hosts Mexico/Canada/USA in A1/B1/D1).
- `data/seed_ratings.csv` — seed strengths on the **eloratings.net scale**
  (June 2026). Top contenders use the exact published values (Spain 2129,
  Argentina 2115, France 2063, England 2024, Portugal 1989, Brazil 1978); the
  remaining teams are close approximations. The engine refines all of them from
  real results, so exact starting values matter less over time.
- `data/results.csv` — real results the engine learns from, generated from the
  official FIFA schedule (`data/fifa_worldcup_2026_schedule.csv`) via
  `wcpredict import-results`. Re-run it as each matchday is played.
- `data/historical_matches.csv` — past matches used by `retune`, generated from
  past tournaments (`data/fifa_worldcup_2018_schedule.csv`,
  `data/fifa_worldcup_2022_schedule.csv`) via `wcpredict import-historical`.
  `retune` tunes only the goal-shape params (`beta`, `mu`, `dc_rho`); home
  advantage and the Elo K-factor are deliberately left fixed (tournament data is
  neutral, and minimising in-sample loss would just switch off the learning).

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
