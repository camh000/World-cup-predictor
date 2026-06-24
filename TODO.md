# TODO

## 1. Validate & tune the tournament-form overlay (once matchday 2+ is played)

The `form` overlay (see "Tournament form" in `README.md`) ships with **unvalidated
defaults** (`form_alpha=0.10`, `form_decay=0.85`, bounds 0.85–1.15 in
`src/wcpredictor/config.py`). It is a no-op on matchday 1 because a team's form
only changes *after* it plays. When matchday 2 (and 3) are in:

1. Re-import results: `wcpredict import-results --file <official schedule>`.
2. A/B test: `wcpredict replay` vs `wcpredict replay --no-form` (compare log-loss/Brier).
3. If it helps, add `form_alpha`/`form_decay` to `TUNABLE_FIELDS`
   (`src/wcpredictor/config.py`) + the optimiser in `learn.py::retune`, then `retune`.
4. If it doesn't help, set `form_alpha = 0`.

## 2. Revisit `dc_rho` / goal-model weights as more data arrives

Findings so far: a sweep on the draw-heavy 2026 matchday 1 wanted a large negative
`dc_rho`, but `retune` on the last two full World Cups (128 matches) drives it to
~0 (WC draws fit independent Poisson well) and sets `beta`≈0.5, `mu`≈1.17. The
committed cold-start defaults keep the literature prior (`dc_rho`=−0.13, `beta`=1,
`mu`=1.35); `retune` writes specialised values to `state/params.json`. As 2026
results accumulate, re-run `retune` and decide whether to fold the tuned goal-model
values into the `config.py` defaults. More historical tournaments (Euros, Copa,
qualifiers) would sharpen the tuning — only WC2018/WC2022 are loaded today.

## 3. Knockout bracket — DONE (verify edge cases)

`src/wcpredictor/tournament.py` now defaults to the **real** 2026 Round-of-32
bracket (`R32_2026`, parsed from the official `1A`/`2B`/`3ABCDF` slot codes) with
constraint-matched best-thirds (`_assign_thirds`). `R32_TEMPLATE` is kept as a
tested, self-consistent fallback (see `tests/test_tournament.py`) — do **not**
delete it. Remaining nicety: spot-check the third-place→R32 allocation against
FIFA's published table for a few group-finish permutations.

## 4. Engineering / workflow

- ~~**CI**: add a GitHub Actions workflow to run `pytest` on push.~~ — already
  provided by `.github/workflows/ci.yml` (runs the suite on every push/PR).
- **Ledger desync**: `update-result` *appends* to `data/predictions.csv` while
  `replay` *regenerates* it — mixing them can desync. Add a guardrail or document
  "use one workflow".
- **`fetch_odds.py`** name-matching is now guarded (refuses to overwrite the latest
  view when <50% of returned events match), but the fuzzy `ALIASES` map could still
  drop a team in a low-volume book — keep an eye on the matched-X/Y log line.

## 4b. Closing-line value (deferred — needs data)

`scripts/fetch_odds.py` now appends timestamped snapshots to
`data/odds_history.csv` / `data/outright_history.csv`. A CLV panel (open = first
snapshot, close = last snapshot before kickoff, scored vs result) is **not** wired
yet: 0 of the 29 settled games are priced, and the history only starts
accumulating once `refresh.yml` runs with `ODDS_API_KEY`. Build the panel once a
few priced fixtures have settled with a real open→close curve.

## 5. Modeling fidelity (lower priority)

- **Group tie-breakers**: currently points/GD/GF/Elo — add head-to-head.
- **Home advantage**: only the 3 hosts are flagged; consider modelling large
  travelling support at "neutral" US venues (e.g. Mexico, Argentina).

## Done

- ~~Strength prior (backlog #1)~~ — added `data/strength_prior.csv` from the real
  FIFA men's ranking (`scripts/fetch_strength_prior.py`, official ranking API),
  folded into the seeds via `respread_seeds.py --strength 0.25`. A modest blend
  improved **both** the walk-forward backtest (log-loss 0.896→0.892, Brier down)
  **and** the leak-free prior-vs-market KL (0.0193→~0.018), peaking around w=0.2–0.3
  and degrading past 0.5 — three signals agreeing, so adopted a conservative 0.25.
  Harness: `scripts/validate_strength_prior.py`. (Re-run the fetch + blend when FIFA
  updates its monthly ranking; it is a periodic step, not the daily refresh.)
- ~~Fix the market mis-calibration~~ — root cause was a **sign-flipping** spread
  error (elite over-confidence, mid-tier about right) plus a structurally
  over-concentrated outright. Fixed with two forecast-only, reversible lenses
  (neither touches the learned Elo): a non-linear **spread compression**
  (`Params.spread_slope`/`spread_threshold`, default T=250/s=0.5) that removed the
  elite over-confidence (replay log-loss 0.987→0.971; fitted γ 0.84→0.95), and
  **tournament rating-uncertainty** (`Params.rating_sigma`=150) that pulled the
  outright favourite ~31%→~17% into the bookies' band. Chosen on a leak-free
  prior-vs-market grid (`scripts/validate_prior.py`,
  `scripts/decompose_outright.py`), not fitted to outcomes. Draw rate and
  `k_factor`/form deliberately left alone (decomposition showed they don't drive
  the outright; the draw excess is ~1.6pt structural Poisson).
- ~~Verify the matchday-1 results~~ — replaced hand-compiled data with the official
  FIFA schedule via `wcpredict import-results` (`data/fifa_worldcup_2026_schedule.csv`).
- ~~Dixon–Coles draw correction~~ — implemented; improved replay skill +5.5% → +7.8%.
- ~~Dixon–Coles sampler was slow~~ — replaced the per-match score-grid build with
  exact rejection sampling (~10x faster, identical distribution); a 20k-sim run
  dropped from ~1–2 min to ~14s.
