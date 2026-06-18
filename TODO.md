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

## 2. Re-tune `dc_rho` once more results exist

Dixon–Coles `dc_rho` defaults to the literature value −0.13. A sweep on the
draw-heavy matchday 1 preferred a much larger magnitude (overfitting one round).
Once more matchdays / historical data are loaded, tune it via `retune`
(it is in `TUNABLE_FIELDS`) rather than by hand, and re-check it isn't overfit.

## 3. Build the real knockout bracket from the official schedule

`src/wcpredictor/tournament.py` uses a *placeholder* Round-of-32 template and a
simplified best-third assignment. The official schedule
(`data/fifa_worldcup_2026_schedule.csv`) encodes the real bracket: knockout rows
have slot codes for Home/Away (`1A`=winner A, `2B`=runner-up B, `3ABCDF`=a
specific third-place combination). Parse those to reproduce FIFA's exact
third-place→R32 mapping and bracket paths.

## 4. Engineering / workflow

- **Open a PR** for branch `claude/world-cup-predictor-xhcezi` (none exists yet).
- **CI**: add a GitHub Actions workflow to run `pytest` on push.
- **Ledger desync**: `update-result` *appends* to `data/predictions.csv` while
  `replay` *regenerates* it — mixing them can desync. Add a guardrail or document
  "use one workflow".
- **`fetch.py`** is unexercised (no API key) and its team-name matching is fuzzy;
  could silently drop matches. Now that `import-results` exists, consider whether
  the live fetcher is still needed.

## 5. Modeling fidelity (lower priority)

- **Group tie-breakers**: currently points/GD/GF/Elo — add head-to-head.
- **Home advantage**: only the 3 hosts are flagged; consider modelling large
  travelling support at "neutral" US venues (e.g. Mexico, Argentina).

## Done

- ~~Verify the matchday-1 results~~ — replaced hand-compiled data with the official
  FIFA schedule via `wcpredict import-results` (`data/fifa_worldcup_2026_schedule.csv`).
- ~~Dixon–Coles draw correction~~ — implemented; improved replay skill +5.5% → +7.8%.
