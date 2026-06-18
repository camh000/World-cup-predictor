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
- ~~Dixon–Coles sampler was slow~~ — replaced the per-match score-grid build with
  exact rejection sampling (~10x faster, identical distribution); a 20k-sim run
  dropped from ~1–2 min to ~14s.
