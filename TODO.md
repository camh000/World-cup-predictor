# TODO

## Validate & tune the tournament-form overlay (once matchday 2+ is played)

The `form` overlay (see "Tournament form" in `README.md`) is implemented with
**unvalidated default weights** (`form_alpha=0.10`, `form_decay=0.85`, bounds
0.85–1.15 in `src/wcpredictor/config.py`). It is a no-op on matchday 1 because a
team's form only changes *after* it plays, so it has not yet affected any real
forecast.

When matchday 2 (and 3) results are in:

1. Add the new results to `data/results.csv`.
2. A/B test it: `wcpredict replay` vs `wcpredict replay --no-form` and compare
   log-loss / Brier. Keep form only if it genuinely improves accuracy.
3. If it helps, let the data choose its strength rather than guessing: add
   `form_alpha` / `form_decay` to `TUNABLE_FIELDS` in `src/wcpredictor/config.py`
   and the optimiser in `src/wcpredictor/learn.py::retune`, then `wcpredict retune`.
4. If it does not help, set `form_alpha = 0` (disables it) or remove the overlay.
