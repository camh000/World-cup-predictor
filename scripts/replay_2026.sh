#!/usr/bin/env bash
#
# Replay every already-played 2026 World Cup match (data/results.csv) through the
# model with the current weights, to validate how accurate the predictions are.
#
# Walk-forward: each match is forecast from the ratings as they stood *before*
# it, the forecast is scored against the real result, then the model learns from
# it and moves to the next game. Prints a play-by-play and an accuracy summary,
# then shows the updated title odds given everything that has happened so far.
#
# Usage:
#   scripts/replay_2026.sh                 # full play-by-play + accuracy + odds
#   scripts/replay_2026.sh --quiet         # accuracy summary only
#   scripts/replay_2026.sh --no-save       # dry run, don't persist ratings
#
set -euo pipefail

echo "=== Replaying played 2026 matches with current model weights ==="
wcpredict replay "$@"

# Only re-simulate when the replay persisted ratings (i.e. not a dry run).
if [[ " $* " != *" --no-save "* ]]; then
  echo
  echo "=== Updated title odds given results so far ==="
  wcpredict simulate --sims 10000 --seed 42 --top 12
fi
