#!/usr/bin/env python3
"""Fetch recent real international matches into data/historical_matches.csv (for retune).

Source: martj42/international_results (a free, well-maintained CSV of every men's
international since 1872). We pull matches since 2023 between two 2026-WC teams,
EXCLUDING "FIFA World Cup" games — those would leak the 2026 results that already
live in data/results.csv (and duplicate the WC2018/2022 finals already in the
historical file). Qualifiers, Nations League, friendlies, Euros, Copa, AFCON etc.
are exactly the extra goal-distribution signal `retune` wants.

Merges into the existing historical_matches.csv (dedup by date+teams), preserving
the committed WC2018/2022 entries. No fabrication — only real recorded results.

Usage:
    python scripts/fetch_history.py
"""

from __future__ import annotations

import csv
import io
import urllib.request
from pathlib import Path

from wcpredictor.data_io import RESULTS_HEADER, _norm_name, build_name_index, read_matches, read_teams

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SRC = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
SINCE = "2023-01-01"
EXCLUDE_TOURNAMENTS = {"FIFA World Cup"}   # leak: 2026 results.csv + WC2018/2022 file


def main() -> None:
    teams = read_teams(DATA / "teams.csv")
    idx = build_name_index(teams)
    wc = {t.team_id for t in teams}

    raw = urllib.request.urlopen(SRC, timeout=90).read().decode("utf-8", "replace")
    src_rows = list(csv.DictReader(io.StringIO(raw)))

    fetched = []
    kept_tourn = {}
    for r in src_rows:
        if r["date"] < SINCE or r["tournament"] in EXCLUDE_TOURNAMENTS:
            continue
        h = idx.get(_norm_name(r["home_team"]))
        a = idx.get(_norm_name(r["away_team"]))
        if h not in wc or a not in wc:
            continue
        try:
            hg, ag = int(r["home_score"]), int(r["away_score"])
        except (ValueError, KeyError):
            continue
        neutral = str(r.get("neutral", "")).strip().lower() in {"true", "1", "yes"}
        fetched.append([r["date"], h, a, hg, ag, "international", r["tournament"], str(neutral).lower()])
        kept_tourn[r["tournament"]] = kept_tourn.get(r["tournament"], 0) + 1

    # Merge with the existing historical file (keep WC2018/2022), dedup by date+teams.
    hist_path = DATA / "historical_matches.csv"
    existing = read_matches(hist_path)
    seen = {(m.date, m.home_team_id, m.away_team_id) for m in existing}
    merged = list(existing)
    added = 0
    new_records = []
    for row in fetched:
        key = (row[0], row[1], row[2])
        if key in seen:
            continue
        seen.add(key)
        new_records.append(row)
        added += 1

    out_rows = ([[m.date, m.home_team_id, m.away_team_id, m.home_goals, m.away_goals,
                  m.stage, m.competition, str(m.neutral).lower()] for m in merged]
                + new_records)
    out_rows.sort(key=lambda r: (r[0], r[1]))
    with hist_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(RESULTS_HEADER)
        w.writerows(out_rows)

    print(f"Source {len(src_rows)} rows; since {SINCE}, {len(fetched)} WC-vs-WC non-WC matches.")
    print("By tournament:", dict(sorted(kept_tourn.items(), key=lambda x: -x[1])))
    print(f"Existing historical: {len(existing)}; added {added} new; total {len(out_rows)} -> {hist_path}")


if __name__ == "__main__":
    main()
