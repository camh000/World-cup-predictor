#!/usr/bin/env python3
"""Fetch the real F1 season calendar into data/f1/calendar_<year>.csv.

The toUpperCase78 dataset we use for results has no 2026 calendar, so the model
was hardcoding TOTAL_RACES=24 and ignoring sprints. f1db (the maintained
Ergast-successor DB, via GitHub releases) carries the full calendar with sprint
flags. The calendar is static for a season, so this is a periodic step (re-run if
the schedule changes) — not part of the daily refresh.

Columns: round, date, grand_prix, sprint (true/false).

Usage:
    python scripts/fetch_f1_calendar.py            # 2026
    python scripts/fetch_f1_calendar.py 2027
"""

from __future__ import annotations

import csv
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "f1"
URL = "https://github.com/f1db/f1db/releases/latest/download/f1db-csv.zip"


def main() -> None:
    year = sys.argv[1] if len(sys.argv) > 1 else "2026"
    raw = urllib.request.urlopen(
        urllib.request.Request(URL, headers={"User-Agent": "wcpredictor/1.0"}), timeout=120).read()
    z = zipfile.ZipFile(io.BytesIO(raw))
    races = [r for r in csv.DictReader(io.TextIOWrapper(z.open("f1db-races.csv"), "utf-8"))
             if r["year"] == year]
    if not races:
        sys.exit(f"error: f1db has no {year} calendar")
    races.sort(key=lambda r: int(r["round"]))
    out = OUT / f"calendar_{year}.csv"
    OUT.mkdir(parents=True, exist_ok=True)
    n_sprint = 0
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["round", "date", "grand_prix", "sprint"])
        for r in races:
            sprint = bool((r.get("sprintQualifyingFormat") or "").strip())
            n_sprint += sprint
            w.writerow([r["round"], r["date"], r["grandPrixId"], str(sprint).lower()])
    print(f"Wrote {out}: {len(races)} rounds ({n_sprint} sprints) for {year}.")


if __name__ == "__main__":
    main()
