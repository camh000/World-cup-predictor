#!/usr/bin/env python3
"""Import REAL bookmaker odds (oddschecker, World Cup 2026) into data/odds.csv.

Fractional 1X2 prices were read from oddschecker on 2026-06-20 for the upcoming
group fixtures. This converts them to decimal, resolves team names to ids and the
fixture date from the official schedule, and writes the odds file consumed by the
dashboard's betting analysis. Re-run scripts/make_sample_odds.py to fall back to
the synthetic demo.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from wcpredictor.data_io import _norm_name, build_name_index, read_teams

ROOT = Path(__file__).resolve().parents[1]
SCHED = ROOT / "data" / "fifa_worldcup_2026_schedule.csv"
TEAMS = ROOT / "data" / "teams.csv"
OUT = ROOT / "data" / "odds.csv"

# Source: oddschecker.com, captured 2026-06-20. (home, away, frac_home, frac_draw, frac_away)
ODDS = [
    ("Netherlands", "Sweden", "3/4", "31/10", "41/10"),
    ("Germany", "Ivory Coast", "8/15", "19/5", "27/5"),
    ("Ecuador", "Curacao", "15/100", "10/1", "25/1"),
    ("Tunisia", "Japan", "6/1", "31/10", "8/13"),
    ("Spain", "Saudi Arabia", "1/8", "11/1", "30/1"),
    ("Belgium", "Iran", "9/20", "19/5", "15/2"),
    ("Uruguay", "Cape Verde", "9/20", "7/2", "17/2"),
    ("New Zealand", "Egypt", "26/5", "33/10", "13/20"),
    ("Argentina", "Austria", "7/12", "3/1", "5/1"),
    ("France", "Iraq", "1/10", "11/1", "33/1"),
    ("Norway", "Senegal", "13/10", "5/2", "11/5"),
    ("Jordan", "Algeria", "5/1", "31/10", "3/5"),
    ("Portugal", "Uzbekistan", "2/9", "6/1", "29/2"),
    ("England", "Ghana", "3/13", "11/2", "14/1"),
    ("Panama", "Croatia", "23/4", "31/10", "6/11"),
    ("Colombia", "DR Congo", "1/2", "10/3", "13/2"),
    ("Switzerland", "Canada", "29/20", "11/5", "47/20"),
    ("Bosnia and Herzegovina", "Qatar", "40/85", "4/1", "13/2"),
    ("Morocco", "Haiti", "2/9", "6/1", "17/1"),
    ("Scotland", "Brazil", "8/1", "17/4", "2/5"),
    ("South Africa", "South Korea", "5/1", "29/10", "4/6"),
    ("Czech Republic", "Mexico", "29/10", "14/5", "21/20"),
    ("Curacao", "Ivory Coast", "22/1", "17/2", "1/7"),
    ("Ecuador", "Germany", "4/1", "29/10", "7/10"),
    ("Japan", "Sweden", "6/5", "12/5", "5/2"),
    ("Tunisia", "Netherlands", "39/4", "9/2", "1/3"),
    ("Paraguay", "Australia", "182/100", "29/20", "29/10"),
    ("Turkey", "USA", "11/5", "3/1", "23/20"),
    ("Norway", "France", "17/5", "14/5", "5/6"),
    ("Senegal", "Iraq", "3/10", "9/2", "21/2"),
    ("Uruguay", "Spain", "11/2", "7/2", "4/7"),
    ("Cape Verde", "Saudi Arabia", "9/5", "27/10", "6/4"),
    ("New Zealand", "Belgium", "14/1", "11/2", "1/4"),
    ("Egypt", "Iran", "11/10", "9/4", "3/1"),
    ("Panama", "England", "12/1", "6/1", "1/4"),
    ("Croatia", "Ghana", "16/25", "29/10", "21/4"),
]

ALIASES = {"czech republic": "CZE", "usa": "USA", "united states": "USA"}


def dec(frac: str) -> float:
    n, d = frac.split("/")
    return round(int(n) / int(d) + 1.0, 3)


def main() -> None:
    teams = read_teams(TEAMS)
    idx = build_name_index(teams)
    idx.update({_norm_name(k): v for k, v in ALIASES.items()})

    # (home_id, away_id) -> iso date, from unplayed schedule rows.
    when = {}
    with SCHED.open("r", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            h = idx.get(_norm_name(r.get("Home Team", "")))
            a = idx.get(_norm_name(r.get("Away Team", "")))
            if not h or not a:
                continue
            try:
                dt = datetime.strptime(r["Date"].strip(), "%d/%m/%Y %H:%M")
            except ValueError:
                continue
            when[(h, a)] = dt.strftime("%Y-%m-%d")

    rows, missed = [], []
    for home, away, fh, fd, fa in ODDS:
        h = idx.get(_norm_name(home))
        a = idx.get(_norm_name(away))
        if not h or not a:
            missed.append(f"name: {home} v {away}")
            continue
        date = when.get((h, a))
        if not date:
            missed.append(f"fixture: {home} v {away} ({h} v {a})")
            continue
        rows.append([date, h, a, dec(fh), dec(fd), dec(fa)])

    rows.sort(key=lambda r: (r[0], r[1]))
    with OUT.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "home_team_id", "away_team_id",
                    "odds_home", "odds_draw", "odds_away"])
        w.writerows(rows)
    print(f"Wrote {OUT} ({len(rows)} real markets)")
    if missed:
        print("UNMATCHED:")
        for m in missed:
            print("  -", m)


if __name__ == "__main__":
    main()
