#!/usr/bin/env python3
"""Import REAL outright 'World Cup 2026 Winner' odds into data/outright_odds.csv.

Best prices read from oddschecker on 2026-06-20. Converts fractional -> decimal,
resolves team names to ids, and writes (team_id, odds_decimal) for the dashboard's
champion-market comparison.
"""

from __future__ import annotations

import csv
from pathlib import Path

from wcpredictor.data_io import _norm_name, build_name_index, read_teams

ROOT = Path(__file__).resolve().parents[1]
TEAMS = ROOT / "data" / "teams.csv"
OUT = ROOT / "data" / "outright_odds.csv"

# (team, fractional outright odds), oddschecker 2026-06-20.
ODDS = [
    ("France", "6/1"), ("Spain", "13/2"), ("England", "13/2"), ("Argentina", "19/2"),
    ("Portugal", "12/1"), ("Brazil", "14/1"), ("Germany", "15/1"), ("Netherlands", "22/1"),
    ("USA", "33/1"), ("Belgium", "50/1"), ("Mexico", "55/1"), ("Japan", "81/1"),
    ("Uruguay", "130/1"), ("Senegal", "150/1"), ("Croatia", "150/1"), ("Morocco", "150/1"),
    ("Switzerland", "150/1"), ("Norway", "150/1"), ("Canada", "175/1"), ("Ivory Coast", "175/1"),
    ("Ecuador", "175/1"), ("Austria", "175/1"), ("Sweden", "175/1"), ("Australia", "200/1"),
    ("Colombia", "250/1"), ("Turkey", "250/1"), ("South Korea", "325/1"), ("Egypt", "400/1"),
    ("Scotland", "400/1"), ("Paraguay", "450/1"), ("Algeria", "550/1"), ("Ghana", "550/1"),
    ("DR Congo", "800/1"), ("Czech Republic", "800/1"), ("Bosnia and Herzegovina", "800/1"),
    ("Iran", "800/1"), ("Saudi Arabia", "1250/1"), ("Tunisia", "1250/1"), ("New Zealand", "1500/1"),
    ("Panama", "2000/1"), ("Cape Verde", "2000/1"), ("South Africa", "2500/1"), ("Iraq", "2500/1"),
    ("Uzbekistan", "2500/1"), ("Jordan", "3000/1"), ("Qatar", "4000/1"), ("Haiti", "4000/1"),
    ("Curacao", "5000/1"),
]

ALIASES = {"czech republic": "CZE", "usa": "USA", "united states": "USA"}


def dec(frac: str) -> float:
    n, d = frac.split("/")
    return round(int(n) / int(d) + 1.0, 2)


def main() -> None:
    teams = read_teams(TEAMS)
    idx = build_name_index(teams)
    idx.update({_norm_name(k): v for k, v in ALIASES.items()})

    rows, missed = [], []
    for team, frac in ODDS:
        tid = idx.get(_norm_name(team))
        if not tid:
            missed.append(team)
            continue
        rows.append([tid, dec(frac)])

    rows.sort(key=lambda r: r[1])
    with OUT.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["team_id", "odds_decimal"])
        w.writerows(rows)
    print(f"Wrote {OUT} ({len(rows)} teams)")
    if missed:
        print("UNMATCHED:", ", ".join(missed))


if __name__ == "__main__":
    main()
