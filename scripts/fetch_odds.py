#!/usr/bin/env python3
"""Fetch real bookmaker odds from the-odds-api.com into the data/ CSVs.

Credit-thrifty by design (free tier = 500/month):
  * the free ``/sports`` endpoint (0 credits) auto-detects the live World Cup
    sport keys, so we never spend a credit on a wrong or inactive key;
  * one region + one market => the match pull (h2h) costs 1 credit and the
    outright "winner" pull costs 1 credit, so a full refresh is 2 credits;
  * we never overwrite a CSV with an empty result, and we print the remaining
    credit balance returned by the API after every call.

Set ODDS_API_KEY (a free key from the-odds-api.com). Without it this no-ops, so
it is safe to wire into CI. Usage:

    python scripts/fetch_odds.py                # match + outright (2 credits)
    python scripts/fetch_odds.py --skip-outrights   # match only (1 credit)
    python scripts/fetch_odds.py --regions uk,eu    # (costs 1 credit PER region)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from wcpredictor.data_io import _norm_name, build_name_index, read_teams

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BASE = "https://api.the-odds-api.com/v4"

# the-odds-api uses long country names; map the handful that differ from teams.csv.
ALIASES = {"czech republic": "CZE", "usa": "USA", "united states": "USA",
           "south korea": "KOR", "korea republic": "KOR", "ivory coast": "CIV",
           "cote divoire": "CIV", "dr congo": "COD", "cape verde": "CPV",
           "turkiye": "TUR"}


def _get(path: str, params: dict):
    """GET JSON; returns (data, remaining_credits). Raises on HTTP error."""
    url = f"{BASE}/{path}?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=20) as resp:
        remaining = resp.headers.get("x-requests-remaining")
        return json.loads(resp.read().decode()), remaining


def _best_prices(event):
    """Best (max) decimal price per outcome name across all bookmakers in an event."""
    best = {}
    for bk in event.get("bookmakers", []):
        for mkt in bk.get("markets", []):
            for oc in mkt.get("outcomes", []):
                nm, price = oc.get("name"), oc.get("price")
                if nm and price and price > best.get(nm, 0.0):
                    best[nm] = price
    return best


def _find_keys(api_key):
    """Return (match_key, winner_key) for the World Cup, or (None, None)."""
    sports, _ = _get("sports", {"apiKey": api_key})
    match_key = winner_key = None
    for s in sports:
        key = s.get("key", "")
        if "world_cup" not in key or not s.get("active"):
            continue
        if key.endswith("_winner"):
            winner_key = winner_key or key
        else:
            match_key = match_key or key
    return match_key, winner_key


def fetch_matches(api_key, key, regions, idx):
    events, rem = _get(f"sports/{key}/odds",
                       {"apiKey": api_key, "regions": regions, "markets": "h2h",
                        "oddsFormat": "decimal", "dateFormat": "iso"})
    rows = []
    for ev in events:
        h = idx.get(_norm_name(ev.get("home_team", "")))
        a = idx.get(_norm_name(ev.get("away_team", "")))
        if not h or not a:
            continue
        best = _best_prices(ev)
        oh = best.get(ev["home_team"])
        oa = best.get(ev["away_team"])
        od = best.get("Draw")
        if not (oh and od and oa):
            continue
        date = ev.get("commence_time", "")[:10]
        rows.append([date, h, a, round(oh, 3), round(od, 3), round(oa, 3)])
    rows.sort(key=lambda r: (r[0], r[1]))
    return rows, rem


def fetch_outrights(api_key, key, regions, idx):
    events, rem = _get(f"sports/{key}/odds",
                       {"apiKey": api_key, "regions": regions, "markets": "outrights",
                        "oddsFormat": "decimal"})
    best = {}
    for ev in events:
        for team, price in _best_prices(ev).items():
            tid = idx.get(_norm_name(team))
            if tid and price > 0:
                best[tid] = max(price, best.get(tid, 0.0))
    rows = sorted(([tid, round(p, 2)] for tid, p in best.items()), key=lambda r: r[1])
    return rows, rem


def _write(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", default="uk", help="comma list; costs 1 credit PER region")
    ap.add_argument("--skip-outrights", action="store_true", help="match odds only (1 credit)")
    ap.add_argument("--skip-matches", action="store_true", help="outright odds only (1 credit)")
    args = ap.parse_args()

    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        print("ODDS_API_KEY not set — skipping odds fetch (no credits spent).")
        return

    idx = build_name_index(read_teams(DATA / "teams.csv"))
    idx.update({_norm_name(k): v for k, v in ALIASES.items()})

    try:
        match_key, winner_key = _find_keys(api_key)  # free
    except urllib.error.HTTPError as e:
        print(f"Could not list sports (HTTP {e.code}); aborting to save credits.")
        sys.exit(0)

    remaining = None
    if not args.skip_matches:
        if match_key:
            rows, remaining = fetch_matches(api_key, match_key, args.regions, idx)
            if rows:
                _write(DATA / "odds.csv",
                       ["date", "home_team_id", "away_team_id",
                        "odds_home", "odds_draw", "odds_away"], rows)
                print(f"Wrote data/odds.csv ({len(rows)} matches) from '{match_key}'.")
            else:
                print("No match odds returned (kept existing data/odds.csv).")
        else:
            print("No active World Cup match market found (kept existing odds).")

    if not args.skip_outrights:
        if winner_key:
            rows, remaining = fetch_outrights(api_key, winner_key, args.regions, idx)
            if rows:
                _write(DATA / "outright_odds.csv", ["team_id", "odds_decimal"], rows)
                print(f"Wrote data/outright_odds.csv ({len(rows)} teams) from '{winner_key}'.")
            else:
                print("No outright odds returned (kept existing outright_odds.csv).")
        else:
            print("No active World Cup winner market found (kept existing outrights).")

    if remaining is not None:
        print(f"the-odds-api credits remaining this month: {remaining}")


if __name__ == "__main__":
    main()
