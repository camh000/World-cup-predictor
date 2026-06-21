#!/usr/bin/env python3
"""Fetch real bookmaker odds from the-odds-api.com into the data/ CSVs.

Credit-thrifty by design (free tier = 500/month):
  * the free ``/sports`` endpoint (0 credits) auto-detects the live World Cup
    sport keys, so we never spend a credit on a wrong or inactive key;
  * one region + one market => the match pull (h2h) costs 1 credit and the
    outright "winner" pull costs 1 credit, so a full refresh is 2 credits;
  * we never overwrite a CSV with an empty result, and we print the remaining
    credit balance returned by the API after every call.

By default it pulls only bet365's prices (the-odds-api ``bookmakers=bet365``,
which costs the same as one region). Set ODDS_API_KEY (a free key from
the-odds-api.com). Without it this no-ops, so it is safe to wire into CI. Usage:

    python scripts/fetch_odds.py                     # bet365 match + outright (2 credits)
    python scripts/fetch_odds.py --skip-outrights    # bet365 match only (1 credit)
    python scripts/fetch_odds.py --bookmaker ''      # best-odds across --regions instead
    python scripts/fetch_odds.py --bookmaker pinnacle  # a different single book
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
from datetime import datetime, timezone
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


def _best_prices(event, bookmaker=None):
    """Best decimal price per outcome name across an event's bookmakers.

    If ``bookmaker`` is given, only that bookmaker's prices are considered (so the
    "best" is simply its quote); otherwise the max across all books is taken.
    """
    best = {}
    for bk in event.get("bookmakers", []):
        if bookmaker and bk.get("key") != bookmaker:
            continue
        for mkt in bk.get("markets", []):
            for oc in mkt.get("outcomes", []):
                nm, price = oc.get("name"), oc.get("price")
                if nm and price and price > best.get(nm, 0.0):
                    best[nm] = price
    return best


def _select_keys(sports):
    """Pick the FIFA World Cup (men's football) match + winner keys from /sports.

    Must require ``fifa_world_cup``, NOT merely ``world_cup``: the-odds-api also
    lists e.g. ``cricket_t20_world_cup_womens``, and matching on ``world_cup``
    alone wrongly grabs cricket (which then yields zero 1X2 rows because cricket
    has no Draw). Qualifier markets are excluded by the ``active`` flag during the
    finals. ``_winner`` is the outright market; anything else is the match market.
    """
    match_key = winner_key = None
    for s in sports:
        key = s.get("key", "")
        if "fifa_world_cup" not in key or not s.get("active"):
            continue
        if key.endswith("_winner"):
            winner_key = winner_key or key
        else:
            match_key = match_key or key
    return match_key, winner_key


def _find_keys(api_key):
    """Return (match_key, winner_key) for the FIFA World Cup, or (None, None)."""
    sports, _ = _get("sports", {"apiKey": api_key})  # free (0 credits)
    return _select_keys(sports)


def fetch_matches(api_key, key, scope, idx, bookmaker=None):
    events, rem = _get(f"sports/{key}/odds",
                       {"apiKey": api_key, **scope, "markets": "h2h",
                        "oddsFormat": "decimal", "dateFormat": "iso"})
    rows = []
    for ev in events:
        h = idx.get(_norm_name(ev.get("home_team", "")))
        a = idx.get(_norm_name(ev.get("away_team", "")))
        if not h or not a:
            continue
        best = _best_prices(ev, bookmaker)
        oh = best.get(ev["home_team"])
        oa = best.get(ev["away_team"])
        od = best.get("Draw")
        if not (oh and od and oa):
            continue
        date = ev.get("commence_time", "")[:10]
        rows.append([date, h, a, round(oh, 3), round(od, 3), round(oa, 3)])
    rows.sort(key=lambda r: (r[0], r[1]))
    return rows, rem, len(events)


def fetch_outrights(api_key, key, scope, idx, bookmaker=None):
    events, rem = _get(f"sports/{key}/odds",
                       {"apiKey": api_key, **scope, "markets": "outrights",
                        "oddsFormat": "decimal"})
    best = {}
    seen_names = set()
    for ev in events:
        for team, price in _best_prices(ev, bookmaker).items():
            seen_names.add(team)
            tid = idx.get(_norm_name(team))
            if tid and price > 0:
                best[tid] = max(price, best.get(tid, 0.0))
    rows = sorted(([tid, round(p, 2)] for tid, p in best.items()), key=lambda r: r[1])
    return rows, rem, len(seen_names)


def _write(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def _append(path, header, rows):
    """Append ``rows`` to ``path``, writing ``header`` first iff the file is new.

    Used to accumulate a timestamped odds history (for closing-line-value) without
    ever destroying past snapshots, in contrast to the overwriting latest view.
    """
    new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(header)
        w.writerows(rows)


def _err(e) -> str:
    """A log-safe description of a network error that NEVER leaks the URL/apiKey."""
    if isinstance(e, urllib.error.HTTPError):
        return f"HTTP {e.code}"
    if isinstance(e, urllib.error.URLError):
        return f"network error ({type(e.reason).__name__})"
    return type(e).__name__


# Errors from a priced call that we catch and turn into a safe no-op (keep existing
# CSVs, spend no further credits). HTTPError (401/429 etc.) and URLError both
# subclass OSError; OSError also covers a bare read-phase TimeoutError /
# ConnectionResetError that is not wrapped in URLError.
_FETCH_ERRORS = (OSError, json.JSONDecodeError)


def _load_dotenv(path: Path = ROOT / ".env") -> None:
    """Load simple ``KEY=value`` lines from a ``.env`` file into ``os.environ``.

    Zero-dependency and deliberately minimal: blank lines and ``#`` comments are
    ignored, surrounding quotes are stripped, and a leading ``export`` is allowed.
    A real environment variable always wins over the file (``setdefault``), and a
    missing file is a quiet no-op — so this never overrides CI secrets and keeps
    the no-key-no-op behaviour intact. ``.env`` is git-ignored (see .gitignore).
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, val)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bookmaker", default="bet365",
                    help="only pull this bookmaker's odds (default bet365); "
                         "pass --bookmaker '' for best-odds across --regions")
    ap.add_argument("--regions", default="uk",
                    help="comma list, used only when --bookmaker is empty; 1 credit PER region")
    ap.add_argument("--skip-outrights", action="store_true", help="match odds only (1 credit)")
    ap.add_argument("--skip-matches", action="store_true", help="outright odds only (1 credit)")
    args = ap.parse_args()

    # the-odds-api takes EITHER a bookmakers filter OR regions. Selecting a single
    # bookmaker costs the same as one region (1 credit per market).
    bookmaker = args.bookmaker or None
    scope = {"bookmakers": bookmaker} if bookmaker else {"regions": args.regions}

    _load_dotenv()
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        print("ODDS_API_KEY not set — skipping odds fetch (no credits spent).")
        return

    idx = build_name_index(read_teams(DATA / "teams.csv"))
    idx.update({_norm_name(k): v for k, v in ALIASES.items()})

    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        match_key, winner_key = _find_keys(api_key)  # free (0 credits)
    except _FETCH_ERRORS as e:
        print(f"Could not list sports ({_err(e)}); aborting to save credits.")
        sys.exit(0)

    remaining = None
    if not args.skip_matches:
        if not match_key:
            print("No active World Cup match market found (kept existing odds).")
        else:
            try:
                rows, remaining, seen = fetch_matches(api_key, match_key, scope, idx, bookmaker)
            except _FETCH_ERRORS as e:
                print(f"Match odds fetch failed ({_err(e)}); kept existing data/odds.csv.")
                rows, seen = [], 0
            header = ["date", "home_team_id", "away_team_id",
                      "odds_home", "odds_draw", "odds_away"]
            if rows and seen and len(rows) < 0.5 * seen:
                print(f"Only matched {len(rows)}/{seen} match events (<50%); kept existing "
                      f"data/odds.csv (suspect name-matching, not overwriting).")
            elif rows:
                _write(DATA / "odds.csv", header, rows)
                _append(DATA / "odds_history.csv", ["fetched_at"] + header,
                        [[fetched_at] + r for r in rows])
                print(f"Wrote data/odds.csv ({len(rows)}/{seen} matches) from '{match_key}' "
                      f"[{bookmaker or args.regions}] + snapshot -> data/odds_history.csv.")
            else:
                print("No match odds returned (kept existing data/odds.csv).")

    if not args.skip_outrights:
        if not winner_key:
            print("No active World Cup winner market found (kept existing outrights).")
        else:
            try:
                rows, remaining, seen = fetch_outrights(api_key, winner_key, scope, idx, bookmaker)
            except _FETCH_ERRORS as e:
                print(f"Outright odds fetch failed ({_err(e)}); kept existing outright_odds.csv.")
                rows, seen = [], 0
            header = ["team_id", "odds_decimal"]
            if rows and seen and len(rows) < 0.5 * seen:
                print(f"Only matched {len(rows)}/{seen} outright teams (<50%); kept existing "
                      f"outright_odds.csv (suspect name-matching, not overwriting).")
            elif rows:
                _write(DATA / "outright_odds.csv", header, rows)
                _append(DATA / "outright_history.csv", ["fetched_at"] + header,
                        [[fetched_at] + r for r in rows])
                print(f"Wrote data/outright_odds.csv ({len(rows)}/{seen} teams) from '{winner_key}' "
                      f"[{bookmaker or args.regions}] + snapshot -> data/outright_history.csv.")
            else:
                print("No outright odds returned (kept existing outright_odds.csv).")

    if remaining is not None:
        print(f"the-odds-api credits remaining this month: {remaining}")


if __name__ == "__main__":
    main()
