#!/usr/bin/env python3
"""Fetch the current FIFA men's World Ranking into data/strength_prior.csv.

A real, external strength signal (FIFA ranking points) that can be folded into
the seed ratings via scripts/respread_seeds.py -- but only after it is validated
on the walk-forward backtest against the current seeds (see ground rules). No
numbers are fabricated: the source is FIFA's own ranking API.

The post-2018 FIFA ranking is itself an Elo-style system, so the points are on a
sensible (if differently-scaled) strength axis. We store the rank and the raw
points; respread_seeds.py handles rescaling/blending.

Usage:
    python scripts/fetch_strength_prior.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
import urllib.request
from pathlib import Path

from wcpredictor.data_io import _norm_name, build_name_index, read_teams

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PAGE = "https://inside.fifa.com/fifa-world-ranking/men"
API = "https://inside.fifa.com/api/ranking-overview?locale=en&dateId={}"
UA = {"User-Agent": "Mozilla/5.0"}
FALLBACK_DATE_ID = "id14870"   # a known-good recent ranking, if page scrape fails

# FIFA spellings not already resolved by data_io's aliases + accent-stripping.
ALIASES: dict[str, str] = {}


def _get(url: str) -> str:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", "replace")


def latest_date_id() -> str:
    """Most recent ranking dateId, scraped from the ranking page (max ``idNNNNN``)."""
    try:
        ids = re.findall(r"id\d{5}", _get(PAGE))
        return max(ids, key=lambda s: int(s[2:])) if ids else FALLBACK_DATE_ID
    except Exception:
        return FALLBACK_DATE_ID


def fetch_ranking(date_id: str):
    """Return [(name, rank, points), ...] from the FIFA ranking API."""
    data = json.loads(_get(API.format(date_id)))
    out = []
    for x in data.get("rankings", []):
        it = x.get("rankingItem", {}) or {}
        nm, rank, pts = it.get("name"), it.get("rank"), it.get("totalPoints")
        if nm and rank is not None and pts is not None:
            out.append((nm.strip(), int(rank), float(pts)))
    return out


def main() -> None:
    teams = read_teams(DATA / "teams.csv")
    idx = build_name_index(teams)
    idx.update({_norm_name(k): v for k, v in ALIASES.items()})
    wc_ids = {t.team_id for t in teams}
    name_of = {t.team_id: t.name for t in teams}

    date_id = latest_date_id()
    ranking = fetch_ranking(date_id)
    if not ranking:
        sys.exit(f"error: FIFA ranking API returned no rows for dateId={date_id}")

    matched: dict[str, tuple[int, float]] = {}
    for nm, rank, pts in ranking:
        tid = idx.get(_norm_name(nm))
        if tid in wc_ids and tid not in matched:
            matched[tid] = (rank, pts)

    missing = sorted(wc_ids - set(matched))
    print(f"FIFA ranking dateId={date_id}: {len(ranking)} nations, "
          f"matched {len(matched)}/{len(wc_ids)} WC teams.")
    if missing:
        print("UNMATCHED WC teams:", ", ".join(f"{m} ({name_of[m]})" for m in missing))
        # Show nearest FIFA names to help extend ALIASES.
        fifa_names = [nm for nm, _, _ in ranking]
        for m in missing:
            hint = [n for n in fifa_names if name_of[m].split()[0].lower() in n.lower()]
            if hint:
                print(f"   {m}: candidate FIFA names -> {hint[:4]}")

    rows = sorted(((tid, r, p) for tid, (r, p) in matched.items()), key=lambda x: x[1])
    out = DATA / "strength_prior.csv"
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["team_id", "fifa_rank", "fifa_points"])
        for tid, rank, pts in rows:
            w.writerow([tid, rank, f"{pts:.2f}"])
    print(f"Wrote {out} ({len(rows)} teams).")


if __name__ == "__main__":
    main()
