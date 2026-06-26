#!/usr/bin/env python3
"""Decide whether a bookmaker-odds snapshot is due *right now*.

Exit code 0 ("snapshot now") iff a fixture kicks off within ``--within`` minutes
AND no odds snapshot has been taken in the last ``--min-gap`` minutes; otherwise
exit 1. The live workflow calls this to capture the *closing line* just before
each game (the only snapshot that makes closing-line-value meaningful) while
spending the fewest credits — bookmakers also tend not to post a match market
until kick-off is near, so an off-peak fetch usually returns nothing anyway.

Schedule kick-off times are treated as UTC (the dashboard renders them with a
``Z`` suffix). ``--now`` overrides the clock for testing.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def parse_kickoffs(schedule_path) -> List[datetime]:
    """All kick-off datetimes (UTC) from the official-format schedule CSV."""
    out: List[datetime] = []
    path = Path(schedule_path)
    if not path.exists():
        return out
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            raw = (row.get("Date") or "").strip()
            try:
                out.append(datetime.strptime(raw, "%d/%m/%Y %H:%M").replace(tzinfo=timezone.utc))
            except ValueError:
                continue
    return out


def last_snapshot(history_path) -> Optional[datetime]:
    """Timestamp of the most recent odds snapshot, or ``None`` if there is none."""
    path = Path(history_path)
    if not path.exists():
        return None
    latest: Optional[datetime] = None
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            ts = (row.get("fetched_at") or "").strip()
            try:
                dt = datetime.fromisoformat(ts)
            except ValueError:
                continue
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if latest is None or dt > latest:
                latest = dt
    return latest


def should_snapshot(now: datetime, kickoffs: List[datetime], last_snap: Optional[datetime],
                    within_min: int, min_gap_min: int) -> Tuple[bool, str]:
    """Return (snapshot_now, human-readable reason)."""
    window_end = now + timedelta(minutes=within_min)
    imminent = sorted(k for k in kickoffs if now <= k <= window_end)
    if not imminent:
        return False, f"no kick-off within the next {within_min} min"
    if last_snap is not None and (now - last_snap) < timedelta(minutes=min_gap_min):
        ago = (now - last_snap).total_seconds() / 60
        return False, f"a snapshot was taken {ago:.0f} min ago (< {min_gap_min} min gap)"
    nxt = imminent[0]
    mins = (nxt - now).total_seconds() / 60
    return True, f"kick-off at {nxt.isoformat()} (in {mins:.0f} min)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--within", type=int, default=35,
                    help="snapshot if a game kicks off within this many minutes")
    ap.add_argument("--min-gap", type=int, default=50,
                    help="don't snapshot if one was taken within this many minutes")
    ap.add_argument("--schedule", default=str(DATA / "fifa_worldcup_2026_schedule.csv"))
    ap.add_argument("--history", default=str(DATA / "odds_history.csv"))
    ap.add_argument("--now", default=None, help="ISO-8601 clock override (testing)")
    args = ap.parse_args()

    now = datetime.fromisoformat(args.now) if args.now else datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    go, why = should_snapshot(now, parse_kickoffs(args.schedule),
                              last_snapshot(args.history), args.within, args.min_gap)
    print(("SNAPSHOT: " if go else "SKIP: ") + why)
    sys.exit(0 if go else 1)


if __name__ == "__main__":
    main()
