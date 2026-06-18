"""Generate an .ics calendar file of World Cup fixtures for a date window.

Reads the official schedule (data/fifa_worldcup_2026_schedule.csv), filters to a
time window, and writes an iCalendar (.ics) file that imports straight into Apple
Calendar (or Google/Outlook). Kickoff times in the schedule are UTC; events are
written in UTC with a trailing 'Z', so calendar apps localise them automatically.

Usage:
    python scripts/make_calendar.py                       # next 7 days
    python scripts/make_calendar.py --from 2026-06-26 --days 7
    python scripts/make_calendar.py --out my_fixtures.ics --reminder 60
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path

SCHEDULE = Path("data/fifa_worldcup_2026_schedule.csv")
MATCH_FMT = "%d/%m/%Y %H:%M"  # same format as data_io._parse_date
MATCH_MINUTES = 120           # event length


def _ics_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace(",", "\\,").replace(";", "\\;").replace("\n", "\\n")


def build_calendar(rows, reminder_min: int) -> str:
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//wcpredictor//World Cup 2026//EN",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        "X-WR-CALNAME:World Cup 2026",
    ]
    for dt, num, home, away, group, venue in rows:
        start = dt.strftime("%Y%m%dT%H%M%SZ")
        end = (dt + timedelta(minutes=MATCH_MINUTES)).strftime("%Y%m%dT%H%M%SZ")
        summary = _ics_escape(f"⚽ {home} v {away} (WC {group})")
        desc = _ics_escape(f"FIFA World Cup 2026 - {group}\n{home} vs {away}")
        lines += [
            "BEGIN:VEVENT",
            f"UID:wc2026-match{num}@worldcup",
            f"DTSTAMP:{stamp}",
            f"DTSTART:{start}",
            f"DTEND:{end}",
            f"SUMMARY:{summary}",
            f"LOCATION:{_ics_escape(venue)}",
            f"DESCRIPTION:{desc}",
        ]
        if reminder_min > 0:
            lines += [
                "BEGIN:VALARM",
                "ACTION:DISPLAY",
                f"DESCRIPTION:{summary}",
                f"TRIGGER:-PT{reminder_min}M",
                "END:VALARM",
            ]
        lines.append("END:VEVENT")
    lines.append("END:VCALENDAR")
    # RFC 5545 requires CRLF line endings.
    return "\r\n".join(lines) + "\r\n"


def load_rows(schedule: Path, start: datetime, end: datetime, rounds=None):
    rows = []
    with schedule.open("r", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if rounds and r.get("Round Number", "").strip() not in rounds:
                continue
            try:
                dt = datetime.strptime(r["Date"].strip(), MATCH_FMT)
            except (ValueError, KeyError):
                continue
            home, away = r.get("Home Team", "").strip(), r.get("Away Team", "").strip()
            if not home or not away or "announce" in home.lower():
                continue  # skip TBD knockout placeholders
            if start <= dt <= end:
                rows.append((dt, r["Match Number"].strip(), home, away,
                             r.get("Group", "").strip(), r.get("Location", "").strip()))
    rows.sort(key=lambda x: (x[0], x[1]))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate an .ics of World Cup fixtures")
    ap.add_argument("--from", dest="frm", default="2026-06-18T10:00",
                    help="window start, UTC (default 2026-06-18T10:00 = 11:00 BST)")
    ap.add_argument("--days", type=int, default=7, help="window length in days (default 7)")
    ap.add_argument("--to", dest="to", default=None,
                    help="explicit window end date (overrides --days), e.g. 2026-06-29")
    ap.add_argument("--rounds", default=None,
                    help="comma-separated Round Number filter, e.g. '1,2,3' for the group stage")
    ap.add_argument("--out", default="worldcup_next_week.ics", help="output .ics path")
    ap.add_argument("--reminder", type=int, default=30, help="reminder minutes before (0 = none)")
    ap.add_argument("--schedule", default=str(SCHEDULE), help="schedule CSV path")
    args = ap.parse_args()

    start = datetime.fromisoformat(args.frm)
    # Include all of the final calendar day so an evening kickoff isn't cut off.
    end_anchor = datetime.fromisoformat(args.to) if args.to else start + timedelta(days=args.days)
    end = datetime.combine(end_anchor.date(), datetime.max.time())
    rounds = {r.strip() for r in args.rounds.split(",")} if args.rounds else None
    rows = load_rows(Path(args.schedule), start, end, rounds)
    if not rows:
        raise SystemExit(f"No fixtures between {start} and {end} in {args.schedule}")

    with open(args.out, "w", encoding="utf-8", newline="") as fh:  # preserve CRLF
        fh.write(build_calendar(rows, args.reminder))
    print(f"Wrote {len(rows)} events to {args.out}")
    print(f"  window: {start:%a %d %b %H:%MZ} -> {end:%a %d %b %H:%MZ}")
    print(f"  first:  {rows[0][0]:%a %d %b %H:%MZ}  {rows[0][2]} v {rows[0][3]}")
    print(f"  last:   {rows[-1][0]:%a %d %b %H:%MZ}  {rows[-1][2]} v {rows[-1][3]}")


if __name__ == "__main__":
    main()
