#!/usr/bin/env python3
"""Fetch Formula 1 datasets (toUpperCase78/formula1-datasets) into data/f1/.

Pulls the 2026 season results plus 2025 history (for rating priors) from raw
GitHub. No API key needed. Network-only; safe to re-run. Files are renamed to
stable, lower-case names the F1 model expects.
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "f1"
BASE = "https://raw.githubusercontent.com/toUpperCase78/formula1-datasets/master"

# remote filename -> local filename
FILES = {
    "Formula1_2026Season_RaceResults.csv": "race_results_2026.csv",
    "Formula1_2026Season_SprintResults.csv": "sprint_results_2026.csv",
    "Formula1_2026Season_QualifyingResults.csv": "qualifying_2026.csv",
    "Formula1_2025Season_RaceResults.csv": "race_results_2025.csv",
    "Formula1_2025Season_SprintResults.csv": "sprint_results_2025.csv",
    "Formula1_2025Season_Teams.csv": "teams_2025.csv",
    "Formula1_2025Season_Drivers.csv": "drivers_2025.csv",
    "Formula1_2025Season_Calendar.csv": "calendar_2025.csv",
}


def _fetch(remote: str) -> bytes | None:
    url = f"{BASE}/{urllib.parse.quote(remote)}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return resp.read()
    except (urllib.error.URLError, OSError) as e:
        print(f"  ! {remote}: {getattr(e, 'code', e)}")
        return None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ok = 0
    for remote, local in FILES.items():
        data = _fetch(remote)
        if data and len(data) > 20:
            (OUT / local).write_bytes(data)
            print(f"  {local}  ({len(data)} bytes)")
            ok += 1
    print(f"Fetched {ok}/{len(FILES)} F1 files into {OUT}")
    if ok == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
