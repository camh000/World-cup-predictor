"""Reading and writing the canonical CSV data files."""

from __future__ import annotations

import csv
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

from .ratings import Rating


RESULTS_HEADER = [
    "date", "home_team_id", "away_team_id",
    "home_goals", "away_goals", "stage", "competition", "neutral",
]


@dataclass(frozen=True)
class Team:
    team_id: str
    name: str
    confederation: str
    group: str
    host: bool = False


@dataclass(frozen=True)
class MatchRecord:
    date: str
    home_team_id: str
    away_team_id: str
    home_goals: int
    away_goals: int
    stage: str = "friendly"
    competition: str = ""
    neutral: bool = True


def _to_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def read_teams(path: Path) -> List[Team]:
    teams: List[Team] = []
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            teams.append(
                Team(
                    team_id=row["team_id"].strip(),
                    name=row["name"].strip(),
                    confederation=row["confederation"].strip(),
                    group=row["group"].strip(),
                    host=_to_bool(row.get("host", "")),
                )
            )
    return teams


def read_seed_ratings(path: Path) -> Dict[str, Rating]:
    path = Path(path)
    out: Dict[str, Rating] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            out[row["team_id"].strip()] = Rating(
                elo=float(row["elo"]),
                attack=float(row.get("attack", 0) or 0),
                defense=float(row.get("defense", 0) or 0),
            )
    return out


def read_matches(path: Path) -> List[MatchRecord]:
    path = Path(path)
    out: List[MatchRecord] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if not row.get("home_team_id"):
                continue
            out.append(
                MatchRecord(
                    date=row.get("date", ""),
                    home_team_id=row["home_team_id"].strip(),
                    away_team_id=row["away_team_id"].strip(),
                    home_goals=int(row["home_goals"]),
                    away_goals=int(row["away_goals"]),
                    stage=row.get("stage", "friendly").strip() or "friendly",
                    competition=row.get("competition", "").strip(),
                    neutral=_to_bool(row.get("neutral", "true")),
                )
            )
    return out


def ensure_results_file(path: Path) -> None:
    path = Path(path)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerow(RESULTS_HEADER)


# Names used by the official FIFA schedule that differ from data/teams.csv names.
TEAM_NAME_ALIASES = {
    "korea republic": "South Korea",
    "turkiye": "Turkey",
    "cote divoire": "Ivory Coast",
    "cabo verde": "Cape Verde",
    "congo dr": "DR Congo",
    "ir iran": "Iran",
    "usa": "United States",
}

HOST_TEAM_IDS = {"MEX", "USA", "CAN"}


def _norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return "".join(c for c in s.lower() if c.isalnum())


def build_name_index(teams: List[Team]) -> Dict[str, str]:
    """Map normalised team names / ids (incl. known aliases) to team ids."""
    idx: Dict[str, str] = {}
    for t in teams:
        idx[_norm_name(t.name)] = t.team_id
        idx[_norm_name(t.team_id)] = t.team_id
    for alias, canonical in TEAM_NAME_ALIASES.items():
        target = idx.get(_norm_name(canonical))
        if target:
            idx[_norm_name(alias)] = target
    return idx


def read_official_schedule(
    path: Path,
    teams: List[Team],
    *,
    allow_unknown: bool = False,
    host_ids: Optional[set] = None,
    competition: str = "WC2026",
) -> List[MatchRecord]:
    """Parse an official FIFA-format schedule CSV into played ``MatchRecord``s.

    Expects columns: Match Number, Round Number, Date (DD/MM/YYYY HH:MM),
    Location, Home Team, Away Team, Group, Result ("H - A"). Rows without a
    numeric result are skipped (so the same file can be re-imported as more
    matches are played).

    ``allow_unknown`` keeps matches involving teams not in ``teams`` by assigning
    a generated id from the team name (used for historical tournaments whose
    fields differ from 2026). ``host_ids`` is the set of team ids treated as
    playing at home (non-neutral); pass an empty set for historical World Cups,
    whose hosts differ from 2026.
    """
    if host_ids is None:
        host_ids = HOST_TEAM_IDS
    idx = build_name_index(teams)

    def resolve(name: str) -> Optional[str]:
        tid = idx.get(_norm_name(name))
        if tid:
            return tid
        if allow_unknown and name.strip():
            return _norm_name(name).upper()
        return None

    out: List[MatchRecord] = []
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            result = (row.get("Result") or "").strip()
            if "-" not in result:
                continue
            home_id = resolve(row.get("Home Team", ""))
            away_id = resolve(row.get("Away Team", ""))
            if not home_id or not away_id:
                continue
            try:
                hg, ag = (int(x) for x in result.replace("–", "-").split("-"))
            except ValueError:
                continue
            out.append(
                MatchRecord(
                    date=_parse_date(row.get("Date", "")),
                    home_team_id=home_id,
                    away_team_id=away_id,
                    home_goals=hg,
                    away_goals=ag,
                    stage="group" if (row.get("Round Number", "").strip() in {"1", "2", "3"})
                          else (row.get("Round Number", "").strip().lower() or "knockout"),
                    competition=competition,
                    # A host playing at home is not a neutral-venue game.
                    neutral=home_id not in host_ids,
                )
            )
    out.sort(key=lambda m: (m.date, m.home_team_id))
    return out


def _parse_date(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    for fmt in ("%d/%m/%Y %H:%M", "%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            continue
    return value[:10]


def write_results(path: Path, records: List[MatchRecord]) -> None:
    """Overwrite the results log with ``records`` (header + rows)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(RESULTS_HEADER)
        for r in records:
            writer.writerow([r.date, r.home_team_id, r.away_team_id, r.home_goals,
                             r.away_goals, r.stage, r.competition, str(r.neutral).lower()])


def append_result(path: Path, record: MatchRecord) -> None:
    """Append one real result to the results log (creating it if needed)."""
    ensure_results_file(path)
    with Path(path).open("a", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerow(
            [
                record.date or date.today().isoformat(),
                record.home_team_id,
                record.away_team_id,
                record.home_goals,
                record.away_goals,
                record.stage,
                record.competition,
                str(record.neutral).lower(),
            ]
        )
