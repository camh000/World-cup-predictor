"""Reading and writing the canonical CSV data files."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List

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
