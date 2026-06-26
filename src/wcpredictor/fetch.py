"""Optional live-results fetcher.

This module is entirely optional. The engine works fully offline; this just
lets you auto-populate ``data/results.csv`` from a public football API. It
imports ``requests`` lazily so the core package has zero network dependencies,
and it fails with a clear message when the extra or API key is missing.

Currently supports football-data.org (set FOOTBALL_DATA_API_KEY). The provider's
team names are mapped to local ids via ``data/teams.csv`` names, with accent
folding plus an alias table for the handful of names that differ (e.g.
"Türkiye" -> TUR, "Korea Republic" -> KOR), and its stage labels are normalised
to match the rest of the pipeline (so group games read ``stage == "group"``).
"""

from __future__ import annotations

import os
import unicodedata
from typing import Dict, List, Optional

from .data_io import HOST_TEAM_IDS, MatchRecord, Team


class FetchError(RuntimeError):
    pass


# football-data.org names that don't match data/teams.csv after accent folding.
# Keys are normalised with ``_norm`` (accent-folded, alphanumeric, lower-case).
_ALIASES = {
    "korearepublic": "KOR", "southkorea": "KOR",
    "czechrepublic": "CZE", "czechia": "CZE",
    "turkiye": "TUR", "turkey": "TUR",
    "cotedivoire": "CIV", "ivorycoast": "CIV",
    "caboverde": "CPV", "capeverde": "CPV", "capeverdeislands": "CPV",
    "bosniaherzegovina": "BIH",  # provider drops the "and" in "Bosnia and Herzegovina"
    "drcongo": "COD", "congodr": "COD",
    "democraticrepublicofcongo": "COD", "congodemocraticrepublic": "COD",
    "usa": "USA", "unitedstates": "USA", "unitedstatesofamerica": "USA",
    "iran": "IRN", "iriran": "IRN", "islamicrepublicofiran": "IRN",
    "curacao": "CUW",
}

# Provider stage labels -> the labels the rest of the pipeline uses. Group games
# MUST become "group" so the dashboard's group-standings filter counts them.
_STAGE = {
    "groupstage": "group", "group": "group",
    "last32": "round of 32", "roundof32": "round of 32",
    "last16": "round of 16", "roundof16": "round of 16",
    "quarterfinals": "quarter finals", "quarterfinal": "quarter finals",
    "semifinals": "semi finals", "semifinal": "semi finals",
    "thirdplace": "third place", "playofffor3rdplace": "third place",
    "final": "final",
}


def _require_requests():
    try:
        import requests  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise FetchError(
            "The 'api' extra is not installed. Run: pip install -e '.[api]'"
        ) from exc
    return requests


def _norm(name: str) -> str:
    """Accent-fold and reduce to lower-case alphanumerics ("Türkiye" -> turkiye)."""
    decomposed = unicodedata.normalize("NFKD", name or "")
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return "".join(ch for ch in stripped.lower() if ch.isalnum())


def _name_index(teams: List[Team]) -> Dict[str, str]:
    idx = {_norm(t.name): t.team_id for t in teams}
    idx.update(_ALIASES)  # explicit aliases win over / extend the canonical names
    return idx


def _norm_stage(stage: str) -> str:
    return _STAGE.get(_norm(stage), (stage or "").strip().lower() or "tournament")


def parse_matches(
    matches: List[dict], teams: List[Team], competition: str = "WC",
    unmapped: Optional[list] = None,
) -> List[MatchRecord]:
    """Map football-data.org match objects to :class:`MatchRecord`s.

    Only finished matches where *both* teams resolve to a known local id are
    returned. A host nation playing at home is treated as a non-neutral game, to
    match the schedule importer and the hand-entered results.

    If ``unmapped`` is given, the provider names of any *finished* match dropped
    because a team didn't resolve are appended to it, so the caller can surface a
    silent name-mapping gap instead of quietly losing the result.
    """
    idx = _name_index(teams)
    out: List[MatchRecord] = []
    for m in matches:
        home_name = m.get("homeTeam", {}).get("name", "")
        away_name = m.get("awayTeam", {}).get("name", "")
        home_id = idx.get(_norm(home_name))
        away_id = idx.get(_norm(away_name))
        score = m.get("score", {}).get("fullTime", {})
        if score.get("home") is None:
            continue  # not finished
        if home_id is None or away_id is None:
            if unmapped is not None:
                if home_id is None and home_name:
                    unmapped.append(home_name)
                if away_id is None and away_name:
                    unmapped.append(away_name)
            continue
        out.append(
            MatchRecord(
                date=(m.get("utcDate", "") or "")[:10],
                home_team_id=home_id,
                away_team_id=away_id,
                home_goals=int(score["home"]),
                away_goals=int(score["away"]),
                stage=_norm_stage(str(m.get("stage", ""))),
                competition=competition,
                # A host playing at home is not a neutral-venue game.
                neutral=home_id not in HOST_TEAM_IDS,
            )
        )
    return out


def fetch_results(
    teams: List[Team],
    source: str = "football-data",
    since: Optional[str] = None,
    competition: str = "WC",
) -> List[MatchRecord]:
    """Fetch finished matches from a provider and map them to ``MatchRecord``s.

    Only matches where *both* teams map to a known local ``team_id`` are returned.
    Raises :class:`FetchError` with an actionable message on any failure.
    """
    if source != "football-data":
        raise FetchError(f"Unknown source {source!r}. Supported: 'football-data'.")

    requests = _require_requests()
    api_key = os.environ.get("FOOTBALL_DATA_API_KEY")
    if not api_key:
        raise FetchError(
            "FOOTBALL_DATA_API_KEY is not set. Get a free key at "
            "https://www.football-data.org/ and export it before fetching."
        )

    params: Dict[str, str] = {"status": "FINISHED"}
    if since:
        params["dateFrom"] = since

    try:
        resp = requests.get(
            f"https://api.football-data.org/v4/competitions/{competition}/matches",
            headers={"X-Auth-Token": api_key},
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:  # pragma: no cover - network path
        raise FetchError(f"Request to football-data.org failed: {exc}") from exc

    unmapped: List[str] = []
    records = parse_matches(payload.get("matches", []), teams, competition, unmapped)
    if unmapped:
        import sys
        names = ", ".join(sorted(set(unmapped)))
        print(f"warning: {len(set(unmapped))} provider team name(s) did not map to a "
              f"local id; their finished matches were skipped: {names}. Add an alias "
              f"in wcpredictor/fetch.py (_ALIASES).", file=sys.stderr)
    return records
