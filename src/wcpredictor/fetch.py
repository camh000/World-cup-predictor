"""Optional live-results fetcher.

This module is entirely optional. The engine works fully offline; this just
lets you auto-populate ``data/results.csv`` from a public football API. It
imports ``requests`` lazily so the core package has zero network dependencies,
and it fails with a clear message when the extra or API key is missing.

Currently supports football-data.org (set FOOTBALL_DATA_API_KEY). Map the
provider's team names to your local team ids via ``data/teams.csv`` names.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from .data_io import MatchRecord, Team


class FetchError(RuntimeError):
    pass


def _require_requests():
    try:
        import requests  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise FetchError(
            "The 'api' extra is not installed. Run: pip install -e '.[api]'"
        ) from exc
    return requests


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

    name_to_id = _name_index(teams)
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

    out: List[MatchRecord] = []
    for m in payload.get("matches", []):
        home_id = name_to_id.get(_norm(m.get("homeTeam", {}).get("name", "")))
        away_id = name_to_id.get(_norm(m.get("awayTeam", {}).get("name", "")))
        score = m.get("score", {}).get("fullTime", {})
        if home_id is None or away_id is None or score.get("home") is None:
            continue
        out.append(
            MatchRecord(
                date=(m.get("utcDate", "") or "")[:10],
                home_team_id=home_id,
                away_team_id=away_id,
                home_goals=int(score["home"]),
                away_goals=int(score["away"]),
                stage=str(m.get("stage", "")).lower() or "tournament",
                competition=competition,
                neutral=True,
            )
        )
    return out


def _norm(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _name_index(teams: List[Team]) -> Dict[str, str]:
    return {_norm(t.name): t.team_id for t in teams}
