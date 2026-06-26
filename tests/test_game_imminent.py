"""Tests for the odds-snapshot timing gate (scripts/game_imminent.py).

scripts/ is not a package, so load the module by path (cf. test_fetch_odds.py).
"""

import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UTC = timezone.utc


def _load():
    spec = importlib.util.spec_from_file_location(
        "game_imminent", ROOT / "scripts" / "game_imminent.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_snapshots_when_kickoff_imminent():
    gi = _load()
    now = datetime(2026, 6, 26, 18, 40, tzinfo=UTC)
    kicks = [datetime(2026, 6, 26, 19, 0, tzinfo=UTC)]
    go, _ = gi.should_snapshot(now, kicks, None, within_min=35, min_gap_min=50)
    assert go


def test_skips_when_no_kickoff_soon():
    gi = _load()
    now = datetime(2026, 6, 26, 12, 0, tzinfo=UTC)
    kicks = [datetime(2026, 6, 26, 19, 0, tzinfo=UTC)]
    go, _ = gi.should_snapshot(now, kicks, None, within_min=35, min_gap_min=50)
    assert not go


def test_skips_when_recently_snapshotted():
    gi = _load()
    now = datetime(2026, 6, 26, 18, 50, tzinfo=UTC)
    kicks = [datetime(2026, 6, 26, 19, 0, tzinfo=UTC)]
    recent = now - timedelta(minutes=10)
    go, _ = gi.should_snapshot(now, kicks, recent, within_min=35, min_gap_min=50)
    assert not go


def test_resnapshots_once_min_gap_has_passed():
    gi = _load()
    now = datetime(2026, 6, 26, 18, 50, tzinfo=UTC)
    kicks = [datetime(2026, 6, 26, 19, 0, tzinfo=UTC)]
    old = now - timedelta(minutes=90)
    go, _ = gi.should_snapshot(now, kicks, old, within_min=35, min_gap_min=50)
    assert go


def test_already_kicked_off_is_not_imminent():
    gi = _load()
    now = datetime(2026, 6, 26, 19, 30, tzinfo=UTC)
    kicks = [datetime(2026, 6, 26, 19, 0, tzinfo=UTC)]
    go, _ = gi.should_snapshot(now, kicks, None, within_min=35, min_gap_min=50)
    assert not go


def test_parses_real_schedule_kickoffs():
    gi = _load()
    kicks = gi.parse_kickoffs(ROOT / "data" / "fifa_worldcup_2026_schedule.csv")
    assert len(kicks) > 50
    assert all(k.tzinfo is not None for k in kicks)
