#!/usr/bin/env python3
"""
Odds Snapshot Utility — Preserves historical odds data before overwrites.

Before any pipeline script overwrites a JSON file in JSON_extract/,
call snapshot_before_overwrite() to copy the current version to:
    JSON_extract/odds_snapshots/{tournament}/{YYYY-MM-DD}/{filename}_{HHMMSS}.json

Used by: quick_datagolf_fetch.py, 2_compare_rankings.py, 1_scrape_vip365.py
"""

import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from tournament_config import load_config

BASE_PATH = Path(__file__).resolve().parent
SNAPSHOTS_DIR = BASE_PATH / "JSON_extract" / "odds_snapshots"

# Max age for snapshot directories (cleanup older than this)
MAX_SNAPSHOT_AGE_DAYS = 14


def snapshot_before_overwrite(filepath, tournament_name=None):
    """Copy existing JSON file to odds_snapshots/ before it gets overwritten.

    Args:
        filepath: Path to the JSON file about to be overwritten.
        tournament_name: Optional tournament name for directory grouping.
                         If None, reads from tournament_config.json.
    """
    filepath = Path(filepath)

    # Skip if file doesn't exist or is empty
    if not filepath.exists() or filepath.stat().st_size == 0:
        return

    # Get tournament name for directory grouping
    if not tournament_name:
        try:
            config = load_config()
            tournament_name = config.get("current_tournament", "unknown")
        except Exception:
            tournament_name = "unknown"

    if not tournament_name:
        tournament_name = "unknown"

    # Sanitize tournament name for filesystem
    safe_name = tournament_name.replace(" ", "_").replace("/", "_").replace("\\", "_")

    # Build snapshot path: odds_snapshots/{tournament}/{date}/{filename}_{time}.json
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H%M%S")

    snap_dir = SNAPSHOTS_DIR / safe_name / date_str
    snap_dir.mkdir(parents=True, exist_ok=True)

    snap_filename = f"{filepath.stem}_{time_str}.json"
    snap_path = snap_dir / snap_filename

    # Copy the file
    try:
        shutil.copy2(filepath, snap_path)
    except (OSError, shutil.Error) as e:
        print(f"  [odds_snapshot] Warning: Could not snapshot {filepath.name}: {e}")


def cleanup_old_snapshots(days=MAX_SNAPSHOT_AGE_DAYS):
    """Remove snapshot directories older than `days` days."""
    if not SNAPSHOTS_DIR.exists():
        return

    cutoff = datetime.now() - timedelta(days=days)
    removed = 0

    for tournament_dir in SNAPSHOTS_DIR.iterdir():
        if not tournament_dir.is_dir():
            continue
        for date_dir in tournament_dir.iterdir():
            if not date_dir.is_dir():
                continue
            try:
                dir_date = datetime.strptime(date_dir.name, "%Y-%m-%d")
                if dir_date < cutoff:
                    shutil.rmtree(date_dir)
                    removed += 1
            except ValueError:
                continue  # Skip dirs that don't match date format

        # Remove empty tournament dirs
        if tournament_dir.exists() and not any(tournament_dir.iterdir()):
            tournament_dir.rmdir()

    if removed > 0:
        print(f"  [odds_snapshot] Cleaned up {removed} snapshot directories older than {days} days")


if __name__ == "__main__":
    # When run directly, just clean up old snapshots
    cleanup_old_snapshots()
    print(f"Snapshot directory: {SNAPSHOTS_DIR}")
    if SNAPSHOTS_DIR.exists():
        for t in sorted(SNAPSHOTS_DIR.iterdir()):
            if t.is_dir():
                count = sum(1 for _ in t.rglob("*.json"))
                print(f"  {t.name}: {count} snapshots")
