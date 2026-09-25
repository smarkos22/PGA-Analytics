#!/usr/bin/env python3
"""
Shared tournament configuration module.

Provides load/save for tournament_config.json and auto-mode detection.
Used by: quick_datagolf_fetch.py, 1_scrape_vip365.py,
         2_compare_rankings.py, 3_select_bets.py
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent / "tournament_config.json"

DEFAULT_CONFIG = {
    "updated_at": None,
    "current_tournament": None,
    "grass_types": {
        "greens": None,
        "fairways": None,
        "approaches": None,
        "rough": None,
    },
    "vip365_checkboxes": ["Futures To win Outright", "Futures Top 5 Finish"],
    "market_types": ["win", "top_5"],
    "kalshi_event_ticker": None,
    "kalshi_event_name": None,
    "kalshi_match_score": None,
    "kalshi_field_overlap": None,
}


def load_config() -> dict:
    """Load tournament config from JSON file, or return defaults if missing."""
    if not CONFIG_PATH.exists():
        return dict(DEFAULT_CONFIG)
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        # Merge with defaults so new keys are always present
        merged = dict(DEFAULT_CONFIG)
        merged.update(data)
        return merged
    except (json.JSONDecodeError, OSError):
        return dict(DEFAULT_CONFIG)


def save_config(data: dict) -> None:
    """Write tournament config to JSON file with timestamp."""
    data["updated_at"] = datetime.now().isoformat(timespec="seconds")
    # Preserve defaults for any missing keys
    merged = dict(DEFAULT_CONFIG)
    merged.update(data)
    CONFIG_PATH.write_text(
        json.dumps(merged, indent=2, sort_keys=False),
        encoding="utf-8",
    )


def is_auto_mode() -> bool:
    """Return True when running non-interactively (--auto flag or piped stdin)."""
    return "--auto" in sys.argv or not sys.stdin.isatty()


# --- Tournament name matching utilities ---

# Minimum Jaccard similarity to consider a VIP365 market as matching the current tournament.
TOURNAMENT_MATCH_THRESHOLD = 0.4

_STOP_WORDS = frozenset(
    {"futures", "to", "win", "outright", "outrights", "tournament",
     "top", "finish", "the", "5", "five"}
)


def normalize_tournament_name(name: str) -> str:
    """Normalize a tournament name for comparison.

    Strips common non-distinctive words, lowercases, removes punctuation.
    """
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    tokens = [t for t in s.split() if t and t not in _STOP_WORDS]
    return " ".join(tokens)


def tournament_name_match_score(name1: str, name2: str) -> float:
    """Return a 0.0-1.0 Jaccard similarity score between two tournament names."""
    tokens1 = set(normalize_tournament_name(name1).split())
    tokens2 = set(normalize_tournament_name(name2).split())
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2
    return len(intersection) / len(union)
