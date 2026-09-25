#!/usr/bin/env python3
"""Wave advantage calculator for tee-time weather edge.

Combines tournament weather forecast (AM vs PM wind) with player wave assignments
from DataGolf field data to compute per-player scoring adjustments.

Historical basis: +1.04 SG per round advantage for calmer wave across 24 qualifying
rounds (92% consistency). See 4_modeling/wind_analysis.py for derivation.

Used by: production_betting_model.py, frl_production_model.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_FORECAST_PATH = SCRIPT_DIR / "JSON_extract" / "tournament_forecast.json"
DEFAULT_FIELD_PATH = SCRIPT_DIR / "JSON_extract" / "datagolf_field.json"

# --- Constants (empirically derived) ---

WIND_DIFF_THRESHOLD_KMH = 5.0   # Minimum AM-PM diff to trigger adjustment
CONFIDENCE_DISCOUNT = 0.70       # Discount for forecast uncertainty
SG_PER_KMH = 0.104              # +1.04 SG per ~10 km/h wind differential


def _load_json(path: Path) -> Optional[dict]:
    """Load JSON file, return None if missing or invalid."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _get_player_wave_schedule(field_data: dict) -> Dict[str, list]:
    """Extract per-player wave assignments for all rounds.

    Returns: {player_name: [("early"|"late", round_num), ...]}
    """
    players = {}
    for player in field_data.get("field", []):
        name = player.get("player_name", "")
        if not name:
            continue
        waves = []
        for tt in player.get("teetimes", []):
            wave = tt.get("wave")
            rnd = tt.get("round_num")
            if wave and rnd:
                waves.append((wave, rnd))
        if waves:
            players[name] = waves
    return players


def _infer_full_schedule(known_waves: list) -> Dict[int, str]:
    """Infer wave assignments for all 4 rounds from known R1-R2 data.

    PGA standard: R1 wave flips for R2. R3 = R1. R4 = R2.
    """
    schedule = {}
    for wave, rnd in known_waves:
        schedule[rnd] = wave

    # If we have R1, infer the rest
    if 1 in schedule:
        r1_wave = schedule[1]
        opposite = "late" if r1_wave == "early" else "early"
        schedule.setdefault(2, opposite)
        schedule.setdefault(3, r1_wave)
        schedule.setdefault(4, opposite)
    elif 2 in schedule:
        r2_wave = schedule[2]
        opposite = "late" if r2_wave == "early" else "early"
        schedule.setdefault(1, opposite)
        schedule.setdefault(3, opposite)
        schedule.setdefault(4, r2_wave)

    return schedule


def compute_wave_adjustments(
    forecast_path: Optional[str] = None,
    field_path: Optional[str] = None,
    frl_mode: bool = False,
) -> Dict[str, float]:
    """Compute per-player wave advantage adjustments.

    Args:
        forecast_path: Path to tournament_forecast.json (default: auto-detect)
        field_path: Path to datagolf_field.json (default: auto-detect)
        frl_mode: If True, only use R1 adjustment (for FRL model)

    Returns:
        dict: {player_name: wave_adj} for all players in field.
              Returns all zeros if data unavailable (graceful degradation).
    """
    fp = Path(forecast_path) if forecast_path else DEFAULT_FORECAST_PATH
    flp = Path(field_path) if field_path else DEFAULT_FIELD_PATH

    forecast = _load_json(fp)
    field_data = _load_json(flp)

    # Graceful degradation — return empty adjustments
    if not forecast or not field_data:
        players = field_data.get("field", []) if field_data else []
        return {p.get("player_name", ""): 0.0 for p in players if p.get("player_name")}

    if forecast.get("error") or not forecast.get("rounds"):
        return {p.get("player_name", ""): 0.0
                for p in field_data.get("field", []) if p.get("player_name")}

    # Get per-round wave advantage from forecast
    round_advantages = {}  # {round_num: {"advantage": "early"|"late"|"none", "diff": float}}
    for round_key, round_info in forecast.get("rounds", {}).items():
        try:
            rnd_num = int(round_key.split("_")[1])
        except (IndexError, ValueError):
            continue
        diff = round_info.get("wind_diff_kmh", 0.0)
        adv = round_info.get("advantage", "none")
        round_advantages[rnd_num] = {"advantage": adv, "diff": diff}

    # Get player wave assignments
    player_waves = _get_player_wave_schedule(field_data)

    # Compute adjustments
    adjustments = {}
    for name, known_waves in player_waves.items():
        schedule = _infer_full_schedule(known_waves)
        total_adj = 0.0

        rounds_to_use = [1] if frl_mode else [1, 2, 3, 4]

        for rnd in rounds_to_use:
            if rnd not in schedule or rnd not in round_advantages:
                continue

            player_wave = schedule[rnd]
            ra = round_advantages[rnd]
            diff = ra["diff"]
            advantage_wave = ra["advantage"]

            if diff < WIND_DIFF_THRESHOLD_KMH or advantage_wave == "none":
                continue

            # Player gets positive adjustment if in the advantaged wave
            sg_magnitude = SG_PER_KMH * diff * CONFIDENCE_DISCOUNT
            if player_wave == advantage_wave:
                total_adj += sg_magnitude
            else:
                total_adj -= sg_magnitude

        adjustments[name] = round(total_adj, 3)

    # Include any players without tee time data (zero adjustment)
    for player in field_data.get("field", []):
        name = player.get("player_name", "")
        if name and name not in adjustments:
            adjustments[name] = 0.0

    return adjustments


def get_forecast_summary(forecast_path: Optional[str] = None) -> dict:
    """Get a summary of the current forecast for UI display.

    Returns dict with:
        - has_data: bool
        - active_rounds: list of round numbers with wave advantage
        - total_rounds: int
        - max_diff: float (largest AM-PM wind diff)
        - advantage_wave: str ("early"|"late"|"mixed"|"none")
    """
    fp = Path(forecast_path) if forecast_path else DEFAULT_FORECAST_PATH
    forecast = _load_json(fp)

    if not forecast or forecast.get("error") or not forecast.get("rounds"):
        return {"has_data": False, "active_rounds": [], "total_rounds": 0,
                "max_diff": 0.0, "advantage_wave": "none"}

    active_rounds = []
    max_diff = 0.0
    advantages = set()

    for round_key, round_info in forecast.get("rounds", {}).items():
        try:
            rnd_num = int(round_key.split("_")[1])
        except (IndexError, ValueError):
            continue
        diff = round_info.get("wind_diff_kmh", 0.0)
        adv = round_info.get("advantage", "none")
        if diff >= WIND_DIFF_THRESHOLD_KMH and adv != "none":
            active_rounds.append(rnd_num)
            advantages.add(adv)
            max_diff = max(max_diff, diff)

    if not advantages:
        wave = "none"
    elif len(advantages) == 1:
        wave = advantages.pop()
    else:
        wave = "mixed"

    return {
        "has_data": True,
        "active_rounds": sorted(active_rounds),
        "total_rounds": len(forecast.get("rounds", {})),
        "max_diff": round(max_diff, 1),
        "advantage_wave": wave,
        "fetched_at": forecast.get("fetched_at", ""),
    }
