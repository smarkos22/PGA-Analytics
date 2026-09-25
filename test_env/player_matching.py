"""
Canonical player name matching for bet slip creation.

Matches extracted player names (from screenshots, OCR, user input) against the
DataGolf field to ensure bet slips use canonical, trackable player names.

Uses the existing 3-tier matching system from 2_compare_rankings.py:
  Tier 1: Exact match after normalization (comma-flip + lowercase)
  Tier 2: Plausible match (same last name + first initial)

Returns matched names in "First Last" format for bet slip consistency.
"""

import json
import re
import sys
import unicodedata
from pathlib import Path

# Import matching functions from the existing pipeline
try:
    from compare_rankings_matching import (
        normalize_name,
        split_name_parts,
        is_plausible_name_match,
    )
except ImportError:
    # Fallback: define locally if import fails
    def strip_diacriticals(text: str) -> str:
        """Transliterate Unicode diacriticals to ASCII (Å→A, é→e, ñ→n, etc.)."""
        nfkd = unicodedata.normalize("NFKD", text)
        return "".join(c for c in nfkd if not unicodedata.combining(c))

    def normalize_name(value: str) -> str:
        raw = value.strip()
        raw = strip_diacriticals(raw)
        if "," in raw:
            parts = [part.strip() for part in raw.split(",", 1)]
            if len(parts) == 2:
                raw = f"{parts[1]} {parts[0]}"
        cleaned = re.sub(r"[^A-Za-z\s'-]+", "", raw).strip().lower()
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned

    def split_name_parts(name: str) -> tuple:
        norm = normalize_name(name)
        parts = norm.split()
        if not parts:
            return "", ""
        if len(parts) == 1:
            return parts[0], ""
        return parts[0], parts[-1]

    def is_plausible_name_match(dg_name: str, vip_name: str) -> bool:
        dg_first, dg_last = split_name_parts(dg_name)
        vip_first, vip_last = split_name_parts(vip_name)
        if not dg_last or not vip_last:
            return False
        if dg_last != vip_last:
            return False
        if not dg_first or not vip_first:
            return True
        return dg_first[0] == vip_first[0]


# Path to DataGolf field data
JSON_EXTRACT_PATH = Path(__file__).parent / "JSON_extract"

# Module-level cache
_field_lookup_cache = None
_field_lookup_path = None


def flip_name_to_first_last(dg_name: str) -> str:
    """Convert 'Last, First' to 'First Last'. Pass through if no comma."""
    if "," in dg_name:
        parts = [p.strip() for p in dg_name.split(",", 1)]
        if len(parts) == 2:
            return f"{parts[1]} {parts[0]}"
    return dg_name


def load_field_lookup(field_path: Path = None) -> dict:
    """Load DataGolf field and build normalized name lookup.

    Returns: {normalized_name: {
        "player_name": "Last, First",
        "dg_id": int,
        "first_last": "First Last"
    }}
    """
    global _field_lookup_cache, _field_lookup_path

    if field_path is None:
        field_path = JSON_EXTRACT_PATH / "datagolf_field.json"

    # Return cached if same path
    if _field_lookup_cache is not None and _field_lookup_path == str(field_path):
        return _field_lookup_cache

    lookup = {}
    if not field_path.exists():
        print(f"  [player_matching] Warning: field file not found: {field_path}", file=sys.stderr)
        return lookup

    try:
        with open(field_path, "r") as f:
            field_data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [player_matching] Warning: could not load field: {e}", file=sys.stderr)
        return lookup

    for player in field_data.get("field", []):
        pname = player.get("player_name", "")
        dg_id = player.get("dg_id")
        if not pname:
            continue
        norm = normalize_name(pname)
        lookup[norm] = {
            "player_name": pname,
            "dg_id": dg_id,
            "first_last": flip_name_to_first_last(pname),
        }

    _field_lookup_cache = lookup
    _field_lookup_path = str(field_path)
    return lookup


def invalidate_cache():
    """Clear the cached field lookup (e.g., after a data refresh)."""
    global _field_lookup_cache, _field_lookup_path
    _field_lookup_cache = None
    _field_lookup_path = None


def match_player_to_field(
    extracted_name: str,
    field_lookup: dict = None,
    field_path: Path = None,
) -> dict:
    """Match an extracted player name against the DataGolf field.

    Returns: {
        "matched": bool,
        "canonical_name": str,       # "First Last" format for bet slip storage
        "datagolf_name": str | None, # "Last, First" format
        "dg_id": int | None,
        "match_tier": int | None,    # 1=exact, 2=plausible, None=unmatched
        "original_name": str,
    }
    """
    if field_lookup is None:
        field_lookup = load_field_lookup(field_path)

    original = extracted_name.strip()
    result = {
        "matched": False,
        "canonical_name": original,
        "datagolf_name": None,
        "dg_id": None,
        "match_tier": None,
        "original_name": original,
    }

    if not original or not field_lookup:
        return result

    # --- Tier 1: Exact match after normalization ---
    norm = normalize_name(original)
    if norm in field_lookup:
        entry = field_lookup[norm]
        result["matched"] = True
        result["canonical_name"] = entry["first_last"]
        result["datagolf_name"] = entry["player_name"]
        result["dg_id"] = entry["dg_id"]
        result["match_tier"] = 1
        return result

    # --- Tier 2: Plausible match (same last name + first initial) ---
    candidates = []
    for norm_key, entry in field_lookup.items():
        if is_plausible_name_match(entry["player_name"], original):
            candidates.append(entry)

    if len(candidates) == 1:
        # Unambiguous plausible match
        entry = candidates[0]
        result["matched"] = True
        result["canonical_name"] = entry["first_last"]
        result["datagolf_name"] = entry["player_name"]
        result["dg_id"] = entry["dg_id"]
        result["match_tier"] = 2
        return result

    # Multiple plausible candidates = ambiguous, don't auto-match
    if len(candidates) > 1:
        print(
            f"  [player_matching] Ambiguous match for '{original}': "
            f"{[c['player_name'] for c in candidates]}",
            file=sys.stderr,
        )

    # --- No match ---
    return result


def match_players_batch(
    names: list,
    field_lookup: dict = None,
    field_path: Path = None,
) -> list:
    """Match multiple player names against the DataGolf field.

    Returns list of match result dicts (same schema as match_player_to_field).
    """
    if field_lookup is None:
        field_lookup = load_field_lookup(field_path)

    results = []
    for name in names:
        results.append(match_player_to_field(name, field_lookup=field_lookup))
    return results
