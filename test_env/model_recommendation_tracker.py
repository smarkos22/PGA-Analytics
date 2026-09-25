#!/usr/bin/env python3
"""
Model Recommendation Tracker — Tracks how production model picks evolve Mon→Thu.

Saves timestamped snapshots of model recommendations with content-hash
deduplication. Provides utilities to compare snapshots and trace per-player
edge/rank/stake changes across the week.

Storage: JSON_extract/model_snapshots/{tournament}/{YYYY-MM-DD}/
           {model_type}_{HHMMSS}.json
           .last_hash_{model_type}.txt   (dedup marker)

Used by: production_betting_model.py, frl_production_model.py (after generating recs)
"""

import hashlib
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from tournament_config import load_config

BASE_PATH = Path(__file__).resolve().parent
MODEL_SNAPSHOTS_DIR = BASE_PATH / "JSON_extract" / "model_snapshots"

MAX_SNAPSHOT_AGE_DAYS = 60  # Longer retention than odds (14d) for calibration analysis


def _hash_recommendations(recs_list: list) -> str:
    """Content hash of recommendations array for deduplication."""
    # Sort by player+market for deterministic ordering
    stable = sorted(recs_list, key=lambda r: (r.get("player_name", ""), r.get("market", "")))
    return hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()[:16]


def _get_tournament_name() -> str:
    """Get current tournament name from config."""
    try:
        config = load_config()
        return config.get("current_tournament", "unknown")
    except Exception:
        return "unknown"


def _safe_dir_name(name: str) -> str:
    """Sanitize tournament name for filesystem."""
    return name.replace(" ", "_").replace("/", "_").replace("'", "")


def save_model_snapshot(model_output: dict, model_type: str = "production",
                        tournament_name: str = None) -> "Path | None":
    """Save a model recommendation snapshot if content has changed.

    Args:
        model_output: Full model output dict (with recommendations, model_config, etc.)
        model_type: "production" or "frl"
        tournament_name: Tournament name (auto-detected from config if None)

    Returns:
        Path to saved snapshot, or None if skipped (duplicate content).
    """
    if not model_output or not isinstance(model_output, dict):
        return None

    recs = model_output.get("recommendations", [])
    if not recs:
        return None

    if not tournament_name:
        tournament_name = _get_tournament_name()

    # Build snapshot directory
    safe_name = _safe_dir_name(tournament_name)
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    snap_dir = MODEL_SNAPSHOTS_DIR / safe_name / date_str
    snap_dir.mkdir(parents=True, exist_ok=True)

    # Content-hash deduplication
    rec_hash = _hash_recommendations(recs)
    hash_file = snap_dir / f".last_hash_{model_type}.txt"

    if hash_file.exists():
        try:
            prev_hash = hash_file.read_text(encoding="utf-8").strip()
            if prev_hash == rec_hash:
                return None  # Identical recommendations, skip
        except OSError:
            pass

    # Save snapshot
    time_str = now.strftime("%H%M%S")
    snap_path = snap_dir / f"{model_type}_{time_str}.json"

    snapshot = {
        "snapshot_ts": now.isoformat(),
        "tournament": tournament_name,
        "model_type": model_type,
        "model": model_output.get("model", "unknown"),
        "model_config": model_output.get("model_config", {}),
        "wave_forecast": model_output.get("wave_forecast", {}),
        "rec_count": len(recs),
        "recommendations": recs,
    }

    # Include exclusions if present (production model tracks why players were excluded)
    if "exclusions" in model_output:
        snapshot["exclusions"] = model_output["exclusions"]

    try:
        snap_path.write_text(json.dumps(snapshot, indent=1), encoding="utf-8")
        hash_file.write_text(rec_hash, encoding="utf-8")
        print(f"  [model_tracker] Saved {model_type} snapshot: {snap_path.name} ({len(recs)} recs)")
        return snap_path
    except OSError as e:
        print(f"  [model_tracker] Warning: could not save snapshot: {e}")
        return None


def get_snapshot_history(tournament_name: str, model_type: str = "production",
                         days_back: int = 30) -> list:
    """Load all model snapshots for a tournament, sorted chronologically.

    Returns:
        List of snapshot dicts sorted by snapshot_ts.
    """
    safe_name = _safe_dir_name(tournament_name)
    tournament_dir = MODEL_SNAPSHOTS_DIR / safe_name

    if not tournament_dir.exists():
        return []

    cutoff = datetime.now() - timedelta(days=days_back)
    snapshots = []

    for date_dir in sorted(tournament_dir.iterdir()):
        if not date_dir.is_dir():
            continue
        try:
            dir_date = datetime.strptime(date_dir.name, "%Y-%m-%d")
            if dir_date < cutoff:
                continue
        except ValueError:
            continue

        for snap_file in sorted(date_dir.glob(f"{model_type}_*.json")):
            try:
                data = json.loads(snap_file.read_text(encoding="utf-8"))
                snapshots.append(data)
            except (json.JSONDecodeError, OSError):
                continue

    return snapshots


def get_player_timeline(tournament_name: str, player_name: str,
                        model_type: str = "production", days_back: int = 30) -> list:
    """Get a player's recommendation history across all snapshots.

    Args:
        tournament_name: Tournament name.
        player_name: Player name (DataGolf format "Last, First").
        model_type: "production" or "frl".
        days_back: How many days back to look.

    Returns:
        List of dicts per snapshot where this player appeared:
        [{"ts", "market", "model_rank", "edge", "odds", "book",
          "model_prob", "stake", "sg_avg", "grass_adj", "wave_adj", "score"}, ...]
        If player was excluded in a snapshot, returns {"ts", "status": "excluded", "reason"}.
    """
    snapshots = get_snapshot_history(tournament_name, model_type, days_back)
    normalized = player_name.strip().lower()
    timeline = []

    for snap in snapshots:
        ts = snap.get("snapshot_ts")
        found = False

        for rec in snap.get("recommendations", []):
            if rec.get("player_name", "").strip().lower() == normalized:
                entry = {
                    "ts": ts,
                    "status": "recommended",
                    "market": rec.get("market"),
                    "model_rank": rec.get("model_rank"),
                    "edge": rec.get("edge"),
                    "odds": rec.get("odds_american"),
                    "book": rec.get("book"),
                    "model_prob": rec.get("model_prob"),
                    "implied_prob": rec.get("implied_prob"),
                    "stake": rec.get("recommended_stake"),
                    "sg_avg": rec.get("sg_avg"),
                    "grass_adj": rec.get("grass_adj"),
                    "wave_adj": rec.get("wave_adj"),
                    "score": rec.get("score"),
                }
                timeline.append(entry)
                found = True
                # Don't break — player may appear in multiple markets

        # Check exclusions if not found in recommendations
        if not found:
            for excl in snap.get("exclusions", []):
                if excl.get("player_name", "").strip().lower() == normalized:
                    timeline.append({
                        "ts": ts,
                        "status": "excluded",
                        "reason": excl.get("reason", "unknown"),
                        "model_rank": excl.get("model_rank"),
                        "score": excl.get("score"),
                    })
                    found = True
                    break

    return timeline


def compare_snapshots(snap_old: dict, snap_new: dict) -> dict:
    """Compare two snapshots and identify changes.

    Args:
        snap_old: Earlier snapshot dict.
        snap_new: Later snapshot dict.

    Returns:
        {
            "ts_old": str, "ts_new": str,
            "new_picks": [{player, market, edge, odds, reason_hint}],
            "removed_picks": [{player, market, edge, odds}],
            "edge_changes": [{player, market, old_edge, new_edge, delta,
                              old_odds, new_odds, reasons: [str]}],
            "summary": {"added": int, "removed": int, "changed": int,
                        "avg_edge_delta": float}
        }
    """
    def _build_lookup(recs):
        """Build {(player, market): rec} lookup."""
        lookup = {}
        for r in recs:
            key = (r.get("player_name", ""), r.get("market", ""))
            lookup[key] = r
        return lookup

    old_recs = _build_lookup(snap_old.get("recommendations", []))
    new_recs = _build_lookup(snap_new.get("recommendations", []))

    old_keys = set(old_recs.keys())
    new_keys = set(new_recs.keys())

    # New picks
    new_picks = []
    for key in sorted(new_keys - old_keys):
        rec = new_recs[key]
        # Check if player was excluded before
        reason = "new pick"
        for excl in snap_old.get("exclusions", []):
            if excl.get("player_name", "").strip().lower() == key[0].strip().lower():
                reason = f"was excluded: {excl.get('reason', 'unknown')}"
                break
        new_picks.append({
            "player": key[0], "market": key[1],
            "edge": rec.get("edge"), "odds": rec.get("odds_american"),
            "reason_hint": reason,
        })

    # Removed picks
    removed_picks = []
    for key in sorted(old_keys - new_keys):
        rec = old_recs[key]
        removed_picks.append({
            "player": key[0], "market": key[1],
            "edge": rec.get("edge"), "odds": rec.get("odds_american"),
        })

    # Edge changes for picks that exist in both
    edge_changes = []
    for key in sorted(old_keys & new_keys):
        old_r = old_recs[key]
        new_r = new_recs[key]
        old_edge = old_r.get("edge", 0)
        new_edge = new_r.get("edge", 0)
        delta = new_edge - old_edge

        if abs(delta) < 0.5:  # Less than 0.5% change, skip
            continue

        # Determine reasons for the change
        reasons = []
        if old_r.get("odds_american") != new_r.get("odds_american"):
            reasons.append(f"odds: {old_r.get('odds_american')} → {new_r.get('odds_american')}")
        if old_r.get("book") != new_r.get("book"):
            reasons.append(f"book: {old_r.get('book')} → {new_r.get('book')}")
        if abs((old_r.get("sg_avg", 0) or 0) - (new_r.get("sg_avg", 0) or 0)) > 0.01:
            reasons.append(f"sg_avg: {old_r.get('sg_avg', 0):.3f} → {new_r.get('sg_avg', 0):.3f}")
        if abs((old_r.get("grass_adj", 0) or 0) - (new_r.get("grass_adj", 0) or 0)) > 0.01:
            reasons.append(f"grass_adj: {old_r.get('grass_adj', 0):.3f} → {new_r.get('grass_adj', 0):.3f}")
        old_wave = old_r.get("wave_adj") or 0
        new_wave = new_r.get("wave_adj") or 0
        if abs(old_wave - new_wave) > 0.01:
            reasons.append(f"wave_adj: {old_wave:.3f} → {new_wave:.3f}")

        if not reasons:
            reasons.append("model recalibration")

        edge_changes.append({
            "player": key[0], "market": key[1],
            "old_edge": round(old_edge, 1), "new_edge": round(new_edge, 1),
            "delta": round(delta, 1),
            "old_odds": old_r.get("odds_american"),
            "new_odds": new_r.get("odds_american"),
            "reasons": reasons,
        })

    # Sort edge changes by absolute delta descending
    edge_changes.sort(key=lambda x: abs(x["delta"]), reverse=True)

    avg_delta = sum(c["delta"] for c in edge_changes) / len(edge_changes) if edge_changes else 0

    return {
        "ts_old": snap_old.get("snapshot_ts"),
        "ts_new": snap_new.get("snapshot_ts"),
        "new_picks": new_picks,
        "removed_picks": removed_picks,
        "edge_changes": edge_changes,
        "summary": {
            "added": len(new_picks),
            "removed": len(removed_picks),
            "changed": len(edge_changes),
            "avg_edge_delta": round(avg_delta, 1),
        },
    }


def cleanup_old_snapshots(days: int = MAX_SNAPSHOT_AGE_DAYS):
    """Remove model snapshot directories older than `days` days."""
    if not MODEL_SNAPSHOTS_DIR.exists():
        return

    cutoff = datetime.now() - timedelta(days=days)
    removed = 0

    for tournament_dir in MODEL_SNAPSHOTS_DIR.iterdir():
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
                continue

        # Remove empty tournament dirs
        if tournament_dir.exists() and not any(tournament_dir.iterdir()):
            tournament_dir.rmdir()

    if removed > 0:
        print(f"  [model_tracker] Cleaned up {removed} snapshot directories older than {days} days")


if __name__ == "__main__":
    # When run directly, show summary + compare latest two snapshots if available
    cleanup_old_snapshots()
    print(f"Model snapshots directory: {MODEL_SNAPSHOTS_DIR}")

    if MODEL_SNAPSHOTS_DIR.exists():
        for t in sorted(MODEL_SNAPSHOTS_DIR.iterdir()):
            if t.is_dir():
                count = sum(1 for _ in t.rglob("*.json"))
                print(f"\n  {t.name}: {count} snapshots")
                for d in sorted(t.iterdir()):
                    if d.is_dir() and not d.name.startswith("."):
                        day_count = sum(1 for _ in d.glob("*.json"))
                        print(f"    {d.name}: {day_count} files")

        # Auto-compare latest two production snapshots for current tournament
        tournament = _get_tournament_name()
        history = get_snapshot_history(tournament, "production", days_back=7)
        if len(history) >= 2:
            diff = compare_snapshots(history[-2], history[-1])
            print(f"\n{'=' * 70}")
            print(f"LATEST CHANGE: {diff['ts_old'][:16]} → {diff['ts_new'][:16]}")
            print(f"{'=' * 70}")
            s = diff["summary"]
            print(f"  +{s['added']} added | -{s['removed']} removed | ~{s['changed']} changed | avg edge Δ: {s['avg_edge_delta']:+.1f}%")

            if diff["new_picks"]:
                print(f"\n  NEW PICKS:")
                for p in diff["new_picks"]:
                    print(f"    + {p['player']:<25s} {p['market']:<8s} {p['odds']:>8s} edge: {p['edge']:+.1f}%  ({p['reason_hint']})")

            if diff["removed_picks"]:
                print(f"\n  REMOVED PICKS:")
                for p in diff["removed_picks"]:
                    print(f"    - {p['player']:<25s} {p['market']:<8s} {p['odds']:>8s} edge: {p['edge']:+.1f}%")

            if diff["edge_changes"][:10]:
                print(f"\n  BIGGEST EDGE CHANGES:")
                for c in diff["edge_changes"][:10]:
                    print(f"    ~ {c['player']:<25s} {c['market']:<8s} {c['old_edge']:+.1f}% → {c['new_edge']:+.1f}% (Δ{c['delta']:+.1f}%)")
                    for r in c["reasons"]:
                        print(f"        → {r}")
        elif len(history) == 1:
            print(f"\n  Only 1 snapshot for {tournament} — need at least 2 to compare.")
        else:
            print(f"\n  No snapshots yet for {tournament}.")
    else:
        print("  No model snapshots captured yet.")
