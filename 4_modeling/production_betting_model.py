#!/usr/bin/env python3
"""
Golf recommendation model used by the weekly research application.

Combines trailing strokes gained, tee-to-green performance, confidence-scaled
course/grass adjustments, and volatility. Win-market probabilities blend model
and market estimates; top-N markets can blend DataGolf pre-tournament estimates.
Recommendations apply edge thresholds and fractional-Kelly sizing constraints.

Parameter selection can bias backtest estimates. No performance result is
claimed by this public source snapshot. Evaluate on
held-out events with explicit pricing and execution assumptions before drawing
performance conclusions.
"""

import json
import math
import statistics
import csv
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "test_env"))
try:
    from wave_advantage import compute_wave_adjustments, get_forecast_summary
except ImportError:
    compute_wave_adjustments = None
    get_forecast_summary = None

CANONICAL_DIR = REPO_ROOT / "3_canonical_data_aggregation" / "canonical_data"
# Auto-detect latest canonical data file
CANONICAL_FILES = sorted(CANONICAL_DIR.glob("canonical_historical_data_*.json"))
CANONICAL_FILE = CANONICAL_FILES[-1].name if CANONICAL_FILES else "canonical_historical_data.json"
GRASS_FILE = REPO_ROOT / "4_modeling" / "grass_type_analysis_output" / "grass_type_all_data.csv"
JSON_EXTRACT_DIR = REPO_ROOT / "test_env" / "JSON_extract"

# Research parameters inherited from the application. These are assumptions,
# not a claim of optimality or independently verified performance.
LOOKBACK = 10
MIN_HIST = 5
MAX_PLAYERS = 15
GRASS_WEIGHT = 0.40
GRASS_CONFIDENCE_EVENTS = 20
VOLATILITY_WEIGHT = -0.10
SG_T2G_WEIGHT = 0.25
WAVE_ENABLED = True
MIN_EDGE = 5.0
WIN_BLEND_ALPHA = 0.50
DG_PT_BLEND = 0.50
DG_PT_LIVE_FILE = REPO_ROOT / "test_env" / "JSON_extract" / "datagolf_pretournament.json"
KELLY_FRACTIONS = {"win": 0.333, "top_5": 0.333, "top_10": 0.125, "top_20": 0.0}
MAX_STAKE_PCT = 0.05
MIN_STAKE = 5.0
DEFAULT_BANKROLL = 1000.0  # Fictional example budget for the public snapshot
MAX_TOURNAMENT_PCT = 0.10

# Market-specific comparison files and softmax temperatures
MARKET_FILES = {
    "win": "golf_rank_comparison_outright_winner.json",
    "top_5": "golf_rank_comparison_top_5_finish.json",
    "top_10": "golf_rank_comparison_top_10_finish.json",
    "top_20": "golf_rank_comparison_top_20_finish.json",
}

def softmax_probabilities(scores, temperature=1.0):
    """Convert raw model scores to probabilities via softmax with temperature scaling.

    Subtracts the maximum score for numerical stability. Temperature is supplied
    by the caller; softmax alone does not establish probability calibration.
    """
    if not scores:
        return []
    max_score = max(scores)
    exp_scores = [math.exp((s - max_score) / temperature) for s in scores]
    total = sum(exp_scores)
    return [e / total for e in exp_scores]


def calculate_kelly_stake(model_prob, decimal_odds, kelly_fraction, bankroll,
                          max_pct=MAX_STAKE_PCT, min_stake=MIN_STAKE):
    """Calculate fractional Kelly criterion stake.

    Returns (kelly_pct, dollar_stake) tuple.
    """
    b = decimal_odds - 1  # net odds per $1 wagered
    if b <= 0:
        return 0.0, 0.0
    q = 1 - model_prob
    kelly_raw = (model_prob * b - q) / b
    if kelly_raw <= 0:
        return 0.0, 0.0
    fraction = min(kelly_raw * kelly_fraction, max_pct)
    stake = round(fraction * bankroll, 2)
    if stake < min_stake:
        return 0.0, 0.0
    return round(fraction, 6), stake


def parse_american_odds(odds_str):
    if not odds_str:
        return None
    try:
        odds = int(odds_str.replace("+", ""))
        return (odds / 100) + 1 if odds > 0 else (100 / abs(odds)) + 1
    except:
        return None


def load_grass_data():
    """Load grass significance data."""
    grass_data = {}
    if not GRASS_FILE.exists():
        return grass_data

    with open(GRASS_FILE, newline="") as f:
        for row in csv.DictReader(f):
            player = row.get("player_name", "").strip().lower()
            surface = row.get("surface", "").strip().lower()
            grass = row.get("grass_type", "").strip().lower()
            mean_diff = row.get("mean_diff")
            n_on = row.get("n_on_type")

            if player and surface and grass and mean_diff:
                try:
                    grass_data.setdefault(player, {}).setdefault(surface, {})[grass] = {
                        "mean_diff": float(mean_diff),
                        "n_on": int(n_on) if n_on else 0,
                    }
                except:
                    pass
    return grass_data


def load_canonical_data():
    """Load canonical data and build player history."""
    with open(CANONICAL_DIR / CANONICAL_FILE) as f:
        canonical = json.load(f)

    events = canonical.get("events_by_id", {})

    # Get event grass types
    event_grass = {}
    for eid, ev in events.items():
        ag = ev.get("aggriculture", {})
        courses = ag.get("courses", [])
        if courses:
            grass_map = {}
            for surface in ["greens", "fairways", "approaches", "rough"]:
                types = []
                for course in courses:
                    tg = course.get("turfgrass", {}).get(surface, {})
                    if tg and tg.get("type"):
                        types.extend(tg["type"])
                if types:
                    grass_map[surface] = "/".join(sorted(set(t.lower() for t in types)))
            event_grass[eid] = grass_map

    # Build player history (SG only - pre-tournament available!)
    player_history = defaultdict(list)

    for eid, ev in events.items():
        scores = ev.get("scores", [])
        event_date = ev.get("event_completed", "")
        if not event_date:
            continue

        for score in scores:
            dg_id = score.get("dg_id")
            if not dg_id:
                continue

            # SG total and tee-to-green are available pre-tournament
            sg_total = 0
            sg_t2g = 0
            for r in range(1, 5):
                rd = score.get(f"round_{r}", {})
                sg_total += rd.get("sg_total", 0) or 0
                sg_t2g += rd.get("sg_t2g", 0) or 0

            fin_text = score.get("fin_text", "")
            finish = 999
            if fin_text:
                s = str(fin_text).strip().upper()
                if s not in ("WD", "DQ", "CUT", "MDF"):
                    try:
                        finish = int(s.replace("T", ""))
                    except:
                        pass

            if sg_total:
                player_history[dg_id].append({
                    "date": event_date,
                    "sg": sg_total,
                    "sg_t2g": sg_t2g,
                    "finish": finish,
                })

    for dg_id in player_history:
        player_history[dg_id].sort(key=lambda x: x["date"])

    return events, player_history, event_grass


def load_pretournament_live():
    """Load DataGolf /preds/pre-tournament cached response (current week).

    Returns dict: {str(dg_id): {win, top_5, top_10, top_20}} with 0..1 probs.
    Empty dict if file missing, stale, or for a different tournament than the
    comparison files — caller is responsible for checking.
    """
    if not DG_PT_LIVE_FILE.exists():
        return {}
    try:
        data = json.loads(DG_PT_LIVE_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    out = {}
    for entry in data.get("baseline", []):
        dg_id = entry.get("dg_id")
        if dg_id is None:
            continue
        out[str(dg_id)] = {
            "win": entry.get("win"),
            "top_5": entry.get("top_5"),
            "top_10": entry.get("top_10"),
            "top_20": entry.get("top_20"),
            "event_name": data.get("event_name"),
        }
    return out


class ProductionBettingModel:
    def __init__(self):
        print("Loading production model...")
        self.grass_data = load_grass_data()
        self.events, self.player_history, self.event_grass = load_canonical_data()
        self._last_exclusions = []  # Populated by generate_recommendations()
        self.dg_pt_live = load_pretournament_live() if DG_PT_BLEND > 0 else {}
        print(f"  Loaded SG history for {len(self.player_history)} players")
        print(f"  Grass data for {len(self.grass_data)} players")
        if DG_PT_BLEND > 0:
            print(f"  DG pre-tournament blend: {DG_PT_BLEND} ({len(self.dg_pt_live)} players loaded)")

    def calculate_score(self, dg_id, event_date, event_grass_types, player_name):
        """Calculate model score for a player.

        Score = blended_sg_avg + grass_adj + volatility_penalty
        where blended_sg_avg = sg_total * (1 - SG_T2G_WEIGHT) + sg_t2g * SG_T2G_WEIGHT
        """
        history = self.player_history.get(dg_id, [])
        prior = [h for h in history if h["date"] < event_date]

        if len(prior) < MIN_HIST:
            return None, 0, 0

        recent = prior[-LOOKBACK:]
        sg_vals = [h["sg"] for h in recent]
        sg_avg = statistics.mean(sg_vals)

        # SG tee-to-green emphasis: blend overall SG with t2g-only SG
        if SG_T2G_WEIGHT > 0:
            t2g_vals = [h.get("sg_t2g", 0) for h in recent]
            t2g_avg = statistics.mean(t2g_vals)
            sg_avg = sg_avg * (1 - SG_T2G_WEIGHT) + t2g_avg * SG_T2G_WEIGHT

        # Grass adjustment
        grass_adj = 0
        if player_name and event_grass_types:
            name_key = player_name.strip().lower()
            player_grass = self.grass_data.get(name_key, {})

            if player_grass and event_grass_types:
                for surface, grass_type in event_grass_types.items():
                    if surface in player_grass and grass_type in player_grass[surface]:
                        n = player_grass[surface][grass_type].get("n_on", 5)
                        conf = min(1.0, n / GRASS_CONFIDENCE_EVENTS)
                        grass_adj += player_grass[surface][grass_type].get("mean_diff", 0) * GRASS_WEIGHT * conf

        # Volatility penalty: penalize inconsistent players
        vol_adj = 0
        if VOLATILITY_WEIGHT != 0 and len(sg_vals) >= 3:
            vol_adj = statistics.stdev(sg_vals) * VOLATILITY_WEIGHT

        total_score = sg_avg + grass_adj + vol_adj

        return total_score, sg_avg, grass_adj

    def generate_recommendations(self, market_data, event_id, event_date=None):
        """Generate betting recommendations across all markets.

        Args:
            market_data: dict of {market_key: [entries]} from load_market_data()
            event_id: canonical event ID (or "current_event")
            event_date: date string for history cutoff
        """
        if event_date is None:
            event_date = datetime.now().strftime("%Y-%m-%d")

        event_grass_types = self.event_grass.get(event_id, {})

        # Step 1: Score all players using outright winner field as master list
        master_field = market_data.get("win", [])
        if not master_field:
            for entries in market_data.values():
                if entries:
                    master_field = entries
                    break

        scored_players = {}  # dg_id -> {score, sg_avg, grass_adj, wave_adj, player_name}
        for entry in master_field:
            dg_id = entry.get("dg_id")
            if not dg_id:
                continue
            player_name = entry.get("datagolf_name") or entry.get("player_name", "")
            score, sg_avg, grass_adj = self.calculate_score(
                dg_id, event_date, event_grass_types, player_name
            )
            if score is not None:
                scored_players[dg_id] = {
                    "score": score,
                    "sg_avg": sg_avg,
                    "grass_adj": grass_adj,
                    "wave_adj": 0.0,
                    "player_name": player_name,
                }

        if not scored_players:
            return []

        # Track Gate 1: field players who couldn't be scored (insufficient history)
        self._last_exclusions = []
        for entry in master_field:
            dg_id = entry.get("dg_id")
            if not dg_id or dg_id in scored_players:
                continue
            player_name = entry.get("datagolf_name") or entry.get("player_name", "")
            history = self.player_history.get(dg_id, [])
            n_prior = len([h for h in history if h["date"] < event_date])
            self._last_exclusions.append({
                "player_name": player_name,
                "dg_id": dg_id,
                "dg_rank": entry.get("datagolf_rank") or entry.get("dg_rank"),
                "reason": "insufficient_history",
                "detail": f"{n_prior} of {MIN_HIST} required completed events with SG data",
                "model_rank": None,
                "sg_avg": None,
            })

        # Apply wave advantage adjustments from weather forecast
        if WAVE_ENABLED and compute_wave_adjustments is not None:
            wave_adjs = compute_wave_adjustments()
            wave_applied = 0
            for dg_id, info in scored_players.items():
                adj = wave_adjs.get(info["player_name"], 0.0)
                if adj != 0.0:
                    info["score"] += adj
                    info["wave_adj"] = adj
                    wave_applied += 1
            if wave_applied:
                print(f"  Wave advantage applied to {wave_applied} players")

        # Step 2: Rank by model score
        ranked = sorted(scored_players.items(), key=lambda x: x[1]["score"], reverse=True)
        for rank, (dg_id, info) in enumerate(ranked, 1):
            info["model_rank"] = rank

        all_scores = [info["score"] for _, info in ranked]
        dg_id_order = [dg_id for dg_id, _ in ranked]

        # Auto-calibrate softmax temperature from score distribution.
        # Using stdev as temperature makes the top player's win probability
        # roughly match the market (e.g., ~24% for a dominant favorite).
        score_stdev = statistics.stdev(all_scores) if len(all_scores) > 1 else 1.0
        base_temperature = max(score_stdev, 0.5)  # Floor to avoid division issues

        # Step 3: For each market, compute softmax probs and find edges
        recommendations = []

        for market_key, entries in market_data.items():
            # Win market: softmax gives P(win) directly.
            # Top-N markets: softmax gives relative model ranking; we scale
            # to match the market's probability level so edges reflect
            # the model's disagreement on WHICH players, not on base rates.
            temperature = base_temperature
            raw_probs = softmax_probabilities(all_scores, temperature)

            # Map dg_id -> raw model prob
            raw_prob_map = {dg_id_order[i]: raw_probs[i] for i in range(len(dg_id_order))}

            # First pass: collect market implied probs and raw model probs
            # for scored players so we can compute the scaling factor.
            player_odds_data = []
            for entry in entries:
                dg_id = entry.get("dg_id")
                if dg_id not in scored_players:
                    continue
                raw_prob = raw_prob_map.get(dg_id, 0)

                best_odds = None
                best_book = None
                best_odds_american = None
                for book, field in [("draftkings", "draftkings_odds"),
                                    ("fanduel", "fanduel_odds"),
                                    ("vip365", "odds")]:
                    odds_str = entry.get(field, "")
                    if odds_str:
                        dec = parse_american_odds(str(odds_str))
                        if dec and (best_odds is None or dec > best_odds):
                            best_odds = dec
                            best_book = book
                            best_odds_american = str(odds_str)
                if best_odds:
                    player_odds_data.append({
                        "dg_id": dg_id,
                        "raw_prob": raw_prob,
                        "implied_prob": 1 / best_odds,
                        "best_odds": best_odds,
                        "best_book": best_book,
                        "best_odds_american": best_odds_american,
                    })

            if not player_odds_data:
                continue

            # Compute scaling factor: match average model prob to average
            # market implied prob so edge reflects relative disagreement.
            avg_raw = statistics.mean([p["raw_prob"] for p in player_odds_data])
            avg_implied = statistics.mean([p["implied_prob"] for p in player_odds_data])
            scale = avg_implied / avg_raw if avg_raw > 0 else 1.0

            # Second pass: compute scaled model probs and edges
            for pod in player_odds_data:
                dg_id = pod["dg_id"]
                player_info = scored_players[dg_id]
                if player_info["model_rank"] > MAX_PLAYERS:
                    continue

                model_prob = min(pod["raw_prob"] * scale, 0.95)
                implied_prob = pod["implied_prob"]

                # DataGolf pre-tournament blend (top-N markets only; win uses WIN_BLEND below).
                # Apply the configured blend to top-N markets; win already has
                # WIN_BLEND with market-implied so DG-blend there would double-shrink.
                if (DG_PT_BLEND > 0 and market_key != "win"
                        and self.dg_pt_live):
                    dg_pt = self.dg_pt_live.get(str(pod["dg_id"]))
                    if dg_pt:
                        dg_prob = dg_pt.get(market_key)
                        if dg_prob is not None:
                            model_prob = (1 - DG_PT_BLEND) * model_prob + DG_PT_BLEND * dg_prob
                            model_prob = min(model_prob, 0.95)

                # Win market: blend model prob with market implied prob
                # Shrinks overconfident estimates toward market consensus
                if market_key == "win":
                    betting_prob = WIN_BLEND_ALPHA * model_prob + (1 - WIN_BLEND_ALPHA) * implied_prob
                else:
                    betting_prob = model_prob

                edge = (betting_prob - implied_prob) / implied_prob * 100

                if edge > MIN_EDGE:
                    market_kelly = KELLY_FRACTIONS.get(market_key, 0.25)
                    kelly_frac, rec_stake = calculate_kelly_stake(
                        betting_prob, pod["best_odds"], market_kelly, DEFAULT_BANKROLL
                    )
                    recommendations.append({
                        "dg_id": dg_id,
                        "player_name": player_info["player_name"],
                        "market": market_key,
                        "model_rank": player_info["model_rank"],
                        "sg_avg": round(player_info["sg_avg"], 4),
                        "grass_adj": round(player_info["grass_adj"], 4),
                        "wave_adj": round(player_info["wave_adj"], 3),
                        "score": round(player_info["score"], 4),
                        "odds_american": pod["best_odds_american"],
                        "book": pod["best_book"],
                        "implied_prob": round(implied_prob, 4),
                        "model_prob": round(betting_prob, 4),
                        "edge": round(edge, 1),
                        "kelly_fraction": kelly_frac,
                        "recommended_stake": rec_stake,
                    })

        recommendations.sort(key=lambda x: x["edge"], reverse=True)

        # Enforce tournament-level bankroll cap: total stakes <= 10% of bankroll
        total_stake = sum(r["recommended_stake"] for r in recommendations)
        max_total = DEFAULT_BANKROLL * MAX_TOURNAMENT_PCT
        if total_stake > max_total and total_stake > 0:
            scale = max_total / total_stake
            for r in recommendations:
                r["recommended_stake"] = round(r["recommended_stake"] * scale, 2)

        # Track Gates 2-4: scored players who didn't make the cut into recommendations
        recommended_ids = {r["dg_id"] for r in recommendations}
        has_odds_in_any_market = set()
        for entries in market_data.values():
            for entry in entries:
                did = entry.get("dg_id")
                if did and any(entry.get(f) for f in ["draftkings_odds", "fanduel_odds", "odds"]):
                    has_odds_in_any_market.add(did)

        for dg_id, info in scored_players.items():
            if dg_id in recommended_ids:
                continue
            if dg_id not in has_odds_in_any_market:
                reason = "no_odds"
                detail = "No DraftKings, FanDuel, or VIP365 market offered"
            elif info["model_rank"] > MAX_PLAYERS:
                reason = "outside_top_n"
                detail = f"Model rank #{info['model_rank']} (cutoff: top {MAX_PLAYERS})"
            else:
                reason = "no_edge"
                detail = f"Edge < {MIN_EDGE}% threshold in all eligible markets"
            self._last_exclusions.append({
                "player_name": info["player_name"],
                "dg_id": dg_id,
                "dg_rank": None,  # annotated in main() via datagolf_field.json
                "reason": reason,
                "detail": detail,
                "model_rank": info["model_rank"],
                "sg_avg": round(info["sg_avg"], 3),
            })

        return recommendations


    def backtest(self, kelly_configs=None, bankroll=1000.0, min_edge=MIN_EDGE):
        """Run time-series cross-validation backtest across historical events.

        For each event with historical odds:
        1. Score players using ONLY prior data
        2. Compute softmax probabilities + market scaling
        3. Find edges vs historical close odds
        4. Apply Kelly sizing, settle using bet_outcome_numeric
        5. Track P&L per strategy/market

        Returns dict with full backtest results per Kelly strategy.
        """
        if kelly_configs is None:
            kelly_configs = {
                "eighth_kelly": {"win": 0.125, "top_5": 0.125, "top_10": 0.125, "top_20": 0.125},
                "quarter_kelly": {"win": 0.25, "top_5": 0.25, "top_10": 0.25, "top_20": 0.25},
                "third_kelly": {"win": 0.333, "top_5": 0.333, "top_10": 0.333, "top_20": 0.333},
                "half_kelly": {"win": 0.5, "top_5": 0.5, "top_10": 0.5, "top_20": 0.5},
                "market_tiered": dict(KELLY_FRACTIONS),
                "flat_10": "flat",
            }

        # Collect events with historical odds, sorted chronologically
        odds_events = []
        for eid, ev in self.events.items():
            ho = ev.get("historical_odds")
            if ho and ho.get("player_odds"):
                odds_events.append((eid, ev))
        odds_events.sort(key=lambda x: x[1].get("event_completed", ""))
        print(f"  Found {len(odds_events)} events with historical odds")

        # Initialize per-strategy results
        results = {}
        for name in kelly_configs:
            results[name] = {
                "bets": [],
                "by_market": {m: {"bets": 0, "wins": 0, "staked": 0.0, "pnl": 0.0}
                              for m in ["win", "top_5", "top_10", "top_20"]},
            }

        events_with_bets = 0

        for event_idx, (eid, ev) in enumerate(odds_events):
            event_date = ev.get("event_completed", "")
            if not event_date:
                continue
            ho = ev["historical_odds"]
            event_grass_types = self.event_grass.get(eid, {})

            # Score all players who have historical odds
            scored = {}
            for dg_id_str, pod in ho["player_odds"].items():
                dg_id = int(dg_id_str) if dg_id_str.isdigit() else dg_id_str
                player_name = pod.get("player_name", "")
                score, sg_avg, grass_adj = self.calculate_score(
                    dg_id, event_date, event_grass_types, player_name
                )
                if score is not None:
                    scored[dg_id] = {
                        "score": score, "sg_avg": sg_avg, "grass_adj": grass_adj,
                        "player_name": player_name, "odds_data": pod,
                    }

            if len(scored) < 10:
                continue

            # Rank and softmax
            ranked = sorted(scored.items(), key=lambda x: x[1]["score"], reverse=True)
            for rank, (dg_id, info) in enumerate(ranked, 1):
                info["model_rank"] = rank

            all_scores = [info["score"] for _, info in ranked]
            dg_id_order = [dg_id for dg_id, _ in ranked]
            score_stdev = statistics.stdev(all_scores) if len(all_scores) > 1 else 1.0
            temperature = max(score_stdev, 0.5)
            raw_probs = softmax_probabilities(all_scores, temperature)
            raw_prob_map = {dg_id_order[i]: raw_probs[i] for i in range(len(dg_id_order))}

            event_had_bets = False

            for market in ["win", "top_5", "top_10", "top_20"]:
                # Collect players with odds in this market
                player_market_data = []
                for dg_id, info in scored.items():
                    dg_id_str = str(dg_id)
                    market_odds = info["odds_data"].get("markets", {}).get(market, {})

                    # Get best close odds between DK and FD
                    best_dec = None
                    best_book = None
                    best_outcome = None
                    for book in ["draftkings", "fanduel"]:
                        bd = market_odds.get(book, {})
                        close_str = bd.get("close_odds")
                        if close_str:
                            dec = parse_american_odds(close_str)
                            if dec and (best_dec is None or dec > best_dec):
                                best_dec = dec
                                best_book = book
                                best_outcome = bd.get("bet_outcome_numeric")

                    if best_dec and best_outcome is not None:
                        player_market_data.append({
                            "dg_id": dg_id,
                            "raw_prob": raw_prob_map.get(dg_id, 0),
                            "implied_prob": 1 / best_dec,
                            "decimal_odds": best_dec,
                            "book": best_book,
                            "bet_outcome": best_outcome,
                            "model_rank": info["model_rank"],
                            "player_name": info["player_name"],
                        })

                if not player_market_data:
                    continue

                # Market scaling
                avg_raw = statistics.mean([p["raw_prob"] for p in player_market_data])
                avg_implied = statistics.mean([p["implied_prob"] for p in player_market_data])
                scale = avg_implied / avg_raw if avg_raw > 0 else 1.0

                for pod in player_market_data:
                    if pod["model_rank"] > MAX_PLAYERS:
                        continue
                    model_prob = min(pod["raw_prob"] * scale, 0.95)
                    implied_prob = pod["implied_prob"]

                    # Win market: blend model prob with market implied prob
                    if market == "win":
                        betting_prob = WIN_BLEND_ALPHA * model_prob + (1 - WIN_BLEND_ALPHA) * implied_prob
                    else:
                        betting_prob = model_prob

                    edge = (betting_prob - implied_prob) / implied_prob * 100
                    if edge <= min_edge:
                        continue

                    # This pick qualifies — apply each strategy
                    for strat_name, fracs in kelly_configs.items():
                        if fracs == "flat":
                            stake = 10.0
                        else:
                            mf = fracs.get(market, 0.25)
                            _, stake = calculate_kelly_stake(
                                betting_prob, pod["decimal_odds"], mf, bankroll
                            )
                        if stake <= 0:
                            continue

                        # Settle: profit = stake * (outcome * odds - 1)
                        profit = round(stake * (pod["bet_outcome"] * pod["decimal_odds"] - 1), 2)

                        results[strat_name]["bets"].append({
                            "event": eid,
                            "event_date": event_date,
                            "market": market,
                            "player": pod["player_name"],
                            "rank": pod["model_rank"],
                            "stake": stake,
                            "odds": pod["decimal_odds"],
                            "model_prob": round(betting_prob, 4),
                            "edge": round(edge, 1),
                            "outcome": pod["bet_outcome"],
                            "profit": profit,
                        })
                        r = results[strat_name]["by_market"][market]
                        r["bets"] += 1
                        r["staked"] += stake
                        r["pnl"] += profit
                        if pod["bet_outcome"] > 0:
                            r["wins"] += 1
                        event_had_bets = True

            if event_had_bets:
                events_with_bets += 1

        # Compute aggregate metrics per strategy
        for strat_name, data in results.items():
            bets = data["bets"]
            if not bets:
                data["summary"] = {"total_bets": 0}
                continue

            total_staked = sum(b["stake"] for b in bets)
            total_profit = sum(b["profit"] for b in bets)
            roi = (total_profit / total_staked * 100) if total_staked > 0 else 0
            wins = sum(1 for b in bets if b["outcome"] > 0)

            # Max drawdown on running P&L
            running = 0.0
            peak = 0.0
            max_dd = 0.0
            for b in bets:
                running += b["profit"]
                if running > peak:
                    peak = running
                dd = peak - running
                if dd > max_dd:
                    max_dd = dd

            # Rolling bankroll simulation
            roll = bankroll
            roll_peak = bankroll
            roll_max_dd_pct = 0.0
            for b in bets:
                roll += b["profit"]
                if roll > roll_peak:
                    roll_peak = roll
                if roll_peak > 0:
                    dd_pct = (roll_peak - roll) / roll_peak * 100
                    if dd_pct > roll_max_dd_pct:
                        roll_max_dd_pct = dd_pct

            # Sharpe ratio (bet-level)
            profits = [b["profit"] for b in bets]
            avg_p = statistics.mean(profits)
            std_p = statistics.stdev(profits) if len(profits) > 1 else 1
            sharpe = avg_p / std_p if std_p > 0 else 0

            data["summary"] = {
                "total_bets": len(bets),
                "total_staked": round(total_staked, 2),
                "total_profit": round(total_profit, 2),
                "roi_pct": round(roi, 1),
                "win_rate_pct": round(wins / len(bets) * 100, 1),
                "avg_edge": round(statistics.mean([b["edge"] for b in bets]), 1),
                "max_drawdown": round(max_dd, 2),
                "sharpe_ratio": round(sharpe, 4),
                "events_tested": events_with_bets,
                "rolling_final_bankroll": round(roll, 2),
                "rolling_max_drawdown_pct": round(roll_max_dd_pct, 1),
            }

        return results


def load_market_data():
    """Load field data for each market type separately.

    Returns dict of {market_key: [entries]} where each market's entries
    have that market's specific DK/FD odds.
    """
    market_data = {}
    for market_key, filename in MARKET_FILES.items():
        filepath = JSON_EXTRACT_DIR / filename
        if filepath.exists():
            with open(filepath) as fp:
                data = json.load(fp)
            results = data.get("results", {})
            entries = []
            for market_name, md in results.items():
                entries.extend(md.get("entries", []))
            if entries:
                market_data[market_key] = entries
    return market_data


def main():
    print("=" * 80)
    print("PRODUCTION GOLF BETTING MODEL")
    print("Research model: SG + grass + volatility + tee-to-green + market blend")
    print("=" * 80)

    model = ProductionBettingModel()

    print("\nLoading current tournament market data...")
    market_data = load_market_data()

    if market_data:
        for mk, entries in market_data.items():
            print(f"  {mk}: {len(entries)} players")

        # Load grass types from tournament_config.json for the current tournament.
        # The canonical data has no "current_event" entry, so without this the
        # grass adjustment would be 0 for all players.
        tournament_config_path = REPO_ROOT / "test_env" / "tournament_config.json"
        try:
            with open(tournament_config_path) as f:
                tc = json.load(f)
            current_grass = tc.get("grass_types", {})
            if current_grass:
                model.event_grass["current_event"] = current_grass
                print(f"  Grass types: {current_grass}")
        except Exception as e:
            print(f"  Warning: could not load tournament_config.json: {e}")

        recommendations = model.generate_recommendations(market_data, "current_event")

        # Annotate dg_rank for scored exclusions (no_odds / outside_top_n / no_edge)
        field_file = JSON_EXTRACT_DIR / "datagolf_field.json"
        if field_file.exists():
            with open(field_file) as f:
                field_json = json.load(f)
            dg_rank_map = {p["dg_id"]: p.get("dg_rank") for p in field_json.get("field", [])}
            for excl in model._last_exclusions:
                if excl["dg_rank"] is None:
                    excl["dg_rank"] = dg_rank_map.get(excl["dg_id"])

        print(f"\nGenerated {len(recommendations)} recommendations")
        print(f"Tracked {len(model._last_exclusions)} excluded players")

        for market in ["win", "top_5", "top_10", "top_20"]:
            market_recs = [r for r in recommendations if r["market"] == market]
            if market_recs:
                print(f"\n--- {market.upper()} ({len(market_recs)} picks) ---")
                print(f"  {'Player':<25s} | {'Odds':>8s} | {'Rank':>4s} | {'Model%':>7s} | {'Mkt%':>7s} | {'Edge':>6s} | {'Stake':>7s}")
                print(f"  {'-'*80}")
                for rec in market_recs[:8]:
                    print(f"  {rec['player_name']:<25s} | {rec['odds_american']:>8s} | {rec['model_rank']:>4d} | {rec['model_prob']*100:>6.1f}% | {rec['implied_prob']*100:>6.1f}% | {rec['edge']:>+5.1f}% | ${rec['recommended_stake']:>6.2f}")

        # Get wave forecast summary for output
        wave_summary = {}
        if WAVE_ENABLED and get_forecast_summary is not None:
            wave_summary = get_forecast_summary()
            if wave_summary.get("has_data") and wave_summary.get("active_rounds"):
                print(f"\n  Wave forecast: {len(wave_summary['active_rounds'])} active round(s), "
                      f"max diff {wave_summary['max_diff']} km/h, "
                      f"advantage: {wave_summary['advantage_wave']}")
            else:
                print("\n  Wave forecast: no significant wave advantage detected")

        # Snapshot previous model output before overwriting
        output_path = JSON_EXTRACT_DIR / "production_model_recommendations.json"
        try:
            from odds_snapshot import snapshot_before_overwrite
            snapshot_before_overwrite(output_path)
        except Exception as e:
            print(f"  Warning: could not snapshot previous model output: {e}")

        # Save
        model_output = {
            "generated_at": datetime.now().isoformat(),
            "model": "sg_grass_v1",
            "model_config": {
                "lookback": LOOKBACK,
                "grass_weight": GRASS_WEIGHT,
                "grass_confidence_events": GRASS_CONFIDENCE_EVENTS,
                "max_players": MAX_PLAYERS,
                "kelly_fractions": KELLY_FRACTIONS,
                "default_bankroll": DEFAULT_BANKROLL,
                "wave_enabled": WAVE_ENABLED,
            },
            "wave_forecast": wave_summary,
            "recommendations": recommendations,
            "exclusions": model._last_exclusions,
        }
        with open(output_path, "w") as f:
            json.dump(model_output, f, indent=2)

        print(f"\nSaved to: {output_path}")

        # Track recommendation evolution across the week
        try:
            from model_recommendation_tracker import save_model_snapshot
            save_model_snapshot(model_output, model_type="production",
                                tournament_name=tc.get("current_tournament") if 'tc' in dir() else None)
        except Exception as e:
            print(f"  Warning: could not save model snapshot: {e}")
    else:
        print("No tournament market data found.")


def run_backtest(bankroll=1000.0, min_edge=MIN_EDGE):
    """Run and display backtest results."""
    print("=" * 90)
    print("PRODUCTION MODEL BACKTEST — Time-Series Cross-Validation")
    print(f"Bankroll: ${bankroll:.0f} | Min Edge: {min_edge}% | Lookback: {LOOKBACK} events")
    print("=" * 90)

    model = ProductionBettingModel()
    results = model.backtest(bankroll=bankroll, min_edge=min_edge)

    # Strategy comparison table
    print(f"\n{'Strategy':<18s} | {'Bets':>6s} | {'Staked':>10s} | {'Profit':>10s} | {'ROI':>7s} | {'Win%':>6s} | {'Sharpe':>7s} | {'MaxDD':>9s} | {'DD%':>6s}")
    print("-" * 105)
    for name in ["eighth_kelly", "quarter_kelly", "third_kelly", "half_kelly", "market_tiered", "flat_10"]:
        s = results.get(name, {}).get("summary", {})
        if not s or s.get("total_bets", 0) == 0:
            continue
        print(f"  {name:<16s} | {s['total_bets']:>6d} | ${s['total_staked']:>9.2f} | ${s['total_profit']:>+9.2f} | {s['roi_pct']:>+6.1f}% | {s['win_rate_pct']:>5.1f}% | {s['sharpe_ratio']:>7.4f} | ${s['max_drawdown']:>8.2f} | {s['rolling_max_drawdown_pct']:>5.1f}%")

    # Per-market breakdown for market_tiered (or best strategy)
    best_name = max(
        (n for n in results if results[n].get("summary", {}).get("total_bets", 0) > 0),
        key=lambda n: results[n]["summary"]["roi_pct"]
    )
    print(f"\nPer-Market Breakdown ({best_name}):")
    print(f"  {'Market':<10s} | {'Bets':>6s} | {'Staked':>10s} | {'Profit':>10s} | {'ROI':>7s} | {'Win%':>6s}")
    print(f"  {'-'*65}")
    for market in ["win", "top_5", "top_10", "top_20"]:
        m = results[best_name]["by_market"].get(market, {})
        if m.get("bets", 0) == 0:
            continue
        wr = m["wins"] / m["bets"] * 100 if m["bets"] > 0 else 0
        roi = m["pnl"] / m["staked"] * 100 if m["staked"] > 0 else 0
        print(f"  {market:<10s} | {m['bets']:>6d} | ${m['staked']:>9.2f} | ${m['pnl']:>+9.2f} | {roi:>+6.1f}% | {wr:>5.1f}%")

    print(f"\n  Best strategy: {best_name} ({results[best_name]['summary']['roi_pct']:+.1f}% ROI)")
    print(f"  Events tested: {results[best_name]['summary']['events_tested']}")

    # Save results
    output_path = REPO_ROOT / "4_modeling" / "backtest_results.json"
    save_data = {
        "generated_at": datetime.now().isoformat(),
        "bankroll": bankroll,
        "min_edge": min_edge,
        "model_config": {
            "lookback": LOOKBACK, "grass_weight": GRASS_WEIGHT,
            "max_players": MAX_PLAYERS, "min_edge": MIN_EDGE,
        },
        "strategies": {},
    }
    for name, data in results.items():
        save_data["strategies"][name] = {
            "summary": data.get("summary", {}),
            "by_market": {k: dict(v) for k, v in data["by_market"].items()},
        }
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Production Golf Betting Model")
    parser.add_argument("--backtest", action="store_true", help="Run historical backtest")
    parser.add_argument("--bankroll", type=float, default=DEFAULT_BANKROLL, help="Backtest bankroll")
    parser.add_argument("--min-edge", type=float, default=MIN_EDGE, help="Minimum edge %%")
    args = parser.parse_args()

    if args.backtest:
        run_backtest(args.bankroll, args.min_edge)
    else:
        main()
