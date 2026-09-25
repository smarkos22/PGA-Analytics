#!/usr/bin/env python3
"""
FRL (First Round Leader) Production Betting Model

Monte Carlo simulation-based model for FRL bets using per-round
strokes gained data from the canonical historical dataset.

Key differences from the outright production model:
- Uses R1-specific SG history (not tournament totals)
- Monte Carlo simulation (not softmax) for tie-aware probabilities
- R1 tendency (R1 vs R2-R4 advantage) as a core feature
- Grass adjustment scaled for single round (~1/4 of tournament scale)
- Calibration backtest (no historical FRL close odds available from DataGolf)

Usage:
    python3.10 4_modeling/frl_production_model.py                # Generate recommendations
    python3.10 4_modeling/frl_production_model.py --backtest     # Run calibration backtest
    python3.10 4_modeling/frl_production_model.py --backtest --bankroll 2000
"""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

import sys

# ── Paths ──
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "test_env"))
try:
    from wave_advantage import compute_wave_adjustments, get_forecast_summary
except ImportError:
    compute_wave_adjustments = None
    get_forecast_summary = None
CANONICAL_DIR = REPO_ROOT / "3_canonical_data_aggregation" / "canonical_data"
CANONICAL_FILES = sorted(CANONICAL_DIR.glob("canonical_historical_data_*.json"))
CANONICAL_FILE = CANONICAL_FILES[-1].name if CANONICAL_FILES else ""
GRASS_FILE = REPO_ROOT / "4_modeling" / "grass_type_analysis_output" / "grass_type_all_data.csv"
JSON_EXTRACT_DIR = REPO_ROOT / "test_env" / "JSON_extract"
FRL_ODDS_FILE = JSON_EXTRACT_DIR / "datagolf_outrights_first_round_leader.json"
OUTPUT_FILE = JSON_EXTRACT_DIR / "frl_production_model_recommendations.json"

# ── Research parameters (out-of-sample validation required) ──
LOOKBACK = 8               # R1 events for trailing SG average (shorter than outright's 10; R1 noisier)
MIN_HIST = 5               # Minimum R1 rounds to score a player
MAX_PLAYERS = 20           # More players than outright (FRL is higher variance)
GRASS_WEIGHT = 0.20        # Single-round scale (~1/2 of outright's 0.40)
GRASS_CONFIDENCE_EVENTS = 20
VOLATILITY_WEIGHT = -0.15  # Penalize inconsistent players (same direction as outright)
R1_TENDENCY_WEIGHT = 0.25  # Small positive R1-specific advantage signal
R1_FORM_WEIGHT = 0.0       # Off — other features capture form; adds noise alone
SG_T2G_WEIGHT = 0.0        # Not used; sub-category weights below replace this
SG_PUTT_WEIGHT = 0.10      # Putting emphasis (FRL leaders avg 1.70 SG putting)
SG_APP_WEIGHT = 0.08       # Approach emphasis (FRL leaders avg 1.27 SG approach)
SG_OTT_WEIGHT = 0.04       # Off-the-tee emphasis
WAVE_ENABLED = True        # Enable tee-time wave advantage from weather forecast
MIN_EDGE = 5.0             # Minimum edge % to recommend
N_SIMULATIONS = 50_000     # Monte Carlo simulation count
TIE_THRESHOLD_SG = 0.01    # Players within this of max are co-leaders

# ── R1 Tendency parameters ──
R1_CONFIDENCE_N_THRESHOLD = 20
R1_CONFIDENCE_COHEN_THRESHOLD = 0.5
RECENT_BLEND_WEIGHT = 0.6       # 60% recent, 40% career
RECENT_WINDOW = 12               # Last 12 tournaments for "recent"
MIN_TOURNAMENTS_FOR_TENDENCY = 8 # Minimum to compute tendency

# ── R1 variance defaults by skill tier ──
DEFAULT_R1_VARIANCE_ELITE = 1.9   # sg_avg > 2.0
DEFAULT_R1_VARIANCE_MID = 2.1     # sg_avg > 0.5
DEFAULT_R1_VARIANCE_FIELD = 2.4   # sg_avg <= 0.5
FALLBACK_R1_VARIANCE = 2.3

# ── Kelly sizing ──
KELLY_FRACTION = 0.25      # Quarter-Kelly for FRL (to be calibrated)
MAX_STAKE_PCT = 0.05
MIN_STAKE = 5.0
DEFAULT_BANKROLL = 1000.0  # Fictional example budget for the public snapshot


# ─────────────────────────────────────────────────
# Statistical functions (Welch's t-test, Cohen's d)
# ─────────────────────────────────────────────────

def _betacf(a, b, x):
    max_iter = 200
    eps = 3.0e-7
    am, bm, az = 1.0, 1.0, 1.0
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    bz = 1.0 - qab * x / qap
    for m in range(1, max_iter + 1):
        em = float(m)
        tem = em + em
        d = em * (b - em) * x / ((qam + tem) * (a + tem))
        ap = az + d * am
        bp = bz + d * bm
        d = -(a + em) * (qab + em) * x / ((a + tem) * (qap + tem))
        app = ap + d * az
        bpp = bp + d * bz
        am = ap / bpp
        bm = bp / bpp
        az = app / bpp
        bz = 1.0
        if abs(az - (app / bpp if bpp else az)) < eps * abs(az):
            return az
    return az


def regularized_beta(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf(t, df):
    if df <= 0:
        return None
    x = df / (df + t * t)
    ib = regularized_beta(df / 2.0, 0.5, x)
    return 1.0 - 0.5 * ib if t >= 0 else 0.5 * ib


def welch_ttest(a, b):
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return None, None
    m1, m2 = statistics.mean(a), statistics.mean(b)
    v1, v2 = statistics.variance(a), statistics.variance(b)
    denom = (v1 / n1) + (v2 / n2)
    if denom == 0:
        return None, None
    t_stat = (m1 - m2) / math.sqrt(denom)
    df_num = denom * denom
    df_den = ((v1 ** 2) / (n1 ** 2 * (n1 - 1))) + ((v2 ** 2) / (n2 ** 2 * (n2 - 1)))
    if df_den == 0:
        return None, None
    df = df_num / df_den
    cdf = t_cdf(abs(t_stat), df)
    if cdf is None:
        return None, None
    return t_stat, 2.0 * (1.0 - cdf)


def cohen_d_func(a, b):
    if len(a) < 2 or len(b) < 2:
        return None
    v1 = statistics.variance(a)
    v2 = statistics.variance(b)
    pooled = math.sqrt((v1 + v2) / 2.0)
    if pooled == 0:
        return None
    return (statistics.mean(a) - statistics.mean(b)) / pooled


# ─────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────

def parse_american_odds(odds_str):
    if not odds_str:
        return None
    try:
        odds = int(str(odds_str).replace("+", "").replace(",", ""))
        return (odds / 100) + 1 if odds > 0 else (100 / abs(odds)) + 1
    except (ValueError, ZeroDivisionError):
        return None


def load_grass_data():
    """Load grass significance data from CSV."""
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
                except (ValueError, TypeError):
                    pass
    return grass_data


# ─────────────────────────────────────────────────
# FRL Production Model
# ─────────────────────────────────────────────────

class FRLProductionModel:
    def __init__(self):
        print("Loading FRL production model...")
        self.grass_data = load_grass_data()
        self.events = {}
        self.r1_history = defaultdict(list)       # dg_id -> [{date, sg_total, sg_t2g, sg_app, sg_putt, sg_ott, sg_arg, event_id}]
        self.other_history = defaultdict(list)     # dg_id -> [{date, sg_total, event_id, round}]
        self.r1_tendencies = {}                    # dg_id -> tendency dict
        self.event_grass = {}                      # event_id -> {surface: grass_type}
        self._load_data()
        self._build_r1_tendencies()
        print(f"  R1 history for {len(self.r1_history)} players")
        print(f"  R1 tendencies for {len(self.r1_tendencies)} players")
        print(f"  Grass data for {len(self.grass_data)} players")

    def _load_data(self):
        """Load canonical data and extract per-round SG breakdowns."""
        canonical_path = CANONICAL_DIR / CANONICAL_FILE
        if not canonical_path.exists():
            raise FileNotFoundError(f"Canonical data not found: {canonical_path}")

        with open(canonical_path) as f:
            canonical = json.load(f)

        self.events = canonical.get("events_by_id", {})

        # Extract event grass types
        for eid, ev in self.events.items():
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
                self.event_grass[eid] = grass_map

        # Build per-round histories
        sorted_events = sorted(self.events.items(), key=lambda x: x[1].get("event_completed", ""))

        for event_id, event in sorted_events:
            event_date = event.get("event_completed", "")
            if not event_date:
                continue
            for score in event.get("scores", []):
                dg_id = score.get("dg_id")
                if not dg_id:
                    continue
                fin_text = str(score.get("fin_text", "")).strip().upper()
                if fin_text in {"WD", "DQ"}:
                    continue

                player_name = score.get("player_name", "")

                # R1 data
                rd1 = score.get("round_1")
                if isinstance(rd1, dict) and rd1.get("sg_total") is not None:
                    self.r1_history[dg_id].append({
                        "event_id": event_id,
                        "event_date": event_date,
                        "player_name": player_name,
                        "sg_total": float(rd1["sg_total"]),
                        "sg_t2g": float(rd1.get("sg_t2g", 0) or 0),
                        "sg_app": float(rd1.get("sg_app", 0) or 0),
                        "sg_putt": float(rd1.get("sg_putt", 0) or 0),
                        "sg_ott": float(rd1.get("sg_ott", 0) or 0),
                        "sg_arg": float(rd1.get("sg_arg", 0) or 0),
                    })

                # R2-R4 data (for tendency calculation)
                for r in range(2, 5):
                    rd = score.get(f"round_{r}")
                    if isinstance(rd, dict) and rd.get("sg_total") is not None:
                        self.other_history[dg_id].append({
                            "event_id": event_id,
                            "event_date": event_date,
                            "round": r,
                            "sg_total": float(rd["sg_total"]),
                        })

        print(f"  Loaded {sum(len(v) for v in self.r1_history.values())} R1 rounds from {len(self.events)} events")

    def _build_r1_tendencies(self):
        """Calculate R1 vs R2-R4 tendency for each player with Welch's t-test."""
        self.r1_tendencies = {}

        for dg_id, r1_hist in self.r1_history.items():
            other_hist = self.other_history.get(dg_id, [])
            n_tournaments = len(r1_hist)
            if n_tournaments < 2:
                continue

            # Career tendency
            r1_sgs = [h["sg_total"] for h in r1_hist]
            other_sgs = [h["sg_total"] for h in other_hist]

            career_r1_mean = statistics.mean(r1_sgs)
            career_other_mean = statistics.mean(other_sgs) if other_sgs else 0.0
            career_advantage = career_r1_mean - career_other_mean

            career_t_stat, career_p_value = welch_ttest(r1_sgs, other_sgs)
            career_cd = cohen_d_func(r1_sgs, other_sgs)
            career_r1_std = statistics.stdev(r1_sgs) if len(r1_sgs) > 1 else FALLBACK_R1_VARIANCE

            # Confidence weighting
            career_n_weight = min(1.0, n_tournaments / R1_CONFIDENCE_N_THRESHOLD)
            career_effect_weight = min(1.0, abs(career_cd) / R1_CONFIDENCE_COHEN_THRESHOLD) if career_cd is not None else 0.0
            career_confidence = career_n_weight * career_effect_weight
            career_weighted = career_advantage * career_confidence

            # Recent tendency (last RECENT_WINDOW)
            sorted_r1 = sorted(r1_hist, key=lambda x: x["event_date"], reverse=True)
            recent_r1 = sorted_r1[:RECENT_WINDOW]
            recent_event_ids = {h["event_id"] for h in recent_r1}

            recent_r1_sgs = [h["sg_total"] for h in recent_r1]
            recent_other_sgs = [h["sg_total"] for h in other_hist if h["event_id"] in recent_event_ids]

            recent_r1_mean = statistics.mean(recent_r1_sgs) if recent_r1_sgs else 0.0
            recent_other_mean = statistics.mean(recent_other_sgs) if recent_other_sgs else 0.0
            recent_advantage = recent_r1_mean - recent_other_mean

            recent_cd = None
            if len(recent_r1_sgs) >= 2 and len(recent_other_sgs) >= 2:
                recent_cd = cohen_d_func(recent_r1_sgs, recent_other_sgs)

            recent_n = len(recent_r1_sgs)
            recent_n_weight = min(1.0, recent_n / R1_CONFIDENCE_N_THRESHOLD)
            recent_effect_weight = min(1.0, abs(recent_cd) / R1_CONFIDENCE_COHEN_THRESHOLD) if recent_cd is not None else 0.0
            recent_confidence = recent_n_weight * recent_effect_weight
            recent_weighted = recent_advantage * recent_confidence

            # Blend
            if recent_n >= MIN_TOURNAMENTS_FOR_TENDENCY:
                blended = RECENT_BLEND_WEIGHT * recent_weighted + (1 - RECENT_BLEND_WEIGHT) * career_weighted
            else:
                blended = career_weighted

            self.r1_tendencies[dg_id] = {
                "career_advantage": career_advantage,
                "career_p_value": career_p_value,
                "career_cohen_d": career_cd,
                "career_confidence": career_confidence,
                "career_n": n_tournaments,
                "career_r1_std": career_r1_std,
                "recent_advantage": recent_advantage,
                "recent_n": recent_n,
                "blended_r1_tendency": blended,
            }

    def _get_r1_variance(self, dg_id, sg_avg):
        """Player-specific R1 SG stdev, with tier-based default."""
        tendency = self.r1_tendencies.get(dg_id)
        if tendency and tendency["career_n"] >= 4:
            return tendency["career_r1_std"]
        if sg_avg > 2.0:
            return DEFAULT_R1_VARIANCE_ELITE
        elif sg_avg > 0.5:
            return DEFAULT_R1_VARIANCE_MID
        elif sg_avg > -1.0:
            return DEFAULT_R1_VARIANCE_FIELD
        return FALLBACK_R1_VARIANCE

    def calculate_r1_score(self, dg_id, event_date, event_grass_types, player_name,
                           lookback=LOOKBACK, min_hist=MIN_HIST, grass_weight=GRASS_WEIGHT,
                           r1_tendency_weight=R1_TENDENCY_WEIGHT, r1_form_weight=R1_FORM_WEIGHT,
                           volatility_weight=VOLATILITY_WEIGHT, sg_t2g_weight=SG_T2G_WEIGHT,
                           sg_putt_weight=SG_PUTT_WEIGHT, sg_app_weight=SG_APP_WEIGHT,
                           sg_ott_weight=SG_OTT_WEIGHT):
        """Calculate expected R1 SG for a player.

        Returns dict with score components, or None if insufficient data.
        """
        history = self.r1_history.get(dg_id, [])
        prior = [h for h in history if h["event_date"] < event_date]

        if len(prior) < min_hist:
            return None

        if lookback >= 999:
            recent = prior
        else:
            recent = prior[-lookback:]

        # Base R1 SG average
        sg_vals = [h["sg_total"] for h in recent]
        r1_sg_avg = statistics.mean(sg_vals)

        # Optional: blend with R1 t2g
        if sg_t2g_weight > 0:
            t2g_vals = [h.get("sg_t2g", 0) for h in recent]
            t2g_avg = statistics.mean(t2g_vals)
            r1_sg_avg = r1_sg_avg * (1 - sg_t2g_weight) + t2g_avg * sg_t2g_weight

        # Configured SG sub-category emphasis; requires independent validation
        sub_total_w = sg_putt_weight + sg_app_weight + sg_ott_weight
        if sub_total_w > 0:
            base_sg = r1_sg_avg * (1.0 - sub_total_w)
            sub_score = 0.0
            if sg_putt_weight > 0:
                putt_vals = [h.get("sg_putt", 0) for h in recent]
                sub_score += statistics.mean(putt_vals) * sg_putt_weight
            if sg_app_weight > 0:
                app_vals = [h.get("sg_app", 0) for h in recent]
                sub_score += statistics.mean(app_vals) * sg_app_weight
            if sg_ott_weight > 0:
                ott_vals = [h.get("sg_ott", 0) for h in recent]
                sub_score += statistics.mean(ott_vals) * sg_ott_weight
            r1_sg_avg = base_sg + sub_score

        # Grass adjustment (scaled for single round)
        grass_adj = 0
        if player_name and event_grass_types and grass_weight > 0:
            name_key = player_name.strip().lower()
            player_grass = self.grass_data.get(name_key, {})
            if player_grass:
                for surface, grass_type in event_grass_types.items():
                    if surface in player_grass and grass_type in player_grass[surface]:
                        n = player_grass[surface][grass_type].get("n_on", 5)
                        conf = min(1.0, n / GRASS_CONFIDENCE_EVENTS)
                        grass_adj += player_grass[surface][grass_type].get("mean_diff", 0) * grass_weight * conf

        # R1 tendency
        tendency = self.r1_tendencies.get(dg_id, {})
        r1_tendency = tendency.get("blended_r1_tendency", 0.0) * r1_tendency_weight

        # Recent R1 form (exponential decay)
        r1_form = 0
        if r1_form_weight > 0:
            form_entries = sorted(prior, key=lambda x: x["event_date"], reverse=True)[:4]
            if form_entries:
                weights = [math.exp(-0.3 * i) for i in range(len(form_entries))]
                total_w = sum(weights)
                r1_form = sum(e["sg_total"] * w for e, w in zip(form_entries, weights)) / total_w * r1_form_weight

        # Volatility adjustment
        vol_adj = 0
        if volatility_weight != 0 and len(sg_vals) >= 3:
            vol_adj = statistics.stdev(sg_vals) * volatility_weight

        total_score = r1_sg_avg + grass_adj + r1_tendency + r1_form + vol_adj
        variance = self._get_r1_variance(dg_id, r1_sg_avg)

        return {
            "score": total_score,
            "r1_sg_avg": r1_sg_avg,
            "grass_adj": grass_adj,
            "r1_tendency": r1_tendency / r1_tendency_weight if r1_tendency_weight else 0,
            "r1_form": r1_form,
            "vol_adj": vol_adj,
            "variance": variance,
        }

    def simulate_frl_probabilities(self, scored_players, n_sims=N_SIMULATIONS):
        """Monte Carlo simulation for tie-aware FRL probabilities.

        Args:
            scored_players: list of {dg_id, score, variance, ...}
            n_sims: number of simulations
        Returns:
            dict of {dg_id: frl_probability}
        """
        if not scored_players:
            return {}

        expected_sgs = [p["score"] for p in scored_players]
        variances = [p["variance"] for p in scored_players]

        if HAS_NUMPY:
            counts = self._simulate_numpy(expected_sgs, variances, n_sims)
        else:
            counts = self._simulate_python(expected_sgs, variances, n_sims)

        return {scored_players[i]["dg_id"]: counts[i] / n_sims for i in range(len(scored_players))}

    def _simulate_numpy(self, expected_sgs, variances, n_sims):
        mu = np.array(expected_sgs)
        sigma = np.array(variances)
        sims = np.random.normal(mu[None, :], sigma[None, :], size=(n_sims, len(expected_sgs)))
        max_per_sim = sims.max(axis=1, keepdims=True)
        is_leader = np.abs(sims - max_per_sim) < TIE_THRESHOLD_SG
        return is_leader.sum(axis=0).tolist()

    def _simulate_python(self, expected_sgs, variances, n_sims):
        import random
        n = len(expected_sgs)
        counts = [0] * n
        for _ in range(n_sims):
            sampled = [random.gauss(expected_sgs[i], variances[i]) for i in range(n)]
            max_sg = max(sampled)
            for i in range(n):
                if abs(sampled[i] - max_sg) < TIE_THRESHOLD_SG:
                    counts[i] += 1
        return counts

    def _find_r1_leaders(self, event):
        """Find actual R1 leader(s) for a historical event."""
        r1_entries = []
        for score in event.get("scores", []):
            fin_text = str(score.get("fin_text", "")).strip().upper()
            if fin_text in {"WD", "DQ"}:
                continue
            rd1 = score.get("round_1")
            if isinstance(rd1, dict) and rd1.get("sg_total") is not None:
                r1_entries.append({
                    "dg_id": score.get("dg_id"),
                    "sg_total": float(rd1["sg_total"]),
                })
        if not r1_entries:
            return set(), 0
        max_sg = max(e["sg_total"] for e in r1_entries)
        leaders = {e["dg_id"] for e in r1_entries if abs(e["sg_total"] - max_sg) < 0.001}
        return leaders, len(leaders)

    def generate_recommendations(self, bankroll=DEFAULT_BANKROLL, kelly_fraction=KELLY_FRACTION):
        """Generate FRL betting recommendations for current tournament."""
        # Load current FRL odds
        if not FRL_ODDS_FILE.exists():
            print(f"FRL odds file not found: {FRL_ODDS_FILE}")
            return []

        with open(FRL_ODDS_FILE) as f:
            frl_odds_data = json.load(f)

        # Load tournament config for grass types
        config_path = REPO_ROOT / "test_env" / "tournament_config.json"
        event_grass_types = {}
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            event_grass_types = config.get("grass_types", {})

        event_date = datetime.now().strftime("%Y-%m-%d")

        # Score ALL players in the FRL field (scored + unscored)
        odds_entries = frl_odds_data if isinstance(frl_odds_data, list) else frl_odds_data.get("players", frl_odds_data.get("odds", []))
        if not isinstance(odds_entries, list):
            print(f"No FRL odds available: {odds_entries if isinstance(odds_entries, str) else 'invalid data format'}")
            return []
        scored = []       # Players with enough history
        unscored = []     # Players without history → get field-average score

        for entry in odds_entries:
            dg_id = entry.get("dg_id")
            if not dg_id:
                continue
            player_name = entry.get("player_name", "")
            result = self.calculate_r1_score(dg_id, event_date, event_grass_types, player_name)

            # Find best odds
            best_odds = None
            best_book = None
            best_odds_str = None
            for book_key, book_name in [("draftkings", "draftkings"), ("fanduel", "fanduel")]:
                odds_str = entry.get(book_key)
                if not odds_str:
                    continue
                decimal = parse_american_odds(odds_str)
                if decimal and (best_odds is None or decimal > best_odds):
                    best_odds = decimal
                    best_book = book_name
                    best_odds_str = odds_str

            player_entry = {
                "dg_id": dg_id,
                "player_name": player_name,
                "best_odds": best_odds,
                "best_book": best_book,
                "best_odds_str": best_odds_str,
            }

            if result is not None:
                player_entry.update(result)
                scored.append(player_entry)
            else:
                unscored.append(player_entry)

        if not scored:
            print("No players could be scored")
            return []

        # Apply wave advantage (R1 only for FRL)
        if WAVE_ENABLED and compute_wave_adjustments is not None:
            wave_adjs = compute_wave_adjustments(frl_mode=True)
            wave_applied = 0
            for p in scored:
                adj = wave_adjs.get(p["player_name"], 0.0)
                if adj != 0.0:
                    p["score"] += adj
                    p["wave_adj"] = adj
                    wave_applied += 1
                else:
                    p["wave_adj"] = 0.0
            if wave_applied:
                print(f"  Wave advantage (R1) applied to {wave_applied} players")
        else:
            for p in scored:
                p["wave_adj"] = 0.0

        # Sort scored by score
        scored.sort(key=lambda x: x["score"], reverse=True)

        # For unscored players, use field-average score + high variance
        # This ensures MC simulates the full field, not just scored players
        avg_score = statistics.mean(p["score"] for p in scored)
        # Unscored = slightly below average (less info = less confidence)
        default_score = avg_score - 0.5
        for p in unscored:
            p.update({
                "score": default_score,
                "r1_sg_avg": default_score,
                "grass_adj": 0,
                "r1_tendency": 0,
                "r1_form": 0,
                "vol_adj": 0,
                "wave_adj": 0.0,
                "variance": FALLBACK_R1_VARIANCE,
            })

        # Monte Carlo simulation on the FULL field (scored + unscored)
        all_players = scored + unscored
        print(f"  MC simulation: {len(scored)} scored + {len(unscored)} unscored = {len(all_players)} total field")
        frl_probs = self.simulate_frl_probabilities(all_players, N_SIMULATIONS)

        # Market scaling: match average model prob to average implied prob
        implied_probs = []
        model_probs = []
        for p in scored:
            if p["best_odds"]:
                imp = 1.0 / p["best_odds"]
                mod = frl_probs.get(p["dg_id"], 0)
                if mod > 0 and imp > 0:
                    implied_probs.append(imp)
                    model_probs.append(mod)

        scale = 1.0
        if implied_probs and model_probs:
            avg_implied = statistics.mean(implied_probs)
            avg_model = statistics.mean(model_probs)
            if avg_model > 0:
                scale = avg_implied / avg_model

        # Generate recommendations
        recommendations = []
        for rank, p in enumerate(scored, 1):
            if rank > MAX_PLAYERS:
                break
            if not p["best_odds"]:
                continue

            raw_prob = frl_probs.get(p["dg_id"], 0)
            model_prob = min(raw_prob * scale, 0.95)
            implied_prob = 1.0 / p["best_odds"]
            edge = (model_prob - implied_prob) / implied_prob * 100 if implied_prob > 0 else 0

            if edge < MIN_EDGE:
                continue

            # Kelly sizing
            b = p["best_odds"] - 1
            if b > 0 and model_prob > 0:
                kelly_raw = (model_prob * b - (1 - model_prob)) / b
                frac = min(kelly_raw * kelly_fraction, MAX_STAKE_PCT)
                stake = round(frac * bankroll, 2)
                if stake < MIN_STAKE:
                    frac, stake = 0.0, 0.0
            else:
                frac, stake = 0.0, 0.0

            recommendations.append({
                "player_name": p["player_name"],
                "dg_id": p["dg_id"],
                "market": "frl",
                "model_rank": rank,
                "model_frl_prob": round(model_prob, 4),
                "implied_prob": round(implied_prob, 4),
                "edge": round(edge, 1),
                "odds_american": p["best_odds_str"],
                "book": p["best_book"],
                "r1_sg_avg": round(p["r1_sg_avg"], 3),
                "grass_adj": round(p["grass_adj"], 3),
                "wave_adj": round(p.get("wave_adj", 0.0), 3),
                "r1_tendency": round(p["r1_tendency"], 3),
                "r1_form": round(p["r1_form"], 3),
                "vol_adj": round(p["vol_adj"], 3),
                "score": round(p["score"], 3),
                "variance": round(p["variance"], 2),
                "kelly_fraction": round(frac, 6),
                "recommended_stake": stake,
            })

        recommendations.sort(key=lambda x: x["edge"], reverse=True)
        return recommendations

    def backtest(self, min_training_events=20, n_sims=10_000, bankroll=DEFAULT_BANKROLL,
                 kelly_fraction=KELLY_FRACTION):
        """Walk-forward calibration backtest.

        For each event after warmup:
        - Score using only prior R1 data
        - Simulate FRL probabilities
        - Check if actual R1 leader was in model's top N
        - Track calibration metrics (Brier score, hit rates)

        Returns dict with calibration and performance metrics.
        """
        sorted_events = sorted(
            self.events.items(),
            key=lambda x: x[1].get("event_completed", "")
        )

        brier_scores = []
        top_5_hits = 0
        top_10_hits = 0
        top_20_hits = 0
        total_tested = 0
        log_loss_scores = []

        # Probability calibration bins
        cal_bins = defaultdict(lambda: {"predicted": [], "actual": []})

        for test_idx in range(min_training_events, len(sorted_events)):
            event_id, event = sorted_events[test_idx]
            event_date = event.get("event_completed", "")
            if not event_date:
                continue

            # Find actual R1 leaders
            leaders, n_tied = self._find_r1_leaders(event)
            if not leaders:
                continue

            # Get event grass types
            event_grass = self.event_grass.get(event_id, {})

            # Score all players with R1 data in this event
            scored = []
            for score in event.get("scores", []):
                dg_id = score.get("dg_id")
                if not dg_id:
                    continue
                fin_text = str(score.get("fin_text", "")).strip().upper()
                if fin_text in {"WD", "DQ"}:
                    continue
                rd1 = score.get("round_1")
                if not isinstance(rd1, dict) or rd1.get("sg_total") is None:
                    continue

                player_name = score.get("player_name", "")
                result = self.calculate_r1_score(dg_id, event_date, event_grass, player_name)
                if result is None:
                    continue

                scored.append({"dg_id": dg_id, **result})

            if len(scored) < 10:
                continue

            # Monte Carlo simulation
            frl_probs = self.simulate_frl_probabilities(scored, n_sims)

            # Evaluate
            total_tested += 1
            ranked = sorted(scored, key=lambda x: frl_probs.get(x["dg_id"], 0), reverse=True)
            ranked_ids = [p["dg_id"] for p in ranked]

            # Brier score
            for p in scored:
                prob = frl_probs.get(p["dg_id"], 0)
                actual = 1.0 if p["dg_id"] in leaders else 0.0
                brier_scores.append((prob - actual) ** 2)
                # Log loss
                prob_clipped = max(min(prob, 0.999), 0.001)
                log_loss_scores.append(-(actual * math.log(prob_clipped) + (1 - actual) * math.log(1 - prob_clipped)))
                # Calibration bins (10 bins)
                bin_idx = min(int(prob * 10), 9)
                cal_bins[bin_idx]["predicted"].append(prob)
                cal_bins[bin_idx]["actual"].append(actual)

            # Top-N hit rates
            for n, counter in [(5, "top_5"), (10, "top_10"), (20, "top_20")]:
                top_n_ids = set(ranked_ids[:n])
                if leaders & top_n_ids:
                    if counter == "top_5":
                        top_5_hits += 1
                    elif counter == "top_10":
                        top_10_hits += 1
                    else:
                        top_20_hits += 1

        if total_tested == 0:
            return {"error": "No events tested"}

        # Calibration analysis
        calibration = {}
        for bin_idx in sorted(cal_bins.keys()):
            b = cal_bins[bin_idx]
            if b["predicted"]:
                calibration[f"bin_{bin_idx}"] = {
                    "avg_predicted": round(statistics.mean(b["predicted"]), 4),
                    "avg_actual": round(statistics.mean(b["actual"]), 4),
                    "count": len(b["predicted"]),
                }

        return {
            "n_tournaments_tested": total_tested,
            "brier_score": round(statistics.mean(brier_scores), 6) if brier_scores else None,
            "log_loss": round(statistics.mean(log_loss_scores), 6) if log_loss_scores else None,
            "top_5_hit_rate": round(top_5_hits / total_tested, 3),
            "top_10_hit_rate": round(top_10_hits / total_tested, 3),
            "top_20_hit_rate": round(top_20_hits / total_tested, 3),
            "calibration": calibration,
        }


# ─────────────────────────────────────────────────
# Main / CLI
# ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FRL Production Betting Model")
    parser.add_argument("--backtest", action="store_true", help="Run calibration backtest")
    parser.add_argument("--bankroll", type=float, default=DEFAULT_BANKROLL, help="Bankroll for Kelly sizing")
    parser.add_argument("--min-edge", type=float, default=MIN_EDGE, help="Minimum edge %% to recommend")
    parser.add_argument("--sims", type=int, default=N_SIMULATIONS, help="Monte Carlo simulation count")
    args = parser.parse_args()

    model = FRLProductionModel()

    if args.backtest:
        print("\n" + "=" * 60)
        print("FRL CALIBRATION BACKTEST")
        print("=" * 60)
        bt = model.backtest(n_sims=min(args.sims, 10_000), bankroll=args.bankroll)
        if "error" in bt:
            print(f"Error: {bt['error']}")
            return

        print(f"\nTournaments tested: {bt['n_tournaments_tested']}")
        print(f"Brier Score:        {bt['brier_score']:.6f}")
        print(f"Log Loss:           {bt['log_loss']:.6f}")
        print(f"Top-5 Hit Rate:     {bt['top_5_hit_rate']:.1%}")
        print(f"Top-10 Hit Rate:    {bt['top_10_hit_rate']:.1%}")
        print(f"Top-20 Hit Rate:    {bt['top_20_hit_rate']:.1%}")

        # Calibration
        print("\nCalibration (predicted vs actual):")
        for bin_key in sorted(bt.get("calibration", {}).keys()):
            b = bt["calibration"][bin_key]
            print(f"  {bin_key}: predicted={b['avg_predicted']:.4f}  actual={b['avg_actual']:.4f}  n={b['count']}")

        return

    # Generate recommendations
    print("\n" + "=" * 60)
    print("FRL PRODUCTION MODEL — RECOMMENDATIONS")
    print("=" * 60)

    recs = model.generate_recommendations(bankroll=args.bankroll)

    if not recs:
        print("No FRL recommendations generated (check odds file and grass config)")
        return

    # Print recommendations
    print(f"\n{'Rank':>4}  {'Player':<25}  {'FRL Prob':>8}  {'Implied':>8}  {'Edge':>6}  {'Odds':>8}  {'Book':<11}  {'Stake':>7}")
    print("-" * 100)
    for r in recs:
        print(f"{r['model_rank']:>4}  {r['player_name']:<25}  {r['model_frl_prob']:>8.4f}  {r['implied_prob']:>8.4f}  "
              f"{r['edge']:>5.1f}%  {r['odds_american']:>8}  {r['book']:<11}  ${r['recommended_stake']:>6.2f}")

    total_stake = sum(r["recommended_stake"] for r in recs)
    avg_edge = statistics.mean(r["edge"] for r in recs) if recs else 0
    print(f"\nTotal: {len(recs)} picks, ${total_stake:.2f} stake, {avg_edge:.1f}% avg edge")

    # Run quick calibration for the output
    cal = model.backtest(n_sims=5000)

    # Write output
    output = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": "frl_mc_v1",
        "n_simulations": N_SIMULATIONS,
        "model_config": {
            "lookback": LOOKBACK,
            "min_hist": MIN_HIST,
            "max_players": MAX_PLAYERS,
            "grass_weight": GRASS_WEIGHT,
            "volatility_weight": VOLATILITY_WEIGHT,
            "r1_tendency_weight": R1_TENDENCY_WEIGHT,
            "r1_form_weight": R1_FORM_WEIGHT,
            "sg_t2g_weight": SG_T2G_WEIGHT,
            "sg_putt_weight": SG_PUTT_WEIGHT,
            "sg_app_weight": SG_APP_WEIGHT,
            "sg_ott_weight": SG_OTT_WEIGHT,
            "kelly_fraction": KELLY_FRACTION,
            "min_edge": MIN_EDGE,
            "bankroll": args.bankroll,
            "wave_enabled": WAVE_ENABLED,
        },
        "wave_forecast": get_forecast_summary() if (WAVE_ENABLED and get_forecast_summary) else {},
        "calibration": {
            "events_tested": cal.get("n_tournaments_tested", 0),
            "brier_score": cal.get("brier_score"),
            "log_loss": cal.get("log_loss"),
            "top_5_hit_rate": cal.get("top_5_hit_rate"),
            "top_10_hit_rate": cal.get("top_10_hit_rate"),
            "top_20_hit_rate": cal.get("top_20_hit_rate"),
        },
        "recommendations": recs,
    }

    # Snapshot previous FRL model output before overwriting
    try:
        from odds_snapshot import snapshot_before_overwrite
        snapshot_before_overwrite(OUTPUT_FILE)
    except Exception as e:
        print(f"  Warning: could not snapshot previous FRL model output: {e}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {len(recs)} recommendations to {OUTPUT_FILE}")

    # Track recommendation evolution across the week
    try:
        from model_recommendation_tracker import save_model_snapshot
        save_model_snapshot(output, model_type="frl")
    except Exception as e:
        print(f"  Warning: could not save FRL model snapshot: {e}")


if __name__ == "__main__":
    main()
