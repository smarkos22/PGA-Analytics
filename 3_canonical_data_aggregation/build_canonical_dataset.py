#!/usr/bin/env python3
import csv
import json
import re
import shutil
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = ROOT_DIR / "1_raw_data_extracts" / "raw_data"
RAW_ARCHIVE_DIR = RAW_DATA_DIR / "raw_data_archive"
MAPPING_PATH = ROOT_DIR / "2_raw_data_mapping" / "results_grass_mapping.csv"
OUTPUT_DIR = ROOT_DIR / "3_canonical_data_aggregation" / "canonical_data"
OUTPUT_ARCHIVE_DIR = OUTPUT_DIR / "canonical_data_archive"
OUTPUT_FILENAME = "canonical_historical_data.json"


def find_latest_file(patterns: Iterable[str], search_dirs: Iterable[Path]) -> Path:
    candidates: List[Path] = []
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for pattern in patterns:
            candidates.extend(search_dir.glob(pattern))
    if not candidates:
        joined_patterns = ", ".join(patterns)
        joined_dirs = ", ".join(str(d) for d in search_dirs)
        raise FileNotFoundError(f"No files found for patterns [{joined_patterns}] in [{joined_dirs}]")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def parse_iso_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return None


def build_historical_index(raw_hist: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    rounds_by_tour_year = raw_hist.get("rounds_by_tour_year", {})
    index: Dict[str, List[Dict[str, Any]]] = {}
    for tour, years in rounds_by_tour_year.items():
        for year, events in years.items():
            for event_id, event in events.items():
                entry = dict(event)
                entry["_tour"] = tour
                entry["_year_key"] = year
                index.setdefault(str(event_id), []).append(entry)
    return index


def select_historical_event(
    events: List[Dict[str, Any]],
    target_completed: Optional[str],
) -> Optional[Dict[str, Any]]:
    if not events:
        return None
    if target_completed:
        for event in events:
            if event.get("event_completed") == target_completed:
                return event
    dated_events: List[Tuple[datetime, Dict[str, Any]]] = []
    for event in events:
        event_date = parse_iso_date(event.get("event_completed"))
        if event_date:
            dated_events.append((event_date, event))
    if dated_events:
        return max(dated_events, key=lambda item: item[0])[1]
    return events[-1]


def build_aggriculture_index(raw_aggriculture: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for record in raw_aggriculture:
        event_id = str(record.get("event_id", "")).strip()
        if not event_id:
            continue
        existing = index.get(event_id)
        if not existing:
            index[event_id] = record
            continue
        existing_ts = existing.get("parsed_at_utc")
        record_ts = record.get("parsed_at_utc")
        if not existing_ts or (record_ts and record_ts > existing_ts):
            index[event_id] = record
    return index


def build_course_entries(courses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    course_entries: List[Dict[str, Any]] = []
    for course in courses:
        tournament_setup = course.get("tournament_setup", {})
        if not isinstance(tournament_setup, dict):
            tournament_setup = {}
        if "par" in tournament_setup:
            tournament_setup = dict(tournament_setup)
            tournament_setup["par"] = normalize_par_value(tournament_setup.get("par"))
        turfgrass = sanitize_turfgrass(course.get("turfgrass", {}))
        course_entries.append(
            {
                "course_name": course.get("course_name"),
                "tournament_setup": tournament_setup,
                "turfgrass": turfgrass,
                "course_statistics": course.get("course_statistics", {}),
            }
        )
    return course_entries


def normalize_par_value(par_value: Any) -> Dict[str, Optional[int]]:
    if isinstance(par_value, dict):
        return {
            "front": _coerce_int(par_value.get("front")),
            "back": _coerce_int(par_value.get("back")),
            "total": _coerce_int(par_value.get("total")),
        }
    if par_value is None:
        return {"front": None, "back": None, "total": None}
    if isinstance(par_value, (int, float)):
        return {"front": None, "back": None, "total": int(par_value)}
    text = str(par_value)
    numbers = [int(value) for value in re.findall(r"\d+", text)]
    if len(numbers) >= 3:
        return {"front": numbers[0], "back": numbers[1], "total": numbers[2]}
    if len(numbers) == 2:
        return {"front": numbers[0], "back": numbers[1], "total": numbers[0] + numbers[1]}
    if len(numbers) == 1:
        return {"front": None, "back": None, "total": numbers[0]}
    return {"front": None, "back": None, "total": None}


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def sanitize_turfgrass(turfgrass: Any) -> Dict[str, Any]:
    if not isinstance(turfgrass, dict):
        return {}
    cleaned: Dict[str, Any] = {}
    for key, value in turfgrass.items():
        if isinstance(value, str):
            cleaned_value = value.replace("\u201d", "").strip()
            cleaned[key] = parse_turfgrass_value(cleaned_value)
        else:
            cleaned[key] = value
    return cleaned


def parse_turfgrass_value(value: str) -> Dict[str, Any]:
    cut_length = extract_cut_length(value)
    type_list = extract_grass_types(value)
    if not type_list:
        type_list = ["other"]
    return {"type": type_list, "cut_length": cut_length}


def extract_cut_length(value: str) -> Optional[float]:
    matches = re.findall(r"\d*\.?\d+", value)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def extract_grass_types(value: str) -> List[str]:
    text = value.lower()
    found: List[str] = []
    if re.search(r"bent", text):
        found.append("bent")
    if re.search(r"bermuda|bermudagrass|tif", text):
        found.append("bermuda")
    if re.search(r"rye", text):
        found.append("ryegrass")
    if re.search(r"poa", text):
        found.append("poa annua")
    if re.search(r"paspalum", text):
        found.append("paspalum")
    if re.search(r"zoysia", text):
        found.append("zoysia")
    if re.search(r"fescue", text):
        found.append("fescue")
    return list(dict.fromkeys(found))


def build_odds_index(raw_odds: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Build index of historical odds keyed by '{dg_event_id}_{year}'."""
    index: Dict[str, Dict[str, Any]] = {}
    odds_data = raw_odds.get("odds_by_tour_year_market_book", {})

    for _tour, years in odds_data.items():
        for year_str, markets in years.items():
            for market, books in markets.items():
                for book, events in books.items():
                    if not isinstance(events, dict):
                        continue
                    for event_id, event_data in events.items():
                        if not isinstance(event_data, dict):
                            continue
                        key = f"{event_id}_{year_str}"
                        if key not in index:
                            index[key] = {
                                "available_markets": set(),
                                "available_books": set(),
                                "raw_odds": {},
                            }
                        index[key]["available_markets"].add(market)
                        index[key]["available_books"].add(book)
                        index[key]["raw_odds"].setdefault(market, {})[book] = event_data.get("odds", [])

    for key in index:
        index[key]["available_markets"] = sorted(index[key]["available_markets"])
        index[key]["available_books"] = sorted(index[key]["available_books"])

    return index


def build_historical_odds_entry(odds_data: Dict[str, Any]) -> Dict[str, Any]:
    """Build the canonical historical_odds structure from indexed raw odds."""
    entry: Dict[str, Any] = {
        "available_markets": odds_data["available_markets"],
        "available_books": odds_data["available_books"],
        "markets": {},
        "player_odds": {},
    }

    raw = odds_data["raw_odds"]

    for market, books in raw.items():
        entry["markets"][market] = {}
        for book, odds_list in books.items():
            if not isinstance(odds_list, list):
                continue

            entry["markets"][market][book] = {"players_count": len(odds_list)}

            for player in odds_list:
                dg_id = str(player.get("dg_id", ""))
                if not dg_id:
                    continue

                if dg_id not in entry["player_odds"]:
                    entry["player_odds"][dg_id] = {
                        "player_name": player.get("player_name", ""),
                        "outcome": player.get("outcome", ""),
                        "markets": {},
                    }

                po = entry["player_odds"][dg_id]
                po.setdefault("markets", {}).setdefault(market, {})[book] = {
                    "open_odds": player.get("open_odds"),
                    "close_odds": player.get("close_odds"),
                    "open_time": player.get("open_time"),
                    "close_time": player.get("close_time"),
                    "bet_outcome_numeric": player.get("bet_outcome_numeric"),
                    "bet_outcome_text": player.get("bet_outcome_text"),
                }

    return entry


def _parse_teetime(teetime_str: str) -> Optional[int]:
    """Parse tee time string to integer hour. '11:39am' -> 11, '2:15pm' -> 14."""
    if not teetime_str or not isinstance(teetime_str, str):
        return None
    t = teetime_str.strip().lower()
    try:
        if "am" in t:
            parts = t.replace("am", "").split(":")
            hour = int(parts[0])
            if hour == 12:
                hour = 0
            return hour
        elif "pm" in t:
            parts = t.replace("pm", "").split(":")
            hour = int(parts[0])
            if hour != 12:
                hour += 12
            return hour
    except (ValueError, IndexError):
        pass
    return None


def _compute_round_wind(hourly_data: Dict, round_date: str, teetime_hour: int,
                        duration: int = 5) -> Optional[Dict[str, Any]]:
    """Compute wind conditions during a player's round.

    Args:
        hourly_data: {date_str: {time: [...], wind_speed_10m: [...], ...}}
        round_date: YYYY-MM-DD string
        teetime_hour: integer hour (0-23)
        duration: hours the round lasts (default 5)

    Returns dict with avg_wind_speed, max_gusts, avg_wind_dir, hours_covered.
    """
    day_data = hourly_data.get(round_date, {})
    times = day_data.get("time", [])
    wind_speeds = day_data.get("wind_speed_10m", [])
    wind_gusts = day_data.get("wind_gusts_10m", [])
    wind_dirs = day_data.get("wind_direction_10m", [])

    if not times or not wind_speeds:
        return None

    # Build hour -> index mapping
    hour_index = {}
    for i, t in enumerate(times):
        # "2025-01-02T11:00" -> hour 11
        try:
            h = int(t[11:13])
            hour_index[h] = i
        except (ValueError, IndexError):
            continue

    # Collect wind data for the player's round hours
    end_hour = min(teetime_hour + duration, 23)
    hours_covered = []
    speeds = []
    gusts = []
    dirs = []
    for h in range(teetime_hour, end_hour + 1):
        if h in hour_index:
            idx = hour_index[h]
            ws = wind_speeds[idx] if idx < len(wind_speeds) else None
            wg = wind_gusts[idx] if idx < len(wind_gusts) else None
            wd = wind_dirs[idx] if idx < len(wind_dirs) else None
            if ws is not None:
                hours_covered.append(h)
                speeds.append(ws)
                if wg is not None:
                    gusts.append(wg)
                if wd is not None:
                    dirs.append(wd)

    if not speeds:
        return None

    return {
        "avg_wind_speed": round(sum(speeds) / len(speeds), 1),
        "max_gusts": round(max(gusts), 1) if gusts else None,
        "avg_wind_dir": round(sum(dirs) / len(dirs)) if dirs else None,
        "hours_covered": hours_covered,
    }


def _compute_round_temp_precip(hourly_data: Dict, round_date: str,
                                teetime_hour: int,
                                duration: int = 5) -> Optional[Dict[str, Any]]:
    """Compute temperature and precipitation during a player's round.

    Same tee-time matching logic as _compute_round_wind().

    Returns dict with avg_temp, min_temp, max_temp, total_precip_mm, hours_covered.
    """
    day_data = hourly_data.get(round_date, {})
    times = day_data.get("time", [])
    temps = day_data.get("temperature_2m", [])
    precips = day_data.get("precipitation", [])

    if not times or (not temps and not precips):
        return None

    # Build hour -> index mapping
    hour_index = {}
    for i, t in enumerate(times):
        try:
            h = int(t[11:13])
            hour_index[h] = i
        except (ValueError, IndexError):
            continue

    end_hour = min(teetime_hour + duration, 23)
    hours_covered = []
    temp_vals = []
    precip_vals = []
    for h in range(teetime_hour, end_hour + 1):
        if h in hour_index:
            idx = hour_index[h]
            tv = temps[idx] if idx < len(temps) else None
            pv = precips[idx] if idx < len(precips) else None
            hours_covered.append(h)
            if tv is not None:
                temp_vals.append(tv)
            if pv is not None:
                precip_vals.append(pv)

    if not temp_vals and not precip_vals:
        return None

    return {
        "avg_temp": round(sum(temp_vals) / len(temp_vals), 1) if temp_vals else None,
        "min_temp": round(min(temp_vals), 1) if temp_vals else None,
        "max_temp": round(max(temp_vals), 1) if temp_vals else None,
        "total_precip_mm": round(sum(precip_vals), 2) if precip_vals else None,
        "hours_covered": hours_covered,
    }


def _build_weather_entry(weather_data: Dict[str, Any],
                         scores: Optional[list]) -> Dict[str, Any]:
    """Build the canonical weather structure from raw weather data.

    Includes:
    1. Event-level round summaries (AM/PM/day wind averages)
    2. Per-player per-round wind_during_round (matched to tee times)
    """
    hourly = weather_data.get("hourly", {})
    round_dates = weather_data.get("round_dates", [])

    # Build event-level round summaries
    round_summaries = {}
    for i, rd in enumerate(round_dates):
        day_data = hourly.get(rd, {})
        wind_speeds = day_data.get("wind_speed_10m", [])
        wind_gusts = day_data.get("wind_gusts_10m", [])
        temps = day_data.get("temperature_2m", [])
        precips = day_data.get("precipitation", [])
        if not wind_speeds and not temps:
            continue

        # AM = hours 6-11, PM = hours 12-18, all playing = hours 6-18
        am_winds = [w for w in wind_speeds[6:12] if w is not None]
        pm_winds = [w for w in wind_speeds[12:19] if w is not None]
        all_playing = [w for w in wind_speeds[6:19] if w is not None]
        all_gusts = [g for g in wind_gusts[6:19] if g is not None] if wind_gusts else []

        playing_temps = [t for t in temps[6:19] if t is not None] if temps else []
        playing_precip = [p for p in precips[6:19] if p is not None] if precips else []

        round_summaries[f"round_{i+1}"] = {
            "date": rd,
            "am_wind_avg": round(sum(am_winds) / len(am_winds), 1) if am_winds else None,
            "pm_wind_avg": round(sum(pm_winds) / len(pm_winds), 1) if pm_winds else None,
            "day_wind_avg": round(sum(all_playing) / len(all_playing), 1) if all_playing else None,
            "max_gusts": round(max(all_gusts), 1) if all_gusts else None,
            "day_temp_avg": round(sum(playing_temps) / len(playing_temps), 1) if playing_temps else None,
            "min_temp": round(min(playing_temps), 1) if playing_temps else None,
            "max_temp": round(max(playing_temps), 1) if playing_temps else None,
            "total_precip_mm": round(sum(playing_precip), 2) if playing_precip else None,
        }

    entry = {
        "course_num": weather_data.get("course_num"),
        "latitude": weather_data.get("latitude"),
        "longitude": weather_data.get("longitude"),
        "timezone": weather_data.get("timezone"),
        "round_dates": round_dates,
        "round_summaries": round_summaries,
    }

    # Enrich per-player round data with weather conditions
    if scores:
        for player in scores:
            for rnd_idx, rnd_key in enumerate(["round_1", "round_2", "round_3", "round_4"]):
                rnd_data = player.get(rnd_key)
                if not rnd_data or not isinstance(rnd_data, dict):
                    continue
                teetime = rnd_data.get("teetime")
                teetime_hour = _parse_teetime(teetime)
                if teetime_hour is None:
                    continue
                if rnd_idx < len(round_dates):
                    wind_info = _compute_round_wind(hourly, round_dates[rnd_idx], teetime_hour)
                    if wind_info:
                        rnd_data["wind_during_round"] = wind_info
                    tp_info = _compute_round_temp_precip(hourly, round_dates[rnd_idx], teetime_hour)
                    if tp_info:
                        rnd_data["temp_during_round"] = {
                            "avg_temp": tp_info["avg_temp"],
                            "min_temp": tp_info["min_temp"],
                            "max_temp": tp_info["max_temp"],
                        }
                        rnd_data["precip_during_round"] = {
                            "total_precip_mm": tp_info["total_precip_mm"],
                        }

    return entry


def archive_existing_output(output_path: Path) -> None:
    if not output_path.exists():
        return
    OUTPUT_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archived_name = f"{output_path.stem}_{timestamp}{output_path.suffix}"
    shutil.move(str(output_path), str(OUTPUT_ARCHIVE_DIR / archived_name))


def main() -> None:
    hist_path = find_latest_file(
        ["raw_historical_data_golf_*.json", "raw_historical_data_golf_*.JSON"],
        [RAW_DATA_DIR, RAW_ARCHIVE_DIR],
    )
    agg_path = find_latest_file(
        ["raw_event_aggriculture_*.json", "raw_event_aggriculture_*.JSON"],
        [RAW_DATA_DIR, RAW_ARCHIVE_DIR],
    )

    raw_hist = load_json(hist_path)
    raw_aggriculture = load_json(agg_path)

    hist_index = build_historical_index(raw_hist)
    agg_index = build_aggriculture_index(raw_aggriculture)

    odds_path: Optional[Path] = None
    odds_index: Dict[str, Dict[str, Any]] = {}
    try:
        odds_path = find_latest_file(
            ["raw_historical_odds_*.json", "raw_historical_odds_*.JSON"],
            [RAW_DATA_DIR, RAW_ARCHIVE_DIR],
        )
        raw_odds = load_json(odds_path)
        odds_index = build_odds_index(raw_odds)
        print(f"Loaded historical odds from: {odds_path} ({len(odds_index)} event/year combos)")
    except FileNotFoundError:
        print("No historical odds data found, skipping.")

    weather_path: Optional[Path] = None
    weather_index: Dict[str, Dict[str, Any]] = {}
    try:
        weather_path = find_latest_file(
            ["raw_historical_weather_*.json", "raw_historical_weather_*.JSON"],
            [RAW_DATA_DIR, RAW_ARCHIVE_DIR],
        )
        raw_weather = load_json(weather_path)
        weather_index = raw_weather.get("events", {})
        print(f"Loaded weather data from: {weather_path} ({len(weather_index)} events)")
    except FileNotFoundError:
        print("No weather data found, skipping.")

    events_by_id: Dict[str, Dict[str, Any]] = {}
    missing_hist: List[str] = []
    missing_agg: List[str] = []

    with MAPPING_PATH.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            can_event_id = row.get("can_event_id")
            can_event_name = row.get("can_event_name")
            dg_event_id = row.get("dg_event_id")
            agg_event_id = row.get("agg_event_id")
            dg_event_completed = row.get("dg_event_completed")

            hist_events = hist_index.get(str(dg_event_id)) if dg_event_id else None
            hist_event = select_historical_event(hist_events or [], dg_event_completed)
            if not hist_event:
                missing_hist.append(str(dg_event_id))

            agg_event = agg_index.get(str(agg_event_id)) if agg_event_id else None
            if not agg_event:
                missing_agg.append(str(agg_event_id))

            courses = agg_event.get("courses", []) if agg_event else []
            tournament = agg_event.get("tournament", {}) if agg_event else {}

            event_entry = {
                "can_event_id": can_event_id,
                "can_event_name": can_event_name,
                "event_completed": (hist_event or {}).get("event_completed", dg_event_completed),
                "year": (hist_event or {}).get("year"),
                "sg_categories": (hist_event or {}).get("sg_categories"),
                "scores": (hist_event or {}).get("scores"),
                "source_event_ids": {
                    "dg_event_id": dg_event_id,
                    "agg_event_id": agg_event_id,
                },
                "historical_event_meta": {
                    "event_id": (hist_event or {}).get("event_id"),
                    "event_name": (hist_event or {}).get("event_name"),
                    "tour": (hist_event or {}).get("_tour"),
                    "season": (hist_event or {}).get("season"),
                },
                "aggriculture": {
                    "event_id": agg_event.get("event_id") if agg_event else None,
                    "tournament": {
                        "name": tournament.get("name"),
                        "tour": tournament.get("tour"),
                        "date_range": tournament.get("date_range"),
                    },
                    "location": tournament.get("location", {}),
                    "courses": build_course_entries(courses),
                    "source_pdf": agg_event.get("source_pdf") if agg_event else None,
                    "parsed_at_utc": agg_event.get("parsed_at_utc") if agg_event else None,
                },
            }

            odds_key = f"{dg_event_id}_{event_entry.get('year', '')}"
            odds_data = odds_index.get(odds_key)
            event_entry["historical_odds"] = build_historical_odds_entry(odds_data) if odds_data else None

            weather_data = weather_index.get(can_event_id) if can_event_id else None
            event_entry["weather"] = _build_weather_entry(weather_data, event_entry.get("scores")) if weather_data else None

            if can_event_id:
                events_by_id[can_event_id] = event_entry

    now_utc = datetime.now(timezone.utc)
    run_timestamp = now_utc.strftime("%Y%m%d_%H%M%S")
    generated_at_utc = now_utc.isoformat()
    output_payload = {
        "generated_at_utc": generated_at_utc,
        "source_files": {
            "mapping": str(MAPPING_PATH),
            "historical_data": str(hist_path),
            "event_aggriculture": str(agg_path),
            "historical_odds": str(odds_path) if odds_path else None,
            "historical_weather": str(weather_path) if weather_path else None,
        },
        "events_by_id": events_by_id,
        "missing_sources": {
            "historical_event_ids": sorted(set(missing_hist)),
            "aggriculture_event_ids": sorted(set(missing_agg)),
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_filename = f"canonical_historical_data_{run_timestamp}.json"
    output_path = OUTPUT_DIR / output_filename
    for existing in OUTPUT_DIR.glob("canonical_historical_data*.json"):
        archive_existing_output(existing)
    with output_path.open("w") as f:
        json.dump(output_payload, f, indent=2)

    print(f"Wrote {output_path}")
    if missing_hist or missing_agg:
        print("Missing sources detected.")
        if missing_hist:
            print(f"  historical_event_ids: {sorted(set(missing_hist))}")
        if missing_agg:
            print(f"  aggriculture_event_ids: {sorted(set(missing_agg))}")


if __name__ == "__main__":
    main()
