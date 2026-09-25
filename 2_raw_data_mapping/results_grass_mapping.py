#!/usr/bin/env python3
"""Create a mapping CSV between aggriculture and historical golf events."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from hashlib import sha1
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


SYSTEM_PROMPT = (
    "You match golf events across two datasets. "
    "Use name similarity and year alignment. "
    "If no solid match, reply with nulls and low confidence."
)


@dataclass
class AggEvent:
    event_id: str
    name: str
    date_range: str
    year: Optional[int]


@dataclass
class DgEvent:
    event_id: str
    name: str
    event_completed: str
    year: Optional[int]


@dataclass
class MatchResult:
    agg_event: AggEvent
    dg_event: Optional[DgEvent]
    confidence: int
    method: str
    rationale: str


def load_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Set OPENAI_API_KEY before running API-backed extraction.")
    return key


def latest_file(directory: Path, pattern: str) -> Path:
    candidates = list(directory.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No files matching {pattern} in {directory}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def normalize_name(name: str) -> str:
    text = name.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_year(text: str) -> Optional[int]:
    match = re.search(r"(19|20)\d{2}", text)
    if match:
        return int(match.group(0))
    return None


def load_aggriculture(path: Path) -> List[AggEvent]:
    data = json.loads(path.read_text())
    events: List[AggEvent] = []
    if not isinstance(data, list):
        raise RuntimeError("Aggriculture JSON expected to be a list.")
    for row in data:
        tournament = row.get("tournament") or {}
        name = str(tournament.get("name") or "").strip()
        date_range = str(tournament.get("date_range") or "").strip()
        event_id = str(row.get("event_id") or "").strip()
        year = extract_year(date_range)
        if not name or not event_id:
            continue
        events.append(AggEvent(event_id=event_id, name=name, date_range=date_range, year=year))
    return events


def load_historical(path: Path) -> List[DgEvent]:
    data = json.loads(path.read_text())
    rounds_by_tour_year = data.get("rounds_by_tour_year")
    if not isinstance(rounds_by_tour_year, dict):
        raise RuntimeError("Historical JSON missing rounds_by_tour_year.")
    events: List[DgEvent] = []
    for _tour, years in rounds_by_tour_year.items():
        if not isinstance(years, dict):
            continue
        for year_key, event_map in years.items():
            if not isinstance(event_map, dict):
                continue
            for _event_key, event in event_map.items():
                if not isinstance(event, dict):
                    continue
                name = str(event.get("event_name") or "").strip()
                event_id = str(event.get("event_id") or "").strip()
                event_completed = str(event.get("event_completed") or "").strip()
                year = extract_year(event_completed) or extract_year(str(event.get("year") or ""))
                if not year:
                    year = extract_year(str(year_key))
                if not name or not event_id:
                    continue
                events.append(
                    DgEvent(
                        event_id=event_id,
                        name=name,
                        event_completed=event_completed,
                        year=year,
                    )
                )
    return events


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def generate_canonical_id(agg_event: AggEvent, dg_event: Optional[DgEvent]) -> str:
    seed = f"{agg_event.event_id}|{dg_event.event_id if dg_event else 'none'}|{agg_event.year}"
    digest = sha1(seed.encode("utf-8"), usedforsecurity=False).hexdigest()[:10]
    return f"CAN-{digest}"


def call_openai(prompt: str, model: str, api_key: str, retries: int) -> str:
    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:
        raise RuntimeError("Missing openai package. Install openai>=1.0.0.") from exc

    client = OpenAI(api_key=api_key)
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"API request failed after {retries} attempts; response details omitted.") from None


def extract_json_block(text: str) -> str:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in response.")
    return match.group(0)


def format_prompt(agg_event: AggEvent, candidates: List[DgEvent]) -> str:
    lines = [
        "Match this aggriculture event to the best historical golf event.",
        "Return JSON with keys: dg_event_id, dg_event_name, confidence (1-10), rationale.",
        "If no solid match, set dg_event_id and dg_event_name to null and confidence <= 6.",
        "",
        f"Agg event: name='{agg_event.name}', date_range='{agg_event.date_range}', year='{agg_event.year}'",
        "Candidates:",
    ]
    for idx, cand in enumerate(candidates, start=1):
        lines.append(
            f"{idx}. name='{cand.name}', event_id='{cand.event_id}', "
            f"event_completed='{cand.event_completed}', year='{cand.year}'"
        )
    return "\n".join(lines)


def match_with_llm(
    agg_event: AggEvent,
    candidates: List[DgEvent],
    model: str,
    api_key: str,
    retries: int,
) -> MatchResult:
    prompt = format_prompt(agg_event, candidates)
    raw = call_openai(prompt, model=model, api_key=api_key, retries=retries)
    data = json.loads(extract_json_block(raw))
    dg_event_id = data.get("dg_event_id")
    dg_event_name = data.get("dg_event_name")
    confidence = int(data.get("confidence") or 1)
    rationale = str(data.get("rationale") or "").strip()

    dg_event = None
    if dg_event_id and dg_event_name:
        for cand in candidates:
            if cand.event_id == dg_event_id:
                dg_event = cand
                break
        if dg_event is None:
            dg_event = DgEvent(
                event_id=str(dg_event_id),
                name=str(dg_event_name),
                event_completed="",
                year=None,
            )
    return MatchResult(
        agg_event=agg_event,
        dg_event=dg_event,
        confidence=confidence,
        method="openai",
        rationale=rationale,
    )


def build_candidate_index(events: Iterable[DgEvent]) -> dict:
    by_year = {}
    for event in events:
        by_year.setdefault(event.year, []).append(event)
    return by_year


def auto_match(
    agg_event: AggEvent,
    candidates: List[DgEvent],
    ratio_threshold: float,
    margin_threshold: float,
) -> Optional[MatchResult]:
    if not candidates:
        return None
    agg_norm = normalize_name(agg_event.name)
    scored: List[Tuple[float, DgEvent]] = []
    for cand in candidates:
        score = similarity(agg_norm, normalize_name(cand.name))
        scored.append((score, cand))
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_event = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0
    if agg_norm == normalize_name(best_event.name):
        return MatchResult(
            agg_event=agg_event,
            dg_event=best_event,
            confidence=9,
            method="exact",
            rationale="Exact normalized name match.",
        )
    if best_score >= ratio_threshold and (best_score - second_score) >= margin_threshold:
        confidence = max(7, int(round(best_score * 10)))
        return MatchResult(
            agg_event=agg_event,
            dg_event=best_event,
            confidence=confidence,
            method="fuzzy",
            rationale=f"Top similarity {best_score:.2f} with clear margin.",
        )
    return None


def write_results(path: Path, matches: List[MatchResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dg_event_name",
                "dg_event_id",
                "agg_event_name",
                "agg_event_id",
                "can_event_name",
                "can_event_id",
                "dg_event_completed",
            ],
        )
        writer.writeheader()
        for match in matches:
            if not match.dg_event:
                continue
            dg_event = match.dg_event
            can_event_name = dg_event.name
            completed_year = extract_year(dg_event.event_completed or "") or dg_event.year
            if completed_year is None:
                completed_year = match.agg_event.year
            can_event_id = (
                f"{dg_event.name}_{completed_year}" if completed_year is not None else dg_event.name
            )
            writer.writerow(
                {
                    "dg_event_name": dg_event.name,
                    "dg_event_id": dg_event.event_id,
                    "agg_event_name": match.agg_event.name,
                    "agg_event_id": match.agg_event.event_id,
                    "can_event_name": can_event_name,
                    "can_event_id": can_event_id,
                    "dg_event_completed": dg_event.event_completed,
                }
            )


def write_issues(path: Path, matches: List[MatchResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dg_event_name",
                "dg_event_id",
                "agg_event_name",
                "agg_event_id",
                "confidence",
                "match_method",
                "rationale",
                "can_event_name",
                "can_event_id",
            ],
        )
        writer.writeheader()
        for match in matches:
            if match.confidence > 6 and match.dg_event:
                continue
            can_event_name = match.agg_event.name
            can_event_id = generate_canonical_id(match.agg_event, match.dg_event)
            writer.writerow(
                {
                    "dg_event_name": match.dg_event.name if match.dg_event else "",
                    "dg_event_id": match.dg_event.event_id if match.dg_event else "",
                    "agg_event_name": match.agg_event.name,
                    "agg_event_id": match.agg_event.event_id,
                    "confidence": match.confidence,
                    "match_method": match.method,
                    "rationale": match.rationale,
                    "can_event_name": can_event_name,
                    "can_event_id": can_event_id,
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Map aggriculture events to historical golf events.")
    parser.add_argument(
        "--raw-dir",
        default=str(Path(__file__).resolve().parents[1] / "1_raw_data_extracts" / "raw_data"),
        help="Directory containing raw JSON files.",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "results_grass_mapping.csv"),
        help="Path to write mapping CSV.",
    )
    parser.add_argument(
        "--issues",
        default=str(Path(__file__).resolve().parent / "mapping_issues.csv"),
        help="Path to write mapping issues CSV.",
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model.")
    parser.add_argument("--retries", type=int, default=3, help="Retry count.")
    parser.add_argument("--ratio-threshold", type=float, default=0.92)
    parser.add_argument("--margin-threshold", type=float, default=0.03)
    parser.add_argument("--candidate-limit", type=int, default=12)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).expanduser().resolve()
    agg_path = latest_file(raw_dir, "raw_event_aggriculture_*.json")
    dg_path = latest_file(raw_dir, "raw_historical_data_golf_*.json")

    agg_events = load_aggriculture(agg_path)
    dg_events = load_historical(dg_path)
    dg_by_year = build_candidate_index(dg_events)

    api_key = load_api_key()

    matches: List[MatchResult] = []
    total = len(agg_events)
    llm_calls = 0
    start = time.time()
    print(f"Loaded {len(agg_events)} aggriculture events from {agg_path.name}.")
    print(f"Loaded {len(dg_events)} historical events from {dg_path.name}.")
    print("Starting mapping...")
    for agg_event in agg_events:
        idx = len(matches) + 1
        if idx % 10 == 1 or idx == total:
            elapsed = time.time() - start
            print(f"Progress {idx}/{total} | LLM calls: {llm_calls} | elapsed {elapsed:.1f}s")
        candidates = dg_by_year.get(agg_event.year, dg_events)
        auto = auto_match(
            agg_event,
            candidates,
            ratio_threshold=args.ratio_threshold,
            margin_threshold=args.margin_threshold,
        )
        if auto:
            if auto.method != "exact":
                print(
                    f"Auto-match ({auto.method}) '{agg_event.name}' -> '{auto.dg_event.name}' "
                    f"(conf {auto.confidence})"
                )
            matches.append(auto)
            continue
        scored = sorted(
            candidates,
            key=lambda cand: similarity(
                normalize_name(agg_event.name), normalize_name(cand.name)
            ),
            reverse=True,
        )
        shortlist = scored[: args.candidate_limit]
        if not shortlist:
            matches.append(
                MatchResult(
                    agg_event=agg_event,
                    dg_event=None,
                    confidence=1,
                    method="none",
                    rationale="No candidates available.",
                )
            )
            continue
        llm_calls += 1
        match = match_with_llm(
            agg_event=agg_event,
            candidates=shortlist,
            model=args.model,
            api_key=api_key,
            retries=args.retries,
        )
        print(
            f"LLM match '{agg_event.name}' -> "
            f"'{match.dg_event.name if match.dg_event else 'None'}' "
            f"(conf {match.confidence})"
        )
        matches.append(match)

    output_path = Path(args.output).expanduser().resolve()
    issues_path = Path(args.issues).expanduser().resolve()
    write_results(output_path, matches)
    write_issues(issues_path, matches)

    print(f"Wrote results to {output_path}")
    print(f"Wrote issues to {issues_path}")
    print(f"Total LLM calls: {llm_calls}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
