#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import zlib
from html.parser import HTMLParser
from pathlib import Path

SYSTEM_PROMPT = """You are an expert document parser specializing in semi-structured PDF extraction.

You will be given the full text of a GCSAA "Tournament Fact Sheet" PDF.
These documents follow a consistent semantic structure but may vary slightly
in formatting, ordering, spacing, or phrasing.

Your task is to extract structured data using meaning and context,
NOT exact formatting or position on the page.

IMPORTANT RULES:
- Ignore repeated boilerplate text (GCSAA mission statement, contact footer, links).
- Do not hallucinate missing fields.
- If a value is not present, return null.
- Multiple courses may exist per tournament; extract each separately.
- Preserve original wording for free-text "Notable" sections.

-----------------------------------
OUTPUT FORMAT (STRICT JSON)
-----------------------------------

{
  "tournament": {
    "name": string,
    "tour": string,
    "location": {
      "city": string,
      "state": string
    },
    "date_range": string
  },
  "courses": [
    {
      "course_name": string,
      "architect": {
        "name": string,
        "year": number | null
      },
      "renovation": string | null,
      "superintendent": {
        "name": string,
        "title": string,
        "gcsaa_membership_years": number | null,
        "years_at_course": number | null,
        "education": string | null,
        "previous_courses": string | null
      },
      "tournament_setup": {
        "par": string,
        "yardage": number,
        "stimpmeter": number | null
      },
      "turfgrass": {
        "greens": string | null,
        "collars": string | null,
        "approaches": string | null,
        "tees": string | null,
        "fairways": string | null,
        "rough": string | null
      },
      "course_statistics": {
        "average_green_size_sq_ft": number | null,
        "acres_fairway": number | null,
        "acres_rough": number | null,
        "number_of_bunkers": string | number | null,
        "number_of_water_hazards": number | null,
        "holes_water_in_play": number | null,
        "soil_conditions": string | null,
        "water_sources": string | null
      },
      "staffing": {
        "agronomy_employees": number | null,
        "tournament_volunteers": number | null,
        "key_personnel": string | null
      },
      "notable_notes": [string]
    }
  ]
}

-----------------------------------
EXTRACTION GUIDANCE
-----------------------------------

- Course names often appear as headers (e.g., "South Course", "Pebble Beach Golf Links").
- Superintendent info usually follows immediately after the course name.
- Turfgrass sections always list surface -> grass type -> mowing height.
- Course statistics may be split across lines; combine logically.
- "Notable" sections are bullet points; return them as an array of strings.
- Do NOT include comparative tables or summary rankings at the end of the document.

Return only valid JSON. No markdown, no code fences, no commentary.
"""

TARGET_PAGE = "https://www.gcsaa.org/who-we-are/media/tournament-fact-sheets"


def read_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Set OPENAI_API_KEY before running API-backed extraction.")
    return key


class PdfLinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "a":
            return
        href = None
        for key, value in attrs:
            if key.lower() == "href":
                href = value
                break
        if href and ".pdf" in href.lower():
            self.links.append(href)


def validate_source_url(url: str) -> str:
    """Allow only HTTPS downloads from the configured provider domain."""
    parsed = urllib.parse.urlsplit(url)
    host = (parsed.hostname or "").lower()
    if (parsed.scheme != "https" or parsed.username or parsed.password
            or parsed.port not in (None, 443)
            or not (host == "gcsaa.org" or host.endswith(".gcsaa.org"))):
        raise ValueError("Source URL must use HTTPS on the GCSAA provider domain")
    return url


class ProviderRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_source_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def open_source_url(url: str):
    req = urllib.request.Request(validate_source_url(url), headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.build_opener(ProviderRedirectHandler()).open(req, timeout=60)


def fetch_pdf_links_from_html(url: str) -> list[dict]:
    with open_source_url(url) as resp:
        html = resp.read().decode("utf-8", errors="replace")
    parser = PdfLinkParser()
    parser.feed(html)
    links = []
    for href in parser.links:
        links.append({"url": urllib.parse.urljoin(url, href)})
    return links


def safe_filename(url: str) -> str:
    split = urllib.parse.urlsplit(url)
    filename = os.path.basename(split.path)
    if not filename.lower().endswith(".pdf"):
        filename = f"{filename}.pdf"
    return filename


def download_pdf(url: str, dest_path: Path) -> None:
    with open_source_url(url) as resp:
        content = resp.read()
    dest_path.write_bytes(content)


def download_new_pdfs(output_dir: Path) -> tuple[int, int, int]:
    print("Checking GCSAA for new PDFs...")
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_links = fetch_pdf_links_from_html(TARGET_PAGE)
    print(f"Found {len(pdf_links)} PDF link(s) on {TARGET_PAGE}")
    if not pdf_links:
        return 0, 0, 0

    downloaded = 0
    skipped = 0
    failed = 0
    existing_files = {p.name for p in output_dir.glob("*.pdf")}
    for link in pdf_links:
        url = link.get("url", "")
        if not url:
            continue
        filename = safe_filename(url)
        dest_path = output_dir / filename
        if filename in existing_files:
            skipped += 1
            continue
        try:
            print(f"Downloading: {url}")
            download_pdf(url, dest_path)
            print(f"Saved: {dest_path.name}")
            downloaded += 1
        except Exception as exc:
            failed += 1
            print(f"Failed: {url} ({exc})", file=sys.stderr)

    print(
        f"Download complete. downloaded={downloaded} skipped={skipped} failed={failed}"
    )
    return downloaded, skipped, failed


def extract_pdf_text(pdf_path: Path) -> str:
    try:
        import pdfplumber  # type: ignore

        chunks = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                chunks.append(text)
        return "\n".join(chunks)
    except Exception:
        pass

    try:
        try:
            from pypdf import PdfReader  # type: ignore
        except Exception:
            from PyPDF2 import PdfReader  # type: ignore

        reader = PdfReader(str(pdf_path))
        chunks = []
        for page in reader.pages:
            text = page.extract_text() or ""
            chunks.append(text)
        return "\n".join(chunks)
    except Exception as exc:
        raise RuntimeError(
            "Failed to read PDF. Install pdfplumber or pypdf/PyPDF2."
        ) from exc


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_json_block(raw: str) -> str:
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response.")
    return raw[start : end + 1]


def generate_event_id(data: dict, fallback_seed: str) -> str:
    tournament = data.get("tournament") or {}
    name = str(tournament.get("name") or "")
    date_range = str(tournament.get("date_range") or "")
    seed = f"{name}|{date_range}|{fallback_seed}"
    letters = re.sub(r"[^A-Za-z]", "", name).upper()
    if len(letters) < 2:
        letters = (letters + "XX")[:2]
    prefix = letters[:2]
    num = zlib.adler32(seed.encode("utf-8")) % 100000
    return f"{prefix}-{num:05d}"


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


def process_pdf(pdf_path: Path, model: str, api_key: str, retries: int) -> dict:
    text = normalize_text(extract_pdf_text(pdf_path))
    if not text:
        raise RuntimeError("Empty text extracted from PDF.")
    raw = call_openai(text, model, api_key, retries)
    json_text = extract_json_block(raw)
    data = json.loads(json_text)
    if isinstance(data, dict):
        data["parsed_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        data["source_pdf"] = pdf_path.name
        if not data.get("event_id"):
            data["event_id"] = generate_event_id(data, pdf_path.name)
    return data


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract structured data from GCSAA Tournament Fact Sheet PDFs."
    )
    parser.add_argument(
        "--input-dir",
        default=str(
            Path(__file__).resolve().parent / "aggriculture_PDFs"
        ),
        help="Directory containing PDF files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "parsed_json"),
        help="Directory to write JSON outputs.",
    )
    parser.add_argument(
        "--aggregate-dir",
        default=str(Path(__file__).resolve().parent / "raw_data"),
        help="Directory to write aggregated JSON output.",
    )
    parser.add_argument(
        "--archive-dir",
        default=str(Path(__file__).resolve().parent / "raw_data" / "raw_data_archive"),
        help="Directory to store previous aggregate files.",
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model.")
    parser.add_argument("--retries", type=int, default=3, help="Retry count.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing JSON files.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_dir = Path(args.aggregate_dir).expanduser().resolve()
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    archive_dir = Path(args.archive_dir).expanduser().resolve()
    archive_dir.mkdir(parents=True, exist_ok=True)

    api_key = read_api_key()

    download_new_pdfs(input_dir)

    pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {input_dir}", file=sys.stderr)
        return 1

    total = len(pdfs)
    processed = 0
    skipped = 0
    failed = 0
    print(f"Found {total} PDF(s) in {input_dir}")

    # Identify PDFs that need processing (skip existing)
    to_process = []
    for idx, pdf_path in enumerate(pdfs, start=1):
        out_path = output_dir / f"{pdf_path.stem}.json"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            print(f"[{idx}/{total}] Skip existing {out_path.name}")
        else:
            to_process.append((idx, pdf_path, out_path))

    # Process PDFs in parallel (4 concurrent OpenAI API calls)
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _parse_one(item):
        idx, pdf_path, out_path = item
        print(f"[{idx}/{total}] Parsing {pdf_path.name}...")
        data = process_pdf(pdf_path, args.model, api_key, args.retries)
        out_path.write_text(json.dumps(data, indent=2))
        print(f"[{idx}/{total}] Wrote {out_path.name}")
        return pdf_path.name

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_parse_one, item): item for item in to_process}
        for future in as_completed(futures):
            item = futures[future]
            try:
                future.result()
                processed += 1
            except Exception as exc:
                failed += 1
                print(f"[{item[0]}/{total}] Failed {item[1].name}: {exc}", file=sys.stderr)

    print(
        f"Done. processed={processed} skipped={skipped} failed={failed} total={total}"
    )

    date_stamp = time.strftime("%m-%d-%Y", time.localtime())
    aggregate_path = aggregate_dir / f"raw_event_aggriculture_{date_stamp}.json"
    for existing in aggregate_dir.glob("raw_event_aggriculture_*.json"):
        if existing.name == aggregate_path.name:
            continue
        archive_target = archive_dir / existing.name
        existing.replace(archive_target)
    aggregated = []
    for json_path in sorted(output_dir.glob("*.json")):
        try:
            data = json.loads(json_path.read_text())
            if isinstance(data, dict) and not data.get("event_id"):
                data["event_id"] = generate_event_id(data, json_path.name)
                json_path.write_text(json.dumps(data, indent=2))
            aggregated.append(data)
        except Exception as exc:
            print(f"Failed to load {json_path.name}: {exc}", file=sys.stderr)
    aggregate_path.write_text(json.dumps(aggregated, indent=2))
    print(f"Wrote aggregate {aggregate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
