"""Check the reviewed publication boundary without printing matching content."""
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def check():
    allowed = set((ROOT / "PUBLICATION_FILES.txt").read_text().splitlines())
    tracked = set(subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT).decode().strip("\0").split("\0"))
    errors = [f"Unreviewed file: {p}" for p in sorted(tracked - allowed)]
    errors += [f"Missing reviewed file: {p}" for p in sorted(allowed - tracked)]
    checks = {
        "personal email": r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}",
        "machine-specific home": r"/(?:Users|home)/[A-Za-z0-9_.-]+/",
        "private network address": r"\b(?:10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+|100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d+\.\d+)\b",
        "credential-shaped literal": r"(?:gh[pousr]_[A-Za-z0-9]{20,}|sk-(?:proj-)?[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----)",
    }
    for name in sorted(tracked):
        path = ROOT / name
        if path.is_symlink():
            errors.append(f"Symlink is outside publication policy: {name}")
            continue
        raw = path.read_bytes()
        if len(raw) > 250_000 or b"\0" in raw:
            errors.append(f"Binary or oversized file: {name}")
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            errors.append(f"Non-text file: {name}")
            continue
        for label, pattern in checks.items():
            if re.search(pattern, text):
                errors.append(f"Review {label} in {name}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"Publication boundary passed: {len(tracked)} reviewed text files.")
    return 0

if __name__ == "__main__":
    raise SystemExit(check())
