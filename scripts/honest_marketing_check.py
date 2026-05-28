#!/usr/bin/env python3
"""Honest-marketing CI grep.

Scans README.md, bench/RESULTS.md (if present), and src/**/*.py
docstrings for banned phrases. Lines containing the comment marker
`# HONEST-CLAIM:` are exempt (so we can quote a banned phrase to refute
it). Numbers in `bench/RESULTS.md` must carry a `[MEASURED YYYY-MM-DD]`
tag at file scope at least once.

Exit code:
  0 — clean
  1 — at least one violation
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Banned phrases (case-insensitive). Keep this list small and curated;
# false positives waste reviewer time more than they help.
BANNED = [
    re.compile(r"world[\s\-]?first", re.IGNORECASE),
    re.compile(r"world[\s\-]?leading", re.IGNORECASE),
    re.compile(r"state[\s\-]?of[\s\-]?the[\s\-]?art", re.IGNORECASE),
    re.compile(r"\bSOTA\b"),
    re.compile(r"\bperfect\b", re.IGNORECASE),
    re.compile(r"\bguaranteed\b", re.IGNORECASE),
    re.compile(r"\bforever\b", re.IGNORECASE),
    re.compile(r"永続"),
    re.compile(r"完全自動"),
]

EXEMPT_MARKER = "# HONEST-CLAIM:"

# Files we always scan, even if README has a banned phrase quoted to
# refute it (those go on EXEMPT_MARKER lines).
TARGETS_ALWAYS: list[Path] = [
    ROOT / "README.md",
]
TARGETS_OPTIONAL: list[Path] = [
    ROOT / "bench" / "RESULTS.md",
]
SRC_GLOB = (ROOT / "src").rglob("*.py")


def scan_file(path: Path) -> list[str]:
    if not path.exists():
        return []
    violations: list[str] = []
    try:
        display = str(path.relative_to(ROOT))
    except ValueError:
        display = str(path)
    for ln, raw in enumerate(path.read_text().splitlines(), start=1):
        if EXEMPT_MARKER in raw:
            continue
        for rx in BANNED:
            m = rx.search(raw)
            if m:
                violations.append(f"{display}:{ln}: banned phrase {m.group(0)!r}")
    return violations


def check_measured_tag() -> list[str]:
    """RESULTS.md must contain at least one [MEASURED YYYY-MM-DD] tag."""
    results = ROOT / "bench" / "RESULTS.md"
    if not results.exists():
        # Pre-S6, RESULTS.md may not exist yet — only enforce when it does.
        return []
    text = results.read_text()
    if not re.search(r"\[MEASURED \d{4}-\d{2}-\d{2}\]", text):
        return [f"{results.relative_to(ROOT)}: missing [MEASURED YYYY-MM-DD] tag"]
    return []


def main() -> int:
    violations: list[str] = []
    for f in TARGETS_ALWAYS:
        violations.extend(scan_file(f))
    for f in TARGETS_OPTIONAL:
        violations.extend(scan_file(f))
    for f in SRC_GLOB:
        violations.extend(scan_file(f))
    violations.extend(check_measured_tag())

    if violations:
        print("honest-marketing check: FAIL", file=sys.stderr)
        for v in violations:
            print(f"  {v}", file=sys.stderr)
        return 1
    print("honest-marketing check: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
