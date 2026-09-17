#!/usr/bin/env python3
"""Validate the structural completeness of a Codex session audit report."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


SECTION_RE = re.compile(r"^## ([1-5])\.\s+", re.MULTILINE)
FINDING_RE = re.compile(r"^### (F-\d+)\b", re.MULTILINE)
OPERATION_RE = re.compile(r"(?:\*\*)?Operation(?:\*\*)?\s*:\s*(INSERT|REPLACE|DELETE)\b")
FIELD_NAMES = (
    "Target",
    "Location",
    "Operation",
    "Current text/logic",
    "Proposed text/logic",
    "Why",
    "Trade-off",
    "Verification",
)


def block_after_heading(text: str, start: int, next_starts: list[int]) -> str:
    end = min((value for value in next_starts if value > start), default=len(text))
    return text[start:end]


def validate(text: str) -> list[str]:
    errors: list[str] = []

    sections = {match.group(1) for match in SECTION_RE.finditer(text)}
    missing_sections = [str(number) for number in range(1, 6) if str(number) not in sections]
    if missing_sections:
        errors.append(f"missing protocol section(s): {', '.join(missing_sections)}")

    finding_matches = list(FINDING_RE.finditer(text))
    if not finding_matches:
        errors.append("no stable finding headings matching '### F-###' found")

    starts = [match.start() for match in finding_matches]
    for index, match in enumerate(finding_matches):
        finding_id = match.group(1)
        block = block_after_heading(text, match.start(), starts[index + 1 :])
        for field in FIELD_NAMES:
            if not re.search(rf"(?:\*\*)?{re.escape(field)}(?:\*\*)?\s*:", block):
                errors.append(f"{finding_id} missing corrective-change field: {field}")
        if not OPERATION_RE.search(block):
            errors.append(f"{finding_id} operation must be INSERT, REPLACE, or DELETE")

    actionable_match = re.search(r"^## 5\.\s+Actionable improvements\s*$", text, re.MULTILINE)
    if actionable_match:
        actionable = text[actionable_match.end() :]
        if "Target:" not in actionable or "Verification:" not in actionable:
            errors.append("section 5 must contain at least one complete corrective-change record")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Markdown audit report to validate, or '-' for stdin")
    args = parser.parse_args()

    if args.report == Path("-"):
        text = sys.stdin.read()
        source = "<stdin>"
    else:
        try:
            text = args.report.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"error: cannot read {args.report}: {exc}", file=sys.stderr)
            return 2
        source = str(args.report)

    errors = validate(text)
    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        return 1

    print(f"PASS: {source} contains protocol sections 1-5 and patch-ready corrective changes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
