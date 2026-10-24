"""Command-line interface for pinned source recovery and integrity checks."""

from __future__ import annotations

import argparse
import json
import sys

from .source_data import AuditError, audit_sources, fetch_pinned_sources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="download missing data from the pinned upstream revisions before auditing",
    )
    parser.add_argument("--write", help="write the sanitized audit JSON to this path")
    parser.add_argument(
        "--require-casa-gold",
        action="store_true",
        help="exit with status 2 unless CASA has complete, approved ASTE gold",
    )
    args = parser.parse_args()
    try:
        if args.fetch:
            fetch_pinned_sources()
        summary = audit_sources()
    except (AuditError, OSError) as error:
        parser.error(str(error))
    serialized = json.dumps(summary, indent=2, sort_keys=True)
    if args.write:
        from pathlib import Path

        path = Path(args.write)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    if args.require_casa_gold and not summary["casa"]["aste_gold_ready"]:
        print("CASA gold gate: blocked; human annotations and mapping are incomplete.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
