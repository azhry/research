#!/usr/bin/env python3
"""Locate a Codex rollout and page through its complete semantic transcript."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

SECRET = re.compile(r"(?i)\b(token|secret|password|api[_-]?key|authorization)\b(\s*[=:]\s*|\s+)([^\s,;]+)")
SECRET_KEY = re.compile(r"(?i)(token|secret|password|api[_-]?key|authorization)")
KEEP_RESPONSE = {
    "message", "custom_tool_call", "custom_tool_call_output",
    "function_call", "function_call_output", "mcp_tool_call", "mcp_tool_call_output",
}
KEEP_EVENT = {
    "task_started", "task_complete", "user_message", "agent_message",
    "context_compacted",
}


def sanitize(value: Any) -> Any:
    if isinstance(value, str):
        return SECRET.sub(lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]", value)
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if SECRET_KEY.search(str(key)) else sanitize(item)
            for key, item in value.items()
            if key not in {"encrypted_content", "internal_chat_message_metadata_passthrough"}
        }
    return value


def bound_tool_data(value: Any, limit: int = 4000) -> Any:
    if isinstance(value, str) and len(value) > limit:
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
        return f"{value[:limit]}...[TRUNCATED length={len(value)} sha256={digest}]"
    if isinstance(value, list):
        return [bound_tool_data(item, limit) for item in value]
    if isinstance(value, dict):
        return {key: bound_tool_data(item, limit) for key, item in value.items()}
    return value


def semantic_record(event: dict[str, Any], line: int) -> dict[str, Any] | None:
    event_type = event.get("type")
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    item_type = payload.get("type")
    if event_type == "response_item" and item_type not in KEEP_RESPONSE:
        return None
    if event_type == "response_item" and item_type == "message" and payload.get("role") not in {"developer", "system"}:
        return None
    if event_type == "event_msg" and item_type not in KEEP_EVENT:
        return None
    if event_type not in {"response_item", "event_msg", "session_meta", "compacted", "turn_context"}:
        return None
    if event_type == "session_meta":
        payload = {
            key: payload.get(key)
            for key in ("session_id", "timestamp", "cwd", "originator", "cli_version", "source", "git")
            if key in payload
        }
    if event_type == "turn_context":
        payload = {
            key: payload.get(key)
            for key in ("turn_id", "cwd", "current_date", "timezone", "model")
            if key in payload
        }
    if event_type == "response_item":
        payload = bound_tool_data(payload, 1500 if item_type == "message" else 500)
    return {
        "line": line,
        "timestamp": event.get("timestamp"),
        "event_type": event_type,
        "payload": sanitize(payload),
    }


def chunk_records(records: list[dict[str, Any]], chunk_chars: int) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for record_index, record in enumerate(records):
        encoded = json.dumps(record, ensure_ascii=False)
        count = max(1, (len(encoded) + chunk_chars - 1) // chunk_chars)
        for chunk_index in range(count):
            start = chunk_index * chunk_chars
            chunks.append({
                "record_index": record_index,
                "chunk_index": chunk_index,
                "chunk_count": count,
                "content": encoded[start:start + chunk_chars],
            })
    return chunks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--sessions-root", type=Path, default=Path(os.environ.get("USERPROFILE", "~")) / ".codex" / "sessions")
    parser.add_argument("--session-file", type=Path, help="immutable rollout snapshot to audit instead of the live session store")
    parser.add_argument("--page", type=int)
    parser.add_argument("--page-size", type=int, default=40)
    parser.add_argument("--chunk-chars", type=int, default=4000)
    args = parser.parse_args()
    if args.page_size < 1 or args.page_size > 200:
        parser.error("--page-size must be between 1 and 200")
    if args.chunk_chars < 500 or args.chunk_chars > 10000:
        parser.error("--chunk-chars must be between 500 and 10000")

    matches = [args.session_file] if args.session_file else sorted(args.sessions_root.glob(f"**/rollout-*-{args.session_id}.jsonl"))
    matches = [path for path in matches if path and path.is_file()]
    if len(matches) != 1:
        print(json.dumps({"session_id": args.session_id, "match_count": len(matches), "error": "expected exactly one rollout"}, indent=2))
        return 2

    records: list[dict[str, Any]] = []
    malformed: list[int] = []
    raw_count = 0
    with matches[0].open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, 1):
            raw_count += 1
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                malformed.append(line_number)
                continue
            if isinstance(event, dict):
                record = semantic_record(event, line_number)
                if record:
                    records.append(record)

    observed_session_ids = sorted({
        record["payload"]["session_id"]
        for record in records
        if record["event_type"] == "session_meta"
        and isinstance(record["payload"].get("session_id"), str)
    })
    if observed_session_ids != [args.session_id]:
        print(json.dumps({
            "session_id": args.session_id,
            "path": str(matches[0]),
            "observed_session_ids": observed_session_ids,
            "error": "snapshot session metadata does not match the requested session",
        }, indent=2))
        return 2

    seen_messages: dict[str, int] = {}
    for record in records:
        payload = record["payload"]
        if record["event_type"] != "response_item" or payload.get("type") != "message":
            continue
        fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        if fingerprint in seen_messages:
            record["payload"] = {"type": "message_duplicate", "duplicate_of_line": seen_messages[fingerprint]}
        else:
            seen_messages[fingerprint] = record["line"]

    chunks = chunk_records(records, args.chunk_chars)
    page_count = (len(chunks) + args.page_size - 1) // args.page_size
    summary: dict[str, Any] = {
        "session_id": args.session_id,
        "path": str(matches[0]),
        "snapshot_sha256": hashlib.sha256(matches[0].read_bytes()).hexdigest(),
        "immutable_snapshot": args.session_file is not None,
        "raw_event_count": raw_count,
        "semantic_record_count": len(records),
        "transcript_chunk_count": len(chunks),
        "malformed_lines": malformed,
        "page_size": args.page_size,
        "page_count": page_count,
        "first_timestamp": records[0]["timestamp"] if records else None,
        "last_timestamp": records[-1]["timestamp"] if records else None,
        "record_type_counts": dict(Counter(record["event_type"] for record in records)),
    }
    if args.page is not None:
        if args.page < 1 or args.page > max(page_count, 1):
            print(json.dumps({**summary, "error": "page out of range"}, indent=2))
            return 2
        start = (args.page - 1) * args.page_size
        summary.update({
            "page": args.page,
            "chunks": chunks[start:start + args.page_size],
            "next_page": args.page + 1 if args.page < page_count else None,
        })
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
