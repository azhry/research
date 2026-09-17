#!/usr/bin/env python3
"""Discover KiloCode's schema and page through every row for one session."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

SECRET = re.compile(
    r"(?i)\b((?:[a-z0-9]+[_-])*(?:token|secret|password|api[_-]?key|authorization))\b"
    r"(\s*[=:]\s*|\s+)([^\s,;]+)"
)
SECRET_KEY = re.compile(r"(?i)(token|secret|password|api[_-]?key|authorization)")


class SessionLookupError(Exception):
    """Report a safe, deterministic session-selector failure."""


def quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def sanitize(value: Any) -> Any:
    if isinstance(value, str):
        return SECRET.sub(lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]", value)
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if SECRET_KEY.search(str(key)) else sanitize(item)
            for key, item in value.items()
        }
    return value


def bound(value: Any, limit: int = 4000) -> Any:
    if isinstance(value, str) and len(value) > limit:
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
        return f"{value[:limit]}...[TRUNCATED length={len(value)} sha256={digest}]"
    if isinstance(value, list):
        return [bound(item, limit) for item in value]
    if isinstance(value, dict):
        return {key: bound(item, limit) for key, item in value.items()}
    return value


def normalize_row(row: sqlite3.Row) -> dict[str, Any]:
    result = {
        key: "[REDACTED]" if SECRET_KEY.search(str(key)) else sanitize(row[key])
        for key in row.keys()
    }
    if isinstance(result.get("data"), str):
        try:
            result["data"] = sanitize(json.loads(result["data"]))
        except json.JSONDecodeError:
            pass
    return bound(result)


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_snapshot(source: Path, snapshot: Path) -> None:
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    source_connection = sqlite3.connect(f"file:{source.as_posix()}?mode=ro", uri=True)
    try:
        target_connection = sqlite3.connect(snapshot)
        try:
            source_connection.backup(target_connection)
        finally:
            target_connection.close()
    finally:
        source_connection.close()


def resolve_session_id(
    connection: sqlite3.Connection,
    session_id: str | None,
    session_title: str | None,
) -> str:
    if session_id is not None:
        return session_id

    table = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name = 'session'"
    ).fetchone()
    if table is None:
        raise SessionLookupError("session title lookup unsupported by schema")

    columns = {
        str(row[1]) for row in connection.execute(f"PRAGMA table_info({quote('session')})")
    }
    if not {"id", "title"}.issubset(columns):
        raise SessionLookupError("session title lookup unsupported by schema")

    matches = connection.execute(
        f"SELECT {quote('id')} FROM {quote('session')} "
        f"WHERE {quote('title')} COLLATE BINARY = ? LIMIT 2",
        (session_title,),
    ).fetchall()
    if not matches:
        raise SessionLookupError("session title not found")
    if len(matches) > 1:
        raise SessionLookupError("session title is ambiguous")
    if matches[0][0] is None:
        raise SessionLookupError("session title lookup returned no session id")
    return str(matches[0][0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--session-id")
    selector.add_argument("--session-title")
    parser.add_argument("--database", type=Path, default=Path(os.environ.get("USERPROFILE", "~")) / ".local" / "share" / "kilo" / "kilo.db")
    parser.add_argument("--snapshot-file", type=Path, help="Create once, then reuse this consistent SQLite snapshot")
    parser.add_argument("--page", type=int)
    parser.add_argument("--page-size", type=int, default=5)
    parser.add_argument("--chunk-chars", type=int, default=4000)
    args = parser.parse_args()
    if args.page_size < 1 or args.page_size > 200:
        parser.error("--page-size must be between 1 and 200")
    if args.chunk_chars < 500 or args.chunk_chars > 10000:
        parser.error("--chunk-chars must be between 500 and 10000")
    selection = (
        {"session_id": args.session_id}
        if args.session_id is not None
        else {"selector": "session-title"}
    )
    if not args.database.is_file():
        print(json.dumps({**selection, "database": str(args.database), "error": "database not found"}, indent=2))
        return 2

    database = args.database
    if args.snapshot_file is not None:
        if not args.snapshot_file.exists():
            create_snapshot(args.database, args.snapshot_file)
        database = args.snapshot_file
    snapshot_sha256 = sha256_file(database) if args.snapshot_file is not None else None

    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        session_id = resolve_session_id(connection, args.session_id, args.session_title)
    except SessionLookupError as error:
        connection.close()
        print(
            json.dumps(
                {**selection, "database": str(database), "error": str(error)},
                indent=2,
            )
        )
        return 2
    tables = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    records: list[dict[str, Any]] = []
    searched: list[str] = []
    for table in tables:
        columns = [dict(row) for row in connection.execute(f"PRAGMA table_info({quote(table)})")]
        names = [str(column["name"]) for column in columns]
        predicates: list[str] = []
        params: list[str] = []
        if "session_id" in names:
            predicates.append(f"{quote('session_id')} = ?")
            params.append(session_id)
        if table == "session" and "id" in names:
            predicates.append(f"{quote('id')} = ?")
            params.append(session_id)
        if not predicates:
            continue
        searched.append(table)
        order = next((name for name in ("seq", "time_created", "time_updated", "id") if name in names), "rowid")
        query = f"SELECT rowid, * FROM {quote(table)} WHERE {' OR '.join(predicates)} ORDER BY {quote(order) if order != 'rowid' else 'rowid'}"
        for row in connection.execute(query, params):
            records.append({"table": table, "row": normalize_row(row)})
    connection.close()
    records.sort(key=lambda record: (
        record["row"].get("time_created") or record["row"].get("time_updated") or 0,
        record["table"],
        record["row"].get("seq") or 0,
        record["row"].get("rowid") or 0,
    ))

    chunks = chunk_records(records, args.chunk_chars)
    page_count = (len(chunks) + args.page_size - 1) // args.page_size
    summary: dict[str, Any] = {
        "session_id": session_id,
        "database": str(database),
        "source_database": str(args.database) if args.snapshot_file is not None else None,
        "snapshot_sha256": snapshot_sha256,
        "immutable_snapshot": args.snapshot_file is not None,
        "searched_tables": searched,
        "record_count": len(records),
        "transcript_chunk_count": len(chunks),
        "page_size": args.page_size,
        "page_count": page_count,
        "first_timestamp": records[0]["row"].get("time_created") if records else None,
        "last_timestamp": records[-1]["row"].get("time_updated") if records else None,
    }
    if not records:
        summary["error"] = "session not found in schema-confirmed session columns"
        print(json.dumps(summary, indent=2))
        return 2
    if args.page is not None:
        if args.page < 1 or args.page > page_count:
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
