---
name: kilocode-session-audit
description: Fetch and audit a local KiloCode session by ID or exact title from its SQLite store. Use when investigating KiloCode tool failures, instruction conflicts, skipped verification, or workflow improvements from an evidence-backed session timeline.
---

# KiloCode session audit

KiloCode session data is local and potentially sensitive. Inspect it read-only. Do not export raw conversations, credentials, tokens, cookies, or private user content into Git, Linear, or chat.

## Workflow

1. Confirm the exact session ID or visible title, audit question, and project root. Prefer the ID when known. Title lookup is exact and must resolve to one session; duplicate titles are reported as ambiguous. Keep targeted questions inside a whole-session audit.
2. Choose a unique temporary `*.db` path outside the repository. Discover the actual source schema and create a consistent SQLite backup by running:

   ```powershell
   <python-executable> .agents/skills/kilocode-session-audit/scripts/fetch_kilocode_session.py --session-id <session_id> --snapshot-file <temporary-snapshot.db>
   ```

   When only the visible title is known, run:

   ```powershell
   <python-executable> .agents/skills/kilocode-session-audit/scripts/fetch_kilocode_session.py --session-title "<exact session title>" --snapshot-file <temporary-snapshot.db>
   ```

   Pass exactly one selector. On Windows, try `py` or `python` only when either resolves to an installed interpreter. Otherwise resolve the workspace-bundled Python. The extractor opens `%USERPROFILE%/.local/share/kilo/kilo.db` read-only, uses SQLite's backup API when the snapshot does not exist, discovers its schema, and reports the resolved session ID, snapshot hash, and `page_count` for the session's redacted rows.
3. If the database is absent, unreadable, has no matching row, or has an unfamiliar schema, report that condition. Do not invent KiloCode table names or fallback SQL.
4. Reuse the same `--snapshot-file` for every command. Read every page in order with `--page 1` through `page_count`; do not form findings from a subset. Build the same coverage ledger required by the Codex skill and mark inaccessible pages as an incomplete audit.
5. Audit KiloCode-relevant sources only when loaded or applicable: `.kilo/rules/`, `kilo.json`, and genuinely project-wide rules. Inspect Codex sources only with explicit cross-tool evidence.
6. Produce every section in the shared [audit protocol](../codex-session-audit/references/audit-protocol.md).

## Evidence and recommendations

Use the same five missing-access classifications and minimal-change recommendation rules as the Codex session audit skill. Do not treat an SQLite access error as evidence that a credential, tool, or instruction was missing.
