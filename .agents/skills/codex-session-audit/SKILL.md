---
name: codex-session-audit
description: Fetch and audit a local Codex session by ID. Use when asked why an agent made a decision, missed access, followed the wrong instruction, skipped verification, or when a Codex conversation needs an evidence-backed workflow audit.
---

# Codex session audit

Audit one requested local Codex session at a time. Treat the session log as sensitive local data: do not print, commit, upload, or quote credentials, tokens, cookies, or private user content. Summarize only the evidence needed to answer the request.

## Workflow

1. Confirm the exact `session_id`, audit question, and project root. Do not substitute a similarly named session. Treat a targeted question as one required focus inside the whole-session audit, never as permission to ignore the rest of the session.
2. Run the included extractor once from the project root to locate the exact live rollout:

   ```powershell
   <python-executable> .agents/skills/codex-session-audit/scripts/fetch_codex_session.py --session-id <session_id>
   ```

   On Windows, try `py` or `python` only when either resolves to an installed interpreter. Otherwise resolve the workspace-bundled Python.
3. Copy the exact rollout to a unique temporary file outside the repository, then run every summary/page command with both `--session-id <id>` and `--session-file <snapshot>`. Record the reported `snapshot_sha256`, last timestamp, and `page_count`. Never page through a live rollout: it may append during the audit.
4. Read **every snapshot transcript page**, in order, with `--page 1`, `--page 2`, and so on through `page_count`. Do not start findings before all pages are reviewed. The transcript covers the full human conversation plus bounded audit metadata for tool operations; encrypted reasoning and token-count noise are excluded.
5. Follow every local attachment or pasted-request file referenced by a user message and read it fully when accessible. For a `[TRUNCATED ...]` tool record that affects a finding, read the referenced raw event/file locally before deciding. Do not silently treat an excerpt as complete evidence.
6. Build a coverage ledger before analysis: snapshot hash/cutoff, pages read, referenced attachments read, user requests, agent decisions, tool failures, verification claims, instruction reads, and unresolved work. If any page or required attachment cannot be read, label the audit incomplete.
7. Inspect Codex-relevant project sources only when the transcript shows they were loaded or applicable: `AGENTS.md`, `.agents/workflows/`, `.agents/skills/`, and `.codex/`. Inspect KiloCode sources only with explicit cross-tool evidence.
8. Produce and return every section in [audit protocol](references/audit-protocol.md). The requested focus must appear in the report, but it must not replace whole-session findings. Do not replace the detailed report with an executive summary, a shortened handoff, or a statement that a report was produced.
9. Before the final response, run `python .agents/skills/codex-session-audit/scripts/validate_audit_report.py <report.md>` against the complete report, then run the completeness check against the coverage ledger: confirm protocol sections 1–5 are present, every material finding has all required evidence mappings, every corrective change satisfies the patch contract in [audit protocol](references/audit-protocol.md), and every unresolved item is labeled. If the complete redacted report will not fit in the response, write it in full to a unique temporary Markdown file outside the repository and link that artifact; a short summary is supplementary, never a substitute.

## Access diagnosis

For each missing capability, classify the first supported condition:

- **Tool unavailable** — the tool was absent from the available tool list or executable discovery.
- **Tool available but unconfigured** — present, but authentication, endpoint, runtime, or required configuration failed.
- **Credential exists but was not exposed** — evidence shows a credential store/config existed but the agent was not permitted or instructed to load the allowlisted value.
- **Instruction flow prevented correct use** — an applicable instruction mandated a conflicting sequence or prohibited the needed safe operation.
- **Incorrect assumption** — the agent claimed availability/unavailability without running the relevant check.

Never infer credential existence from an error alone. Never recommend placing credentials in source, prompts, or skill files.

## Recommendations

Prefer the smallest change that removes a repeated failure mode. Every corrective change must be patch-ready: name one exact repository-relative target file, give the current line or stable heading/symbol anchor, identify the operation as `INSERT`, `REPLACE`, or `DELETE`, quote the exact text/logic to insert, replace, or delete, and include the expected benefit and instruction-weight trade-off. A vague recommendation such as “clarify the workflow”, “add validation”, or “update the template” is incomplete until it specifies the concrete patch. Do not duplicate an existing rule across several files.
