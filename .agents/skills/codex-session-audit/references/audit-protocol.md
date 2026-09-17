# Session audit protocol

Use this report structure for both session sources.

Before writing it, state coverage as `complete` only after every extractor page was read. A targeted question may shape emphasis but cannot narrow collection or omit unrelated major failures.

## Output completeness gate

Return the complete detailed audit to the user. Do not compress it into an executive summary, omit evidence mappings for brevity, or merely state that findings were produced.

For every confirmed material finding, include:

- A stable finding ID and concise title.
- Impact and the observed outcome.
- Session evidence: timestamp plus raw JSONL line/event identifier when available.
- Governing instruction evidence: exact repository-relative file, heading, line number, and the operative sentence. If no applicable instruction existed, say so explicitly.
- Judgment: followed, violated, incomplete, ambiguous, harmful, or not demonstrated to have influenced the session.
- Root cause and classification: reasoning, instruction/workflow, tooling, configuration, credential exposure, or external-state mutation.
- The smallest corrective change: a patch-ready change record following the corrective-change patch contract below. It must state the exact target file, current line number or stable heading/symbol anchor, one operation (`INSERT`, `REPLACE`, or `DELETE`), the exact text/logic to insert, replace, or delete, why it prevents recurrence, and its context/maintenance trade-off.

Keep distinct failures as distinct findings even when they share a root cause. Do not collapse them into a generic theme. Include recoveries only after recording the original failure and its evidence.

If the complete redacted report cannot fit in one response, write the entire report to a unique temporary Markdown file outside the repository, provide a clickable link, and state exactly which sections are in that artifact. Never silently shorten the report.

## 1. Session summary

- Requested question and source/session ID.
- What the user asked, what the agent attempted, and the first point where the outcome diverged.
- Evidence coverage and limitations (missing/truncated/inaccessible events).
- Coverage ledger: immutable snapshot SHA-256/cutoff, page range, session time range, referenced attachments, requests, decisions, failures, verification, and unresolved work.

## 2. Timeline of key events

List ordered, relevant events only: user instruction, instruction-file read, agent decision, tool discovery/call, result, and verification outcome. Cite event IDs/timestamps where available. Describe sensitive values rather than reproducing them.

## 3. Root-cause analysis

- Direct cause.
- Contributing factors.
- Classification: tooling, configuration, credential exposure, instruction/workflow, or reasoning.
- Confirmed findings vs. labelled hypotheses.
- Map each root-cause claim to the finding IDs and session evidence that establish it.

## 4. Instruction and workflow findings

For each relevant source, cite its path, heading/rule, line number, operative sentence, and evidence that it governed the audited session. State whether it was correct, harmful, incomplete, ambiguous, or not demonstrated to have influenced the session. Exclude source-specific rules for another agent platform unless an event or project-wide rule establishes cross-tool relevance.

### Corrective-change patch contract

Every confirmed material finding and every item in section 5 must include a concrete change record with this exact information:

```text
Target: <repository-relative file path>
Location: <current line or line range, plus stable heading/symbol anchor>
Operation: INSERT | REPLACE | DELETE
Current text/logic: <exact existing text/logic; write “none — new insertion after <anchor>” for INSERT>
Proposed text/logic: <exact complete text/logic to insert or replacement; write “delete the Current text/logic” for DELETE>
Why: <the observed failure this patch prevents>
Trade-off: <maintenance, scope, or instruction-precedence cost>
Verification: <exact check that proves the patch is present and effective>
```

Place one such record under each finding and under each section-5 recommendation. Run `.agents/skills/codex-session-audit/scripts/validate_audit_report.py <report.md>` before returning the report. The validator is a structural gate, not a substitute for checking that the proposed patch is technically correct.

Line numbers are required as they exist when the audit is written; the stable heading or symbol anchor is required so the location remains findable after nearby edits. For `REPLACE`, quote both the old and new text. For `INSERT`, state the exact insertion point and full inserted text. For `DELETE`, quote the exact text/logic being removed. If several files are needed, create one change record per file and explain why the smallest single-file change is insufficient. Do not use “clarify”, “improve”, “add validation”, “update the workflow”, or similar wording without the concrete patch above.

## 5. Actionable improvements

For every recommendation provide the complete corrective-change patch contract above, including exact target file, line/anchor, operation, current text/logic, proposed text/logic, why, trade-off, and verification. Tie it to one or more finding IDs. Avoid broad new rules when reordering or clarifying an existing rule is enough.
