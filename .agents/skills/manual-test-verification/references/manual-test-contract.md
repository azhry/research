# Manual Test Contract

This reference defines the evidence required when a task or pull request asks for manual verification.

## Before execution

- Treat the issue or PR's manual steps as the contract. Preserve its command text and expected result unless the task explicitly changes them.
- Read `AGENTS.md`, the relevant `.agents/knowledge/` files, and any service-specific run instructions. Resolve the actual repository root, environment name, URL, and port from those sources or the running process.
- Use documented accounts, providers, roles, tiers, endpoints, and fixtures. Do not invent a record or substitute a local fake for a live acceptance boundary.
- If a command needs `.agents/.env`, load only the allowlisted key required for that command into the current process with a non-printing lookup. Never dot-source/source the file, load it wholesale, print the file, echo values, or place a secret in a command, report, commit, or handoff.

## Bash execution

Use Bash only for this contract. Keep setup separate from verification and make each numbered verification step one independently pasteable fenced Bash block containing one target request, normally one simple `curl`, plus its immediate status capture. Do not put health, login, fixture creation, the behavior under test, assertions, or cleanup into one block. Do not use `set -e`, `set -Eeuo pipefail`, `set -o pipefail`, a trap that exits early, a helper script, a full-flow script, a loop, a function, or a bulk runner that can hide which command failed. Never hand off an endpoint label such as `POST /api/apps` in place of the runnable command.

For each target command, capture its immediate status before any assertion, formatter, cleanup, or follow-up command:

```bash
command_under_test
target_status=$?
printf 'target command exit status: %s\n' "$target_status"
```

The printed status belongs to `command_under_test`. A later assertion or the wrapper's final exit status must not replace it. If a step intentionally expects a nonzero status, record that expected status and continue without converting it into a false pass.

## Classification

For every step, report all of these fields:

| Field | Required value |
| --- | --- |
| Command | The exact independently runnable Bash command or one-request block that ran; never an endpoint label or an all-in-one script |
| Immediate status | The target command's captured exit status |
| Sanitized response | Relevant output with credentials and sensitive identifiers removed |
| Expected outcome | The task/PR contract, including an expected nonzero status when applicable |
| Classification | `pass`, `expected result`, or `fail` |
| Environment/fixture | Documented environment and real fixture/account identity, without secrets |
| Limitation | Unrun live boundary, unavailable dependency, or `none` |

Use `pass` only when the observed status and response match the contract. Use `expected result` when the contract intentionally expects a nonzero status and that status occurred. Use `fail` for an unexpected status, response mismatch, missing fixture, or unavailable required boundary. A command that was not run is not a pass.

## Sanitization and evidence boundaries

Remove API keys, JWTs, cookies, passwords, Vault values, provider tokens, authorization headers, personal secrets, and opaque identifiers that would expose a protected record. Keep safe status codes, error classes, field names, counts, and non-sensitive response shape when they are enough to support the claim.

Unit tests, mocks, fakes, stub servers, isolated protocol checks, and local fixtures can support regression claims only. They cannot prove live API, authentication, persistence, Vault, or cross-service behavior. If the live API or authentication boundary was not run, state that exact limitation and do not report the live flow as passed.

## Failure handling

Preserve complete failure output in temporary or ignored storage when needed, but keep credentials out of it. Leave the terminal open after a failure so a human can inspect the state. Stop and report when a required dependency, credential, fixture, or documented environment is unavailable; do not silently downgrade a live check to a mock or change the port.

## Audit-derived readiness matrix

Apply the following matrix when the task includes the corresponding readiness artifact:

| Finding | Required safeguard | Evidence to record |
| --- | --- | --- |
| F-1 | Visual references use the requested UML sequence semantics, including lifelines, directional messages, and return/activation markers. | Re-read the saved rendering; attachment and text counts alone are insufficient. |
| F-2 | Human descriptions use the exact closed heading schema from the repository template. | Compare the saved heading sequence with the template; do not import agent-comment headings. |
| F-3 | Every target command has an immediate status captured before assertions or wrapper work. | Record the target status separately from the wrapper or assertion status. |
| F-4 | Manual verification includes Steps 0–4, separate Bash blocks, per-step statuses, no fake records/placeholders, and live-flow limitations. | Re-read the saved issue/PR and record each step's observed result. |
| F-5 | The active repository's local guardrails are checked directly. | Record the active repository path and local instruction source; sibling rules are not substitutes. |

## Report template

Use one compact record per step:

```text
Step N — <name>
Bash command: <exact command>
Immediate exit status: <integer>
Sanitized observed response: <relevant output>
Expected outcome: <contract>
Classification: <pass | expected result | fail>
Environment/fixture: <documented identity>
Limitation: <none or exact live-boundary limitation>
```
