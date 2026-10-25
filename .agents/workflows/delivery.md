# Delivery workflow

Apply this workflow to Linear, GitHub, issue, pull-request, and release work.

## Linear

- Treat an issue identifier (for example, `AZH-385`) as a Linear task. Use the connected Linear app to read it; if the tool is not immediately visible, discover available tools before selecting any fallback.
- Do not use direct HTTP, `curl`, guessed GraphQL queries, or credentials from project configuration for Linear while the connected app is available. If the connector is genuinely unavailable after discovery, immediately use the documented API fallback with the official schema or documentation; load only the required credential without output, and report authorization failures rather than bypassing them.
- Read the named issue before changing it; verify its team, project, state, relations, and constraints.
- Read its comments and the team's valid statuses before mutation. Read related issues and project context when they affect scope or sequencing.
- Before implementation, assess whether the issue has a sufficient human-readable description. When it is empty or ambiguous, update it first using the [issue-description template](../templates/linear-issue-description.md), whose top-level structure is limited to TL;DR, Process Flow, Before-After, and Implementation Manual Test and Verification.
- Preserve supplied reference material verbatim in the appropriate human-description section, including full HTML, mockups, screenshots, and example payloads. Do not overwrite, shorten, or substitute it with a summary.
- After saving the human description, inspect the relevant code, tests, and related work. Then complete the [agent-comment template](../templates/linear-issue-comment.md) and post it as an agent-facing Linear comment. The comment must identify the task category, confirmed problem analysis and scope boundaries, implementation plan, Definition of Done, correctness checks, and execution controls. Clearly label hypotheses and open questions; do not present them as confirmed defects.
- A design artifact alone (including a complete HTML file, screenshot, mockup, or prototype) is not sufficient. It describes the reference, not the task contract. The detailed agent comment must add the required analysis and acceptance contract before implementation.
- **Readiness gate:** verify both tracker mutations succeeded. Re-read the saved human description and agent comment, confirm the description follows the four-section human template, and confirm the comment contains Category, Problem analysis, Scope boundary, Implementation plan, Definition of Done, and Correctness checks. Do not create an implementation branch, change issue state, delegate implementation, or edit source files before this gate passes. Private reasoning, a todo list, or a final response is not a substitute for the tracker updates.
- Treat the human description and detailed agent comment together as the implementation contract. Do not begin implementation until they give a future agent enough information to understand the intended behavior, out-of-scope work, and how success will be verified.
- Move work to the team's active state when implementation begins. Attach the PR and move it to the available review state when handoff starts.
- Mark work complete only when the PR is merged or the user/team workflow confirms completion. For an intentional red-test issue, use the paired delivery workflow below to distinguish its designated expected-red GitHub check from unrelated checks that must remain green.
- Use exact IDs for mutations and report changed fields plus blockers.
- When an API fallback is permitted, obtain the needed credential from a non-printing secret store at runtime. Never use a generic file reader that serializes config contents, and never place a literal token in a command, source file, temporary file, transcript, URL, or issue comment. If a safe secret-loading path is unavailable, record the blocker and stop.

### Human reviewability and PR sequencing

- Treat human attention as a finite review budget. A PR is complete only when it carries one coherent behavior or one independently verifiable delivery unit that a human can understand, test, and manually verify in one focused review.
- Split the work before implementation when the issue contains multiple independent outcomes, crosses unrelated product areas, combines separate migration/behavior or infrastructure/application concerns, or cannot be explained and verified as one focused unit. Do not enforce an arbitrary line-count threshold; reviewability, dependency order, and manual verification effort are the hard boundary.
- Record the PR shape before editing: focused scope, base branch, review position, dependency chain, merge condition, parallel group, and manual verification boundary for every planned PR.
- Use a stack when a later unit depends on an earlier unit. Later branches must name their predecessor as the base, and PRs must be reviewed and merged from the bottom of the stack upward.
- Use parallel PRs only when units have no required dependency or conflicting shared change. State the parallel group and that its members are independently reviewable and mergeable.
- Every PR body must contain the `Review and merge order` section from the applicable PR template. It is the handoff contract for reviewer focus, stack order, dependencies, parallelism, merge conditions, and manual verification.
- Manual request/response is an executable artifact, not a prose report: every numbered manual step must contain a fenced `bash` block with the actual request command and an expected response. Never replace those command blocks with narrative saying that a request was sent. If live execution is unavailable, keep the runnable command using documented configuration or dynamic fixture creation and put only the exact blocker in Known limitations; never present an unrun response as observed.
- Do not require a reviewer to supply bare `TOKEN`, `SESSION_ID`, `API_BASE_URL`, or similar values. Every variable used by a manual block must be initialized in that sequence from documented project configuration, a real authentication exchange, or a dynamically created persisted fixture; secrets remain process-local and never enter the PR or tracker.
- Manual verification must be safe to paste into an interactive shell: do not use `set -e`, `set -u`, `set -o pipefail`, `set -euo pipefail`, `exit`, `exit 1`, or cleanup traps that call `exit`. Use explicit checks that print failures and leave the terminal open.
- Keep every manual step short and independently pasteable for a human: put the command in the issue or PR description, use at most one simple request (normally one `curl`) and one concise expected response per numbered step, keep setup/auth separate, and avoid helper functions, loops, generated fixtures, and bulk scripts. Never create or attach a verification script as the manual handoff; if the live flow was not run, record the exact limitation instead of inventing an observed response.

### Paired red-test delivery

Use this workflow only when the tracker explicitly defines separate test and implementation issues and requires the test issue to capture currently missing behavior before production code changes.

1. Keep the issues separate. The test issue owns only the test contract, fixtures/harness changes, and reproducible red evidence. The paired implementation issue owns the production behavior that makes that test green.
2. Start the test branch from `main`. Commit the test-only change and open its own PR against `main`. The targeted test must compile, complete setup, reach its intended contract assertion, and fail for that reason; setup, migration, container, authentication, or unrelated failures are blockers.
3. Identify the exact GitHub check that is expected to be red for this issue. Do not skip, quarantine, weaken, or convert the assertion into a false pass. Build, lint, unit, infrastructure/setup, and unaffected integration checks must remain green.
4. Do not mark the test issue or PR blocked because the designated check fails at the explicitly expected assertion. That failed check is the required evidence. Failures in compilation, environment setup, migrations, fixtures, authentication, cleanup, or unrelated tests remain blockers.
5. Record the exact test commit SHA, branch, PR, command, nonzero exit, intended assertion failure, successful setup stages, designated expected-red GitHub check, and unrelated green checks in both the test issue and paired implementation issue.
6. When the issue contract explicitly authorizes an intentional-red merge and the designated GitHub check fails only for the recorded assertion while unrelated checks remain green, merge the test-only PR into `main`. The test issue then completes through its own merged PR.
7. After that merge, start the paired implementation issue from the updated `main`. Preserve the test's contract and assertion in substance; do not skip, quarantine, delete, or weaken it to obtain green.
8. The implementation issue owns a separate production PR to `main`. It must make the previously red command exit 0 and pass every required check before merge.

The expected branch topology is:

```text
main ── red-test PR merge (documented test command remains intentionally red)
     └── implementation PR merge (same test becomes green)
```

## Git and GitHub

- Inspect `git status --short --branch`, the current branch, remote, and project instructions before editing or staging.
- For new work, branch from `main`. For work already in a PR, continue from that branch and update the same PR.
- Complete the GitHub authentication/repository-access preflight and create the required branch before the first implementation write. A later branch creation does not repair edits made on `main`. If the worktree is dirty, preserve it; use an isolated worktree when the task can proceed safely, otherwise report the blocker.
- Stage only intended files. Run proportionate verification and distinguish pre-existing failures from regressions.
- Before staging and before handoff, inspect `git status --short` and the intended diff. Keep generated diagnostics (screenshots, browser traces, lint logs, downloaded samples) outside the repository or in ignored temporary storage. Clean up only artifacts created by the current task.
- A verification claim requires a successful recorded exit status for the exact command or browser flow. A command that reaches partial compilation but exits non-zero is a failure. Report baseline failures separately with their command and affected path; never describe them as a passing build, lint, test, or check.
- Run build, typecheck, lint, and test commands unfiltered before using filters for diagnostics. A pipe to `findstr`, `Select-String`, or a similar filter cannot establish that the original command passed.
- Build the PR body from the applicable repository template: [backend](../templates/github-pr-description-backend.md), [frontend](../templates/github-pr-description-frontend.md), or [tests](../templates/github-pr-description-tests.md). Use the tests template for test-only work, including intentional red-test handoffs. For a cross-cutting PR, start with the template matching the primary implementation and include every applicable section from the others.
- Before PR creation or tracker transition, validate each PR body against the applicable template: include the direct PR URL in `Review and merge order`, link every predecessor and successor in the stated order, keep automated checks separate from manual acceptance evidence, use Bash for manual command blocks, do not use `set -o pipefail`, `set -e`, `set -Eeuo pipefail`, or another fail-fast wrapper in interactive verification, and remove every placeholder, fake record, mock endpoint, and unrun flow presented as a pass.
- Replace every bracketed placeholder and remove a conditional section only when it genuinely does not apply. Re-read the saved PR body and verify that it records scope, observable outcomes, exact commands and exit statuses, limitations, and the linked work item.
- For API, authentication, persistence, migration, or infrastructure changes, do not declare the live boundary blocked until the documented project knowledge and configuration sources have been checked, including any available read-only infrastructure path. Record whether the blocker is dependency access, application deployment, or an unverified fixture.
- Push and create/update the PR without asking for confirmation when this repository is in scope. Include summary, verification, limitations, and linked issue reference. Preserve an existing ready-for-review state; do not convert it back to draft unless the user explicitly asks.
- Prefer connected integration tools; use an authenticated CLI when necessary. Do not expose credentials or bypass authorization failures.
