# Linear agent comment template

Use this template after the human-readable Linear issue description has been created or updated. Fill it out and post the completed content as an agent-facing Linear comment before implementation begins. Replace every bracketed placeholder. Remove a conditional section only when it genuinely does not apply.

The human-readable issue description is the concise entry point. This agent comment carries the detailed implementation contract: analysis, scope, dependencies, plan, acceptance criteria, verification, and execution controls. An agent must be able to implement the issue correctly from the human description, this completed comment, and the repository instructions.

## Authoring and readiness gate

- Read the complete human issue description, source request, parent issue, linked references, existing comments, relevant code, and tests before finalizing this comment.
- Translate every source requirement into an explicit, independently checkable item below. Do not replace literal requirements with a shorter summary.
- Preserve user-supplied reference material verbatim.
- If several symptoms share one root cause, one issue may cover them only when every affected surface, reproduction path, and acceptance check is enumerated.
- If a source task requires one issue or dependency pair per operation/work item, create them literally. Do not silently aggregate operations.
- Put API contracts, affected flows, dependencies, and mandatory checks in this comment. Do not overload the human-readable description with the agent execution contract.
- Re-read both the saved human description and the saved agent comment, then perform the Completion self-audit before moving the issue out of backlog/todo or beginning implementation.

## Category

**[Bug | Feature | Refactor | Chore | Research] — [short category qualifier].**

> Readiness rule: a human-readable description, design reference, private plan, or parent issue by itself is incomplete. Before implementation begins, the agent comment must contain every applicable heading in this template with evidence-based content.

## Problem analysis

### Reference material

- Identify the user-supplied HTML, mockups, screenshots, and example data preserved verbatim in the human issue description. Treat it as the visual/behavioral source of truth unless the issue explicitly says otherwise.
- Parent/source issue: [exact issue ID and the inherited requirements this issue must satisfy].

### Confirmed findings

1. **[Symptom or missing behavior]** — [evidence, affected route/module, and user impact].
2. **[Cause or implementation gap]** — [evidence and affected path].

### Discovery evidence

- Source-code review: [finding and exact paths, or "Not performed / not applicable"].
- Runtime/browser/API validation: [finding, viewport/scenario, and observable result, or "Not performed" with reason].
- Discovery classification: [source review | runtime validation | both].

### Affected surfaces and flows

- [Entry point] → [normal in-product navigation] → [affected page/control/outcome].
- [Additional page, viewport, role, state, or shared occurrence].

### Reproduction steps

1. [Start from the normal user entry point; do not rely on a direct URL unless the route is only reachable that way].
2. [Perform the user action].
3. [State the actual result and the expected result].

### Scope boundary

- In scope: [specific behavior, files, routes, or components].
- Out of scope: [nearby work deliberately excluded].
- Open questions / assumptions: [only unresolved decisions; label assumptions explicitly].

## Data / API contract (required for data-backed work)

Repeat this subsection for every operation used or required by the UI. Do not group distinct operations unless the source issue explicitly permits one work item.

### [UI surface or flow] — [operation name or endpoint]

- API type: [GraphQL query | GraphQL mutation | REST method and path].
- Operation/path: [exact operation name or HTTP method/path].
- Request variables/payload:
  - `[field]`: [type, required/optional, meaning].
- Response structure:
  - `[field]`: [type, meaning, nullable/default behavior].
- UI transformation: [how API fields map to view/store/domain fields].
- Missing fields or mismatches: [exact omissions, naming differences, lossy transformations, or "None"].
- Error/loading/empty behavior: [required observable behavior].

## Dependencies and sequencing

- Delivery mode: [Independent green PR | Intentional-red test PR | Paired implementation consuming merged red test].
- Blocked by: [exact issue IDs and what each must deliver, or "None"].
- Blocks: [exact issue IDs, or "None"].
- Required order: [for example, "Create/complete test coverage first; implementation must remain blocked until then"].
- Start gate: [state/evidence that must exist before this issue may begin].
- Handoff artifact: [exact commit/branch/PR and expected red or green state, or "Not applicable"].
- Merge vehicle: [this issue's PR | other explicitly named issue].

For an intentional-red test issue, keep its PR and the paired production PR separate. The test PR may merge into `main` when the issue explicitly authorizes that delivery mode, its designated GitHub check fails only at the recorded contract assertion, and compilation, infrastructure/setup, cleanup, and unrelated checks remain green. Do not treat that designated expected-red check as a blocker. The paired implementation issue then starts from updated `main`, makes the same test green without weakening it, and merges through its own PR.

## Implementation plan

1. [Concrete change and expected behavior].
2. [Concrete change and expected behavior].
3. Add or update regression coverage at [unit / integration / E2E seam].

## Definition of Done

- [Observable user or system outcome].
- [Accessibility, data, error, or responsive behavior when relevant].
- [No interactive-looking control is left as a no-op, if this is UI work].
- [Relevant automated tests pass, or pre-existing failures are identified separately].
- [Required delivery artifact: commit, PR, tracker update].
- [For intentional-red delivery: the designated GitHub check fails at the exact assertion after successful setup, unrelated checks remain green, the test-only PR merges, and the paired implementation issue is updated to start from that merged test].

## Correctness checks

- Automated: [exact command/test suite, expected assertions, and test seam].
- Interactive: [desktop/mobile/browser flows, keyboard checks, or API scenarios].
- Visual (when a design reference exists): [desktop and mobile comparison against the supplied reference; content/images/colors/typography/layout checklist].
- Build/lint: [exact commands or CI checks].
- Known limitations / pre-existing failures: [exact failing command, affected path, and why it is unrelated; or "None known"].

## Required execution protocol

1. Read the human issue description, this entire agent comment, and the repository instructions before changing code.
2. Treat every enumerated surface, operation, field, state, viewport, and acceptance criterion as required scope. Do not silently omit, combine, or substitute items.
3. Respect the dependency start gate above. A blocked implementation issue must not begin before its prerequisite delivers the specified evidence.
4. Keep required implementation-contract information in this comment. If new evidence changes the contract, update and re-read the comment before continuing; update the human description too when its summary, flow, before/after behavior, or manual verification changes.
5. Do not move the issue to review until the Completion self-audit is recorded with evidence.
6. For paired red-test delivery, keep the issues and PRs separate: merge the explicitly authorized test-only PR first, then start the implementation issue from updated `main` and make that test green.

## Completion self-audit

Before handoff, update this agent comment or add a concise completion comment, and link the PR evidence:

- [ ] Every in-scope requirement and every source/parent requirement is mapped to an implemented outcome or an explicit blocker.
- [ ] Every affected surface and normal navigation path was exercised where applicable.
- [ ] Every listed API operation, request field, response field, and transformation was implemented and tested where applicable.
- [ ] Every required viewport, accessibility behavior, error state, empty state, and loading state was verified where applicable.
- [ ] Dependency ordering was respected and blocker relationships remain correct.
- [ ] Delivery mode, handoff artifact, merge vehicle, and paired issue are explicit and consistent.
- [ ] Exact verification commands and exit statuses are recorded; failures are not described as passes.
- [ ] The final diff, commit, PR, and tracker state contain only this issue's intended work.
