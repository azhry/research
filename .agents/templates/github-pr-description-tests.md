# Test-task PR description template

Use this template for test-only work: unit, contract, integration, E2E, regression, or an intentional red-test handoff. Replace every bracketed placeholder. Remove a conditional section only when it genuinely does not apply.

Choose exactly one outcome mode:

- **Green verification:** the PR adds or repairs tests that must exit 0.
- **Red-test handoff:** the PR deliberately captures missing production behavior for a paired implementation issue. The targeted test and its designated GitHub check must reach the intended assertion and fail for that reason only. Infrastructure/setup and unrelated failures are blockers, not acceptable red evidence. When the issue explicitly authorizes an intentional-red merge, this test-only PR may merge into `main` with that designated check still red; do not misclassify the expected failure as a blocker.

## Linked work

- Test issue: [AZH-000 and URL]
- Paired implementation issue: [AZH-000 and URL, or "None"]
- Parent/source issue: [issue ID and URL, or "None"]
- Outcome mode: [Green verification | Red-test handoff]
- Handoff commit/branch: [Required for red-test handoff; otherwise "Not applicable"]
- Merge authorization: [Issue text authorizing intentional-red merge, or "Not applicable"]
- Paired issue start point: [Updated `main` after this PR merges, or "Not applicable"]

## Review and merge order

- Delivery shape: [Single focused PR | Stacked PR | Parallel PR group]
- This PR's review position: [Standalone | PR 1 of N | PR N of N | Parallel member A/B]
- Base branch: [main or predecessor branch]
- Depends on: [PR/commit and the exact delivered behavior, or "None"]
- Review order: [Exact order, or "Any order within <parallel group>"]
- Merge order and conditions: [Exact merge sequence and prerequisite checks, or "Any order; all required checks green"]
- Parallel group: [Group name and independent members, or "None"]
- Human-verification focus: [The contract assertion, fixture boundary, or test seam a reviewer should prioritize]

## Test contract

- Behavior under test: [Observable contract.]
- Confirmed pre-test behavior: [Evidence and affected seam.]
- Expected result in this PR:
  - Green mode: [Passing behavior.]
  - Red mode: [Exact assertion expected to fail until the paired implementation.]
- Production-code boundary: [State whether production files are intentionally unchanged.]

## Coverage matrix

| Scenario | Test seam | Setup/fixture | Expected assertion |
| --- | --- | --- | --- |
| Success/current behavior | [Unit/service/API/browser] | [Deterministic setup] | [Result] |
| Missing/invalid input | [Seam] | [Setup] | [Error/result] |
| Authentication/ownership | [Seam] | [Users/resources] | [Isolation result] |
| Compatibility/legacy data | [Seam] | [Fixture] | [Result] |
| Boundary/error state | [Seam] | [Setup] | [Result] |

## Environment and isolation

- Runtime/dependencies: [Versions or relevant services]
- Fresh-state strategy: [New process/database/container/context per run]
- Migrations/setup: [What runs and how success is established]
- Fixtures: [File/helper and deterministic records]
- Mocking: [What is mocked and why; identify what remains real]
- Cleanup: [Files, processes, pools, databases, and containers removed]
- Repeatability: [Evidence that a second run does not reuse state]

## Database/API integration

Remove only when the task has no database or API behavior.

- Database harness: [For example, `testutil.StartPostgres`]
- API boundary: [Actual GraphQL operation or REST method/path]
- Seed boundary: [Direct SQL may prepare prerequisites; state which behavior is exercised through the API]
- Persistence boundary: [Write/create → new request/client/process → read/reload]
- Ownership/isolation: [Cross-user/resource cases]
- Cleanup evidence: [Container/pool termination, including failure path]

## Results

Record commands unfiltered and preserve their real exit statuses.

| Command | Expected mode | Actual exit | Evidence |
| --- | --- | ---: | --- |
| [Focused test command] | [Green 0 | Intentional red nonzero] | [Exit] | [Exact passing assertion or expected failing assertion] |
| [Broader unit suite] | Green | [Exit] | [Result] |
| `make test-integration` | [Green | Intentional red] | [Exit] | [Infrastructure lifecycle and contract assertion] |
| [Build/typecheck/lint] | Green | [Exit] | [Result] |

## Red-test handoff evidence

Complete this section only for red-test handoffs.

- Exact failing test: [Package/test name]
- Exact intended assertion failure: [Concise failure text]
- Why this proves the missing contract: [Evidence]
- Setup stages that succeeded before the assertion: [Compile, container, migrations, fixtures, API request]
- Unrelated suites that remain green: [Commands and exits]
- Paired implementation start point: [Commit SHA/branch]
- Test issue completion evidence: [Tracker update linking this commit, PR, command, exit, and assertion failure]
- Designated expected-red GitHub check: [Name, command, nonzero exit, and intended assertion failure]
- Unrelated GitHub checks: [Names and green results, including build and infrastructure/setup coverage]
- Merge path: [This test-only PR → `main`; paired implementation starts from updated `main` → separate green PR]
- Protection against weakening:
  - [ ] Test is not skipped or quarantined.
  - [ ] Failure is not caused by environment/setup.
  - [ ] No production implementation was added.
  - [ ] Paired implementation issue must make this test green unchanged in substance.
  - [ ] The issue description explicitly authorizes the intentional-red merge.
  - [ ] The designated GitHub check is red only for the intended assertion.
  - [ ] Compilation, infrastructure/setup, cleanup, and unrelated checks are green.
  - [ ] The paired implementation issue is updated to start from `main` after this PR merges.

## Green verification evidence

Complete this section for green verification or after the paired implementation turns red coverage green.

- Previously failing behavior: [Test/evidence]
- Passing behavior now: [Test/evidence]
- Regression protection: [Why the test fails if behavior regresses]
- Full required suite: [Commands and zero exits]

## Known limitations and pre-existing failures

- [Exact command, exit status, affected path, and why it is unrelated; or "None known".]

## Reviewer focus

- [Assertion quality, fixture realism, test seam, isolation, or flakiness risk.]
- [Any deliberate mock/real-boundary choice.]

## Completion self-audit

- [ ] The selected outcome mode is explicit and internally consistent.
- [ ] Every required scenario has a distinct assertion at the correct seam.
- [ ] Fixtures are deterministic and isolated.
- [ ] Mocks do not substitute for a required real API/database/browser boundary.
- [ ] Red mode reaches the intended assertion; setup failures are not presented as expected red evidence.
- [ ] Red mode records a complete handoff and does not treat the intended contract assertion as a task blocker.
- [ ] Green mode records zero exits for every required suite.
- [ ] Cleanup and repeat-run evidence are recorded.
- [ ] No test was weakened, skipped, or made order-dependent to obtain the reported result.
- [ ] Exact commands, exits, commit/branch, paired issue, and remaining blocker are recorded.
