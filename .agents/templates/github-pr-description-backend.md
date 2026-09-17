# Backend PR description template

Use this template for Go service, GraphQL/API, authentication, persistence, or migration work. Replace every bracketed placeholder. Remove a conditional section only when it genuinely does not apply.

The PR description is the review and handoff record. Do not claim a check passed unless the exact command exited 0. Put blockers and pre-existing failures in their own section rather than presenting partial execution as success.

## Linked work

- Linear issue: [AZH-000 and URL]
- Parent/source issue: [issue ID and URL, or "None"]
- Depends on: [issue/PR and delivered artifact, or "None"]
- Unblocks: [issue IDs, or "None"]

## Review and merge order

- Delivery shape: [Single focused PR | Stacked PR | Parallel PR group]
- PR/MR link: [Direct URL for this pull request]
- This PR's review position: [Standalone | PR 1 of N | PR N of N | Parallel member A/B]
- Base branch: [main or predecessor branch]
- Depends on: [PR/commit and the exact delivered behavior, or "None"]
- Review order: [Exact order, or "Any order within <parallel group>"]
- Merge order and conditions: [Exact merge sequence and prerequisite checks, or "Any order; all required checks green"]
- Parallel group: [Group name and independent members, or "None"]
- Human-verification focus: [The one behavior and manual check a reviewer should prioritize]

## Summary

- [Observable backend outcome.]
- [API, security, persistence, or migration outcome.]
- [Important compatibility outcome.]

## Scope

### Included

- [Implemented behavior and affected package/path.]
- [Additional in-scope behavior.]

### Excluded

- [Nearby work intentionally not changed.]

## Verification

### Automated code checks (supporting only)

Record exact unfiltered commands and exit statuses here. Unit tests, integration tests, lint, builds, mocks, and local protocol checks are regression evidence at a code seam; they do not prove live API, authentication, PostgreSQL, Vault, or ownership behavior.

### Manual request/response sequence

Write this as a sequence of small, copy-pasteable Bash steps in the PR description.
Do not attach a script file or combine the whole verification into one bulk script.
Use real staging data and the documented staging fixture for the issue; do not use
placeholder values, fake records, or mocks. Never paste credentials, tokens, or
API keys into the PR. Do not use `set -o pipefail`, `set -e`, `set -Eeuo pipefail`,
or another fail-fast wrapper: keep each step independently runnable, print its
exit status, and leave the interactive terminal open when a request fails.
Do not use `exit`, `exit 1`, or cleanup traps that call `exit`. Keep setup in a
separate short block, use at most one network request per numbered step, avoid
helper functions and bulk scripts, and wrap long commands with continuations.

#### Step 0 — Load the verified staging environment

```bash
set -a
. .agents/.env
set +a
: "${STAGING_API_BASE_URL:?Set this to the verified staging API URL before running the step}"
export API_BASE_URL="$STAGING_API_BASE_URL"
printf 'staging target: %s\n' "$API_BASE_URL"
```

Expected response:

```text
staging target: https://<verified-staging-host>
```

The command must exit 0 and identify the real staging target without printing
any secret values. Replace the example host with the actual target before handoff;
do not leave a placeholder in the completed PR.

#### Step 1 — Authenticate with the real staging fixture (when required)

If the flow requires authentication, use the account variables supplied by
`.agents/.env`. Keep the login request and token export as separate commands so
the token remains in the current shell only and is never committed or printed.

```bash
login_response="$({
  curl --fail-with-body --silent --show-error \
    --request POST \
    --header 'Content-Type: application/json' \
    --data "{\"username\":\"$CASDOOR_TEST_USERNAME\",\"password\":\"$CASDOOR_TEST_PASSWORD\"}" \
    "$API_BASE_URL/api/auth/login"
})"
printf '%s\n' "$login_response" | jq 'del(.token)'
export TOKEN="$(printf '%s' "$login_response" | jq -er '.token')"
```

Expected response:

```json
{"authenticated":true,"user":{"id":"<verified-fixture-user-id>","tier":"<verified-fixture-tier>"}}
```

The login command must exit 0, the response must identify the real fixture
account, and `TOKEN` must be exported for the following steps. Replace the
fixture variable/path only with the documented staging values for the issue.

#### Step 2 — [real behavior under test]

Command:

```bash
curl \
  --header "Authorization: Bearer $TOKEN" \
  "$API_BASE_URL/<documented-staging-path>"
```

Expected response:

```json
{"<field>":"<verified-staging-value>"}
```

Replace the path, payload, and expected response with the real staging
operation and its observed contract before handoff. Add one numbered step per
meaningful action (for example: create, read, update, delete, retry, or verify
ownership); each step must include its own Bash command and expected response.

#### Step 3 — [next real behavior or regression check]

Command:

```bash
[one command for the next real staging action]
```

Expected response:

```text
[observable status, response field, or assertion from staging]
```

Keep adding explicit steps until the complete user/API journey is covered.
Record the exact command exit status and distinguish staging or pre-existing
failures from regressions introduced by this PR.

## Known limitations and pre-existing failures

- [Exact command, exit status, affected path, and why it is unrelated; or "None known".]

## Reviewer focus

- [Highest-risk contract, migration, authorization, or concurrency decision.]
- [Specific file or behavior that merits close review.]

## Completion self-audit

- [ ] Every issue requirement is mapped to an implemented outcome or explicit blocker.
- [ ] Every changed operation lists its inputs, outputs, errors, and authorization behavior.
- [ ] Persistence claims cross a new request/client/process boundary rather than reuse an in-memory object.
- [ ] Migration up/down and existing-data behavior are documented and tested where applicable.
- [ ] Success, validation, unauthenticated, cross-user, absent-resource, and persistence-error paths are covered where applicable.
- [ ] Exact unfiltered commands and exit statuses are recorded.
- [ ] Generated coverage, logs, dumps, credentials, and unrelated files are absent from the diff.
- [ ] Automated checks are clearly separated from manual acceptance evidence and are not presented as proof that the live work is complete.
- [ ] For API work, the PR body includes a copy-paste manual request/response sequence using real configured fixtures; automated tests alone do not satisfy this check.
- [ ] The linked Linear issue and dependencies reflect the actual handoff state.
