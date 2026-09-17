# Linear operations

Prefer a connected Linear app. Discover the exact callable names from the active tool list; Codex commonly exposes these operations as `mcp__codex_apps__linear_*`.

| Intent | Connector call |
| --- | --- |
| Read one issue | `linear_get_issue({ id: "TEAM-123", includeRelations: true })` |
| Find work by text | `linear_search({ query: "authentication", type: "issue" })` |
| List/filter work | `linear_list_issues({ team, project, state, assignee })` |
| List valid states | `linear_list_issue_statuses({ team })` |
| Create/update an issue | `linear_save_issue({ id, state, assignee, labels, project, ... })` |
| Create/update a comment | `linear_save_comment({ issueId, body })` |
| Read projects/teams/cycles | `linear_list_projects`, `linear_list_teams`, `linear_list_cycles` |
| Create/update project docs | `linear_save_project`, `linear_save_document` |

`linear_save_issue` creates when `id` is omitted and updates when it is supplied. Resolve state names per team before setting `state`.

## HTTP API fallback

When no connector exists, use Linear's GraphQL API at `POST https://api.linear.app/graphql` with an authenticated `Authorization` header supplied by the runtime's secret store. Do not put tokens in source, logs, URLs, or issue text. Use the official schema/docs to construct the requested query or mutation; do not guess field names.

Invoke the bundled [`linear_tooling.py`](scripts/linear_tooling.py) operation before writing custom GraphQL. If an operation is missing, consult the official schema; GraphQL validation errors must not be used as schema discovery.

On Windows, resolve Python with `Get-Command py, python -ErrorAction SilentlyContinue`. When neither command resolves to an installed interpreter, use the workspace-bundled runtime.
