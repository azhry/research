# GitHub operations

Use Git for local history, a connected GitHub app for GitHub actions, and an authenticated `gh` CLI when the connector is unavailable or lacks a needed feature.

## Git commands

```sh
git status --short --branch
git switch main && git switch -c <topic-branch>
git diff --check
git add <intended-paths>
git commit -m "feat(scope): summary"
git push -u origin <topic-branch>
```

Run the push only after the user authorizes external publication.

## GitHub app calls

Discover the exact callable names from the active tool list; Codex commonly exposes these as `mcp__codex_apps__github_*`.

| Intent | Connector call |
| --- | --- |
| Create PR | `github_create_pull_request({ repository_full_name, head, base, title, body, draft })` |
| Inspect PR | `github_fetch_pr` or `github_get_pr_info` |
| Find PRs | `github_search_prs` |
| Comment/review | `github_add_comment_to_issue`, `github_add_review_to_pr` |
| Inspect checks | `github_get_commit_combined_status`, `github_fetch_commit_workflow_runs`, or `github_fetch_workflow_run_jobs` |
| Compare branches | `github_compare_commits({ repo_full_name, base, head })` |

## GitHub CLI fallback

### Resolve and preflight the official CLI

Do this before any source edit, branch creation, commit, push, or PR command. Do not assume that `gh` on PATH is the GitHub CLI: Node/npm packages can shadow it and may prompt for an interactive login.

On PowerShell, prefer a real `gh.exe` outside Node/npm paths, then retain its full path for every subsequent call:

```powershell
$gh = Get-Command gh.exe -All |
  Where-Object { $_.Source -notmatch '(?i)node_modules|\\nodejs\\gh(?:\.cmd|\.ps1)?$' } |
  Select-Object -First 1 -ExpandProperty Source
if (-not $gh) { throw 'Official GitHub CLI executable not found' }
& $gh --version
& $gh auth status
& $gh repo view OWNER/REPO --json nameWithOwner,viewerPermission
```

On POSIX shells, inspect all candidates and use the verified executable path:

```sh
type -a gh
gh_bin="$(command -v gh)"
"$gh_bin" --version
"$gh_bin" auth status
"$gh_bin" repo view OWNER/REPO --json nameWithOwner,viewerPermission
```

The preflight passes only when all three commands succeed and the repository result identifies the intended repository. Do not continue with a different `gh`, an interactive `gh auth login`, `GH_TOKEN` populated from a project config file, or a direct HTTP request containing a project token. Use a connected GitHub app if available; otherwise record the blocker and stop.

The bundled preflight helper performs those checks atomically:

```powershell
& .agents/skills/github/scripts/gh_preflight.ps1 -Repository OWNER/REPO
```

### Deterministic PR handoff

Always pass the repository, base, and head explicitly. Check for an existing PR before creating one so a retry updates the same PR instead of creating a duplicate:

```sh
gh pr list --repo OWNER/REPO --head <branch> --json number,url,state,title,headRefName,baseRefName
gh pr create --repo OWNER/REPO --base main --head <branch> --title "<title>" --body "<body>" --draft
gh pr view <number> --repo OWNER/REPO --json url,state,isDraft,mergeable,statusCheckRollup
```

Use the resolved full executable path in place of `gh` in those commands. Merge only when the user explicitly asks, after confirming the PR is mergeable and reporting its check state.

If `gh` is not authenticated, report that blocker instead of starting an interactive login in an unattended workflow.

## REST API fallback

Use `POST /repos/{owner}/{repo}/pulls` to create a PR and `GET /repos/{owner}/{repo}/pulls/{number}` to inspect one. Obtain credentials only from the execution environment's secret store; never include a token in source, logs, or PR text.

Invoke the bundled [`gh_tooling.py`](scripts/gh_tooling.py) operation before writing custom scripts. If an operation is missing, consult the official schema; API validation errors must not be used as schema discovery.
