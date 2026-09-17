#!/usr/bin/env python3
"""HTTP API fallback scripts for GitHub tooling when the connected GitHub app and gh CLI are unavailable.

Reads the GitHub personal access token from the GITHUB_TOKEN environment variable only.
Never reads a token from a project config file, source comments, or logs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

GITHUB_API_URL = "https://api.github.com"
SCRIPT_NAME = "gh_tooling.py"
API_VERSION = "2022-11-28"


def get_api_key() -> str:
    key = os.environ.get("GITHUB_TOKEN", "")
    if not key:
        print("Error: GITHUB_TOKEN environment variable is not set.", file=sys.stderr)
        sys.exit(1)
    return key


def github_request(
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    url = f"{GITHUB_API_URL}{path}"
    data: bytes | None = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    req = Request(
        url,
        data=data,
        headers={
            "Accept": f"application/vnd.github+json;api-version={API_VERSION}",
            "Authorization": f"Bearer {get_api_key()}",
            "X-GitHub-Api-Version": API_VERSION,
            "Content-Type": "application/json",
        },
        method=method,
    )

    try:
        with urlopen(req, timeout=30) as response:
            body = response.read().decode("utf-8")
            if not body:
                return {}
            return json.loads(body)
    except HTTPError as e:
        body = e.read().decode("utf-8") if e.fp else ""
        print(f"HTTP error {e.code}: {body}", file=sys.stderr)
        sys.exit(1)
    except URLError as e:
        print(f"Connection error: {e.reason}", file=sys.stderr)
        sys.exit(1)


def extract_response(result: dict[str, Any]) -> dict[str, Any]:
    if "errors" in result and result["errors"]:
        for error in result["errors"]:
            print(f"API error: {error}", file=sys.stderr)
        sys.exit(1)
    return result


# --- PR commands ---


def cmd_create_pr(args: argparse.Namespace) -> None:
    payload: dict[str, Any] = {
        "title": args.title,
        "head": args.head,
        "base": args.base,
    }
    if args.body is not None:
        payload["body"] = args.body
    if args.draft:
        payload["draft"] = True
    if args.milestone is not None:
        payload["milestone"] = int(args.milestone)

    path = f"/repos/{args.owner}/{args.repo}/pulls"
    result = github_request("POST", path, payload)
    print(json.dumps(result, indent=2))


def cmd_update_pr(args: argparse.Namespace) -> None:
    payload: dict[str, Any] = {}
    if args.title is not None:
        payload["title"] = args.title
    if args.body is not None:
        payload["body"] = args.body
    if args.state is not None:
        payload["state"] = args.state
    if args.base is not None:
        payload["base"] = args.base

    if not payload:
        print("Error: at least one of --title, --body, --state, --base is required.", file=sys.stderr)
        sys.exit(1)

    path = f"/repos/{args.owner}/{args.repo}/pulls/{args.number}"
    result = github_request("PATCH", path, payload)
    print(json.dumps(result, indent=2))


def cmd_view_pr(args: argparse.Namespace) -> None:
    path = f"/repos/{args.owner}/{args.repo}/pulls/{args.number}"
    result = github_request("GET", path)
    print(json.dumps(result, indent=2))


def cmd_list_prs(args: argparse.Namespace) -> None:
    query_params = []
    if args.state is not None:
        query_params.append(f"state={args.state}")
    if args.head is not None:
        query_params.append(f"head={args.head}")
    if args.base is not None:
        query_params.append(f"base={args.base}")
    if args.per_page is not None:
        query_params.append(f"per_page={args.per_page}")

    query = f"?{'&'.join(query_params)}" if query_params else ""
    path = f"/repos/{args.owner}/{args.repo}/pulls{query}"
    result = github_request("GET", path)
    print(json.dumps(result, indent=2))


def cmd_create_comment(args: argparse.Namespace) -> None:
    payload = {"body": args.body}
    path = f"/repos/{args.owner}/{args.repo}/issues/{args.number}/comments"
    result = github_request("POST", path, payload)
    print(json.dumps(result, indent=2))


def cmd_list_comments(args: argparse.Namespace) -> None:
    path = f"/repos/{args.owner}/{args.repo}/issues/{args.number}/comments"
    result = github_request("GET", path)
    print(json.dumps(result, indent=2))


# --- Issue commands ---


def cmd_view_issue(args: argparse.Namespace) -> None:
    path = f"/repos/{args.owner}/{args.repo}/issues/{args.number}"
    result = github_request("GET", path)
    print(json.dumps(result, indent=2))


def cmd_update_issue(args: argparse.Namespace) -> None:
    payload: dict[str, Any] = {}
    if args.title is not None:
        payload["title"] = args.title
    if args.body is not None:
        payload["body"] = args.body
    if args.state is not None:
        payload["state"] = args.state
    if args.labels is not None:
        payload["labels"] = [l.strip() for l in args.labels.split(",") if l.strip()]

    if not payload:
        print("Error: at least one of --title, --body, --state, --labels is required.", file=sys.stderr)
        sys.exit(1)

    path = f"/repos/{args.owner}/{args.repo}/issues/{args.number}"
    result = github_request("PATCH", path, payload)
    print(json.dumps(result, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=SCRIPT_NAME,
        description="GitHub HTTP API fallback scripts for tooling when the MCP connector and gh CLI are unavailable.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # create-pr
    p_create = subparsers.add_parser("create-pr", help="Create a pull request")
    p_create.add_argument("--owner", required=True, help="Repository owner")
    p_create.add_argument("--repo", required=True, help="Repository name")
    p_create.add_argument("--head", required=True, help="Branch the changes are on")
    p_create.add_argument("--base", required=True, help="Branch to merge into")
    p_create.add_argument("--title", required=True, help="PR title")
    p_create.add_argument("--body", help="PR description body")
    p_create.add_argument("--draft", action="store_true", help="Create as a draft PR")
    p_create.add_argument("--milestone", help="Milestone number")
    p_create.set_defaults(func=cmd_create_pr)

    # update-pr
    p_update = subparsers.add_parser("update-pr", help="Update an existing pull request")
    p_update.add_argument("--owner", required=True, help="Repository owner")
    p_update.add_argument("--repo", required=True, help="Repository name")
    p_update.add_argument("--number", required=True, type=int, help="Pull request number")
    p_update.add_argument("--title", help="New PR title")
    p_update.add_argument("--body", help="New PR description body")
    p_update.add_argument("--state", choices=["open", "closed"], help="PR state")
    p_update.add_argument("--base", help="New base branch")
    p_update.set_defaults(func=cmd_update_pr)

    # view-pr
    p_view = subparsers.add_parser("view-pr", help="View a pull request")
    p_view.add_argument("--owner", required=True, help="Repository owner")
    p_view.add_argument("--repo", required=True, help="Repository name")
    p_view.add_argument("--number", required=True, type=int, help="Pull request number")
    p_view.set_defaults(func=cmd_view_pr)

    # list-prs
    p_list = subparsers.add_parser("list-prs", help="List pull requests")
    p_list.add_argument("--owner", required=True, help="Repository owner")
    p_list.add_argument("--repo", required=True, help="Repository name")
    p_list.add_argument("--state", choices=["open", "closed", "all"], help="Filter by PR state")
    p_list.add_argument("--head", help="Filter by head branch")
    p_list.add_argument("--base", help="Filter by base branch")
    p_list.add_argument("--per-page", type=int, help="Results per page (max 100)")
    p_list.set_defaults(func=cmd_list_prs)

    # create-comment
    p_comment = subparsers.add_parser("create-comment", help="Add a comment to a PR or issue")
    p_comment.add_argument("--owner", required=True, help="Repository owner")
    p_comment.add_argument("--repo", required=True, help="Repository name")
    p_comment.add_argument("--number", required=True, type=int, help="PR or issue number")
    p_comment.add_argument("--body", required=True, help="Comment body")
    p_comment.set_defaults(func=cmd_create_comment)

    # list-comments
    p_list_comments = subparsers.add_parser("list-comments", help="List comments on a PR or issue")
    p_list_comments.add_argument("--owner", required=True, help="Repository owner")
    p_list_comments.add_argument("--repo", required=True, help="Repository name")
    p_list_comments.add_argument("--number", required=True, type=int, help="PR or issue number")
    p_list_comments.set_defaults(func=cmd_list_comments)

    # view-issue
    p_view_issue = subparsers.add_parser("view-issue", help="View an issue")
    p_view_issue.add_argument("--owner", required=True, help="Repository owner")
    p_view_issue.add_argument("--repo", required=True, help="Repository name")
    p_view_issue.add_argument("--number", required=True, type=int, help="Issue number")
    p_view_issue.set_defaults(func=cmd_view_issue)

    # update-issue
    p_update_issue = subparsers.add_parser("update-issue", help="Update an issue")
    p_update_issue.add_argument("--owner", required=True, help="Repository owner")
    p_update_issue.add_argument("--repo", required=True, help="Repository name")
    p_update_issue.add_argument("--number", required=True, type=int, help="Issue number")
    p_update_issue.add_argument("--title", help="New issue title")
    p_update_issue.add_argument("--body", help="New issue description body")
    p_update_issue.add_argument("--state", choices=["open", "closed"], help="Issue state")
    p_update_issue.add_argument("--labels", help="Comma-separated label names")
    p_update_issue.set_defaults(func=cmd_update_issue)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
