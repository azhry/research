#!/usr/bin/env python3
"""HTTP API fallback scripts for Linear tooling when the MCP connector is unavailable."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

LINEAR_API_URL = "https://api.linear.app/graphql"
SCRIPT_NAME = "linear_tooling.py"


def get_api_key() -> str:
    key = os.environ.get("LINEAR_API_KEY", "")
    if not key:
        print("Error: LINEAR_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)
    return key


def graphql_request(query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {"query": query}
    if variables is not None:
        payload["variables"] = variables

    data = json.dumps(payload).encode("utf-8")
    req = Request(
        LINEAR_API_URL,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": get_api_key(),
        },
        method="POST",
    )

    try:
        with urlopen(req, timeout=30) as response:
            body = response.read().decode("utf-8")
            return json.loads(body)
    except HTTPError as e:
        body = e.read().decode("utf-8") if e.fp else ""
        print(f"HTTP error {e.code}: {body}", file=sys.stderr)
        sys.exit(1)
    except URLError as e:
        print(f"Connection error: {e.reason}", file=sys.stderr)
        sys.exit(1)


def extract_data(result: dict[str, Any]) -> Any:
    if "errors" in result and result["errors"]:
        for error in result["errors"]:
            print(f"GraphQL error: {error}", file=sys.stderr)
        sys.exit(1)
    return result.get("data")


def cmd_read_issue(args: argparse.Namespace) -> None:
    query = """
    query ReadIssue($id: String!) {
      issue(id: $id) {
        id
        title
        description
        state { name }
        priority
        assignee { name }
        project { name }
        cycle { name }
        labels { nodes { name } }
        url
        createdAt
        updatedAt
      }
    }
    """
    variables = {"id": args.id}
    result = graphql_request(query, variables)
    print(json.dumps(extract_data(result), indent=2))


def cmd_search(args: argparse.Namespace) -> None:
    query = """
    query SearchIssues($query: String!) {
      issues(query: $query) {
        nodes {
          id
          title
          state { name }
          priority
          assignee { name }
          project { name }
          url
        }
      }
    }
    """
    variables = {"query": args.query}
    result = graphql_request(query, variables)
    print(json.dumps(extract_data(result), indent=2))


def cmd_list_issues(args: argparse.Namespace) -> None:
    query = """
    query ListIssues($teamId: String, $projectId: String, $state: String, $assigneeId: String, $first: Int, $after: String) {
      issues(teamId: $teamId, projectId: $projectId, stateId: $state, assigneeId: $assigneeId, first: $first, after: $after) {
        nodes {
          id
          title
          state { name }
          priority
          assignee { name }
          project { name }
          url
        }
        pageInfo {
          endCursor
          hasNextPage
        }
      }
    }
    """
    variables: dict[str, Any] = {}
    if args.team_id:
        variables["teamId"] = args.team_id
    if args.project_id:
        variables["projectId"] = args.project_id
    if args.state:
        variables["stateId"] = args.state
    if args.assignee_id:
        variables["assigneeId"] = args.assignee_id
    if args.first:
        variables["first"] = args.first
    if args.after:
        variables["after"] = args.after
    result = graphql_request(query, variables)
    print(json.dumps(extract_data(result), indent=2))


def cmd_list_statuses(args: argparse.Namespace) -> None:
    query = """
    query ListStatuses($teamId: String!) {
      issuesStatuses(teamId: $teamId) {
        nodes {
          id
          name
          type
        }
      }
    }
    """
    if not args.team_id:
        print("Error: --team-id is required for listing statuses.", file=sys.stderr)
        sys.exit(1)
    variables = {"teamId": args.team_id}
    result = graphql_request(query, variables)
    print(json.dumps(extract_data(result), indent=2))


def cmd_save_issue(args: argparse.Namespace) -> None:
    query = """
    mutation SaveIssue($input: IssueInput!) {
      issueCreate(input: $input) {
        success
        issue {
          id
          title
          state { name }
          priority
          assignee { name }
          project { name }
          url
        }
      }
    }
    """
    input_data: dict[str, Any] = {"title": args.title}
    if args.description:
        input_data["description"] = args.description
    if args.team_id:
        input_data["teamId"] = args.team_id
    if args.project_id:
        input_data["projectId"] = args.project_id
    if args.state_id:
        input_data["stateId"] = args.state_id
    if args.priority is not None:
        input_data["priority"] = args.priority
    if args.assignee_id:
        input_data["assigneeId"] = args.assignee_id
    if args.labels:
        input_data["labelIds"] = args.labels.split(",")
    variables = {"input": input_data}
    result = graphql_request(query, variables)
    print(json.dumps(extract_data(result), indent=2))


def cmd_save_comment(args: argparse.Namespace) -> None:
    query = """
    mutation SaveComment($input: CommentInput!) {
      commentCreate(input: $input) {
        success
        comment {
          id
          body
          issue { id title }
          url
        }
      }
    }
    """
    if not args.issue_id:
        print("Error: --issue-id is required for creating a comment.", file=sys.stderr)
        sys.exit(1)
    if not args.body:
        print("Error: --body is required for creating a comment.", file=sys.stderr)
        sys.exit(1)
    input_data = {"issueId": args.issue_id, "body": args.body}
    variables = {"input": input_data}
    result = graphql_request(query, variables)
    print(json.dumps(extract_data(result), indent=2))


def cmd_list_projects(args: argparse.Namespace) -> None:
    query = """
    query ListProjects {
      projects {
        nodes {
          id
          name
          icon
          state { name }
          startDate
          targetDate
        }
      }
    }
    """
    result = graphql_request(query)
    print(json.dumps(extract_data(result), indent=2))


def cmd_list_teams(args: argparse.Namespace) -> None:
    query = """
    query ListTeams {
      teams {
        nodes {
          id
          name
          key
          memberCount
        }
      }
    }
    """
    result = graphql_request(query)
    print(json.dumps(extract_data(result), indent=2))


def cmd_list_cycles(args: argparse.Namespace) -> None:
    query = """
    query ListCycles($teamId: String!) {
      cycles(teamId: $teamId) {
        nodes {
          id
          name
          startDate
          endDate
          state { name }
        }
      }
    }
    """
    if not args.team_id:
        print("Error: --team-id is required for listing cycles.", file=sys.stderr)
        sys.exit(1)
    variables = {"teamId": args.team_id}
    result = graphql_request(query, variables)
    print(json.dumps(extract_data(result), indent=2))


def cmd_save_project(args: argparse.Namespace) -> None:
    query = """
    mutation SaveProject($input: ProjectInput!) {
      projectCreate(input: $input) {
        success
        project {
          id
          name
          icon
          state { name }
        }
      }
    }
    """
    if not args.name:
        print("Error: --name is required for creating a project.", file=sys.stderr)
        sys.exit(1)
    input_data: dict[str, Any] = {"name": args.name}
    if args.team_id:
        input_data["teamId"] = args.team_id
    if args.description:
        input_data["description"] = args.description
    if args.icon:
        input_data["icon"] = args.icon
    if args.state_id:
        input_data["stateId"] = args.state_id
    if args.start_date:
        input_data["startDate"] = args.start_date
    if args.target_date:
        input_data["targetDate"] = args.target_date
    variables = {"input": input_data}
    result = graphql_request(query, variables)
    print(json.dumps(extract_data(result), indent=2))


def cmd_save_document(args: argparse.Namespace) -> None:
    query = """
    mutation SaveDocument($input: DocumentInput!) {
      documentCreate(input: $input) {
        success
        document {
          id
          title
          url
        }
      }
    }
    """
    if not args.title:
        print("Error: --title is required for creating a document.", file=sys.stderr)
        sys.exit(1)
    input_data: dict[str, Any] = {"title": args.title}
    if args.project_id:
        input_data["projectId"] = args.project_id
    if args.body:
        input_data["body"] = args.body
    if args.parent_doc_id:
        input_data["parentDocId"] = args.parent_doc_id
    variables = {"input": input_data}
    result = graphql_request(query, variables)
    print(json.dumps(extract_data(result), indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=SCRIPT_NAME,
        description="Linear HTTP API fallback scripts for tooling when the MCP connector is unavailable.",
    )
    subparsers = parser.add_subparsers(dest="command")

    p_read = subparsers.add_parser("read-issue", help="Read a single Linear issue by ID")
    p_read.add_argument("--id", required=True, help="Linear issue ID (e.g. AZH-417)")
    p_read.set_defaults(func=cmd_read_issue)

    p_search = subparsers.add_parser("search", help="Find work by text query")
    p_search.add_argument("--query", required=True, help="Search query string")
    p_search.set_defaults(func=cmd_search)

    p_list = subparsers.add_parser("list-issues", help="List and filter Linear issues")
    p_list.add_argument("--team-id", help="Team ID to filter by")
    p_list.add_argument("--project-id", help="Project ID to filter by")
    p_list.add_argument("--state", help="State ID to filter by")
    p_list.add_argument("--assignee-id", help="Assignee ID to filter by")
    p_list.add_argument("--first", type=int, help="Maximum number of issues to return")
    p_list.add_argument("--after", help="Pagination cursor")
    p_list.set_defaults(func=cmd_list_issues)

    p_statuses = subparsers.add_parser("list-statuses", help="List valid issue statuses for a team")
    p_statuses.add_argument("--team-id", required=True, help="Team ID")
    p_statuses.set_defaults(func=cmd_list_statuses)

    p_save = subparsers.add_parser("save-issue", help="Create or update a Linear issue")
    p_save.add_argument("--title", required=True, help="Issue title")
    p_save.add_argument("--description", help="Issue description")
    p_save.add_argument("--team-id", help="Team ID")
    p_save.add_argument("--project-id", help="Project ID")
    p_save.add_argument("--state-id", help="State ID")
    p_save.add_argument("--priority", type=int, help="Issue priority")
    p_save.add_argument("--assignee-id", help="Assignee ID")
    p_save.add_argument("--labels", help="Comma-separated label IDs")
    p_save.set_defaults(func=cmd_save_issue)

    p_comment = subparsers.add_parser("save-comment", help="Create a comment on an issue")
    p_comment.add_argument("--issue-id", required=True, help="Linear issue ID")
    p_comment.add_argument("--body", required=True, help="Comment body text")
    p_comment.set_defaults(func=cmd_save_comment)

    p_projects = subparsers.add_parser("list-projects", help="List Linear projects")
    p_projects.set_defaults(func=cmd_list_projects)

    p_teams = subparsers.add_parser("list-teams", help="List Linear teams")
    p_teams.set_defaults(func=cmd_list_teams)

    p_cycles = subparsers.add_parser("list-cycles", help="List cycles for a team")
    p_cycles.add_argument("--team-id", required=True, help="Team ID")
    p_cycles.set_defaults(func=cmd_list_cycles)

    p_save_project = subparsers.add_parser("save-project", help="Create or update a Linear project")
    p_save_project.add_argument("--name", help="Project name")
    p_save_project.add_argument("--team-id", help="Team ID")
    p_save_project.add_argument("--description", help="Project description")
    p_save_project.add_argument("--icon", help="Project icon")
    p_save_project.add_argument("--state-id", help="Project state ID")
    p_save_project.add_argument("--start-date", help="Project start date (ISO 8601)")
    p_save_project.add_argument("--target-date", help="Project target date (ISO 8601)")
    p_save_project.set_defaults(func=cmd_save_project)

    p_save_doc = subparsers.add_parser("save-document", help="Create or update a Linear document")
    p_save_doc.add_argument("--title", help="Document title")
    p_save_doc.add_argument("--project-id", help="Project ID")
    p_save_doc.add_argument("--body", help="Document body text")
    p_save_doc.add_argument("--parent-doc-id", help="Parent document ID")
    p_save_doc.set_defaults(func=cmd_save_document)

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