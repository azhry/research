---
description:
---

# Skill workflow

Apply this workflow when creating or changing reusable agent skills.

- Keep a skill portable by default. Do not hard-code a repository, team, ticket prefix, credential file, or branch convention unless the user specifically requests a project-bound skill.
- Explain what platform or domain the skill covers, what it is capable of, and when it should trigger.
- Include concrete connector tool names, CLI commands, or API endpoints plus discovery and safe fallback rules when the skill covers an integration.
- Never place credentials in skills, examples, source, logs, or prompts.
- Keep `SKILL.md` concise; put detailed command/API mappings in a directly linked reference file.
- Validate structure and examples before handoff. For executable integration helpers, add isolated tests that assert request method, URL, headers, payload, successful response parsing, and HTTP/network failures without using live credentials.
