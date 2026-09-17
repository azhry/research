# Research

Workspace for independent research projects.

## Projects

Project-specific agent context lives under `.agents/knowledge/<project-slug>/`.
The current study is:

- [Code Retrieval](.agents/knowledge/code-retrieval/project.md) — Python retrieval
  experiments on COIR/CosQA comparing E5, HyDE, and cross-encoder re-ranking with
  `nDCG@10` as the primary metric.

Add future studies as separate project-slug directories instead of placing their
assumptions in the workspace-wide agent context.

## Shared agent context

- `.agents/project.md` — workspace-wide conventions
- `.agents/knowledge/` — project routing and study specifications
- `.agents/workflows/` — shared research and delivery workflows
- `.agents/skills/` — reusable project skills
- `.agents/templates/` — shared issue, PR, and report templates

## Runtime

Projects may use Google Colab, Jupyter, or another documented environment. Record
dataset/model revisions, parameters, seeds, execution status, and artifact
provenance for every benchmark result.
