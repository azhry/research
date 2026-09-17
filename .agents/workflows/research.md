# Multi-project research workflow

Apply this workflow to notebooks, data preparation, model pipelines, evaluation
code, experiment reports, and research artifacts in any project in this workspace.

## Before editing

1. Identify the project slug and read its `.agents/projects/<slug>/project.md`.
2. Read that project's study specification and reproducibility notes when present.
3. Inspect repository status, local instructions, requirements, and current layout.
   Preserve unrelated dirty files.
4. State the research question, hypothesis, controlled variables, deliberate change,
   and evidence artifact for the task.

## Implementation

- Keep loading, preprocessing, model inference, retrieval/ranking, evaluation,
  caching, and reporting at separate seams.
- Make model/dataset revisions, prompts, parameters, seeds, device, and output paths
  explicit configuration.
- Use the same declared data, labels, evaluator, and controls for comparable runs.
- Keep exploratory deviations out of the primary comparison or label them clearly.
- Do not use fake LLM/Codex responses, mocked core agent clients, or synthetic scores
  as evidence for model-quality claims. Small deterministic fixtures may test
  plumbing only and must be labeled as such.

## Verification

Run checks from narrow to broad:

1. Import/compile checks.
2. Deterministic regression tests for adapters, shapes, metrics, and cache rules.
3. Small real-model smoke run for wiring.
4. Complete declared experiment or benchmark run.

Capture the immediate status of each command. A smoke run, partial execution, stale
cache, or failed notebook is not a successful benchmark. Report exact commands,
exits, artifacts, and limitations.

## Interpretation and handoff

Separate measured effects from hypotheses. Report the primary metric, comparison
baseline, operational cost, run identity, artifact locations, and any unrun or
unavailable boundary. Keep one coherent research outcome per PR or tracker issue.
