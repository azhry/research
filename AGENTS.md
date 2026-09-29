# Workspace agent instructions

## Read workspace guidance first

Before doing any other task work, read all non-secret guidance and project
context under `.agents/`, including shared instructions, workflows, relevant
skills, templates, and project knowledge, specifications, and reproducibility
notes. Use `.agents/knowledge/index.md` to identify the project and read its
context before changing notebooks, code, or artifacts.

Treat credential-bearing files such as `.agents/config.md` and `.agents/.env` as
secrets. Consult them only when a task requires those credentials, and never
print, copy into reports, or disclose their values.

## Work from the existing experiment

For an experiment that already has a notebook, scripts, caches, or results, do
not create a custom one-off script or parallel notebook that duplicates it.
Start with the existing entry point and saved artifacts. Run the existing code
when appropriate, compare its output with the documented specification and
goal, then refine the existing notebook or reusable scripts until the result
complies. Keep analysis attached to the experiment workflow where researchers
can inspect and reproduce it.

Add a new script only when the existing code has no suitable entry point or
reusable capability. In that case, make it reusable, integrate it with the
existing workflow, and document why it is needed.

## Preserve research integrity

- Keep each study's question, hypotheses, dataset, models, metrics, and evidence
  boundaries explicit.
- Preserve run identities, source revisions, caches, notebook outputs, and
  historical artifacts. Do not overwrite or relabel prior results.
- Distinguish smoke runs, partial runs, exploratory analyses, and complete
  benchmark evidence.
- Never use fake model responses or synthetic scores as evidence for model
  quality.
- Separate measured findings from hypotheses and state material limitations.
