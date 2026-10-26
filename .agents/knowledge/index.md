# Research workspace knowledge index

The root knowledge directory routes context for multiple research projects. Each
project's study-specific documents live under its own slug directory:

- [Code Retrieval](code-retrieval/project.md) — COIR/CosQA study using E5, HyDE,
  and re-ranking.
- [Sentiment Analysis](sentiment-analysis/project.md) — Indonesian ASTE
  experiments on HoASA and CASA comparing EMCGCN and SSGCN.

For a new study, add `.agents/knowledge/<project-slug>/project.md` and link it here.
Do not treat another project's context as a default. Load only the selected
project's documents plus the shared workflow needed for the task.
