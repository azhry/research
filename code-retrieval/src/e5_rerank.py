"""Reusable pieces for the CosQA E5 plus cross-encoder reranking experiment.

The notebook remains the ordered experiment entry point.  This module keeps
cross-encoder input construction, score validation, candidate-pool preserving
reranking, cache identity, and result serialization testable without loading a
model or fabricating benchmark output.
"""

from __future__ import annotations

import time
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from e5_baseline import (
    BaselineConfig,
    CosQAData,
    RunData,
    canonical_json,
    choose_device,
    environment_metadata,
    evaluate_ndcg_at_10,
    git_commit,
    load_valid_embedding_cache,
    load_valid_json_cache,
    rank_with_faiss,
    save_embedding_cache,
    save_json_cache,
    set_seed,
    sha256_ids,
    sha256_text,
    write_result_artifacts,
)


CODE_VERSION = "e5-rerank-v2-paper-faiss-ranking"
DEFAULT_RERANKER_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"
DEFAULT_RERANKER_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"


@dataclass(frozen=True)
class RerankConfig(BaselineConfig):
    """Explicit controls for the E5 plus reranking experiment."""

    reranker_id: str = DEFAULT_RERANKER_ID
    reranker_revision: str = DEFAULT_RERANKER_REVISION
    reranker_max_seq_length: int = 512
    reranker_batch_size: int = 32
    candidate_depth: int = 1000
    cache_dir: str = "artifacts/e5_rerank/cache"
    artifact_dir: str = "artifacts/e5_rerank"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.reranker_max_seq_length <= 0 or self.reranker_batch_size <= 0:
            raise ValueError(
                "reranker_max_seq_length and reranker_batch_size must be positive"
            )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def rerank_run_identity(
    config: RerankConfig, *, repo_root: Path | None = None
) -> str:
    """Return an identity that includes every first- and second-stage control."""

    payload = {
        "code_version": CODE_VERSION,
        "config": config.as_dict(),
        "git_commit": git_commit(repo_root),
    }
    return sha256_text(canonical_json(payload))[:16]


def rerank_cache_paths(config: RerankConfig, identity: str) -> dict[str, Path]:
    root = Path(config.cache_dir) / identity
    return {
        "root": root,
        "corpus_embeddings": root / "corpus_embeddings.npy",
        "corpus_metadata": root / "corpus_embeddings.metadata.json",
        "query_embeddings": root / "query_embeddings.npy",
        "query_metadata": root / "query_embeddings.metadata.json",
        "rankings": root / "rankings.json",
        "rankings_metadata": root / "rankings.metadata.json",
        "reranked_rankings": root / "reranked_rankings.json",
        "reranked_rankings_metadata": root / "reranked_rankings.metadata.json",
    }


def expected_rerank_cache_metadata(
    config: RerankConfig,
    *,
    identity: str,
    kind: str,
    ids: Sequence[str],
    repo_root: Path | None = None,
) -> dict[str, Any]:
    return {
        "cache_format": 1,
        "kind": kind,
        "identity": identity,
        "code_version": CODE_VERSION,
        "git_commit": git_commit(repo_root),
        "config": config.as_dict(),
        "ids_sha256": sha256_ids(ids),
        "count": len(ids),
    }


def _validate_scores(scores: Sequence[float] | np.ndarray, expected_count: int) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    if values.shape != (expected_count,):
        raise ValueError(
            f"cross-encoder returned {values.size} scores for {expected_count} pairs"
        )
    if not np.isfinite(values).all():
        raise ValueError("cross-encoder returned non-finite scores")
    return values


def build_candidate_pairs(
    queries: Mapping[str, str],
    corpus: Mapping[str, Mapping[str, str]],
    rankings: Mapping[str, Mapping[str, float]],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Build raw query/passage pairs in ranking order."""

    pairs: list[tuple[str, str]] = []
    keys: list[tuple[str, str]] = []
    for query_id, candidate_scores in rankings.items():
        if query_id not in queries:
            raise ValueError(f"ranking references an unknown query: {query_id}")
        if not candidate_scores:
            raise ValueError(f"ranking has no candidates for query: {query_id}")
        for corpus_id in candidate_scores:
            if corpus_id not in corpus:
                raise ValueError(
                    f"ranking references an unknown corpus ID: {corpus_id}"
                )
            text = corpus[corpus_id].get("text")
            if text is None or not str(text).strip():
                raise ValueError(f"corpus document has no usable text: {corpus_id}")
            pairs.append((str(queries[query_id]), str(text)))
            keys.append((str(query_id), str(corpus_id)))
    if not pairs:
        raise ValueError("candidate rankings must be non-empty")
    return pairs, keys


def score_candidate_pool(
    queries: Mapping[str, str],
    corpus: Mapping[str, Mapping[str, str]],
    rankings: Mapping[str, Mapping[str, float]],
    score_pairs_fn: Callable[[Sequence[tuple[str, str]]], Sequence[float] | np.ndarray],
) -> dict[str, dict[str, float]]:
    """Score exactly the supplied first-stage candidate pairs."""

    pairs, keys = build_candidate_pairs(queries, corpus, rankings)
    values = _validate_scores(score_pairs_fn(pairs), len(pairs))
    grouped: dict[str, dict[str, float]] = {}
    for (query_id, corpus_id), score in zip(keys, values):
        row = grouped.setdefault(query_id, {})
        if corpus_id in row:
            raise ValueError(f"duplicate candidate pair: {query_id}/{corpus_id}")
        row[corpus_id] = float(score)
    return grouped


def rerank_rankings(
    first_stage_rankings: Mapping[str, Mapping[str, float]],
    reranker_scores: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    """Reorder each first-stage ranking without changing its candidate set."""

    if not first_stage_rankings:
        raise ValueError("candidate rankings must be non-empty")
    if set(first_stage_rankings) != set(reranker_scores):
        raise ValueError("reranker scores must cover exactly the ranked queries")

    reranked: dict[str, dict[str, float]] = {}
    for query_id, first_stage_scores in first_stage_rankings.items():
        if not first_stage_scores:
            raise ValueError(f"ranking has no candidates for query: {query_id}")
        candidate_ids = set(first_stage_scores)
        scores = reranker_scores[query_id]
        if candidate_ids != set(scores):
            raise ValueError(
                f"reranker scores must cover exactly the candidates for query: {query_id}"
            )
        for corpus_id, value in scores.items():
            if not np.isfinite(float(value)):
                raise ValueError("reranker scores must be finite")
        ordered_ids = sorted(
            scores,
            key=lambda corpus_id: (-float(scores[corpus_id]), str(corpus_id)),
        )
        reranked[query_id] = {
            corpus_id: float(scores[corpus_id]) for corpus_id in ordered_ids
        }
    return reranked


def candidate_pool_preserved(
    first_stage_rankings: Mapping[str, Mapping[str, float]],
    reranked_rankings: Mapping[str, Mapping[str, float]],
) -> bool:
    """Return whether reranking preserved every query and candidate ID."""

    if set(first_stage_rankings) != set(reranked_rankings):
        return False
    return all(
        set(first_stage_rankings[query_id]) == set(reranked_rankings[query_id])
        for query_id in first_stage_rankings
    )


class CrossEncoderReranker:
    """Thin wrapper around the declared Sentence Transformers cross-encoder."""

    def __init__(self, config: RerankConfig):
        # Transformers may discover an unrelated TensorFlow/Keras installation
        # even though this experiment is PyTorch-only.  Keep model imports
        # deterministic in mixed notebook runtimes.
        os.environ.setdefault("USE_TF", "0")
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for the cross-encoder reranker"
            ) from exc

        self.config = config
        self.device = choose_device(config)
        self.model = CrossEncoder(
            config.reranker_id,
            revision=config.reranker_revision,
            max_length=config.reranker_max_seq_length,
            device=self.device,
        )

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        if not pairs:
            raise ValueError("cannot score an empty pair collection")
        scores = self.model.predict(
            list(pairs),
            batch_size=self.config.reranker_batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        return _validate_scores(scores, len(pairs))


def _system_result(
    *,
    system_id: str,
    config: RerankConfig,
    metric: Mapping[str, Any],
    run_data: RunData,
    reranker: bool,
    candidate_pool_was_preserved: bool,
) -> dict[str, Any]:
    return {
        "system_id": system_id,
        "system": {
            "first_stage_retriever": config.model_id,
            "query_expansion": None,
            "reranker": config.reranker_id if reranker else None,
        },
        "metric": "nDCG@10",
        "ndcg_at_10": float(metric["ndcg_at_10"]),
        "evaluator": dict(metric),
        "query_count": len(run_data.queries),
        "corpus_count": len(run_data.corpus),
        "qrels_query_count": len(run_data.qrels),
        "qrels_judgment_count": sum(len(rels) for rels in run_data.qrels.values()),
        "candidate_depth": config.candidate_depth,
        "candidate_pool_preserved": candidate_pool_was_preserved,
        "model": {
            "id": config.reranker_id if reranker else config.model_id,
            "revision": (
                config.reranker_revision if reranker else config.model_revision
            ),
            "max_seq_length": (
                config.reranker_max_seq_length if reranker else config.max_seq_length
            ),
            "batch_size": (
                config.reranker_batch_size if reranker else config.batch_size
            ),
            "device": choose_device(config),
            **(
                {
                    "query_prefix": config.query_prefix,
                    "passage_prefix": config.passage_prefix,
                    "normalize_embeddings": config.normalize_embeddings,
                }
                if not reranker
                else {}
            ),
        },
    }


def build_comparison_result(
    config: RerankConfig,
    data: CosQAData,
    run_data: RunData,
    baseline_metric: Mapping[str, Any],
    reranked_metric: Mapping[str, Any],
    *,
    identity: str,
    environment: Mapping[str, Any],
    timings: Mapping[str, float],
    candidate_pool_was_preserved: bool,
    notebook_sha256: str | None = None,
    cache_paths: Mapping[str, str] | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    if not candidate_pool_was_preserved:
        raise ValueError("cannot serialize a comparison with a changed candidate pool")
    status = "smoke" if config.run_mode == "smoke" else "benchmark"
    baseline = _system_result(
        system_id="e5",
        config=config,
        metric=baseline_metric,
        run_data=run_data,
        reranker=False,
        candidate_pool_was_preserved=True,
    )
    reranked = _system_result(
        system_id="e5_rerank",
        config=config,
        metric=reranked_metric,
        run_data=run_data,
        reranker=True,
        candidate_pool_was_preserved=candidate_pool_was_preserved,
    )
    return {
        "experiment_id": "e5_rerank",
        "status": status,
        "benchmark_evidence": status == "benchmark",
        "systems": {"e5": baseline, "e5_rerank": reranked},
        "delta_ndcg_at_10": reranked["ndcg_at_10"] - baseline["ndcg_at_10"],
        "candidate_pool": {
            "depth": config.candidate_depth,
            "preserved": candidate_pool_was_preserved,
        },
        "dataset": {
            "id": config.dataset_id,
            "revision": config.dataset_revision,
            "configs": {
                "queries": config.queries_config,
                "corpus": config.corpus_config,
                "qrels": config.qrels_config,
            },
            "splits": {
                "queries": config.queries_split,
                "corpus": config.corpus_split,
                "qrels": config.qrels_split,
            },
        },
        "data_schema": data.schema,
        "query_count": len(run_data.queries),
        "corpus_count": len(run_data.corpus),
        "qrels_query_count": len(run_data.qrels),
        "qrels_judgment_count": sum(len(rels) for rels in run_data.qrels.values()),
        "exclusions": run_data.exclusions,
        "seed": config.seed,
        "run_mode": config.run_mode,
        "cache_identity": identity,
        "code_version": CODE_VERSION,
        "git_commit": git_commit(repo_root),
        "notebook_sha256": notebook_sha256,
        "environment": dict(environment),
        "timings_seconds": dict(timings),
        "cache_paths": dict(cache_paths or {}),
        "limitations": (
            [
                "Smoke subset is wiring evidence only; do not compare its scores as benchmark results."
            ]
            if status == "smoke"
            else []
        ),
        "artifact_provenance": {
            "generated_at_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),
            "source": "code-retrieval/notebooks/e5_rerank_experiment.ipynb",
            "real_model_inference": True,
            "synthetic_scores": False,
        },
    }


__all__ = [
    "CODE_VERSION",
    "CrossEncoderReranker",
    "DEFAULT_RERANKER_ID",
    "DEFAULT_RERANKER_REVISION",
    "RerankConfig",
    "build_candidate_pairs",
    "build_comparison_result",
    "candidate_pool_preserved",
    "environment_metadata",
    "evaluate_ndcg_at_10",
    "expected_rerank_cache_metadata",
    "load_valid_embedding_cache",
    "load_valid_json_cache",
    "rank_with_faiss",
    "rerank_cache_paths",
    "rerank_rankings",
    "rerank_run_identity",
    "save_embedding_cache",
    "save_json_cache",
    "score_candidate_pool",
    "set_seed",
    "write_result_artifacts",
]
