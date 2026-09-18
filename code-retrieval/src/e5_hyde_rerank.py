"""Controlled E5, HyDE, and cross-encoder CosQA experiments.

The notebook is the ordered experiment entry point.  This module keeps model
loading, candidate-set invariants, cache identity, comparison reporting, and
artifact persistence reusable and testable without replacing real model calls
with fixtures or synthetic scores.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from e5_baseline import (
    BaselineConfig,
    CosQAData,
    E5Encoder,
    RunData,
    choose_device,
    environment_metadata,
    evaluate_ndcg_at_10,
    git_commit,
    load_cosqa,
    load_valid_embedding_cache,
    load_valid_json_cache,
    rank_with_faiss,
    save_embedding_cache,
    save_json_cache,
    select_run_data,
    set_seed,
    sha256_ids,
)


CODE_VERSION = "e5-hyde-rerank-v2"
SYSTEM_IDS = ["e5", "e5_hyde", "e5_rerank", "e5_hyde_rerank"]
DEFAULT_HYDE_PROMPT = (
    "Answer this programming question with a concise solution: {query}"
)


@dataclass(frozen=True)
class ExperimentConfig(BaselineConfig):
    """Explicit controls shared by all four comparison systems."""

    candidate_depth: int = 100
    batch_size: int = 32
    cache_dir: str = "artifacts/e5_hyde_rerank/cache"
    artifact_dir: str = "artifacts/e5_hyde_rerank"
    hyde_model_id: str = "google/flan-t5-base"
    hyde_model_revision: str = "main"
    hyde_prompt: str = DEFAULT_HYDE_PROMPT
    hyde_num_hypotheses: int = 1
    hyde_temperature: float = 1.0
    hyde_max_new_tokens: int = 64
    hyde_do_sample: bool = False
    hyde_stop_behavior: str = "eos_token"
    hyde_combination_strategy: str = "hypothesis_only"
    reranker_model_id: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    reranker_model_revision: str = "main"
    reranker_batch_size: int = 32
    reranker_max_length: int = 512
    reranker_device: str = "auto"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.hyde_num_hypotheses <= 0:
            raise ValueError("hyde_num_hypotheses must be positive")
        if self.hyde_temperature <= 0:
            raise ValueError("hyde_temperature must be positive")
        if self.hyde_max_new_tokens <= 0:
            raise ValueError("hyde_max_new_tokens must be positive")
        if self.hyde_combination_strategy not in {"hypothesis_only", "query_plus_hypotheses"}:
            raise ValueError(
                "hyde_combination_strategy must be 'hypothesis_only' or "
                "'query_plus_hypotheses'"
            )
        if self.reranker_batch_size <= 0 or self.reranker_max_length <= 0:
            raise ValueError("reranker_batch_size and reranker_max_length must be positive")


@dataclass(frozen=True)
class ModelRevisions:
    """Configured and resolved revisions used by cache identities."""

    e5: str
    hyde: str
    reranker: str

    def as_dict(self) -> dict[str, str]:
        return {"e5": self.e5, "hyde": self.hyde, "reranker": self.reranker}


@dataclass
class CachePaths:
    root: Path
    corpus_embeddings: Path
    corpus_metadata: Path
    original_query_embeddings: Path
    original_query_metadata: Path
    hyde_generations: Path
    hyde_generations_metadata: Path
    hyde_query_embeddings: Path
    hyde_query_metadata: Path
    original_rankings: Path
    original_rankings_metadata: Path
    hyde_rankings: Path
    hyde_rankings_metadata: Path
    original_reranked: Path
    original_reranked_metadata: Path
    hyde_reranked: Path
    hyde_reranked_metadata: Path


def resolved_model_revision(model: Any) -> str | None:
    """Find the immutable Hugging Face commit recorded on a loaded model."""

    pending: list[Any] = [model]
    seen: set[int] = set()
    while pending:
        current = pending.pop(0)
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        config = getattr(current, "config", None)
        commit = getattr(config, "_commit_hash", None)
        if commit:
            return str(commit)
        for name in ("model", "auto_model", "_first_module", "encoder"):
            child = getattr(current, name, None)
            if child is not None:
                pending.append(child)
    return None


def _require_resolved_revision(model: Any, configured: str, label: str) -> str:
    resolved = resolved_model_revision(model)
    if not resolved:
        raise RuntimeError(
            f"{label} loaded without a resolved Hugging Face revision; "
            f"configured revision was {configured!r}"
        )
    return resolved


def _revision_for_configured_model(configured: str, model: Any, label: str) -> str:
    """Use the runtime commit, falling back only for a pinned SHA."""

    resolved = resolved_model_revision(model)
    if resolved:
        return resolved
    if len(configured) == 40 and all(character in "0123456789abcdef" for character in configured.lower()):
        return configured
    raise RuntimeError(f"{label} has no observable resolved revision")


def combine_hypotheses(
    query: str,
    hypotheses: Sequence[str],
    *,
    strategy: str,
) -> str:
    """Build the E5 query-side representation without accessing labels or docs."""

    clean = [str(value).strip() for value in hypotheses if str(value).strip()]
    if not clean:
        raise ValueError("HyDE generation returned no non-empty hypotheses")
    if strategy == "hypothesis_only":
        return "\n\n".join(clean)
    if strategy == "query_plus_hypotheses":
        return f"{query.strip()}\n\n" + "\n\n".join(clean)
    raise ValueError(f"unsupported HyDE combination strategy: {strategy}")


class HyDEGenerator:
    """Local text-to-text generator used for real HyDE inference."""

    def __init__(self, config: ExperimentConfig):
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers and torch are required for HyDE generation") from exc

        self.config = config
        self.torch = torch
        self.device = choose_device(config)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.hyde_model_id,
            revision=config.hyde_model_revision,
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            config.hyde_model_id,
            revision=config.hyde_model_revision,
        )
        self.model.to(self.device)
        self.model.eval()
        self.resolved_revision = _require_resolved_revision(
            self.model, config.hyde_model_revision, "HyDE generator"
        )

    def _prompts(self, queries: Mapping[str, str]) -> list[tuple[str, str]]:
        prompts: list[tuple[str, str]] = []
        for query_id, query in queries.items():
            if not str(query).strip():
                raise ValueError(f"query {query_id} is empty")
            prompts.append((str(query_id), self.config.hyde_prompt.format(query=query)))
        return prompts

    def generate(self, queries: Mapping[str, str]) -> dict[str, list[str]]:
        prompts = self._prompts(queries)
        generated: dict[str, list[str]] = {}
        sample = self.config.hyde_do_sample or self.config.hyde_num_hypotheses > 1
        for start in range(0, len(prompts), self.config.batch_size):
            batch = prompts[start : start + self.config.batch_size]
            encoded = self.tokenizer(
                [prompt for _, prompt in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_seq_length,
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": self.config.hyde_max_new_tokens,
                "num_return_sequences": self.config.hyde_num_hypotheses,
                "do_sample": sample,
            }
            if sample:
                generation_kwargs["temperature"] = self.config.hyde_temperature
            with self.torch.no_grad():
                output = self.model.generate(**encoded, **generation_kwargs)
            decoded = self.tokenizer.batch_decode(output, skip_special_tokens=True)
            expected = len(batch) * self.config.hyde_num_hypotheses
            if len(decoded) != expected:
                raise RuntimeError(
                    f"HyDE generator returned {len(decoded)} outputs; expected {expected}"
                )
            for index, (query_id, _) in enumerate(batch):
                values = decoded[
                    index * self.config.hyde_num_hypotheses : (index + 1)
                    * self.config.hyde_num_hypotheses
                ]
                clean = [value.strip() for value in values if value.strip()]
                if len(clean) != self.config.hyde_num_hypotheses:
                    raise RuntimeError(f"HyDE generator returned empty text for query {query_id}")
                generated[query_id] = clean
        if set(generated) != set(queries):
            raise RuntimeError("HyDE output query IDs do not match the input query IDs")
        return generated

    def close(self) -> None:
        del self.model
        del self.tokenizer
        if self.device == "cuda" and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


class CrossEncoderReranker:
    """Thin wrapper around the real sentence-transformers cross-encoder."""

    def __init__(self, config: ExperimentConfig):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is required for re-ranking") from exc

        self.config = config
        self.device = choose_device(config) if config.reranker_device == "auto" else config.reranker_device
        self.model = CrossEncoder(
            config.reranker_model_id,
            revision=config.reranker_model_revision,
            max_length=config.reranker_max_length,
            device=self.device,
        )
        self.resolved_revision = _require_resolved_revision(
            self.model, config.reranker_model_revision, "cross-encoder"
        )

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        if not pairs:
            raise ValueError("cannot score an empty candidate set")
        scores = self.model.predict(
            list(pairs),
            batch_size=self.config.reranker_batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if scores.shape[0] != len(pairs) or not np.isfinite(scores).all():
            raise ValueError("cross-encoder returned invalid candidate scores")
        return scores

    def rerank(
        self,
        query_representations: Mapping[str, str],
        corpus: Mapping[str, Mapping[str, str]],
        rankings: Mapping[str, Mapping[str, float]],
    ) -> dict[str, dict[str, float]]:
        return rerank_candidate_rankings(
            rankings,
            query_representations,
            corpus,
            self.score_pairs,
        )


def rerank_candidate_rankings(
    first_stage_rankings: Mapping[str, Mapping[str, float]],
    query_representations: Mapping[str, str],
    corpus: Mapping[str, Mapping[str, str]],
    score_pairs: Callable[[Sequence[tuple[str, str]]], Sequence[float]],
) -> dict[str, dict[str, float]]:
    """Reorder exactly the first-stage candidate IDs, never add new documents."""

    reranked: dict[str, dict[str, float]] = {}
    for query_id, candidate_scores in first_stage_rankings.items():
        if query_id not in query_representations:
            raise ValueError(f"missing query representation for {query_id}")
        candidate_ids = list(candidate_scores)
        if not candidate_ids:
            raise ValueError(f"empty candidate set for {query_id}")
        try:
            passages = [corpus[doc_id]["text"] for doc_id in candidate_ids]
        except KeyError as exc:
            raise ValueError(f"candidate is missing from corpus: {exc.args[0]}") from exc
        scores = np.asarray(
            score_pairs([(query_representations[query_id], passage) for passage in passages]),
            dtype=np.float32,
        ).reshape(-1)
        if scores.shape[0] != len(candidate_ids) or not np.isfinite(scores).all():
            raise ValueError(f"invalid reranker scores for {query_id}")
        positions = sorted(
            range(len(candidate_ids)),
            key=lambda index: (-float(scores[index]), index),
        )
        reranked[query_id] = {
            candidate_ids[index]: float(scores[index]) for index in positions
        }
        if set(reranked[query_id]) != set(candidate_ids):
            raise AssertionError(f"reranker changed the candidate set for {query_id}")
    return reranked


def experiment_identity(
    config: ExperimentConfig,
    revisions: ModelRevisions,
    *,
    repo_root: Path | None = None,
) -> str:
    payload = {
        "code_version": CODE_VERSION,
        "config": config.as_dict(),
        "model_revisions": revisions.as_dict(),
        "git_commit": git_commit(repo_root),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()[:16]


def cache_paths(config: ExperimentConfig, identity: str) -> CachePaths:
    root = Path(config.cache_dir) / identity
    return CachePaths(
        root=root,
        corpus_embeddings=root / "corpus_embeddings.npy",
        corpus_metadata=root / "corpus_embeddings.metadata.json",
        original_query_embeddings=root / "original_query_embeddings.npy",
        original_query_metadata=root / "original_query_embeddings.metadata.json",
        hyde_generations=root / "hyde_generations.json",
        hyde_generations_metadata=root / "hyde_generations.metadata.json",
        hyde_query_embeddings=root / "hyde_query_embeddings.npy",
        hyde_query_metadata=root / "hyde_query_embeddings.metadata.json",
        original_rankings=root / "original_rankings.json",
        original_rankings_metadata=root / "original_rankings.metadata.json",
        hyde_rankings=root / "hyde_rankings.json",
        hyde_rankings_metadata=root / "hyde_rankings.metadata.json",
        original_reranked=root / "original_reranked.json",
        original_reranked_metadata=root / "original_reranked.metadata.json",
        hyde_reranked=root / "hyde_reranked.json",
        hyde_reranked_metadata=root / "hyde_reranked.metadata.json",
    )


def cache_metadata(
    config: ExperimentConfig,
    revisions: ModelRevisions,
    *,
    identity: str,
    kind: str,
    ids: Iterable[str],
    repo_root: Path | None = None,
    representation_sha256: str | None = None,
) -> dict[str, Any]:
    ordered_ids = list(ids)
    metadata: dict[str, Any] = {
        "cache_format": 1,
        "kind": kind,
        "identity": identity,
        "code_version": CODE_VERSION,
        "git_commit": git_commit(repo_root),
        "config": config.as_dict(),
        "model_revisions": revisions.as_dict(),
        "ids_sha256": sha256_ids(ordered_ids),
        "count": len(ordered_ids),
    }
    if representation_sha256 is not None:
        metadata["representation_sha256"] = representation_sha256
    return metadata


def _representation_hash(representations: Mapping[str, str]) -> str:
    payload = [[key, representations[key]] for key in sorted(representations)]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = ["system_id", "ndcg_at_10", "delta_vs_e5", "query_count", "corpus_count"]
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    temporary.replace(path)


def _json_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")
    temporary.replace(path)


def _result_rows(
    metrics: Mapping[str, Mapping[str, Any] | None],
    *,
    query_count: int,
    corpus_count: int,
) -> list[dict[str, Any]]:
    baseline_metric = metrics.get("e5")
    baseline = None if baseline_metric is None else float(baseline_metric["ndcg_at_10"])
    rows: list[dict[str, Any]] = []
    for system_id in SYSTEM_IDS:
        metric = metrics.get(system_id)
        score = None if metric is None else float(metric["ndcg_at_10"])
        rows.append(
            {
                "system_id": system_id,
                "ndcg_at_10": score,
                "delta_vs_e5": None if score is None or baseline is None else score - baseline,
                "query_count": query_count,
                "corpus_count": corpus_count,
            }
        )
    return rows


def build_comparison_result(
    config: ExperimentConfig,
    data: CosQAData,
    run_data: RunData,
    metrics: Mapping[str, Mapping[str, Any] | None],
    *,
    identity: str,
    revisions: ModelRevisions,
    environment: Mapping[str, Any],
    timings: Mapping[str, float],
    notebook_sha256: str | None = None,
    repo_root: Path | None = None,
    status: str | None = None,
    blocker: str | None = None,
    real_model_inference: bool = True,
) -> dict[str, Any]:
    if status is None:
        status = "smoke" if config.run_mode == "smoke" else "benchmark"
    if status not in {"smoke", "benchmark", "blocked"}:
        raise ValueError("status must be smoke, benchmark, or blocked")
    if status == "benchmark" and any(metrics.get(system_id) is None for system_id in SYSTEM_IDS):
        raise ValueError("benchmark results require all four evaluated systems")
    rows = _result_rows(
        metrics,
        query_count=len(run_data.queries),
        corpus_count=len(run_data.corpus),
    )
    exclusions = list(run_data.exclusions)
    limitations: list[dict[str, str]] = []
    if status == "smoke":
        limitations.append(
            {
                "type": "evidence_level",
                "reason": "smoke execution is wiring evidence only; do not compare its score as benchmark evidence",
            }
        )
    if blocker:
        limitations.append({"type": "blocker", "reason": blocker})
    return {
        "system_ids": list(SYSTEM_IDS),
        "metric": "nDCG@10",
        "status": status,
        "benchmark_evidence": status == "benchmark",
        "results": rows,
        "query_count": len(run_data.queries),
        "corpus_count": len(run_data.corpus),
        "qrels_query_count": len(run_data.qrels),
        "qrels_judgment_count": sum(len(rels) for rels in run_data.qrels.values()),
        "exclusions": exclusions,
        "limitations": limitations,
        "candidate_depth": config.candidate_depth,
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
            "schema": data.schema,
        },
        "model": {
            "id": config.model_id,
            "configured_revision": config.model_revision,
            "resolved_revision": revisions.e5,
            "query_prefix": config.query_prefix,
            "passage_prefix": config.passage_prefix,
            "max_seq_length": config.max_seq_length,
            "normalize_embeddings": config.normalize_embeddings,
            "batch_size": config.batch_size,
            "device": choose_device(config),
        },
        "hyde": {
            "generator_model": config.hyde_model_id,
            "configured_revision": config.hyde_model_revision,
            "resolved_revision": revisions.hyde,
            "prompt": config.hyde_prompt,
            "num_hypotheses": config.hyde_num_hypotheses,
            "temperature": config.hyde_temperature,
            "max_new_tokens": config.hyde_max_new_tokens,
            "do_sample": config.hyde_do_sample,
            "effective_sampling": config.hyde_do_sample or config.hyde_num_hypotheses > 1,
            "stop_behavior": config.hyde_stop_behavior,
            "combination_strategy": config.hyde_combination_strategy,
        },
        "reranker": {
            "model_id": config.reranker_model_id,
            "configured_revision": config.reranker_model_revision,
            "resolved_revision": revisions.reranker,
            "batch_size": config.reranker_batch_size,
            "max_length": config.reranker_max_length,
            "device": choose_device(config) if config.reranker_device == "auto" else config.reranker_device,
        },
        "seed": config.seed,
        "run_mode": config.run_mode,
        "cache_identity": identity,
        "code_version": CODE_VERSION,
        "git_commit": git_commit(repo_root),
        "environment": dict(environment),
        "timings_seconds": dict(timings),
        "cost_observations": {
            "local_model_inference": real_model_inference,
            "gpu_memory_captured": False,
            "latency_is_wall_clock_seconds": True,
        },
        "notebook_sha256": notebook_sha256,
        "evaluator": "coir.beir.retrieval.evaluation.EvaluateRetrieval",
        "evaluator_package": config.evaluator_package,
        "artifact_provenance": {
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": "code-retrieval/notebooks/e5_hyde_rerank_experiment.ipynb",
            "real_model_inference": real_model_inference,
            "synthetic_scores": False,
        },
    }


def blocked_result(
    config: ExperimentConfig,
    *,
    blocker: str,
    environment: Mapping[str, Any] | None = None,
    repo_root: Path | None = None,
    notebook_sha256: str | None = None,
    real_model_inference: bool = False,
) -> dict[str, Any]:
    """Create a non-benchmark artifact with null scores and an exact blocker."""

    empty_data = CosQAData(corpus={}, queries={}, qrels={}, schema={})
    empty_run = RunData(corpus={}, queries={}, qrels={})
    revisions = ModelRevisions(
        e5=config.model_revision,
        hyde=config.hyde_model_revision,
        reranker=config.reranker_model_revision,
    )
    result = build_comparison_result(
        config,
        empty_data,
        empty_run,
        {system_id: None for system_id in SYSTEM_IDS},
        identity="blocked",
        revisions=revisions,
        environment=dict(environment or {}),
        timings={},
        notebook_sha256=notebook_sha256,
        repo_root=repo_root,
        status="blocked",
        blocker=blocker,
        real_model_inference=real_model_inference,
    )
    result["run_mode"] = config.run_mode
    return result


def write_comparison_artifacts(
    config: ExperimentConfig,
    result: Mapping[str, Any],
    *,
    cache_metadata: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Persist JSON, CSV, and metadata atomically below the configured artifact directory."""

    artifact_dir = Path(config.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    result_path = artifact_dir / "result.json"
    comparison_path = artifact_dir / "comparison.csv"
    metadata_path = artifact_dir / "metadata.json"
    _json_write(result_path, dict(result))
    _write_csv(comparison_path, list(result.get("results", [])))
    _json_write(
        metadata_path,
        {
            "result": dict(result),
            "cache_metadata": dict(cache_metadata or {}),
        },
    )
    return {
        "result": str(result_path),
        "comparison": str(comparison_path),
        "metadata": str(metadata_path),
    }


def _embedding_or_cache(
    *,
    path: Path,
    metadata_path: Path,
    metadata: Mapping[str, Any],
    encode: Callable[[], np.ndarray],
) -> tuple[np.ndarray, float]:
    cached = load_valid_embedding_cache(path, metadata_path, metadata)
    if cached is not None:
        return cached, 0.0
    started = time.perf_counter()
    embeddings = encode()
    elapsed = time.perf_counter() - started
    save_embedding_cache(path, metadata_path, embeddings, metadata)
    return embeddings, elapsed


def _json_or_cache(
    *,
    path: Path,
    metadata_path: Path,
    metadata: Mapping[str, Any],
    produce: Callable[[], Mapping[str, Any]],
) -> tuple[dict[str, Any], float]:
    cached = load_valid_json_cache(path, metadata_path, metadata)
    if cached is not None:
        return cached, 0.0
    started = time.perf_counter()
    value = dict(produce())
    elapsed = time.perf_counter() - started
    save_json_cache(path, metadata_path, value, metadata)
    return value, elapsed


def run_experiment(
    config: ExperimentConfig,
    *,
    repo_root: Path,
    notebook_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Run all four systems with real model inference and write their artifacts."""

    set_seed(config.seed)
    data = load_cosqa(config)
    run_data = select_run_data(data, config)
    if config.run_mode == "benchmark" and (
        len(run_data.queries) != len(data.queries) or len(run_data.corpus) != len(data.corpus)
    ):
        raise RuntimeError("benchmark mode must use the complete declared CosQA test queries and corpus")

    encoder = E5Encoder(config)
    e5_revision = _revision_for_configured_model(config.model_revision, encoder.model, "E5")
    hyde = HyDEGenerator(config)
    hyde_revision = hyde.resolved_revision
    reranker = CrossEncoderReranker(config)
    reranker_revision = reranker.resolved_revision
    revisions = ModelRevisions(e5=e5_revision, hyde=hyde_revision, reranker=reranker_revision)
    identity = experiment_identity(config, revisions, repo_root=repo_root)
    paths = cache_paths(config, identity)

    environment = environment_metadata(config, repo_root=repo_root)
    environment["model_revisions"] = revisions.as_dict()
    corpus_ids = list(run_data.corpus)
    query_ids = list(run_data.queries)
    corpus_texts = [run_data.corpus[doc_id]["text"] for doc_id in corpus_ids]
    query_texts = {query_id: run_data.queries[query_id] for query_id in query_ids}
    base_ids = corpus_ids

    corpus_meta = cache_metadata(
        config, revisions, identity=identity, kind="corpus_embeddings", ids=base_ids, repo_root=repo_root
    )
    corpus_embeddings, corpus_seconds = _embedding_or_cache(
        path=paths.corpus_embeddings,
        metadata_path=paths.corpus_metadata,
        metadata=corpus_meta,
        encode=lambda: encoder.encode_corpus(corpus_texts),
    )
    original_query_meta = cache_metadata(
        config, revisions, identity=identity, kind="original_query_embeddings", ids=query_ids, repo_root=repo_root
    )
    original_query_embeddings, original_query_seconds = _embedding_or_cache(
        path=paths.original_query_embeddings,
        metadata_path=paths.original_query_metadata,
        metadata=original_query_meta,
        encode=lambda: encoder.encode_queries(list(query_texts.values())),
    )

    hyde_ids = query_ids
    hyde_meta = cache_metadata(
        config, revisions, identity=identity, kind="hyde_generations", ids=hyde_ids, repo_root=repo_root
    )
    generated, hyde_seconds = _json_or_cache(
        path=paths.hyde_generations,
        metadata_path=paths.hyde_generations_metadata,
        metadata=hyde_meta,
        produce=lambda: hyde.generate(query_texts),
    )
    hyde.close()
    hyde_representations = {
        query_id: combine_hypotheses(
            query_texts[query_id],
            [str(value) for value in generated[query_id]],
            strategy=config.hyde_combination_strategy,
        )
        for query_id in query_ids
    }
    hyde_query_meta = cache_metadata(
        config,
        revisions,
        identity=identity,
        kind="hyde_query_embeddings",
        ids=query_ids,
        repo_root=repo_root,
        representation_sha256=_representation_hash(hyde_representations),
    )
    hyde_query_embeddings, hyde_query_seconds = _embedding_or_cache(
        path=paths.hyde_query_embeddings,
        metadata_path=paths.hyde_query_metadata,
        metadata=hyde_query_meta,
        encode=lambda: encoder.encode_queries(list(hyde_representations.values())),
    )

    ranking_ids = query_ids + ["__corpus__"] + corpus_ids
    original_rank_meta = cache_metadata(
        config, revisions, identity=identity, kind="original_rankings", ids=ranking_ids, repo_root=repo_root
    )
    original_rankings, original_ranking_seconds = _json_or_cache(
        path=paths.original_rankings,
        metadata_path=paths.original_rankings_metadata,
        metadata=original_rank_meta,
        produce=lambda: rank_with_faiss(
            original_query_embeddings,
            corpus_embeddings,
            query_ids,
            corpus_ids,
            top_k=config.candidate_depth,
        ),
    )
    hyde_rank_meta = cache_metadata(
        config, revisions, identity=identity, kind="hyde_rankings", ids=ranking_ids, repo_root=repo_root
    )
    hyde_rankings, hyde_ranking_seconds = _json_or_cache(
        path=paths.hyde_rankings,
        metadata_path=paths.hyde_rankings_metadata,
        metadata=hyde_rank_meta,
        produce=lambda: rank_with_faiss(
            hyde_query_embeddings,
            corpus_embeddings,
            query_ids,
            corpus_ids,
            top_k=config.candidate_depth,
        ),
    )

    original_rerank_meta = cache_metadata(
        config,
        revisions,
        identity=identity,
        kind="original_reranked",
        ids=ranking_ids,
        repo_root=repo_root,
        representation_sha256=_representation_hash(query_texts),
    )
    original_reranked, original_rerank_seconds = _json_or_cache(
        path=paths.original_reranked,
        metadata_path=paths.original_reranked_metadata,
        metadata=original_rerank_meta,
        produce=lambda: reranker.rerank(query_texts, run_data.corpus, original_rankings),
    )
    hyde_rerank_meta = cache_metadata(
        config,
        revisions,
        identity=identity,
        kind="hyde_reranked",
        ids=ranking_ids,
        repo_root=repo_root,
        representation_sha256=_representation_hash(hyde_representations),
    )
    hyde_reranked, hyde_rerank_seconds = _json_or_cache(
        path=paths.hyde_reranked,
        metadata_path=paths.hyde_reranked_metadata,
        metadata=hyde_rerank_meta,
        produce=lambda: reranker.rerank(hyde_representations, run_data.corpus, hyde_rankings),
    )

    evaluation_started = time.perf_counter()
    metric_by_system: dict[str, Mapping[str, Any]] = {
        "e5": evaluate_ndcg_at_10(run_data.qrels, original_rankings, cutoff=10),
        "e5_hyde": evaluate_ndcg_at_10(run_data.qrels, hyde_rankings, cutoff=10),
        "e5_rerank": evaluate_ndcg_at_10(run_data.qrels, original_reranked, cutoff=10),
        "e5_hyde_rerank": evaluate_ndcg_at_10(run_data.qrels, hyde_reranked, cutoff=10),
    }
    evaluation_seconds = time.perf_counter() - evaluation_started
    timings = {
        "corpus_encoding": corpus_seconds,
        "original_query_encoding": original_query_seconds,
        "hyde_generation": hyde_seconds,
        "hyde_query_encoding": hyde_query_seconds,
        "original_ranking": original_ranking_seconds,
        "hyde_ranking": hyde_ranking_seconds,
        "original_reranking": original_rerank_seconds,
        "hyde_reranking": hyde_rerank_seconds,
        "evaluation": evaluation_seconds,
    }
    result = build_comparison_result(
        config,
        data,
        run_data,
        metric_by_system,
        identity=identity,
        revisions=revisions,
        environment=environment,
        timings=timings,
        notebook_sha256=notebook_sha256,
        repo_root=repo_root,
    )
    artifacts = write_comparison_artifacts(
        config,
        result,
        cache_metadata={
            "corpus": corpus_meta,
            "original_queries": original_query_meta,
            "hyde_generations": hyde_meta,
            "hyde_queries": hyde_query_meta,
            "original_rankings": original_rank_meta,
            "hyde_rankings": hyde_rank_meta,
            "original_reranked": original_rerank_meta,
            "hyde_reranked": hyde_rerank_meta,
        },
    )
    return result, artifacts


__all__ = [
    "CODE_VERSION",
    "DEFAULT_HYDE_PROMPT",
    "ExperimentConfig",
    "ModelRevisions",
    "SYSTEM_IDS",
    "blocked_result",
    "build_comparison_result",
    "cache_metadata",
    "cache_paths",
    "combine_hypotheses",
    "experiment_identity",
    "load_cosqa",
    "run_experiment",
    "rerank_candidate_rankings",
    "select_run_data",
    "write_comparison_artifacts",
]
