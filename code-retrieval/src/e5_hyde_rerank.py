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
import subprocess
import time
import unicodedata
from dataclasses import asdict, dataclass, replace
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
    RANKING_ORDER_POLICY,
)


CODE_VERSION = "e5-hyde-rerank-v15-reproducible-selection"
SYSTEM_IDS = ["e5", "e5_hyde", "e5_rerank", "e5_hyde_rerank"]
DEFAULT_RRF_K = 60
DEFAULT_HYDE_PROMPT = (
    "Answer this programming question with a concise solution: {query}"
)
DEFAULT_HYDE_FALLBACK_PROMPTS = (
    "Provide a short answer to the following: {query}",
    "Python solution: {query}",
    "Write code for: {query}",
    "Give a Python example for: {query}",
)


def experiment_notebook_sha256(path: str | Path) -> str:
    """Hash experiment cells while excluding tagged posthoc analysis cells."""

    notebook = json.loads(Path(path).read_text(encoding="utf-8"))
    cells = [
        cell
        for cell in notebook.get("cells", [])
        if "posthoc-analysis" not in cell.get("metadata", {}).get("tags", [])
    ]
    payload = json.dumps(
        cells,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ExperimentConfig(BaselineConfig):
    """Explicit controls shared by all four comparison systems."""

    candidate_depth: int = 1000
    batch_size: int = 128
    cache_dir: str = "artifacts/e5_hyde_rerank/cache"
    artifact_dir: str = "artifacts/e5_hyde_rerank"
    hyde_model_id: str = "google/flan-t5-base"
    hyde_model_revision: str = "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2"
    hyde_prompt: str = DEFAULT_HYDE_PROMPT
    hyde_fallback_prompts: tuple[str, ...] = DEFAULT_HYDE_FALLBACK_PROMPTS
    hyde_num_hypotheses: int = 1
    hyde_temperature: float = 1.0
    hyde_max_new_tokens: int = 64
    hyde_do_sample: bool = False
    hyde_stop_behavior: str = "eos_token"
    hyde_combination_strategy: str = "query_plus_hypotheses"
    reranker_model_id: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    reranker_model_revision: str = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
    reranker_batch_size: int = 128
    reranker_max_length: int = 512
    reranker_device: str = "cpu"
    rrf_k: int = DEFAULT_RRF_K
    hyde_fusion_weights: tuple[float, float] = (0.65, 0.35)
    reranker_fusion_weights: tuple[float, float] = (0.8, 0.2)
    combined_fusion_weights: tuple[float, float] = (0.75, 0.25)

    def as_dict(self) -> dict[str, Any]:
        """Return cache metadata that is stable across JSON round-trips."""

        values = asdict(self)
        values["hyde_fallback_prompts"] = list(self.hyde_fallback_prompts)
        values["hyde_fusion_weights"] = list(self.hyde_fusion_weights)
        values["reranker_fusion_weights"] = list(self.reranker_fusion_weights)
        values["combined_fusion_weights"] = list(self.combined_fusion_weights)
        return values

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.hyde_num_hypotheses <= 0:
            raise ValueError("hyde_num_hypotheses must be positive")
        if self.hyde_temperature <= 0:
            raise ValueError("hyde_temperature must be positive")
        if self.hyde_max_new_tokens <= 0:
            raise ValueError("hyde_max_new_tokens must be positive")
        if not self.hyde_fallback_prompts:
            raise ValueError("hyde_fallback_prompts must not be empty")
        if self.hyde_combination_strategy not in {"hypothesis_only", "query_plus_hypotheses"}:
            raise ValueError(
                "hyde_combination_strategy must be 'hypothesis_only' or "
                "'query_plus_hypotheses'"
            )
        if self.reranker_batch_size <= 0 or self.reranker_max_length <= 0:
            raise ValueError("reranker_batch_size and reranker_max_length must be positive")
        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        for name, weights in (
            ("hyde_fusion_weights", self.hyde_fusion_weights),
            ("reranker_fusion_weights", self.reranker_fusion_weights),
            ("combined_fusion_weights", self.combined_fusion_weights),
        ):
            if len(weights) != 2 or any(weight < 0 for weight in weights) or not any(weights):
                raise ValueError(f"{name} must contain two non-negative weights, one positive")


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
    hyde_fused: Path
    hyde_fused_metadata: Path
    rerank_fused: Path
    rerank_fused_metadata: Path
    combined_fused: Path
    combined_fused_metadata: Path


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
        self.fallback_query_ids: list[str] = []
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

    def _generate_batch(self, batch: Sequence[tuple[str, str]], *, sample: bool) -> list[list[str]]:
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
        return [
            [
                value.strip()
                for value in decoded[
                    index * self.config.hyde_num_hypotheses : (index + 1)
                    * self.config.hyde_num_hypotheses
                ]
                if value.strip()
            ]
            for index in range(len(batch))
        ]

    def generate(self, queries: Mapping[str, str]) -> dict[str, list[str]]:
        prompts = self._prompts(queries)
        generated: dict[str, list[str]] = {}
        fallback_ids: list[str] = []
        sample = self.config.hyde_do_sample or self.config.hyde_num_hypotheses > 1
        for start in range(0, len(prompts), self.config.batch_size):
            batch = prompts[start : start + self.config.batch_size]
            decoded_values = self._generate_batch(batch, sample=sample)
            for (query_id, _), clean in zip(batch, decoded_values):
                if len(clean) != self.config.hyde_num_hypotheses:
                    fallback_ids.append(query_id)
                else:
                    generated[query_id] = clean

        pending_fallback_ids = list(fallback_ids)
        for fallback_prompt in self.config.hyde_fallback_prompts:
            if not pending_fallback_ids:
                break
            next_pending_ids: list[str] = []
            fallback_prompts = [
                (query_id, fallback_prompt.format(query=queries[query_id]))
                for query_id in pending_fallback_ids
            ]
            for start in range(0, len(fallback_prompts), self.config.batch_size):
                batch = fallback_prompts[start : start + self.config.batch_size]
                decoded_values = self._generate_batch(batch, sample=sample)
                for (query_id, _), clean in zip(batch, decoded_values):
                    if len(clean) != self.config.hyde_num_hypotheses:
                        next_pending_ids.append(query_id)
                    else:
                        generated[query_id] = clean
                        self.fallback_query_ids.append(query_id)
            pending_fallback_ids = next_pending_ids
        if pending_fallback_ids:
            raise RuntimeError(f"HyDE generator returned empty text for query {pending_fallback_ids[0]}")
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

    candidate_groups: list[tuple[str, list[str]]] = []
    pairs: list[tuple[str, str]] = []
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
        candidate_groups.append((query_id, candidate_ids))
        pairs.extend((query_representations[query_id], passage) for passage in passages)

    if not pairs:
        raise ValueError("cannot score an empty candidate collection")
    all_scores = np.asarray(score_pairs(pairs), dtype=np.float32).reshape(-1)
    if all_scores.shape[0] != len(pairs) or not np.isfinite(all_scores).all():
        raise ValueError("invalid reranker scores")

    reranked: dict[str, dict[str, float]] = {}
    offset = 0
    for query_id, candidate_ids in candidate_groups:
        scores = all_scores[offset : offset + len(candidate_ids)]
        offset += len(candidate_ids)
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


def reciprocal_rank_fusion(
    rankings: Sequence[Mapping[str, Mapping[str, float]]],
    *,
    weights: Sequence[float],
    top_k: int,
    rrf_k: int = DEFAULT_RRF_K,
) -> dict[str, dict[str, float]]:
    """Fuse ranked candidate lists while preserving deterministic source order.

    Each input ranking is a Faiss- or reranker-ordered mapping. The returned
    scores are unique enough for the official evaluator to retain the explicit
    fused order, including when the first-stage model produced tied scores.
    """

    if not rankings:
        raise ValueError("at least one ranking collection is required")
    if len(rankings) != len(weights):
        raise ValueError("weights must contain one value per ranking collection")
    if top_k <= 0 or rrf_k <= 0:
        raise ValueError("top_k and rrf_k must be positive")
    if any(weight < 0 for weight in weights) or not any(weights):
        raise ValueError("weights must be non-negative and contain a positive value")

    query_ids = list(rankings[0])
    expected_query_ids = set(query_ids)
    if any(set(ranking) != expected_query_ids for ranking in rankings[1:]):
        raise ValueError("ranking collections must cover the same query IDs")

    fused: dict[str, dict[str, float]] = {}
    for query_id in query_ids:
        scores: dict[str, float] = {}
        tie_break: dict[str, tuple[int, int]] = {}
        for source_index, (ranking, weight) in enumerate(zip(rankings, weights)):
            if weight == 0:
                continue
            source_items = list(ranking[query_id].items())
            if any(not np.isfinite(float(score)) for _, score in source_items):
                raise ValueError("ranking scores must be finite")
            # Scores recover each source rank, and Python's stable sort keeps
            # Faiss's order for equal-score ties across a cache round-trip.
            source_items.sort(key=lambda item: -float(item[1]))
            for rank, (document_id, _) in enumerate(source_items, start=1):
                scores[document_id] = scores.get(document_id, 0.0) + float(weight) / (rrf_k + rank)
                tie_break.setdefault(document_id, (source_index, rank))
        ordered_ids = sorted(
            scores,
            key=lambda document_id: (-scores[document_id], tie_break[document_id]),
        )[:top_k]
        fused[query_id] = {document_id: scores[document_id] for document_id in ordered_ids}
    return fused


def experiment_identity(
    config: ExperimentConfig,
    revisions: ModelRevisions,
    *,
    repo_root: Path | None = None,
    notebook_sha256: str | None = None,
    environment: Mapping[str, Any] | None = None,
) -> str:
    execution_environment = dict(
        environment or environment_metadata(config, repo_root=repo_root)
    )
    payload = {
        "code_version": CODE_VERSION,
        "config": config.as_dict(),
        "model_revisions": revisions.as_dict(),
        "git_commit": git_commit(repo_root),
        "notebook_sha256": notebook_sha256,
        "execution_environment": execution_environment,
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
        hyde_fused=root / "e5_hyde_fused.json",
        hyde_fused_metadata=root / "e5_hyde_fused.metadata.json",
        rerank_fused=root / "e5_rerank_fused.json",
        rerank_fused_metadata=root / "e5_rerank_fused.metadata.json",
        combined_fused=root / "e5_hyde_rerank_fused.json",
        combined_fused_metadata=root / "e5_hyde_rerank_fused.metadata.json",
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
        "execution_environment": environment_metadata(config, repo_root=repo_root),
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


def candidate_contract(rankings: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    payload = [
        [str(query_id), list(candidate_scores)]
        for query_id, candidate_scores in rankings.items()
    ]
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    counts = [len(candidate_scores) for candidate_scores in rankings.values()]
    return {
        "query_count": len(rankings),
        "minimum_candidates_per_query": min(counts, default=0),
        "maximum_candidates_per_query": max(counts, default=0),
        "ordered_candidate_ids_sha256": digest,
    }


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_query_text(value: str) -> str:
    """Normalize query text for split-overlap checks without storing raw text."""

    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def query_population_fingerprint(queries: Mapping[str, str]) -> dict[str, Any]:
    id_text_hashes = [
        [str(query_id), hashlib.sha256(normalized_query_text(text).encode("utf-8")).hexdigest()]
        for query_id, text in queries.items()
    ]
    groups: dict[str, list[str]] = {}
    for query_id, text_hash in id_text_hashes:
        groups.setdefault(text_hash, []).append(query_id)
    ordered_groups = [
        {"normalized_text_sha256": text_hash, "query_ids": sorted(query_ids)}
        for text_hash, query_ids in sorted(groups.items())
    ]
    ordered_id_hashes = sorted(id_text_hashes)
    query_ids = [str(query_id) for query_id in queries]
    return {
        "query_count": len(query_ids),
        "query_ids": query_ids,
        "query_ids_sha256": sha256_ids(query_ids),
        "query_text_hashes": ordered_id_hashes,
        "query_text_hashes_sha256": _canonical_sha256(ordered_id_hashes),
        "query_group_count": len(ordered_groups),
        "query_groups": ordered_groups,
        "query_groups_sha256": _canonical_sha256(ordered_groups),
    }


def qrels_fingerprint(qrels: Mapping[str, Mapping[str, int]]) -> dict[str, Any]:
    records = [
        [str(query_id), str(document_id), int(grade)]
        for query_id, judgments in qrels.items()
        for document_id, grade in judgments.items()
    ]
    records.sort()
    query_ids = [str(query_id) for query_id in qrels]
    return {
        "qrels_query_count": len(query_ids),
        "qrels_query_ids_sha256": sha256_ids(query_ids),
        "qrels_judgment_count": len(records),
        "qrels_sha256": _canonical_sha256(records),
    }


def corpus_fingerprint(corpus: Mapping[str, Mapping[str, str]]) -> dict[str, Any]:
    ids = [str(document_id) for document_id in corpus]
    content = [
        [str(document_id), hashlib.sha256(str(record["text"]).encode("utf-8")).hexdigest()]
        for document_id, record in corpus.items()
    ]
    return {
        "corpus_count": len(ids),
        "corpus_ids_sha256": sha256_ids(ids),
        "corpus_text_sha256": _canonical_sha256(sorted(content)),
    }


def _selection_grid() -> list[float]:
    return [round(index / 20, 2) for index in range(21)]


def _selection_controls(config: ExperimentConfig) -> dict[str, Any]:
    """Return controls that affect selection and final evaluation, excluding paths."""

    return {
        "dataset": {
            "id": config.dataset_id,
            "revision": config.dataset_revision,
            "queries_config": config.queries_config,
            "queries_split": config.queries_split,
            "corpus_config": config.corpus_config,
            "corpus_split": config.corpus_split,
            "qrels_config": config.qrels_config,
            "selection_qrels_split": "valid",
            "evaluation_qrels_split": "test",
        },
        "e5": {
            "model_id": config.model_id,
            "model_revision": config.model_revision,
            "query_prefix": config.query_prefix,
            "passage_prefix": config.passage_prefix,
            "max_seq_length": config.max_seq_length,
            "normalize_embeddings": config.normalize_embeddings,
            "batch_size": config.batch_size,
            "device": config.device,
        },
        "hyde": {
            "model_id": config.hyde_model_id,
            "model_revision": config.hyde_model_revision,
            "prompt": config.hyde_prompt,
            "fallback_prompts": list(config.hyde_fallback_prompts),
            "num_hypotheses": config.hyde_num_hypotheses,
            "temperature": config.hyde_temperature,
            "max_new_tokens": config.hyde_max_new_tokens,
            "do_sample": config.hyde_do_sample,
            "stop_behavior": config.hyde_stop_behavior,
            "combination_strategy": config.hyde_combination_strategy,
        },
        "reranker": {
            "model_id": config.reranker_model_id,
            "model_revision": config.reranker_model_revision,
            "batch_size": config.reranker_batch_size,
            "max_length": config.reranker_max_length,
            "device": config.reranker_device,
        },
        "runtime_controls": {
            "run_mode": config.run_mode,
            "seed": config.seed,
            "candidate_depth": config.candidate_depth,
            "rrf_k": config.rrf_k,
            "evaluator_package": config.evaluator_package,
            "evaluator": "coir.beir.retrieval.evaluation.EvaluateRetrieval",
            "evaluation_order_policy": RANKING_ORDER_POLICY,
        },
        "selection": {
            "metric": "nDCG@10",
            "weight_grid": _selection_grid(),
            "tie_break_policy": "prefer_the_larger_first_input_weight_after_rounding_nDCG_at_10_to_5_decimals",
            "rrf_k": config.rrf_k,
            "candidate_depth": config.candidate_depth,
        },
    }


def _selection_runtime_fingerprint(environment: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "python",
        "platform",
        "processor",
        "device",
        "packages",
        "torch_version",
        "torch_num_threads",
        "torch_deterministic_algorithms",
        "torch_float32_matmul_precision",
        "torch_cuda_matmul_allow_tf32",
        "torch_default_dtype",
        "cuda_available",
        "cuda_device_count",
        "cuda_device_name",
    )
    return {key: environment.get(key) for key in keys}


def _experiment_source_hashes(
    repo_root: Path,
    *,
    notebook_sha256: str | None,
    source_notebook_sha256: str | None,
) -> dict[str, str | None]:
    return {
        "e5_hyde_rerank_source_sha256": sha256_file(
            repo_root / "code-retrieval" / "src" / "e5_hyde_rerank.py"
        ),
        "e5_baseline_source_sha256": sha256_file(
            repo_root / "code-retrieval" / "src" / "e5_baseline.py"
        ),
        "experiment_notebook_sha256": notebook_sha256,
        "notebook_file_sha256": source_notebook_sha256,
    }


def _manifest_body(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in manifest.items()
        if key != "manifest_sha256" and not str(key).startswith("_")
    }


def seal_selection_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    sealed = dict(_manifest_body(manifest))
    sealed["manifest_sha256"] = _canonical_sha256(sealed)
    return sealed


def verify_selection_manifest_integrity(manifest: Mapping[str, Any]) -> str:
    recorded = manifest.get("manifest_sha256")
    actual = _canonical_sha256(_manifest_body(manifest))
    if not isinstance(recorded, str) or recorded != actual:
        raise ValueError("selection manifest integrity digest is missing or does not match")
    return actual


def validate_selection_manifest(
    path: str | Path,
    config: ExperimentConfig,
    *,
    repo_root: Path,
    notebook_sha256: str | None,
    source_notebook_sha256: str | None,
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed on a stale or incompatible manifest before model loading."""

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"required validation-selection manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("selection manifest must be a JSON object")
    verify_selection_manifest_integrity(manifest)
    if manifest.get("schema") != "azh-527-validation-selection-v1":
        raise ValueError("unsupported validation-selection manifest schema")
    if manifest.get("code_version") != CODE_VERSION:
        raise ValueError("selection manifest was produced by a different experiment code version")
    if manifest.get("selection_qrels_split") != "valid":
        raise ValueError("test evaluation requires a manifest selected on valid qrels")
    if manifest.get("target_evaluation_qrels_split") != "test":
        raise ValueError("selection manifest is not frozen for test evaluation")
    if manifest.get("status") != "complete":
        raise ValueError("selection manifest is not complete")
    if manifest.get("baseline_parity", {}).get("passed") is not True:
        raise ValueError("validation baseline parity failed; refusing final evaluation")
    if config.run_mode != "benchmark" or config.qrels_split != "test":
        raise ValueError("frozen selection manifests may only be used for benchmark test evaluation")
    if manifest.get("controls") != _selection_controls(config):
        raise ValueError("selection manifest controls do not match the current experiment configuration")
    if manifest.get("model_revisions") != {
        "e5": config.model_revision,
        "hyde": config.hyde_model_revision,
        "reranker": config.reranker_model_revision,
    }:
        raise ValueError("selection manifest model revisions do not match the current configuration")
    if manifest.get("runtime") != _selection_runtime_fingerprint(environment):
        raise ValueError("selection manifest runtime does not match the current execution environment")
    if manifest.get("source_hashes") != _experiment_source_hashes(
        repo_root,
        notebook_sha256=notebook_sha256,
        source_notebook_sha256=source_notebook_sha256,
    ):
        raise ValueError("selection manifest source or notebook hash does not match this checkout")
    if manifest.get("git_commit") != git_commit(repo_root):
        raise ValueError("selection manifest was produced from a different Git commit")
    if manifest.get("git_worktree_clean") is not True:
        raise ValueError("selection manifest was not produced from a clean committed worktree")
    return manifest


def validate_test_population_boundary(
    manifest: Mapping[str, Any],
    run_data: RunData,
) -> dict[str, Any]:
    """Check query ID, normalized-text, and corpus separation before inference."""

    validation = manifest.get("selection_population", {})
    test_queries = query_population_fingerprint(run_data.queries)
    test_corpus = corpus_fingerprint(run_data.corpus)
    validation_ids = set(validation.get("queries", {}).get("query_ids", []))
    test_ids = set(test_queries["query_ids"])
    validation_texts = {
        group["normalized_text_sha256"]
        for group in validation.get("queries", {}).get("query_groups", [])
    }
    test_texts = {
        group["normalized_text_sha256"] for group in test_queries["query_groups"]
    }
    if validation_ids & test_ids:
        raise ValueError("validation and test query IDs overlap")
    if validation_texts & test_texts:
        raise ValueError("validation and test normalized query text overlaps")
    if validation.get("corpus") != test_corpus:
        raise ValueError("test corpus does not match the validation-selection corpus")
    if len(test_queries["query_ids"]) != 500 or len(run_data.qrels) != 500:
        raise ValueError("test evaluation must use the complete declared 500-query population")
    if validation.get("queries", {}).get("query_count") != 500:
        raise ValueError("selection manifest does not contain the complete 500-query valid population")
    return {
        "verified": True,
        "validation_query_count": len(validation_ids),
        "test_query_count": len(test_ids),
        "query_id_overlap_count": 0,
        "normalized_query_overlap_count": 0,
        "validation_query_ids_sha256": validation["queries"]["query_ids_sha256"],
        "test_query_ids_sha256": test_queries["query_ids_sha256"],
        "validation_query_groups_sha256": validation["queries"]["query_groups_sha256"],
        "test_query_groups_sha256": test_queries["query_groups_sha256"],
        "test_qrels": qrels_fingerprint(run_data.qrels),
        "test_corpus": test_corpus,
    }


def _validate_rankings(
    rankings: Mapping[str, Mapping[str, float]],
    run_data: RunData,
    *,
    label: str,
    candidate_depth: int,
) -> None:
    expected_ids = list(run_data.queries)
    if list(rankings) != expected_ids:
        raise ValueError(f"{label} rankings do not preserve the declared query order/population")
    corpus_ids = set(run_data.corpus)
    for query_id, candidate_scores in rankings.items():
        if len(candidate_scores) != candidate_depth:
            raise ValueError(
                f"{label} ranking for {query_id} contains {len(candidate_scores)} candidates; "
                f"expected {candidate_depth}"
            )
        if len(candidate_scores) != len(set(candidate_scores)):
            raise ValueError(f"{label} ranking contains duplicate candidate IDs for {query_id}")
        if not set(candidate_scores) <= corpus_ids:
            raise ValueError(f"{label} ranking includes candidates outside the declared corpus")
        if any(not np.isfinite(float(score)) for score in candidate_scores.values()):
            raise ValueError(f"{label} ranking contains non-finite scores")


def _rankings_file_sha256(rankings: Mapping[str, Mapping[str, float]]) -> str:
    return _canonical_sha256(
        [[str(query_id), [[str(doc_id), float(score)] for doc_id, score in docs.items()]]
         for query_id, docs in rankings.items()]
    )


def _write_ordered_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _select_rrf_weights(
    qrels: Mapping[str, Mapping[str, int]],
    first_rankings: Mapping[str, Mapping[str, float]],
    second_rankings: Mapping[str, Mapping[str, float]],
    *,
    rrf_k: int,
    candidate_depth: int,
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    scores: list[dict[str, Any]] = []
    rankings_by_weight: dict[float, dict[str, dict[str, float]]] = {}
    for first_weight in _selection_grid():
        weights = (first_weight, round(1.0 - first_weight, 2))
        fused = reciprocal_rank_fusion(
            [first_rankings, second_rankings],
            weights=weights,
            top_k=candidate_depth,
            rrf_k=rrf_k,
        )
        metric = evaluate_ndcg_at_10(qrels, fused, cutoff=10)
        scores.append(
            {
                "weights": list(weights),
                "ndcg_at_10": float(metric["ndcg_at_10"]),
                "selection_ndcg_at_10": round(float(metric["ndcg_at_10"]), 5),
            }
        )
        rankings_by_weight[first_weight] = fused
    selected = max(
        scores,
        key=lambda row: (row["selection_ndcg_at_10"], row["weights"][0]),
    )
    return {
        "grid": scores,
        "selected": {
            "weights": list(selected["weights"]),
            "ndcg_at_10": selected["ndcg_at_10"],
            "selection_ndcg_at_10": selected["selection_ndcg_at_10"],
        },
    }, rankings_by_weight[float(selected["weights"][0])]


def _baseline_parity(observed: float, reference: float, *, tolerance: float) -> dict[str, Any]:
    return {
        "observed_ndcg_at_10": float(observed),
        "audited_reference_ndcg_at_10": float(reference),
        "absolute_difference": abs(float(observed) - float(reference)),
        "tolerance": tolerance,
        "passed": abs(float(observed) - float(reference)) <= tolerance,
    }


def load_ranking_replay(
    sources: Mapping[str, Mapping[str, str | Path]],
    run_data: RunData,
    config: ExperimentConfig,
    *,
    split: str,
    selection_reference_path: str | Path | None = None,
    validation_reference_path: str | Path | None = None,
    test_result_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load saved real-model rankings after checking their source records."""

    required = {"e5_for_hyde", "hyde", "e5_for_rerank", "reranker"}
    if set(sources) != required:
        raise ValueError(f"ranking replay must provide exactly {sorted(required)}")
    if split != config.qrels_split or split not in {"valid", "test"}:
        raise ValueError("ranking replay split must match the configured valid or test qrels split")
    if config.run_mode != "benchmark":
        raise ValueError("historical ranking replay requires complete benchmark mode")
    expected_model_revisions = {
        "e5": config.model_revision,
        "hyde": config.hyde_model_revision,
        "reranker": config.reranker_model_revision,
    }
    if split == "valid":
        if not selection_reference_path or not validation_reference_path:
            raise ValueError("valid replay requires the historical ranking hashes and reranker validation record")
        selection_reference = json.loads(
            Path(selection_reference_path).read_text(encoding="utf-8")
        )
        validation_reference = json.loads(
            Path(validation_reference_path).read_text(encoding="utf-8")
        )
        if (
            selection_reference.get("selection_qrels_split") != "valid"
            or selection_reference.get("query_count") != 500
            or selection_reference.get("candidate_depth") != config.candidate_depth
            or selection_reference.get("real_model_inference_inputs") is not True
            or selection_reference.get("synthetic_scores") is not False
        ):
            raise ValueError("historical selection record does not describe a complete real-model valid run")
        selection_environment = selection_reference.get("environment", {})
        if selection_environment.get("model_revisions") != {
            "e5": config.model_revision,
            "hyde": config.hyde_model_revision,
        }:
            raise ValueError("historical selection record has incompatible model revisions")
        if (
            validation_reference.get("dataset") != config.dataset_id
            or validation_reference.get("dataset_revision") != config.dataset_revision
            or validation_reference.get("qrels_split") != "valid"
            or validation_reference.get("candidate_depth") != config.candidate_depth
            or validation_reference.get("query_count") != 500
            or validation_reference.get("corpus_count") != len(run_data.corpus)
            or validation_reference.get("query_ids_sha256")
            != query_population_fingerprint(run_data.queries)["query_ids_sha256"]
            or validation_reference.get("corpus_ids_sha256")
            != corpus_fingerprint(run_data.corpus)["corpus_ids_sha256"]
            or validation_reference.get("e5_revision") != config.model_revision
            or validation_reference.get("reranker_revision") != config.reranker_model_revision
            or validation_reference.get("real_model_inference_inputs") is not True
            or validation_reference.get("synthetic_scores") is not False
        ):
            raise ValueError("reranker validation record does not match the declared valid population or controls")
        historical_hash_names = {
            "e5_for_hyde": "e5_for_hyde",
            "hyde": "hyde_query_plus",
            "e5_for_rerank": "e5_for_rerank",
            "reranker": "reranker_raw",
        }
    else:
        if not test_result_path:
            raise ValueError("test replay requires its original complete-run result record")
        test_result = json.loads(Path(test_result_path).read_text(encoding="utf-8"))
        test_result_provenance = test_result.get("artifact_provenance", {})
        if (
            test_result.get("status") != "benchmark"
            or test_result.get("benchmark_evidence") is not True
            or test_result_provenance.get("real_model_inference") is not True
            or test_result_provenance.get("synthetic_scores") is not False
            or test_result.get("query_count") != 500
            or test_result.get("qrels_query_count") != 500
            or test_result.get("corpus_count") != len(run_data.corpus)
            or test_result.get("candidate_depth") != config.candidate_depth
            or test_result.get("dataset", {}).get("id") != config.dataset_id
            or test_result.get("dataset", {}).get("revision") != config.dataset_revision
            or test_result.get("dataset", {}).get("splits", {}).get("qrels") != "test"
            or test_result.get("evaluator_package") != config.evaluator_package
            or test_result.get("model", {}).get("resolved_revision") != config.model_revision
            or test_result.get("hyde", {}).get("resolved_revision") != config.hyde_model_revision
            or test_result.get("reranker", {}).get("resolved_revision") != config.reranker_model_revision
        ):
            raise ValueError("test replay result does not describe the complete pinned real-model test run")
        expected_source_identity = test_result.get("cache_identity")
        historical_hash_names = {}

    expected_ranking_ids_sha256 = sha256_ids(
        list(run_data.queries) + ["__corpus__"] + list(run_data.corpus)
    )
    rankings: dict[str, dict[str, dict[str, float]]] = {}
    file_records: list[dict[str, Any]] = []
    for label, source in sources.items():
        path = Path(source["path"])
        metadata_path = Path(source["metadata_path"])
        if not path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"ranking replay source or metadata is missing for {label}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"ranking replay source {label} must contain an object")
        if metadata.get("config"):
            source_config = metadata["config"]
            if (
                source_config.get("dataset_id") != config.dataset_id
                or source_config.get("dataset_revision") != config.dataset_revision
                or source_config.get("qrels_split") != split
                or source_config.get("candidate_depth") != config.candidate_depth
                or source_config.get("run_mode") != "benchmark"
                or source_config.get("device") != "cpu"
            ):
                raise ValueError(f"cached source controls do not match for {label}")
            source_e5_revision = source_config.get("model_revision")
            source_hyde_revision = source_config.get(
                "hyde_model_revision", source_config.get("generator_revision")
            )
            source_reranker_revision = source_config.get("reranker_model_revision")
            if source_e5_revision != config.model_revision:
                raise ValueError(f"cached E5 revision does not match for {label}")
            if source_hyde_revision is not None and source_hyde_revision != config.hyde_model_revision:
                raise ValueError(f"cached HyDE revision does not match for {label}")
            if source_reranker_revision is not None and source_reranker_revision != config.reranker_model_revision:
                raise ValueError(f"cached reranker revision does not match for {label}")
            if metadata.get("count") != len(run_data.queries) + 1 + len(run_data.corpus):
                raise ValueError(f"cached source count does not match the declared population for {label}")
            if metadata.get("ids_sha256") != expected_ranking_ids_sha256:
                raise ValueError(f"cached source IDs do not match the declared population for {label}")
            if metadata.get("model_revisions"):
                if metadata["model_revisions"] != expected_model_revisions:
                    raise ValueError(f"cached model revisions do not match for {label}")
            if split == "test" and metadata.get("identity") != expected_source_identity:
                raise ValueError(f"cached ranking identity does not match the recorded test run for {label}")
        elif split == "valid" and metadata.get("identity"):
            if metadata["identity"] != selection_reference.get("hyde_cache_identity"):
                raise ValueError(f"valid cached ranking identity does not match the historical hash record for {label}")
        expected_hash = None
        if split == "valid":
            historical_key = historical_hash_names[label]
            expected_hash = selection_reference["input_rankings"][historical_key]["sha256"]
            if label == "e5_for_rerank":
                expected_hash = validation_reference["e5_rankings_sha256"]
            elif label == "reranker":
                expected_hash = validation_reference["reranker_rankings_sha256"]
        actual_hash = sha256_file(path)
        if expected_hash and actual_hash != expected_hash:
            raise ValueError(f"saved ranking file hash does not match its prior record for {label}")
        _validate_rankings(
            raw,
            run_data,
            label=label,
            candidate_depth=config.candidate_depth,
        )
        rankings[label] = {
            str(query_id): {str(doc_id): float(score) for doc_id, score in scores.items()}
            for query_id, scores in raw.items()
        }
        file_records.append(
            {
                "label": label,
                "path": str(path),
                "sha256": actual_hash,
                "metadata_path": str(metadata_path),
                "metadata_sha256": sha256_file(metadata_path),
                "source_identity": metadata.get("identity"),
                "source_code_version": metadata.get("code_version"),
                "source_git_commit": metadata.get("git_commit"),
                "source_kind": metadata.get("kind"),
                "expected_historical_sha256": expected_hash,
            }
        )

    for query_id in run_data.queries:
        first = set(rankings["e5_for_rerank"][query_id])
        reranked = set(rankings["reranker"][query_id])
        if first != reranked:
            raise ValueError(f"reranker changed the E5 candidate pool for {query_id}")

    revisions = ModelRevisions(
        e5=config.model_revision,
        hyde=config.hyde_model_revision,
        reranker=config.reranker_model_revision,
    )
    source_result_record = None
    if split == "valid":
        source_result_record = {
            "selection_reference_path": str(Path(selection_reference_path)),
            "selection_reference_sha256": sha256_file(selection_reference_path),
            "reranker_validation_path": str(Path(validation_reference_path)),
            "reranker_validation_sha256": sha256_file(validation_reference_path),
            "source_selection_git_commit": selection_environment.get("git_commit"),
            "real_model_inference_inputs": True,
            "synthetic_scores": False,
        }
    else:
        source_result_record = {
            "test_result_path": str(Path(test_result_path)),
            "test_result_sha256": sha256_file(test_result_path),
            "source_run_id": test_result.get("run_id"),
            "source_git_commit": test_result.get("git_commit"),
            "source_notebook_sha256": test_result.get("experiment_notebook_sha256"),
            "real_model_inference_inputs": True,
            "synthetic_scores": False,
        }
        for label, result_key in (
            ("e5_for_hyde", "e5"),
            ("hyde", "e5_hyde_raw"),
            ("e5_for_rerank", "e5"),
            ("reranker", "e5_rerank_raw"),
        ):
            expected_contract = test_result.get("candidate_contract", {}).get(result_key)
            core_expected_contract = (
                {
                    key: expected_contract[key]
                    for key in (
                        "query_count",
                        "minimum_candidates_per_query",
                        "maximum_candidates_per_query",
                        "ordered_candidate_ids_sha256",
                    )
                }
                if expected_contract
                else None
            )
            if core_expected_contract and candidate_contract(rankings[label]) != core_expected_contract:
                raise ValueError(f"test result candidate contract does not match cached rankings for {label}")
    return {
        "rankings": rankings,
        "revisions": revisions,
        "source_provenance": {
            "classification": "retrospective_replay_of_saved_real_model_rankings",
            "fresh_model_inference": False,
            "real_model_inference_inputs": True,
            "synthetic_scores": False,
            "split": split,
            "files": file_records,
            "source_run_record": source_result_record,
        },
    }


def run_validation_selection(
    config: ExperimentConfig,
    data: CosQAData,
    run_data: RunData,
    rankings: Mapping[str, Mapping[str, Mapping[str, float]]],
    *,
    revisions: ModelRevisions,
    environment: Mapping[str, Any],
    repo_root: Path,
    notebook_sha256: str | None,
    source_notebook_sha256: str | None,
    ranking_provenance: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Select all fusion weights on valid qrels and persist a sealed manifest."""

    if config.run_mode != "benchmark" or config.qrels_split != "valid":
        raise ValueError("validation selection requires benchmark mode and valid qrels")
    if len(run_data.queries) != 500 or len(run_data.qrels) != 500 or len(run_data.corpus) != 20604:
        raise ValueError("validation selection requires the complete declared CosQA population")
    _validate_rankings(rankings["e5_for_hyde"], run_data, label="E5 for HyDE", candidate_depth=config.candidate_depth)
    _validate_rankings(rankings["hyde"], run_data, label="HyDE", candidate_depth=config.candidate_depth)
    _validate_rankings(rankings["e5_for_rerank"], run_data, label="E5 for reranking", candidate_depth=config.candidate_depth)
    _validate_rankings(rankings["reranker"], run_data, label="reranker", candidate_depth=config.candidate_depth)

    baseline_metric = evaluate_ndcg_at_10(run_data.qrels, rankings["e5_for_hyde"], cutoff=10)
    baseline_parity = _baseline_parity(
        baseline_metric["ndcg_at_10"],
        0.44119,
        tolerance=0.00001,
    )
    hyde_sweep, selected_hyde = _select_rrf_weights(
        run_data.qrels,
        rankings["e5_for_hyde"],
        rankings["hyde"],
        rrf_k=config.rrf_k,
        candidate_depth=config.candidate_depth,
    )
    reranker_sweep, selected_reranker = _select_rrf_weights(
        run_data.qrels,
        rankings["e5_for_rerank"],
        rankings["reranker"],
        rrf_k=config.rrf_k,
        candidate_depth=config.candidate_depth,
    )
    combined_sweep, selected_combined = _select_rrf_weights(
        run_data.qrels,
        selected_hyde,
        selected_reranker,
        rrf_k=config.rrf_k,
        candidate_depth=config.candidate_depth,
    )

    run_id = f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}-{time.time_ns()}"
    selection_root = Path(config.artifact_dir) / "selection_runs" / run_id
    selection_root.mkdir(parents=True, exist_ok=False)
    fused_paths: dict[str, Path] = {}
    fused_hashes: dict[str, str] = {}
    for name, value in (
        ("e5_hyde", selected_hyde),
        ("e5_rerank", selected_reranker),
        ("e5_hyde_rerank", selected_combined),
    ):
        path = selection_root / f"{name}_selected_rankings.json"
        _write_ordered_json(path, value)
        fused_paths[name] = path
        fused_hashes[name] = sha256_file(path)

    current_source_hashes = _experiment_source_hashes(
        repo_root,
        notebook_sha256=notebook_sha256,
        source_notebook_sha256=source_notebook_sha256,
    )
    status = "complete" if baseline_parity["passed"] else "parity_failed"
    manifest = seal_selection_manifest(
        {
            "schema": "azh-527-validation-selection-v1",
            "status": status,
            "selection_run_id": run_id,
            "selection_qrels_split": "valid",
            "target_evaluation_qrels_split": "test",
            "selection_population": {
                "queries": query_population_fingerprint(run_data.queries),
                "qrels": qrels_fingerprint(run_data.qrels),
                "corpus": corpus_fingerprint(run_data.corpus),
            },
            "controls": _selection_controls(config),
            "model_revisions": revisions.as_dict(),
            "runtime": _selection_runtime_fingerprint(environment),
            "code_version": CODE_VERSION,
            "source_hashes": current_source_hashes,
            "git_commit": git_commit(repo_root),
            "git_worktree_clean": not bool(
                subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=repo_root,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            ),
            "evaluator": {
                "name": "coir.beir.retrieval.evaluation.EvaluateRetrieval",
                "package": config.evaluator_package,
                "metric": "nDCG@10",
                "order_policy": RANKING_ORDER_POLICY,
            },
            "sweeps": {
                "e5_hyde": {
                    "input_order": ["e5_for_hyde", "hyde"],
                    **hyde_sweep,
                },
                "e5_rerank": {
                    "input_order": ["e5_for_rerank", "reranker"],
                    **reranker_sweep,
                },
                "e5_hyde_rerank": {
                    "input_order": ["selected_e5_hyde", "selected_e5_rerank"],
                    **combined_sweep,
                },
            },
            "baseline_parity": baseline_parity,
            "baseline_metric": {
                "system_id": "e5",
                "ndcg_at_10": float(baseline_metric["ndcg_at_10"]),
                "audited_valid_reference": 0.44119,
            },
            "ranking_inputs": {
                **dict(ranking_provenance),
                "ranking_content": {
                    name: {
                        "canonical_sha256": _rankings_file_sha256(value),
                        "candidate_contract": candidate_contract(value),
                    }
                    for name, value in rankings.items()
                },
            },
            "selected_rankings": {
                name: {
                    "path": str(path),
                    "sha256": fused_hashes[name],
                    "candidate_contract": candidate_contract(value),
                }
                for name, path, value in (
                    ("e5_hyde", fused_paths["e5_hyde"], selected_hyde),
                    ("e5_rerank", fused_paths["e5_rerank"], selected_reranker),
                    ("e5_hyde_rerank", fused_paths["e5_hyde_rerank"], selected_combined),
                )
            },
            "evidence_boundary": {
                "test_qrels_loaded": False,
                "test_results_previously_examined": True,
                "future_test_interpretation": "retrospective_and_exploratory_not_independent_confirmation",
                "synthetic_scores": False,
                "fresh_model_inference": bool(ranking_provenance.get("fresh_model_inference", True)),
                "real_model_inference_inputs": bool(ranking_provenance.get("real_model_inference_inputs", True)),
            },
        }
    )
    manifest_path = selection_root / "selection_manifest.json"
    _write_ordered_json(manifest_path, manifest)
    manifest_path_sha256 = sha256_file(manifest_path)
    result = {
        "kind": "validation_selection",
        "status": status,
        "benchmark_evidence": False,
        "selection_manifest_path": str(manifest_path),
        "selection_manifest_sha256": manifest["manifest_sha256"],
        "selection_manifest_file_sha256": manifest_path_sha256,
        "selection_run_id": run_id,
        "selection_qrels_split": "valid",
        "query_count": len(run_data.queries),
        "corpus_count": len(run_data.corpus),
        "qrels_query_count": len(run_data.qrels),
        "qrels_judgment_count": sum(len(rels) for rels in run_data.qrels.values()),
        "baseline_parity": baseline_parity,
        "systems": manifest["sweeps"],
        "selected_rankings": manifest["selected_rankings"],
        "ranking_inputs": dict(ranking_provenance),
        "evidence_boundary": manifest["evidence_boundary"],
    }
    result_path = selection_root / "selection_result.json"
    _write_ordered_json(result_path, result)
    paths = {
        "selection_run_dir": str(selection_root),
        "selection_manifest": str(manifest_path),
        "selection_result": str(result_path),
        "e5_hyde_selected_rankings": str(fused_paths["e5_hyde"]),
        "e5_rerank_selected_rankings": str(fused_paths["e5_rerank"]),
        "e5_hyde_rerank_selected_rankings": str(fused_paths["e5_hyde_rerank"]),
    }
    return result, paths


def run_replayed_test_evaluation(
    config: ExperimentConfig,
    data: CosQAData,
    run_data: RunData,
    ranking_replay: Mapping[str, Any],
    selection_manifest: Mapping[str, Any],
    *,
    repo_root: Path,
    notebook_sha256: str | None,
    source_notebook_sha256: str | None,
    environment: Mapping[str, Any],
    test_boundary: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Evaluate saved test rankings with the frozen validation-selected weights."""

    revisions = ranking_replay["revisions"]
    rankings = ranking_replay["rankings"]
    selected = selection_manifest["sweeps"]
    config = replace(
        config,
        hyde_fusion_weights=tuple(selected["e5_hyde"]["selected"]["weights"]),
        reranker_fusion_weights=tuple(selected["e5_rerank"]["selected"]["weights"]),
        combined_fusion_weights=tuple(selected["e5_hyde_rerank"]["selected"]["weights"]),
    )
    e5_rankings = rankings["e5_for_hyde"]
    e5_rerank_rankings = rankings["e5_for_rerank"]
    hyde_rankings = rankings["hyde"]
    reranker_rankings = rankings["reranker"]
    fused_hyde = reciprocal_rank_fusion(
        [e5_rankings, hyde_rankings],
        weights=config.hyde_fusion_weights,
        top_k=config.candidate_depth,
        rrf_k=config.rrf_k,
    )
    fused_reranker = reciprocal_rank_fusion(
        [e5_rerank_rankings, reranker_rankings],
        weights=config.reranker_fusion_weights,
        top_k=config.candidate_depth,
        rrf_k=config.rrf_k,
    )
    fused_combined = reciprocal_rank_fusion(
        [fused_hyde, fused_reranker],
        weights=config.combined_fusion_weights,
        top_k=config.candidate_depth,
        rrf_k=config.rrf_k,
    )
    evaluation_started = time.perf_counter()
    metrics: dict[str, Mapping[str, Any]] = {
        "e5": evaluate_ndcg_at_10(run_data.qrels, e5_rankings, cutoff=10),
        "e5_hyde": evaluate_ndcg_at_10(run_data.qrels, fused_hyde, cutoff=10),
        "e5_rerank": evaluate_ndcg_at_10(run_data.qrels, fused_reranker, cutoff=10),
        "e5_hyde_rerank": evaluate_ndcg_at_10(run_data.qrels, fused_combined, cutoff=10),
        "e5_hyde_raw": evaluate_ndcg_at_10(run_data.qrels, hyde_rankings, cutoff=10),
        "e5_rerank_raw": evaluate_ndcg_at_10(run_data.qrels, reranker_rankings, cutoff=10),
    }
    evaluation_seconds = time.perf_counter() - evaluation_started
    identity = experiment_identity(
        config,
        revisions,
        repo_root=repo_root,
        notebook_sha256=notebook_sha256,
        environment=environment,
    )
    result = build_comparison_result(
        config,
        data,
        run_data,
        metrics,
        identity=identity,
        revisions=revisions,
        environment=environment,
        timings={"cached_ranking_evaluation": evaluation_seconds},
        notebook_sha256=notebook_sha256,
        source_notebook_sha256=source_notebook_sha256,
        repo_root=repo_root,
        status="cache_replay",
        real_model_inference=False,
    )
    result["candidate_contract"] = {
        "e5": candidate_contract(e5_rankings),
        "e5_hyde_raw": candidate_contract(hyde_rankings),
        "e5_hyde": {**candidate_contract(fused_hyde), "pool_policy": "union_of_e5_and_hyde_top_k_truncated_to_candidate_depth"},
        "e5_rerank_raw": {**candidate_contract(reranker_rankings), "pool_policy": "same_candidate_ids_as_e5"},
        "e5_rerank": {**candidate_contract(fused_reranker), "pool_policy": "same_candidate_ids_as_e5"},
        "e5_hyde_rerank": {**candidate_contract(fused_combined), "pool_policy": "union_of_selected_component_rankings_truncated_to_candidate_depth"},
    }
    result["selection_manifest"] = {
        "path": str(selection_manifest.get("_path", "")),
        "manifest_sha256": selection_manifest["manifest_sha256"],
        "selection_run_id": selection_manifest["selection_run_id"],
        "selected_weights": {
            name: selection_manifest["sweeps"][name]["selected"]["weights"]
            for name in ("e5_hyde", "e5_rerank", "e5_hyde_rerank")
        },
    }
    result["split_boundary"] = dict(test_boundary)
    result["test_data_interpretation"] = {
        "classification": "retrospective_and_exploratory",
        "independent_confirmation": False,
        "reason": "CosQA test results had already been examined before this replay",
    }
    result["baseline_parity"] = _baseline_parity(
        metrics["e5"]["ndcg_at_10"],
        0.45876,
        tolerance=0.00001,
    )
    result["ranking_replay"] = dict(ranking_replay["source_provenance"])
    result["artifact_provenance"]["ranking_replay"] = dict(ranking_replay["source_provenance"])
    result["benchmark_evidence"] = False
    result["limitations"].append(
        {
            "type": "evidence_level",
            "reason": "Metrics were recomputed from saved real-model rankings; this run performed no fresh model inference and is retrospective cache-replay evidence.",
        }
    )
    artifacts = write_comparison_artifacts(
        config,
        result,
        per_query_metrics=_per_query_metric_rows(metrics, cache_identity=identity),
        cache_metadata={"ranking_replay": dict(ranking_replay["source_provenance"])},
    )
    return result, artifacts


def _per_query_metric_rows(
    metrics: Mapping[str, Mapping[str, Any] | None],
    *,
    cache_identity: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for system_id, metric in metrics.items():
        if metric is None:
            continue
        for query_id, values in metric.get("per_query", {}).items():
            rows.append(
                {
                    "query_id": query_id,
                    "system_id": system_id,
                    **values,
                    "cache_identity": cache_identity,
                }
            )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = [
        "system_id",
        "ndcg_at_10",
        "delta_vs_e5",
        "map_at_10",
        "delta_map_at_10",
        "recall_at_10",
        "delta_recall_at_10",
        "precision_at_10",
        "delta_precision_at_10",
        "recall_at_1000",
        "delta_recall_at_1000",
        "query_count",
        "corpus_count",
        "qrels_query_count",
        "qrels_judgment_count",
        "candidate_depth",
        "evaluator",
        "evaluator_package",
        "evaluation_order_policy",
        "status",
        "benchmark_evidence",
    ]
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    temporary.replace(path)


def _write_per_query_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = [
        "query_id",
        "system_id",
        "ndcg_at_10",
        "map_at_10",
        "recall_at_10",
        "precision_at_10",
        "recall_at_1000",
        "first_relevant_rank",
        "relevant_document_count",
        "relevant_retrieved_at_10",
        "relevant_retrieved_at_1000",
        "cache_identity",
    ]
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
    qrels_query_count: int,
    qrels_judgment_count: int,
    candidate_depth: int,
    evaluator_package: str,
    status: str,
) -> list[dict[str, Any]]:
    baseline_metric = metrics.get("e5")
    metric_names = (
        "ndcg_at_10",
        "map_at_10",
        "recall_at_10",
        "precision_at_10",
        "recall_at_1000",
    )
    baseline_values = {
        name: None if baseline_metric is None else float(baseline_metric[name])
        for name in metric_names
    }
    rows: list[dict[str, Any]] = []
    for system_id in SYSTEM_IDS:
        metric = metrics.get(system_id)
        row: dict[str, Any] = {"system_id": system_id}
        for metric_name in metric_names:
            score = None if metric is None else float(metric[metric_name])
            baseline = baseline_values[metric_name]
            row[metric_name] = score
            row[f"delta_{metric_name}"] = (
                None if score is None or baseline is None else score - baseline
            )
        row.update({
            "delta_vs_e5": row["delta_ndcg_at_10"],
            "query_count": query_count,
            "corpus_count": corpus_count,
            "qrels_query_count": qrels_query_count,
            "qrels_judgment_count": qrels_judgment_count,
            "candidate_depth": candidate_depth,
            "evaluator": "coir.beir.retrieval.evaluation.EvaluateRetrieval",
            "evaluator_package": evaluator_package,
            "evaluation_order_policy": RANKING_ORDER_POLICY,
            "status": status,
            "benchmark_evidence": status == "benchmark",
        })
        rows.append(row)
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
    source_notebook_sha256: str | None = None,
    repo_root: Path | None = None,
    status: str | None = None,
    blocker: str | None = None,
    real_model_inference: bool = True,
) -> dict[str, Any]:
    if status is None:
        status = "smoke" if config.run_mode == "smoke" else "benchmark"
    if status not in {"smoke", "benchmark", "cache_replay", "blocked"}:
        raise ValueError("status must be smoke, benchmark, cache_replay, or blocked")
    if status == "benchmark" and any(metrics.get(system_id) is None for system_id in SYSTEM_IDS):
        raise ValueError("benchmark results require all four evaluated systems")
    rows = _result_rows(
        metrics,
        query_count=len(run_data.queries),
        corpus_count=len(run_data.corpus),
        qrels_query_count=len(run_data.qrels),
        qrels_judgment_count=sum(len(rels) for rels in run_data.qrels.values()),
        candidate_depth=config.candidate_depth,
        evaluator_package=config.evaluator_package,
        status=status,
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
    if status == "cache_replay":
        limitations.append(
            {
                "type": "evidence_level",
                "reason": "metrics were recomputed from saved real-model rankings; this run performed no fresh model inference",
            }
        )
    if blocker:
        limitations.append({"type": "blocker", "reason": blocker})
    return {
        "system_ids": list(SYSTEM_IDS),
        "metric": "nDCG@10",
        "reported_metrics": ["nDCG@10", "MAP@10", "Recall@10", "Recall@1000"],
        "metric_notes": {
            "map_at_10": "With one judged relevant item per query, MAP@10 equals MRR@10.",
            "precision_at_10": "Not emphasized in the paper because with one relevant item per query P@10 equals Recall@10 divided by 10.",
            "recall_at_10": "Fraction of queries with the judged relevant item in the top 10.",
            "recall_at_1000": "Fraction of queries with the judged relevant item in the depth-1000 candidate set.",
        },
        "run_id": f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}-{time.time_ns()}-{identity}-metrics-v1",
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
            "fallback_prompts": list(config.hyde_fallback_prompts),
            "fallback_query_count": int(environment.get("hyde_fallback_query_count", 0)),
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
        "fusion": {
            "method": "weighted_reciprocal_rank_fusion",
            "rrf_k": config.rrf_k,
            "component_weights": {
                "e5_hyde": list(config.hyde_fusion_weights),
                "e5_rerank": list(config.reranker_fusion_weights),
                "e5_hyde_rerank": list(config.combined_fusion_weights),
            },
            "sources": {
                "e5_hyde": ["e5_original_query", "e5_query_plus_hyde"],
                "e5_rerank": ["e5_original_query", "e5_cross_encoder_order"],
                "e5_hyde_rerank": ["e5_hyde", "e5_rerank"],
            },
            "candidate_depth": config.candidate_depth,
            "selection_qrels_split": "valid",
            "selection_metric": "nDCG@10",
        },
        "component_diagnostics": {
            system_id: {
                **{
                    metric_name: float(metrics[system_id][metric_name])
                    for metric_name in (
                        "ndcg_at_10",
                        "map_at_10",
                        "recall_at_10",
                        "precision_at_10",
                        "recall_at_1000",
                    )
                },
                "delta_vs_e5": float(metrics[system_id]["ndcg_at_10"])
                - float(metrics["e5"]["ndcg_at_10"]),
                "delta_map_at_10": float(metrics[system_id]["map_at_10"])
                - float(metrics["e5"]["map_at_10"]),
                "delta_recall_at_10": float(metrics[system_id]["recall_at_10"])
                - float(metrics["e5"]["recall_at_10"]),
                "delta_recall_at_1000": float(metrics[system_id]["recall_at_1000"])
                - float(metrics["e5"]["recall_at_1000"]),
                "ranking_method": "unfused_component_ranking",
            }
            for system_id in ("e5_hyde_raw", "e5_rerank_raw")
            if metrics.get(system_id) is not None
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
        "notebook_sha256": source_notebook_sha256 or notebook_sha256,
        "experiment_notebook_sha256": notebook_sha256,
        "evaluator": "coir.beir.retrieval.evaluation.EvaluateRetrieval",
        "evaluator_package": config.evaluator_package,
        "evaluation_order_policy": RANKING_ORDER_POLICY,
        "artifact_provenance": {
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": "code-retrieval/notebooks/e5_hyde_rerank_experiment.ipynb",
            "notebook_sha256": source_notebook_sha256 or notebook_sha256,
            "experiment_notebook_sha256": notebook_sha256,
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
    source_notebook_sha256: str | None = None,
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
        source_notebook_sha256=source_notebook_sha256,
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
    per_query_metrics: Sequence[Mapping[str, Any]] = (),
) -> dict[str, str]:
    """Persist JSON, CSV, and metadata atomically below the configured artifact directory."""

    artifact_dir = Path(config.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    result_path = artifact_dir / "result.json"
    comparison_path = artifact_dir / "comparison.csv"
    metadata_path = artifact_dir / "metadata.json"
    identity = str(result.get("cache_identity") or "unidentified")
    run_id = str(result.get("run_id") or identity)
    run_dir = artifact_dir / "runs" / run_id
    run_result_path = run_dir / "result.json"
    run_comparison_path = run_dir / "comparison.csv"
    run_metadata_path = run_dir / "metadata.json"
    per_query_path = run_dir / "per_query_metrics.csv"
    metadata = {
        "result": dict(result),
        "cache_metadata": dict(cache_metadata or {}),
    }
    _json_write(run_result_path, dict(result))
    _write_csv(run_comparison_path, list(result.get("results", [])))
    if per_query_metrics:
        _write_per_query_csv(per_query_path, per_query_metrics)
        metadata["artifacts"] = {
            "per_query_metrics": {
                "path": str(per_query_path),
                "sha256": hashlib.sha256(per_query_path.read_bytes()).hexdigest(),
                "row_count": len(per_query_metrics),
            }
        }
    _json_write(run_metadata_path, metadata)
    _json_write(result_path, dict(result))
    _write_csv(comparison_path, list(result.get("results", [])))
    _json_write(metadata_path, metadata)
    return {
        "result": str(result_path),
        "comparison": str(comparison_path),
        "metadata": str(metadata_path),
        "run_result": str(run_result_path),
        "run_comparison": str(run_comparison_path),
        "run_metadata": str(run_metadata_path),
        "run_dir": str(run_dir),
        "per_query_metrics": str(per_query_path) if per_query_metrics else "",
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
    # Ranking dictionaries must retain Faiss/cross-encoder tie order across a
    # JSON cache round-trip; sorted keys would silently change metric results.
    save_json_cache(path, metadata_path, value, metadata, sort_keys=False)
    return value, elapsed


def run_experiment(
    config: ExperimentConfig,
    *,
    repo_root: Path,
    notebook_sha256: str | None = None,
    source_notebook_sha256: str | None = None,
    selection_manifest_path: str | Path | None = None,
    ranking_replay: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Select on valid or evaluate test using the existing ordered workflow."""

    set_seed(config.seed)
    pre_model_environment = environment_metadata(config, repo_root=repo_root)
    selection_manifest: dict[str, Any] | None = None
    test_boundary: dict[str, Any] | None = None
    if config.run_mode == "benchmark" and config.qrels_split == "test":
        if selection_manifest_path is None:
            raise ValueError(
                "benchmark test evaluation requires E5_HYDE_RERANK_SELECTION_MANIFEST "
                "from a completed valid-only selection"
            )
        selection_manifest = validate_selection_manifest(
            selection_manifest_path,
            config,
            repo_root=repo_root,
            notebook_sha256=notebook_sha256,
            source_notebook_sha256=source_notebook_sha256,
            environment=pre_model_environment,
        )
        selection_manifest["_path"] = str(Path(selection_manifest_path))
        selection = selection_manifest["sweeps"]
        config = replace(
            config,
            hyde_fusion_weights=tuple(selection["e5_hyde"]["selected"]["weights"]),
            reranker_fusion_weights=tuple(selection["e5_rerank"]["selected"]["weights"]),
            combined_fusion_weights=tuple(selection["e5_hyde_rerank"]["selected"]["weights"]),
        )
    data = load_cosqa(config)
    run_data = select_run_data(data, config)
    if config.run_mode == "benchmark" and (
        len(run_data.queries) != len(data.queries) or len(run_data.corpus) != len(data.corpus)
    ):
        raise RuntimeError("benchmark mode must use the complete declared CosQA query population and corpus")

    if selection_manifest is not None:
        test_boundary = validate_test_population_boundary(selection_manifest, run_data)
    if ranking_replay is not None:
        if config.run_mode != "benchmark":
            raise ValueError("ranking replay is available only for complete benchmark populations")
        if ranking_replay.get("split") != config.qrels_split:
            raise ValueError("ranking replay split does not match the configured qrels split")
        loaded_replay = load_ranking_replay(
            ranking_replay["sources"],
            run_data,
            config,
            split=config.qrels_split,
            selection_reference_path=ranking_replay.get("selection_reference_path"),
            validation_reference_path=ranking_replay.get("validation_reference_path"),
            test_result_path=ranking_replay.get("test_result_path"),
        )
        replay_environment = dict(pre_model_environment)
        replay_environment["model_revisions"] = loaded_replay["revisions"].as_dict()
        replay_provenance = loaded_replay["source_provenance"]
        if config.qrels_split == "valid":
            return run_validation_selection(
                config,
                data,
                run_data,
                loaded_replay["rankings"],
                revisions=loaded_replay["revisions"],
                environment=replay_environment,
                repo_root=repo_root,
                notebook_sha256=notebook_sha256,
                source_notebook_sha256=source_notebook_sha256,
                ranking_provenance=replay_provenance,
            )
        if selection_manifest is None or test_boundary is None:
            raise ValueError("test ranking replay requires a validated selection manifest")
        return run_replayed_test_evaluation(
            config,
            data,
            run_data,
            loaded_replay,
            selection_manifest,
            repo_root=repo_root,
            notebook_sha256=notebook_sha256,
            source_notebook_sha256=source_notebook_sha256,
            environment=replay_environment,
            test_boundary=test_boundary,
        )

    encoder = E5Encoder(config)
    e5_revision = _revision_for_configured_model(config.model_revision, encoder.model, "E5")
    hyde = HyDEGenerator(config)
    hyde_revision = hyde.resolved_revision
    reranker = CrossEncoderReranker(config)
    reranker_revision = reranker.resolved_revision
    revisions = ModelRevisions(e5=e5_revision, hyde=hyde_revision, reranker=reranker_revision)
    environment = environment_metadata(config, repo_root=repo_root)
    environment["model_revisions"] = revisions.as_dict()
    identity = experiment_identity(
        config,
        revisions,
        repo_root=repo_root,
        notebook_sha256=notebook_sha256,
        environment=environment,
    )
    paths = cache_paths(config, identity)
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
    environment["hyde_fallback_query_count"] = len(hyde.fallback_query_ids)
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
    if config.run_mode == "benchmark":
        for label, ranking in (
            ("E5", original_rankings),
            ("HyDE", hyde_rankings),
            ("cross-encoder", original_reranked),
        ):
            _validate_rankings(
                ranking,
                run_data,
                label=label,
                candidate_depth=config.candidate_depth,
            )
        for query_id in query_ids:
            if set(original_rankings[query_id]) != set(original_reranked[query_id]):
                raise ValueError(f"reranker changed the E5 candidate pool for {query_id}")
    if config.run_mode == "benchmark" and config.qrels_split == "valid":
        ranking_inputs = {
            "classification": "fresh_real_model_inference",
            "fresh_model_inference": True,
            "real_model_inference_inputs": True,
            "synthetic_scores": False,
            "cache_identity": identity,
            "files": [
                {
                    "label": label,
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "metadata_path": str(metadata_path),
                    "metadata_sha256": sha256_file(metadata_path),
                }
                for label, path, metadata_path in (
                    ("e5_for_hyde", paths.original_rankings, paths.original_rankings_metadata),
                    ("hyde", paths.hyde_rankings, paths.hyde_rankings_metadata),
                    ("e5_for_rerank", paths.original_rankings, paths.original_rankings_metadata),
                    ("reranker", paths.original_reranked, paths.original_reranked_metadata),
                )
            ],
        }
        return run_validation_selection(
            config,
            data,
            run_data,
            {
                "e5_for_hyde": original_rankings,
                "hyde": hyde_rankings,
                "e5_for_rerank": original_rankings,
                "reranker": original_reranked,
            },
            revisions=revisions,
            environment=environment,
            repo_root=repo_root,
            notebook_sha256=notebook_sha256,
            source_notebook_sha256=source_notebook_sha256,
            ranking_provenance=ranking_inputs,
        )
    hyde_fused_meta = cache_metadata(
        config, revisions, identity=identity, kind="e5_hyde_fused_rankings", ids=ranking_ids, repo_root=repo_root
    )
    fused_hyde, hyde_fusion_seconds = _json_or_cache(
        path=paths.hyde_fused,
        metadata_path=paths.hyde_fused_metadata,
        metadata=hyde_fused_meta,
        produce=lambda: reciprocal_rank_fusion(
            [original_rankings, hyde_rankings],
            weights=config.hyde_fusion_weights,
            top_k=config.candidate_depth,
            rrf_k=config.rrf_k,
        ),
    )
    rerank_fused_meta = cache_metadata(
        config, revisions, identity=identity, kind="e5_rerank_fused_rankings", ids=ranking_ids, repo_root=repo_root
    )
    fused_rerank, rerank_fusion_seconds = _json_or_cache(
        path=paths.rerank_fused,
        metadata_path=paths.rerank_fused_metadata,
        metadata=rerank_fused_meta,
        produce=lambda: reciprocal_rank_fusion(
            [original_rankings, original_reranked],
            weights=config.reranker_fusion_weights,
            top_k=config.candidate_depth,
            rrf_k=config.rrf_k,
        ),
    )
    combined_fused_meta = cache_metadata(
        config, revisions, identity=identity, kind="e5_hyde_rerank_fused_rankings", ids=ranking_ids, repo_root=repo_root
    )
    fused_hyde_reranked, combined_fusion_seconds = _json_or_cache(
        path=paths.combined_fused,
        metadata_path=paths.combined_fused_metadata,
        metadata=combined_fused_meta,
        produce=lambda: reciprocal_rank_fusion(
            [fused_hyde, fused_rerank],
            weights=config.combined_fusion_weights,
            top_k=config.candidate_depth,
            rrf_k=config.rrf_k,
        ),
    )
    evaluation_started = time.perf_counter()
    metric_by_system: dict[str, Mapping[str, Any]] = {
        "e5": evaluate_ndcg_at_10(run_data.qrels, original_rankings, cutoff=10),
        "e5_hyde": evaluate_ndcg_at_10(run_data.qrels, fused_hyde, cutoff=10),
        "e5_rerank": evaluate_ndcg_at_10(run_data.qrels, fused_rerank, cutoff=10),
        "e5_hyde_rerank": evaluate_ndcg_at_10(run_data.qrels, fused_hyde_reranked, cutoff=10),
        "e5_hyde_raw": evaluate_ndcg_at_10(run_data.qrels, hyde_rankings, cutoff=10),
        "e5_rerank_raw": evaluate_ndcg_at_10(run_data.qrels, original_reranked, cutoff=10),
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
        "hyde_rrf_fusion": hyde_fusion_seconds,
        "reranker_rrf_fusion": rerank_fusion_seconds,
        "combined_rrf_fusion": combined_fusion_seconds,
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
        source_notebook_sha256=source_notebook_sha256,
        repo_root=repo_root,
    )
    if selection_manifest is not None:
        result["selection_manifest"] = {
            "path": str(selection_manifest.get("_path", "")),
            "manifest_sha256": selection_manifest["manifest_sha256"],
            "selection_run_id": selection_manifest["selection_run_id"],
            "selected_weights": {
                name: selection_manifest["sweeps"][name]["selected"]["weights"]
                for name in ("e5_hyde", "e5_rerank", "e5_hyde_rerank")
            },
        }
        result["split_boundary"] = dict(test_boundary or {})
        result["test_data_interpretation"] = {
            "classification": "retrospective_and_exploratory",
            "independent_confirmation": False,
            "reason": "CosQA test results had already been examined before this evaluation",
        }
        result["baseline_parity"] = _baseline_parity(
            metric_by_system["e5"]["ndcg_at_10"],
            0.45876,
            tolerance=0.00001,
        )
    result["candidate_contract"] = {
        "e5": candidate_contract(original_rankings),
        "e5_hyde_raw": candidate_contract(hyde_rankings),
        "e5_hyde": {
            **candidate_contract(fused_hyde),
            "pool_policy": "union_of_e5_and_hyde_top_k_truncated_to_candidate_depth",
        },
        "e5_rerank_raw": {
            **candidate_contract(original_reranked),
            "pool_policy": "same_candidate_ids_as_e5",
        },
        "e5_rerank": {
            **candidate_contract(fused_rerank),
            "pool_policy": "same_candidate_ids_as_e5",
        },
        "e5_hyde_rerank": {
            **candidate_contract(fused_hyde_reranked),
            "pool_policy": "union_of_selected_component_rankings_truncated_to_candidate_depth",
        },
    }
    artifacts = write_comparison_artifacts(
        config,
        result,
        per_query_metrics=_per_query_metric_rows(
            metric_by_system,
            cache_identity=identity,
        ),
        cache_metadata={
            "corpus": corpus_meta,
            "original_queries": original_query_meta,
            "hyde_generations": hyde_meta,
            "hyde_queries": hyde_query_meta,
            "original_rankings": original_rank_meta,
            "hyde_rankings": hyde_rank_meta,
            "original_reranked": original_rerank_meta,
            "hyde_fused": hyde_fused_meta,
            "rerank_fused": rerank_fused_meta,
            "combined_fused": combined_fused_meta,
        },
    )
    return result, artifacts


__all__ = [
    "CODE_VERSION",
    "DEFAULT_HYDE_PROMPT",
    "DEFAULT_RRF_K",
    "ExperimentConfig",
    "ModelRevisions",
    "SYSTEM_IDS",
    "blocked_result",
    "build_comparison_result",
    "cache_metadata",
    "cache_paths",
    "candidate_contract",
    "combine_hypotheses",
    "experiment_identity",
    "load_ranking_replay",
    "load_cosqa",
    "run_experiment",
    "run_validation_selection",
    "rerank_candidate_rankings",
    "reciprocal_rank_fusion",
    "normalized_query_text",
    "query_population_fingerprint",
    "qrels_fingerprint",
    "select_run_data",
    "seal_selection_manifest",
    "sha256_file",
    "validate_selection_manifest",
    "validate_test_population_boundary",
    "verify_selection_manifest_integrity",
    "write_comparison_artifacts",
]
