"""Reusable components for the CosQA E5 + HyDE experiment.

The notebook remains the ordered experiment entry point.  This module keeps
HyDE configuration, real text generation, expanded-query construction, cache
validation, and result provenance testable without loading a generator model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from e5_baseline import (
    BaselineConfig,
    CosQAData,
    E5Encoder,
    RunData,
    build_result,
    canonical_json,
    choose_device,
    environment_metadata,
    git_commit,
    load_valid_json_cache,
    rank_with_faiss,
    save_json_cache,
    sha256_ids,
    sha256_text,
)


CODE_VERSION = "e5-hyde-v12-query-preserving-e5-rrf"
DEFAULT_GENERATOR_REVISION = "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2"
DEFAULT_HYDE_PROMPT = "Answer this programming question with a concise solution: {query}"
DEFAULT_HYDE_FALLBACK_PROMPTS = (
    "Provide a short answer to the following: {query}",
    "Python solution: {query}",
    "Write code for: {query}",
    "Give a Python example for: {query}",
)


@dataclass(frozen=True)
class HyDEConfig(BaselineConfig):
    """Explicit controls for one E5 + HyDE execution."""

    batch_size: int = 128
    generator_id: str = "google/flan-t5-base"
    generator_revision: str = DEFAULT_GENERATOR_REVISION
    prompt_template: str = DEFAULT_HYDE_PROMPT
    fallback_prompts: tuple[str, ...] = DEFAULT_HYDE_FALLBACK_PROMPTS
    num_hypotheses: int = 1
    temperature: float = 1.0
    do_sample: bool = False
    max_new_tokens: int = 64
    generation_batch_size: int = 128
    stop_behavior: str = "eos_token"
    combination_strategy: str = "query_plus_hypotheses"
    empty_hypothesis_behavior: str = "error"
    rrf_k: int = 60
    fusion_weights: tuple[float, float] = (0.65, 0.35)
    cache_dir: str = "artifacts/e5_hyde/cache"
    artifact_dir: str = "artifacts/e5_hyde"

    def as_dict(self) -> dict[str, Any]:
        values = super().as_dict()
        values["fallback_prompts"] = list(self.fallback_prompts)
        values["fusion_weights"] = list(self.fusion_weights)
        return values

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.generator_id or not self.generator_revision:
            raise ValueError("generator_id and generator_revision must be non-empty")
        if not self.prompt_template or "{query}" not in self.prompt_template:
            raise ValueError("prompt_template must be non-empty and contain {query}")
        if self.num_hypotheses != 1:
            raise ValueError("this experiment requires exactly one hypothesis")
        if not self.fallback_prompts or any(
            "{query}" not in prompt for prompt in self.fallback_prompts
        ):
            raise ValueError("fallback prompts must be non-empty and contain {query}")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if self.do_sample and self.temperature <= 0:
            raise ValueError("temperature must be positive when sampling is enabled")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.generation_batch_size <= 0:
            raise ValueError("generation_batch_size must be positive")
        if self.stop_behavior != "eos_token":
            raise ValueError("stop_behavior must be eos_token")
        if self.combination_strategy not in {"hypothesis_only", "query_plus_hypotheses"}:
            raise ValueError(
                "combination_strategy must be hypothesis_only or query_plus_hypotheses"
            )
        if self.empty_hypothesis_behavior != "error":
            raise ValueError(
                "empty_hypothesis_behavior must be error"
            )
        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        if (
            len(self.fusion_weights) != 2
            or any(weight < 0 for weight in self.fusion_weights)
            or not any(self.fusion_weights)
        ):
            raise ValueError("fusion_weights must contain two non-negative values, one positive")


def hyde_run_identity(
    config: HyDEConfig,
    *,
    repo_root: Path | None = None,
    notebook_sha256: str | None = None,
    environment: Mapping[str, Any] | None = None,
) -> str:
    """Hash all declared controls, code identity, and notebook identity."""

    execution_environment = dict(
        environment or environment_metadata(config, repo_root=repo_root)
    )
    payload = {
        "code_version": CODE_VERSION,
        "config": config.as_dict(),
        "git_commit": git_commit(repo_root),
        "notebook_sha256": notebook_sha256,
        "execution_environment": execution_environment,
    }
    return sha256_text(canonical_json(payload))[:16]


def hyde_cache_paths(config: HyDEConfig, identity: str) -> dict[str, Path]:
    root = Path(config.cache_dir) / identity
    return {
        "root": root,
        "hypotheses": root / "hypotheses.json",
        "hypotheses_metadata": root / "hypotheses.metadata.json",
        "expanded_queries": root / "expanded_queries.json",
        "expanded_queries_metadata": root / "expanded_queries.metadata.json",
        "corpus_embeddings": root / "corpus_embeddings.npy",
        "corpus_metadata": root / "corpus_embeddings.metadata.json",
        "query_embeddings": root / "query_embeddings.npy",
        "query_metadata": root / "query_embeddings.metadata.json",
        "original_query_embeddings": root / "original_query_embeddings.npy",
        "original_query_metadata": root / "original_query_embeddings.metadata.json",
        "rankings": root / "rankings.json",
        "rankings_metadata": root / "rankings.metadata.json",
        "original_rankings": root / "original_rankings.json",
        "original_rankings_metadata": root / "original_rankings.metadata.json",
        "fused_rankings": root / "fused_rankings.json",
        "fused_rankings_metadata": root / "fused_rankings.metadata.json",
    }


def expected_hyde_cache_metadata(
    config: HyDEConfig,
    *,
    identity: str,
    kind: str,
    ids: Sequence[str],
    repo_root: Path | None = None,
    notebook_sha256: str | None = None,
    environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return strict metadata for every HyDE and retrieval cache."""

    execution_environment = dict(
        environment or environment_metadata(config, repo_root=repo_root)
    )
    return {
        "cache_format": 1,
        "kind": kind,
        "identity": identity,
        "code_version": CODE_VERSION,
        "git_commit": git_commit(repo_root),
        "notebook_sha256": notebook_sha256,
        "execution_environment": execution_environment,
        "config": config.as_dict(),
        "ids_sha256": sha256_ids(ids),
        "count": len(ids),
    }


def build_hyde_prompts(
    queries: Mapping[str, str], prompt_template: str = DEFAULT_HYDE_PROMPT
) -> dict[str, str]:
    """Build prompts from only original query IDs and text.

    The function deliberately accepts no qrels, answers, labels, or corpus so
    the data boundary is visible at the call site.
    """

    if not queries:
        raise ValueError("queries must be non-empty")
    if "{query}" not in prompt_template:
        raise ValueError("prompt_template must contain {query}")
    prompts: dict[str, str] = {}
    for query_id, query in queries.items():
        query_id = str(query_id)
        query = str(query).strip()
        if not query:
            raise ValueError(f"query {query_id} is empty")
        if query_id in prompts:
            raise ValueError(f"duplicate query ID: {query_id}")
        prompts[query_id] = prompt_template.format(query=query)
    return prompts


def validate_hypotheses(
    queries: Mapping[str, str],
    hypotheses: Mapping[str, str],
    *,
    empty_hypothesis_behavior: str = "error",
) -> dict[str, str]:
    """Validate one hypothesis per query with an explicit empty-output policy."""

    if empty_hypothesis_behavior not in {"error", "original_query_fallback"}:
        raise ValueError(
            "empty_hypothesis_behavior must be error or original_query_fallback"
        )

    expected_ids = list(queries)
    if list(hypotheses) != expected_ids:
        raise ValueError("hypothesis IDs must match the ordered query IDs")
    normalized: dict[str, str] = {}
    for query_id in expected_ids:
        value = str(hypotheses[query_id]).strip()
        if not value:
            if empty_hypothesis_behavior == "original_query_fallback":
                value = str(queries[query_id]).strip()
            else:
                raise ValueError(f"hypothesis for query {query_id} is empty")
        normalized[query_id] = value
    return normalized


def build_expanded_queries(
    queries: Mapping[str, str],
    hypotheses: Mapping[str, str],
    *,
    strategy: str = "query_plus_hypotheses",
) -> dict[str, str]:
    """Combine original queries and generated text without changing IDs."""

    if strategy not in {"hypothesis_only", "query_plus_hypotheses"}:
        raise ValueError("unsupported HyDE combination strategy")
    normalized_hypotheses = validate_hypotheses(queries, hypotheses)
    if strategy == "hypothesis_only":
        return normalized_hypotheses
    return {
        query_id: "\n\n".join(
            [str(queries[query_id]).strip(), normalized_hypotheses[query_id]]
        )
        for query_id in queries
    }


class HyDEGenerator:
    """Thin adapter around real Transformers sequence-to-sequence inference."""

    def __init__(self, config: HyDEConfig):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.config = config
        self.device = choose_device(config)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.generator_id,
            revision=config.generator_revision,
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            config.generator_id,
            revision=config.generator_revision,
        )
        self.model.to(self.device)
        self.model.eval()
        self._torch = torch
        self.resolved_revision = (
            getattr(getattr(self.model, "config", None), "_commit_hash", None)
            or config.generator_revision
        )
        self.fallback_query_ids: list[str] = []

    def generate(self, queries: Mapping[str, str]) -> dict[str, str]:
        """Generate one deterministic, query-only hypothesis per query."""

        prompts = build_hyde_prompts(queries, self.config.prompt_template)
        query_ids = list(prompts)
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.config.max_new_tokens,
            "num_return_sequences": self.config.num_hypotheses,
            "do_sample": self.config.do_sample,
        }
        if self.config.do_sample:
            generation_kwargs["temperature"] = self.config.temperature
        hypotheses: dict[str, str] = {}
        pending_ids: list[str] = []
        for start in range(0, len(query_ids), self.config.generation_batch_size):
            batch_ids = query_ids[start : start + self.config.generation_batch_size]
            encoded = self.tokenizer(
                [prompts[query_id] for query_id in batch_ids],
                padding=True,
                truncation=True,
                max_length=self.config.max_seq_length,
                return_tensors="pt",
            )
            encoded = {name: value.to(self.device) for name, value in encoded.items()}
            with self._torch.inference_mode():
                generated = self.model.generate(**encoded, **generation_kwargs)
            decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            if len(decoded) != len(batch_ids):
                raise RuntimeError(
                    f"generator returned {len(decoded)} outputs for {len(batch_ids)} queries"
                )
            for query_id, text in zip(batch_ids, decoded):
                if text.strip():
                    hypotheses[query_id] = text.strip()
                else:
                    pending_ids.append(query_id)
        for fallback_prompt in self.config.fallback_prompts:
            if not pending_ids:
                break
            next_pending_ids: list[str] = []
            fallback_batches = [
                (query_id, fallback_prompt.format(query=queries[query_id]))
                for query_id in pending_ids
            ]
            for start in range(0, len(fallback_batches), self.config.generation_batch_size):
                batch = fallback_batches[start : start + self.config.generation_batch_size]
                encoded = self.tokenizer(
                    [prompt for _, prompt in batch],
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_seq_length,
                    return_tensors="pt",
                )
                encoded = {name: value.to(self.device) for name, value in encoded.items()}
                with self._torch.inference_mode():
                    generated = self.model.generate(**encoded, **generation_kwargs)
                decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
                if len(decoded) != len(batch):
                    raise RuntimeError(
                        f"generator returned {len(decoded)} outputs for {len(batch)} queries"
                    )
                for (query_id, _), text in zip(batch, decoded):
                    if text.strip():
                        hypotheses[query_id] = text.strip()
                        self.fallback_query_ids.append(query_id)
                    else:
                        next_pending_ids.append(query_id)
            pending_ids = next_pending_ids
        if pending_ids:
            raise RuntimeError(f"HyDE generator returned empty text for query {pending_ids[0]}")
        ordered_hypotheses = {
            query_id: hypotheses[query_id] for query_id in queries
        }
        return validate_hypotheses(
            queries,
            ordered_hypotheses,
        empty_hypothesis_behavior=self.config.empty_hypothesis_behavior,
        )


def build_hyde_result(
    config: HyDEConfig,
    data: CosQAData,
    run_data: RunData,
    metric: Mapping[str, Any],
    *,
    identity: str,
    environment: Mapping[str, Any],
    timings: Mapping[str, float],
    hypothesis_count: int,
    repo_root: Path | None = None,
    notebook_sha256: str | None = None,
) -> dict[str, Any]:
    """Build the machine-readable E5 + HyDE result contract."""

    result = build_result(
        config,
        data,
        run_data,
        metric,
        identity=identity,
        environment=environment,
        timings=timings,
        repo_root=repo_root,
    )
    result["system_id"] = "e5_hyde"
    result["system"] = {
        "first_stage_retriever": config.model_id,
        "query_expansion": "HyDE",
        "reranker": None,
    }
    result["tqe"] = {
        "method": "HyDE",
        "generator": {
            "id": config.generator_id,
            "revision": config.generator_revision,
            "resolved_revision": environment.get("model_revisions", {}).get(
                "hyde", config.generator_revision
            ),
        },
        "prompt_template": config.prompt_template,
        "fallback_prompts": list(config.fallback_prompts),
        "num_hypotheses": config.num_hypotheses,
        "temperature": config.temperature,
        "do_sample": config.do_sample,
        "max_new_tokens": config.max_new_tokens,
        "stop_behavior": config.stop_behavior,
        "combination_strategy": config.combination_strategy,
        "empty_hypothesis_behavior": config.empty_hypothesis_behavior,
        "fallback_query_count": int(environment.get("hyde_fallback_query_count", 0)),
        "hypothesis_count": hypothesis_count,
    }
    result["fusion"] = {
        "method": "weighted_reciprocal_rank_fusion",
        "sources": ["e5_original_query", "e5_query_plus_hyde"],
        "weights": list(config.fusion_weights),
        "rrf_k": config.rrf_k,
        "candidate_depth": config.candidate_depth,
        "selection_qrels_split": "valid",
        "selection_metric": "nDCG@10",
    }
    result["notebook_sha256"] = notebook_sha256
    result["code_version"] = CODE_VERSION
    result["artifact_provenance"] = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "code-retrieval/notebooks/e5_hyde_experiment.ipynb",
        "real_model_inference": True,
        "synthetic_scores": False,
        "synthetic_hypotheses": False,
    }
    return result


__all__ = [
    "CODE_VERSION",
    "DEFAULT_GENERATOR_REVISION",
    "DEFAULT_HYDE_PROMPT",
    "E5Encoder",
    "HyDEConfig",
    "HyDEGenerator",
    "build_expanded_queries",
    "build_hyde_prompts",
    "build_hyde_result",
    "environment_metadata",
    "expected_hyde_cache_metadata",
    "hyde_cache_paths",
    "hyde_run_identity",
    "load_valid_json_cache",
    "rank_with_faiss",
    "save_json_cache",
    "validate_hypotheses",
]
