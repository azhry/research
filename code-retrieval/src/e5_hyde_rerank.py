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

from runtime_costs import RuntimeCostRecorder, observe_module_batches, synchronize_device

from e5_baseline import (
    BaselineConfig,
    build_faiss_index,
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


CODE_VERSION = "e5-hyde-rerank-v18-code-reranker"
SYSTEM_IDS = ["e5", "e5_hyde", "e5_rerank", "e5_hyde_rerank"]
CODE_RERANKER_SYSTEM_ID = "e5_code_rerank"
CODE_RERANKER_MODEL_ID = "NamanAgnih0tri/code-reranker-miniLM-staqc"
CODE_RERANKER_MODEL_REVISION = "9bcf5f1cceef4914e2fc564b2e4ee78410626702"
CODE_RERANKER_MODEL_CARD = (
    "https://huggingface.co/NamanAgnih0tri/code-reranker-miniLM-staqc"
)
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


def _code_reranker_training_context() -> dict[str, Any]:
    """Return publisher-reported provenance and the unresolved overlap boundary."""

    return {
        "model_card_url": CODE_RERANKER_MODEL_CARD,
        "model_card_claims_audited": False,
        "training_dataset": "StaQC SCA Stack Overflow question-code pairs",
        "training_language": "Python",
        "publisher_reported_training_examples": 85294,
        "publisher_reported_samples_including_negatives": 170588,
        "evaluation_dataset": "CoIR-Retrieval/cosqa",
        "domain_note": (
            "The model card describes Stack Overflow question-to-Python-code "
            "training; this study evaluates the pinned CosQA query/corpus collection."
        ),
        "training_data_overlap_with_cosqa": (
            "unknown; record-level overlap was not audited"
        ),
        "publisher_reported_performance_used_as_evidence": False,
    }


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
    """Explicit controls shared by the CosQA comparison systems."""

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
    code_reranker_enabled: bool = False
    code_reranker_model_id: str = CODE_RERANKER_MODEL_ID
    code_reranker_model_revision: str = CODE_RERANKER_MODEL_REVISION
    code_reranker_batch_size: int = 128
    code_reranker_max_length: int = 512
    code_reranker_device: str = "cpu"
    rrf_k: int = DEFAULT_RRF_K
    hyde_fusion_weights: tuple[float, float] = (0.65, 0.35)
    hypothesis_only_fusion_weights: tuple[float, float] = (0.65, 0.35)
    reranker_fusion_weights: tuple[float, float] = (0.8, 0.2)
    code_reranker_fusion_weights: tuple[float, float] = (0.8, 0.2)
    combined_fusion_weights: tuple[float, float] = (0.75, 0.25)

    def as_dict(self) -> dict[str, Any]:
        """Return cache metadata that is stable across JSON round-trips."""

        values = asdict(self)
        values["hyde_fallback_prompts"] = list(self.hyde_fallback_prompts)
        values["hyde_fusion_weights"] = list(self.hyde_fusion_weights)
        values["hypothesis_only_fusion_weights"] = list(self.hypothesis_only_fusion_weights)
        values["reranker_fusion_weights"] = list(self.reranker_fusion_weights)
        values["code_reranker_fusion_weights"] = list(self.code_reranker_fusion_weights)
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
        if self.hyde_combination_strategy != "query_plus_hypotheses":
            raise ValueError(
                "the paired experiment uses query_plus_hypotheses for its existing system; "
                "hypothesis_only is evaluated as a separate ablation"
            )
        if self.reranker_batch_size <= 0 or self.reranker_max_length <= 0:
            raise ValueError("reranker_batch_size and reranker_max_length must be positive")
        if self.code_reranker_batch_size <= 0 or self.code_reranker_max_length <= 0:
            raise ValueError(
                "code_reranker_batch_size and code_reranker_max_length must be positive"
            )
        if self.code_reranker_enabled and self.code_reranker_device != "cpu":
            raise ValueError("the pinned code-reranker comparison requires CPU inference")
        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        for name, weights in (
            ("hyde_fusion_weights", self.hyde_fusion_weights),
            ("hypothesis_only_fusion_weights", self.hypothesis_only_fusion_weights),
            ("reranker_fusion_weights", self.reranker_fusion_weights),
            ("code_reranker_fusion_weights", self.code_reranker_fusion_weights),
            ("combined_fusion_weights", self.combined_fusion_weights),
        ):
            if len(weights) != 2 or any(weight < 0 for weight in weights) or not any(weights):
                raise ValueError(f"{name} must contain two non-negative weights, one positive")


def experiment_system_ids(config: ExperimentConfig) -> list[str]:
    """Return primary systems for this configured run."""

    systems = list(SYSTEM_IDS)
    if config.code_reranker_enabled:
        systems.append(CODE_RERANKER_SYSTEM_ID)
    return systems


@dataclass(frozen=True)
class ModelRevisions:
    """Configured and resolved revisions used by cache identities."""

    e5: str
    hyde: str
    reranker: str
    code_reranker: str | None = None

    def as_dict(self) -> dict[str, str]:
        revisions = {"e5": self.e5, "hyde": self.hyde, "reranker": self.reranker}
        if self.code_reranker is not None:
            revisions["code_reranker"] = self.code_reranker
        return revisions


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
    hypothesis_only_query_embeddings: Path
    hypothesis_only_query_metadata: Path
    original_rankings: Path
    original_rankings_metadata: Path
    hyde_rankings: Path
    hyde_rankings_metadata: Path
    hypothesis_only_rankings: Path
    hypothesis_only_rankings_metadata: Path
    original_reranked: Path
    original_reranked_metadata: Path
    hyde_fused: Path
    hyde_fused_metadata: Path
    hypothesis_only_fused: Path
    hypothesis_only_fused_metadata: Path
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


def build_hyde_representation_conditions(
    queries: Mapping[str, str],
    generated: Mapping[str, Sequence[str]],
) -> dict[str, dict[str, str]]:
    """Build paired query-plus and hypothesis-only text from one generation set."""

    if set(queries) != set(generated):
        raise ValueError("HyDE generations must match the query IDs")
    query_plus = {
        query_id: combine_hypotheses(
            queries[query_id], generated[query_id], strategy="query_plus_hypotheses"
        )
        for query_id in queries
    }
    hypothesis_only = {
        query_id: combine_hypotheses(
            queries[query_id], generated[query_id], strategy="hypothesis_only"
        )
        for query_id in queries
    }
    return {
        "query_plus_hypotheses": query_plus,
        "hypothesis_only": hypothesis_only,
    }


class HyDEGenerator:
    """Local text-to-text generator used for real HyDE inference."""

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        batch_observer: Callable[..., None] | None = None,
    ):
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers and torch are required for HyDE generation") from exc

        self.config = config
        self.torch = torch
        self.batch_observer = batch_observer
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
        synchronize_device(self.device)
        started = time.perf_counter()
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
        synchronize_device(self.device)
        decoded = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        expected = len(batch) * self.config.hyde_num_hypotheses
        if len(decoded) != expected:
            raise RuntimeError(
                f"HyDE generator returned {len(decoded)} outputs; expected {expected}"
            )
        result = [
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
        if self.batch_observer is not None:
            self.batch_observer(
                stage="hyde_generation",
                seconds=time.perf_counter() - started,
                unit_count=len(batch),
                unit="queries",
            )
        return result

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

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        batch_observer: Callable[..., None] | None = None,
    ):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is required for re-ranking") from exc

        self.config = config
        self.batch_observer = batch_observer
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
        if self.batch_observer is None:
            scores = self.model.predict(
                list(pairs),
                batch_size=self.config.reranker_batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
            )
        else:
            with observe_module_batches(
                self.model,
                device=self.device,
                stage="cross_encoder_scoring",
                observer=self.batch_observer,
                unit="query_candidate_pairs",
                fallback_batch_size=self.config.reranker_batch_size,
            ):
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


class CodeCrossEncoderReranker:
    """Pinned StaQC-trained pair scorer for the same E5 candidate pool."""

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        batch_observer: Callable[..., None] | None = None,
    ):
        if not config.code_reranker_enabled:
            raise ValueError("enable the code-trained reranker before loading it")
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is required for code re-ranking") from exc

        self.config = config
        self.batch_observer = batch_observer
        self.device = config.code_reranker_device
        self.model = CrossEncoder(
            config.code_reranker_model_id,
            revision=config.code_reranker_model_revision,
            max_length=config.code_reranker_max_length,
            device=self.device,
        )
        self.resolved_revision = _require_resolved_revision(
            self.model,
            config.code_reranker_model_revision,
            "code cross-encoder",
        )

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        if not pairs:
            raise ValueError("cannot score an empty candidate set")
        if self.batch_observer is None:
            scores = self.model.predict(
                list(pairs),
                batch_size=self.config.code_reranker_batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
            )
        else:
            with observe_module_batches(
                self.model,
                device=self.device,
                stage="cross_encoder_scoring",
                observer=self.batch_observer,
                unit="query_candidate_pairs",
                fallback_batch_size=self.config.code_reranker_batch_size,
            ):
                scores = self.model.predict(
                    list(pairs),
                    batch_size=self.config.code_reranker_batch_size,
                    show_progress_bar=True,
                    convert_to_numpy=True,
                )
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if scores.shape[0] != len(pairs) or not np.isfinite(scores).all():
            raise ValueError("code cross-encoder returned invalid candidate scores")
        return scores

    def preflight(
        self,
        query_text: str,
        code_text: str,
        *,
        query_id: str,
        document_id: str,
    ) -> dict[str, Any]:
        score = float(self.score_pairs([(query_text, code_text)])[0])
        if not np.isfinite(score):
            raise ValueError("code reranker preflight produced a non-finite score")
        model_config = getattr(getattr(self.model, "model", None), "config", None)
        tokenizer = getattr(self.model, "tokenizer", None)
        activation = getattr(self.model, "default_activation_function", None)
        return {
            "status": "passed",
            "real_model_weights": True,
            "pair_count": 1,
            "query_id": str(query_id),
            "document_id": str(document_id),
            "finite_score": score,
            "score_direction": "higher_score_ranks_first",
            "score_output": "sentence-transformers CrossEncoder.predict default output",
            "configured_revision": self.config.code_reranker_model_revision,
            "resolved_revision": self.resolved_revision,
            "device": self.device,
            "tokenizer": {
                "class": None if tokenizer is None else type(tokenizer).__name__,
                "name_or_path": None if tokenizer is None else getattr(tokenizer, "name_or_path", None),
            },
            "model_output": {
                "num_labels": None if model_config is None else getattr(model_config, "num_labels", None),
                "problem_type": None if model_config is None else getattr(model_config, "problem_type", None),
                "activation": None if activation is None else type(activation).__name__,
                "parameter_dtype": _module_precision(self.model),
            },
            "input_format": "CrossEncoder.predict([(query_text, corpus_text)])",
            "max_length": self.config.code_reranker_max_length,
            "truncation": "sentence-transformers tokenizer truncation at max_length",
        }

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

    def close(self) -> None:
        del self.model


def add_code_reranker_rankings(
    loaded_replay: dict[str, Any],
    config: ExperimentConfig,
    run_data: RunData,
    *,
    cost_recorder: RuntimeCostRecorder,
) -> dict[str, Any]:
    """Score the replayed E5 candidate pool with real pinned code-model weights."""

    if not config.code_reranker_enabled:
        return loaded_replay
    rankings = loaded_replay["rankings"]
    source_rankings = rankings["e5_for_rerank"]
    if not source_rankings:
        raise ValueError("code-reranker replay has no E5 candidate rankings")
    query_id = next(iter(source_rankings))
    document_id = next(iter(source_rankings[query_id]))
    with cost_recorder.phase("model_load_code_reranker", "startup"):
        reranker = CodeCrossEncoderReranker(
            config,
            batch_observer=cost_recorder.observe_batch,
        )
    try:
        preflight_started = time.perf_counter()
        with cost_recorder.phase("code_reranker_preflight", "online_query"):
            preflight = reranker.preflight(
                run_data.queries[query_id],
                run_data.corpus[document_id]["text"],
                query_id=query_id,
                document_id=document_id,
            )
        preflight_seconds = time.perf_counter() - preflight_started
        scoring_started = time.perf_counter()
        with cost_recorder.phase("code_reranker_scoring", "online_query"):
            code_rankings = reranker.rerank(
                run_data.queries,
                run_data.corpus,
                source_rankings,
            )
        scoring_seconds = time.perf_counter() - scoring_started
    finally:
        reranker.close()
    _validate_rankings(
        code_rankings,
        run_data,
        label="code reranker",
        candidate_depth=config.candidate_depth,
    )
    for current_query_id in run_data.queries:
        if set(code_rankings[current_query_id]) != set(source_rankings[current_query_id]):
            raise ValueError(
                f"code reranker changed the E5 candidate pool for {current_query_id}"
            )
    loaded_replay["rankings"]["code_reranker"] = code_rankings
    loaded_replay["revisions"] = replace(
        loaded_replay["revisions"],
        code_reranker=reranker.resolved_revision,
    )
    loaded_replay["code_reranker_preflight"] = preflight
    source_provenance = loaded_replay["source_provenance"]
    source_provenance.update(
        {
            "classification": "saved_baseline_rankings_plus_fresh_code_model_inference",
            "fresh_model_inference": True,
            "real_model_inference_inputs": True,
            "synthetic_scores": False,
            "code_reranker_preflight": preflight,
            "code_reranker": {
                "model_id": config.code_reranker_model_id,
                "configured_revision": config.code_reranker_model_revision,
                "resolved_revision": loaded_replay["revisions"].code_reranker,
                "device": config.code_reranker_device,
                "batch_size": config.code_reranker_batch_size,
                "max_length": config.code_reranker_max_length,
                "pair_count": sum(
                    len(values) for values in code_rankings.values()
                ),
                "preflight_seconds": preflight_seconds,
                "scoring_seconds": scoring_seconds,
                "candidate_contract": candidate_contract(code_rankings),
                "score_direction": "higher_score_ranks_first",
                "training_context": _code_reranker_training_context(),
            },
        }
    )
    return loaded_replay


def run_code_reranker_preflight(
    config: ExperimentConfig,
    run_data: RunData,
    first_stage_rankings: Mapping[str, Mapping[str, float]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load pinned real weights on CPU and score one declared query/candidate pair."""

    if not config.code_reranker_enabled:
        raise ValueError("enable the code-trained reranker before running its preflight")
    if not first_stage_rankings:
        raise ValueError("code-reranker preflight requires a saved E5 ranking")
    query_id = next(iter(first_stage_rankings))
    candidate_ids = first_stage_rankings[query_id]
    if not candidate_ids:
        raise ValueError("code-reranker preflight requires a non-empty E5 candidate list")
    document_id = next(iter(candidate_ids))
    if query_id not in run_data.queries or document_id not in run_data.corpus:
        raise ValueError("code-reranker preflight pair is outside the declared population")
    run_id = f"azh531-preflight-{time.time_ns()}"
    environment = environment_metadata(config)
    with RuntimeCostRecorder() as cost_recorder:
        with cost_recorder.phase("model_load_code_reranker", "startup"):
            reranker = CodeCrossEncoderReranker(
                config,
                batch_observer=cost_recorder.observe_batch,
            )
        try:
            with cost_recorder.phase("code_reranker_preflight", "online_query"):
                preflight = reranker.preflight(
                    run_data.queries[query_id],
                    run_data.corpus[document_id]["text"],
                    query_id=query_id,
                    document_id=document_id,
                )
            precision = _module_precision(reranker.model)
        finally:
            reranker.close()
    environment["model_precision"] = {"code_reranker": precision}
    environment["model_devices"] = {"code_reranker": config.code_reranker_device}
    costs = cost_recorder.summary(
        run_id=run_id,
        cache_identity="preflight",
        status="preflight",
        run_mode=config.run_mode,
        environment=environment,
        config=config,
        query_count=1,
        corpus_count=len(run_data.corpus),
        reranked_pair_count=1,
        model_precision={"code_reranker": precision},
        offline_index_build_seconds=0.0,
    )
    costs["evidence_level"] = "preflight"
    costs["benchmark_evidence"] = False
    preflight["run_id"] = run_id
    preflight["parameter_dtype"] = precision
    preflight["model_load_seconds"] = next(
        (
            item["seconds"]
            for item in costs["phase_timings"]
            if item["name"] == "model_load_code_reranker"
        ),
        None,
    )
    preflight["score_seconds"] = next(
        (
            item["seconds"]
            for item in costs["phase_timings"]
            if item["name"] == "code_reranker_preflight"
        ),
        None,
    )
    return preflight, costs


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
        hypothesis_only_query_embeddings=root / "hypothesis_only_query_embeddings.npy",
        hypothesis_only_query_metadata=root / "hypothesis_only_query_embeddings.metadata.json",
        original_rankings=root / "original_rankings.json",
        original_rankings_metadata=root / "original_rankings.metadata.json",
        hyde_rankings=root / "hyde_rankings.json",
        hyde_rankings_metadata=root / "hyde_rankings.metadata.json",
        hypothesis_only_rankings=root / "hypothesis_only_rankings.json",
        hypothesis_only_rankings_metadata=root / "hypothesis_only_rankings.metadata.json",
        original_reranked=root / "original_reranked.json",
        original_reranked_metadata=root / "original_reranked.metadata.json",
        hyde_fused=root / "e5_hyde_fused.json",
        hyde_fused_metadata=root / "e5_hyde_fused.metadata.json",
        hypothesis_only_fused=root / "e5_hypothesis_only_fused.json",
        hypothesis_only_fused_metadata=root / "e5_hypothesis_only_fused.metadata.json",
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


def _ranking_artifact_record(
    rankings: Mapping[str, Mapping[str, float]],
    *,
    path: Path,
    metadata_path: Path,
) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "canonical_content_sha256": _rankings_file_sha256(rankings),
        "candidate_contract": candidate_contract(rankings),
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

    controls = {
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
            "representation_conditions": [
                "query_plus_hypotheses",
                "hypothesis_only",
            ],
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
    if config.code_reranker_enabled:
        controls["code_reranker"] = {
            "model_id": config.code_reranker_model_id,
            "model_revision": config.code_reranker_model_revision,
            "batch_size": config.code_reranker_batch_size,
            "max_length": config.code_reranker_max_length,
            "device": config.code_reranker_device,
            "input_format": "CrossEncoder.predict([(query_text, corpus_text)])",
            "score_direction": "higher_score_ranks_first",
        }
    return controls


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
    expected_schema = (
        "code-retrieval-validation-selection-v3-code-reranker"
        if config.code_reranker_enabled
        else "code-retrieval-validation-selection-v2-representation-ablations"
    )
    if manifest.get("schema") != expected_schema:
        raise ValueError("unsupported validation-selection manifest schema")
    if manifest.get("code_version") != CODE_VERSION:
        raise ValueError("selection manifest was produced by a different experiment code version")
    if manifest.get("selection_qrels_split") != "valid":
        raise ValueError("test evaluation requires a manifest selected on valid qrels")
    if manifest.get("target_evaluation_qrels_split") != "test":
        raise ValueError("selection manifest is not frozen for test evaluation")
    if manifest.get("status") != "complete":
        raise ValueError("selection manifest is not complete")
    required_sweeps = {
        "e5_hyde",
        "e5_hypothesis_only",
        "e5_rerank",
        "e5_hyde_rerank",
    }
    if config.code_reranker_enabled:
        required_sweeps.add(CODE_RERANKER_SYSTEM_ID)
    if set(manifest.get("sweeps", {})) != required_sweeps:
        raise ValueError("selection manifest is missing a required representation or fusion sweep")
    if manifest.get("baseline_parity", {}).get("passed") is not True:
        raise ValueError("validation baseline parity failed; refusing final evaluation")
    if config.run_mode != "benchmark" or config.qrels_split != "test":
        raise ValueError("frozen selection manifests may only be used for benchmark test evaluation")
    if manifest.get("controls") != _selection_controls(config):
        raise ValueError("selection manifest controls do not match the current experiment configuration")
    expected_model_revisions = {
        "e5": config.model_revision,
        "hyde": config.hyde_model_revision,
        "reranker": config.reranker_model_revision,
    }
    if config.code_reranker_enabled:
        expected_model_revisions["code_reranker"] = config.code_reranker_model_revision
    if manifest.get("model_revisions") != expected_model_revisions:
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


def load_validation_ranking_replay(
    selection_manifest_path: str | Path,
    run_data: RunData,
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Load the complete saved valid rankings from a sealed prior selection."""

    if config.run_mode != "benchmark" or config.qrels_split != "valid":
        raise ValueError("valid ranking replay requires benchmark mode and valid qrels")
    path = Path(selection_manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"valid selection manifest is missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    verify_selection_manifest_integrity(manifest)
    if (
        manifest.get("schema")
        != "code-retrieval-validation-selection-v2-representation-ablations"
        or manifest.get("status") != "complete"
        or manifest.get("selection_qrels_split") != "valid"
        or manifest.get("target_evaluation_qrels_split") != "test"
        or manifest.get("baseline_parity", {}).get("passed") is not True
    ):
        raise ValueError("valid ranking replay requires a complete sealed AZH-529 selection")
    if len(run_data.queries) != 500 or len(run_data.qrels) != 500 or len(run_data.corpus) != 20604:
        raise ValueError("valid ranking replay requires the complete declared CosQA population")
    population = manifest.get("selection_population", {})
    if (
        population.get("queries") != query_population_fingerprint(run_data.queries)
        or population.get("qrels") != qrels_fingerprint(run_data.qrels)
        or population.get("corpus") != corpus_fingerprint(run_data.corpus)
    ):
        raise ValueError("valid selection manifest does not match the loaded population")
    controls = manifest.get("controls", {})
    if (
        controls.get("dataset", {}).get("id") != config.dataset_id
        or controls.get("dataset", {}).get("revision") != config.dataset_revision
        or controls.get("e5", {}).get("model_revision") != config.model_revision
        or controls.get("hyde", {}).get("model_revision") != config.hyde_model_revision
        or controls.get("reranker", {}).get("model_revision") != config.reranker_model_revision
        or controls.get("runtime_controls", {}).get("candidate_depth") != config.candidate_depth
        or controls.get("runtime_controls", {}).get("rrf_k") != config.rrf_k
    ):
        raise ValueError("valid selection manifest has incompatible frozen controls")
    expected_revisions = {
        "e5": config.model_revision,
        "hyde": config.hyde_model_revision,
        "reranker": config.reranker_model_revision,
    }
    if manifest.get("model_revisions") != expected_revisions:
        raise ValueError("valid selection manifest has incompatible model revisions")

    labels = {
        "e5_for_hyde",
        "hyde",
        "hyde_hypothesis_only",
        "e5_for_rerank",
        "reranker",
    }
    input_record = manifest.get("ranking_inputs", {})
    file_records = {
        item.get("label"): item for item in input_record.get("files", [])
    }
    if set(file_records) != labels:
        raise ValueError("valid selection manifest does not record every required ranking input")
    rankings: dict[str, dict[str, dict[str, float]]] = {}
    verified_files: list[dict[str, Any]] = []
    ranking_content = input_record.get("ranking_content", {})
    for label in (
        "e5_for_hyde",
        "hyde",
        "hyde_hypothesis_only",
        "e5_for_rerank",
        "reranker",
    ):
        record = file_records[label]
        source_path = Path(record.get("path", ""))
        metadata_path = Path(record.get("metadata_path", ""))
        if not source_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"valid ranking source or metadata is missing for {label}")
        actual_hash = sha256_file(source_path)
        metadata_hash = sha256_file(metadata_path)
        if actual_hash != record.get("sha256") or metadata_hash != record.get("metadata_sha256"):
            raise ValueError(f"valid ranking source digest does not match the sealed manifest for {label}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata_config = metadata.get("config", {})
        if (
            metadata_config.get("dataset_id") != config.dataset_id
            or metadata_config.get("dataset_revision") != config.dataset_revision
            or metadata_config.get("qrels_split") != "valid"
            or metadata_config.get("candidate_depth") != config.candidate_depth
            or metadata_config.get("run_mode") != "benchmark"
            or metadata_config.get("device") != "cpu"
            or metadata.get("identity") != input_record.get("cache_identity")
            or metadata.get("model_revisions") != expected_revisions
        ):
            raise ValueError(f"valid ranking metadata controls do not match for {label}")
        raw = json.loads(source_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"valid ranking source {label} must contain an object")
        _validate_rankings(raw, run_data, label=label, candidate_depth=config.candidate_depth)
        actual_contract = candidate_contract(raw)
        expected_content = ranking_content.get(label, {})
        if (
            expected_content.get("canonical_sha256") != _rankings_file_sha256(raw)
            or expected_content.get("candidate_contract") != actual_contract
        ):
            raise ValueError(f"valid ranking content does not match the sealed manifest for {label}")
        rankings[label] = {
            str(query_id): {str(doc_id): float(score) for doc_id, score in docs.items()}
            for query_id, docs in raw.items()
        }
        verified_files.append(
            {
                "label": label,
                "path": str(source_path),
                "sha256": actual_hash,
                "metadata_path": str(metadata_path),
                "metadata_sha256": metadata_hash,
                "source_identity": metadata.get("identity"),
                "source_code_version": metadata.get("code_version"),
                "source_git_commit": metadata.get("git_commit"),
                "source_kind": metadata.get("kind"),
            }
        )
    for query_id in run_data.queries:
        if set(rankings["e5_for_hyde"][query_id]) != set(rankings["e5_for_rerank"][query_id]):
            raise ValueError(f"valid E5 candidate pools disagree for {query_id}")
        if set(rankings["e5_for_rerank"][query_id]) != set(rankings["reranker"][query_id]):
            raise ValueError(f"valid passage reranker changed the E5 candidate pool for {query_id}")
    return {
        "rankings": rankings,
        "revisions": ModelRevisions(**expected_revisions),
        "source_provenance": {
            "classification": "retrospective_replay_of_saved_real_model_valid_rankings",
            "fresh_model_inference": False,
            "real_model_inference_inputs": True,
            "synthetic_scores": False,
            "split": "valid",
            "selection_manifest_path": str(path),
            "selection_manifest_sha256": sha256_file(path),
            "selection_run_id": manifest.get("selection_run_id"),
            "files": verified_files,
        },
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

    if split == "valid":
        raise ValueError(
            "historical valid ranking replay predates the hypothesis-only representation; "
            "rerun the existing notebook with fresh model inference for validation selection"
        )
    required = {
        "e5_for_hyde",
        "hyde",
        "hyde_hypothesis_only",
        "e5_for_rerank",
        "reranker",
    }
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
    replay_hyde_generation: dict[str, Any] | None = None
    test_result_ranking_artifacts: dict[str, Any] = {}
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
        replay_hyde_generation = dict(test_result.get("hyde_generation_provenance", {}))
        test_result_ranking_artifacts = dict(test_result.get("ranking_artifacts", {}))
        test_result_provenance = test_result.get("artifact_provenance", {})
        required_ranking_artifacts = {
            "e5",
            "query_plus_hypothesis_raw",
            "hypothesis_only_raw",
            "reranker_raw",
        }
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
            or test_result.get("hyde_generation_provenance", {}).get(
                "shared_across_representations"
            ) is not True
            or test_result.get("candidate_contract", {}).get(
                "e5_hypothesis_only_raw"
            ) is None
            or not required_ranking_artifacts.issubset(test_result_ranking_artifacts)
        ):
            raise ValueError("test replay result does not describe the complete pinned real-model test run")
        generation_path = Path(str(replay_hyde_generation.get("path", "")))
        generation_metadata_path = Path(
            str(replay_hyde_generation.get("metadata_path", ""))
        )
        if (
            not generation_path.is_file()
            or sha256_file(generation_path) != replay_hyde_generation.get("sha256")
            or not generation_metadata_path.is_file()
            or sha256_file(generation_metadata_path)
            != replay_hyde_generation.get("metadata_sha256")
            or replay_hyde_generation.get("query_count") != len(run_data.queries)
            or replay_hyde_generation.get("query_ids_sha256")
            != sha256_ids(list(run_data.queries))
        ):
            raise ValueError("test replay generation artifact or query population hash does not match")
        expected_source_identity = test_result.get("cache_identity")
        historical_hash_names = {}

    expected_ranking_ids_sha256 = sha256_ids(
        list(run_data.queries) + ["__corpus__"] + list(run_data.corpus)
    )
    test_ranking_artifact_labels = {
        "e5_for_hyde": "e5",
        "hyde": "query_plus_hypothesis_raw",
        "hyde_hypothesis_only": "hypothesis_only_raw",
        "e5_for_rerank": "e5",
        "reranker": "reranker_raw",
    }
    expected_representation_hashes = (
        replay_hyde_generation.get("representation_text_sha256", {})
        if replay_hyde_generation
        else {}
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
        if split == "test":
            artifact_key = test_ranking_artifact_labels[label]
            expected_artifact_hash = test_result_ranking_artifacts[artifact_key].get("sha256")
            if actual_hash != expected_artifact_hash:
                raise ValueError(f"saved ranking file hash does not match the test run record for {label}")
            expected_representation_key = {
                "hyde": "query_plus_hypotheses",
                "hyde_hypothesis_only": "hypothesis_only",
            }.get(label)
            if expected_representation_key and metadata.get("representation_sha256") != expected_representation_hashes.get(
                expected_representation_key
            ):
                raise ValueError(f"cached representation hash does not match the test run record for {label}")
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
            ("hyde_hypothesis_only", "e5_hypothesis_only_raw"),
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
            "hyde_generation_provenance": replay_hyde_generation,
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
    if "hyde_hypothesis_only" not in rankings:
        raise ValueError(
            "validation selection requires a fresh hypothesis-only ranking; "
            "historical AZH-527 ranking replay does not contain this representation"
        )
    if len(run_data.queries) != 500 or len(run_data.qrels) != 500 or len(run_data.corpus) != 20604:
        raise ValueError("validation selection requires the complete declared CosQA population")
    _validate_rankings(rankings["e5_for_hyde"], run_data, label="E5 for HyDE", candidate_depth=config.candidate_depth)
    _validate_rankings(rankings["hyde"], run_data, label="HyDE", candidate_depth=config.candidate_depth)
    _validate_rankings(
        rankings["hyde_hypothesis_only"],
        run_data,
        label="hypothesis-only HyDE",
        candidate_depth=config.candidate_depth,
    )
    _validate_rankings(rankings["e5_for_rerank"], run_data, label="E5 for reranking", candidate_depth=config.candidate_depth)
    _validate_rankings(rankings["reranker"], run_data, label="reranker", candidate_depth=config.candidate_depth)
    if config.code_reranker_enabled:
        if "code_reranker" not in rankings:
            raise ValueError("code-reranker selection requires real code-reranker rankings")
        if revisions.code_reranker != config.code_reranker_model_revision:
            raise ValueError("resolved code-reranker revision does not match the frozen checkpoint")
        _validate_rankings(
            rankings["code_reranker"],
            run_data,
            label="code reranker",
            candidate_depth=config.candidate_depth,
        )
        for query_id in run_data.queries:
            if set(rankings["code_reranker"][query_id]) != set(rankings["e5_for_rerank"][query_id]):
                raise ValueError(f"code reranker changed the E5 candidate pool for {query_id}")

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
    hypothesis_only_sweep, selected_hypothesis_only = _select_rrf_weights(
        run_data.qrels,
        rankings["e5_for_hyde"],
        rankings["hyde_hypothesis_only"],
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
    code_reranker_sweep = None
    selected_code_reranker = None
    if config.code_reranker_enabled:
        code_reranker_sweep, selected_code_reranker = _select_rrf_weights(
            run_data.qrels,
            rankings["e5_for_rerank"],
            rankings["code_reranker"],
            rrf_k=config.rrf_k,
            candidate_depth=config.candidate_depth,
        )
    representation_metrics: dict[str, Mapping[str, Any] | None] = {
        "e5": baseline_metric,
        "e5_rerank_raw": evaluate_ndcg_at_10(
            run_data.qrels, rankings["reranker"], cutoff=10
        ),
        "e5_hyde_raw": evaluate_ndcg_at_10(run_data.qrels, rankings["hyde"], cutoff=10),
        "e5_hypothesis_only_raw": evaluate_ndcg_at_10(
            run_data.qrels, rankings["hyde_hypothesis_only"], cutoff=10
        ),
        "e5_hyde": evaluate_ndcg_at_10(run_data.qrels, selected_hyde, cutoff=10),
        "e5_hypothesis_only": evaluate_ndcg_at_10(
            run_data.qrels, selected_hypothesis_only, cutoff=10
        ),
    }
    if config.code_reranker_enabled:
        representation_metrics["e5_code_rerank_raw"] = evaluate_ndcg_at_10(
            run_data.qrels, rankings["code_reranker"], cutoff=10
        )
        representation_metrics[CODE_RERANKER_SYSTEM_ID] = evaluate_ndcg_at_10(
            run_data.qrels, selected_code_reranker, cutoff=10
        )
    ablation_config = replace(
        config,
        hyde_fusion_weights=tuple(hyde_sweep["selected"]["weights"]),
        hypothesis_only_fusion_weights=tuple(
            hypothesis_only_sweep["selected"]["weights"]
        ),
    )

    run_id = f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}-{time.time_ns()}"
    selection_root = Path(config.artifact_dir) / "selection_runs" / run_id
    selection_root.mkdir(parents=True, exist_ok=False)
    fused_paths: dict[str, Path] = {}
    fused_hashes: dict[str, str] = {}
    for name, value in (
        ("e5_hyde", selected_hyde),
        ("e5_hypothesis_only", selected_hypothesis_only),
        ("e5_rerank", selected_reranker),
        ("e5_hyde_rerank", selected_combined),
    ):
        path = selection_root / f"{name}_selected_rankings.json"
        _write_ordered_json(path, value)
        fused_paths[name] = path
        fused_hashes[name] = sha256_file(path)
    if config.code_reranker_enabled:
        path = selection_root / f"{CODE_RERANKER_SYSTEM_ID}_selected_rankings.json"
        _write_ordered_json(path, selected_code_reranker)
        fused_paths[CODE_RERANKER_SYSTEM_ID] = path
        fused_hashes[CODE_RERANKER_SYSTEM_ID] = sha256_file(path)
    code_raw_rankings_path = None
    code_raw_rankings_record = None
    if config.code_reranker_enabled:
        code_raw_rankings_path = selection_root / "e5_code_reranker_raw_rankings.json"
        _write_ordered_json(code_raw_rankings_path, rankings["code_reranker"])
        code_raw_rankings_record = {
            "path": str(code_raw_rankings_path),
            "sha256": sha256_file(code_raw_rankings_path),
            "candidate_contract": candidate_contract(rankings["code_reranker"]),
            "pair_count": sum(len(values) for values in rankings["code_reranker"].values()),
        }
    code_per_query_path = None
    selection_cache_identity = str(ranking_provenance.get("cache_identity") or run_id)
    if config.code_reranker_enabled:
        code_per_query_path = selection_root / "code_reranker_per_query_metrics.csv"
        code_metric_ids = (
            "e5",
            "e5_rerank_raw",
            "e5_code_rerank_raw",
            CODE_RERANKER_SYSTEM_ID,
        )
        _write_records_csv(
            code_per_query_path,
            _per_query_metric_rows(
                {name: representation_metrics[name] for name in code_metric_ids},
                cache_identity=selection_cache_identity,
            ),
        )

    representation_ablation_rows = _representation_ablation_rows(
        ablation_config,
        representation_metrics,
        query_count=len(run_data.queries),
        corpus_count=len(run_data.corpus),
        qrels_query_count=len(run_data.qrels),
        qrels_judgment_count=sum(len(rels) for rels in run_data.qrels.values()),
        status="validation_selection",
    )
    representation_ablation_per_query = _representation_ablation_per_query_rows(
        representation_metrics,
        cache_identity=selection_cache_identity,
    )
    representation_ablation_path = selection_root / "representation_ablation.csv"
    representation_ablation_per_query_path = (
        selection_root / "representation_ablation_per_query.csv"
    )
    _write_records_csv(representation_ablation_path, representation_ablation_rows)
    _write_records_csv(representation_ablation_per_query_path, representation_ablation_per_query)
    representation_ablation_artifacts = {
        "representation_ablation": {
            "path": str(representation_ablation_path),
            "sha256": sha256_file(representation_ablation_path),
            "row_count": len(representation_ablation_rows),
        },
        "representation_ablation_per_query": {
            "path": str(representation_ablation_per_query_path),
            "sha256": sha256_file(representation_ablation_per_query_path),
            "row_count": len(representation_ablation_per_query),
            "paired_query_count": len(representation_ablation_per_query),
        },
    }

    current_source_hashes = _experiment_source_hashes(
        repo_root,
        notebook_sha256=notebook_sha256,
        source_notebook_sha256=source_notebook_sha256,
    )
    status = "complete" if baseline_parity["passed"] else "parity_failed"
    sweep_records = {
        "e5_hyde": {
            "input_order": ["e5_for_hyde", "hyde"],
            **hyde_sweep,
        },
        "e5_hypothesis_only": {
            "input_order": ["e5_for_hyde", "hyde_hypothesis_only"],
            **hypothesis_only_sweep,
        },
        "e5_rerank": {
            "input_order": ["e5_for_rerank", "reranker"],
            **reranker_sweep,
        },
        "e5_hyde_rerank": {
            "input_order": ["selected_e5_hyde", "selected_e5_rerank"],
            **combined_sweep,
        },
    }
    if config.code_reranker_enabled:
        sweep_records[CODE_RERANKER_SYSTEM_ID] = {
            "input_order": ["e5_for_rerank", "code_reranker"],
            **code_reranker_sweep,
        }
    selected_items = [
        ("e5_hyde", fused_paths["e5_hyde"], selected_hyde),
        ("e5_hypothesis_only", fused_paths["e5_hypothesis_only"], selected_hypothesis_only),
        ("e5_rerank", fused_paths["e5_rerank"], selected_reranker),
        ("e5_hyde_rerank", fused_paths["e5_hyde_rerank"], selected_combined),
    ]
    if config.code_reranker_enabled:
        selected_items.append(
            (
                CODE_RERANKER_SYSTEM_ID,
                fused_paths[CODE_RERANKER_SYSTEM_ID],
                selected_code_reranker,
            )
        )
    manifest = seal_selection_manifest(
        {
            "schema": (
                "code-retrieval-validation-selection-v3-code-reranker"
                if config.code_reranker_enabled
                else "code-retrieval-validation-selection-v2-representation-ablations"
            ),
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
            "sweeps": sweep_records,
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
                for name, path, value in selected_items
            },
            "code_reranker": (
                {
                    "model_id": config.code_reranker_model_id,
                    "configured_revision": config.code_reranker_model_revision,
                    "resolved_revision": revisions.code_reranker,
                    "batch_size": config.code_reranker_batch_size,
                    "max_length": config.code_reranker_max_length,
                    "device": config.code_reranker_device,
                    "preflight": ranking_provenance.get("code_reranker_preflight"),
                    "training_context": _code_reranker_training_context(),
                    "raw_rankings": code_raw_rankings_record,
            "per_query_metrics": (
                {
                    "path": str(code_per_query_path),
                    "sha256": sha256_file(code_per_query_path),
                    "row_count": 4 * len(run_data.queries),
                    "paired_query_count": len(run_data.queries),
                }
                if code_per_query_path is not None
                else None
            ),
                }
                if config.code_reranker_enabled
                else None
            ),
            "representation_ablation_artifacts": representation_ablation_artifacts,
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
        "run_id": run_id,
        "cache_identity": selection_cache_identity,
        "run_mode": config.run_mode,
        "system_ids": experiment_system_ids(config),
        "code_version": CODE_VERSION,
        "git_commit": git_commit(repo_root),
        "config": config.as_dict(),
        "environment": dict(environment),
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
        "representation_ablations": representation_ablation_rows,
        "representation_ablation_artifacts": representation_ablation_artifacts,
        "representation_ablation_per_query_count": len(representation_ablation_per_query),
        "representation_ablation_notes": {
            "query_plus_generated_text_unfused": "Raw E5 ranking from the original query plus the saved generated hypothesis.",
            "generated_text_only_unfused": "Raw E5 ranking from the same saved hypothesis without the original query.",
            "fusion_selection": "Each original-query E5 fusion weight was selected independently on the complete valid split using nDCG@10.",
            "interpretation": "Do not attribute a fused score change to representation quality alone; inspect the paired raw conditions and candidate contracts.",
        },
        "ranking_inputs": dict(ranking_provenance),
        "evidence_boundary": manifest["evidence_boundary"],
    }
    if config.code_reranker_enabled:
        result["code_reranker"] = manifest["code_reranker"]
        result["raw_code_reranker_metrics"] = dict(
            representation_metrics["e5_code_rerank_raw"]
        )
        result["code_reranker_metrics"] = dict(
            representation_metrics[CODE_RERANKER_SYSTEM_ID]
        )
        result["reranked_candidate_pair_count"] = code_raw_rankings_record["pair_count"]
    result_path = selection_root / "selection_result.json"
    _write_ordered_json(result_path, result)
    selection_metadata_path = selection_root / "selection_metadata.json"
    _json_write(
        selection_metadata_path,
        {
            "result": result,
            "ranking_provenance": dict(ranking_provenance),
        },
    )
    paths = {
        "run_dir": str(selection_root),
        "selection_run_dir": str(selection_root),
        "selection_manifest": str(manifest_path),
        "selection_result": str(result_path),
        "run_result": str(result_path),
        "run_metadata": str(selection_metadata_path),
        "e5_hyde_selected_rankings": str(fused_paths["e5_hyde"]),
        "e5_hypothesis_only_selected_rankings": str(fused_paths["e5_hypothesis_only"]),
        "e5_rerank_selected_rankings": str(fused_paths["e5_rerank"]),
        "e5_hyde_rerank_selected_rankings": str(fused_paths["e5_hyde_rerank"]),
        "representation_ablation": str(representation_ablation_path),
        "representation_ablation_per_query": str(representation_ablation_per_query_path),
    }
    if config.code_reranker_enabled:
        paths["e5_code_rerank_selected_rankings"] = str(
            fused_paths[CODE_RERANKER_SYSTEM_ID]
        )
        paths["e5_code_rerank_raw_rankings"] = str(code_raw_rankings_path)
        paths["code_reranker_per_query_metrics"] = str(code_per_query_path)
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
    if "hyde_hypothesis_only" not in rankings:
        raise ValueError(
            "test ranking replay requires a hypothesis-only HyDE ranking; "
            "historical AZH-527 artifacts cannot satisfy the representation ablation"
        )
    config = replace(
        config,
        hyde_fusion_weights=tuple(selected["e5_hyde"]["selected"]["weights"]),
        hypothesis_only_fusion_weights=tuple(
            selected["e5_hypothesis_only"]["selected"]["weights"]
        ),
        reranker_fusion_weights=tuple(selected["e5_rerank"]["selected"]["weights"]),
        code_reranker_fusion_weights=(
            tuple(selected[CODE_RERANKER_SYSTEM_ID]["selected"]["weights"])
            if config.code_reranker_enabled
            else config.code_reranker_fusion_weights
        ),
        combined_fusion_weights=tuple(selected["e5_hyde_rerank"]["selected"]["weights"]),
    )
    e5_rankings = rankings["e5_for_hyde"]
    e5_rerank_rankings = rankings["e5_for_rerank"]
    hyde_rankings = rankings["hyde"]
    hypothesis_only_rankings = rankings["hyde_hypothesis_only"]
    reranker_rankings = rankings["reranker"]
    fused_hyde = reciprocal_rank_fusion(
        [e5_rankings, hyde_rankings],
        weights=config.hyde_fusion_weights,
        top_k=config.candidate_depth,
        rrf_k=config.rrf_k,
    )
    fused_hypothesis_only = reciprocal_rank_fusion(
        [e5_rankings, hypothesis_only_rankings],
        weights=config.hypothesis_only_fusion_weights,
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
    fused_code_reranker = None
    if config.code_reranker_enabled:
        if "code_reranker" not in rankings:
            raise ValueError("test evaluation requires the fresh code-reranker ranking")
        if revisions.code_reranker != config.code_reranker_model_revision:
            raise ValueError("test ranking revision does not match the pinned code checkpoint")
        for query_id in run_data.queries:
            if set(rankings["code_reranker"][query_id]) != set(e5_rerank_rankings[query_id]):
                raise ValueError(f"test code reranker changed the E5 candidate pool for {query_id}")
        fused_code_reranker = reciprocal_rank_fusion(
            [e5_rerank_rankings, rankings["code_reranker"]],
            weights=config.code_reranker_fusion_weights,
            top_k=config.candidate_depth,
            rrf_k=config.rrf_k,
        )
    evaluation_started = time.perf_counter()
    metrics: dict[str, Mapping[str, Any]] = {
        "e5": evaluate_ndcg_at_10(run_data.qrels, e5_rankings, cutoff=10),
        "e5_hyde": evaluate_ndcg_at_10(run_data.qrels, fused_hyde, cutoff=10),
        "e5_hypothesis_only": evaluate_ndcg_at_10(
            run_data.qrels, fused_hypothesis_only, cutoff=10
        ),
        "e5_rerank": evaluate_ndcg_at_10(run_data.qrels, fused_reranker, cutoff=10),
        "e5_hyde_rerank": evaluate_ndcg_at_10(run_data.qrels, fused_combined, cutoff=10),
        "e5_hyde_raw": evaluate_ndcg_at_10(run_data.qrels, hyde_rankings, cutoff=10),
        "e5_hypothesis_only_raw": evaluate_ndcg_at_10(
            run_data.qrels, hypothesis_only_rankings, cutoff=10
        ),
        "e5_rerank_raw": evaluate_ndcg_at_10(run_data.qrels, reranker_rankings, cutoff=10),
    }
    if config.code_reranker_enabled:
        metrics["e5_code_rerank_raw"] = evaluate_ndcg_at_10(
            run_data.qrels, rankings["code_reranker"], cutoff=10
        )
        metrics[CODE_RERANKER_SYSTEM_ID] = evaluate_ndcg_at_10(
            run_data.qrels, fused_code_reranker, cutoff=10
        )
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
        timings={
            "cached_ranking_evaluation": evaluation_seconds,
            **(
                {
                    "code_reranker_preflight": ranking_replay["source_provenance"]
                    .get("code_reranker", {})
                    .get("preflight_seconds"),
                    "code_reranker_scoring": ranking_replay["source_provenance"]
                    .get("code_reranker", {})
                    .get("scoring_seconds"),
                }
                if config.code_reranker_enabled
                else {}
            ),
        },
        notebook_sha256=notebook_sha256,
        source_notebook_sha256=source_notebook_sha256,
        repo_root=repo_root,
        status="benchmark" if config.code_reranker_enabled else "cache_replay",
        real_model_inference=config.code_reranker_enabled,
    )
    result["hyde_generation_provenance"] = dict(
        ranking_replay["source_provenance"].get("hyde_generation_provenance", {})
    )
    result["candidate_contract"] = {
        "e5": candidate_contract(e5_rankings),
        "e5_hyde_raw": candidate_contract(hyde_rankings),
        "e5_hypothesis_only_raw": candidate_contract(hypothesis_only_rankings),
        "e5_hyde": {**candidate_contract(fused_hyde), "pool_policy": "union_of_e5_and_hyde_top_k_truncated_to_candidate_depth"},
        "e5_hypothesis_only": {
            **candidate_contract(fused_hypothesis_only),
            "pool_policy": "union_of_e5_and_hypothesis_only_top_k_truncated_to_candidate_depth",
        },
        "e5_rerank_raw": {**candidate_contract(reranker_rankings), "pool_policy": "same_candidate_ids_as_e5"},
        "e5_rerank": {**candidate_contract(fused_reranker), "pool_policy": "same_candidate_ids_as_e5"},
        "e5_hyde_rerank": {**candidate_contract(fused_combined), "pool_policy": "union_of_selected_component_rankings_truncated_to_candidate_depth"},
    }
    if config.code_reranker_enabled:
        result["candidate_contract"]["e5_code_rerank_raw"] = {
            **candidate_contract(rankings["code_reranker"]),
            "pool_policy": "same_candidate_ids_as_e5",
        }
        result["candidate_contract"][CODE_RERANKER_SYSTEM_ID] = {
            **candidate_contract(fused_code_reranker),
            "pool_policy": "same_candidate_ids_as_e5",
        }
    result["selection_manifest"] = {
        "path": str(selection_manifest.get("_path", "")),
        "manifest_sha256": selection_manifest["manifest_sha256"],
        "selection_run_id": selection_manifest["selection_run_id"],
        "selected_weights": {
            name: selection_manifest["sweeps"][name]["selected"]["weights"]
            for name in (
                "e5_hyde",
                "e5_hypothesis_only",
                "e5_rerank",
                "e5_hyde_rerank",
                *([CODE_RERANKER_SYSTEM_ID] if config.code_reranker_enabled else []),
            )
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
    result["benchmark_evidence"] = config.code_reranker_enabled is True
    result["limitations"].append(
        {
            "type": "evidence_level",
            "reason": (
                "The full code-reranker test ranking was scored with fresh real pinned weights; "
                "the E5, HyDE, and passage-reranker comparison inputs were replayed from the "
                "recorded complete historical benchmark. Test results remain retrospective and exploratory."
                if config.code_reranker_enabled
                else "Metrics were recomputed from saved real-model rankings; this run performed no fresh model inference and is retrospective cache-replay evidence."
            ),
        }
    )
    if config.code_reranker_enabled:
        code_evidence = dict(ranking_replay["source_provenance"].get("code_reranker", {}))
        code_evidence.update(result["code_reranker"])
        code_evidence["preflight"] = ranking_replay.get("code_reranker_preflight")
        raw_path = Path(config.artifact_dir) / "runs" / result["run_id"] / "e5_code_reranker_raw_rankings.json"
        _write_ordered_json(raw_path, rankings["code_reranker"])
        code_evidence.update(
            {
                "raw_rankings": {
                    "path": str(raw_path),
                    "sha256": sha256_file(raw_path),
                    "candidate_contract": candidate_contract(rankings["code_reranker"]),
                },
                "metrics": {
                    "raw": dict(metrics["e5_code_rerank_raw"]),
                    "validation_selected_fusion": dict(metrics[CODE_RERANKER_SYSTEM_ID]),
                },
            }
        )
        result["code_reranker"] = code_evidence
        result["reranked_candidate_pair_count"] = code_evidence.get("pair_count", 0)
    artifacts = write_comparison_artifacts(
        config,
        result,
        per_query_metrics=_per_query_metric_rows(metrics, cache_identity=identity),
        representation_ablation_per_query=_representation_ablation_per_query_rows(
            metrics, cache_identity=identity
        ),
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
    system_ids: Sequence[str] | None = None,
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
    for system_id in system_ids or SYSTEM_IDS:
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


REPRESENTATION_ABLATION_CONDITIONS = (
    ("e5", "original_query", "none", None),
    ("e5_hyde_raw", "query_plus_generated_text", "none", None),
    ("e5_hypothesis_only_raw", "generated_text_only", "none", None),
    ("e5_hyde", "query_plus_generated_text", "e5_rrf", "hyde_fusion_weights"),
    (
        "e5_hypothesis_only",
        "generated_text_only",
        "e5_rrf",
        "hypothesis_only_fusion_weights",
    ),
)
REPRESENTATION_METRIC_NAMES = (
    "ndcg_at_10",
    "map_at_10",
    "recall_at_10",
    "precision_at_10",
    "recall_at_1000",
)


def _representation_ablation_rows(
    config: ExperimentConfig,
    metrics: Mapping[str, Mapping[str, Any] | None],
    *,
    query_count: int,
    corpus_count: int,
    qrels_query_count: int,
    qrels_judgment_count: int,
    status: str,
) -> list[dict[str, Any]]:
    baseline = metrics.get("e5")
    if baseline is None:
        return []
    rows: list[dict[str, Any]] = []
    for system_id, representation, fusion, weight_field in REPRESENTATION_ABLATION_CONDITIONS:
        metric = metrics.get(system_id)
        if metric is None:
            continue
        weights = None if weight_field is None else list(getattr(config, weight_field))
        row: dict[str, Any] = {
            "system_id": system_id,
            "representation": representation,
            "fusion": fusion,
            "fusion_weights": None if weights is None else json.dumps(weights),
            "selection_qrels_split": (
                "valid"
                if weights is not None
                and status in {"benchmark", "cache_replay", "validation_selection"}
                else "not_selected" if weights is not None else "not_applicable"
            ),
            "ndcg_at_10": float(metric["ndcg_at_10"]),
            "delta_ndcg_at_10_vs_e5": float(metric["ndcg_at_10"])
            - float(baseline["ndcg_at_10"]),
            "query_count": query_count,
            "corpus_count": corpus_count,
            "qrels_query_count": qrels_query_count,
            "qrels_judgment_count": qrels_judgment_count,
            "candidate_depth": config.candidate_depth,
            "status": status,
            "benchmark_evidence": status == "benchmark",
            "evaluator_package": config.evaluator_package,
            "evaluation_order_policy": RANKING_ORDER_POLICY,
        }
        for metric_name in REPRESENTATION_METRIC_NAMES:
            score = float(metric[metric_name])
            row[metric_name] = score
            row[f"delta_{metric_name}_vs_e5"] = score - float(baseline[metric_name])
        rows.append(row)
    return rows


def _representation_ablation_per_query_rows(
    metrics: Mapping[str, Mapping[str, Any] | None],
    *,
    cache_identity: str,
) -> list[dict[str, Any]]:
    available = [
        (system_id, system_id.replace("e5_hypothesis_only", "generated_text_only"))
        for system_id, _, _, _ in REPRESENTATION_ABLATION_CONDITIONS
        if metrics.get(system_id) is not None
    ]
    if "e5" not in {system_id for system_id, _ in available}:
        return []
    query_orders: dict[str, list[str]] = {}
    per_query_maps: dict[str, Mapping[str, Mapping[str, Any]]] = {}
    for system_id, _ in available:
        per_query = metrics[system_id].get("per_query", {})
        per_query_maps[system_id] = per_query
        query_orders[system_id] = list(per_query)
    baseline_order = query_orders["e5"]
    if any(order != baseline_order for order in query_orders.values()):
        raise ValueError("representation ablation metrics do not share the same ordered query population")

    rows: list[dict[str, Any]] = []
    for query_id in baseline_order:
        baseline = per_query_maps["e5"][query_id]
        row: dict[str, Any] = {"query_id": query_id, "cache_identity": cache_identity}
        for system_id, condition in available:
            values = per_query_maps[system_id][query_id]
            for metric_name in REPRESENTATION_METRIC_NAMES:
                value = values.get(metric_name)
                row[f"{condition}__{metric_name}"] = value
                row[f"{condition}__delta_{metric_name}_vs_e5"] = (
                    None if value is None else float(value) - float(baseline[metric_name])
                )
            for detail_name in (
                "first_relevant_rank",
                "relevant_document_count",
                "relevant_retrieved_at_10",
                "relevant_retrieved_at_1000",
            ):
                row[f"{condition}__{detail_name}"] = values.get(detail_name)
        rows.append(row)
    return rows


def _write_records_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


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
    system_ids = experiment_system_ids(config)
    if status == "benchmark" and any(metrics.get(system_id) is None for system_id in system_ids):
        raise ValueError("benchmark results require every configured primary system")
    rows = _result_rows(
        metrics,
        query_count=len(run_data.queries),
        corpus_count=len(run_data.corpus),
        qrels_query_count=len(run_data.qrels),
        qrels_judgment_count=sum(len(rels) for rels in run_data.qrels.values()),
        candidate_depth=config.candidate_depth,
        evaluator_package=config.evaluator_package,
        status=status,
        system_ids=system_ids,
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
    representation_rows = _representation_ablation_rows(
        config,
        metrics,
        query_count=len(run_data.queries),
        corpus_count=len(run_data.corpus),
        qrels_query_count=len(run_data.qrels),
        qrels_judgment_count=sum(len(rels) for rels in run_data.qrels.values()),
        status=status,
    )
    result = {
        "system_ids": system_ids,
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
        "representation_ablations": representation_rows,
        "representation_ablation_notes": {
            "query_plus_generated_text_unfused": "Raw E5 ranking from the original query plus the saved generated hypothesis.",
            "generated_text_only_unfused": "Raw E5 ranking from the same saved hypothesis without the original query.",
            "fusion_selection": "Each original-query E5 fusion weight is selected independently on the complete valid split using nDCG@10.",
            "interpretation": "Do not attribute a fused score change to representation quality alone; inspect the paired raw conditions and candidate contracts.",
        },
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
                "e5_hypothesis_only": list(config.hypothesis_only_fusion_weights),
                "e5_rerank": list(config.reranker_fusion_weights),
                "e5_hyde_rerank": list(config.combined_fusion_weights),
            },
            "sources": {
                "e5_hyde": ["e5_original_query", "e5_query_plus_hypothesis"],
                "e5_hypothesis_only": ["e5_original_query", "e5_hypothesis_only"],
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
            for system_id in (
                "e5_hyde_raw",
                "e5_hypothesis_only_raw",
                "e5_rerank_raw",
                "e5_code_rerank_raw",
            )
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
    if config.code_reranker_enabled:
        result["code_reranker"] = {
            "model_id": config.code_reranker_model_id,
            "configured_revision": config.code_reranker_model_revision,
            "resolved_revision": revisions.code_reranker,
            "batch_size": config.code_reranker_batch_size,
            "max_length": config.code_reranker_max_length,
            "device": config.code_reranker_device,
            "input_format": "CrossEncoder.predict([(query_text, corpus_text)])",
            "score_direction": "higher_score_ranks_first",
            "fusion_weights": list(config.code_reranker_fusion_weights),
            "training_context": _code_reranker_training_context(),
        }
        result["fusion"]["component_weights"][CODE_RERANKER_SYSTEM_ID] = list(
            config.code_reranker_fusion_weights
        )
        result["fusion"]["sources"][CODE_RERANKER_SYSTEM_ID] = [
            "e5_original_query",
            "e5_code_cross_encoder_order",
        ]
    return result


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
        {system_id: None for system_id in experiment_system_ids(config)},
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
    representation_ablation_per_query: Sequence[Mapping[str, Any]] = (),
) -> dict[str, str]:
    """Persist JSON, CSV, and metadata atomically below the configured artifact directory."""

    artifact_dir = Path(config.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    result_path = artifact_dir / "result.json"
    comparison_path = artifact_dir / "comparison.csv"
    metadata_path = artifact_dir / "metadata.json"
    representation_ablation_path = artifact_dir / "representation_ablation.csv"
    identity = str(result.get("cache_identity") or "unidentified")
    run_id = str(result.get("run_id") or identity)
    run_dir = artifact_dir / "runs" / run_id
    run_result_path = run_dir / "result.json"
    run_comparison_path = run_dir / "comparison.csv"
    run_metadata_path = run_dir / "metadata.json"
    per_query_path = run_dir / "per_query_metrics.csv"
    run_representation_ablation_path = run_dir / "representation_ablation.csv"
    representation_ablation_per_query_path = run_dir / "representation_ablation_per_query.csv"
    metadata = {
        "result": dict(result),
        "cache_metadata": dict(cache_metadata or {}),
    }
    _json_write(run_result_path, dict(result))
    _write_csv(run_comparison_path, list(result.get("results", [])))
    if per_query_metrics:
        _write_per_query_csv(per_query_path, per_query_metrics)
        metadata.setdefault("artifacts", {})["per_query_metrics"] = {
            "path": str(per_query_path),
            "sha256": hashlib.sha256(per_query_path.read_bytes()).hexdigest(),
            "row_count": len(per_query_metrics),
        }
    ablation_rows = list(result.get("representation_ablations", []))
    if ablation_rows:
        _write_records_csv(run_representation_ablation_path, ablation_rows)
        _write_records_csv(representation_ablation_path, ablation_rows)
        metadata.setdefault("artifacts", {})["representation_ablation"] = {
            "path": str(run_representation_ablation_path),
            "latest_path": str(representation_ablation_path),
            "sha256": hashlib.sha256(run_representation_ablation_path.read_bytes()).hexdigest(),
            "row_count": len(ablation_rows),
        }
    if representation_ablation_per_query:
        _write_records_csv(
            representation_ablation_per_query_path,
            representation_ablation_per_query,
        )
        metadata.setdefault("artifacts", {})["representation_ablation_per_query"] = {
            "path": str(representation_ablation_per_query_path),
            "sha256": hashlib.sha256(
                representation_ablation_per_query_path.read_bytes()
            ).hexdigest(),
            "row_count": len(representation_ablation_per_query),
            "paired_query_count": len(representation_ablation_per_query),
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
        "representation_ablation": str(run_representation_ablation_path)
        if ablation_rows
        else "",
        "representation_ablation_latest": str(representation_ablation_path)
        if ablation_rows
        else "",
        "representation_ablation_per_query": str(representation_ablation_per_query_path)
        if representation_ablation_per_query
        else "",
    }


def _embedding_or_cache(
    *,
    path: Path,
    metadata_path: Path,
    metadata: Mapping[str, Any],
    encode: Callable[[], np.ndarray],
    cost_recorder: RuntimeCostRecorder | None = None,
    stage_name: str = "embedding",
    category: str = "online_query",
    item_count: int = 0,
) -> tuple[np.ndarray, float]:
    load_started = time.perf_counter()
    cached = load_valid_embedding_cache(path, metadata_path, metadata)
    load_seconds = time.perf_counter() - load_started
    if cached is not None:
        if cost_recorder is not None:
            cost_recorder.record_cache_stage(
                name=stage_name,
                category=category,
                cache_hit=True,
                load_seconds=load_seconds,
                compute_seconds=0.0,
                save_seconds=0.0,
                item_count=item_count,
            )
        return cached, 0.0
    started = time.perf_counter()
    embeddings = encode()
    elapsed = time.perf_counter() - started
    save_started = time.perf_counter()
    save_embedding_cache(path, metadata_path, embeddings, metadata)
    save_seconds = time.perf_counter() - save_started
    if cost_recorder is not None:
        cost_recorder.record_cache_stage(
            name=stage_name,
            category=category,
            cache_hit=False,
            load_seconds=load_seconds,
            compute_seconds=elapsed,
            save_seconds=save_seconds,
            item_count=item_count,
        )
    return embeddings, elapsed


def _json_or_cache(
    *,
    path: Path,
    metadata_path: Path,
    metadata: Mapping[str, Any],
    produce: Callable[[], Mapping[str, Any]],
    cost_recorder: RuntimeCostRecorder | None = None,
    stage_name: str = "json_stage",
    category: str = "online_query",
    item_count: int = 0,
) -> tuple[dict[str, Any], float]:
    load_started = time.perf_counter()
    cached = load_valid_json_cache(path, metadata_path, metadata)
    load_seconds = time.perf_counter() - load_started
    if cached is not None:
        if cost_recorder is not None:
            cost_recorder.record_cache_stage(
                name=stage_name,
                category=category,
                cache_hit=True,
                load_seconds=load_seconds,
                compute_seconds=0.0,
                save_seconds=0.0,
                item_count=item_count,
            )
        return cached, 0.0
    started = time.perf_counter()
    value = dict(produce())
    elapsed = time.perf_counter() - started
    # Ranking dictionaries must retain Faiss/cross-encoder tie order across a
    # JSON cache round-trip; sorted keys would silently change metric results.
    save_started = time.perf_counter()
    save_json_cache(path, metadata_path, value, metadata, sort_keys=False)
    save_seconds = time.perf_counter() - save_started
    if cost_recorder is not None:
        cost_recorder.record_cache_stage(
            name=stage_name,
            category=category,
            cache_hit=False,
            load_seconds=load_seconds,
            compute_seconds=elapsed,
            save_seconds=save_seconds,
            item_count=item_count,
        )
    return value, elapsed


def _module_precision(module: Any) -> str | None:
    candidate = getattr(module, "model", module)
    parameters = getattr(candidate, "parameters", None)
    if not callable(parameters):
        return None
    try:
        return str(next(parameters()).dtype)
    except (StopIteration, AttributeError, TypeError):
        return None


def _run_experiment(
    config: ExperimentConfig,
    *,
    cost_recorder: RuntimeCostRecorder,
    repo_root: Path,
    notebook_sha256: str | None = None,
    source_notebook_sha256: str | None = None,
    selection_manifest_path: str | Path | None = None,
    ranking_replay: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Select on valid or evaluate test using the existing ordered workflow."""

    set_seed(config.seed)
    selected_device = choose_device(config)
    if selected_device.startswith("cuda"):
        try:
            import torch

            cost_recorder.set_cuda_device(torch, selected_device)
        except ImportError:
            pass
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
            hypothesis_only_fusion_weights=tuple(
                selection["e5_hypothesis_only"]["selected"]["weights"]
            ),
            reranker_fusion_weights=tuple(selection["e5_rerank"]["selected"]["weights"]),
            combined_fusion_weights=tuple(selection["e5_hyde_rerank"]["selected"]["weights"]),
        )
    with cost_recorder.phase("dataset_load", "startup"):
        data = load_cosqa(config)
    with cost_recorder.phase("run_population_selection", "startup"):
        run_data = select_run_data(data, config)
    if config.run_mode == "benchmark" and (
        len(run_data.queries) != len(data.queries) or len(run_data.corpus) != len(data.corpus)
    ):
        raise RuntimeError("benchmark mode must use the complete declared CosQA query population and corpus")

    if selection_manifest is not None:
        test_boundary = validate_test_population_boundary(selection_manifest, run_data)
    if config.code_reranker_enabled and ranking_replay is None:
        raise ValueError(
            "the AZH-531 code reranker requires provenance-validated saved E5 candidate rankings"
        )
    if ranking_replay is not None:
        if config.run_mode != "benchmark":
            raise ValueError("ranking replay is available only for complete benchmark populations")
        if ranking_replay.get("split") != config.qrels_split:
            raise ValueError("ranking replay split does not match the configured qrels split")
        if config.code_reranker_enabled and config.qrels_split == "valid":
            if not ranking_replay.get("selection_reference_path"):
                raise ValueError("valid code-reranker replay requires its sealed source manifest")
            loaded_replay = load_validation_ranking_replay(
                ranking_replay["selection_reference_path"],
                run_data,
                config,
            )
        else:
            loaded_replay = load_ranking_replay(
                ranking_replay["sources"],
                run_data,
                config,
                split=config.qrels_split,
                selection_reference_path=ranking_replay.get("selection_reference_path"),
                validation_reference_path=ranking_replay.get("validation_reference_path"),
                test_result_path=ranking_replay.get("test_result_path"),
            )
        if config.code_reranker_enabled:
            loaded_replay = add_code_reranker_rankings(
                loaded_replay,
                config,
                run_data,
                cost_recorder=cost_recorder,
            )
        replay_environment = dict(pre_model_environment)
        replay_environment["model_revisions"] = loaded_replay["revisions"].as_dict()
        if config.code_reranker_enabled:
            replay_environment.setdefault("model_precision", {})["code_reranker"] = (
                loaded_replay["code_reranker_preflight"].get("parameter_dtype")
            )
            replay_environment.setdefault("model_devices", {})["code_reranker"] = (
                config.code_reranker_device
            )
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

    with cost_recorder.phase("model_load_e5", "startup"):
        encoder = E5Encoder(config, batch_observer=cost_recorder.observe_batch)
    e5_revision = _revision_for_configured_model(config.model_revision, encoder.model, "E5")
    with cost_recorder.phase("model_load_hyde", "startup"):
        hyde = HyDEGenerator(config, batch_observer=cost_recorder.observe_batch)
    hyde_revision = hyde.resolved_revision
    with cost_recorder.phase("model_load_reranker", "startup"):
        reranker = CrossEncoderReranker(config, batch_observer=cost_recorder.observe_batch)
    reranker_revision = reranker.resolved_revision
    revisions = ModelRevisions(e5=e5_revision, hyde=hyde_revision, reranker=reranker_revision)
    environment = environment_metadata(config, repo_root=repo_root)
    environment["model_revisions"] = revisions.as_dict()
    environment["model_precision"] = {
        "e5": _module_precision(encoder.model),
        "hyde": _module_precision(hyde.model),
        "reranker": _module_precision(reranker.model),
    }
    environment["model_devices"] = {
        "e5": encoder.device,
        "hyde": hyde.device,
        "reranker": reranker.device,
    }
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
        cost_recorder=cost_recorder,
        stage_name="corpus_embeddings",
        category="offline_corpus_encoding",
        item_count=len(corpus_texts),
    )
    original_query_meta = cache_metadata(
        config, revisions, identity=identity, kind="original_query_embeddings", ids=query_ids, repo_root=repo_root
    )
    original_query_embeddings, original_query_seconds = _embedding_or_cache(
        path=paths.original_query_embeddings,
        metadata_path=paths.original_query_metadata,
        metadata=original_query_meta,
        encode=lambda: encoder.encode_queries(
            list(query_texts.values()), stage="original_query_encoding"
        ),
        cost_recorder=cost_recorder,
        stage_name="original_query_embeddings",
        category="online_query",
        item_count=len(query_texts),
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
        cost_recorder=cost_recorder,
        stage_name="hyde_generation",
        category="query_generation",
        item_count=len(query_texts),
    )
    environment["hyde_fallback_query_count"] = len(hyde.fallback_query_ids)
    hyde.close()
    with cost_recorder.phase("hyde_representation", "online_query"):
        representations = build_hyde_representation_conditions(query_texts, generated)
        hyde_representations = representations["query_plus_hypotheses"]
        hypothesis_only_representations = representations["hypothesis_only"]
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
        encode=lambda: encoder.encode_queries(
            list(hyde_representations.values()), stage="hyde_query_encoding"
        ),
        cost_recorder=cost_recorder,
        stage_name="hyde_query_embeddings",
        category="online_query",
        item_count=len(hyde_representations),
    )
    hypothesis_only_query_meta = cache_metadata(
        config,
        revisions,
        identity=identity,
        kind="hypothesis_only_query_embeddings",
        ids=query_ids,
        repo_root=repo_root,
        representation_sha256=_representation_hash(hypothesis_only_representations),
    )
    hypothesis_only_query_embeddings, hypothesis_only_query_seconds = _embedding_or_cache(
        path=paths.hypothesis_only_query_embeddings,
        metadata_path=paths.hypothesis_only_query_metadata,
        metadata=hypothesis_only_query_meta,
        encode=lambda: encoder.encode_queries(
            list(hypothesis_only_representations.values()),
            stage="hypothesis_only_query_encoding",
        ),
        cost_recorder=cost_recorder,
        stage_name="hypothesis_only_query_embeddings",
        category="online_query",
        item_count=len(hypothesis_only_representations),
    )

    faiss_index: Any | None = None

    def rank_queries(query_embeddings: np.ndarray) -> dict[str, dict[str, float]]:
        nonlocal faiss_index
        if faiss_index is None:
            with cost_recorder.phase("faiss_index_build", "offline_index"):
                faiss_index = build_faiss_index(corpus_embeddings)
        return rank_with_faiss(
            query_embeddings,
            corpus_embeddings,
            query_ids,
            corpus_ids,
            top_k=config.candidate_depth,
            index=faiss_index,
            timing_observer=cost_recorder.observe_batch,
        )

    ranking_ids = query_ids + ["__corpus__"] + corpus_ids
    original_rank_meta = cache_metadata(
        config, revisions, identity=identity, kind="original_rankings", ids=ranking_ids, repo_root=repo_root
    )
    original_rankings, original_ranking_seconds = _json_or_cache(
        path=paths.original_rankings,
        metadata_path=paths.original_rankings_metadata,
        metadata=original_rank_meta,
        produce=lambda: rank_queries(original_query_embeddings),
        cost_recorder=cost_recorder,
        stage_name="original_rankings",
        category="query_search",
        item_count=len(query_ids),
    )
    hyde_rank_meta = cache_metadata(
        config,
        revisions,
        identity=identity,
        kind="hyde_rankings",
        ids=ranking_ids,
        repo_root=repo_root,
        representation_sha256=_representation_hash(hyde_representations),
    )
    hyde_rankings, hyde_ranking_seconds = _json_or_cache(
        path=paths.hyde_rankings,
        metadata_path=paths.hyde_rankings_metadata,
        metadata=hyde_rank_meta,
        produce=lambda: rank_queries(hyde_query_embeddings),
        cost_recorder=cost_recorder,
        stage_name="hyde_rankings",
        category="query_search",
        item_count=len(query_ids),
    )
    hypothesis_only_rank_meta = cache_metadata(
        config,
        revisions,
        identity=identity,
        kind="hypothesis_only_rankings",
        ids=ranking_ids,
        repo_root=repo_root,
        representation_sha256=_representation_hash(hypothesis_only_representations),
    )
    hypothesis_only_rankings, hypothesis_only_ranking_seconds = _json_or_cache(
        path=paths.hypothesis_only_rankings,
        metadata_path=paths.hypothesis_only_rankings_metadata,
        metadata=hypothesis_only_rank_meta,
        produce=lambda: rank_queries(hypothesis_only_query_embeddings),
        cost_recorder=cost_recorder,
        stage_name="hypothesis_only_rankings",
        category="query_search",
        item_count=len(query_ids),
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
        cost_recorder=cost_recorder,
        stage_name="original_reranked",
        category="query_reranking",
        item_count=sum(len(ranking) for ranking in original_rankings.values()),
    )
    if config.run_mode == "benchmark":
        for label, ranking in (
            ("E5", original_rankings),
            ("query-plus-hypothesis HyDE", hyde_rankings),
            ("hypothesis-only HyDE", hypothesis_only_rankings),
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
            "generation_provenance": {
                "path": str(paths.hyde_generations),
                "sha256": sha256_file(paths.hyde_generations),
                "metadata_path": str(paths.hyde_generations_metadata),
                "metadata_sha256": sha256_file(paths.hyde_generations_metadata),
                "generation_content_sha256": _canonical_sha256(generated),
                "query_ids_sha256": sha256_ids(query_ids),
                "query_count": len(generated),
                "shared_across_representations": True,
                "representation_text_sha256": {
                    "query_plus_hypotheses": _representation_hash(hyde_representations),
                    "hypothesis_only": _representation_hash(hypothesis_only_representations),
                },
            },
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
                    (
                        "hyde_hypothesis_only",
                        paths.hypothesis_only_rankings,
                        paths.hypothesis_only_rankings_metadata,
                    ),
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
                "hyde_hypothesis_only": hypothesis_only_rankings,
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
        cost_recorder=cost_recorder,
        stage_name="hyde_fusion",
        category="online_query",
        item_count=len(query_ids),
    )
    hypothesis_only_fused_meta = cache_metadata(
        config,
        revisions,
        identity=identity,
        kind="e5_hypothesis_only_fused_rankings",
        ids=ranking_ids,
        repo_root=repo_root,
    )
    fused_hypothesis_only, hypothesis_only_fusion_seconds = _json_or_cache(
        path=paths.hypothesis_only_fused,
        metadata_path=paths.hypothesis_only_fused_metadata,
        metadata=hypothesis_only_fused_meta,
        produce=lambda: reciprocal_rank_fusion(
            [original_rankings, hypothesis_only_rankings],
            weights=config.hypothesis_only_fusion_weights,
            top_k=config.candidate_depth,
            rrf_k=config.rrf_k,
        ),
        cost_recorder=cost_recorder,
        stage_name="hypothesis_only_fusion",
        category="online_query",
        item_count=len(query_ids),
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
        cost_recorder=cost_recorder,
        stage_name="reranker_fusion",
        category="online_query",
        item_count=len(query_ids),
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
        cost_recorder=cost_recorder,
        stage_name="combined_fusion",
        category="online_query",
        item_count=len(query_ids),
    )
    with cost_recorder.phase("metric_evaluation", "evaluation"):
        metric_by_system: dict[str, Mapping[str, Any]] = {
            "e5": evaluate_ndcg_at_10(run_data.qrels, original_rankings, cutoff=10),
            "e5_hyde": evaluate_ndcg_at_10(run_data.qrels, fused_hyde, cutoff=10),
            "e5_hypothesis_only": evaluate_ndcg_at_10(
                run_data.qrels, fused_hypothesis_only, cutoff=10
            ),
            "e5_rerank": evaluate_ndcg_at_10(run_data.qrels, fused_rerank, cutoff=10),
            "e5_hyde_rerank": evaluate_ndcg_at_10(run_data.qrels, fused_hyde_reranked, cutoff=10),
            "e5_hyde_raw": evaluate_ndcg_at_10(run_data.qrels, hyde_rankings, cutoff=10),
            "e5_hypothesis_only_raw": evaluate_ndcg_at_10(
                run_data.qrels, hypothesis_only_rankings, cutoff=10
            ),
            "e5_rerank_raw": evaluate_ndcg_at_10(run_data.qrels, original_reranked, cutoff=10),
        }
        evaluation_seconds = cost_recorder.phase_timings[-1]["seconds"]
    timings = {
        "corpus_encoding": corpus_seconds,
        "original_query_encoding": original_query_seconds,
        "hyde_generation": hyde_seconds,
        "hyde_query_encoding": hyde_query_seconds,
        "hypothesis_only_query_encoding": hypothesis_only_query_seconds,
        "original_ranking": original_ranking_seconds,
        "hyde_ranking": hyde_ranking_seconds,
        "hypothesis_only_ranking": hypothesis_only_ranking_seconds,
        "original_reranking": original_rerank_seconds,
        "hyde_rrf_fusion": hyde_fusion_seconds,
        "hypothesis_only_rrf_fusion": hypothesis_only_fusion_seconds,
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
    result["hyde_generation_provenance"] = {
        "path": str(paths.hyde_generations),
        "sha256": sha256_file(paths.hyde_generations),
        "metadata_path": str(paths.hyde_generations_metadata),
        "metadata_sha256": sha256_file(paths.hyde_generations_metadata),
        "generation_content_sha256": _canonical_sha256(generated),
        "query_ids_sha256": sha256_ids(query_ids),
        "query_count": len(generated),
        "shared_across_representations": True,
        "representation_text_sha256": {
            "query_plus_hypotheses": _representation_hash(hyde_representations),
            "hypothesis_only": _representation_hash(hypothesis_only_representations),
        },
    }
    if selection_manifest is not None:
        result["selection_manifest"] = {
            "path": str(selection_manifest.get("_path", "")),
            "manifest_sha256": selection_manifest["manifest_sha256"],
            "selection_run_id": selection_manifest["selection_run_id"],
            "selected_weights": {
                name: selection_manifest["sweeps"][name]["selected"]["weights"]
                for name in (
                    "e5_hyde",
                    "e5_hypothesis_only",
                    "e5_rerank",
                    "e5_hyde_rerank",
                )
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
        "e5_hypothesis_only_raw": candidate_contract(hypothesis_only_rankings),
        "e5_hyde": {
            **candidate_contract(fused_hyde),
            "pool_policy": "union_of_e5_and_hyde_top_k_truncated_to_candidate_depth",
        },
        "e5_hypothesis_only": {
            **candidate_contract(fused_hypothesis_only),
            "pool_policy": "union_of_e5_and_hypothesis_only_top_k_truncated_to_candidate_depth",
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
    result["reranked_candidate_pair_count"] = sum(
        len(ranking) for ranking in original_reranked.values()
    )
    result["ranking_artifacts"] = {
        "e5": _ranking_artifact_record(
            original_rankings, path=paths.original_rankings, metadata_path=paths.original_rankings_metadata
        ),
        "query_plus_hypothesis_raw": _ranking_artifact_record(
            hyde_rankings, path=paths.hyde_rankings, metadata_path=paths.hyde_rankings_metadata
        ),
        "hypothesis_only_raw": _ranking_artifact_record(
            hypothesis_only_rankings,
            path=paths.hypothesis_only_rankings,
            metadata_path=paths.hypothesis_only_rankings_metadata,
        ),
        "reranker_raw": _ranking_artifact_record(
            original_reranked,
            path=paths.original_reranked,
            metadata_path=paths.original_reranked_metadata,
        ),
        "e5_query_plus_hypothesis_fused": _ranking_artifact_record(
            fused_hyde, path=paths.hyde_fused, metadata_path=paths.hyde_fused_metadata
        ),
        "e5_hypothesis_only_fused": _ranking_artifact_record(
            fused_hypothesis_only,
            path=paths.hypothesis_only_fused,
            metadata_path=paths.hypothesis_only_fused_metadata,
        ),
        "e5_reranker_fused": _ranking_artifact_record(
            fused_rerank, path=paths.rerank_fused, metadata_path=paths.rerank_fused_metadata
        ),
        "e5_query_plus_hypothesis_reranker_fused": _ranking_artifact_record(
            fused_hyde_reranked,
            path=paths.combined_fused,
            metadata_path=paths.combined_fused_metadata,
        ),
    }
    artifacts = write_comparison_artifacts(
        config,
        result,
        per_query_metrics=_per_query_metric_rows(
            metric_by_system,
            cache_identity=identity,
        ),
        representation_ablation_per_query=_representation_ablation_per_query_rows(
            metric_by_system,
            cache_identity=identity,
        ),
        cache_metadata={
            "corpus": corpus_meta,
            "original_queries": original_query_meta,
            "hyde_generations": hyde_meta,
            "hyde_queries": hyde_query_meta,
            "hypothesis_only_queries": hypothesis_only_query_meta,
            "original_rankings": original_rank_meta,
            "hyde_rankings": hyde_rank_meta,
            "hypothesis_only_rankings": hypothesis_only_rank_meta,
            "original_reranked": original_rerank_meta,
            "hyde_fused": hyde_fused_meta,
            "hypothesis_only_fused": hypothesis_only_fused_meta,
            "rerank_fused": rerank_fused_meta,
            "combined_fused": combined_fused_meta,
        },
    )
    return result, artifacts


def _runtime_cost_csv_rows(costs: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add_row(
        *,
        measurement_kind: str,
        stage: str,
        category: str,
        elapsed_seconds: float,
        cache_status: str = "",
        unit_count: int | None = None,
        unit: str = "",
    ) -> None:
        rows.append(
            {
                "schema_version": costs["schema_version"],
                "run_id": costs["run_id"],
                "cache_identity": costs["cache_identity"],
                "evidence_level": costs["evidence_level"],
                "measurement_kind": measurement_kind,
                "stage": stage,
                "category": category,
                "cache_status": cache_status,
                "elapsed_seconds": float(elapsed_seconds),
                "unit_count": unit_count,
                "unit": unit,
            }
        )

    for item in costs["cache_stages"]:
        for component in ("cache_load", "compute", "cache_save"):
            add_row(
                measurement_kind=component,
                stage=item["name"],
                category=item["category"],
                elapsed_seconds=item[f"{component}_seconds"]
                if component != "compute"
                else item["compute_seconds"],
                cache_status=item["cache_status"],
                unit_count=item["item_count"],
            )
    for item in costs["phase_timings"]:
        add_row(
            measurement_kind="phase",
            stage=item["name"],
            category=item["category"],
            elapsed_seconds=item["seconds"],
        )
    for item in costs["batch_observations"]:
        add_row(
            measurement_kind="batch_latency",
            stage=item["stage"],
            category="model_or_retrieval_batch",
            elapsed_seconds=item["seconds"],
            unit_count=item["unit_count"],
            unit=item["unit"],
        )
    add_row(
        measurement_kind="pipeline_wall",
        stage="pipeline",
        category="end_to_end",
        elapsed_seconds=costs["pipeline_wall_seconds"],
    )
    add_row(
        measurement_kind="online_query_cost",
        stage="online_query_path",
        category="end_to_end_query_cost",
        elapsed_seconds=costs["end_to_end_query_cost"]["online_stage_seconds"],
        unit_count=costs["end_to_end_query_cost"]["query_count"],
        unit="queries",
    )
    return rows


def _write_runtime_cost_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "schema_version",
        "run_id",
        "cache_identity",
        "evidence_level",
        "measurement_kind",
        "stage",
        "category",
        "cache_status",
        "elapsed_seconds",
        "unit_count",
        "unit",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    temporary.replace(path)


def _runtime_costs_with_effective_fusion_weights(
    result: Mapping[str, Any], costs: Mapping[str, Any]
) -> dict[str, Any]:
    """Record the validation-selected weights actually used by this result."""

    updated = dict(costs)
    controls = dict(costs.get("measurement_controls", {}))
    settings = dict(controls.get("experiment_settings", {}))
    component_weights = result.get("fusion", {}).get("component_weights", {})
    setting_by_system = {
        "e5_hyde": "hyde_fusion_weights",
        "e5_hypothesis_only": "hypothesis_only_fusion_weights",
        "e5_rerank": "reranker_fusion_weights",
        "e5_hyde_rerank": "combined_fusion_weights",
        "e5_code_rerank": "code_reranker_fusion_weights",
    }
    for system_id, setting_name in setting_by_system.items():
        if system_id in component_weights:
            settings[setting_name] = list(component_weights[system_id])
    if settings:
        controls["experiment_settings"] = settings
        updated["measurement_controls"] = controls
    return updated


def _persist_runtime_costs(
    result: dict[str, Any],
    artifacts: dict[str, str],
    costs: Mapping[str, Any],
) -> dict[str, str]:
    effective_costs = _runtime_costs_with_effective_fusion_weights(result, costs)
    result_path = artifacts.get("run_result") or artifacts.get("result") or artifacts.get(
        "selection_result"
    )
    run_dir_value = artifacts.get("run_dir") or artifacts.get("selection_run_dir")
    run_dir = Path(run_dir_value or (Path(result_path).parent if result_path else "artifacts/runs/unidentified"))
    run_dir.mkdir(parents=True, exist_ok=True)
    cost_json_path = run_dir / "runtime_costs.json"
    cost_csv_path = run_dir / "runtime_costs.csv"
    _json_write(cost_json_path, effective_costs)
    cost_rows = _runtime_cost_csv_rows(effective_costs)
    _write_runtime_cost_csv(cost_csv_path, cost_rows)

    result["runtime_costs"] = effective_costs
    result["cost_observations"] = {
        "local_model_inference": any(
            item["stage"]
            in {
                "corpus_encoding",
                "original_query_encoding",
                "hyde_query_encoding",
                "hypothesis_only_query_encoding",
                "hyde_generation",
                "cross_encoder_scoring",
            }
            for item in effective_costs["batch_observations"]
        ),
        "gpu_memory_captured": effective_costs["memory"]["gpu_peak_allocated_bytes"] is not None,
        "latency_is_wall_clock_seconds": True,
        "runtime_costs_json": str(cost_json_path),
        "runtime_costs_csv": str(cost_csv_path),
    }
    metadata_value = artifacts.get("run_metadata") or artifacts.get("metadata")
    metadata_path = Path(metadata_value) if metadata_value else run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    metadata["result"] = dict(result)
    metadata.setdefault("artifacts", {})["runtime_costs"] = {
        "json_path": str(cost_json_path),
        "json_sha256": sha256_file(cost_json_path),
        "csv_path": str(cost_csv_path),
        "csv_sha256": sha256_file(cost_csv_path),
        "row_count": len(cost_rows),
    }
    if artifacts.get("run_result"):
        _json_write(Path(artifacts["run_result"]), result)
    if artifacts.get("result"):
        _json_write(Path(artifacts["result"]), result)
    if artifacts.get("selection_result"):
        _json_write(Path(artifacts["selection_result"]), result)
    if artifacts.get("run_metadata"):
        _json_write(Path(artifacts["run_metadata"]), metadata)
    if artifacts.get("metadata"):
        _json_write(Path(artifacts["metadata"]), metadata)
    artifacts["runtime_costs_json"] = str(cost_json_path)
    artifacts["runtime_costs_csv"] = str(cost_csv_path)
    return artifacts


def run_experiment(
    config: ExperimentConfig,
    *,
    repo_root: Path,
    notebook_sha256: str | None = None,
    source_notebook_sha256: str | None = None,
    selection_manifest_path: str | Path | None = None,
    ranking_replay: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Run the experiment and attach measured costs to its existing artifacts."""

    with RuntimeCostRecorder() as recorder:
        result, artifacts = _run_experiment(
            config,
            cost_recorder=recorder,
            repo_root=repo_root,
            notebook_sha256=notebook_sha256,
            source_notebook_sha256=source_notebook_sha256,
            selection_manifest_path=selection_manifest_path,
            ranking_replay=ranking_replay,
        )
    environment = result.get("environment", {})
    index_build_seconds = sum(
        item["seconds"]
        for item in recorder.phase_timings
        if item["name"] == "faiss_index_build"
    )
    costs = recorder.summary(
        run_id=str(result.get("run_id") or "unidentified"),
        cache_identity=str(result.get("cache_identity") or "unidentified"),
        status=str(result.get("status") or "unknown"),
        run_mode=str(result.get("run_mode") or config.run_mode),
        environment=environment,
        config=config,
        query_count=int(result.get("query_count") or 0),
        corpus_count=int(result.get("corpus_count") or 0),
        reranked_pair_count=int(result.get("reranked_candidate_pair_count") or 0),
        model_precision=environment.get("model_precision", {}),
        offline_index_build_seconds=index_build_seconds,
    )
    artifacts = _persist_runtime_costs(result, artifacts, costs)
    return result, artifacts


__all__ = [
    "CODE_VERSION",
    "CODE_RERANKER_MODEL_ID",
    "CODE_RERANKER_MODEL_REVISION",
    "DEFAULT_HYDE_PROMPT",
    "DEFAULT_RRF_K",
    "ExperimentConfig",
    "ModelRevisions",
    "SYSTEM_IDS",
    "blocked_result",
    "build_comparison_result",
    "build_hyde_representation_conditions",
    "cache_metadata",
    "cache_paths",
    "candidate_contract",
    "combine_hypotheses",
    "experiment_identity",
    "experiment_system_ids",
    "load_ranking_replay",
    "load_validation_ranking_replay",
    "load_cosqa",
    "run_code_reranker_preflight",
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
