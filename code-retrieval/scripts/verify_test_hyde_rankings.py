"""Verify the saved held-out query-plus-HyDE ranking without reading qrels."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from run_benchmark_notebook import require_pinned_runtime

from e5_baseline import (  # noqa: E402
    rank_with_faiss,
    set_seed,
    sha256_ids,
)
from e5_hyde import (  # noqa: E402
    E5Encoder,
    HyDEConfig,
    build_expanded_queries,
    environment_metadata,
)


SOURCE_ID = "f09f39783d2d818f"
SOURCE_CACHE = PROJECT_ROOT / "artifacts" / "e5_hyde_rerank" / "cache" / SOURCE_ID
VALIDATION_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "e5_hyde_rerank"
    / "validation"
    / "test_hyde_rrf_0.5_0.5"
)
RANKINGS_PATH = VALIDATION_DIR / "hyde_queryplus_rankings.json"
METADATA_PATH = VALIDATION_DIR / "hyde_queryplus_rankings.metadata.json"


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_shared_environment(
    source: Mapping[str, Any], current: Mapping[str, Any]
) -> None:
    keys = (
        "python",
        "platform",
        "processor",
        "device",
        "packages",
        "torch_version",
        "torch_default_dtype",
        "torch_num_threads",
        "torch_deterministic_algorithms",
        "torch_float32_matmul_precision",
        "torch_cuda_matmul_allow_tf32",
        "cuda_available",
        "cuda_device_count",
        "cuda_device_name",
    )
    differences = [key for key in keys if source.get(key) != current.get(key)]
    if differences:
        raise ValueError("test source runtime mismatch: " + ", ".join(differences))


def main() -> None:
    require_pinned_runtime()
    import torch
    from datasets import load_dataset

    config = HyDEConfig(run_mode="benchmark", qrels_split="test")
    torch.set_num_threads(8)
    set_seed(config.seed)
    environment = environment_metadata(config, repo_root=PROJECT_ROOT)

    generation_path = SOURCE_CACHE / "hyde_generations.json"
    generation_metadata_path = SOURCE_CACHE / "hyde_generations.metadata.json"
    corpus_path = SOURCE_CACHE / "corpus_embeddings.npy"
    corpus_metadata_path = SOURCE_CACHE / "corpus_embeddings.metadata.json"
    for path in (generation_path, generation_metadata_path, corpus_path, corpus_metadata_path, RANKINGS_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)

    source_generation_metadata = _read_json(generation_metadata_path)
    source_corpus_metadata = _read_json(corpus_metadata_path)
    source_generation_config = source_generation_metadata.get("config", {})
    source_corpus_config = source_corpus_metadata.get("config", {})
    for key, expected in (
        ("dataset_id", config.dataset_id),
        ("dataset_revision", config.dataset_revision),
        ("qrels_split", "test"),
        ("hyde_model_id", config.generator_id),
        ("hyde_model_revision", config.generator_revision),
        ("hyde_prompt", config.prompt_template),
        ("hyde_num_hypotheses", config.num_hypotheses),
        ("hyde_temperature", config.temperature),
        ("hyde_max_new_tokens", config.max_new_tokens),
        ("hyde_do_sample", config.do_sample),
        ("hyde_stop_behavior", config.stop_behavior),
        ("hyde_fallback_prompts", list(config.fallback_prompts)),
    ):
        if source_generation_config.get(key) != expected:
            raise ValueError(f"test generation source has incompatible {key}")
    for key, expected in (
        ("dataset_id", config.dataset_id),
        ("dataset_revision", config.dataset_revision),
        ("corpus_config", config.corpus_config),
        ("corpus_split", config.corpus_split),
        ("model_id", config.model_id),
        ("model_revision", config.model_revision),
        ("passage_prefix", config.passage_prefix),
        ("max_seq_length", config.max_seq_length),
        ("normalize_embeddings", config.normalize_embeddings),
        ("batch_size", config.batch_size),
        ("device", config.device),
    ):
        if source_corpus_config.get(key) != expected:
            raise ValueError(f"test corpus source has incompatible {key}")
    _assert_shared_environment(source_generation_metadata.get("execution_environment", {}), environment)
    _assert_shared_environment(source_corpus_metadata.get("execution_environment", {}), environment)

    generation_record = _read_json(generation_path)
    saved = _read_json(RANKINGS_PATH)
    if set(generation_record) != set(saved):
        raise ValueError("test query IDs differ between generation and saved ranking artifacts")
    # The generation JSON is canonicalized by key; the ranking map preserves
    # the source query order recorded by its generation cache metadata.
    query_ids = list(saved)
    if source_generation_metadata.get("ids_sha256") != sha256_ids(query_ids):
        raise ValueError("test generation metadata has a different ordered query set")
    if source_generation_metadata.get("count") != len(query_ids) or len(query_ids) != 500:
        raise ValueError("test generation is not the complete 500-query split")

    raw_corpus = load_dataset(
        config.dataset_id,
        config.corpus_config,
        split=config.corpus_split,
        revision=config.dataset_revision,
    )
    raw_queries = load_dataset(
        config.dataset_id,
        config.queries_config,
        split=config.queries_split,
        revision=config.dataset_revision,
    )
    corpus_ids = [str(row["_id"]) for row in raw_corpus]
    all_queries = {str(row["_id"]): str(row["text"]) for row in raw_queries}
    if source_corpus_metadata.get("ids_sha256") != sha256_ids(corpus_ids):
        raise ValueError("test corpus cache does not match the ordered pinned corpus")
    queries = {query_id: all_queries[query_id] for query_id in query_ids}
    hypotheses = {
        query_id: str(generation_record[query_id][0]).strip()
        for query_id in query_ids
        if len(generation_record[query_id]) == 1
    }
    if set(hypotheses) != set(query_ids) or any(not text for text in hypotheses.values()):
        raise ValueError("test generation must contain one non-empty hypothesis per query")
    expanded_queries = build_expanded_queries(
        queries, hypotheses, strategy="query_plus_hypotheses"
    )
    corpus_embeddings = np.load(corpus_path, allow_pickle=False)
    if (
        corpus_embeddings.ndim != 2
        or corpus_embeddings.shape[0] != len(corpus_ids)
        or not np.isfinite(corpus_embeddings).all()
    ):
        raise ValueError("test corpus embedding cache has invalid shape or values")

    encoder = E5Encoder(config)
    query_embeddings = encoder.encode_queries(
        [expanded_queries[query_id] for query_id in query_ids]
    )
    recomputed = rank_with_faiss(
        query_embeddings,
        corpus_embeddings,
        query_ids,
        corpus_ids,
        top_k=config.candidate_depth,
    )
    if recomputed != saved:
        mismatch = next(query_id for query_id in query_ids if recomputed[query_id] != saved[query_id])
        raise RuntimeError(
            f"saved held-out HyDE ranking differs from real pinned inference at query {mismatch}"
        )

    metadata = {
        "benchmark_evidence": True,
        "dataset": {
            "id": config.dataset_id,
            "revision": config.dataset_revision,
            "queries_split": config.queries_split,
            "corpus_split": config.corpus_split,
            "qrels_loaded": False,
        },
        "query_count": len(query_ids),
        "corpus_count": len(corpus_ids),
        "candidate_depth": config.candidate_depth,
        "evaluator": "not run; this step only verifies rankings and does not inspect qrels",
        "system": {
            "first_stage_model_id": config.model_id,
            "first_stage_model_revision": config.model_revision,
            "query_prefix": config.query_prefix,
            "passage_prefix": config.passage_prefix,
            "max_seq_length": config.max_seq_length,
            "normalize_embeddings": config.normalize_embeddings,
            "combination_strategy": "query_plus_hypotheses",
            "query_separator": "\\n\\n",
            "ranking": "exact Faiss IndexFlatIP over normalized embeddings",
        },
        "generation": {
            "model_id": config.generator_id,
            "revision": config.generator_revision,
            "prompt_template": config.prompt_template,
            "num_hypotheses": config.num_hypotheses,
            "temperature": config.temperature,
            "do_sample": config.do_sample,
            "max_new_tokens": config.max_new_tokens,
            "fallback_prompts": list(config.fallback_prompts),
            "source_generation_path": str(generation_path.relative_to(PROJECT_ROOT)),
            "source_generation_sha256": _sha256(generation_path),
            "source_generation_metadata_sha256": _sha256(generation_metadata_path),
        },
        "source_corpus_embeddings": {
            "cache_identity": SOURCE_ID,
            "path": str(corpus_path.relative_to(PROJECT_ROOT)),
            "sha256": _sha256(corpus_path),
            "metadata_path": str(corpus_metadata_path.relative_to(PROJECT_ROOT)),
            "metadata_sha256": _sha256(corpus_metadata_path),
        },
        "runtime": environment,
        "ranking_sha256": _sha256(RANKINGS_PATH),
        "exact_recomputed_ranking_match": True,
        "recomputed_query_count": len(query_ids),
        "synthetic_scores": False,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "verified",
        "query_count": len(query_ids),
        "corpus_count": len(corpus_ids),
        "candidate_depth": config.candidate_depth,
        "exact_recomputed_ranking_match": True,
        "ranking_sha256": metadata["ranking_sha256"],
        "metadata_path": str(METADATA_PATH),
    }, indent=2))


if __name__ == "__main__":
    main()
