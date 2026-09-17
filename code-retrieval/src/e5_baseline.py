"""Reusable pieces for the CosQA E5 baseline experiment.

The notebook is intentionally the ordered experiment entry point.  This module
keeps the data contract, cache validation, ranking, and artifact serialization
testable without loading a model or fabricating benchmark output.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


CODE_VERSION = "e5-baseline-v1"
DEFAULT_DATASET_REVISION = "0846fa3b963a21bead36e9fab61451fc83b777a6"
DEFAULT_MODEL_REVISION = "f52bf8ec8c7124536f0efb74aca902b2995e5bcd"


@dataclass(frozen=True)
class BaselineConfig:
    """Explicit controls for one baseline execution."""

    dataset_id: str = "CoIR-Retrieval/cosqa"
    dataset_revision: str = DEFAULT_DATASET_REVISION
    queries_config: str = "queries"
    queries_split: str = "queries"
    corpus_config: str = "corpus"
    corpus_split: str = "corpus"
    qrels_config: str = "default"
    qrels_split: str = "test"
    model_id: str = "intfloat/e5-base-v2"
    model_revision: str = DEFAULT_MODEL_REVISION
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
    max_seq_length: int = 512
    batch_size: int = 32
    candidate_depth: int = 10
    normalize_embeddings: bool = True
    seed: int = 42
    device: str = "auto"
    run_mode: str = "smoke"
    smoke_queries: int = 8
    smoke_corpus: int = 256
    cache_dir: str = "artifacts/e5_baseline/cache"
    artifact_dir: str = "artifacts/e5_baseline"
    evaluator_package: str = "coir-eval==0.7.0"

    def __post_init__(self) -> None:
        if self.run_mode not in {"smoke", "benchmark"}:
            raise ValueError("run_mode must be 'smoke' or 'benchmark'")
        if self.max_seq_length <= 0 or self.batch_size <= 0:
            raise ValueError("max_seq_length and batch_size must be positive")
        if self.candidate_depth <= 0:
            raise ValueError("candidate_depth must be positive")
        if self.smoke_queries <= 0 or self.smoke_corpus <= 0:
            raise ValueError("smoke limits must be positive")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CosQAData:
    """Normalized COIR records while preserving source identifiers."""

    corpus: dict[str, dict[str, str]]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]
    schema: dict[str, Any]


@dataclass
class RunData:
    """Data selected for either the smoke or complete benchmark run."""

    corpus: dict[str, dict[str, str]]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]
    exclusions: list[dict[str, str]] = field(default_factory=list)


def _json_default(value: Any) -> str:
    return str(value)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_ids(ids: Iterable[str]) -> str:
    return sha256_text(canonical_json(list(ids)))


def git_commit(repo_root: Path | None = None) -> str | None:
    """Return the current repository commit when available."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def set_seed(seed: int) -> None:
    """Set the seeds used by the local Python/NumPy/Torch execution."""

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def choose_device(config: BaselineConfig) -> str:
    if config.device != "auto":
        return config.device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def package_versions(packages: Sequence[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def environment_metadata(config: BaselineConfig, repo_root: Path | None = None) -> dict[str, Any]:
    """Collect the execution environment without making a benchmark claim."""

    metadata: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "device": choose_device(config),
        "packages": package_versions(
            [
                "coir-eval",
                "datasets",
                "faiss-cpu",
                "numpy",
                "pytrec-eval-terrier",
                "sentence-transformers",
                "torch",
                "transformers",
            ]
        ),
        "git_commit": git_commit(repo_root),
    }
    try:
        import torch

        metadata["torch_version"] = torch.__version__
        metadata["torch_num_threads"] = int(torch.get_num_threads())
        metadata["cuda_available"] = bool(torch.cuda.is_available())
        metadata["cuda_device_count"] = int(torch.cuda.device_count())
        metadata["cuda_device_name"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
    except ImportError:
        metadata["torch_version"] = None
        metadata["torch_num_threads"] = None
        metadata["cuda_available"] = False
        metadata["cuda_device_count"] = 0
        metadata["cuda_device_name"] = None
    return metadata


def _dataset_schema(dataset: Any) -> dict[str, Any]:
    features = getattr(dataset, "features", None)
    return {
        "columns": list(getattr(dataset, "column_names", [])),
        "features": str(features),
    }


def _row_schema(rows: Any) -> dict[str, Any]:
    return {
        "columns": list(getattr(rows, "column_names", [])),
        "features": str(getattr(rows, "features", None)),
    }


def _source_id(row: Mapping[str, Any], label: str) -> str:
    for key in ("_id", "id", "query-id", "corpus-id"):
        if key in row and row[key] is not None:
            value = str(row[key])
            if value:
                return value
    raise ValueError(f"{label} row has no supported identifier field: {sorted(row)}")


def _text(row: Mapping[str, Any], label: str) -> str:
    value = row.get("text")
    if value is None:
        raise ValueError(f"{label} row has no text field: {sorted(row)}")
    value = str(value).strip()
    if not value:
        raise ValueError(f"{label} row has an empty text field")
    return value


def normalize_cosqa_records(
    corpus_rows: Iterable[Mapping[str, Any]],
    query_rows: Iterable[Mapping[str, Any]],
    qrel_rows: Iterable[Mapping[str, Any]],
    *,
    schema: Mapping[str, Any] | None = None,
) -> CosQAData:
    """Normalize the observed CosQA rows into the COIR/BEIR contract."""

    corpus: dict[str, dict[str, str]] = {}
    for row in corpus_rows:
        doc_id = _source_id(row, "corpus")
        text = _text(row, "corpus")
        previous = corpus.get(doc_id)
        if previous is not None and previous["text"] != text:
            raise ValueError(f"duplicate corpus ID with different text: {doc_id}")
        corpus[doc_id] = {"text": text}

    all_queries: dict[str, str] = {}
    for row in query_rows:
        query_id = _source_id(row, "query")
        text = _text(row, "query")
        previous = all_queries.get(query_id)
        if previous is not None and previous != text:
            raise ValueError(f"duplicate query ID with different text: {query_id}")
        all_queries[query_id] = text

    qrels: dict[str, dict[str, int]] = {}
    for row in qrel_rows:
        if "query-id" not in row or "corpus-id" not in row or "score" not in row:
            raise ValueError(f"qrels row is missing a required field: {sorted(row)}")
        query_id = str(row["query-id"])
        corpus_id = str(row["corpus-id"])
        score = int(row["score"])
        qrels.setdefault(query_id, {})[corpus_id] = score

    if not corpus or not all_queries or not qrels:
        raise ValueError("CosQA corpus, queries, and qrels must all be non-empty")

    missing_queries = sorted(set(qrels) - set(all_queries))
    missing_docs = sorted(
        {doc_id for rels in qrels.values() for doc_id in rels} - set(corpus)
    )
    if missing_queries:
        raise ValueError(f"qrels reference missing queries: {missing_queries[:5]}")
    if missing_docs:
        raise ValueError(f"qrels reference missing corpus IDs: {missing_docs[:5]}")

    queries = {query_id: all_queries[query_id] for query_id in qrels}
    return CosQAData(
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        schema=dict(schema or {}),
    )


def load_cosqa(
    config: BaselineConfig,
    *,
    load_dataset_fn: Callable[..., Any] | None = None,
) -> CosQAData:
    """Load and validate the actual Hugging Face CosQA configs and splits."""

    if load_dataset_fn is None:
        from datasets import load_dataset

        load_dataset_fn = load_dataset

    loader_kwargs = {"revision": config.dataset_revision}
    corpus_rows = load_dataset_fn(
        config.dataset_id,
        config.corpus_config,
        split=config.corpus_split,
        **loader_kwargs,
    )
    query_rows = load_dataset_fn(
        config.dataset_id,
        config.queries_config,
        split=config.queries_split,
        **loader_kwargs,
    )
    qrel_rows = load_dataset_fn(
        config.dataset_id,
        config.qrels_config,
        split=config.qrels_split,
        **loader_kwargs,
    )
    schema = {
        "corpus": _row_schema(corpus_rows),
        "queries": _row_schema(query_rows),
        "qrels": _row_schema(qrel_rows),
        "dataset_id": config.dataset_id,
        "dataset_revision": config.dataset_revision,
        "configs": {
            "corpus": config.corpus_config,
            "queries": config.queries_config,
            "qrels": config.qrels_config,
        },
        "splits": {
            "corpus": config.corpus_split,
            "queries": config.queries_split,
            "qrels": config.qrels_split,
        },
        "corpus_title_policy": "excluded; COIR paper-compatible text-only input",
    }
    return normalize_cosqa_records(corpus_rows, query_rows, qrel_rows, schema=schema)


def select_run_data(data: CosQAData, config: BaselineConfig) -> RunData:
    """Select a labeled smoke subset or the complete declared test data."""

    if config.run_mode == "benchmark":
        return RunData(
            corpus=dict(data.corpus),
            queries=dict(data.queries),
            qrels={query_id: dict(rels) for query_id, rels in data.qrels.items()},
        )

    selected_query_ids = list(data.queries)[: config.smoke_queries]
    selected_qrels = {
        query_id: dict(data.qrels[query_id]) for query_id in selected_query_ids
    }
    selected_corpus_ids = list(data.corpus)[: config.smoke_corpus]
    for rels in selected_qrels.values():
        for corpus_id in rels:
            if corpus_id not in selected_corpus_ids:
                selected_corpus_ids.append(corpus_id)
    selected_corpus_ids = list(dict.fromkeys(selected_corpus_ids))
    selected_corpus = {corpus_id: data.corpus[corpus_id] for corpus_id in selected_corpus_ids}
    exclusions = [
        {
            "type": "smoke_subset",
            "reason": "smoke execution is intentionally not benchmark evidence",
            "queries_excluded": str(len(data.queries) - len(selected_query_ids)),
            "corpus_excluded": str(len(data.corpus) - len(selected_corpus)),
        }
    ]
    return RunData(
        corpus=selected_corpus,
        queries={query_id: data.queries[query_id] for query_id in selected_query_ids},
        qrels=selected_qrels,
        exclusions=exclusions,
    )


def run_identity(config: BaselineConfig, *, repo_root: Path | None = None) -> str:
    payload = {
        "code_version": CODE_VERSION,
        "config": config.as_dict(),
        "git_commit": git_commit(repo_root),
    }
    return sha256_text(canonical_json(payload))[:16]


def cache_paths(config: BaselineConfig, identity: str) -> dict[str, Path]:
    root = Path(config.cache_dir) / identity
    return {
        "root": root,
        "corpus_embeddings": root / "corpus_embeddings.npy",
        "corpus_metadata": root / "corpus_embeddings.metadata.json",
        "query_embeddings": root / "query_embeddings.npy",
        "query_metadata": root / "query_embeddings.metadata.json",
        "rankings": root / "rankings.json",
        "rankings_metadata": root / "rankings.metadata.json",
    }


def expected_cache_metadata(
    config: BaselineConfig,
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


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _atomic_write_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array)
    temporary.replace(path)


def load_valid_embedding_cache(
    data_path: Path,
    metadata_path: Path,
    expected_metadata: Mapping[str, Any],
) -> np.ndarray | None:
    """Load only a cache whose full identity and shape match the request."""

    if not data_path.exists() or not metadata_path.exists():
        return None
    try:
        actual_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if any(actual_metadata.get(key) != value for key, value in expected_metadata.items()):
            return None
        array = np.load(data_path, allow_pickle=False)
        if array.ndim != 2 or array.shape[0] != int(expected_metadata["count"]):
            return None
        if actual_metadata.get("shape") not in (None, list(array.shape)):
            return None
        if not np.isfinite(array).all():
            return None
        return np.asarray(array, dtype=np.float32)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def save_embedding_cache(
    data_path: Path,
    metadata_path: Path,
    embeddings: np.ndarray,
    metadata: Mapping[str, Any],
) -> None:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or not np.isfinite(embeddings).all():
        raise ValueError("embeddings must be a finite 2D array")
    metadata = dict(metadata)
    if embeddings.shape[0] != int(metadata["count"]):
        raise ValueError("embedding row count does not match cache metadata")
    metadata["shape"] = list(embeddings.shape)
    _atomic_write_npy(data_path, embeddings)
    _atomic_write_json(metadata_path, metadata)


def load_valid_json_cache(
    data_path: Path,
    metadata_path: Path,
    expected_metadata: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not data_path.exists() or not metadata_path.exists():
        return None
    try:
        actual_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if actual_metadata != dict(expected_metadata):
            return None
        value = json.loads(data_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            return None
        return value
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def save_json_cache(
    data_path: Path,
    metadata_path: Path,
    value: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> None:
    _atomic_write_json(data_path, dict(value))
    _atomic_write_json(metadata_path, dict(metadata))


class E5Encoder:
    """Thin wrapper that makes E5's query/passage input contract explicit."""

    def __init__(self, config: BaselineConfig):
        from sentence_transformers import SentenceTransformer

        self.config = config
        self.device = choose_device(config)
        self.model = SentenceTransformer(
            config.model_id,
            revision=config.model_revision,
            device=self.device,
        )
        self.model.max_seq_length = config.max_seq_length

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            raise ValueError("cannot encode an empty text collection")
        embeddings = self.model.encode(
            list(texts),
            batch_size=self.config.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize_embeddings,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
            raise ValueError(f"unexpected embedding shape: {embeddings.shape}")
        if not np.isfinite(embeddings).all():
            raise ValueError("model returned non-finite embeddings")
        return embeddings

    def encode_queries(self, queries: Sequence[str]) -> np.ndarray:
        return self._encode([self.config.query_prefix + text for text in queries])

    def encode_corpus(self, passages: Sequence[str]) -> np.ndarray:
        return self._encode([self.config.passage_prefix + text for text in passages])


def rank_with_faiss(
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray,
    query_ids: Sequence[str],
    corpus_ids: Sequence[str],
    *,
    top_k: int,
) -> dict[str, dict[str, float]]:
    """Exact inner-product search using the COIR paper's Flat index family."""

    if len(query_ids) != query_embeddings.shape[0]:
        raise ValueError("query ID count does not match query embeddings")
    if len(corpus_ids) != corpus_embeddings.shape[0]:
        raise ValueError("corpus ID count does not match corpus embeddings")
    if not query_ids or not corpus_ids:
        raise ValueError("query and corpus collections must be non-empty")
    if query_embeddings.shape[1] != corpus_embeddings.shape[1]:
        raise ValueError("query and corpus embedding dimensions differ")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    try:
        faiss = importlib.import_module("faiss")
    except ImportError as exc:
        raise RuntimeError("faiss-cpu is required for the exact COIR baseline") from exc

    query_embeddings = np.ascontiguousarray(query_embeddings, dtype=np.float32)
    corpus_embeddings = np.ascontiguousarray(corpus_embeddings, dtype=np.float32)
    index = faiss.IndexFlatIP(corpus_embeddings.shape[1])
    index.add(corpus_embeddings)
    scores, positions = index.search(query_embeddings, min(top_k, len(corpus_ids)))

    rankings: dict[str, dict[str, float]] = {}
    for query_index, query_id in enumerate(query_ids):
        row: dict[str, float] = {}
        for score, position in zip(scores[query_index], positions[query_index]):
            if int(position) < 0:
                continue
            row[str(corpus_ids[int(position)])] = float(score)
        rankings[str(query_id)] = row
    return rankings


def evaluate_ndcg_at_10(
    qrels: Mapping[str, Mapping[str, int]],
    rankings: Mapping[str, Mapping[str, float]],
    *,
    cutoff: int = 10,
) -> dict[str, Any]:
    """Call the official COIR evaluator implementation at the primary cutoff."""

    if cutoff != 10:
        raise ValueError("this baseline's primary evaluator cutoff is fixed at 10")
    try:
        evaluator_module = importlib.import_module("coir.beir.retrieval.evaluation")
        evaluator = evaluator_module.EvaluateRetrieval
    except ImportError as exc:
        raise RuntimeError(
            "coir-eval==0.7.0 is required for the official COIR evaluator"
        ) from exc

    # The COIR evaluator accepts plain nested dictionaries and may remove
    # identical query/document IDs in-place. Copy the rankings so the cached
    # ranking remains the original retrieval output.
    safe_rankings = {query_id: dict(scores) for query_id, scores in rankings.items()}
    ndcg, average_precision, recall, precision = evaluator.evaluate(
        {query_id: dict(scores) for query_id, scores in qrels.items()},
        safe_rankings,
        [cutoff],
    )
    key = f"NDCG@{cutoff}"
    if key not in ndcg:
        raise RuntimeError(f"COIR evaluator did not return {key}: {ndcg}")
    return {
        "ndcg_at_10": float(ndcg[key]),
        "ndcg": ndcg,
        "map": average_precision,
        "recall": recall,
        "precision": precision,
        "evaluator": "coir.beir.retrieval.evaluation.EvaluateRetrieval",
        "evaluator_package": "coir-eval==0.7.0",
        "cutoff": cutoff,
    }


def build_result(
    config: BaselineConfig,
    data: CosQAData,
    run_data: RunData,
    metric: Mapping[str, Any],
    *,
    identity: str,
    environment: Mapping[str, Any],
    timings: Mapping[str, float],
    repo_root: Path | None = None,
) -> dict[str, Any]:
    status = "smoke" if config.run_mode == "smoke" else "benchmark"
    return {
        "system_id": "e5",
        "system": {
            "first_stage_retriever": config.model_id,
            "query_expansion": None,
            "reranker": None,
        },
        "status": status,
        "benchmark_evidence": status == "benchmark",
        "metric": "nDCG@10",
        "ndcg_at_10": float(metric["ndcg_at_10"]),
        "evaluator": dict(metric),
        "query_count": len(run_data.queries),
        "corpus_count": len(run_data.corpus),
        "qrels_query_count": len(run_data.qrels),
        "qrels_judgment_count": sum(len(rels) for rels in run_data.qrels.values()),
        "exclusions": run_data.exclusions,
        "candidate_depth": config.candidate_depth,
        "data_schema": data.schema,
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
        "model": {
            "id": config.model_id,
            "revision": config.model_revision,
            "query_prefix": config.query_prefix,
            "passage_prefix": config.passage_prefix,
            "max_seq_length": config.max_seq_length,
            "normalize_embeddings": config.normalize_embeddings,
            "batch_size": config.batch_size,
            "device": choose_device(config),
        },
        "seed": config.seed,
        "run_mode": config.run_mode,
        "cache_identity": identity,
        "code_version": CODE_VERSION,
        "git_commit": git_commit(repo_root),
        "environment": dict(environment),
        "timings_seconds": dict(timings),
        "limitations": (
            ["Smoke subset is wiring evidence only; do not compare its score as a benchmark result."]
            if status == "smoke"
            else []
        ),
        "artifact_provenance": {
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": "code-retrieval/notebooks/e5_baseline_experiment.ipynb",
            "real_model_inference": True,
            "synthetic_scores": False,
        },
    }


def write_result_artifacts(
    config: BaselineConfig,
    result: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    artifact_dir = Path(config.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    result_path = artifact_dir / "result.json"
    metadata_path = artifact_dir / "metadata.json"
    _atomic_write_json(result_path, dict(result))
    _atomic_write_json(metadata_path, dict(metadata or result))
    return {"result": str(result_path), "metadata": str(metadata_path)}
