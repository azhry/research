"""Export per-query metric and qualitative retrieval traces from a benchmark result."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from e5_baseline import BaselineConfig, load_cosqa, select_run_data  # noqa: E402


PAPER_E5_COSQA_NDCG_AT_10 = 0.3259
ANALYSIS_SCHEMA_VERSION = 1
TOP_K = 10
TEXT_PREVIEW_CHARS = 400
COMBINED_CACHE_FILES = {
    "e5": ("original_rankings.json", "original_rankings"),
    "e5_hyde": ("e5_hyde_fused.json", "e5_hyde_fused_rankings"),
    "e5_rerank": ("e5_rerank_fused.json", "e5_rerank_fused_rankings"),
    "e5_hyde_rerank": ("e5_hyde_rerank_fused.json", "e5_hyde_rerank_fused_rankings"),
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object at {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_artifact_path(value: str, *, result_path: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"ranking artifact referenced by result is missing: {path}")
    return path


def _combined_cache_root(result: Mapping[str, Any]) -> Path | None:
    identity = str(result.get("cache_identity") or "")
    if not identity or any(character not in "0123456789abcdef" for character in identity.lower()):
        return None
    return PROJECT_ROOT / "artifacts" / "e5_hyde_rerank" / "cache" / identity


def _is_combined_cache_path(path: Path, result: Mapping[str, Any]) -> bool:
    cache_root = _combined_cache_root(result)
    return cache_root is not None and path.resolve().is_relative_to(cache_root.resolve())


def _validate_cache_provenance(path: Path, *, identity: str, kind: str) -> None:
    metadata_path = path.with_name(f"{path.stem}.metadata.json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"cache provenance metadata is missing: {metadata_path}")
    metadata = _read_json(metadata_path)
    if metadata.get("identity") != identity or metadata.get("kind") != kind:
        raise ValueError(
            f"cache provenance mismatch for {path}: expected identity={identity!r}, kind={kind!r}"
        )


def _ranking_paths(result: Mapping[str, Any]) -> dict[str, str]:
    cache_paths = result.get("cache_paths") or {}
    source_candidates = [
        result.get("source_artifacts") or {},
        cache_paths.get("source_artifacts") or {},
    ]
    systems: dict[str, str] = {}

    if result.get("system_ids"):
        baseline = next(
            (
                source.get("e5_original", {}).get("path")
                for source in source_candidates
                if source.get("e5_original", {}).get("path")
            ),
            None,
        )
        fused = cache_paths.get("fused_rankings") or {}
        if baseline:
            systems["e5"] = baseline
        for system_id in ("e5_hyde", "e5_rerank", "e5_hyde_rerank"):
            if fused.get(system_id):
                systems[system_id] = fused[system_id]
        cache_root = _combined_cache_root(result)
        if cache_root is not None:
            for system_id, (filename, _) in COMBINED_CACHE_FILES.items():
                systems.setdefault(system_id, str(cache_root / filename))
    elif result.get("system_id") == "e5":
        if cache_paths.get("rankings"):
            systems["e5"] = cache_paths["rankings"]
    elif result.get("system_id") == "e5_hyde":
        if cache_paths.get("original_rankings"):
            systems["e5"] = cache_paths["original_rankings"]
        if cache_paths.get("fused_rankings"):
            systems["e5_hyde"] = cache_paths["fused_rankings"]
    elif result.get("experiment_id") == "e5_rerank":
        if cache_paths.get("rankings"):
            systems["e5"] = cache_paths["rankings"]
        if cache_paths.get("fused_rankings"):
            systems["e5_rerank"] = cache_paths["fused_rankings"]

    if not systems:
        raise ValueError("result does not identify any persisted scored rankings")
    required_systems: set[str] = set()
    if result.get("system_ids"):
        required_systems = {"e5", "e5_hyde", "e5_rerank", "e5_hyde_rerank"}
    elif result.get("system_id") == "e5_hyde":
        required_systems = {"e5", "e5_hyde"}
    elif result.get("experiment_id") == "e5_rerank":
        required_systems = {"e5", "e5_rerank"}
    missing = required_systems - systems.keys()
    if missing:
        raise ValueError(f"result is missing ranking artifacts for: {sorted(missing)}")
    return systems


def _candidate_depth(result: Mapping[str, Any]) -> int | None:
    value = result.get("candidate_depth")
    if value is not None:
        return int(value)
    fusion = result.get("fusion") or {}
    if isinstance(fusion, Mapping) and fusion.get("candidate_depth") is not None:
        return int(fusion["candidate_depth"])
    rows = result.get("results") or []
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, Mapping) and row.get("candidate_depth") is not None:
            return int(row["candidate_depth"])
    systems = result.get("systems") or {}
    if isinstance(systems, Mapping):
        for row in systems.values():
            if isinstance(row, Mapping) and row.get("candidate_depth") is not None:
                return int(row["candidate_depth"])
    return None


def _ndcg_at_k(
    qrels: Mapping[str, int], ranking: Mapping[str, float], *, cutoff: int = TOP_K
) -> float:
    def dcg(relevances: list[int]) -> float:
        return sum(
            (2**relevance - 1) / math.log2(rank + 2)
            for rank, relevance in enumerate(relevances[:cutoff])
        )

    ordered_docs = sorted(ranking.items(), key=lambda item: -float(item[1]))[:cutoff]
    observed = [int(qrels.get(document_id, 0)) for document_id, _ in ordered_docs]
    ideal = sorted((int(value) for value in qrels.values()), reverse=True)[:cutoff]
    denominator = dcg(ideal)
    return dcg(observed) / denominator if denominator else 0.0


def _result_score(result: Mapping[str, Any], system_id: str) -> float | None:
    rows = result.get("results") or []
    for row in rows:
        if row.get("system_id") == system_id:
            return float(row["ndcg_at_10"])
    systems = result.get("systems") or {}
    if system_id in systems:
        return float(systems[system_id]["ndcg_at_10"])
    if system_id == "e5" and result.get("system_id") == "e5_hyde":
        diagnostics = result.get("component_diagnostics") or {}
        if isinstance(diagnostics, dict) and diagnostics.get("e5_baseline_ndcg_at_10") is not None:
            return float(diagnostics["e5_baseline_ndcg_at_10"])
    if result.get("system_id") == system_id:
        return float(result["ndcg_at_10"])
    return None


def _hyde_text(result: Mapping[str, Any]) -> dict[str, str]:
    cache_paths = result.get("cache_paths") or {}
    path_value = None
    if result.get("system_ids"):
        for source in (
            result.get("source_artifacts") or {},
            cache_paths.get("source_artifacts") or {},
        ):
            path_value = source.get("hyde_generations", {}).get("path")
            if path_value:
                break
        if not path_value:
            cache_root = _combined_cache_root(result)
            candidate = cache_root / "hyde_generations.json" if cache_root is not None else None
            if candidate is not None and candidate.is_file():
                path_value = str(candidate)
    else:
        path_value = cache_paths.get("hypotheses") or cache_paths.get("hyde_generations")
    if not path_value:
        return {}
    path = _safe_artifact_path(path_value, result_path=Path("."))
    if result.get("system_ids") and _is_combined_cache_path(path, result):
        _validate_cache_provenance(
            path,
            identity=str(result.get("cache_identity") or ""),
            kind="hyde_generations",
        )
    raw = _read_json(path)
    normalized: dict[str, str] = {}
    for query_id, value in raw.items():
        if isinstance(value, list):
            normalized[str(query_id)] = "\n".join(str(item) for item in value)
        else:
            normalized[str(query_id)] = str(value)
    return normalized


def _load_data(result: Mapping[str, Any]):
    dataset = result.get("dataset") or {}
    splits = dataset.get("splits") or {}
    configs = dataset.get("configs") or {}
    config = BaselineConfig(
        dataset_id=str(dataset.get("id", "CoIR-Retrieval/cosqa")),
        dataset_revision=str(
            dataset.get("revision", "0846fa3b963a21bead36e9fab61451fc83b777a6")
        ),
        queries_config=str(configs.get("queries", "queries")),
        queries_split=str(splits.get("queries", "queries")),
        corpus_config=str(configs.get("corpus", "corpus")),
        corpus_split=str(splits.get("corpus", "corpus")),
        qrels_config=str(configs.get("qrels", "default")),
        qrels_split=str(splits.get("qrels", "test")),
        candidate_depth=int(_candidate_depth(result) or TOP_K),
        run_mode="benchmark" if result.get("status") == "benchmark" else "smoke",
    )
    data = load_cosqa(config)
    run_data = select_run_data(data, config)
    return config, data, run_data


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "query_id",
        "system_id",
        "ndcg_at_10",
        "delta_vs_e5",
        "relevant_document_count",
        "relevant_retrieved_at_10",
        "first_relevant_rank",
        "query_count",
        "cache_identity",
        "ranking_sha256",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export(result_path: Path, output_dir: Path) -> dict[str, Any]:
    result_path = result_path.resolve()
    result = _read_json(result_path)
    if result.get("status") != "benchmark":
        raise ValueError("analysis exports require a complete benchmark result")
    if result.get("run_mode") != "benchmark" or not result.get("benchmark_evidence"):
        raise ValueError("result is not marked as complete benchmark evidence")

    config, data, run_data = _load_data(result)
    if len(run_data.queries) != len(data.queries) or len(run_data.corpus) != len(data.corpus):
        raise ValueError("analysis export refuses an incomplete benchmark population")
    expected_queries = int(result.get("query_count", 0))
    expected_corpus = int(result.get("corpus_count", 0))
    if len(run_data.queries) != expected_queries or len(run_data.corpus) != expected_corpus:
        raise ValueError("loaded populations do not match the benchmark result")

    paths = _ranking_paths(result)
    rankings: dict[str, dict[str, dict[str, float]]] = {}
    ranking_hashes: dict[str, str] = {}
    for system_id, path_value in paths.items():
        path = _safe_artifact_path(path_value, result_path=result_path)
        if result.get("system_ids") and _is_combined_cache_path(path, result):
            expected_kind = COMBINED_CACHE_FILES[system_id][1]
            _validate_cache_provenance(
                path,
                identity=str(result.get("cache_identity") or ""),
                kind=expected_kind,
            )
        rankings[system_id] = _read_json(path)
        ranking_hashes[system_id] = _sha256(path)
        if set(rankings[system_id]) != set(run_data.queries):
            raise ValueError(f"{system_id} rankings do not cover the complete query set")

    if "e5" not in rankings:
        raise ValueError("analysis export needs the E5 baseline ranking")
    system_scores = {system_id: _result_score(result, system_id) for system_id in rankings}
    query_scores: dict[str, dict[str, float]] = {}
    for system_id, system_rankings in rankings.items():
        query_scores[system_id] = {
            query_id: _ndcg_at_k(run_data.qrels.get(query_id, {}), system_rankings[query_id])
            for query_id in run_data.queries
        }
        mean_score = sum(query_scores[system_id].values()) / expected_queries
        reported_score = system_scores[system_id]
        if reported_score is not None and abs(mean_score - reported_score) > 1e-5:
            raise ValueError(
                f"{system_id} rankings average to {mean_score:.8f}, but result reports "
                f"{reported_score:.8f}"
            )
        system_scores[system_id] = mean_score
    baseline_score = system_scores.get("e5")
    if baseline_score is None:
        raise ValueError("result does not report or resolve an E5 baseline score")

    metric_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    hypotheses = _hyde_text(result)
    for query_id, query_text in run_data.queries.items():
        qrels = run_data.qrels.get(query_id, {})
        case: dict[str, Any] = {
            "query_id": query_id,
            "query": query_text,
            "hyde_hypothesis": hypotheses.get(query_id),
            "qrels_relevant_document_count": sum(int(value) > 0 for value in qrels.values()),
            "systems": {},
        }
        for system_id, system_rankings in rankings.items():
            ranking = system_rankings[query_id]
            ordered = sorted(ranking.items(), key=lambda item: -float(item[1]))
            query_ndcg = query_scores[system_id][query_id]
            retrieved_relevance = [int(qrels.get(document_id, 0)) for document_id, _ in ordered[:TOP_K]]
            relevant_positions = [index + 1 for index, value in enumerate(retrieved_relevance) if value > 0]
            metric_rows.append(
                {
                    "query_id": query_id,
                    "system_id": system_id,
                    "ndcg_at_10": query_ndcg,
                    "delta_vs_e5": query_ndcg - query_scores["e5"][query_id],
                    "relevant_document_count": sum(int(value) > 0 for value in qrels.values()),
                    "relevant_retrieved_at_10": len(relevant_positions),
                    "first_relevant_rank": min(relevant_positions) if relevant_positions else None,
                    "query_count": expected_queries,
                    "cache_identity": result.get("cache_identity"),
                    "ranking_sha256": ranking_hashes[system_id],
                }
            )
            items: list[dict[str, Any]] = []
            for rank, (document_id, score) in enumerate(ordered[:TOP_K], start=1):
                document = run_data.corpus.get(document_id, {})
                items.append(
                    {
                        "rank": rank,
                        "document_id": document_id,
                        "score": float(score),
                        "relevance": int(qrels.get(document_id, 0)),
                        "title": document.get("title", ""),
                        "language": document.get("language", ""),
                        "text_preview": str(document.get("text", ""))[:TEXT_PREVIEW_CHARS],
                    }
                )
            case["systems"][system_id] = {
                "aggregate_ndcg_at_10": system_scores[system_id],
                "query_ndcg_at_10": query_ndcg,
                "top_10": items,
            }
        case_rows.append(case)

    means = {
        system_id: sum(query_scores[system_id].values()) / expected_queries
        for system_id in rankings
    }
    for system_id, mean in means.items():
        aggregate = system_scores.get(system_id)
        if aggregate is None or abs(mean - aggregate) > 1e-5:
            raise ValueError(
                f"per-query nDCG mean for {system_id} ({mean:.8f}) does not match "
                f"the official aggregate ({aggregate})"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "per_query_metrics.csv"
    cases_path = output_dir / "qualitative_cases.jsonl"
    metadata_path = output_dir / "analysis_metadata.json"
    _write_csv(metrics_path, metric_rows)
    with cases_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in case_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    metadata = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "evidence_level": "benchmark",
        "result_path": str(result_path),
        "result_sha256": _sha256(result_path),
        "cache_identity": result.get("cache_identity"),
        "dataset_id": config.dataset_id,
        "dataset_revision": config.dataset_revision,
        "qrels_split": config.qrels_split,
        "query_count": expected_queries,
        "corpus_count": expected_corpus,
        "candidate_depth": _candidate_depth(result),
        "evaluator": result.get("evaluator", result.get("evaluator_package")),
        "systems": {
            system_id: {
                "reported_ndcg_at_10": system_scores[system_id],
                "per_query_mean_ndcg_at_10": means[system_id],
                "ranking_path": str(_safe_artifact_path(paths[system_id], result_path=result_path)),
                "ranking_sha256": ranking_hashes[system_id],
            }
            for system_id in rankings
        },
        "paper_reference": {
            "source": "https://arxiv.org/pdf/2407.02883",
            "table": 3,
            "system": "E5-base on CosQA",
            "ndcg_at_10": PAPER_E5_COSQA_NDCG_AT_10,
            "local_e5_absolute_delta": baseline_score - PAPER_E5_COSQA_NDCG_AT_10,
            "note": "Published reference; protocol differences must be disclosed.",
        },
        "artifacts": {
            "per_query_metrics": metrics_path.name,
            "qualitative_cases": cases_path.name,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "metadata": str(metadata_path),
        "per_query_metrics": str(metrics_path),
        "qualitative_cases": str(cases_path),
        "query_count": expected_queries,
        "systems": list(rankings),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, help="Path to the completed benchmark result JSON")
    parser.add_argument("--output-dir", required=True, help="Analysis artifact directory")
    args = parser.parse_args()
    result_path = Path(args.result)
    if not result_path.is_absolute():
        result_path = PROJECT_ROOT / result_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    artifacts = export(result_path, output_dir)
    print(json.dumps(artifacts, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
