"""Select E5-anchored RRF weights from complete CosQA validation rankings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from run_benchmark_notebook import require_pinned_runtime

from e5_hyde_rerank import (
    ExperimentConfig,
    evaluate_ndcg_at_10,
    load_cosqa,
    reciprocal_rank_fusion,
    sha256_ids,
    select_run_data,
)


VALIDATION_DIR = PROJECT_ROOT / "artifacts" / "e5_hyde_rerank" / "validation"
RERANK_DIR = VALIDATION_DIR / "rerank_depth1000"
HYDE_RESULT = PROJECT_ROOT / "artifacts" / "e5_hyde" / "result.json"
RERANK_RESULT = PROJECT_ROOT / "artifacts" / "e5_rerank" / "result.json"
OUTPUT_PATH = VALIDATION_DIR / "weight_selection.json"
RRF_K = 60
CANDIDATE_DEPTH = 1000
WEIGHT_GRID = tuple(round(index / 20, 2) for index in range(21))


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _ranking_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_rankings(
    name: str,
    rankings: Mapping[str, Mapping[str, float]],
    query_ids: set[str],
) -> None:
    if set(rankings) != query_ids:
        raise ValueError(f"{name} query IDs do not match the complete validation qrels set")
    depths = {len(candidates) for candidates in rankings.values()}
    if depths != {CANDIDATE_DEPTH}:
        raise ValueError(f"{name} rankings must contain exactly {CANDIDATE_DEPTH} candidates per query")


def _sweep(
    qrels: Mapping[str, Mapping[str, int]],
    e5_rankings: Mapping[str, Mapping[str, float]],
    component_rankings: Mapping[str, Mapping[str, float]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, dict[str, float]]]:
    results: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    selected_rankings: dict[str, dict[str, float]] | None = None
    for e5_weight in WEIGHT_GRID:
        component_weight = round(1.0 - e5_weight, 2)
        fused = reciprocal_rank_fusion(
            [e5_rankings, component_rankings],
            weights=(e5_weight, component_weight),
            top_k=CANDIDATE_DEPTH,
            rrf_k=RRF_K,
        )
        metric = evaluate_ndcg_at_10(qrels, fused, cutoff=10)
        results.append(
            {
                "weights": [e5_weight, component_weight],
                "ndcg_at_10": float(metric["ndcg_at_10"]),
            }
        )
        result = results[-1]
        if selected is None or (
            result["ndcg_at_10"], result["weights"][0]
        ) > (selected["ndcg_at_10"], selected["weights"][0]):
            selected = result
            selected_rankings = fused
    if selected is None or selected_rankings is None:
        raise RuntimeError("weight grid produced no configurations")
    return results, selected, selected_rankings


def _save_rankings(path: Path, rankings: Mapping[str, Mapping[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(rankings, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    require_pinned_runtime()
    config = ExperimentConfig(run_mode="benchmark", qrels_split="valid")
    data = load_cosqa(config)
    run_data = select_run_data(data, config)
    if len(run_data.queries) != len(data.queries) or len(run_data.qrels) != len(data.qrels):
        raise RuntimeError("validation selection requires the complete declared valid split")
    query_ids = set(run_data.qrels)

    hyde_result = _read_json(HYDE_RESULT)
    if hyde_result.get("status") != "benchmark" or not hyde_result.get("benchmark_evidence"):
        raise ValueError("HyDE input must be a complete benchmark artifact")
    if hyde_result.get("dataset", {}).get("splits", {}).get("qrels") != "valid":
        raise ValueError("HyDE input rankings must come from the complete valid qrels split")
    if (
        hyde_result.get("candidate_depth") != CANDIDATE_DEPTH
        or hyde_result.get("query_count") != len(run_data.queries)
        or hyde_result.get("corpus_count") != len(run_data.corpus)
        or hyde_result.get("qrels_query_count") != len(run_data.qrels)
        or hyde_result.get("dataset", {}).get("revision") != config.dataset_revision
    ):
        raise ValueError("HyDE artifact does not match the complete pinned validation dataset")
    hyde_paths = hyde_result["cache_paths"]
    hyde_e5_path = PROJECT_ROOT / hyde_paths["original_rankings"]
    hyde_raw_path = PROJECT_ROOT / hyde_paths["rankings"]
    ranking_ids = list(run_data.queries) + ["__corpus__"] + list(run_data.corpus)
    for name, metadata_key, kind in (
        ("E5 baseline", "original_rankings_metadata", "original_rankings"),
        ("HyDE", "rankings_metadata", "rankings"),
    ):
        metadata_path = PROJECT_ROOT / hyde_paths[metadata_key]
        metadata = _read_json(metadata_path)
        if (
            metadata.get("identity") != hyde_result.get("cache_identity")
            or metadata.get("kind") != kind
            or metadata.get("config", {}).get("qrels_split") != "valid"
            or metadata.get("config", {}).get("candidate_depth") != CANDIDATE_DEPTH
            or metadata.get("ids_sha256") != sha256_ids(ranking_ids)
        ):
            raise ValueError(f"{name} ranking metadata does not match the valid run identity")
    rerank_result = _read_json(RERANK_RESULT) if RERANK_RESULT.is_file() else {}
    if rerank_result.get("dataset", {}).get("splits", {}).get("qrels") == "valid":
        rerank_paths = rerank_result.get("cache_paths", {})
        rerank_e5_path = PROJECT_ROOT / rerank_paths["rankings"]
        rerank_raw_path = PROJECT_ROOT / rerank_paths["reranked_rankings"]
        rerank_artifact = RERANK_RESULT
        reranker_validation_summary_path = RERANK_RESULT
    else:
        rerank_e5_path = RERANK_DIR / "e5_rankings.json"
        rerank_raw_path = RERANK_DIR / "e5_reranked.json"
        rerank_artifact = RERANK_DIR
        reranker_validation_summary_path = RERANK_DIR / "validation_results.json"
        if not reranker_validation_summary_path.is_file():
            raise FileNotFoundError(reranker_validation_summary_path)
        reranker_validation_summary = _read_json(reranker_validation_summary_path)
        expected_reranker_validation = {
            "qrels_split": "valid",
            "dataset_revision": config.dataset_revision,
            "candidate_depth": CANDIDATE_DEPTH,
            "query_count": len(run_data.queries),
            "corpus_count": len(run_data.corpus),
            "qrels_query_count": len(run_data.qrels),
            "qrels_judgment_count": sum(map(len, run_data.qrels.values())),
            "e5_revision": config.model_revision,
            "reranker_revision": config.reranker_model_revision,
            "device": config.reranker_device,
            "torch_threads": 8,
        }
        mismatches = [
            key
            for key, expected in expected_reranker_validation.items()
            if reranker_validation_summary.get(key) != expected
        ]
        if mismatches:
            raise ValueError(
                "reranker validation metadata does not match the complete pinned valid run: "
                + ", ".join(mismatches)
            )
    source_paths = {
        "e5_for_hyde": hyde_e5_path,
        "hyde_query_plus": hyde_raw_path,
        "e5_for_rerank": rerank_e5_path,
        "reranker_raw": rerank_raw_path,
    }
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing complete validation rankings:\n" + "\n".join(missing))

    rankings = {name: _read_json(path) for name, path in source_paths.items()}
    for name, ranking in rankings.items():
        _validate_rankings(name, ranking, query_ids)
    if rankings["e5_for_hyde"] != rankings["e5_for_rerank"]:
        raise ValueError("standalone E5 rankings differ across the HyDE and reranker validations")
    for query_id, candidates in rankings["e5_for_rerank"].items():
        if set(candidates) != set(rankings["reranker_raw"][query_id]):
            raise ValueError(f"reranker changed the E5 candidate set for {query_id}")

    hyde_grid, hyde_selected, hyde_fused = _sweep(
        run_data.qrels, rankings["e5_for_hyde"], rankings["hyde_query_plus"]
    )
    rerank_grid, rerank_selected, rerank_fused = _sweep(
        run_data.qrels, rankings["e5_for_rerank"], rankings["reranker_raw"]
    )
    combined_grid, combined_selected, combined_fused = _sweep(
        run_data.qrels, hyde_fused, rerank_fused
    )

    baseline_score = float(evaluate_ndcg_at_10(run_data.qrels, rankings["e5_for_hyde"], cutoff=10)["ndcg_at_10"])
    if hyde_grid[-1]["ndcg_at_10"] != baseline_score or rerank_grid[-1]["ndcg_at_10"] != baseline_score:
        raise RuntimeError("one-source RRF does not reproduce the E5 baseline under the evaluator tie policy")
    if combined_grid[-1]["ndcg_at_10"] != hyde_selected["ndcg_at_10"]:
        raise RuntimeError("one-source combined RRF does not reproduce its selected HyDE component")

    fused_paths = {
        "e5_hyde": VALIDATION_DIR / "e5_hyde_fused_rankings.json",
        "e5_rerank": VALIDATION_DIR / "e5_rerank_fused_rankings.json",
        "e5_hyde_rerank": VALIDATION_DIR / "e5_hyde_rerank_fused_rankings.json",
    }
    for name, path in fused_paths.items():
        _save_rankings(path, {
            "e5_hyde": hyde_fused,
            "e5_rerank": rerank_fused,
            "e5_hyde_rerank": combined_fused,
        }[name])

    baseline = evaluate_ndcg_at_10(run_data.qrels, rankings["e5_for_hyde"], cutoff=10)
    raw_metrics = {
        "e5_hyde": evaluate_ndcg_at_10(run_data.qrels, rankings["hyde_query_plus"], cutoff=10),
        "e5_rerank": evaluate_ndcg_at_10(run_data.qrels, rankings["reranker_raw"], cutoff=10),
    }
    environment = hyde_result.get("environment", {})
    payload = {
        "selection_qrels_split": "valid",
        "selection_metric": "nDCG@10",
        "rrf_k": RRF_K,
        "candidate_depth": CANDIDATE_DEPTH,
        "query_count": len(run_data.queries),
        "corpus_count": len(run_data.corpus),
        "qrels_query_count": len(run_data.qrels),
        "qrels_judgment_count": sum(map(len, run_data.qrels.values())),
        "evaluator": "coir.beir.retrieval.evaluation.EvaluateRetrieval",
        "evaluator_package": config.evaluator_package,
        "evaluation_order_policy": "descending_score_then_saved_insertion_order_with_unique_ordinal_evaluation_scores",
        "baseline_ndcg_at_10": float(baseline["ndcg_at_10"]),
        "raw_component_ndcg_at_10": {
            name: float(metric["ndcg_at_10"]) for name, metric in raw_metrics.items()
        },
        "weight_grid": list(WEIGHT_GRID),
        "tie_break_policy": "prefer_the_larger_e5_weight_when_validation_scores_tie",
        "systems": {
            "e5_hyde": {"grid": hyde_grid, "selected": hyde_selected},
            "e5_rerank": {"grid": rerank_grid, "selected": rerank_selected},
            "e5_hyde_rerank": {"grid": combined_grid, "selected": combined_selected},
        },
        "input_rankings": {
            name: {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": _ranking_sha256(path)}
            for name, path in source_paths.items()
        },
        "fused_rankings": {
            name: str(path.relative_to(PROJECT_ROOT)) for name, path in fused_paths.items()
        },
        "hyde_cache_identity": hyde_result.get("cache_identity"),
        "reranker_cache_identity": rerank_result.get("cache_identity"),
        "reranker_validation_artifact": str(rerank_artifact.relative_to(PROJECT_ROOT)),
        "reranker_validation_summary": {
            "path": str(reranker_validation_summary_path.relative_to(PROJECT_ROOT)),
            "sha256": _ranking_sha256(reranker_validation_summary_path),
        },
        "environment": environment,
        "real_model_inference_inputs": True,
        "synthetic_scores": False,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    reranker_summary_path = RERANK_DIR / "validation_results.json"
    if reranker_summary_path.is_file():
        reranker_summary = _read_json(reranker_summary_path)
        reranker_summary.setdefault(
            "legacy_metric_snapshot_before_order_preserving_evaluation",
            {
                "baseline_ndcg_at_10": reranker_summary.get("baseline_ndcg_at_10"),
                "raw_reranker_ndcg_at_10": reranker_summary.get("raw_reranker_ndcg_at_10"),
                "weight_grid": reranker_summary.get("weight_grid"),
                "best": reranker_summary.get("best"),
                "tie_policy": "raw scores; evaluator treats equal-score documents as ties",
            },
        )
        reranker_summary.update(
            {
                "baseline_ndcg_at_10": payload["baseline_ndcg_at_10"],
                "raw_reranker_ndcg_at_10": payload["raw_component_ndcg_at_10"]["e5_rerank"],
                "evaluation_order_policy": payload["evaluation_order_policy"],
                "weight_grid": [
                    {
                        "e5_weight": item["weights"][0],
                        "reranker_weight": item["weights"][1],
                        "ndcg_at_10": item["ndcg_at_10"],
                    }
                    for item in rerank_grid
                ],
                "best": {
                    "e5_weight": rerank_selected["weights"][0],
                    "reranker_weight": rerank_selected["weights"][1],
                    "ndcg_at_10": rerank_selected["ndcg_at_10"],
                },
                "selection_artifact": str(OUTPUT_PATH.relative_to(PROJECT_ROOT)),
                "benchmark_evidence": True,
                "real_model_inference_inputs": True,
                "synthetic_scores": False,
                "evaluator": "coir.beir.retrieval.evaluation.EvaluateRetrieval",
                "evaluator_package": config.evaluator_package,
                "e5_rankings_sha256": _ranking_sha256(source_paths["e5_for_rerank"]),
                "reranker_rankings_sha256": _ranking_sha256(source_paths["reranker_raw"]),
            }
        )
        reranker_summary_path.write_text(
            json.dumps(reranker_summary, indent=2) + "\n", encoding="utf-8"
        )
        payload["reranker_validation_summary"]["sha256"] = _ranking_sha256(
            reranker_summary_path
        )
        OUTPUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT_PATH),
        "baseline_ndcg_at_10": payload["baseline_ndcg_at_10"],
        "systems": {
            name: system["selected"] for name, system in payload["systems"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
