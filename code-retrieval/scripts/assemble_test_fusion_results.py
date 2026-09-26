"""Apply validation-selected RRF settings once to the complete CosQA test split."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from run_benchmark_notebook import require_pinned_runtime

from e5_hyde_rerank import (  # noqa: E402
    ExperimentConfig,
    ModelRevisions,
    build_comparison_result,
    cache_metadata,
    cache_paths,
    candidate_contract,
    environment_metadata,
    evaluate_ndcg_at_10,
    experiment_identity,
    load_cosqa,
    reciprocal_rank_fusion,
    save_json_cache,
    set_seed,
    select_run_data,
    write_comparison_artifacts,
)


VALIDATION_DIR = PROJECT_ROOT / "artifacts" / "e5_hyde_rerank" / "validation"
SELECTION_PATH = VALIDATION_DIR / "weight_selection.json"
TEST_HYDE_DIR = VALIDATION_DIR / "test_hyde_rrf_0.5_0.5"
TEST_HYDE_RANKING_METADATA = TEST_HYDE_DIR / "hyde_queryplus_rankings.metadata.json"
SOURCE_RUN_ID = "f09f39783d2d818f"
SOURCE_CACHE_DIR = PROJECT_ROOT / "artifacts" / "e5_hyde_rerank" / "cache" / SOURCE_RUN_ID
SOURCE_RESULT_PATH = (
    PROJECT_ROOT / "artifacts" / "e5_hyde_rerank" / "runs" / SOURCE_RUN_ID / "result.json"
)
SOURCE_RUN_METADATA_PATH = (
    PROJECT_ROOT / "artifacts" / "e5_hyde_rerank" / "runs" / SOURCE_RUN_ID / "metadata.json"
)


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_rankings(
    name: str,
    rankings: Mapping[str, Mapping[str, float]],
    query_ids: set[str],
    depth: int,
) -> None:
    if set(rankings) != query_ids:
        raise ValueError(f"{name} query IDs do not match the complete test qrels set")
    if any(len(candidates) != depth for candidates in rankings.values()):
        raise ValueError(f"{name} rankings must contain exactly {depth} candidates per query")


def _assert_shared_environment(source: Mapping[str, Any], current: Mapping[str, Any], name: str) -> None:
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
    mismatches = [key for key in keys if source.get(key) != current.get(key)]
    if mismatches:
        raise ValueError(f"{name} runtime mismatch: " + ", ".join(mismatches))


def main() -> None:
    require_pinned_runtime()
    if not SELECTION_PATH.is_file():
        raise FileNotFoundError("Run the complete valid-split weight selector before reading test qrels")
    selection = _read_json(SELECTION_PATH)
    if selection.get("selection_qrels_split") != "valid":
        raise ValueError("fusion settings must be selected using valid qrels")
    if (
        selection.get("candidate_depth") != 1000
        or selection.get("query_count") != 500
        or selection.get("evaluator_package") != "coir-eval==0.7.0"
    ):
        raise ValueError("fusion selection must cover the complete pinned valid benchmark")
    if not SOURCE_RESULT_PATH.is_file():
        raise FileNotFoundError(f"missing complete source benchmark: {SOURCE_RESULT_PATH}")

    selected = selection["systems"]
    config = ExperimentConfig(
        run_mode="benchmark",
        qrels_split="test",
        rrf_k=int(selection["rrf_k"]),
        hyde_fusion_weights=tuple(selected["e5_hyde"]["selected"]["weights"]),
        reranker_fusion_weights=tuple(selected["e5_rerank"]["selected"]["weights"]),
        combined_fusion_weights=tuple(selected["e5_hyde_rerank"]["selected"]["weights"]),
    )
    data = load_cosqa(config)
    run_data = select_run_data(data, config)
    if len(run_data.queries) != len(data.queries) or len(run_data.qrels) != len(data.qrels):
        raise RuntimeError("test evaluation requires the complete declared test qrels split")
    query_ids = set(run_data.qrels)

    source_result = _read_json(SOURCE_RESULT_PATH)
    source_metadata = _read_json(SOURCE_CACHE_DIR / "original_reranked.metadata.json")
    source_run_metadata = _read_json(SOURCE_RUN_METADATA_PATH)
    if source_result.get("dataset", {}).get("splits", {}).get("qrels") != "test":
        raise ValueError("source model outputs must come from the complete test split")
    for key, expected in (
        ("dataset_revision", config.dataset_revision),
        ("model_revision", config.model_revision),
        ("reranker_model_revision", config.reranker_model_revision),
    ):
        actual = (
            source_metadata.get("config", {}).get(key)
            if key != "dataset_revision"
            else source_metadata.get("config", {}).get("dataset_revision")
        )
        if actual != expected:
            raise ValueError(f"source artifact {key} does not match the selected benchmark configuration")
    import torch

    source_threads = int(source_metadata["execution_environment"]["torch_num_threads"])
    torch.set_num_threads(source_threads)
    set_seed(config.seed)

    source_paths = {
        "e5_original": SOURCE_CACHE_DIR / "original_rankings.json",
        "cross_encoder_raw": SOURCE_CACHE_DIR / "original_reranked.json",
        "hyde_query_plus": TEST_HYDE_DIR / "hyde_queryplus_rankings.json",
    }
    generation_path = SOURCE_CACHE_DIR / "hyde_generations.json"
    metadata_paths = {
        "hyde_generations_metadata": SOURCE_CACHE_DIR / "hyde_generations.metadata.json",
        "hyde_query_plus_metadata": TEST_HYDE_RANKING_METADATA,
    }
    legacy_paths = {
        "legacy_hyde_hypothesis_only": SOURCE_CACHE_DIR / "hyde_rankings.json",
        "legacy_hyde_then_cross_encoder": SOURCE_CACHE_DIR / "hyde_reranked.json",
    }
    all_source_paths = {
        **source_paths,
        **legacy_paths,
        "hyde_generations": generation_path,
        **metadata_paths,
    }
    missing = [str(path) for path in all_source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing complete test rankings:\n" + "\n".join(missing))
    source_rankings = {name: _read_json(path) for name, path in source_paths.items()}
    query_plus_metadata = _read_json(TEST_HYDE_RANKING_METADATA)
    if (
        not query_plus_metadata.get("exact_recomputed_ranking_match")
        or query_plus_metadata.get("dataset", {}).get("qrels_loaded") is not False
        or query_plus_metadata.get("dataset", {}).get("revision") != config.dataset_revision
        or query_plus_metadata.get("query_count") != len(query_ids)
        or query_plus_metadata.get("corpus_count") != len(run_data.corpus)
        or query_plus_metadata.get("candidate_depth") != config.candidate_depth
        or query_plus_metadata.get("ranking_sha256") != _sha256(source_paths["hyde_query_plus"])
    ):
        raise ValueError("held-out HyDE rankings lack matching real-inference provenance")
    _assert_shared_environment(
        source_result.get("environment", {}),
        query_plus_metadata.get("runtime", {}),
        "test HyDE ranking",
    )
    _assert_shared_environment(
        source_result.get("environment", {}),
        source_metadata.get("execution_environment", {}),
        "source test model run",
    )
    for name, ranking in source_rankings.items():
        _validate_rankings(name, ranking, query_ids, config.candidate_depth)
    for query_id, e5_candidates in source_rankings["e5_original"].items():
        if set(e5_candidates) != set(source_rankings["cross_encoder_raw"][query_id]):
            raise ValueError(f"cross-encoder changed the E5 candidate set for {query_id}")

    start = time.perf_counter()
    fused_hyde = reciprocal_rank_fusion(
        [source_rankings["e5_original"], source_rankings["hyde_query_plus"]],
        weights=config.hyde_fusion_weights,
        top_k=config.candidate_depth,
        rrf_k=config.rrf_k,
    )
    fused_rerank = reciprocal_rank_fusion(
        [source_rankings["e5_original"], source_rankings["cross_encoder_raw"]],
        weights=config.reranker_fusion_weights,
        top_k=config.candidate_depth,
        rrf_k=config.rrf_k,
    )
    fused_combined = reciprocal_rank_fusion(
        [fused_hyde, fused_rerank],
        weights=config.combined_fusion_weights,
        top_k=config.candidate_depth,
        rrf_k=config.rrf_k,
    )
    metric_by_system = {
        "e5": evaluate_ndcg_at_10(run_data.qrels, source_rankings["e5_original"], cutoff=10),
        "e5_hyde": evaluate_ndcg_at_10(run_data.qrels, fused_hyde, cutoff=10),
        "e5_rerank": evaluate_ndcg_at_10(run_data.qrels, fused_rerank, cutoff=10),
        "e5_hyde_rerank": evaluate_ndcg_at_10(run_data.qrels, fused_combined, cutoff=10),
        "e5_hyde_raw": evaluate_ndcg_at_10(run_data.qrels, source_rankings["hyde_query_plus"], cutoff=10),
        "e5_rerank_raw": evaluate_ndcg_at_10(run_data.qrels, source_rankings["cross_encoder_raw"], cutoff=10),
    }
    legacy_config = source_run_metadata["cache_metadata"]["corpus"]["config"]
    legacy_weights = tuple(float(weight) for weight in legacy_config["fusion_weights"])
    if (
        legacy_config.get("qrels_split") != "test"
        or legacy_config.get("candidate_depth") != config.candidate_depth
        or legacy_config.get("hyde_combination_strategy") != "hypothesis_only"
        or legacy_weights != (0.90, 0.05, 0.04, 0.01)
    ):
        raise ValueError("historical test reference no longer matches its recorded fixed configuration")
    legacy_rankings = {
        "e5": source_rankings["e5_original"],
        "hyde_hypothesis_only": _read_json(legacy_paths["legacy_hyde_hypothesis_only"]),
        "cross_encoder_raw": source_rankings["cross_encoder_raw"],
        "hyde_then_cross_encoder": _read_json(legacy_paths["legacy_hyde_then_cross_encoder"]),
    }
    for name, ranking in legacy_rankings.items():
        _validate_rankings(name, ranking, query_ids, config.candidate_depth)
    legacy_fused = reciprocal_rank_fusion(
        [
            legacy_rankings["e5"],
            legacy_rankings["hyde_hypothesis_only"],
            legacy_rankings["cross_encoder_raw"],
            legacy_rankings["hyde_then_cross_encoder"],
        ],
        weights=legacy_weights,
        top_k=config.candidate_depth,
        rrf_k=int(legacy_config["rrf_k"]),
    )
    legacy_metric = evaluate_ndcg_at_10(run_data.qrels, legacy_fused, cutoff=10)
    postprocessing_seconds = time.perf_counter() - start

    baseline_reference = float(
        next(row["ndcg_at_10"] for row in source_result["results"] if row["system_id"] == "e5")
    )
    baseline_current = float(metric_by_system["e5"]["ndcg_at_10"])

    revisions = ModelRevisions(
        e5=config.model_revision,
        hyde=config.hyde_model_revision,
        reranker=config.reranker_model_revision,
    )
    environment = environment_metadata(config, repo_root=PROJECT_ROOT)
    _assert_shared_environment(source_result.get("environment", {}), environment, "current test assembly")
    environment["model_revisions"] = revisions.as_dict()
    notebook_path = PROJECT_ROOT / "notebooks" / "e5_hyde_rerank_experiment.ipynb"
    notebook_sha256 = _sha256(notebook_path)
    identity = experiment_identity(
        config,
        revisions,
        repo_root=PROJECT_ROOT,
        notebook_sha256=notebook_sha256,
        environment=environment,
    )
    paths = cache_paths(config, identity)
    corpus_ids = list(run_data.corpus)
    ranking_ids = list(run_data.queries) + ["__corpus__"] + corpus_ids
    fused_rankings = {
        "e5_hyde": fused_hyde,
        "e5_rerank": fused_rerank,
        "e5_hyde_rerank": fused_combined,
    }
    fused_cache_metadata: dict[str, Any] = {}
    for name, ranking, data_path, metadata_path, kind in (
        ("e5_hyde", fused_hyde, paths.hyde_fused, paths.hyde_fused_metadata, "e5_hyde_fused_rankings"),
        ("e5_rerank", fused_rerank, paths.rerank_fused, paths.rerank_fused_metadata, "e5_rerank_fused_rankings"),
        ("e5_hyde_rerank", fused_combined, paths.combined_fused, paths.combined_fused_metadata, "e5_hyde_rerank_fused_rankings"),
    ):
        metadata = cache_metadata(
            config,
            revisions,
            identity=identity,
            kind=kind,
            ids=ranking_ids,
            repo_root=PROJECT_ROOT,
        )
        save_json_cache(data_path, metadata_path, ranking, metadata, sort_keys=False)
        fused_cache_metadata[name] = metadata

    result = build_comparison_result(
        config,
        data,
        run_data,
        metric_by_system,
        identity=identity,
        revisions=revisions,
        environment=environment,
        timings={"fusion_and_official_evaluation": postprocessing_seconds},
        notebook_sha256=notebook_sha256,
        repo_root=PROJECT_ROOT,
    )
    result["candidate_contract"] = {
        "e5": candidate_contract(source_rankings["e5_original"]),
        "e5_hyde_raw": candidate_contract(source_rankings["hyde_query_plus"]),
        "e5_hyde": {
            **candidate_contract(fused_hyde),
            "pool_policy": "union_of_e5_and_hyde_top_k_truncated_to_candidate_depth",
        },
        "e5_rerank_raw": {
            **candidate_contract(source_rankings["cross_encoder_raw"]),
            "pool_policy": "same_candidate_ids_as_e5",
        },
        "e5_rerank": {
            **candidate_contract(fused_rerank),
            "pool_policy": "same_candidate_ids_as_e5",
        },
        "e5_hyde_rerank": {
            **candidate_contract(fused_combined),
            "pool_policy": "union_of_selected_component_rankings_truncated_to_candidate_depth",
        },
    }
    result["artifact_provenance"].update(
        {
            "derived_from_complete_real_model_rankings": True,
            "new_model_inference_in_this_postprocessing_step": False,
            "validation_selection_path": str(SELECTION_PATH.relative_to(PROJECT_ROOT)),
            "source_run_identity": SOURCE_RUN_ID,
        }
    )
    result["source_artifacts"] = {
        name: {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": _sha256(path)}
        for name, path in all_source_paths.items()
    }
    result["source_artifacts"]["source_test_result"] = {
        "path": str(SOURCE_RESULT_PATH.relative_to(PROJECT_ROOT)),
        "sha256": _sha256(SOURCE_RESULT_PATH),
    }
    result["source_artifacts"]["source_test_run_metadata"] = {
        "path": str(SOURCE_RUN_METADATA_PATH.relative_to(PROJECT_ROOT)),
        "sha256": _sha256(SOURCE_RUN_METADATA_PATH),
    }
    result["source_model_environment"] = source_result.get("environment", {})
    result["cache_paths"] = {
        "identity": identity,
        "source_artifacts": result["source_artifacts"],
        "fused_rankings": {
            "e5_hyde": str(paths.hyde_fused),
            "e5_rerank": str(paths.rerank_fused),
            "e5_hyde_rerank": str(paths.combined_fused),
        },
    }
    result["baseline_recomputation"] = {
        "reference_cache_identity": SOURCE_RUN_ID,
        "reference_legacy_ndcg_at_10": baseline_reference,
        "reference_legacy_metric_policy": "raw similarity scores; evaluator treats equal-score documents as ties",
        "current_ndcg_at_10": baseline_current,
        "current_metric_policy": metric_by_system["e5"]["ranking_order_policy"],
        "same_source_ranking_reused": True,
        "source_ranking_sha256": _sha256(source_paths["e5_original"]),
        "legacy_and_current_scores_directly_comparable": False,
        "reason": "Current evaluation preserves the deterministic saved order when source similarities tie.",
    }
    published_legacy_score = float(
        next(
            row["ndcg_at_10"]
            for row in source_result["results"]
            if row["system_id"] == "e5_hyde_rerank"
        )
    )
    legacy_current_score = float(legacy_metric["ndcg_at_10"])
    if abs(legacy_current_score - published_legacy_score) > 1e-12:
        raise RuntimeError(
            "historical fixed-method score did not reproduce under the current ordered-ranking evaluator"
        )
    result["historical_fixed_method_comparison"] = {
        "reference_cache_identity": SOURCE_RUN_ID,
        "reference_system": "legacy four-source weighted RRF",
        "reference_query_representation": "hypothesis_only",
        "reference_weights": list(legacy_weights),
        "reference_published_ndcg_at_10": published_legacy_score,
        "reference_recomputed_ndcg_at_10": legacy_current_score,
        "reference_evaluation_order_policy": legacy_metric["ranking_order_policy"],
        "current_selected_ndcg_at_10": float(metric_by_system["e5_hyde_rerank"]["ndcg_at_10"]),
        "absolute_improvement": float(metric_by_system["e5_hyde_rerank"]["ndcg_at_10"])
        - legacy_current_score,
        "test_weights_were_tuned": False,
    }
    artifacts = write_comparison_artifacts(
        config,
        result,
        cache_metadata={
            "validation_selection": selection,
            "source_test_artifacts": result["source_artifacts"],
            "fused_rankings": fused_cache_metadata,
        },
    )
    print(json.dumps({
        "cache_identity": identity,
        "results": result["results"],
        "component_diagnostics": result["component_diagnostics"],
        "artifact_paths": artifacts,
    }, indent=2))


if __name__ == "__main__":
    main()
