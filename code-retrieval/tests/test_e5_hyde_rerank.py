import json

from e5_baseline import CosQAData, RunData, load_valid_json_cache, save_json_cache
from e5_hyde_rerank import (
    CODE_VERSION,
    ExperimentConfig,
    ModelRevisions,
    SYSTEM_IDS,
    blocked_result,
    build_comparison_result,
    cache_metadata,
    combine_hypotheses,
    experiment_identity,
    rerank_candidate_rankings,
    reciprocal_rank_fusion,
    write_comparison_artifacts,
)


def test_config_keeps_shared_candidate_depth_and_explicit_hyde_controls():
    config = ExperimentConfig()

    assert config.candidate_depth == 1000
    assert CODE_VERSION == "e5-hyde-rerank-v7-cache-order-aware-rrf"
    assert config.hyde_model_id == "google/flan-t5-base"
    assert config.hyde_num_hypotheses == 1
    assert config.hyde_combination_strategy == "hypothesis_only"
    assert config.reranker_model_id == "cross-encoder/ms-marco-MiniLM-L6-v2"
    assert config.rrf_k == 60
    assert config.fusion_weights == (0.90, 0.05, 0.04, 0.01)
    assert isinstance(config.as_dict()["hyde_fallback_prompts"], list)
    assert isinstance(config.as_dict()["fusion_weights"], list)


def test_combine_hypotheses_rejects_empty_generation_and_preserves_strategy():
    assert combine_hypotheses("find a parser", ["a parser implementation"], strategy="hypothesis_only") == (
        "a parser implementation"
    )
    assert combine_hypotheses("find a parser", ["a parser implementation"], strategy="query_plus_hypotheses") == (
        "find a parser\n\na parser implementation"
    )
    try:
        combine_hypotheses("find a parser", ["", "  "], strategy="hypothesis_only")
    except ValueError as error:
        assert "non-empty" in str(error)
    else:
        raise AssertionError("empty HyDE output was accepted")


def test_reranker_reorders_without_changing_the_candidate_set():
    rankings = {"q1": {"d1": 0.8, "d2": 0.7, "d3": 0.6}}
    queries = {"q1": "find a parser"}
    corpus = {
        "d1": {"text": "first"},
        "d2": {"text": "second"},
        "d3": {"text": "third"},
    }

    calls = []

    def deterministic_scores(pairs):
        calls.append(len(pairs))
        return [len(passage) for _, passage in pairs]

    reranked = rerank_candidate_rankings(rankings, queries, corpus, deterministic_scores)

    assert list(reranked["q1"]) == ["d2", "d1", "d3"]
    assert set(reranked["q1"]) == set(rankings["q1"])
    assert calls == [3]


def test_reciprocal_rank_fusion_is_weighted_and_depth_limited():
    original = {"q1": {"d1": 1.0, "d2": 0.9, "d3": 0.8}}
    expanded = {"q1": {"d3": 1.0, "d4": 0.9, "d1": 0.8}}

    fused = reciprocal_rank_fusion(
        [original, expanded],
        weights=(0.9, 0.1),
        top_k=3,
        rrf_k=60,
    )

    assert list(fused["q1"]) == ["d1", "d3", "d2"]
    assert len(fused["q1"]) == 3
    assert fused["q1"]["d1"] > fused["q1"]["d2"]


def test_reciprocal_rank_fusion_recovers_rank_from_cached_scores():
    # JSON cache persistence sorts mapping keys, so iteration order is not
    # retrieval order after a cache round-trip. Fusion must use the stored
    # ranking scores instead of assuming dictionary insertion order.
    cached_ranking = {"d1": 0.2, "d2": 0.9, "d3": 0.8}

    fused = reciprocal_rank_fusion(
        [{"q1": cached_ranking}],
        weights=(1.0,),
        top_k=3,
        rrf_k=60,
    )

    assert list(fused["q1"]) == ["d2", "d3", "d1"]


def test_reciprocal_rank_fusion_preserves_equal_score_source_order():
    source_ranking = {"d2": 0.9, "d1": 0.9, "d3": 0.8}

    fused = reciprocal_rank_fusion(
        [{"q1": source_ranking}],
        weights=(1.0,),
        top_k=3,
        rrf_k=60,
    )

    assert list(fused["q1"]) == ["d2", "d1", "d3"]


def test_ranking_cache_round_trip_preserves_equal_score_order(tmp_path):
    data_path = tmp_path / "rankings.json"
    metadata_path = tmp_path / "rankings.metadata.json"
    ranking = {"q1": {"d2": 0.9, "d1": 0.9, "d3": 0.8}}

    save_json_cache(
        data_path,
        metadata_path,
        ranking,
        {"kind": "rankings"},
        sort_keys=False,
    )

    persisted = json.loads(data_path.read_text(encoding="utf-8"))
    assert list(persisted["q1"]) == ["d2", "d1", "d3"]


def test_experiment_cache_metadata_round_trip_accepts_tuple_controls(tmp_path):
    config = ExperimentConfig(cache_dir=str(tmp_path))
    revisions = ModelRevisions("e5", "hyde", "reranker")
    metadata = cache_metadata(
        config,
        revisions,
        identity="identity",
        kind="rankings",
        ids=["q1", "d1"],
    )
    data_path = tmp_path / "value.json"
    metadata_path = tmp_path / "value.metadata.json"
    save_json_cache(data_path, metadata_path, {"q1": {"d1": 1.0}}, metadata)

    assert load_valid_json_cache(data_path, metadata_path, metadata) == {
        "q1": {"d1": 1.0}
    }


def test_cache_metadata_and_identity_include_all_model_revisions(tmp_path):
    config = ExperimentConfig(cache_dir=str(tmp_path))
    revisions = ModelRevisions(e5="e5-revision", hyde="hyde-revision", reranker="reranker-revision")
    identity = experiment_identity(config, revisions)
    metadata = cache_metadata(
        config,
        revisions,
        identity=identity,
        kind="hyde_generations",
        ids=["q1", "q2"],
    )

    assert metadata["model_revisions"] == revisions.as_dict()
    assert metadata["ids_sha256"]
    changed = experiment_identity(
        config,
        ModelRevisions(e5="e5-revision", hyde="changed", reranker="reranker-revision"),
    )
    assert changed != identity


def test_comparison_result_contains_four_rows_and_deltas():
    config = ExperimentConfig(run_mode="smoke")
    data = CosQAData(
        corpus={"d1": {"text": "code"}},
        queries={"q1": "query"},
        qrels={"q1": {"d1": 1}},
        schema={"columns": ["_id", "text"]},
    )
    run_data = RunData(
        corpus=data.corpus,
        queries=data.queries,
        qrels=data.qrels,
        exclusions=[],
    )
    metrics = {
        "e5": {"ndcg_at_10": 0.5},
        "e5_hyde": {"ndcg_at_10": 0.6},
        "e5_rerank": {"ndcg_at_10": 0.55},
        "e5_hyde_rerank": {"ndcg_at_10": 0.7},
    }
    result = build_comparison_result(
        config,
        data,
        run_data,
        metrics,
        identity="identity",
        revisions=ModelRevisions("e5", "hyde", "reranker"),
        environment={"device": "cpu"},
        timings={},
    )

    assert result["system_ids"] == SYSTEM_IDS
    assert [row["system_id"] for row in result["results"]] == SYSTEM_IDS
    assert result["results"][0]["delta_vs_e5"] == 0.0
    assert abs(result["results"][-1]["delta_vs_e5"] - 0.2) < 1e-9
    assert result["status"] == "smoke"
    assert result["benchmark_evidence"] is False
    assert result["exclusions"] == []
    assert result["limitations"][0]["type"] == "evidence_level"
    assert result["artifact_provenance"]["synthetic_scores"] is False


def test_blocked_result_has_null_scores_and_explicit_blocker():
    result = blocked_result(
        ExperimentConfig(run_mode="benchmark"),
        blocker="model download unavailable",
    )

    assert result["status"] == "blocked"
    assert result["benchmark_evidence"] is False
    assert all(row["ndcg_at_10"] is None for row in result["results"])
    assert any(item["type"] == "blocker" for item in result["limitations"])
    assert result["artifact_provenance"]["synthetic_scores"] is False


def test_write_comparison_artifacts_writes_json_csv_and_metadata(tmp_path):
    config = ExperimentConfig(artifact_dir=str(tmp_path))
    result = blocked_result(config, blocker="test plumbing only")

    paths = write_comparison_artifacts(config, result)

    persisted = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    csv_text = (tmp_path / "comparison.csv").read_text(encoding="utf-8")
    assert set(paths) == {"result", "comparison", "metadata"}
    assert persisted["system_ids"] == SYSTEM_IDS
    assert metadata["result"]["status"] == "blocked"
    assert "system_id,ndcg_at_10,delta_vs_e5" in csv_text
