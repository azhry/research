import json
from pathlib import Path

from e5_baseline import CosQAData, RunData, load_valid_json_cache, save_json_cache
from e5_hyde_rerank import (
    CODE_VERSION,
    ExperimentConfig,
    ModelRevisions,
    SYSTEM_IDS,
    blocked_result,
    build_comparison_result,
    build_hyde_representation_conditions,
    cache_metadata,
    candidate_contract,
    combine_hypotheses,
    corpus_fingerprint,
    experiment_identity,
    load_ranking_replay,
    query_population_fingerprint,
    seal_selection_manifest,
    _select_rrf_weights,
    validate_selection_manifest,
    validate_test_population_boundary,
    verify_selection_manifest_integrity,
    rerank_candidate_rankings,
    reciprocal_rank_fusion,
    write_comparison_artifacts,
    _representation_ablation_per_query_rows,
)


def test_config_keeps_shared_candidate_depth_and_explicit_hyde_controls():
    config = ExperimentConfig()

    assert config.candidate_depth == 1000
    assert CODE_VERSION == "e5-hyde-rerank-v17-representation-ablations"
    assert config.hyde_model_id == "google/flan-t5-base"
    assert config.hyde_num_hypotheses == 1
    assert config.hyde_combination_strategy == "query_plus_hypotheses"
    assert config.reranker_model_id == "cross-encoder/ms-marco-MiniLM-L6-v2"
    assert config.rrf_k == 60
    assert config.hyde_fusion_weights == (0.65, 0.35)
    assert config.hypothesis_only_fusion_weights == (0.65, 0.35)
    assert config.reranker_fusion_weights == (0.8, 0.2)
    assert config.combined_fusion_weights == (0.75, 0.25)
    assert isinstance(config.as_dict()["hyde_fallback_prompts"], list)
    assert isinstance(config.as_dict()["hyde_fusion_weights"], list)
    assert isinstance(config.as_dict()["hypothesis_only_fusion_weights"], list)


def test_paired_representations_reuse_the_same_generation_text():
    representations = build_hyde_representation_conditions(
        {"q1": "find a parser", "q2": "sort nested values"},
        {"q2": ["nested sorting example"], "q1": ["parser implementation details"]},
    )

    assert representations["query_plus_hypotheses"]["q1"] == (
        "find a parser\n\nparser implementation details"
    )
    assert representations["hypothesis_only"]["q1"] == "parser implementation details"
    assert list(representations["query_plus_hypotheses"]) == ["q1", "q2"]
    assert list(representations["hypothesis_only"]) == ["q1", "q2"]
    try:
        build_hyde_representation_conditions(
            {"q1": "find a parser"}, {"other": ["parser implementation details"]}
        )
    except ValueError as error:
        assert "must match the query IDs" in str(error)
    else:
        raise AssertionError("a generation for different query IDs was accepted")


def test_paired_experiment_keeps_query_plus_as_the_existing_system_definition():
    try:
        ExperimentConfig(hyde_combination_strategy="hypothesis_only")
    except ValueError as error:
        assert "hypothesis_only is evaluated as a separate ablation" in str(error)
    else:
        raise AssertionError("the existing HyDE system accepted a hypothesis-only definition")


def test_historical_valid_ranking_replay_fails_closed_without_hypothesis_only():
    try:
        load_ranking_replay(
            {},
            RunData(corpus={}, queries={}, qrels={}),
            ExperimentConfig(run_mode="benchmark", qrels_split="valid"),
            split="valid",
        )
    except ValueError as error:
        assert "predates the hypothesis-only representation" in str(error)
    else:
        raise AssertionError("historical ranking replay entered validation selection")


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
    # Ranking caches preserve insertion order, while scores recover rank
    # order when a source returns scores that are not monotone by insertion.
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


def test_candidate_contract_records_ordered_ids_and_population():
    contract = candidate_contract({"q1": {"d2": 0.9, "d1": 0.8}})

    assert contract["query_count"] == 1
    assert contract["minimum_candidates_per_query"] == 2
    assert contract["maximum_candidates_per_query"] == 2
    assert len(contract["ordered_candidate_ids_sha256"]) == 64


def test_selection_sweep_uses_real_evaluator_and_prefers_higher_first_weight_on_ties():
    qrels = {"q1": {"d1": 1}}
    same_ranking = {"q1": {"d1": 3.0, "d2": 2.0, "d3": 1.0}}

    sweep, selected_ranking = _select_rrf_weights(
        qrels,
        same_ranking,
        same_ranking,
        rrf_k=60,
        candidate_depth=3,
    )

    assert len(sweep["grid"]) == 21
    assert sweep["selected"]["weights"] == [1.0, 0.0]
    assert len({row["selection_ndcg_at_10"] for row in sweep["grid"]}) == 1
    assert list(selected_ranking["q1"]) == ["d1", "d2", "d3"]


def test_query_population_fingerprint_normalizes_text_and_preserves_query_ids():
    fingerprint = query_population_fingerprint(
        {"q1": "  Find\tA parser ", "q2": "find a PARSER", "q3": "another query"}
    )

    assert fingerprint["query_count"] == 3
    assert fingerprint["query_ids"] == ["q1", "q2", "q3"]
    assert fingerprint["query_group_count"] == 2
    duplicate_group = next(
        group for group in fingerprint["query_groups"] if group["query_ids"] == ["q1", "q2"]
    )
    assert len(duplicate_group["normalized_text_sha256"]) == 64


def test_selection_manifest_digest_detects_mutation():
    manifest = seal_selection_manifest(
        {"schema": "test", "sweeps": {"e5_hyde": {"selected": [0.65, 0.35]}}}
    )

    assert len(verify_selection_manifest_integrity(manifest)) == 64
    manifest["sweeps"]["e5_hyde"]["selected"] = [0.5, 0.5]
    try:
        verify_selection_manifest_integrity(manifest)
    except ValueError as error:
        assert "integrity digest" in str(error)
    else:
        raise AssertionError("a modified selection manifest passed its digest check")


def test_test_evaluation_rejects_a_missing_selection_manifest(tmp_path):
    try:
        validate_selection_manifest(
            tmp_path / "missing-selection.json",
            ExperimentConfig(run_mode="benchmark", qrels_split="test"),
            repo_root=Path(__file__).resolve().parents[2],
            notebook_sha256=None,
            source_notebook_sha256=None,
            environment={},
        )
    except FileNotFoundError as error:
        assert "required validation-selection manifest" in str(error)
    else:
        raise AssertionError("test evaluation accepted an absent selection manifest")


def test_test_evaluation_rejects_manifest_with_mismatched_controls(tmp_path):
    config = ExperimentConfig(run_mode="benchmark", qrels_split="test")
    manifest_path = tmp_path / "selection.json"
    manifest_path.write_text(
        json.dumps(
            seal_selection_manifest(
                {
                    "schema": "code-retrieval-validation-selection-v2-representation-ablations",
                    "code_version": CODE_VERSION,
                    "status": "complete",
                    "selection_qrels_split": "valid",
                    "target_evaluation_qrels_split": "test",
                    "baseline_parity": {"passed": True},
                    "sweeps": {
                        "e5_hyde": {},
                        "e5_hypothesis_only": {},
                        "e5_rerank": {},
                        "e5_hyde_rerank": {},
                    },
                    "controls": {"runtime_controls": {"candidate_depth": 10}},
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    try:
        validate_selection_manifest(
            manifest_path,
            config,
            repo_root=Path(__file__).resolve().parents[2],
            notebook_sha256=None,
            source_notebook_sha256=None,
            environment={},
        )
    except ValueError as error:
        assert "controls do not match" in str(error)
    else:
        raise AssertionError("test evaluation accepted a mismatched selection manifest")


def test_test_population_boundary_rejects_normalized_query_overlap():
    corpus = {f"d{index}": {"text": f"document {index}"} for index in range(20604)}
    valid_queries = {f"v{index}": f"valid query {index}" for index in range(500)}
    test_queries = {f"t{index}": f"test query {index}" for index in range(500)}
    test_queries["t0"] = "  VALID\tQUERY 0 "
    manifest = {
        "selection_population": {
            "queries": query_population_fingerprint(valid_queries),
            "corpus": corpus_fingerprint(corpus),
        }
    }
    run_data = RunData(
        corpus=corpus,
        queries=test_queries,
        qrels={query_id: {"d0": 1} for query_id in test_queries},
    )

    try:
        validate_test_population_boundary(manifest, run_data)
    except ValueError as error:
        assert "normalized query text overlaps" in str(error)
    else:
        raise AssertionError("test evaluation accepted normalized query overlap")


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
        system_id: {
            "ndcg_at_10": ndcg,
            "map_at_10": ndcg * 0.8,
            "recall_at_10": ndcg * 0.7,
            "precision_at_10": ndcg * 0.07,
            "recall_at_1000": 0.9,
        }
        for system_id, ndcg in {
            "e5": 0.5,
            "e5_hyde": 0.6,
            "e5_rerank": 0.55,
            "e5_hyde_rerank": 0.7,
            "e5_hyde_raw": 0.3,
            "e5_hypothesis_only_raw": 0.35,
            "e5_hypothesis_only": 0.58,
            "e5_rerank_raw": 0.4,
        }.items()
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
    assert result["fusion"]["selection_qrels_split"] == "valid"
    assert result["component_diagnostics"]["e5_hyde_raw"]["ndcg_at_10"] == 0.3
    assert result["component_diagnostics"]["e5_hypothesis_only_raw"]["ndcg_at_10"] == 0.35
    assert result["component_diagnostics"]["e5_rerank_raw"]["ndcg_at_10"] == 0.4
    assert [row["system_id"] for row in result["representation_ablations"]] == [
        "e5",
        "e5_hyde_raw",
        "e5_hypothesis_only_raw",
        "e5_hyde",
        "e5_hypothesis_only",
    ]
    assert result["representation_ablations"][-1]["ndcg_at_10"] == 0.58
    assert result["representation_ablations"][-1]["fusion_weights"] == "[0.65, 0.35]"
    assert result["representation_ablations"][-1]["selection_qrels_split"] == "not_selected"


def test_representation_ablation_export_keeps_paired_metrics_and_headline_rows(tmp_path):
    config = ExperimentConfig(run_mode="smoke", artifact_dir=str(tmp_path))
    metric_values = {
        "e5": 0.5,
        "e5_hyde_raw": 0.3,
        "e5_hypothesis_only_raw": 0.35,
        "e5_hyde": 0.6,
        "e5_hypothesis_only": 0.58,
        "e5_rerank": 0.55,
        "e5_hyde_rerank": 0.7,
        "e5_rerank_raw": 0.4,
    }
    metrics = {}
    for system_id, ndcg in metric_values.items():
        per_query = {}
        for query_id, offset in (("q1", 0.0), ("q2", 0.1)):
            per_query[query_id] = {
                "ndcg_at_10": ndcg + offset,
                "map_at_10": ndcg + offset,
                "recall_at_10": ndcg + offset,
                "precision_at_10": (ndcg + offset) / 10,
                "recall_at_1000": 1.0,
                "first_relevant_rank": 1,
                "relevant_document_count": 1,
                "relevant_retrieved_at_10": 1,
                "relevant_retrieved_at_1000": 1,
            }
        metrics[system_id] = {
            "ndcg_at_10": ndcg,
            "map_at_10": ndcg,
            "recall_at_10": ndcg,
            "precision_at_10": ndcg / 10,
            "recall_at_1000": 1.0,
            "per_query": per_query,
        }
    data = CosQAData(
        corpus={"d1": {"text": "code"}},
        queries={"q1": "query one", "q2": "query two"},
        qrels={"q1": {"d1": 1}, "q2": {"d1": 1}},
        schema={},
    )
    run_data = RunData(corpus=data.corpus, queries=data.queries, qrels=data.qrels)
    result = build_comparison_result(
        config,
        data,
        run_data,
        metrics,
        identity="paired-fixture",
        revisions=ModelRevisions("e5", "hyde", "reranker"),
        environment={"device": "cpu"},
        timings={},
    )
    paired_rows = _representation_ablation_per_query_rows(
        metrics, cache_identity="paired-fixture"
    )
    paths = write_comparison_artifacts(
        config,
        result,
        representation_ablation_per_query=paired_rows,
    )

    headline = Path(paths["run_comparison"]).read_text(encoding="utf-8")
    ablation = Path(paths["representation_ablation"]).read_text(encoding="utf-8")
    paired = Path(paths["representation_ablation_per_query"]).read_text(encoding="utf-8")
    metadata = json.loads(Path(paths["run_metadata"]).read_text(encoding="utf-8"))
    assert headline.count("\n") == 5
    assert "e5_hypothesis_only_raw" in ablation
    assert "generated_text_only__ndcg_at_10" in paired
    assert "generated_text_only__delta_ndcg_at_10_vs_e5" in paired
    assert metadata["artifacts"]["representation_ablation_per_query"]["paired_query_count"] == 2


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
    assert set(paths) == {
        "result",
        "comparison",
        "metadata",
        "run_dir",
        "run_result",
        "run_comparison",
        "run_metadata",
        "per_query_metrics",
        "representation_ablation",
        "representation_ablation_latest",
        "representation_ablation_per_query",
    }
    assert Path(paths["run_result"]).is_file()
    assert Path(paths["run_comparison"]).is_file()
    assert Path(paths["run_metadata"]).is_file()
    assert paths["per_query_metrics"] == ""
    assert paths["representation_ablation"] == ""
    assert paths["representation_ablation_per_query"] == ""
    assert persisted["system_ids"] == SYSTEM_IDS
    assert metadata["result"]["status"] == "blocked"
    assert "system_id,ndcg_at_10,delta_vs_e5" in csv_text
