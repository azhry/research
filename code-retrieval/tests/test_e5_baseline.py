import json

import numpy as np

from e5_baseline import (
    BaselineConfig,
    CODE_VERSION,
    expected_cache_metadata,
    evaluate_ndcg_at_10,
    load_valid_embedding_cache,
    load_valid_json_cache,
    normalize_cosqa_records,
    rank_with_faiss,
    run_identity,
    save_embedding_cache,
    save_json_cache,
    select_run_data,
)
from e5_hyde import HyDEConfig
from e5_hyde_rerank import ExperimentConfig, reciprocal_rank_fusion
from e5_rerank import RerankConfig


def test_baseline_uses_paper_faiss_candidate_contract():
    assert BaselineConfig().candidate_depth == 1000
    assert CODE_VERSION == "e5-baseline-v5-runtime-and-notebook-cache-identity"


def test_primary_experiments_share_retrieval_and_model_controls():
    baseline = BaselineConfig(run_mode="benchmark")
    hyde = HyDEConfig(run_mode="benchmark")
    rerank = RerankConfig(run_mode="benchmark")
    combined = ExperimentConfig(run_mode="benchmark")

    for config in (baseline, hyde, rerank, combined):
        assert config.candidate_depth == 1000
        assert config.batch_size == 128
        assert config.device == "cpu"
        assert config.dataset_revision == baseline.dataset_revision
        assert config.model_revision == baseline.model_revision

    assert hyde.generator_revision == combined.hyde_model_revision
    assert hyde.prompt_template == combined.hyde_prompt
    assert hyde.fallback_prompts == combined.hyde_fallback_prompts
    assert hyde.max_new_tokens == combined.hyde_max_new_tokens
    assert hyde.do_sample == combined.hyde_do_sample
    assert hyde.combination_strategy == combined.hyde_combination_strategy
    assert hyde.fusion_weights == combined.hyde_fusion_weights
    assert hyde.rrf_k == combined.rrf_k
    assert hyde.stop_behavior == combined.hyde_stop_behavior
    assert rerank.reranker_revision == combined.reranker_model_revision
    assert rerank.reranker_batch_size == combined.reranker_batch_size
    assert rerank.reranker_max_seq_length == combined.reranker_max_length
    assert rerank.fusion_weights == combined.reranker_fusion_weights
    assert rerank.rrf_k == combined.rrf_k
    assert hyde.fusion_weights == (0.65, 0.35)
    assert rerank.fusion_weights == (0.8, 0.2)
    assert combined.hyde_fusion_weights == (0.65, 0.35)
    assert combined.reranker_fusion_weights == (0.8, 0.2)
    assert combined.combined_fusion_weights == (0.75, 0.25)


def test_normalize_cosqa_filters_queries_to_qrels_and_preserves_text_only_contract():
    data = normalize_cosqa_records(
        [
            {"_id": "d1", "text": "def one(): pass", "title": "ignored"},
            {"_id": "d2", "text": "def two(): pass", "title": "ignored"},
        ],
        [
            {"_id": "q1", "text": "find one", "partition": "test"},
            {"_id": "q2", "text": "train only", "partition": "train"},
        ],
        [{"query-id": "q1", "corpus-id": "d1", "score": 1}],
    )

    assert list(data.queries) == ["q1"]
    assert data.corpus["d1"] == {"text": "def one(): pass"}
    assert data.qrels == {"q1": {"d1": 1}}


def test_smoke_selection_keeps_relevant_documents_in_the_reduced_corpus():
    data = normalize_cosqa_records(
        [{"_id": f"d{i}", "text": f"doc {i}"} for i in range(5)],
        [{"_id": f"q{i}", "text": f"query {i}"} for i in range(2)],
        [
            {"query-id": "q0", "corpus-id": "d4", "score": 1},
            {"query-id": "q1", "corpus-id": "d3", "score": 1},
        ],
    )
    config = BaselineConfig(run_mode="smoke", smoke_queries=1, smoke_corpus=1)

    selected = select_run_data(data, config)

    assert list(selected.queries) == ["q0"]
    assert "d4" in selected.corpus
    assert selected.qrels == {"q0": {"d4": 1}}
    assert selected.exclusions[0]["type"] == "smoke_subset"


def test_cache_rejects_changed_identity_and_accepts_matching_metadata(tmp_path):
    config = BaselineConfig(cache_dir=str(tmp_path))
    identity = run_identity(config)
    metadata = expected_cache_metadata(
        config,
        identity=identity,
        kind="corpus_embeddings",
        ids=["d1", "d2"],
    )
    data_path = tmp_path / "embeddings.npy"
    metadata_path = tmp_path / "embeddings.metadata.json"
    embeddings = np.ones((2, 3), dtype=np.float32)
    save_embedding_cache(data_path, metadata_path, embeddings, metadata)

    assert np.array_equal(
        load_valid_embedding_cache(data_path, metadata_path, metadata), embeddings
    )
    changed = dict(metadata)
    changed["ids_sha256"] = "different"
    assert load_valid_embedding_cache(data_path, metadata_path, changed) is None

    persisted = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert persisted["shape"] == [2, 3]


def test_cache_identity_changes_with_notebook_and_execution_environment():
    config = BaselineConfig()
    first = run_identity(
        config,
        notebook_sha256="notebook-one",
        environment={"device": "cpu", "torch": "2.14.0"},
    )
    changed_notebook = run_identity(
        config,
        notebook_sha256="notebook-two",
        environment={"device": "cpu", "torch": "2.14.0"},
    )
    changed_runtime = run_identity(
        config,
        notebook_sha256="notebook-one",
        environment={"device": "cpu", "torch": "2.15.0"},
    )

    assert first != changed_notebook
    assert first != changed_runtime


def test_rankings_cache_round_trip_preserves_retrieval_order(tmp_path):
    config = BaselineConfig(cache_dir=str(tmp_path))
    identity = run_identity(config)
    metadata = expected_cache_metadata(
        config,
        identity=identity,
        kind="rankings",
        ids=["q1", "__corpus__", "d1", "d2", "d3"],
    )
    rankings = {"q1": {"d9": 0.9, "d2": 0.8, "d10": 0.7}}
    data_path = tmp_path / "rankings.json"
    metadata_path = tmp_path / "rankings.metadata.json"

    save_json_cache(data_path, metadata_path, rankings, metadata)
    restored = load_valid_json_cache(data_path, metadata_path, metadata)

    assert restored is not None
    assert list(restored["q1"]) == ["d9", "d2", "d10"]


def test_official_metric_uses_saved_order_to_break_similarity_ties():
    qrels = {"q1": {"d-relevant": 1}}
    tied_ranking = {"q1": {"d-relevant": 0.5, "d-other": 0.5}}
    uniquely_scored_ranking = {"q1": {"d-relevant": 2.0, "d-other": 1.0}}

    tied_metric = evaluate_ndcg_at_10(qrels, tied_ranking)
    unique_metric = evaluate_ndcg_at_10(qrels, uniquely_scored_ranking)

    assert tied_metric["ndcg_at_10"] == unique_metric["ndcg_at_10"] == 1.0
    assert tied_metric["ranking_order_policy"] == (
        "descending_score_then_saved_insertion_order_with_unique_ordinal_evaluation_scores"
    )


def test_one_source_rrf_has_the_same_metric_as_the_source_ranking():
    qrels = {"q1": {"d-relevant": 1}}
    ranking = {
        "q1": {
            "d-other": 0.5,
            "d-relevant": 0.5,
            "d-last": 0.25,
        }
    }
    one_source_rrf = reciprocal_rank_fusion(
        [ranking, ranking], weights=(1.0, 0.0), top_k=3, rrf_k=60
    )

    assert evaluate_ndcg_at_10(qrels, one_source_rrf)["ndcg_at_10"] == (
        evaluate_ndcg_at_10(qrels, ranking)["ndcg_at_10"]
    )


def test_config_rejects_unknown_run_mode():
    try:
        BaselineConfig(run_mode="full")
    except ValueError as error:
        assert "run_mode" in str(error)
    else:
        raise AssertionError("invalid run mode was accepted")


def test_faiss_ranking_uses_configured_candidate_depth_and_preserves_corpus_contract():
    corpus_ids = ["d1", "d2", "d3"]
    rankings = rank_with_faiss(
        np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        np.asarray(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        ["q1", "q2"],
        corpus_ids,
        top_k=1000,
    )

    assert set(rankings) == {"q1", "q2"}
    assert all(set(row) == set(corpus_ids) for row in rankings.values())
    assert all(len(row) == len(corpus_ids) for row in rankings.values())
