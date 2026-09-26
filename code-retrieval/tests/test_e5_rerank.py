import numpy as np
import pytest

from e5_baseline import RunData, normalize_cosqa_records
from e5_rerank import (
    RerankConfig,
    build_candidate_pairs,
    build_comparison_result,
    candidate_pool_preserved,
    expected_rerank_cache_metadata,
    rerank_rankings,
    rerank_run_identity,
    score_candidate_pool,
)


def test_candidate_pairs_use_raw_query_and_corpus_text_in_ranking_order():
    pairs, keys = build_candidate_pairs(
        {"q1": "find one"},
        {"d2": {"text": "second"}, "d1": {"text": "first"}},
        {"q1": {"d2": 0.8, "d1": 0.7}},
    )

    assert pairs == [("find one", "second"), ("find one", "first")]
    assert keys == [("q1", "d2"), ("q1", "d1")]


def test_score_candidate_pool_groups_scores_without_expanding_candidates():
    rankings = {"q1": {"d2": 0.8, "d1": 0.7}}

    scores = score_candidate_pool(
        {"q1": "find one"},
        {"d1": {"text": "first"}, "d2": {"text": "second"}},
        rankings,
        lambda pairs: np.asarray([0.2, 0.9], dtype=np.float32),
    )

    assert scores["q1"]["d2"] == pytest.approx(0.2)
    assert scores["q1"]["d1"] == pytest.approx(0.9)


def test_rerank_orders_descending_scores_with_deterministic_id_ties():
    first_stage = {"q1": {"d3": 0.3, "d1": 0.8, "d2": 0.7}}
    reranker_scores = {"q1": {"d1": 0.5, "d2": 0.9, "d3": 0.9}}

    reranked = rerank_rankings(first_stage, reranker_scores)

    assert list(reranked["q1"]) == ["d3", "d2", "d1"]
    assert candidate_pool_preserved(first_stage, reranked)


def test_rerank_rejects_score_membership_and_nonfinite_values():
    with pytest.raises(ValueError, match="cover exactly"):
        rerank_rankings({"q1": {"d1": 1.0}}, {"q1": {"d2": 0.5}})

    with pytest.raises(ValueError, match="finite"):
        rerank_rankings({"q1": {"d1": 1.0}}, {"q1": {"d1": np.nan}})


def test_score_candidate_pool_rejects_wrong_score_count():
    with pytest.raises(ValueError, match="scores for 2 pairs"):
        score_candidate_pool(
            {"q1": "find one"},
            {"d1": {"text": "first"}, "d2": {"text": "second"}},
            {"q1": {"d1": 0.8, "d2": 0.7}},
            lambda pairs: [0.2],
        )


def test_rerank_cache_identity_includes_reranker_controls(tmp_path):
    config = RerankConfig(cache_dir=str(tmp_path))
    identity = rerank_run_identity(config)
    metadata = expected_rerank_cache_metadata(
        config,
        identity=identity,
        kind="reranked_rankings",
        ids=["q1", "d1", "d2"],
    )

    changed = RerankConfig(cache_dir=str(tmp_path), reranker_batch_size=64)

    assert metadata["code_version"] == "e5-rerank-v6-validation-selected-e5-fusion-stable-ties"
    assert rerank_run_identity(changed) != identity
    assert metadata["config"]["reranker_revision"] == config.reranker_revision


def test_comparison_result_contains_both_measured_systems_and_delta():
    data = normalize_cosqa_records(
        [{"_id": "d1", "text": "first"}],
        [{"_id": "q1", "text": "find first"}],
        [{"query-id": "q1", "corpus-id": "d1", "score": 1}],
    )
    config = RerankConfig(run_mode="smoke")
    metric = {"ndcg_at_10": 0.5, "NDCG@10": {"NDCG@10": 0.5}}
    reranked_metric = {"ndcg_at_10": 0.75, "NDCG@10": {"NDCG@10": 0.75}}

    result = build_comparison_result(
        config,
        data,
        RunData(data.corpus, data.queries, data.qrels),
        metric,
        reranked_metric,
        raw_reranked_metric={"ndcg_at_10": 0.4},
        identity="identity",
        environment={"device": "cpu"},
        timings={"reranking": 1.0},
        candidate_pool_was_preserved=True,
    )

    assert result["status"] == "smoke"
    assert result["benchmark_evidence"] is False
    assert result["systems"]["e5"]["metric"] == "nDCG@10"
    assert result["systems"]["e5_rerank"]["candidate_pool_preserved"] is True
    assert result["delta_ndcg_at_10"] == pytest.approx(0.25)
    assert result["fusion"]["selection_qrels_split"] == "valid"
    assert result["component_diagnostics"]["e5_rerank_raw"]["ndcg_at_10"] == 0.4
