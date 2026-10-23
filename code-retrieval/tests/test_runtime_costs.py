import numpy as np

from e5_hyde_rerank import ExperimentConfig, _embedding_or_cache
from runtime_costs import RuntimeCostRecorder


def test_embedding_cache_records_load_compute_and_hit_separately(tmp_path):
    data_path = tmp_path / "embeddings.npy"
    metadata_path = tmp_path / "embeddings.metadata.json"
    metadata = {"identity": "test-identity", "kind": "corpus", "count": 2}
    values = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    recorder = RuntimeCostRecorder()

    cold, compute_seconds = _embedding_or_cache(
        path=data_path,
        metadata_path=metadata_path,
        metadata=metadata,
        encode=lambda: values,
        cost_recorder=recorder,
        stage_name="corpus_embeddings",
        category="offline_corpus_encoding",
        item_count=2,
    )
    warm, warm_compute_seconds = _embedding_or_cache(
        path=data_path,
        metadata_path=metadata_path,
        metadata=metadata,
        encode=lambda: (_ for _ in ()).throw(AssertionError("cache hit invoked inference")),
        cost_recorder=recorder,
        stage_name="corpus_embeddings",
        category="offline_corpus_encoding",
        item_count=2,
    )

    assert np.array_equal(cold, values)
    assert np.array_equal(warm, values)
    assert compute_seconds >= 0
    assert warm_compute_seconds == 0
    assert [stage["cache_status"] for stage in recorder.cache_stages] == ["miss", "hit"]
    assert recorder.cache_stages[0]["compute_seconds"] == compute_seconds
    assert recorder.cache_stages[1]["compute_seconds"] == 0


def test_runtime_cost_summary_records_controls_latency_and_cpu_memory_scope():
    config = ExperimentConfig(batch_size=4, reranker_batch_size=2, candidate_depth=16)
    environment = {
        "device": "cpu",
        "processor": "test processor",
        "platform": "test platform",
        "cuda_available": False,
        "torch_num_threads": 3,
        "torch_default_dtype": "torch.float32",
        "torch_float32_matmul_precision": "highest",
    }
    with RuntimeCostRecorder(sample_interval_seconds=0.005) as recorder:
        with recorder.phase("model_load_e5", "startup"):
            pass
        recorder.observe_batch(
            stage="hyde_generation",
            seconds=0.25,
            unit_count=2,
            unit="queries",
        )
        recorder.record_cache_stage(
            name="original_rankings",
            category="query_search",
            cache_hit=False,
            load_seconds=0.01,
            compute_seconds=0.2,
            save_seconds=0.02,
            item_count=2,
        )

    summary = recorder.summary(
        run_id="run-id",
        cache_identity="cache-id",
        status="smoke",
        run_mode="smoke",
        environment=environment,
        config=config,
        query_count=2,
        corpus_count=16,
        reranked_pair_count=8,
        model_precision={"e5": "torch.float32"},
        offline_index_build_seconds=0.04,
    )

    assert summary["evidence_level"] == "smoke"
    assert summary["measurement_controls"]["batch_size"] == 4
    assert summary["measurement_controls"]["candidate_count"] == 8
    assert summary["offline_costs"]["faiss_index_build_seconds"] == 0.04
    assert summary["latency_summaries"]["hyde_generation"]["sample_count"] == 1
    assert summary["end_to_end_query_cost"]["online_stage_seconds"] >= 0.19
    assert "sampled process" in summary["memory"]["method"]
    assert summary["memory"]["gpu_memory_reason"] == "Measured model device was CPU"
