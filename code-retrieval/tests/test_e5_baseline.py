import json

import numpy as np

from e5_baseline import (
    BaselineConfig,
    CODE_VERSION,
    expected_cache_metadata,
    load_valid_embedding_cache,
    normalize_cosqa_records,
    run_identity,
    save_embedding_cache,
    select_run_data,
)


def test_baseline_uses_paper_faiss_candidate_contract():
    assert BaselineConfig().candidate_depth == 1000
    assert CODE_VERSION == "e5-baseline-v3-paper-faiss-ranking"


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


def test_config_rejects_unknown_run_mode():
    try:
        BaselineConfig(run_mode="full")
    except ValueError as error:
        assert "run_mode" in str(error)
    else:
        raise AssertionError("invalid run mode was accepted")
