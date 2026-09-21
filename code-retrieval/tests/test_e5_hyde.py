import json

import pytest

from e5_baseline import CosQAData, RunData
from e5_hyde import (
    DEFAULT_GENERATOR_REVISION,
    DEFAULT_HYDE_PROMPT,
    HyDEConfig,
    build_expanded_queries,
    build_rrf_rankings,
    build_hyde_prompts,
    build_hyde_result,
    expected_hyde_cache_metadata,
    hyde_run_identity,
    hyde_cache_paths,
    load_valid_json_cache,
    save_json_cache,
    validate_hypotheses,
)


def test_config_records_explicit_hyde_controls():
    config = HyDEConfig()

    assert config.generator_id == "google/flan-t5-base"
    assert config.generator_revision == DEFAULT_GENERATOR_REVISION
    assert config.prompt_template == DEFAULT_HYDE_PROMPT
    assert config.num_hypotheses == 1
    assert config.candidate_depth == 1000
    assert config.temperature == 0.0
    assert config.max_new_tokens == 32
    assert config.stop_behavior == "eos_or_pad"
    assert config.combination_strategy == "original_plus_hypothesis"
    assert config.empty_hypothesis_behavior == "original_query_fallback"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prompt_template": "no query placeholder"},
        {"num_hypotheses": 2},
        {"temperature": -0.1},
        {"max_new_tokens": 0},
        {"stop_behavior": "custom"},
        {"combination_strategy": "hypothesis_only"},
        {"rrf_k": 0},
        {"rrf_hyde_weight": -0.1},
        {"empty_hypothesis_behavior": "ignore"},
    ],
)
def test_config_rejects_unsupported_or_invalid_controls(kwargs):
    with pytest.raises(ValueError):
        HyDEConfig(**kwargs)


def test_prompts_are_built_from_original_queries_only():
    prompts = build_hyde_prompts({"q1": "find a Python parser"})

    assert list(prompts) == ["q1"]
    assert "find a Python parser" in prompts["q1"]
    assert "qrels" not in prompts["q1"].lower()
    assert "answer key" not in prompts["q1"].lower()


def test_expanded_query_plumbing_preserves_order_and_original_text():
    queries = {"q1": "find a parser", "q2": "sort a list"}
    hypotheses = {"q1": "def parse(text): ...", "q2": "sorted(values)"}

    expanded = build_expanded_queries(queries, hypotheses)

    assert list(expanded) == ["q1", "q2"]
    assert expanded["q1"] == "find a parser\n\ndef parse(text): ..."
    assert expanded["q2"] == "sort a list\n\nsorted(values)"


def test_rrf_fusion_preserves_original_candidates_and_weights_hyde():
    fused = build_rrf_rankings(
        {
            "q1": {"d1": 0.9, "d2": 0.8},
        },
        {
            "q1": {"d2": 0.9, "d3": 0.8},
        },
        rrf_k=5,
        hyde_weight=0.5,
    )

    assert set(fused["q1"]) == {"d1", "d2", "d3"}
    assert fused["q1"]["d2"] > fused["q1"]["d1"] > fused["q1"]["d3"]


def test_hypotheses_must_match_query_ids_and_be_non_empty():
    with pytest.raises(ValueError, match="ordered query IDs"):
        validate_hypotheses({"q1": "one"}, {"q2": "generated"})
    with pytest.raises(ValueError, match="empty"):
        validate_hypotheses({"q1": "one"}, {"q1": "  "})


def test_empty_hypothesis_can_explicitly_fallback_to_original_query():
    assert validate_hypotheses(
        {"q1": "find a parser"},
        {"q1": "  "},
        empty_hypothesis_behavior="original_query_fallback",
    ) == {"q1": "find a parser"}


def test_hyde_cache_identity_changes_with_prompt_or_generator_revision(tmp_path):
    config = HyDEConfig(cache_dir=str(tmp_path))
    first = hyde_run_identity(config, repo_root=tmp_path, notebook_sha256="notebook-a")
    changed_prompt = HyDEConfig(
        cache_dir=str(tmp_path), prompt_template="Request: {query}\nHypothesis:"
    )
    changed_revision = HyDEConfig(
        cache_dir=str(tmp_path), generator_revision="different-revision"
    )
    changed_depth = HyDEConfig(cache_dir=str(tmp_path), candidate_depth=10)

    assert first != hyde_run_identity(
        changed_prompt, repo_root=tmp_path, notebook_sha256="notebook-a"
    )
    assert first != hyde_run_identity(
        changed_revision, repo_root=tmp_path, notebook_sha256="notebook-a"
    )
    assert first != hyde_run_identity(
        changed_depth, repo_root=tmp_path, notebook_sha256="notebook-a"
    )
    assert first != hyde_run_identity(
        config, repo_root=tmp_path, notebook_sha256="notebook-b"
    )


def test_hyde_json_cache_requires_exact_metadata(tmp_path):
    config = HyDEConfig(cache_dir=str(tmp_path))
    identity = hyde_run_identity(config, repo_root=tmp_path)
    paths = hyde_cache_paths(config, identity)
    metadata = expected_hyde_cache_metadata(
        config,
        identity=identity,
        kind="hypotheses",
        ids=["q1"],
        repo_root=tmp_path,
    )
    value = {"q1": "def parse(text): ..."}
    save_json_cache(paths["hypotheses"], paths["hypotheses_metadata"], value, metadata)

    assert load_valid_json_cache(
        paths["hypotheses"], paths["hypotheses_metadata"], metadata
    ) == value
    changed = dict(metadata)
    changed["config"] = dict(metadata["config"])
    changed["config"]["max_new_tokens"] = 64
    assert load_valid_json_cache(
        paths["hypotheses"], paths["hypotheses_metadata"], changed
    ) is None

    persisted = json.loads(paths["hypotheses_metadata"].read_text())
    assert persisted["kind"] == "hypotheses"
    assert persisted["config"]["generator_revision"] == DEFAULT_GENERATOR_REVISION


def test_result_contract_identifies_hyde_and_provenance(tmp_path):
    # This uses a deterministic plumbing fixture only; it is not benchmark evidence.
    config = HyDEConfig(cache_dir=str(tmp_path), artifact_dir=str(tmp_path / "artifacts"))
    data = CosQAData(
        corpus={"d1": {"text": "def parse(text): ..."}},
        queries={"q1": "find a parser"},
        qrels={"q1": {"d1": 1}},
        schema={"source": "fixture; plumbing test only"},
    )
    run_data = RunData(data.corpus, data.queries, data.qrels)

    result = build_hyde_result(
        config,
        data,
        run_data,
        {"ndcg_at_10": 0.5},
        identity="fixture-identity",
        environment={"fixture": True},
        timings={"generation": 0.0},
        hypothesis_count=1,
        repo_root=tmp_path,
        notebook_sha256="fixture-notebook",
    )

    assert result["system_id"] == "e5_hyde"
    assert result["system"]["query_expansion"] == "HyDE"
    assert result["tqe"]["generator"]["revision"] == DEFAULT_GENERATOR_REVISION
    assert result["tqe"]["combination_strategy"] == "original_plus_hypothesis"
    assert result["tqe"]["empty_hypothesis_behavior"] == "original_query_fallback"
    assert result["artifact_provenance"]["synthetic_hypotheses"] is False
    assert result["artifact_provenance"]["synthetic_scores"] is False
