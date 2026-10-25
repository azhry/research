"""Post-hoc paired analysis for the saved CosQA test runs.

This module reads completed run artifacts only. It does not load models, retrieve
documents, or alter any historical run or paper output.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


DATASET_ID = "CoIR-Retrieval/cosqa"
DATASET_REVISION = "0846fa3b963a21bead36e9fab61451fc83b777a6"
QUERY_COUNT = 500
CORPUS_COUNT = 20_604
CANDIDATE_DEPTH = 1_000
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 518
TIE_TOLERANCE = 1e-12
AGGREGATE_TOLERANCE = 1e-5

METRICS = {
    "ndcg_at_10": "nDCG@10",
    "map_at_10": "MAP@10",
    "recall_at_10": "Recall@10",
    "recall_at_1000": "Recall@1000",
}

SYSTEM_LABELS = {
    "e5": "E5",
    "e5_hyde_raw": "Query + generated hypothesis (raw)",
    "e5_hypothesis_only_raw": "Generated hypothesis only (raw)",
    "e5_hyde": "E5 + query/hypothesis fusion",
    "e5_hypothesis_only": "E5 + hypothesis-only fusion",
    "e5_rerank_raw": "MS MARCO reranker (raw)",
    "e5_rerank": "E5 + MS MARCO reranker fusion",
    "e5_hyde_rerank": "Combined HyDE + MS MARCO reranker",
    "e5_code_rerank_raw": "Code-trained reranker (raw)",
    "e5_code_rerank": "E5 + code-trained reranker fusion",
}

REPRESENTATION_SYSTEMS = (
    "e5",
    "e5_hyde_raw",
    "e5_hypothesis_only_raw",
    "e5_hyde",
    "e5_hypothesis_only",
    "e5_rerank_raw",
    "e5_rerank",
    "e5_hyde_rerank",
)
CODE_RERANKER_SYSTEMS = REPRESENTATION_SYSTEMS + (
    "e5_code_rerank_raw",
    "e5_code_rerank",
)

CONTRASTS = (
    ("e5_hyde_raw", "e5", "Raw query-plus-hypothesis representation vs E5"),
    ("e5_hypothesis_only_raw", "e5", "Raw hypothesis-only representation vs E5"),
    ("e5_hyde", "e5", "HyDE fusion vs E5"),
    ("e5_hypothesis_only", "e5", "Hypothesis-only fusion vs E5"),
    ("e5_rerank_raw", "e5", "Raw MS MARCO reranker vs E5"),
    ("e5_rerank", "e5", "MS MARCO reranker fusion vs E5"),
    ("e5_hyde_rerank", "e5", "Combined system vs E5"),
    ("e5_code_rerank_raw", "e5", "Raw code-trained reranker vs E5"),
    ("e5_code_rerank", "e5", "Code-trained reranker fusion vs E5"),
    ("e5_hyde_rerank", "e5_hyde", "Combined system vs HyDE component"),
    ("e5_hyde_rerank", "e5_rerank", "Combined system vs MS MARCO component"),
)

CSV_COLUMNS = (
    "query_id",
    "system_id",
    "ndcg_at_10",
    "map_at_10",
    "recall_at_10",
    "recall_at_1000",
    "first_relevant_rank",
    "relevant_document_count",
    "relevant_retrieved_at_10",
    "relevant_retrieved_at_1000",
    "cache_identity",
)


class AnalysisInputError(ValueError):
    """Raised when a saved run does not satisfy this analysis contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AnalysisInputError(f"Required artifact is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisInputError(f"Could not read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AnalysisInputError(f"Expected a JSON object in {path}")
    return value


def _int_value(value: Any, field: str, context: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise AnalysisInputError(f"Invalid {field} for {context}: {value!r}") from exc


def _float_value(value: Any, field: str, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AnalysisInputError(f"Invalid {field} for {context}: {value!r}") from exc
    if not math.isfinite(result):
        raise AnalysisInputError(f"Non-finite {field} for {context}: {value!r}")
    return result


def _validate_run_identity(result: dict[str, Any], run_dir: Path) -> None:
    label = result.get("run_id", run_dir.name)
    if result.get("status") != "benchmark" or result.get("run_mode") != "benchmark":
        raise AnalysisInputError(f"{label} is not a completed benchmark run")
    if result.get("benchmark_evidence") is not True:
        raise AnalysisInputError(f"{label} is not marked as benchmark evidence")
    dataset = result.get("dataset") or {}
    if dataset.get("id") != DATASET_ID or dataset.get("revision") != DATASET_REVISION:
        raise AnalysisInputError(f"{label} does not use the pinned CosQA dataset revision")
    if (dataset.get("splits") or {}).get("qrels") != "test":
        raise AnalysisInputError(f"{label} does not use the held-out test qrels")
    expected_counts = {
        "query_count": QUERY_COUNT,
        "qrels_query_count": QUERY_COUNT,
        "qrels_judgment_count": QUERY_COUNT,
        "corpus_count": CORPUS_COUNT,
        "candidate_depth": CANDIDATE_DEPTH,
    }
    for field, expected in expected_counts.items():
        if _int_value(result.get(field), field, label) != expected:
            raise AnalysisInputError(
                f"{label} has {field}={result.get(field)!r}; expected {expected}"
            )
    if result.get("synthetic_scores") is True:
        raise AnalysisInputError(f"{label} is marked as containing synthetic scores")


def _load_run(run_dir: Path, expected_systems: tuple[str, ...]) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    result_path = run_dir / "result.json"
    metadata_path = run_dir / "metadata.json"
    metrics_path = run_dir / "per_query_metrics.csv"
    runtime_path = run_dir / "runtime_costs.json"
    result = _read_json(result_path)
    metadata = _read_json(metadata_path)
    runtime = _read_json(runtime_path)
    _validate_run_identity(result, run_dir)
    run_id = result["run_id"]
    if runtime.get("run_id") != run_id or runtime.get("status") != "benchmark":
        raise AnalysisInputError(f"Runtime-cost record does not match {run_id}")
    if metadata.get("result", {}).get("run_id") != run_id:
        raise AnalysisInputError(f"Metadata record does not match {run_id}")
    if runtime.get("evidence_level") != "benchmark":
        raise AnalysisInputError(f"Runtime costs for {run_id} are not benchmark evidence")

    systems: dict[str, dict[str, dict[str, Any]]] = {
        system_id: {} for system_id in expected_systems
    }
    row_count = 0
    if not metrics_path.is_file():
        raise AnalysisInputError(f"Required paired export is missing: {metrics_path}")
    with metrics_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing_columns = sorted(set(CSV_COLUMNS) - set(reader.fieldnames or ()))
        if missing_columns:
            raise AnalysisInputError(
                f"{metrics_path} is missing columns: {', '.join(missing_columns)}"
            )
        for row in reader:
            row_count += 1
            query_id = (row.get("query_id") or "").strip()
            system_id = (row.get("system_id") or "").strip()
            if not query_id or system_id not in systems:
                raise AnalysisInputError(
                    f"Unexpected query or system in {metrics_path}: "
                    f"{query_id!r}, {system_id!r}"
                )
            if query_id in systems[system_id]:
                raise AnalysisInputError(
                    f"Duplicate metric row for {query_id}, {system_id} in {metrics_path}"
                )
            if row.get("cache_identity") != result.get("cache_identity"):
                raise AnalysisInputError(
                    f"Cache identity mismatch in {metrics_path} for {system_id}"
                )
            values = {
                metric: _float_value(row[metric], metric, f"{system_id}/{query_id}")
                for metric in METRICS
            }
            if any(value < 0 or value > 1 for value in values.values()):
                raise AnalysisInputError(
                    f"Metric outside [0, 1] for {system_id}/{query_id}"
                )
            rank_text = (row.get("first_relevant_rank") or "").strip()
            first_relevant_rank = int(rank_text) if rank_text else None
            relevant_count = _int_value(
                row.get("relevant_document_count"),
                "relevant_document_count",
                f"{system_id}/{query_id}",
            )
            hit10 = _int_value(
                row.get("relevant_retrieved_at_10"),
                "relevant_retrieved_at_10",
                f"{system_id}/{query_id}",
            )
            hit1000 = _int_value(
                row.get("relevant_retrieved_at_1000"),
                "relevant_retrieved_at_1000",
                f"{system_id}/{query_id}",
            )
            if relevant_count != 1 or hit10 not in (0, 1) or hit1000 not in (0, 1):
                raise AnalysisInputError(
                    f"Unexpected qrels shape for {system_id}/{query_id}"
                )
            if first_relevant_rank is None:
                if hit10 or hit1000:
                    raise AnalysisInputError(
                        f"Missing rank conflicts with a retrieved judgment for {system_id}/{query_id}"
                    )
            else:
                if first_relevant_rank < 1:
                    raise AnalysisInputError(
                        f"Invalid first relevant rank for {system_id}/{query_id}"
                    )
                if hit10 != int(first_relevant_rank <= 10):
                    raise AnalysisInputError(
                        f"Top-10 hit conflicts with first relevant rank for {system_id}/{query_id}"
                    )
                if hit1000 != int(first_relevant_rank <= CANDIDATE_DEPTH):
                    raise AnalysisInputError(
                        f"Depth-1000 hit conflicts with first relevant rank for {system_id}/{query_id}"
                    )
            if abs(values["recall_at_10"] - hit10) > TIE_TOLERANCE:
                raise AnalysisInputError(
                    f"Recall@10 conflicts with the judged hit for {system_id}/{query_id}"
                )
            if abs(values["recall_at_1000"] - hit1000) > TIE_TOLERANCE:
                raise AnalysisInputError(
                    f"Recall@1000 conflicts with the judged hit for {system_id}/{query_id}"
                )
            systems[system_id][query_id] = {
                **values,
                "first_relevant_rank": first_relevant_rank,
                "relevant_document_count": relevant_count,
                "relevant_retrieved_at_10": hit10,
                "relevant_retrieved_at_1000": hit1000,
            }

    expected_rows = QUERY_COUNT * len(expected_systems)
    if row_count != expected_rows:
        raise AnalysisInputError(
            f"{metrics_path} has {row_count} rows; expected {expected_rows}"
        )
    query_ids = set(next(iter(systems.values())))
    if len(query_ids) != QUERY_COUNT:
        raise AnalysisInputError(
            f"{metrics_path} contains {len(query_ids)} queries; expected {QUERY_COUNT}"
        )
    for system_id, query_rows in systems.items():
        if set(query_rows) != query_ids:
            raise AnalysisInputError(
                f"{metrics_path} is not fully paired for {system_id}"
            )
    if runtime.get("measurement_controls", {}).get("query_count") != QUERY_COUNT:
        raise AnalysisInputError(f"Runtime query count does not match {run_id}")

    return {
        "run_dir": run_dir,
        "result": result,
        "runtime": runtime,
        "systems": systems,
        "query_ids": sorted(query_ids),
        "input_files": {
            "result.json": {"path": result_path.as_posix(), "sha256": sha256_file(result_path)},
            "metadata.json": {
                "path": metadata_path.as_posix(),
                "sha256": sha256_file(metadata_path),
            },
            "per_query_metrics.csv": {
                "path": metrics_path.as_posix(),
                "sha256": sha256_file(metrics_path),
            },
            "runtime_costs.json": {
                "path": runtime_path.as_posix(),
                "sha256": sha256_file(runtime_path),
            },
        },
    }


def _reported_means(result: dict[str, Any]) -> dict[str, dict[str, float]]:
    reported: dict[str, dict[str, float]] = {}
    for section_name in ("results", "representation_ablations"):
        entries = result.get(section_name) or []
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict) or not entry.get("system_id"):
                continue
            system_id = entry["system_id"]
            metrics = {
                metric: _float_value(entry.get(metric), metric, f"result/{system_id}")
                for metric in METRICS
                if entry.get(metric) is not None
            }
            if system_id in reported:
                for metric, value in metrics.items():
                    prior = reported[system_id].get(metric)
                    if prior is not None and abs(prior - value) > AGGREGATE_TOLERANCE:
                        raise AnalysisInputError(
                            f"Result sections disagree for {system_id}/{metric}"
                        )
                    reported[system_id][metric] = value
            else:
                reported[system_id] = metrics
    for system_id, entry in (result.get("component_diagnostics") or {}).items():
        metrics = {
            metric: _float_value(entry.get(metric), metric, f"component_diagnostics/{system_id}")
            for metric in METRICS
            if entry.get(metric) is not None
        }
        if system_id in reported:
            for metric, value in metrics.items():
                prior = reported[system_id].get(metric)
                if prior is not None and abs(prior - value) > AGGREGATE_TOLERANCE:
                    raise AnalysisInputError(
                        f"Result sections disagree for {system_id}/{metric}"
                    )
                reported[system_id][metric] = value
        else:
            reported[system_id] = metrics
    code_reranker = result.get("code_reranker") or {}
    raw_metrics = code_reranker.get("metrics", {}).get("raw") or {}
    raw_values = {
        metric: raw_metrics.get(metric)
        for metric in METRICS
        if raw_metrics.get(metric) is not None
    }
    if raw_values:
        reported.setdefault("e5_code_rerank_raw", {}).update(
            {metric: _float_value(value, metric, "result/e5_code_rerank_raw") for metric, value in raw_values.items()}
        )
    return reported


def _reconcile_aggregates(run: dict[str, Any]) -> dict[str, Any]:
    reported = _reported_means(run["result"])
    observed: dict[str, dict[str, float]] = {}
    for system_id, query_rows in run["systems"].items():
        observed[system_id] = {
            metric: float(np.mean([row[metric] for row in query_rows.values()]))
            for metric in METRICS
        }
    checks: list[dict[str, Any]] = []
    for system_id, metric_values in observed.items():
        if system_id not in reported:
            raise AnalysisInputError(
                f"No aggregate result is available to reconcile for {system_id}"
            )
        for metric, value in metric_values.items():
            if metric not in reported[system_id]:
                raise AnalysisInputError(
                    f"No aggregate {metric} is available to reconcile for {system_id}"
                )
            matches = round(value, 5) == round(reported[system_id][metric], 5)
            checks.append(
                {
                    "system_id": system_id,
                    "metric": metric,
                    "per_query_mean": value,
                    "reported_mean": reported[system_id][metric],
                    "rounding_precision": 5,
                    "passed": matches,
                }
            )
            if not matches:
                raise AnalysisInputError(
                    f"Per-query {metric} mean for {system_id} does not match "
                    f"the run aggregate ({value:.8f} vs {reported[system_id][metric]:.8f})"
                )
    return {"passed": True, "checks": checks}


def _cross_run_parity(
    representation_run: dict[str, Any], code_run: dict[str, Any]
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for system_id in REPRESENTATION_SYSTEMS:
        for query_id in representation_run["query_ids"]:
            left = representation_run["systems"][system_id][query_id]
            right = code_run["systems"][system_id][query_id]
            for metric in METRICS:
                difference = abs(left[metric] - right[metric])
                checks.append(
                    {
                        "system_id": system_id,
                        "query_id": query_id,
                        "metric": metric,
                        "absolute_difference": difference,
                    }
                )
                if difference > TIE_TOLERANCE:
                    raise AnalysisInputError(
                        f"Shared run outputs differ for {system_id}/{query_id}/{metric}"
                    )
    return {
        "passed": True,
        "shared_system_count": len(REPRESENTATION_SYSTEMS),
        "shared_query_count": QUERY_COUNT,
        "shared_metric_count": len(METRICS),
        "maximum_absolute_difference": max(
            (check["absolute_difference"] for check in checks), default=0.0
        ),
    }


def _classify_interval(low: float, high: float) -> str:
    if low > 0:
        return "improvement"
    if high < 0:
        return "regression"
    if abs(low) <= TIE_TOLERANCE and abs(high) <= TIE_TOLERANCE:
        return "no_observed_difference"
    return "uncertain"


def _paired_effect(
    candidate: np.ndarray, reference: np.ndarray, sample_indices: np.ndarray
) -> dict[str, Any]:
    deltas = candidate - reference
    bootstrap_means = deltas[sample_indices].mean(axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975], method="linear")
    mean_delta = float(deltas.mean())
    low_value, high_value = float(low), float(high)
    return {
        "mean_delta": mean_delta,
        "ci95": [low_value, high_value],
        "query_wins": int((deltas > TIE_TOLERANCE).sum()),
        "query_ties": int((np.abs(deltas) <= TIE_TOLERANCE).sum()),
        "query_losses": int((deltas < -TIE_TOLERANCE).sum()),
        "evidence_reading": _classify_interval(low_value, high_value),
    }


def _resolve_generation_path(
    run: dict[str, Any], representation_run: dict[str, Any]
) -> tuple[Path, dict[str, Any]]:
    provenance = run["result"].get("hyde_generation_provenance") or {}
    declared_path = provenance.get("path")
    candidates = [Path(declared_path).expanduser()] if declared_path else []
    for source_run in (representation_run, run):
        cache_identity = source_run["result"].get("cache_identity")
        if cache_identity:
            candidates.append(
                source_run["run_dir"].parent.parent / "cache" / cache_identity / "hyde_generations.json"
            )
    for candidate in candidates:
        if candidate.is_file():
            resolved = candidate.resolve()
            expected_sha = provenance.get("sha256")
            observed_sha = sha256_file(resolved)
            if expected_sha and observed_sha != expected_sha:
                raise AnalysisInputError(
                    f"HyDE generation artifact hash does not match result provenance: {resolved}"
                )
            return resolved, {
                "path": resolved.as_posix(),
                "sha256": observed_sha,
                "declared_sha256": expected_sha,
                "query_count": provenance.get("query_count"),
            }
    raise AnalysisInputError(
        "Could not locate the provenance-linked 500-query HyDE generation artifact"
    )


def _generation_summary(
    run: dict[str, Any], representation_run: dict[str, Any]
) -> dict[str, Any]:
    generations_path, source = _resolve_generation_path(run, representation_run)
    generations = _read_json(generations_path)
    query_ids = set(run["query_ids"])
    if set(generations) != query_ids:
        raise AnalysisInputError(
            "HyDE generation IDs do not match the complete paired test query IDs"
        )
    texts: dict[str, str] = {}
    output_counts: Counter[int] = Counter()
    for query_id, value in generations.items():
        if isinstance(value, list):
            output_counts[len(value)] += 1
            text = value[0].strip() if value and isinstance(value[0], str) else ""
        elif isinstance(value, str):
            output_counts[1] += 1
            text = value.strip()
        else:
            output_counts[0] += 1
            text = ""
        texts[query_id] = text
    token_counts = np.asarray([len(text.split()) for text in texts.values()], dtype=np.int64)
    exact_counts = Counter(texts.values())
    return {
        "scope": "surface-form diagnostics only; no semantic or human quality labels",
        "query_count": len(texts),
        "empty_outputs": sum(not text for text in texts.values()),
        "unique_exact_strings": len(exact_counts),
        "repeated_output_occurrences_after_first": len(texts) - len(exact_counts),
        "queries_with_at_most_two_whitespace_tokens": int((token_counts <= 2).sum()),
        "whitespace_token_count": {
            "minimum": int(token_counts.min()),
            "median": float(np.median(token_counts)),
            "mean": float(token_counts.mean()),
            "p90": float(np.quantile(token_counts, 0.9, method="linear")),
            "maximum": int(token_counts.max()),
        },
        "outputs_per_query": {str(key): value for key, value in sorted(output_counts.items())},
        "source_artifact": source,
        "interpretation_limit": (
            "Length, emptiness, and exact duplication do not measure semantic correctness, "
            "code validity, or usefulness. The study retains Flan-T5-base; its quality "
            "concern remains unresolved."
        ),
    }


def _retrieval_failures(run: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for system_id in CODE_RERANKER_SYSTEMS:
        query_rows = run["systems"][system_id]
        ranks = [
            row["first_relevant_rank"]
            for row in query_rows.values()
            if row["first_relevant_rank"] is not None
        ]
        missing_at_10 = sum(
            row["relevant_retrieved_at_10"] == 0 for row in query_rows.values()
        )
        missing_at_1000 = sum(
            row["relevant_retrieved_at_1000"] == 0 for row in query_rows.values()
        )
        rows.append(
            {
                "system_id": system_id,
                "system": SYSTEM_LABELS[system_id],
                "query_count": QUERY_COUNT,
                "judged_relevant_in_top_10": QUERY_COUNT - missing_at_10,
                "judged_relevant_missing_from_top_10": missing_at_10,
                "judged_relevant_missing_from_depth_1000": missing_at_1000,
                "judged_relevant_at_ranks_11_to_1000": missing_at_10 - missing_at_1000,
                "median_first_relevant_rank_when_retrieved": (
                    float(np.median(ranks)) if ranks else None
                ),
            }
        )
    return rows


def _parse_weight(value: Any) -> list[float] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if isinstance(value, (list, tuple)) and all(
        isinstance(item, (int, float)) for item in value
    ):
        return [float(item) for item in value]
    return None


def _runtime_settings_audit(run: dict[str, Any]) -> list[dict[str, Any]]:
    result = run["result"]
    settings = (
        run["runtime"].get("measurement_controls", {}).get("experiment_settings", {})
    )
    expected: dict[str, Any] = {}
    for entry in result.get("representation_ablations") or []:
        if entry.get("system_id") == "e5_hypothesis_only":
            expected["hypothesis_only_fusion_weights"] = _parse_weight(
                entry.get("fusion_weights")
            )
    code_weights = (result.get("code_reranker") or {}).get("fusion_weights")
    if code_weights is not None:
        expected["code_reranker_fusion_weights"] = _parse_weight(code_weights)
    discrepancies = []
    for key, actual in expected.items():
        recorded = _parse_weight(settings.get(key))
        if actual is not None and recorded is not None and actual != recorded:
            discrepancies.append(
                {
                    "setting": key,
                    "used_for_result": actual,
                    "recorded_in_runtime_sidecar": recorded,
                    "impact": (
                        "Fusion is applied after pair scoring, so this configuration snapshot "
                        "discrepancy does not change measured pair-scoring time or memory."
                    ),
                }
            )
    return discrepancies


def _cost_summary(run: dict[str, Any], label: str) -> dict[str, Any]:
    runtime = run["runtime"]
    phases = runtime.get("phase_timings") or []
    caches = runtime.get("cache_stages") or []
    scoring_name = (
        "code_reranker_scoring"
        if run["result"].get("code_reranker")
        else "original_reranked"
    )
    scoring_seconds = None
    for cache_stage in caches:
        if cache_stage.get("name") == scoring_name:
            scoring_seconds = cache_stage.get("compute_seconds")
            break
    if scoring_seconds is None:
        for phase in phases:
            if phase.get("name") == scoring_name:
                scoring_seconds = phase.get("seconds")
                break
    if scoring_seconds is None and scoring_name == "original_reranked":
        for cache_stage in caches:
            if cache_stage.get("name") == "original_reranked":
                scoring_seconds = cache_stage.get("compute_seconds")
                break
    controls = runtime.get("measurement_controls") or {}
    memory = runtime.get("memory") or {}
    query_cost = runtime.get("end_to_end_query_cost") or {}
    offline = runtime.get("offline_costs") or {}
    return {
        "label": label,
        "run_id": runtime.get("run_id"),
        "cache_identity": runtime.get("cache_identity"),
        "source_commit": controls.get("git_commit"),
        "device": controls.get("device"),
        "query_count": controls.get("query_count"),
        "corpus_count": controls.get("corpus_count"),
        "candidate_depth": controls.get("candidate_depth"),
        "candidate_pair_count": controls.get("candidate_count"),
        "pipeline_wall_seconds": runtime.get("pipeline_wall_seconds"),
        "online_stage_seconds": query_cost.get("online_stage_seconds"),
        "online_seconds_per_query": query_cost.get("online_seconds_per_query"),
        "full_pipeline_seconds_per_query": query_cost.get(
            "full_pipeline_seconds_per_query"
        ),
        "pair_scoring_stage": scoring_name,
        "pair_scoring_seconds": scoring_seconds,
        "phase_timings": phases,
        "offline_costs": offline,
        "cache_stages": caches,
        "cache_policy": (runtime.get("measurement_protocol") or {}).get(
            "cache_policy"
        ),
        "memory": {
            "method": memory.get("method"),
            "process_peak_rss_bytes": memory.get("process_peak_rss_bytes"),
            "peak_delta_from_run_start_bytes": memory.get(
                "peak_delta_from_run_start_bytes"
            ),
            "sample_interval_ms": (runtime.get("measurement_protocol") or {}).get(
                "memory_sample_interval_ms"
            ),
            "sample_count": memory.get("sample_count"),
            "limitation": memory.get("limitation"),
        },
        "runtime_sidecar_setting_discrepancies": _runtime_settings_audit(run),
    }


def _format_effect(effect: dict[str, Any]) -> str:
    low, high = effect["ci95"]
    return f'{effect["mean_delta"]:+.5f} [{low:+.5f}, {high:+.5f}]'


def _markdown_report(analysis: dict[str, Any]) -> str:
    systems = analysis["systems"]
    contrasts = analysis["contrasts"]
    lines = [
        "# AZH-533 paired effects and failure analysis",
        "",
        f'Analysis run: {analysis["analysis_run_id"]}',
        "",
        "## Evidence scope",
        "",
        (
            "This is a retrospective, exploratory analysis of the preserved complete "
            "CosQA test runs. The 500 test queries and validation-selected settings "
            "have already been examined, so this is not independent confirmatory evidence."
        ),
        "",
        (
            f'Dataset: {DATASET_ID} at revision {DATASET_REVISION}; '
            f'{QUERY_COUNT} test queries; {CORPUS_COUNT:,} corpus documents; '
            f'depth-{CANDIDATE_DEPTH:,} candidate rankings. The evaluator uses the '
            "saved insertion order for ties and converts it to unique ordinal scores "
            "before the pinned COIR evaluator. This is the local declared tie policy."
        ),
        "",
        (
            f'Paired uncertainty uses {analysis["bootstrap"]["replicates"]:,} query '
            f'resamples with seed {analysis["bootstrap"]["seed"]} and percentile 95% '
            "intervals. Intervals are descriptive and unadjusted across metrics and "
            "contrasts; they are not simultaneous bounds. All contrasts remain "
            "exploratory under the approved scope."
        ),
        "",
        "## Recomputed system metrics",
        "",
        "| System | nDCG@10 | MAP@10 | Recall@10 | Recall@1000 |",
        "|---|---:|---:|---:|---:|",
    ]
    for system_id in CODE_RERANKER_SYSTEMS:
        values = systems[system_id]["mean_metrics"]
        lines.append(
            f'| {SYSTEM_LABELS[system_id]} | {values["ndcg_at_10"]:.5f} | '
            f'{values["map_at_10"]:.5f} | {values["recall_at_10"]:.5f} | '
            f'{values["recall_at_1000"]:.5f} |'
        )
    lines.extend(
        [
            "",
            (
                "Every per-query mean was reconciled to the corresponding result or "
                "representation-ablation aggregate after five-decimal rounding. The "
                "AZH-529 and AZH-531 exports also match for all eight shared systems, "
                "all 500 query IDs, and all four metrics."
            ),
            "",
            "## Primary outcome: paired nDCG@10 effects",
            "",
            "| Contrast (candidate minus reference) | Delta (95% paired CI) | Query wins / ties / losses | Reading |",
            "|---|---:|---:|---|",
        ]
    )
    for candidate, reference, label in CONTRASTS:
        effect = contrasts[f"{candidate}_minus_{reference}"]["metrics"][
            "ndcg_at_10"
        ]
        lines.append(
            f'| {label} | {_format_effect(effect)} | '
            f'{effect["query_wins"]} / {effect["query_ties"]} / '
            f'{effect["query_losses"]} | {effect["evidence_reading"]} |'
        )
    lines.extend(
        [
            "",
            (
                "A positive interval is descriptive evidence of an improvement in this "
                "test sample; an interval spanning zero leaves the effect uncertain; "
                "a negative interval is descriptive evidence of a regression. These "
                "readings do not correct for multiple comparisons."
            ),
            "",
            "## Secondary paired effects",
            "",
        ]
    )
    for metric, display_name in METRICS.items():
        if metric == "ndcg_at_10":
            continue
        lines.extend(
            [
                f"### {display_name}",
                "",
                "| Contrast (candidate minus reference) | Delta (95% paired CI) | Query wins / ties / losses |",
                "|---|---:|---:|",
            ]
        )
        for candidate, reference, label in CONTRASTS:
            effect = contrasts[f"{candidate}_minus_{reference}"]["metrics"][metric]
            lines.append(
                f'| {label} | {_format_effect(effect)} | '
                f'{effect["query_wins"]} / {effect["query_ties"]} / '
                f'{effect["query_losses"]} |'
            )
        lines.append("")
    lines.extend(
        [
            "## Judged retrieval failure patterns",
            "",
            (
                "Each CosQA test query has one judged relevant code item. The counts "
                "below track only that judged item; retrieved documents without a "
                "positive qrel are unjudged, not known irrelevant."
            ),
            "",
            "| System | Judged item in top 10 | Missing from top 10 | Missing from depth 1,000 | Found at ranks 11–1,000 | Median judged rank when found |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in analysis["retrieval_failures"]:
        median_rank = (
            f'{row["median_first_relevant_rank_when_retrieved"]:.1f}'
            if row["median_first_relevant_rank_when_retrieved"] is not None
            else "—"
        )
        lines.append(
            f'| {row["system"]} | {row["judged_relevant_in_top_10"]}/500 | '
            f'{row["judged_relevant_missing_from_top_10"]} | '
            f'{row["judged_relevant_missing_from_depth_1000"]} | '
            f'{row["judged_relevant_at_ranks_11_to_1000"]} | {median_rank} |'
        )
    lines.extend(
        [
            "",
            (
                "These are complete-export counts under the one-positive-per-query "
                "qrels. They do not label individual misses, assess the semantic "
                "relevance of unjudged candidates, or generalize beyond this test set."
            ),
            "",
            "## Generator output diagnostics",
            "",
        ]
    )
    generation = analysis["generation_outputs"]
    token_counts = generation["whitespace_token_count"]
    lines.extend(
        [
            (
                f'Across {generation["query_count"]} saved hypotheses: '
                f'{generation["empty_outputs"]} empty outputs, '
                f'{generation["unique_exact_strings"]} unique exact strings, '
                f'{generation["repeated_output_occurrences_after_first"]} repeated '
                "occurrences after the first, and "
                f'{generation["queries_with_at_most_two_whitespace_tokens"]} outputs '
                "with at most two whitespace tokens. Token counts have median "
                f'{token_counts["median"]:.1f}, mean {token_counts["mean"]:.2f}, and '
                f'p90 {token_counts["p90"]:.1f}.'
            ),
            "",
            generation["interpretation_limit"],
            "",
            "## Runtime, memory, and cost",
            "",
            (
                "These are single recorded CPU executions. AZH-529 records the complete "
                "representation/fusion pipeline with uncached stage computation. "
                "AZH-531 records the added code-reranker scoring run against the saved "
                "E5 candidate pool; it did not re-encode the corpus."
            ),
            "",
            "| Run / measured path | Pair-scoring stage | Scoring seconds | Online seconds/query | Full-pipeline seconds/query | Pipeline wall seconds | Peak process RSS (GiB) |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for cost in analysis["runtime_costs"]:
        memory_bytes = cost["memory"]["process_peak_rss_bytes"]
        memory_gib = memory_bytes / (1024**3) if memory_bytes is not None else None
        lines.append(
            f'| {cost["label"]} | {cost["pair_scoring_stage"]} | '
            f'{cost["pair_scoring_seconds"]:.2f} | '
            f'{cost["online_seconds_per_query"]:.3f} | '
            f'{cost["full_pipeline_seconds_per_query"]:.3f} | '
            f'{cost["pipeline_wall_seconds"]:.2f} | '
            f'{memory_gib:.2f} |'
        )
    lines.extend(
        [
            "",
            "### Cache-stage measurements",
            "",
            "| Run | Category | Stage | Cache state | Items | Cache load (s) | Compute (s) | Cache save (s) |",
            "|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for cost in analysis["runtime_costs"]:
        if not cost["cache_stages"]:
            lines.append(
                f'| {cost["label"]} | — | No cache-stage observations | — | — | — | — | — |'
            )
            continue
        for stage in cost["cache_stages"]:
            lines.append(
                f'| {cost["label"]} | {stage.get("category", "—")} | '
                f'{stage.get("name", "—")} | {stage.get("cache_status", "—")} | '
                f'{stage.get("item_count", "—")} | '
                f'{stage.get("cache_load_seconds", 0):.3f} | '
                f'{stage.get("compute_seconds", 0):.3f} | '
                f'{stage.get("cache_save_seconds", 0):.3f} |'
            )
    lines.extend(
        [
            "",
            "### Other recorded pipeline phases",
            "",
            "| Run | Category | Phase | Seconds |",
            "|---|---|---|---:|",
        ]
    )
    for cost in analysis["runtime_costs"]:
        for phase in cost["phase_timings"]:
            lines.append(
                f'| {cost["label"]} | {phase.get("category", "—")} | '
                f'{phase.get("name", "—")} | {phase.get("seconds", 0):.3f} |'
            )
    discrepancies = [
        (cost["label"], discrepancy)
        for cost in analysis["runtime_costs"]
        for discrepancy in cost["runtime_sidecar_setting_discrepancies"]
    ]
    if discrepancies:
        lines.extend(
            [
                "",
                "### Runtime-sidecar settings audit",
                "",
                "| Run | Setting | Weight used for result | Weight in runtime sidecar |",
                "|---|---|---:|---:|",
            ]
        )
        for label, discrepancy in discrepancies:
            lines.append(
                f'| {label} | {discrepancy["setting"]} | '
                f'{discrepancy["used_for_result"]} | '
                f'{discrepancy["recorded_in_runtime_sidecar"]} |'
            )
    lines.extend(
        [
            "",
            (
                "AZH-529 offline corpus encoding took "
                f'{analysis["runtime_costs"][0]["offline_costs"].get("corpus_encoding_seconds", 0):.2f} '
                "seconds and the depth-1,000 Faiss index build took "
                f'{analysis["runtime_costs"][0]["offline_costs"].get("faiss_index_build_seconds", 0):.2f} '
                "seconds. AZH-531 reports zero corpus-encoding and index-build time "
                "because it replays the saved candidate rankings. Cache load, stage "
                "compute, and cache-save time are shown separately above."
            ),
            "",
            (
                "Peak memory is sampled process working set at the run's recorded "
                f'{analysis["runtime_costs"][0]["memory"]["sample_interval_ms"]} ms '
                "interval; it can miss shorter transients and is not total host memory. "
                "The runtime-sidecar settings differences shown above are preselection "
                "snapshots. Fusion occurs after pair scoring, so those differences do "
                "not change the recorded pair-scoring time or memory."
            ),
            "",
            "## Limits",
            "",
        ]
    )
    lines.extend(f"- {limit}" for limit in analysis["limitations"])
    lines.extend(
        [
            "",
            "## Preserved inputs and outputs",
            "",
            f'Representation run: {analysis["source_runs"]["representation"]["run_id"]}',
            f'Code-reranker run: {analysis["source_runs"]["code_reranker"]["run_id"]}',
            "Machine-readable effects, per-query deltas, input hashes, aggregate checks, and stage-level costs are saved next to this report.",
            "",
        ]
    )
    return "\n".join(lines)


def _build_analysis(
    representation_run: dict[str, Any], code_run: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    query_ids = code_run["query_ids"]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sample_indices = rng.integers(
        0,
        len(query_ids),
        size=(BOOTSTRAP_REPLICATES, len(query_ids)),
        dtype=np.int32,
    )
    systems: dict[str, dict[str, Any]] = {}
    baseline = code_run["systems"]["e5"]
    for system_id in CODE_RERANKER_SYSTEMS:
        metric_means = {
            metric: float(
                np.mean(
                    [
                        code_run["systems"][system_id][query_id][metric]
                        for query_id in query_ids
                    ]
                )
            )
            for metric in METRICS
        }
        systems[system_id] = {
            "label": SYSTEM_LABELS[system_id],
            "mean_metrics": metric_means,
            "delta_vs_e5": {
                metric: metric_means[metric] - float(
                    np.mean([baseline[qid][metric] for qid in query_ids])
                )
                for metric in METRICS
            },
        }

    contrasts: dict[str, Any] = {}
    paired_deltas: list[dict[str, Any]] = []
    for candidate_id, reference_id, label in CONTRASTS:
        key = f"{candidate_id}_minus_{reference_id}"
        metric_results: dict[str, Any] = {}
        for metric in METRICS:
            candidate = np.asarray(
                [
                    code_run["systems"][candidate_id][query_id][metric]
                    for query_id in query_ids
                ],
                dtype=np.float64,
            )
            reference = np.asarray(
                [
                    code_run["systems"][reference_id][query_id][metric]
                    for query_id in query_ids
                ],
                dtype=np.float64,
            )
            metric_results[metric] = _paired_effect(
                candidate, reference, sample_indices
            )
            deltas = candidate - reference
            for query_id, candidate_value, reference_value, delta in zip(
                query_ids, candidate, reference, deltas, strict=True
            ):
                paired_deltas.append(
                    {
                        "query_id": query_id,
                        "contrast_id": key,
                        "candidate_system": candidate_id,
                        "reference_system": reference_id,
                        "metric": metric,
                        "candidate_value": float(candidate_value),
                        "reference_value": float(reference_value),
                        "delta": float(delta),
                        "outcome": (
                            "win"
                            if delta > TIE_TOLERANCE
                            else "loss"
                            if delta < -TIE_TOLERANCE
                            else "tie"
                        ),
                    }
                )
        contrasts[key] = {
            "candidate_system": candidate_id,
            "reference_system": reference_id,
            "label": label,
            "metrics": metric_results,
        }

    return (
        {
            "analysis_scope": "retrospective exploratory paired analysis of saved CosQA test runs",
            "query_count": len(query_ids),
            "systems": systems,
            "contrasts": contrasts,
            "bootstrap": {
                "method": "paired query bootstrap with replacement",
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": BOOTSTRAP_SEED,
                "interval": "95% percentile",
                "multiplicity_policy": (
                    "No adjustment; all estimates and intervals are descriptive and "
                    "exploratory, not simultaneous or confirmatory."
                ),
                "tie_tolerance": TIE_TOLERANCE,
            },
        },
        paired_deltas,
    )


def analyze_saved_runs(
    representation_run_dir: str | Path,
    code_reranker_run_dir: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Validate two complete saved runs, analyze paired outcomes, and preserve outputs."""
    representation_run = _load_run(
        Path(representation_run_dir), REPRESENTATION_SYSTEMS
    )
    code_run = _load_run(Path(code_reranker_run_dir), CODE_RERANKER_SYSTEMS)
    if representation_run["query_ids"] != code_run["query_ids"]:
        raise AnalysisInputError("The two complete runs do not have identical query IDs")
    aggregate_reconciliation = {
        "representation_run": _reconcile_aggregates(representation_run),
        "code_reranker_run": _reconcile_aggregates(code_run),
    }
    cross_run = _cross_run_parity(representation_run, code_run)
    effects, paired_deltas = _build_analysis(representation_run, code_run)
    generation_outputs = _generation_summary(code_run, representation_run)
    runtime_costs = [
        _cost_summary(representation_run, "AZH-529 full representation/fusion run"),
        _cost_summary(code_run, "AZH-531 code-reranker saved-ranking replay"),
    ]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    analysis_run_id = (
        f"azh533-paired-{timestamp}-"
        f"{representation_run['result']['cache_identity']}-"
        f"{code_run['result']['cache_identity']}"
    )
    module_path = Path(__file__).resolve()
    notebook_path = module_path.parents[1] / "notebooks" / "e5_hyde_rerank_experiment.ipynb"
    source_runs = {
        "representation": {
            "run_id": representation_run["result"]["run_id"],
            "cache_identity": representation_run["result"]["cache_identity"],
            "source_commit": representation_run["result"].get("git_commit"),
            "run_dir": representation_run["run_dir"].as_posix(),
            "input_files": representation_run["input_files"],
        },
        "code_reranker": {
            "run_id": code_run["result"]["run_id"],
            "cache_identity": code_run["result"]["cache_identity"],
            "source_commit": code_run["result"].get("git_commit"),
            "run_dir": code_run["run_dir"].as_posix(),
            "input_files": code_run["input_files"],
        },
    }
    generation_outputs["source_artifact"] = _resolve_generation_path(
        code_run, representation_run
    )[1]
    analysis: dict[str, Any] = {
        "analysis_run_id": analysis_run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_source": {
            "module_path": module_path.as_posix(),
            "module_sha256": sha256_file(module_path),
            "notebook_path": notebook_path.as_posix(),
            "notebook_sha256": (
                sha256_file(notebook_path) if notebook_path.is_file() else None
            ),
        },
        "dataset": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "split": "test",
            "query_count": QUERY_COUNT,
            "corpus_count": CORPUS_COUNT,
            "candidate_depth": CANDIDATE_DEPTH,
            "positive_judgments_per_query": 1,
            "evaluator": "coir-eval==0.7.0",
            "tie_policy": (
                "descending score, saved insertion order for ties, then unique "
                "ordinal evaluation scores"
            ),
        },
        **effects,
        "aggregate_reconciliation": aggregate_reconciliation,
        "shared_run_parity": cross_run,
        "source_runs": source_runs,
        "generation_outputs": generation_outputs,
        "retrieval_failures": _retrieval_failures(code_run),
        "runtime_costs": runtime_costs,
        "limitations": [
            "CosQA test outcomes and validation-selected settings were already examined; this analysis is retrospective and exploratory, not independent confirmation.",
            "The 95% paired intervals are unadjusted across the system matrix and metrics. They are not simultaneous bounds and do not support confirmatory claims.",
            "Each query has one judged relevant item. Metrics and failure counts concern recovery of that item; other retrieved code is unjudged, not known irrelevant.",
            "String length, emptiness, and duplication are surface-form diagnostics only. There are no human semantic-quality labels for generated hypotheses; the Flan-T5-base quality concern remains unresolved.",
            "The code-trained reranker is a separate system arm. No combined HyDE plus code-reranker system was run, so that component interaction is unknown.",
            "The code-reranker training overlap with CosQA is unknown; no broad generalization claim is supported.",
            "Runtime is one CPU execution per recorded run. Peak process working set is sampled and can miss short transients; it is not total host memory.",
            "The AZH-531 runtime sidecar stores preselection fusion defaults for the code-reranker and hypothesis-only settings. The frozen result/selection records provide the settings used for metrics; pair-scoring time and memory are unaffected because fusion occurs after scoring.",
        ],
    }
    output_root_path = Path(output_root).expanduser().resolve()
    output_dir = output_root_path / analysis_run_id
    analysis["output_dir"] = output_dir.as_posix()
    analysis["output_files"] = {
        "summary": "paired_effects.json",
        "report": "paired_effects.md",
        "per_query_deltas": "paired_deltas.csv",
    }
    report = _markdown_report(analysis)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "paired_effects.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "paired_effects.md").write_text(report, encoding="utf-8")
    with (output_dir / "paired_deltas.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "query_id",
            "contrast_id",
            "candidate_system",
            "reference_system",
            "metric",
            "candidate_value",
            "reference_value",
            "delta",
            "outcome",
        ))
        writer.writeheader()
        writer.writerows(paired_deltas)
    return {
        "analysis_run_id": analysis_run_id,
        "output_dir": output_dir,
        "summary_path": output_dir / "paired_effects.json",
        "report_path": output_dir / "paired_effects.md",
        "deltas_path": output_dir / "paired_deltas.csv",
        "analysis": analysis,
        "report": report,
    }


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze preserved paired CosQA test runs. This command reads existing "
            "artifacts and does not run model inference or retrieval."
        )
    )
    parser.add_argument(
        "--representation-run-dir",
        required=True,
        type=Path,
        help="Complete AZH-529 representation/fusion run directory.",
    )
    parser.add_argument(
        "--code-reranker-run-dir",
        required=True,
        type=Path,
        help="Complete AZH-531 code-reranker run directory.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Parent directory for a new timestamped AZH-533 analysis run.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    try:
        result = analyze_saved_runs(
            args.representation_run_dir,
            args.code_reranker_run_dir,
            args.output_root,
        )
    except (AnalysisInputError, OSError, ValueError) as exc:
        print(f"AZH-533 analysis failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": "complete",
                "analysis_run_id": result["analysis_run_id"],
                "output_dir": result["output_dir"].as_posix(),
                "summary": result["summary_path"].as_posix(),
                "report": result["report_path"].as_posix(),
                "per_query_deltas": result["deltas_path"].as_posix(),
                "query_count": QUERY_COUNT,
                "system_count": len(CODE_RERANKER_SYSTEMS),
                "contrast_count": len(CONTRASTS),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
