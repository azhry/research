"""Build a seed-aware report from immutable per-run JSON manifests."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "configs" / "experiment-matrix.json"
DEFAULT_RUNS = ROOT / "runs"


def _load_runs(runs_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(runs_dir.glob("*/manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        metrics_path = path.parent / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else None
        records.append({"path": path, "manifest": manifest, "metrics": metrics})
    return records


def _cell_stats(records: list[dict[str, Any]], expected_seeds: list[int]) -> dict[str, Any]:
    observed_statuses = [row["manifest"].get("status") for row in records]
    failed_runs = sum(status == "failed" for status in observed_statuses)
    interrupted_runs = sum(status == "interrupted" for status in observed_statuses)
    smoke_runs = sum(status == "smoke" for status in observed_statuses)
    running_runs = sum(status == "running" for status in observed_statuses)
    complete = [
        row
        for row in records
        if row["manifest"].get("status") == "complete"
        and row["metrics"] is not None
        and row["metrics"].get("test", {}).get("triplet", {}).get("f1") is not None
    ]
    scores: dict[int, float] = {}
    for row in complete:
        seed = int(row["manifest"]["seed"])
        if seed in scores:
            raise ValueError(f"multiple complete runs for seed {seed} in one cell")
        scores[seed] = 100.0 * float(row["metrics"]["test"]["triplet"]["f1"])
    if any(seed not in expected_seeds for seed in scores):
        raise ValueError("run manifest contains a seed outside the predeclared seed set")
    values = list(scores.values())
    expected_count = len(expected_seeds)
    if len(values) == expected_count:
        status = "complete"
    elif values:
        status = "partial; run in progress" if running_runs else "partial"
        if failed_runs:
            status += "; failed attempts"
        if interrupted_runs:
            status += "; interrupted attempts"
    elif running_runs:
        status = "running"
        if smoke_runs:
            status += "; smoke only so far"
        if failed_runs:
            status += "; failed attempts"
        if interrupted_runs:
            status += "; interrupted attempts"
    elif (failed_runs or interrupted_runs) and smoke_runs:
        status = "smoke only"
        if failed_runs:
            status += "; failed attempts"
        if interrupted_runs:
            status += "; interrupted attempts"
    elif failed_runs:
        status = "failed"
        if interrupted_runs:
            status += "; interrupted attempts"
    elif interrupted_runs:
        status = "interrupted"
        if smoke_runs:
            status = "smoke only; interrupted attempts"
    elif smoke_runs:
        status = "smoke only"
    else:
        status = "not run"
    return {
        "status": status,
        "scores": scores,
        "failed_runs": failed_runs,
        "interrupted_runs": interrupted_runs,
        "smoke_runs": smoke_runs,
        "running_runs": running_runs,
        "n": len(values),
        "expected": expected_count,
        "mean": statistics.mean(values) if values else None,
        "sd": statistics.stdev(values) if len(values) > 1 else None,
        "median": statistics.median(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def _mean_metric(records: list[dict[str, Any]], *path: str) -> dict[str, float] | None:
    values = []
    for row in records:
        current: Any = row["metrics"]
        for key in path:
            current = current.get(key) if isinstance(current, dict) else None
        if not isinstance(current, dict) or any(current.get(name) is None for name in ("precision", "recall", "f1")):
            continue
        values.append(current)
    if not values:
        return None
    return {
        name: statistics.mean(float(value[name]) for value in values)
        for name in ("precision", "recall", "f1")
    }


def _format_prf(value: dict[str, float] | None) -> str:
    if value is None:
        return "—"
    return "/".join(f"{100.0 * value[name]:.2f}" for name in ("precision", "recall", "f1"))


def _optimizer_learning_rates(run: dict[str, Any]) -> list[float]:
    resolved = run["manifest"].get("resolved_config", {})
    groups = resolved.get("pyabsa", {}).get("optimizer_groups", [])
    try:
        return [float(group["learning_rate"]) for group in groups]
    except (TypeError, KeyError, ValueError):
        return []


def _rates_match(actual: list[float], expected: list[float]) -> bool:
    return len(actual) == len(expected) and all(
        abs(left - right) <= 1e-12 for left, right in zip(actual, expected)
    )


def build_report(
    runs_dir: Path = DEFAULT_RUNS,
    matrix_path: Path = MATRIX_PATH,
    title: str | None = None,
) -> str:
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    title = title or matrix.get("study", "Sentiment Analysis")
    protocol_id = matrix["experiment_protocol_id"]
    uniform_history = matrix.get("historical_protocols", {}).get("uniform_optimizer_groups", {})
    uniform_protocol_id = uniform_history.get("id")
    uniform_protocol_label = uniform_history.get("label", "superseded uniform-rate protocol")
    seeds = [int(seed) for seed in matrix["seed_protocol"]["seeds"]]
    all_runs = _load_runs(runs_dir) if runs_dir.exists() else []
    runs = [
        run for run in all_runs
        if run["manifest"].get("experiment_protocol_id") == protocol_id
    ]
    baseline_scope = [
        run for run in all_runs
        if run["manifest"].get("experiment_protocol_id") != protocol_id
        and run["manifest"].get("train_dataset", run["manifest"].get("dataset")) == "hoasa"
        and run["manifest"].get("test_dataset", run["manifest"].get("dataset")) == "hoasa"
        and run["manifest"].get("architecture") == "emcgcn"
        and run["manifest"].get("encoder") in matrix["encoders"]
    ]
    baseline_candidates = [
        run for run in baseline_scope
        if run["manifest"].get("status") == "complete"
        and run["metrics"] is not None
        and run["metrics"].get("test", {}).get("triplet", {}).get("f1") is not None
    ]
    configured_lr = float(matrix["training"]["learning_rate"])
    native_group_rates = [configured_lr, configured_lr, 1e-3, 1e-3]
    native_schedule_runs = [
        run for run in baseline_candidates
        if _rates_match(_optimizer_learning_rates(run), native_group_rates)
    ]
    native_schedule_attempts = [
        run for run in baseline_scope
        if _rates_match(_optimizer_learning_rates(run), native_group_rates)
    ]
    uniform_schedule_runs = [
        run for run in baseline_candidates
        if _rates_match(_optimizer_learning_rates(run), [configured_lr] * 4)
    ]
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        manifest = run["manifest"]
        key = (
            manifest.get("train_dataset", manifest.get("dataset", "")),
            manifest.get("test_dataset", manifest.get("dataset", "")),
            manifest.get("architecture", ""),
            manifest.get("encoder", ""),
        )
        grouped[key].append(run)

    rows: list[tuple[str, str, str, str, dict[str, Any], float | None]] = []
    cell_stats: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for dataset in matrix["datasets"]:
        for architecture in matrix["architectures"]:
            for encoder in matrix["encoders"]:
                key = (dataset, dataset, architecture, encoder)
                stats = _cell_stats(grouped.get(key, []), seeds)
                cell_stats[key] = stats
                target = (
                    matrix["encoders"][encoder]["target_mean_triplet_f1"]
                    if dataset == "hoasa" and architecture == "emcgcn"
                    else None
                )
                if dataset == "casa" and stats["status"] == "not run":
                    stats["status"] = "blocked: human gold"
                rows.append((dataset, dataset, architecture, encoder, stats, target))
    for transfer in matrix["transfer_cells"]:
        key = (
            transfer["train_dataset"],
            transfer["test_dataset"],
            transfer["architecture"],
            transfer["encoder"],
        )
        stats = _cell_stats(grouped.get(key, []), seeds)
        cell_stats[key] = stats
        if stats["status"] == "not run":
            stats["status"] = "blocked: CASA human gold"
        rows.append((*key, stats, None))

    target_rows = [row for row in rows if row[5] is not None]
    protocol_complete_count = sum(row[4]["n"] for row in target_rows)
    protocol_expected_count = sum(row[4]["expected"] for row in target_rows)
    target_cells_complete = all(row[4]["status"] == "complete" for row in target_rows)
    target_met_count = sum(
        row[4]["status"] == "complete"
        and row[4]["mean"] is not None
        and row[4]["mean"] >= row[5]
        for row in target_rows
    )
    if not target_rows or not target_cells_complete:
        target_outcome = "**Chapter 5 target outcome: NOT YET FULLY MEASURED.**"
    elif target_met_count == len(target_rows):
        target_outcome = f"**Chapter 5 target outcome: MET ({target_met_count}/{len(target_rows)} baseline means).**"
    else:
        target_outcome = f"**Chapter 5 target outcome: NOT MET ({target_met_count}/{len(target_rows)} baseline means).**"

    reference_diagnostics = []
    for train, test, architecture, encoder, stats, target in target_rows:
        complete = [
            row
            for row in grouped.get((train, test, architecture, encoder), [])
            if row["manifest"].get("status") == "complete" and row["metrics"] is not None
        ]
        explicit = _mean_metric(complete, "test", "explicit_only_triplet")
        explicit_f1 = 100.0 * explicit["f1"] if explicit is not None else None
        reference_diagnostics.append((encoder, stats, target, explicit_f1))
    explicit_only_deltas = [
        explicit_f1 - stats["mean"]
        for _encoder, stats, _target, explicit_f1 in reference_diagnostics
        if explicit_f1 is not None and stats["mean"] is not None
    ]
    explicit_only_target_count = sum(
        explicit_f1 is not None and explicit_f1 >= target
        for _encoder, _stats, target, explicit_f1 in reference_diagnostics
    )
    if explicit_only_deltas:
        implicit_sensitivity = (
            f"On the same selected predictions, excluding implicit-aspect gold raises mean F1 by "
            f"{min(explicit_only_deltas):.2f}–{max(explicit_only_deltas):.2f} points; "
            f"{explicit_only_target_count}/{len(target_rows)} explicit-only means meet the historical references. "
            "Chapter 5 does not document whether its targets excluded implicit triplets, so this quantifies metric sensitivity without establishing exact comparability."
        )
    else:
        implicit_sensitivity = "The active native-group protocol has no complete runs yet, so its implicit-aspect sensitivity is not measured. Completed native-group and superseded uniform-rate runs are summarized separately below."

    lines = [
        f"# {title} results",
        "",
        "Status: **independent replication**. Original per-seed artifacts and the full seed list were not recovered.",
        "",
        f"Active run protocol: `{protocol_id}`. {matrix['training']['learning_rate_policy']}",
        "",
        target_outcome,
        "",
        "The Chapter 5 draft describes selecting `Max-Test-F1` on the test data during training. This replication selects checkpoints on validation and evaluates the held-out test once, as required by the approved protocol. The original seed list, checkpoints, and run outputs were not recovered, so this is not a method-matched rerun. The primary measure retains all 418 implicit-aspect HoASA test triplets, which the current PyABSA EMCGCN adapter cannot predict; explicit-only scores below are secondary diagnostics and do not replace the primary outcome.",
        "",
        implicit_sensitivity,
        "",
        "## Optimizer schedule correction",
        "",
        "Source inspection of pinned PyABSA 2.4.3 shows that `config.learning_rate=2e-5` is used for the two Transformer parameter groups; its EMCGCN instructor separately assigns `1e-3` to both graph/classifier groups. A superseded adapter incorrectly forced all four groups to `2e-5`. In matched mBERT seed-0 runs with the same data hashes, model revision, seed, and sequence length, native rates measured 70.15 full-gold F1 / 76.76 explicit-only F1, while the uniform-rate run measured 67.51 / 73.69. The native schedule is restored in the active code and configuration. Across the five-seed native-rate history, all four full-gold means remain below the Chapter 5 references; only three of four explicit-only means meet them, and that secondary metric does not establish exact reproduction. Completed uniform-rate runs remain historical evidence under their original protocol and are not pooled with the active protocol.",
        "",
        "Scores are exact test triplet micro-F1 percentages. Only run manifests marked `complete` contribute; smoke, partial, blocked, failed, interrupted, or absent runs never become scores. Attempt counts for the HoASA EMCGCN baseline are summarized below.",
        "",
        f"Declared seeds: `{', '.join(map(str, seeds))}`. Expected runs per cell: {len(seeds)}.",
        "",
        "## Completed historical optimizer schedules",
        "",
        "The native-rate history contains 20 complete measured runs (five per encoder) but predates explicit protocol IDs; its manifests record the expected group rates and all runs remain under their original identities. Those full-gold means are below all four Chapter 5 references. The second schedule is the completed uniform-rate protocol, now superseded because its adapter overrode PyABSA's graph/classifier rates. Neither historical schedule is relabeled as an active-protocol run. "
        f"The active v3 protocol has {protocol_complete_count}/{protocol_expected_count} complete HoASA EMCGCN seed runs; those measurements are reported separately below and are not pooled with historical schedules.",
        "",
        "The explicit-only scores below use the same predictions and exclude implicit-aspect triplets from the gold denominator. Chapter 5 does not document its implicit-label policy, and it selects the best test-set epoch while this replication selects on validation. These figures are sensitivity diagnostics, not evidence that a target has been reproduced.",
        "",
        "| Schedule | Encoder | Runs | Full-gold seed F1 (%) | Full-gold mean | Explicit-only seed F1 (%) | Explicit-only mean | Chapter 5 reference | Full-gold Δ | Explicit-only Δ |",
        "| --- | --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for schedule_name, schedule_runs in (
        ("Native PyABSA groups", native_schedule_runs),
        ("Superseded uniform groups", uniform_schedule_runs),
    ):
        for encoder in matrix["encoders"]:
            prior = sorted(
                (run for run in schedule_runs if run["manifest"].get("encoder") == encoder),
                key=lambda run: (int(run["manifest"]["seed"]), run["manifest"].get("run_id", "")),
            )
            scores = [
                100.0 * float(run["metrics"]["test"]["triplet"]["f1"])
                for run in prior
            ]
            explicit_by_run = [
                (run, 100.0 * float(run["metrics"]["test_explicit_aspects_only"]["triplet"]["f1"]))
                for run in prior
                if run["metrics"].get("test_explicit_aspects_only", {}).get("triplet", {}).get("f1") is not None
            ]
            explicit_scores = [score for _run, score in explicit_by_run]
            seed_scores = ", ".join(
                f"{run['manifest']['seed']}:{score:.2f}"
                for run, score in zip(prior, scores)
            ) or "—"
            explicit_seed_scores = ", ".join(
                f"{run['manifest']['seed']}:{score:.2f}"
                for run, score in explicit_by_run
            ) or "—"
            full_mean = statistics.mean(scores) if scores else None
            explicit_mean = statistics.mean(explicit_scores) if explicit_scores else None
            target = matrix["encoders"][encoder].get("target_mean_triplet_f1")
            target_text = f"{float(target):.2f}" if target is not None else "—"
            full_delta = f"{full_mean - float(target):+.2f}" if full_mean is not None and target is not None else "—"
            explicit_delta = f"{explicit_mean - float(target):+.2f}" if explicit_mean is not None and target is not None else "—"
            full_mean_text = f"{full_mean:.2f}" if full_mean is not None else "—"
            explicit_mean_text = f"{explicit_mean:.2f}" if explicit_mean is not None else "—"
            lines.append(
                f"| {schedule_name} | {encoder} | {len(scores)} | {seed_scores} | {full_mean_text} | "
                f"{explicit_seed_scores} | {explicit_mean_text} | {target_text} | "
                f"{full_delta} | {explicit_delta} |"
            )

    lines.extend(
        [
        "",
        "| Train → test | Architecture | Encoder | State | n | Seed F1 values | Mean | SD | Median | Min | Max | Chapter 5 target | Δ | Target status |",
        "| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for train, test, architecture, encoder, stats, target in rows:
        display = train if train == test else f"{train} → {test}"
        score_text = ", ".join(
            f"{seed}:{stats['scores'][seed]:.2f}" if seed in stats["scores"] else f"{seed}:—"
            for seed in seeds
        )
        target_text = f"{target:.2f}" if target is not None else "—"
        complete_seed_set = stats["status"] == "complete"
        delta_text = (
            f"{stats['mean'] - target:+.2f}"
            if target is not None and stats["mean"] is not None and complete_seed_set
            else "—"
        )
        if target is None:
            target_status = "not applicable"
        elif stats["mean"] is None or not complete_seed_set:
            target_status = "not measured"
        else:
            target_status = "meets" if stats["mean"] >= target else "below"
        fields = [stats[name] for name in ("mean", "sd", "median", "min", "max")]
        formatted = [f"{value:.2f}" if value is not None else "—" for value in fields]
        lines.append(
            f"| {display} | {architecture} | {encoder} | {stats['status']} | {stats['n']}/{stats['expected']} | {score_text} | "
            + " | ".join(formatted)
            + f" | {target_text} | {delta_text} | {target_status} |"
        )

    lines.extend(
        [
            "",
            "## Chapter 5 reference diagnostics",
            "",
            "These rows summarize the active native-group protocol. The primary full-gold metric retains implicit-aspect triplets; explicit-only diagnostics do not establish a Chapter 5 match because its implicit-label policy is undocumented.",
            "",
            "| Encoder | Chapter 5 target | Primary full-gold mean | Primary Δ | Explicit-only mean (diagnostic) | Diagnostic Δ | F1 change when implicit gold is excluded |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for encoder, stats, target, explicit_f1 in reference_diagnostics:
        primary_delta = stats["mean"] - target if stats["mean"] is not None else None
        diagnostic_delta = explicit_f1 - target if explicit_f1 is not None else None
        implicit_delta = explicit_f1 - stats["mean"] if explicit_f1 is not None and stats["mean"] is not None else None
        values = (stats["mean"], primary_delta, explicit_f1, diagnostic_delta, implicit_delta)
        formatted = [f"{value:.2f}" if value is not None else "—" for value in values]
        lines.append(f"| {encoder} | {target:.2f} | " + " | ".join(formatted) + " |")

    lines.extend(
        [
            "",
            "## HoASA EMCGCN attempt history",
            "",
            "Complete seed runs contribute to the score table. Smoke, failed, and interrupted attempts remain in the run archive and are excluded from benchmark metrics.",
            "",
            "| Encoder | Complete seed runs | Smoke attempts | Failed attempts | Interrupted attempts |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for _train, _test, _architecture, encoder, stats, _target in target_rows:
        lines.append(
            f"| {encoder} | {stats['n']}/{stats['expected']} | {stats['smoke_runs']} | "
            f"{stats['failed_runs']} | {stats['interrupted_runs']} |"
        )

    superseded_attempts = [
        run for run in all_runs
        if uniform_protocol_id
        and run["manifest"].get("experiment_protocol_id") == uniform_protocol_id
        and run["manifest"].get("train_dataset", run["manifest"].get("dataset")) == "hoasa"
        and run["manifest"].get("test_dataset", run["manifest"].get("dataset")) == "hoasa"
        and run["manifest"].get("architecture") == "emcgcn"
        and run["manifest"].get("encoder") in matrix["encoders"]
    ]
    if superseded_attempts:
        lines.extend(
            [
                "",
                "## Superseded uniform-rate attempt history",
                "",
                f"The {uniform_protocol_label} queue ended with all 20 declared seed conditions complete and no queue-level failures. Earlier failed and interrupted attempts remain in the run archive and are not scores.",
                "",
                "| Encoder | Complete runs | Failed attempts | Interrupted attempts |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for encoder in matrix["encoders"]:
            attempts = [
                run for run in superseded_attempts
                if run["manifest"].get("encoder") == encoder
            ]
            complete_count = sum(run["manifest"].get("status") == "complete" for run in attempts)
            failed_count = sum(run["manifest"].get("status") == "failed" for run in attempts)
            interrupted_count = sum(run["manifest"].get("status") == "interrupted" for run in attempts)
            lines.append(
                f"| {encoder} | {complete_count}/{len(seeds)} | {failed_count} | {interrupted_count} |"
            )

    if native_schedule_attempts:
        lines.extend(
            [
                "",
                "## Historical native-rate attempt history",
                "",
                "These unprotocolized manifests record the native group rates. Only complete runs with metrics contribute to the historical score table; smoke, failed, and interrupted attempts remain excluded.",
                "",
                "| Encoder | Complete runs | Failed attempts | Interrupted attempts | Smoke attempts |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for encoder in matrix["encoders"]:
            attempts = [
                run for run in native_schedule_attempts
                if run["manifest"].get("encoder") == encoder
            ]
            complete_count = sum(
                run["manifest"].get("status") == "complete"
                and run["metrics"] is not None
                and run["metrics"].get("test", {}).get("triplet", {}).get("f1") is not None
                for run in attempts
            )
            failed_count = sum(run["manifest"].get("status") == "failed" for run in attempts)
            interrupted_count = sum(run["manifest"].get("status") == "interrupted" for run in attempts)
            smoke_count = sum(run["manifest"].get("status") == "smoke" for run in attempts)
            lines.append(
                f"| {encoder} | {complete_count}/{len(seeds)} | {failed_count} | "
                f"{interrupted_count} | {smoke_count} |"
            )

    unclassified_attempts = [
        run for run in all_runs
        if not run["manifest"].get("experiment_protocol_id")
        and run["manifest"].get("train_dataset", run["manifest"].get("dataset")) == "hoasa"
        and run["manifest"].get("test_dataset", run["manifest"].get("dataset")) == "hoasa"
        and run["manifest"].get("architecture") == "emcgcn"
        and run["manifest"].get("encoder") in matrix["encoders"]
        and run["manifest"].get("status") in {"failed", "interrupted", "smoke"}
        and not _optimizer_learning_rates(run)
    ]
    if unclassified_attempts:
        lines.extend(
            [
                "",
                "## Unprotocolized attempts without optimizer metadata",
                "",
                "These older failed, interrupted, or smoke manifests do not record resolved optimizer groups, so they cannot be assigned to either schedule and contribute no scores. They remain in the run archive.",
                "",
                "| Encoder | Failed | Interrupted | Smoke |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for encoder in matrix["encoders"]:
            attempts = [
                run for run in unclassified_attempts
                if run["manifest"].get("encoder") == encoder
            ]
            failed_count = sum(run["manifest"].get("status") == "failed" for run in attempts)
            interrupted_count = sum(run["manifest"].get("status") == "interrupted" for run in attempts)
            smoke_count = sum(run["manifest"].get("status") == "smoke" for run in attempts)
            lines.append(f"| {encoder} | {failed_count} | {interrupted_count} | {smoke_count} |")

    lines.extend(
        [
            "",
            "## Secondary metrics",
            "",
            "Each value is mean precision/recall/F1 (%) over complete seeds only. The full-source opinion metric includes opinion spans linked to implicit-aspect gold even though those triplets are absent from the PyABSA training view; the adjacent explicit-triplet-only opinion metric removes those spans from its denominator.",
            "",
            "| Train → test | Architecture | Encoder | Explicit-only triplet P/R/F1 | Aspect P/R/F1 | Opinion P/R/F1 (full source) | Opinion P/R/F1 (explicit triplets) | POS P/R/F1 | NEU P/R/F1 | NEG P/R/F1 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for train, test, architecture, encoder, _stats, _target in rows:
        key = (train, test, architecture, encoder)
        complete = [
            row
            for row in grouped.get(key, [])
            if row["manifest"].get("status") == "complete" and row["metrics"] is not None
        ]
        metrics = [
            _mean_metric(complete, "test", "explicit_only_triplet"),
            _mean_metric(complete, "test", "aspect_term"),
            _mean_metric(complete, "test", "opinion_term"),
            _mean_metric(complete, "test_explicit_aspects_only", "opinion_term"),
            _mean_metric(complete, "test", "by_polarity", "POS"),
            _mean_metric(complete, "test", "by_polarity", "NEU"),
            _mean_metric(complete, "test", "by_polarity", "NEG"),
        ]
        display = train if train == test else f"{train} → {test}"
        lines.append(
            f"| {display} | {architecture} | {encoder} | "
            + " | ".join(_format_prf(value) for value in metrics)
            + " |"
        )

    lines.extend(
        [
            "",
            "## Protocol and evidence limits",
            "",
            "- Checkpoints are selected on validation triplet F1; held-out test predictions are produced and evaluated once per model.",
            "- HoASA source `[-1]` implicit aspects remain in canonical gold and the primary test denominator. The report must also show explicit-only diagnostics when the selected model cannot represent implicit aspects.",
            "- CASA cells stay blocked until all splits have complete, human-adjudicated, sentiment-bearing ASTE annotations, an approved aspect-to-category mapping, and frozen hashes.",
            "- Every score must resolve to one immutable run directory with model/data revisions, seed, configuration, environment, checkpoint, predictions, and digests.",
            "- No seed-level results are present unless an immutable run manifest and metrics file exist.",
            "",
        ]
    )
    lines.extend(
        [
            "## Paired architecture deltas",
            "",
            "Delta is SSGCN minus EMCGCN in test triplet micro-F1 percentage points, paired by dataset, encoder, and seed. Only seeds with complete runs for both architectures are included.",
            "",
            "| Train → test | Encoder | Paired seeds | Per-seed Δ | Mean Δ |",
            "| --- | --- | --- | --- | ---: |",
        ]
    )
    pairs: list[tuple[str, str, str]] = [
        (dataset, dataset, encoder)
        for dataset in matrix["datasets"]
        for encoder in matrix["encoders"]
    ]
    pairs.extend(
        sorted(
            {
                (item["train_dataset"], item["test_dataset"], item["encoder"])
                for item in matrix["transfer_cells"]
            }
        )
    )
    for train, test, encoder in pairs:
        emc_key = (train, test, "emcgcn", encoder)
        ssg_key = (train, test, "ssgcn", encoder)
        emc = cell_stats.get(emc_key, {"scores": {}})["scores"]
        ssg = cell_stats.get(ssg_key, {"scores": {}})["scores"]
        paired = sorted(set(emc) & set(ssg))
        deltas = {seed: ssg[seed] - emc[seed] for seed in paired}
        delta_text = ", ".join(
            f"{seed}:{deltas[seed]:+.2f}" if seed in deltas else f"{seed}:—"
            for seed in seeds
        )
        mean_delta = f"{statistics.mean(deltas.values()):+.2f}" if deltas else "—"
        display = train if train == test else f"{train} → {test}"
        lines.append(
            f"| {display} | {encoder} | {len(paired)}/{len(seeds)} | {delta_text} | {mean_delta} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"wrote {args.output} ({len(report)} characters)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
