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


def build_report(
    runs_dir: Path = DEFAULT_RUNS,
    matrix_path: Path = MATRIX_PATH,
    title: str = "Sentiment Analysis",
) -> str:
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in matrix["seed_protocol"]["seeds"]]
    runs = _load_runs(runs_dir) if runs_dir.exists() else []
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

    lines = [
        f"# {title} Experiment Results",
        "",
        "Status: **independent replication**. Original per-seed artifacts and the full seed list were not recovered.",
        "",
        "Scores are exact test triplet micro-F1 percentages. Only run manifests marked `complete` contribute; smoke, partial, blocked, failed, interrupted, or absent runs never become scores. Smoke, failed attempts, and interrupted attempts remain visible in the cell status.",
        "",
        f"Declared seeds: `{', '.join(map(str, seeds))}`. Expected runs per cell: {len(seeds)}.",
        "",
        "| Train → test | Architecture | Encoder | State | n | Seed F1 values | Mean | SD | Median | Min | Max | Chapter 5 target | Δ | Target status |",
        "| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
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
            "## Secondary metrics",
            "",
            "Each value is mean precision/recall/F1 (%) over complete seeds only. Explicit-only triplet scores exclude implicit-aspect gold; primary triplet scores above retain it.",
            "",
            "| Train → test | Architecture | Encoder | Explicit-only triplet P/R/F1 | Aspect P/R/F1 | Opinion P/R/F1 | POS P/R/F1 | NEU P/R/F1 | NEG P/R/F1 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
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
    title = args.output.stem.removesuffix("-report").upper()
    report = build_report(args.runs, title=title)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"wrote {args.output} ({len(report)} characters)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
