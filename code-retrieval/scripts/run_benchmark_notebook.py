"""Execute a notebook with this project's pinned Python environment.

Jupyter's default ``python3`` kernelspec can silently select a global Python
installation instead of the interpreter that launched ``nbconvert``. This
runner creates a temporary kernelspec that points at the current interpreter
and rejects environments that do not match ``.python-version`` and
``requirements.txt``. Every invocation also preserves the executed notebook,
console output, per-cell lifecycle trace, run metadata, and query-level analysis.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import uuid

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_EXPORTER = PROJECT_ROOT / "scripts" / "export_analysis_traces.py"
RESULT_ARTIFACTS = {
    "e5_baseline_experiment": PROJECT_ROOT / "artifacts" / "e5_baseline" / "result.json",
    "e5_baseline_full_experiment": PROJECT_ROOT / "artifacts" / "e5_baseline_full" / "result.json",
    "e5_hyde_experiment": PROJECT_ROOT / "artifacts" / "e5_hyde" / "result.json",
    "e5_rerank_experiment": PROJECT_ROOT / "artifacts" / "e5_rerank" / "result.json",
    "e5_hyde_rerank_experiment": PROJECT_ROOT / "artifacts" / "e5_hyde_rerank" / "result.json",
}
SECRET_VALUE_KEYS = re.compile(r"(?i)(token|password|secret|api[_-]?key)")
AUTHORIZATION_VALUE = re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s\"']+")


def require_pinned_runtime() -> None:
    expected_python = (PROJECT_ROOT / ".python-version").read_text(encoding="utf-8").strip()
    actual_python = ".".join(map(str, sys.version_info[:3]))
    if actual_python != expected_python:
        raise SystemExit(
            f"Python runtime mismatch: expected {expected_python}, got {actual_python}. "
            "Activate code-retrieval/.venv before running this script."
        )

    mismatches: list[str] = []
    for raw_line in (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        requirement = raw_line.strip()
        if not requirement or requirement.startswith("#"):
            continue
        if "==" not in requirement:
            raise SystemExit(f"requirements.txt contains an unpinned dependency: {requirement}")
        package, expected_version = requirement.split("==", maxsplit=1)
        try:
            actual_version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual_version = "not installed"
        if actual_version != expected_version:
            mismatches.append(f"{package}: expected {expected_version}, got {actual_version}")
    if mismatches:
        raise SystemExit(
            "Pinned dependency mismatch; refusing to label this a benchmark run:\n  "
            + "\n  ".join(mismatches)
        )


def _project_path(value: str, *, must_exist: bool) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if must_exist and (not path.is_file() or path.suffix.lower() != ".ipynb"):
        raise SystemExit(f"notebook does not exist or is not an .ipynb file: {path}")
    return path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _redact(text: str, environment: dict[str, str] | None = None) -> str:
    redacted = AUTHORIZATION_VALUE.sub(r"\1[REDACTED]", text)
    for key, value in (environment or {}).items():
        if SECRET_VALUE_KEYS.search(key) and value and len(value) >= 6:
            redacted = redacted.replace(value, "[REDACTED]")
    return redacted


def _write_jsonl(path: Path, events: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _append_jsonl(path: Path, event: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")


def _cell_contexts(notebook: nbformat.NotebookNode) -> list[dict[str, object]]:
    contexts: list[dict[str, object]] = []
    section = ""
    for cell_index, cell in enumerate(notebook.cells):
        source = str(cell.get("source", ""))
        if cell.cell_type == "markdown":
            heading = next(
                (line.strip() for line in source.splitlines() if line.lstrip().startswith("#")),
                None,
            )
            if heading:
                section = heading
        elif cell.cell_type == "code":
            contexts.append(
                {
                    "cell_index": cell_index,
                    "cell_id": str(cell.get("id", "")),
                    "section": section,
                    "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                    "source_line_count": len(source.splitlines()),
                }
            )
    return contexts


def _trace_bootstrap(contexts: list[dict[str, object]]) -> str:
    encoded_contexts = json.dumps(contexts, ensure_ascii=False)
    return f'''# Codex run trace bootstrap; execution code remains unmodified.
import datetime as __codex_datetime
import hashlib as __codex_hashlib
import json as __codex_json
import os as __codex_os
import time as __codex_time

__codex_trace_path = __codex_os.environ["CODE_RETRIEVAL_KERNEL_TRACE_PATH"]
__codex_trace_contexts = {encoded_contexts}
__codex_trace_state = {{"sequence": 0, "started": {{}}}}

def __codex_trace_emit(event_name, **fields):
    event = {{
        "event": event_name,
        "timestamp_utc": __codex_datetime.datetime.now(__codex_datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        **fields,
    }}
    with open(__codex_trace_path, "a", encoding="utf-8", newline="\\n") as trace_file:
        trace_file.write(__codex_json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\\n")

def __codex_before_cell(info):
    __codex_trace_state["sequence"] += 1
    sequence = __codex_trace_state["sequence"]
    execution_count = getattr(info, "execution_count", None)
    context = __codex_trace_contexts[sequence - 1] if sequence <= len(__codex_trace_contexts) else {{}}
    started = __codex_time.perf_counter()
    __codex_trace_state["started"][sequence] = started
    __codex_trace_emit(
        "cell_started",
        cell_sequence=sequence,
        execution_count=execution_count,
        cell_index=context.get("cell_index"),
        cell_id=context.get("cell_id"),
        section=context.get("section"),
        source_sha256=__codex_hashlib.sha256(str(getattr(info, "raw_cell", "")).encode("utf-8")).hexdigest(),
    )

def __codex_after_cell(result):
    execution_count = getattr(result, "execution_count", None)
    sequence = __codex_trace_state["sequence"]
    started = __codex_trace_state["started"].pop(sequence, None)
    if started is None:
        return
    context = __codex_trace_contexts[sequence - 1] if sequence <= len(__codex_trace_contexts) else {{}}
    error = getattr(result, "error_before_exec", None) or getattr(result, "error_in_exec", None)
    __codex_trace_emit(
        "cell_completed",
        cell_sequence=sequence,
        execution_count=execution_count,
        cell_index=context.get("cell_index"),
        cell_id=context.get("cell_id"),
        section=context.get("section"),
        status="failed" if error else "completed",
        elapsed_seconds=__codex_time.perf_counter() - started,
        error_type=type(error).__name__ if error else None,
        error_message=str(error)[:2000] if error else None,
    )

__codex_trace_emit("notebook_kernel_started", code_cell_count=len(__codex_trace_contexts))
get_ipython().events.register("pre_run_cell", __codex_before_cell)
get_ipython().events.register("post_run_cell", __codex_after_cell)
'''


def _run_streaming(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    log_path: Path,
    append: bool,
    trace_events: list[dict[str, object]],
    label: str,
) -> int:
    mode = "a" if append else "w"
    with log_path.open(mode, encoding="utf-8", newline="") as log_file:
        log_file.write(f"\n[{_utc_now()}] {label}: {' '.join(command)}\n")
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            safe_line = _redact(line, environment)
            sys.stdout.write(safe_line)
            sys.stdout.flush()
            log_file.write(safe_line)
            log_file.flush()
        exit_code = process.wait()
    trace_events.append(
        {
            "event": f"{label}_completed",
            "timestamp_utc": _utc_now(),
            "exit_code": exit_code,
        }
    )
    return exit_code


def _cell_output_text(cell: nbformat.NotebookNode, environment: dict[str, str]) -> list[dict[str, object]]:
    outputs: list[dict[str, object]] = []
    for output in cell.get("outputs", []):
        output_type = output.get("output_type")
        entry: dict[str, object] = {"output_type": output_type}
        if output_type == "stream":
            entry["name"] = output.get("name")
            entry["text"] = _redact("".join(output.get("text", "")), environment)
        elif output_type == "error":
            entry["ename"] = output.get("ename")
            entry["evalue"] = _redact(str(output.get("evalue", "")), environment)
            entry["traceback"] = [_redact(str(line), environment) for line in output.get("traceback", [])]
        else:
            data = output.get("data", {})
            text_data = {
                mime_type: _redact(
                    value if isinstance(value, str) else json.dumps(value, ensure_ascii=False),
                    environment,
                )
                for mime_type, value in data.items()
                if mime_type in {"text/plain", "text/markdown", "text/html", "application/json"}
            }
            if text_data:
                entry["data"] = text_data
        outputs.append(entry)
    return outputs


def _complete_missing_cell_events(
    events: list[dict[str, object]],
    *,
    process_finished_at: str,
    process_exit: int,
) -> list[dict[str, object]]:
    """Close lifecycle spans when a kernel hook omits its post-cell callback.

    In IPython, the pre-cell event does not promise an execution count. Pairing
    by cell sequence lets the normal callback record exact durations; this
    fallback uses the next cell's start (or process exit for the final cell)
    only when a completion callback is absent, and labels that timing source.
    """
    started = {
        int(event["cell_sequence"]): event
        for event in events
        if event.get("event") == "cell_started" and event.get("cell_sequence") is not None
    }
    completed = {
        int(event["cell_sequence"])
        for event in events
        if event.get("event") == "cell_completed" and event.get("cell_sequence") is not None
    }
    completed_events: list[dict[str, object]] = []
    for sequence in sorted(started):
        if sequence in completed:
            continue
        start_event = started[sequence]
        next_started = started.get(sequence + 1)
        end_timestamp = (
            str(next_started.get("timestamp_utc")) if next_started else process_finished_at
        )
        try:
            start_time = datetime.fromisoformat(str(start_event["timestamp_utc"]).replace("Z", "+00:00"))
            end_time = datetime.fromisoformat(end_timestamp.replace("Z", "+00:00"))
            elapsed = max(0.0, (end_time - start_time).total_seconds())
        except (KeyError, ValueError):
            elapsed = None
        completed_events.append(
            {
                "event": "cell_completed",
                "timestamp_utc": end_timestamp,
                "cell_sequence": sequence,
                "execution_count": start_event.get("execution_count"),
                "cell_index": start_event.get("cell_index"),
                "cell_id": start_event.get("cell_id"),
                "section": start_event.get("section"),
                "status": "completed_inferred" if process_exit == 0 else "failed_or_interrupted_inferred",
                "elapsed_seconds": elapsed,
                "timing_source": "next_cell_start" if next_started else "notebook_process_exit",
                "outputs": [],
            }
        )
    if not completed_events:
        return events
    combined = [*events, *completed_events]
    combined.sort(key=lambda event: (str(event.get("timestamp_utc", "")), 0 if event.get("event") == "cell_completed" else 1))
    return combined


def _load_fresh_result(notebook: Path, started_timestamp: float) -> tuple[Path | None, dict[str, object] | None]:
    result_path = RESULT_ARTIFACTS.get(notebook.stem)
    if not result_path or not result_path.is_file() or result_path.stat().st_mtime < started_timestamp:
        return None, None
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None, None
    return result_path, result if isinstance(result, dict) else None


def _result_rows(result: dict[str, object]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for row in result.get("results", []) if isinstance(result.get("results"), list) else []:
        if isinstance(row, dict) and row.get("ndcg_at_10") is not None:
            scores[str(row.get("system_id"))] = float(row["ndcg_at_10"])
    systems = result.get("systems")
    if isinstance(systems, dict):
        for system_id, row in systems.items():
            if isinstance(row, dict) and row.get("ndcg_at_10") is not None:
                scores[str(system_id)] = float(row["ndcg_at_10"])
    if result.get("system_id") and result.get("ndcg_at_10") is not None:
        scores[str(result["system_id"])] = float(result["ndcg_at_10"])
    return scores


def _result_candidate_depth(result: dict[str, object]) -> int | None:
    value = result.get("candidate_depth")
    if value is not None:
        return int(value)
    fusion = result.get("fusion")
    if isinstance(fusion, dict) and fusion.get("candidate_depth") is not None:
        return int(fusion["candidate_depth"])
    for row in result.get("results", []) if isinstance(result.get("results"), list) else []:
        if isinstance(row, dict) and row.get("candidate_depth") is not None:
            return int(row["candidate_depth"])
    systems = result.get("systems")
    if isinstance(systems, dict):
        for row in systems.values():
            if isinstance(row, dict) and row.get("candidate_depth") is not None:
                return int(row["candidate_depth"])
    return None


def _result_evaluator(result: dict[str, object]) -> object | None:
    evaluator = result.get("evaluator")
    if evaluator is not None:
        return evaluator
    if result.get("evaluator_package") is not None:
        return {"evaluator_package": result["evaluator_package"]}
    rows = result.get("results") if isinstance(result.get("results"), list) else []
    systems = result.get("systems")
    if isinstance(systems, dict):
        rows = [*rows, *systems.values()]
    for row in rows:
        if not isinstance(row, dict):
            continue
        evaluator = row.get("evaluator")
        if evaluator is not None:
            return evaluator
        if row.get("evaluator_package") is not None:
            return {"evaluator_package": row["evaluator_package"]}
    return None


def _baseline_parity(
    notebook: Path,
    result: dict[str, object] | None,
    reference: dict[str, object] | None,
) -> dict[str, object] | None:
    if notebook.stem != "e5_baseline_full_experiment" or result is None:
        return None
    if reference is None:
        return {"status": "unavailable", "reason": "no prior four-system result was present at run start"}

    ref_scores = _result_rows(reference)
    current_scores = _result_rows(result)
    reference_score = ref_scores.get("e5")
    current_score = current_scores.get("e5")
    ref_dataset = reference.get("dataset", {})
    current_dataset = result.get("dataset", {})
    ref_splits = ref_dataset.get("splits", {}) if isinstance(ref_dataset, dict) else {}
    current_splits = current_dataset.get("splits", {}) if isinstance(current_dataset, dict) else {}
    ref_revision = reference.get("model", {}).get("configured_revision") if isinstance(reference.get("model"), dict) else None
    current_revision = result.get("model", {}).get("revision") if isinstance(result.get("model"), dict) else None
    ref_evaluator = reference.get("evaluator_package")
    if ref_evaluator is None and isinstance(reference.get("evaluator"), dict):
        ref_evaluator = reference["evaluator"].get("evaluator_package")
    current_evaluator = result.get("evaluator_package")
    if current_evaluator is None and isinstance(result.get("evaluator"), dict):
        current_evaluator = result["evaluator"].get("evaluator_package")
    comparisons = {
        "candidate_depth": (reference.get("candidate_depth"), result.get("candidate_depth")),
        "query_count": (reference.get("query_count"), result.get("query_count")),
        "corpus_count": (reference.get("corpus_count"), result.get("corpus_count")),
        "qrels_split": (ref_splits.get("qrels"), current_splits.get("qrels")),
        "dataset_revision": (ref_dataset.get("revision"), current_dataset.get("revision")),
        "model_revision": (ref_revision, current_revision),
        "evaluator_package": (ref_evaluator, current_evaluator),
    }
    mismatches = {
        key: {"reference": pair[0], "current": pair[1]}
        for key, pair in comparisons.items()
        if pair[0] != pair[1]
    }
    tolerance = 1e-5
    score_delta = (
        abs(current_score - reference_score)
        if current_score is not None and reference_score is not None
        else None
    )
    passed = not mismatches and score_delta is not None and score_delta <= tolerance
    return {
        "status": "passed" if passed else "failed",
        "reference_source": "artifacts/e5_hyde_rerank/result.json captured before the run",
        "reference_ndcg_at_10": reference_score,
        "current_ndcg_at_10": current_score,
        "absolute_score_delta": score_delta,
        "tolerance": tolerance,
        "control_mismatches": mismatches,
    }


def _acceptance_checks(
    result: dict[str, object] | None,
    *,
    baseline_reference: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    if result is None or result.get("run_mode") != "benchmark":
        return []
    scores = _result_rows(result)
    checks: list[dict[str, object]] = []
    if "e5" in scores:
        checks.append(
            {
                "name": "e5_exceeds_paper_reference",
                "passed": scores["e5"] > 0.3259,
                "observed": scores["e5"],
                "reference": 0.3259,
                "absolute_delta": scores["e5"] - 0.3259,
            }
        )
    if {"e5", "e5_hyde"} <= set(scores):
        checks.append(
            {"name": "e5_hyde_exceeds_e5", "passed": scores["e5_hyde"] > scores["e5"], "observed": scores["e5_hyde"] - scores["e5"]}
        )
    elif result.get("system_id") == "e5_hyde":
        diagnostics = result.get("component_diagnostics") or {}
        baseline_score = diagnostics.get("e5_baseline_ndcg_at_10") if isinstance(diagnostics, dict) else None
        baseline_label = "same-run E5 diagnostic"
        if baseline_score is None and baseline_reference is not None:
            baseline_score = _result_rows(baseline_reference).get("e5")
            baseline_label = "fresh full E5 baseline artifact"
        if baseline_score is not None and "e5_hyde" in scores:
            checks.append(
                {
                    "name": "e5_hyde_exceeds_e5",
                    "passed": scores["e5_hyde"] > float(baseline_score),
                    "observed": scores["e5_hyde"] - float(baseline_score),
                    "baseline_source": baseline_label,
                }
            )
    if {"e5", "e5_rerank"} <= set(scores):
        checks.append(
            {"name": "e5_rerank_exceeds_e5", "passed": scores["e5_rerank"] > scores["e5"], "observed": scores["e5_rerank"] - scores["e5"]}
        )
    if {"e5_hyde_rerank", "e5_hyde", "e5_rerank"} <= set(scores):
        checks.extend(
            [
                {
                    "name": "combined_exceeds_e5_hyde",
                    "passed": scores["e5_hyde_rerank"] > scores["e5_hyde"],
                    "observed": scores["e5_hyde_rerank"] - scores["e5_hyde"],
                },
                {
                    "name": "combined_exceeds_e5_rerank",
                    "passed": scores["e5_hyde_rerank"] > scores["e5_rerank"],
                    "observed": scores["e5_hyde_rerank"] - scores["e5_rerank"],
                },
            ]
        )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "notebook",
        nargs="?",
        help="Notebook path, relative to the project root or absolute",
    )
    parser.add_argument(
        "--check-runtime",
        action="store_true",
        help="Check the pinned interpreter and dependencies without executing a notebook",
    )
    parser.add_argument("--output", help="Executed notebook filename (default: <name>_executed.ipynb)")
    parser.add_argument(
        "--output-dir",
        default="artifacts/notebook_runs",
        help="Output directory, relative to the project root unless absolute",
    )
    args = parser.parse_args()

    require_pinned_runtime()
    if args.check_runtime:
        print(f"Pinned runtime verified: Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        return 0
    if not args.notebook:
        parser.error("notebook is required unless --check-runtime is set")
    notebook = _project_path(args.notebook, must_exist=True)
    output_root = _project_path(args.output_dir, must_exist=False)
    output_name = args.output or f"{notebook.stem}_executed.ipynb"
    if Path(output_name).name != output_name or not output_name.lower().endswith(".ipynb"):
        raise SystemExit("--output must be a plain .ipynb filename")

    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')}_{uuid.uuid4().hex[:8]}"
    output_dir = output_root / "runs" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = _utc_now()
    started_timestamp = time.time()
    notebook_sha256 = _sha256_file(notebook)
    log_path = output_dir / "execution.log"
    trace_path = output_dir / "execution_trace.jsonl"
    kernel_trace_path = output_dir / "kernel_events.jsonl"
    metadata_path = output_dir / "execution.json"
    trace_events: list[dict[str, object]] = [
        {
            "event": "notebook_execution_started",
            "timestamp_utc": started_at,
            "run_id": run_id,
            "notebook": str(notebook.relative_to(PROJECT_ROOT)),
            "notebook_sha256": notebook_sha256,
        }
    ]
    reference_path = PROJECT_ROOT / "artifacts" / "e5_hyde_rerank" / "result.json"
    try:
        reference_result = json.loads(reference_path.read_text(encoding="utf-8")) if reference_path.is_file() else None
    except (OSError, ValueError, json.JSONDecodeError):
        reference_result = None

    source_notebook = nbformat.read(notebook, as_version=4)
    injected_notebook = copy.deepcopy(source_notebook)
    bootstrap = nbformat.v4.new_code_cell(source=_trace_bootstrap(_cell_contexts(source_notebook)))
    bootstrap.id = "codex_trace_bootstrap"
    bootstrap.metadata = {"tags": ["codex-trace-bootstrap"], "jupyter": {"source_hidden": True}}
    injected_notebook.cells.insert(0, bootstrap)

    with tempfile.TemporaryDirectory(prefix="code-retrieval-kernel-") as temporary_dir:
        kernel_root = Path(temporary_dir)
        kernel_name = "code-retrieval-pinned"
        kernel_dir = kernel_root / "kernels" / kernel_name
        kernel_dir.mkdir(parents=True)
        kernel_spec = {
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Python (code-retrieval pinned runtime)",
            "language": "python",
            "metadata": {"debugger": True},
        }
        (kernel_dir / "kernel.json").write_text(
            json.dumps(kernel_spec, indent=2) + "\n", encoding="utf-8"
        )

        environment = os.environ.copy()
        existing_jupyter_path = environment.get("JUPYTER_PATH", "")
        environment["JUPYTER_PATH"] = os.pathsep.join(
            part for part in (str(kernel_root), existing_jupyter_path) if part
        )
        environment["PYTHONNOUSERSITE"] = "1"
        environment["CODE_RETRIEVAL_KERNEL_TRACE_PATH"] = str(kernel_trace_path)
        temporary_notebook_handle = tempfile.NamedTemporaryFile(
            prefix=".codex-trace-", suffix=".ipynb", dir=PROJECT_ROOT, delete=False
        )
        instrumented_notebook = Path(temporary_notebook_handle.name)
        temporary_notebook_handle.close()
        nbformat.write(injected_notebook, instrumented_notebook)
        command = [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            str(instrumented_notebook),
            "--output",
            output_name,
            "--output-dir",
            str(output_dir),
            "--ExecutePreprocessor.timeout=0",
            f"--ExecutePreprocessor.kernel_name={kernel_name}",
        ]
        process_started_event = {
            "event": "notebook_process_started",
            "timestamp_utc": _utc_now(),
            "command": command,
            "run_output_directory": str(output_dir),
        }
        trace_events.append(process_started_event)
        try:
            notebook_exit = _run_streaming(
                command,
                cwd=PROJECT_ROOT,
                environment=environment,
                log_path=log_path,
                append=False,
                trace_events=trace_events,
                label="notebook_process",
            )
        finally:
            instrumented_notebook.unlink(missing_ok=True)

    executed_path = output_dir / output_name
    output_by_id: dict[str, list[dict[str, object]]] = {}
    executed_cells: list[nbformat.NotebookNode] = []
    if executed_path.is_file():
        executed_notebook = nbformat.read(executed_path, as_version=4)
        for cell in executed_notebook.cells:
            if cell.get("id") == "codex_trace_bootstrap":
                continue
            executed_cells.append(cell)
            if cell.cell_type == "code":
                output_by_id[str(cell.get("id", ""))] = _cell_output_text(cell, environment)
        executed_notebook.cells = [
            cell for cell in executed_notebook.cells if cell.get("id") != "codex_trace_bootstrap"
        ]
        nbformat.write(executed_notebook, executed_path)

    kernel_events: list[dict[str, object]] = []
    if kernel_trace_path.is_file():
        for line in kernel_trace_path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") == "cell_completed":
                event["outputs"] = output_by_id.get(str(event.get("cell_id", "")), [])
            kernel_events.append(event)

    for event in kernel_events:
        if event.get("error_message") is not None:
            event["error_message"] = _redact(str(event["error_message"]), environment)
        trace_events.append(event)

    process_completion_events = [
        event for event in trace_events if event.get("event") == "notebook_process_completed"
    ]
    process_completion = process_completion_events[-1] if process_completion_events else {}
    kernel_events = _complete_missing_cell_events(
        kernel_events,
        process_finished_at=str(process_completion.get("timestamp_utc", _utc_now())),
        process_exit=int(process_completion.get("exit_code", notebook_exit)),
    )
    trace_events = [trace_events[0], process_started_event, *kernel_events, *process_completion_events]
    context_by_id = {str(context["cell_id"]): context for context in _cell_contexts(source_notebook)}
    for cell_index, cell in enumerate(executed_cells):
        if cell.cell_type != "code" or cell.get("execution_count") is None:
            continue
        outputs = output_by_id.get(str(cell.get("id", "")), [])
        traced = context_by_id.get(str(cell.get("id", "")), {})
        with log_path.open("a", encoding="utf-8", newline="") as log_file:
            log_file.write(
                f"\n=== Cell {cell_index} / execution {cell.execution_count}: {traced.get('section', '')} ===\n"
            )
            for output in outputs:
                text_values = [
                    str(output.get(key, ""))
                    for key in ("text", "evalue")
                    if output.get(key)
                ]
                if isinstance(output.get("data"), dict):
                    text_values.extend(str(value) for value in output["data"].values())
                if isinstance(output.get("traceback"), list):
                    text_values.extend(str(value) for value in output["traceback"])
                for value in text_values:
                    log_file.write(_redact(value, environment))
                    if not value.endswith("\n"):
                        log_file.write("\n")

    result_path, result = _load_fresh_result(notebook, started_timestamp)
    result_summary: dict[str, object] | None = None
    if result is not None and result_path is not None:
        result_summary = {
            "path": str(result_path.relative_to(PROJECT_ROOT)),
            "sha256": _sha256_file(result_path),
            "cache_identity": result.get("cache_identity"),
            "status": result.get("status"),
            "run_mode": result.get("run_mode"),
            "query_count": result.get("query_count"),
            "corpus_count": result.get("corpus_count"),
            "candidate_depth": _result_candidate_depth(result),
            "evaluator": _result_evaluator(result),
            "scores": _result_rows(result),
            "dataset": result.get("dataset"),
            "model": result.get("model"),
            "fusion": result.get("fusion"),
            "timings_seconds": result.get("timings_seconds"),
        }

    parity = _baseline_parity(notebook, result, reference_result)
    current_baseline_path = RESULT_ARTIFACTS["e5_baseline_full_experiment"]
    try:
        current_baseline = (
            json.loads(current_baseline_path.read_text(encoding="utf-8"))
            if current_baseline_path.is_file()
            else None
        )
    except (OSError, ValueError, json.JSONDecodeError):
        current_baseline = None
    acceptance_checks = _acceptance_checks(result, baseline_reference=current_baseline)
    analysis_exit: int | None = None
    analysis_summary: dict[str, object] | None = None
    if notebook_exit == 0 and result_path is not None and result is not None and result.get("status") == "benchmark":
        analysis_directory = output_dir / "analysis"
        analysis_command = [
            sys.executable,
            str(ANALYSIS_EXPORTER),
            "--result",
            str(result_path),
            "--output-dir",
            str(analysis_directory),
        ]
        trace_events.append({"event": "analysis_export_started", "timestamp_utc": _utc_now()})
        analysis_exit = _run_streaming(
            analysis_command,
            cwd=PROJECT_ROOT,
            environment=environment,
            log_path=log_path,
            append=True,
            trace_events=trace_events,
            label="analysis_export",
        )
        if analysis_exit == 0:
            analysis_summary = {
                "directory": str(analysis_directory),
                "metadata": str(analysis_directory / "analysis_metadata.json"),
                "per_query_metrics": str(analysis_directory / "per_query_metrics.csv"),
                "qualitative_cases": str(analysis_directory / "qualitative_cases.jsonl"),
            }

    execution_codes = {"notebook": notebook_exit, "analysis_export": analysis_exit}
    parity_passed = parity is None or parity.get("status") == "passed"
    score_checks_passed = all(bool(check.get("passed")) for check in acceptance_checks)
    completed_successfully = (
        notebook_exit == 0
        and result is not None
        and (analysis_exit in (None, 0))
        and parity_passed
        and score_checks_passed
    )
    finished_at = _utc_now()
    trace_events.append(
        {
            "event": "notebook_execution_completed",
            "timestamp_utc": finished_at,
            "run_id": run_id,
            "status": "completed" if completed_successfully else "failed_or_blocked",
            "exit_codes": execution_codes,
            "result": result_summary,
            "baseline_parity": parity,
            "acceptance_checks": acceptance_checks,
            "analysis_artifacts": analysis_summary,
        }
    )
    _write_jsonl(trace_path, trace_events)

    git_commit = None
    git_worktree_dirty = None
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
        git_worktree_dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        pass
    safe_controls = {
        key: value
        for key, value in os.environ.items()
        if key.startswith(("E5_BASELINE_", "E5_HYDE_", "E5_RERANK_"))
        and not SECRET_VALUE_KEYS.search(key)
    }
    metadata = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "completed" if completed_successfully else "failed_or_blocked",
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "elapsed_seconds": time.time() - started_timestamp,
        "notebook": str(notebook.relative_to(PROJECT_ROOT)),
        "notebook_sha256": notebook_sha256,
        "executed_notebook": str(executed_path),
        "executed_notebook_sha256": _sha256_file(executed_path) if executed_path.is_file() else None,
        "runtime": {
            "python": ".".join(map(str, sys.version_info[:3])),
            "platform": sys.platform,
            "pinned_runtime_check": "passed",
        },
        "git_commit": git_commit,
        "git_worktree_dirty": git_worktree_dirty,
        "source_code_sha256": {
            str(path.relative_to(PROJECT_ROOT)): _sha256_file(path)
            for path in (PROJECT_ROOT / "scripts" / "run_benchmark_notebook.py", ANALYSIS_EXPORTER)
            if path.is_file()
        },
        "execution_controls": safe_controls,
        "result": result_summary,
        "baseline_parity": parity,
        "acceptance_checks": acceptance_checks,
        "analysis_artifacts": analysis_summary,
        "exit_codes": execution_codes,
        "artifacts": {
            "executed_notebook": executed_path.name if executed_path.is_file() else None,
            "execution_log": log_path.name,
            "execution_trace": trace_path.name,
            "analysis_directory": "analysis" if analysis_summary else None,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if kernel_trace_path.exists():
        kernel_trace_path.unlink()
    print(f"Run directory: {output_dir}")
    print(f"Execution log: {log_path}")
    print(f"Execution trace: {trace_path}")
    if analysis_summary:
        print(f"Analysis artifacts: {analysis_summary}")
    return 0 if completed_successfully else (notebook_exit or analysis_exit or 1)


if __name__ == "__main__":
    raise SystemExit(main())
