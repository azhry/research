"""Notebook-integrated queue for the unchanged PyABSA reference scripts."""

from __future__ import annotations

import json
import hashlib
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from aste_repro.report import (
    _parsed_hoasa_rows,
    _verify_isolated_reference_attempt,
    _verify_reference_artifacts,
    build_report,
)


PROTOCOL_ID = "pyabsa-2.4.3-emcgcn-reference-max-test-v2-seed-cache-compatible"
CACHE_PROCESS_PROTOCOL_ID = "pyabsa-2.4.3-emcgcn-reference-max-test-v3-fresh-seed-process-cache-hit"
ENCODER_ORDER = ("mbert", "indobert", "xlmr", "deberta_absa")
ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
QUEUE_NAME = "hoasa-emcgcn-reference-baseline-queue.status.json"
BASELINE_QUEUE_NAME = "hoasa-emcgcn-baseline-queue.status.json"
WORKER_TOKEN = "reference_baseline.py"
QUEUE_TOKEN = "ipykernel_launcher"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finalize_artifact_hashes(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "complete":
        return manifest
    # Hash after the Python process exits: interpreter shutdown can append
    # warnings to the redirected training log after the runner returns.
    manifest["artifact_sha256"] = {
        path.relative_to(run_dir).as_posix(): _sha256(path)
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path != manifest_path
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def _process_identity(pid: int) -> dict[str, Any] | None:
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except OSError:
            return None
        return {"pid": pid, "name": None, "command_line": None, "created_at": None}

    powershell = shutil.which("pwsh") or shutil.which("powershell")
    if powershell is None:
        raise RuntimeError("PowerShell is required to verify Windows process identity; refusing to resume.")
    query = (
        f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId = {int(pid)}'; "
        "if ($null -ne $p) { "
        "$c=$null; if ($null -ne $p.CreationDate) { $c=$p.CreationDate.ToUniversalTime().ToString('o') }; "
        "[pscustomobject]@{pid=[int]$p.ProcessId; name=[string]$p.Name; "
        "command_line=[string]$p.CommandLine; created_at=$c} | ConvertTo-Json -Compress }"
    )
    try:
        result = subprocess.run(
            [powershell, "-NoProfile", "-NonInteractive", "-Command", query],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"could not verify process identity for PID {pid}: {error}") from error
    if result.returncode != 0:
        raise RuntimeError(f"PowerShell process identity check failed for PID {pid}: {result.stderr[-500:]}")
    output = result.stdout.strip()
    if not output:
        return None
    try:
        record = json.loads(output)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid process identity response for PID {pid}") from error
    if int(record.get("pid", -1)) != int(pid):
        raise RuntimeError(f"process identity response returned the wrong PID for {pid}")
    return record


def _process_matches(pid_value: Any, expected_token: str, started_at: str | None) -> tuple[bool, dict[str, Any] | None]:
    try:
        pid = int(pid_value)
    except (TypeError, ValueError):
        return False, None
    process = _process_identity(pid)
    if process is None:
        return False, None
    command_line = process.get("command_line")
    if os.name == "nt" and expected_token.casefold() not in str(command_line or "").casefold():
        return False, process
    if started_at and process.get("created_at"):
        try:
            process_created = datetime.fromisoformat(process["created_at"].replace("Z", "+00:00"))
            queue_started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        except ValueError:
            # An unreadable start time must not turn a possibly live process into
            # permission to launch an overlapping queue.
            return True, process
        if process_created < queue_started - timedelta(seconds=5):
            return False, process
    return True, process


def _process_evidence(process: dict[str, Any] | None, expected_token: str) -> dict[str, Any] | None:
    if process is None:
        return None
    command_line = str(process.get("command_line") or "")
    return {
        "pid": process.get("pid"),
        "name": process.get("name"),
        "expected_command_match": expected_token.casefold() in command_line.casefold(),
        "created_at": process.get("created_at"),
    }


def _atomic_manifest(manifest_path: Path, manifest: dict[str, Any]) -> None:
    _atomic_json(manifest_path, manifest)


def _latest_training_progress(log_path: Path) -> str | None:
    if not log_path.is_file():
        return None
    progress = re.compile(r"Epoch:\s*\d+\s*\|\s*Smooth Loss:\s*[0-9.eE+-]+:.*?\d+%.*")
    last = None
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                match = progress.search(line)
                if match:
                    # Restrict saved text to epoch/loss/progress diagnostics;
                    # dataset sentences or source text never enter queue state.
                    last = match.group(0).strip()[-240:]
    except OSError:
        return None
    return last


def _interrupt_stale_attempt(
    manifest_path: Path,
    queue_record: dict[str, Any] | None,
    reason: str,
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "running":
        return manifest
    pid = manifest.get("runner_pid")
    matches, process = _process_matches(pid, WORKER_TOKEN, manifest.get("created_at"))
    if matches:
        raise RuntimeError(
            f"source worker PID {pid} still matches {WORKER_TOKEN}; preserve it and do not launch another run"
        )
    run_dir = manifest_path.parent
    log_path = run_dir / "training.log"
    queue_process = None
    if queue_record and str(queue_record.get("queue_pid", "")).isdigit():
        queue_process = _process_identity(int(queue_record["queue_pid"]))
    manifest.update(
        {
            "status": "interrupted",
            "completed_at": _now(),
            "failure_type": "UnexpectedWorkerExit",
            "failure": reason,
            "interruption_evidence": {
                "queue_pid": queue_record.get("queue_pid") if queue_record else None,
                "queue_process": _process_evidence(queue_process, QUEUE_TOKEN),
                "worker_pid": queue_record.get("worker_pid") if queue_record else None,
                "runner_pid": pid,
                "runner_process": _process_evidence(process, WORKER_TOKEN),
                "training_log_bytes": log_path.stat().st_size if log_path.is_file() else None,
                "training_log_last_write": datetime.fromtimestamp(
                    log_path.stat().st_mtime, timezone.utc
                ).isoformat() if log_path.is_file() else None,
                "last_training_progress": _latest_training_progress(log_path),
                "partial_artifacts_preserved": True,
            },
        }
    )
    _atomic_manifest(manifest_path, manifest)
    return manifest


def _reference_manifests(
    runs_dir: Path, protocol_id: str, encoder: str | None = None
) -> list[tuple[Path, dict[str, Any]]]:
    rows = []
    for path in sorted(runs_dir.glob("hoasa-emcgcn-reference-*-attempt-*/manifest.json")):
        try:
            manifest = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("experiment_protocol_id") != protocol_id:
            continue
        if encoder is not None and manifest.get("encoder") != encoder:
            continue
        rows.append((path, manifest))
    return rows


def _write_reference_queue(
    marker: Path,
    status: str,
    queue_pid: int,
    started_at: str,
    scope: str,
    **extra: Any,
) -> dict[str, Any]:
    record = {
        "status": status,
        "queue_pid": queue_pid,
        "queue_process_identity": QUEUE_TOKEN,
        "started_at": started_at,
        "updated_at": _now(),
        "scope": scope,
        "experiment_protocol_id": PROTOCOL_ID,
        **extra,
    }
    _atomic_json(marker, record)
    return record


def _recover_old_reference_queue(marker: Path, runs_dir: Path) -> dict[str, Any] | None:
    if not marker.is_file():
        return None
    old = _read_json(marker)
    if old.get("status") != "running":
        return {
            key: old[key]
            for key in ("status", "queue_pid", "worker_pid", "started_at", "completed_at", "failure_type", "failure")
            if key in old
        }

    queue_pid = old.get("queue_pid")
    queue_live, queue_process = _process_matches(queue_pid, QUEUE_TOKEN, old.get("started_at"))
    worker_pid = old.get("worker_pid")
    worker_live, worker_process = _process_matches(
        worker_pid, WORKER_TOKEN, old.get("worker_started_at") or old.get("started_at")
    )
    own_queue = str(queue_pid) == str(os.getpid())
    if worker_live or (queue_live and not own_queue):
        active_pid = worker_pid if worker_live else queue_pid
        raise RuntimeError(
            f"reference queue still has a live process (PID {active_pid}); do not start a duplicate queue"
        )

    reason = (
        f"Recovered stale running queue: queue PID {queue_pid} no longer identifies its notebook kernel; "
        f"worker PID {worker_pid} no longer identifies {WORKER_TOKEN}. Partial run files were preserved."
    )
    stale_attempts = []
    for manifest_path, manifest in _reference_manifests(runs_dir, PROTOCOL_ID):
        if manifest.get("status") != "running":
            continue
        recovered = _interrupt_stale_attempt(manifest_path, old, reason)
        stale_attempts.append(recovered.get("run_id", manifest_path.parent.name))

    old.update(
        {
            "status": "interrupted",
            "completed_at": _now(),
            "failure_type": "StaleQueueRecovery",
            "failure": reason,
            "recovered_attempts": stale_attempts,
            "queue_process_observed": _process_evidence(queue_process, QUEUE_TOKEN),
            "worker_process_observed": _process_evidence(worker_process, WORKER_TOKEN),
            "partial_artifacts_preserved": True,
        }
    )
    _atomic_json(marker, old)
    return {
        key: old[key]
        for key in ("status", "queue_pid", "worker_pid", "started_at", "completed_at", "failure_type", "failure")
        if key in old
    }


def _assert_no_active_queue(current_process_id: int) -> None:
    for marker_path in (RUNS / BASELINE_QUEUE_NAME, RUNS / QUEUE_NAME):
        if not marker_path.is_file():
            continue
        marker = _read_json(marker_path)
        if marker.get("status") != "running":
            continue
        queue_pid = marker.get("queue_pid")
        queue_live, _queue_process = _process_matches(queue_pid, QUEUE_TOKEN, marker.get("started_at"))
        worker_live = False
        worker_pid = marker.get("worker_pid")
        if worker_pid is not None:
            worker_live, _worker_process = _process_matches(
                worker_pid, WORKER_TOKEN, marker.get("worker_started_at") or marker.get("started_at")
            )
        if worker_live or (queue_live and int(queue_pid) != int(current_process_id)):
            active_pid = worker_pid if worker_live else queue_pid
            raise RuntimeError(
                f"an ASTE benchmark queue is active (PID {active_pid}; scope: {marker.get('scope')}); "
                "preserve its files and do not launch an overlapping queue"
            )


def _next_attempt_dir(runs_dir: Path, encoder: str) -> Path:
    prefix = f"hoasa-emcgcn-reference-{encoder}-attempt-"
    numbers = []
    for path in runs_dir.glob(prefix + "*"):
        suffix = path.name.removeprefix(prefix)
        if suffix.isdigit():
            numbers.append(int(suffix))
    attempt = max(numbers, default=0) + 1
    return runs_dir / f"{prefix}{attempt:03d}"


def _refresh_report(report_path: Path, matrix_path: Path, runs_dir: Path) -> None:
    rendered = build_report(runs_dir=runs_dir, matrix_path=matrix_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(rendered, encoding="utf-8")
    print(f"Refreshed aggregate report: {report_path}", flush=True)


def _runtime_instructor_source(python_exe: Path) -> Path:
    """Locate the exact instructor source in the isolated PyABSA runtime."""
    query = (
        "import pathlib, sysconfig; "
        "print(pathlib.Path(sysconfig.get_paths()['purelib']) / "
        "'pyabsa/tasks/AspectSentimentTripletExtraction/instructor/instructor.py')"
    )
    env = os.environ.copy()
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})
    result = subprocess.run(
        [str(python_exe), "-c", query],
        cwd=python_exe.parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"could not locate installed PyABSA instructor source: {result.stderr[-1000:]}")
    source = Path(result.stdout.strip().splitlines()[-1]).resolve() if result.stdout.strip() else None
    if source is None or not source.is_file():
        raise RuntimeError("isolated PyABSA runtime returned no instructor source path")
    return source


def _validated_legacy_log_step_record(
    run_dir: Path,
    manifest: dict[str, Any],
    protocol: dict[str, Any],
    python_exe: Path,
    runtime_source: Path,
) -> tuple[int, list[float], dict[str, Any]] | None:
    """Accept only complete three-seed runs rejected by the old -1 sentinel check."""
    failure = str(manifest.get("failure") or "")
    match = re.fullmatch(r"source config mismatch for log_step: expected -1, observed (\d+)", failure)
    if manifest.get("status") != "failed" or match is None:
        return None
    encoder = manifest.get("encoder")
    spec = protocol.get("source_scripts", {}).get(encoder)
    if not spec or spec.get("settings", {}).get("log_step") != -1:
        return None
    resolved_step = int(match.group(1))
    manifest_path = run_dir / "manifest.json"
    log_path = run_dir / "training.log"
    if not log_path.is_file() or not manifest_path.is_file():
        return None
    if manifest.get("failure_type") != "RuntimeError" or manifest.get("runner_pid") is None:
        return None

    reference = manifest.get("reference", {})
    source_copy = run_dir / reference.get("source_script_copy", "")
    source_script = Path(reference.get("script_path", ""))
    if (
        reference.get("script") != spec.get("path")
        or reference.get("script_sha256") != spec.get("sha256")
        or reference.get("source_script_sha256") != spec.get("sha256")
        or not source_copy.is_file()
        or _sha256(source_copy) != spec.get("sha256")
        or not source_script.is_file()
        or _sha256(source_script) != spec.get("sha256")
    ):
        return None
    if manifest.get("requested_config") != spec.get("settings"):
        return None
    if manifest.get("model", {}).get("repository") != spec.get("model_id"):
        return None
    if manifest.get("model", {}).get("revision") != spec.get("model_revision"):
        return None

    # Require the exact original workspace files and keep byte-for-byte copies
    # before correcting this historical validation classification.
    code_hashes = manifest.get("code", {}).get("file_sha256", {})
    if not code_hashes:
        return None
    for relative, expected_hash in code_hashes.items():
        source = ROOT / relative
        source_snapshot = run_dir / "provenance" / "workspace-source" / relative
        if source_snapshot.is_file():
            if _sha256(source_snapshot) != expected_hash:
                return None
        elif not source.is_file() or _sha256(source) != expected_hash:
            return None

    source_files = manifest.get("source_files", {})
    if set(source_files) != {"train", "dev", "test"}:
        return None
    for split, record in source_files.items():
        run_copy = run_dir / record.get("run_copy_path", "")
        reference_path = Path(record.get("reference_path", ""))
        canonical_path = Path(record.get("canonical_path", ""))
        if not run_copy.is_file() or not reference_path.is_file() or not canonical_path.is_file():
            return None
        expected_hash = record.get("run_copy_sha256")
        if (
            _sha256(run_copy) != expected_hash
            or _sha256(reference_path) != record.get("reference_sha256")
            or record.get("reference_sha256") != expected_hash
            or _sha256(canonical_path) != record.get("canonical_sha256")
            or record.get("parsed_lines_equal") is not True
            or _parsed_hoasa_rows(run_copy) != _parsed_hoasa_rows(canonical_path)
        ):
            return None

    cache = reference.get("model_cache") or {}
    if cache.get("revision_directory") != spec.get("model_revision") or not cache.get("small_file_sha256"):
        return None
    cache_snapshot = Path(cache.get("snapshot_path", ""))
    if not cache_snapshot.is_dir():
        return None
    for relative, expected_hash in cache["small_file_sha256"].items():
        cached = cache_snapshot / relative
        if not cached.is_file() or _sha256(cached) != expected_hash:
            return None

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    logged_steps = [int(value) for value in re.findall(r"\blog_step:(-?\d+)\s+-->\s+Calling Count", log_text)]
    logged_seeds = [int(value) for value in re.findall(r"\bseed:(\d+)\s+-->\s+Calling Count", log_text)]
    logged_num_steps = [int(value) for value in re.findall(r"\bNum steps\s*=\s*(\d+)", log_text)]
    epochs = int(spec["settings"]["num_epoch"])
    if logged_steps != [-1, resolved_step, resolved_step] or logged_seeds != [1, 2, 3]:
        return None
    if logged_num_steps != [resolved_step * epochs] * 3:
        return None
    if len(re.findall(r"Loading best model:", log_text)) != 3:
        return None
    checkpoint_root = run_dir / "checkpoints"
    if not checkpoint_root.is_dir() or len([item for item in checkpoint_root.iterdir() if item.is_dir()]) != 3:
        return None
    metric_lines = [line for line in log_text.splitlines() if "Max-Test-F1" in line]
    if not metric_lines:
        return None
    score_lists = re.findall(r"Max-Test-F1[^\n]*?\[([^\]]+)\]", metric_lines[-1])
    if not score_lists:
        return None
    try:
        scores = [float(value.strip()) for value in score_lists[-1].split(",")]
    except ValueError:
        return None
    if len(scores) != 3 or any(not math.isfinite(score) or not 0 <= score <= 100 for score in scores):
        return None

    runtime_text = runtime_source.read_text(encoding="utf-8")
    if not re.search(
        r"if\s+self\.config\.log_step\s*<\s*0\s*:\s*"
        r"self\.config\.log_step\s*=\s*\(\s*"
        r"len\(self\.train_dataloaders\[0\]\)",
        runtime_text,
        re.DOTALL,
    ):
        return None
    return resolved_step, scores, {
        "requested": -1,
        "resolved": resolved_step,
        "resolution_rule": "PyABSA 2.4.3 replaces negative log_step with len(train_dataloaders[0])",
        "runtime_source_path": str(runtime_source),
        "runtime_source_sha256": _sha256(runtime_source),
        "runtime_source_snapshot": "provenance/runtime/pyabsa_instructor.py",
        "training_log_path": "training.log",
        "logged_seed_order": logged_seeds,
        "logged_log_step_values": logged_steps,
        "logged_num_steps": logged_num_steps,
        "num_epoch": epochs,
        "evidence_verified": True,
    }


def _reconcile_false_log_step_failures(
    runs_dir: Path, protocol: dict[str, Any], python_exe: Path
) -> list[str]:
    """Preserve and correct only old failures proved to be sentinel-validation errors."""
    runtime_source = _runtime_instructor_source(python_exe)
    corrected: list[str] = []
    for manifest_path, manifest in _reference_manifests(runs_dir, PROTOCOL_ID):
        if manifest.get("validation_correction") is not None:
            continue
        evidence = _validated_legacy_log_step_record(
            manifest_path.parent, manifest, protocol, python_exe, runtime_source
        )
        if evidence is None:
            continue
        run_dir = manifest_path.parent
        backup = run_dir / "manifest.pre-validation-correction.json"
        if backup.exists():
            raise RuntimeError(f"refusing to overwrite preserved pre-correction manifest in {run_dir}")
        prior_bytes = manifest_path.read_bytes()
        prior_hash = hashlib.sha256(prior_bytes).hexdigest()
        backup.write_bytes(prior_bytes)

        runtime_snapshot = run_dir / evidence[2]["runtime_source_snapshot"]
        runtime_snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(runtime_source, runtime_snapshot)
        if _sha256(runtime_snapshot) != evidence[2]["runtime_source_sha256"]:
            raise RuntimeError(f"runtime source snapshot checksum mismatch in {run_dir}")

        code_root = run_dir / "provenance" / "workspace-source"
        for relative, expected_hash in manifest["code"]["file_sha256"].items():
            source = ROOT / relative
            destination = code_root / Path(relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.is_file():
                if not source.is_file() or _sha256(source) != expected_hash:
                    raise RuntimeError(f"workspace source changed during legacy reconciliation: {relative}")
                shutil.copy2(source, destination)
            if _sha256(destination) != expected_hash:
                raise RuntimeError(f"workspace source snapshot checksum mismatch for {relative} in {run_dir}")
        manifest["code"]["snapshot_directory"] = "provenance/workspace-source"

        resolved_step, scores, resolution = evidence
        spec = protocol["source_scripts"][manifest["encoder"]]
        actual = dict(spec["settings"])
        actual["model"] = "emcgcn"
        actual["log_step"] = resolved_step
        actual["log_step_resolution"] = resolution
        manifest["code"]["snapshot_directory"] = "provenance/workspace-source"
        manifest["resolved_config"]["actual_settings"] = actual
        manifest["reference"]["runtime_log_step_resolution"] = resolution
        manifest["environment"] = {
            "python": manifest["resolved_config"].get("python"),
            "packages": manifest["resolved_config"].get("packages", {}),
            "working_directory": str(run_dir.resolve()),
        }
        manifest["seed_metrics"] = [
            {"seed": seed, "max_test_f1_pct": score}
            for seed, score in zip((1, 2, 3), scores)
        ]
        manifest["validation_correction"] = {
            "schema_version": 1,
            "applied_at": _now(),
            "original_status": "failed",
            "original_failure": manifest.get("failure"),
            "original_manifest_backup": backup.name,
            "original_manifest_sha256": prior_hash,
            "reason": "The prior validator compared PyABSA's runtime-normalized log_step sentinel to the source setting -1 as if it were immutable. All three declared source-script seed outcomes and runtime/source/data/model evidence were verified; no training setting or score was changed.",
            "seed_outcomes_verified": 3,
            "source_script_unchanged": True,
            "configuration_unchanged": True,
            "runtime_normalization": resolution,
        }
        manifest["status"] = "complete"
        manifest["completed_at"] = _now()
        manifest.pop("failure_type", None)
        manifest.pop("failure", None)
        metrics = {
            "metric": "PyABSA MetricVisualizer Max-Test-F1",
            "unit": "percent",
            "selection_boundary": "test set used during training for checkpoint selection; test-leaked reference only",
            "seeds": manifest["seed_metrics"],
        }
        _atomic_json(run_dir / "reference_metrics.json", metrics)
        manifest["artifact_sha256"] = {
            path.relative_to(run_dir).as_posix(): _sha256(path)
            for path in sorted(run_dir.rglob("*"))
            if path.is_file() and path != manifest_path
        }
        _atomic_json(manifest_path, manifest)
        _verify_reference_artifacts({"path": manifest_path, "manifest": manifest}, protocol)
        corrected.append(run_dir.name)
        print(
            f"Corrected a proven PyABSA log_step sentinel validation false failure for {manifest['encoder']}; "
            f"preserved the prior manifest and verified seeds 1, 2, 3.",
            flush=True,
        )
    return corrected


def run_missing_reference_baselines(
    *,
    project: Path,
    repo_root: Path,
    runs_dir: Path,
    report_path: Path,
    matrix_path: Path,
    encoders: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    """Run selected unchanged source scripts, preserving prior attempts and run identities."""
    project = project.resolve()
    repo_root = repo_root.resolve()
    runs_dir = runs_dir.resolve()
    matrix_path = matrix_path.resolve()
    report_path = report_path.resolve()
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    protocol = matrix.get("reference_baseline_protocol", {})
    if protocol.get("id") != PROTOCOL_ID:
        raise RuntimeError("matrix does not declare the expected source-script reference protocol")
    if list(protocol.get("seeds", [])) != [1, 2, 3]:
        raise RuntimeError("source-script protocol seed declaration changed")
    if set(protocol.get("source_scripts", {})) != set(matrix.get("encoders", {})):
        raise RuntimeError("source-script protocol must cover all four matrix encoders")
    selected_encoders = tuple(ENCODER_ORDER if encoders is None else encoders)
    if not selected_encoders or len(set(selected_encoders)) != len(selected_encoders):
        raise ValueError("encoder selection must be non-empty and contain no duplicates")
    if not set(selected_encoders).issubset(ENCODER_ORDER):
        raise ValueError(f"unknown encoder in selection: {selected_encoders!r}")

    reference_repo = Path(os.environ.get("PYABSA_REFERENCE_REPO", protocol["source_repository"])).resolve()
    python_exe = Path(
        os.environ.get("PYABSA_PYTHON", str(reference_repo / ".venv" / "Scripts" / "python.exe"))
    ).resolve()
    runner = project / "src" / "aste_repro" / "reference_baseline.py"
    if not reference_repo.is_dir():
        raise FileNotFoundError(f"local PyABSA reference repository is unavailable: {reference_repo}")
    if not python_exe.is_file():
        raise FileNotFoundError(f"source PyABSA Python runtime is unavailable: {python_exe}")
    if not runner.is_file():
        raise FileNotFoundError(f"integrated source-script runner is missing: {runner}")

    marker = runs_dir / QUEUE_NAME
    _assert_no_active_queue(os.getpid())
    previous_queue = _recover_old_reference_queue(marker, runs_dir)
    corrected_attempts = _reconcile_false_log_step_failures(runs_dir, protocol, python_exe)
    if corrected_attempts and previous_queue is not None:
        previous_queue["reconciled_validation_attempts"] = corrected_attempts
    _refresh_report(report_path, matrix_path, runs_dir)
    started_at = _now()
    scope = (
        f"{PROTOCOL_ID}: unchanged source scripts; seeds [1, 2, 3] "
        f"for selected encoders {list(selected_encoders)}"
    )
    failures: list[dict[str, Any]] = []
    completed_encoders: list[str] = []
    skipped_encoders: list[str] = []
    current: dict[str, Any] = {"current_encoder": None, "worker_pid": None, "worker_started_at": None}

    def write(status: str, **extra: Any) -> dict[str, Any]:
        queue_fields = {**current, **extra}
        return _write_reference_queue(
            marker,
            status,
            os.getpid(),
            started_at,
            scope,
            recovered_queue=previous_queue,
            completed_encoders=list(completed_encoders),
            skipped_encoders=list(skipped_encoders),
            failures=list(failures),
            **queue_fields,
        )

    write("running", phase="starting")
    try:
        for encoder in selected_encoders:
            attempts = _reference_manifests(runs_dir, PROTOCOL_ID, encoder)
            complete = [(path, manifest) for path, manifest in attempts if manifest.get("status") == "complete"]
            if len(complete) > 1:
                raise RuntimeError(f"multiple complete source-script runs exist for {encoder}; refusing to pool runs")
            if complete:
                _verify_reference_artifacts(
                    {"path": complete[0][0], "manifest": complete[0][1]}, protocol
                )
                skipped_encoders.append(encoder)
                write(
                    "running",
                    phase="skipping_verified_complete",
                    current_encoder=encoder,
                    worker_pid=None,
                    worker_started_at=None,
                )
                print(f"Already complete and checksum-verified; skipping source script for {encoder}.", flush=True)
                continue

            # Change a running manifest only after checking both process command
            # identity and its creation time, so PID reuse is not treated as a
            # live or stale worker by PID alone.
            for manifest_path, manifest in attempts:
                if manifest.get("status") != "running":
                    continue
                _interrupt_stale_attempt(
                    manifest_path,
                    previous_queue,
                    "Recovered a stale running source-script attempt before resume; no score was finalized.",
                )

            run_dir = _next_attempt_dir(runs_dir, encoder)
            run_dir.mkdir(parents=True, exist_ok=False)
            log_path = run_dir / "training.log"
            current.update(
                {
                    "current_encoder": encoder,
                    "worker_pid": None,
                    "worker_started_at": None,
                    "attempt": run_dir.name,
                }
            )
            write("running", phase="launching_source_script")
            env = os.environ.copy()
            env.update(
                {
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "HF_DATASETS_OFFLINE": "1",
                    "MPLBACKEND": "Agg",
                    "PYTHONUNBUFFERED": "1",
                    "PYTHONIOENCODING": "utf-8",
                }
            )
            command = [
                str(python_exe),
                str(runner),
                "--encoder", encoder,
                "--run-dir", str(run_dir),
                "--reference-repo", str(reference_repo),
            ]
            print(
                f"Launching unchanged {protocol['source_scripts'][encoder]['path']} "
                f"for seeds {protocol['seeds']} into {run_dir.name}.",
                flush=True,
            )
            with log_path.open("wb") as log_stream:
                worker = subprocess.Popen(
                    command,
                    # The unchanged source scripts derive datasets/aste from
                    # os.getcwd(); run from the immutable attempt so their
                    # custom dataset is the verified local copy, not a
                    # different workspace path or a downloaded fallback.
                    cwd=run_dir,
                    env=env,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
                )
                current.update(
                    {
                        "worker_pid": worker.pid,
                        "worker_started_at": _now(),
                    }
                )
                write("running", phase="source_script_training")
                last_write = time.monotonic()
                while worker.poll() is None:
                    time.sleep(10)
                    if time.monotonic() - last_write >= 30:
                        write(
                            "running",
                            phase="source_script_training",
                            worker_log_bytes=log_path.stat().st_size if log_path.exists() else 0,
                            last_training_progress=_latest_training_progress(log_path),
                        )
                        last_write = time.monotonic()
                return_code = worker.wait()

            manifest_path = run_dir / "manifest.json"
            if not manifest_path.is_file():
                failures.append(
                    {"encoder": encoder, "attempt": run_dir.name, "error": "worker exited without writing a manifest"}
                )
                current.update({"worker_pid": None, "worker_started_at": None})
                write("running", phase="script_failed")
                _refresh_report(report_path, matrix_path, runs_dir)
                continue

            manifest = _finalize_artifact_hashes(run_dir)
            if manifest.get("status") == "running":
                matches, process = _process_matches(
                    manifest.get("runner_pid"), WORKER_TOKEN, manifest.get("created_at")
                )
                if matches:
                    raise RuntimeError(
                        f"source worker PID {manifest.get('runner_pid')} remains live after its parent wait; refusing duplicate recovery"
                    )
                manifest = _interrupt_stale_attempt(
                    manifest_path,
                    _read_json(marker),
                    f"source worker exited with status {return_code} before writing a complete manifest; no score was recorded.",
                )
                manifest["interruption_evidence"]["runner_process"] = _process_evidence(process, WORKER_TOKEN)
                _atomic_manifest(manifest_path, manifest)
            if return_code == 0 and manifest.get("status") == "complete":
                _verify_reference_artifacts({"path": manifest_path, "manifest": manifest}, protocol)
                completed_encoders.append(encoder)
                print(f"Completed and verified all three declared seeds for {encoder}.", flush=True)
            else:
                error = manifest.get("failure") or f"source worker exited with status {return_code}"
                failures.append(
                    {
                        "encoder": encoder,
                        "attempt": run_dir.name,
                        "status": manifest.get("status"),
                        "return_code": return_code,
                        "error": str(error)[:1000],
                    }
                )
                print(
                    f"Source script for {encoder} ended as {manifest.get('status')} "
                    f"(exit {return_code}); evidence preserved, continuing to the next encoder.",
                    flush=True,
                )
            current.update({"worker_pid": None, "worker_started_at": None, "current_encoder": encoder})
            write("running", phase="refreshing_report")
            _refresh_report(report_path, matrix_path, runs_dir)

        complete_count = sum(
            1
            for encoder in selected_encoders
            if any(
                manifest.get("status") == "complete"
                for _path, manifest in _reference_manifests(runs_dir, PROTOCOL_ID, encoder)
            )
        )
        status = "complete" if complete_count == len(selected_encoders) and not failures else "complete_with_failures"
        write(
            status,
            phase="finished",
            completed_at=_now(),
            complete_scripts=complete_count,
            selected_encoders=list(selected_encoders),
            expected_scripts=len(selected_encoders),
            seed_outcomes=complete_count * len(protocol["seeds"]),
            expected_seed_outcomes=len(selected_encoders) * len(protocol["seeds"]),
        )
        _refresh_report(report_path, matrix_path, runs_dir)
        print(
            f"Reference baseline queue {status}: {complete_count}/{len(selected_encoders)} scripts, "
            f"{complete_count * len(protocol['seeds'])}/"
            f"{len(selected_encoders) * len(protocol['seeds'])} seed outcomes; "
            f"{len(failures)} failed/interrupted attempt(s).",
            flush=True,
        )
        return _read_json(marker)
    except KeyboardInterrupt:
        write(
            "interrupted",
            phase="interrupted",
            completed_at=_now(),
            failure_type="KeyboardInterrupt",
            failure="notebook queue interrupted; partial run files and checkpoints are preserved",
        )
        raise


def run_cache_process_reference_baselines(
    *,
    project: Path,
    repo_root: Path,
    runs_dir: Path,
    report_path: Path,
    matrix_path: Path,
    encoders: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    """Reproduce cache-affected source cells with one fresh process per declared seed."""
    project = project.resolve()
    repo_root = repo_root.resolve()
    runs_dir = runs_dir.resolve()
    report_path = report_path.resolve()
    matrix_path = matrix_path.resolve()
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    source_protocol = matrix.get("reference_baseline_protocol", {})
    protocol = matrix.get("reference_baseline_cache_process_protocol", {})
    if source_protocol.get("id") != PROTOCOL_ID or protocol.get("id") != CACHE_PROCESS_PROTOCOL_ID:
        raise RuntimeError("matrix does not declare the cache-process reference protocol")
    declared_seeds = [int(value) for value in protocol.get("seeds", [])]
    if declared_seeds != [1, 2, 3]:
        raise RuntimeError("cache-process protocol seed declaration changed")
    if protocol.get("reuse_cache_for") != declared_seeds or protocol.get("cache_hit_required") is not True:
        raise RuntimeError("cache-process protocol must require the pinned cache for every declared seed")
    configured_encoders = tuple(protocol.get("encoders", []))
    selected_encoders = tuple(configured_encoders if encoders is None else encoders)
    if not selected_encoders or len(set(selected_encoders)) != len(selected_encoders):
        raise ValueError("encoder selection must be non-empty and contain no duplicates")
    if not set(selected_encoders).issubset(configured_encoders):
        raise ValueError(f"encoder selection is outside the predeclared cache-affected scope: {selected_encoders!r}")
    cache_source_runs = protocol.get("cache_source_runs", {})
    if set(selected_encoders) - set(cache_source_runs):
        raise RuntimeError("matrix must pin one completed v2 cache-source attempt per cache-affected encoder")

    reference_repo = Path(os.environ.get("PYABSA_REFERENCE_REPO", source_protocol["source_repository"])).resolve()
    python_exe = Path(
        os.environ.get("PYABSA_PYTHON", str(reference_repo / ".venv" / "Scripts" / "python.exe"))
    ).resolve()
    runner = project / "src" / "aste_repro" / "reference_baseline.py"
    if not reference_repo.is_dir():
        raise FileNotFoundError(f"local PyABSA reference repository is unavailable: {reference_repo}")
    if not python_exe.is_file():
        raise FileNotFoundError(f"source PyABSA Python runtime is unavailable: {python_exe}")
    if not runner.is_file():
        raise FileNotFoundError(f"integrated source-script runner is missing: {runner}")

    marker = runs_dir / QUEUE_NAME
    _assert_no_active_queue(os.getpid())
    previous_queue = _recover_old_reference_queue(marker, runs_dir)
    # Keep the finished v2 queue record byte-for-byte before v3 takes the active marker.
    if marker.is_file():
        old_marker = _read_json(marker)
        archive = runs_dir / "hoasa-emcgcn-reference-baseline-queue-v2-complete.status.json"
        if old_marker.get("experiment_protocol_id") == PROTOCOL_ID and old_marker.get("status") == "complete":
            if archive.exists():
                if archive.read_bytes() != marker.read_bytes():
                    raise FileExistsError(f"v2 queue archive already exists with different content: {archive}")
            else:
                shutil.copy2(marker, archive)

    # Recover only v3 root attempts whose recorded source worker has been proved stale.
    for manifest_path, manifest in _reference_manifests(runs_dir, CACHE_PROCESS_PROTOCOL_ID):
        if manifest.get("status") != "running":
            continue
        worker_live, worker_process = _process_matches(
            manifest.get("runner_pid"), WORKER_TOKEN, manifest.get("worker_started_at") or manifest.get("created_at")
        )
        if worker_live:
            raise RuntimeError(
                f"cache-process source worker is still live (PID {manifest.get('runner_pid')}); do not duplicate it"
            )
        manifest = _interrupt_stale_attempt(
            manifest_path,
            _read_json(marker) if marker.is_file() else {},
            "Recovered a stale cache-process attempt after confirming its source worker no longer exists; partial child artifacts were preserved.",
        )
        manifest["interruption_evidence"] = manifest.get("interruption_evidence", {})
        manifest["interruption_evidence"]["runner_process"] = _process_evidence(worker_process, WORKER_TOKEN)
        for child_manifest_path in sorted(manifest_path.parent.glob("seed-*/manifest.json")):
            child_manifest = _read_json(child_manifest_path)
            if child_manifest.get("status") == "running":
                _interrupt_stale_attempt(
                    child_manifest_path,
                    _read_json(marker) if marker.is_file() else {},
                    "The enclosing notebook queue stopped after this source process exited; child logs and partial artifacts were preserved.",
                )
        manifest["artifact_sha256"] = {
            path.relative_to(manifest_path.parent).as_posix(): _sha256(path)
            for path in sorted(manifest_path.parent.rglob("*"))
            if path.is_file() and path != manifest_path
        }
        _atomic_manifest(manifest_path, manifest)

    _refresh_report(report_path, matrix_path, runs_dir)
    cache_sources: dict[str, dict[str, Any]] = {}
    for encoder in selected_encoders:
        source_run_id = str(cache_source_runs[encoder])
        source_manifest_path = runs_dir / source_run_id / "manifest.json"
        if not source_manifest_path.is_file():
            raise FileNotFoundError(f"pinned cache-source run is missing for {encoder}: {source_run_id}")
        source_manifest = _read_json(source_manifest_path)
        if (
            source_manifest.get("status") != "complete"
            or source_manifest.get("experiment_protocol_id") != PROTOCOL_ID
            or source_manifest.get("encoder") != encoder
        ):
            raise RuntimeError(f"pinned cache-source run is not a complete v2 run for {encoder}: {source_run_id}")
        _verify_reference_artifacts(
            {"path": source_manifest_path, "manifest": source_manifest}, source_protocol
        )
        source_log = source_manifest_path.parent / str(source_manifest["reference"]["runner_log"])
        save_events = re.findall(
            r"Save cache dataset to ([^\r\n]+\.cache)",
            source_log.read_text(encoding="utf-8", errors="replace"),
        )
        if not save_events:
            raise RuntimeError(f"pinned v2 source run did not create a reusable dataset cache: {source_run_id}")
        cache_name = Path(save_events[0].strip()).name
        cache_path = source_manifest_path.parent / cache_name
        expected_hash = source_manifest.get("artifact_sha256", {}).get(cache_name)
        if not cache_path.is_file() or not expected_hash or _sha256(cache_path) != expected_hash:
            raise RuntimeError(f"pinned v2 cache artifact is absent or checksum-invalid: {cache_path}")
        cache_sources[encoder] = {
            "run_id": source_run_id,
            "relative_path": cache_name,
            "sha256": expected_hash,
            "path": cache_path,
        }
    started_at = _now()
    scope = (
        f"{CACHE_PROCESS_PROTOCOL_ID}: cache-affected unchanged source scripts; "
        f"one fresh process for each seed {declared_seeds} across {list(selected_encoders)}"
    )
    failures: list[dict[str, Any]] = []
    completed_encoders: list[str] = []
    skipped_encoders: list[str] = []
    current: dict[str, Any] = {
        "current_encoder": None,
        "current_seed": None,
        "worker_pid": None,
        "worker_started_at": None,
    }

    def write(status: str, **extra: Any) -> dict[str, Any]:
        return _write_reference_queue(
            marker,
            status,
            os.getpid(),
            started_at,
            scope,
            recovered_queue=previous_queue,
            completed_encoders=list(completed_encoders),
            skipped_encoders=list(skipped_encoders),
            failures=list(failures),
            experiment_protocol_id=CACHE_PROCESS_PROTOCOL_ID,
            expected_scripts=len(selected_encoders),
            expected_seed_outcomes=len(selected_encoders) * len(declared_seeds),
            selected_encoders=list(selected_encoders),
            **{**current, **extra},
        )

    def write_parent(run_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
        manifest_path = run_dir / "manifest.json"
        manifest["artifact_sha256"] = {
            path.relative_to(run_dir).as_posix(): _sha256(path)
            for path in sorted(run_dir.rglob("*"))
            if path.is_file() and path != manifest_path
        }
        _atomic_json(manifest_path, manifest)
        return manifest

    def write_metrics(run_dir: Path, seed_metrics: list[dict[str, Any]]) -> None:
        _atomic_json(
            run_dir / "reference_metrics.json",
            {
                "run_id": run_dir.name,
                "metric": "PyABSA MetricVisualizer Max-Test-F1",
                "unit": "percent",
                "selection_boundary": "test set used during training for checkpoint selection; test-leaked reference only",
                "execution_protocol_id": CACHE_PROCESS_PROTOCOL_ID,
                "seeds": seed_metrics,
            },
        )

    def verify_seed_child(child_manifest_path: Path, encoder: str, seed: int) -> dict[str, Any] | None:
        if not child_manifest_path.is_file():
            return None
        child_manifest = _read_json(child_manifest_path)
        if child_manifest.get("status") != "complete":
            return None
        child_protocol = {**source_protocol, "id": CACHE_PROCESS_PROTOCOL_ID, "seeds": [seed]}
        verified = _verify_reference_artifacts(
            {"path": child_manifest_path, "manifest": child_manifest}, child_protocol
        )
        if len(verified) != 1 or int(verified[0]["seed"]) != seed:
            raise RuntimeError(f"verified child run does not match {encoder} seed {seed}")
        cache_record = child_manifest.get("reference", {}).get("dataset_cache_evidence", {})
        if cache_record.get("cache_hit_required") is not True:
            raise RuntimeError(f"{encoder} seed {seed} did not load its pinned shared dataset cache")
        return verified[0]

    write("running", phase="starting")
    try:
        for encoder in selected_encoders:
            attempts = _reference_manifests(runs_dir, CACHE_PROCESS_PROTOCOL_ID, encoder)
            complete = [(path, record) for path, record in attempts if record.get("status") == "complete"]
            if len(complete) > 1:
                raise RuntimeError(f"multiple complete cache-process attempts exist for {encoder}; refusing to pool")
            if complete:
                _verify_isolated_reference_attempt(
                    {"path": complete[0][0], "manifest": complete[0][1]}, source_protocol, protocol
                )
                skipped_encoders.append(encoder)
                current.update({"current_encoder": encoder, "current_seed": None, "worker_pid": None, "worker_started_at": None})
                write("running", phase="skipping_verified_complete")
                print(f"Already complete and checksum-verified under the cache-process protocol; skipping {encoder}.", flush=True)
                continue

            resumable = [
                (path, record)
                for path, record in attempts
                if record.get("status") in {"interrupted", "complete_with_failures", "failed"}
            ]
            if len(resumable) > 1:
                raise RuntimeError(f"multiple incomplete cache-process attempts exist for {encoder}; refusing to choose between them")

            for manifest_path, old_manifest in attempts:
                if old_manifest.get("status") == "running":
                    live, _process = _process_matches(
                        old_manifest.get("runner_pid"), WORKER_TOKEN,
                        old_manifest.get("worker_started_at") or old_manifest.get("created_at"),
                    )
                    if live:
                        raise RuntimeError(f"cache-process worker is active for {old_manifest.get('run_id')}; refusing a duplicate")
                    _interrupt_stale_attempt(
                        manifest_path,
                        _read_json(marker),
                        "Resumed after a prior cache-process queue stopped; its partial seed artifacts are preserved.",
                    )

            source_spec = source_protocol["source_scripts"][encoder]
            if resumable:
                parent_manifest_path, parent_manifest = resumable[0]
                run_dir = parent_manifest_path.parent
                _verify_isolated_reference_attempt(
                    {"path": parent_manifest_path, "manifest": parent_manifest}, source_protocol, protocol
                )
                valid_seed_runs: dict[str, str] = {}
                for seed in declared_seeds:
                    relative = parent_manifest.get("seed_runs", {}).get(str(seed))
                    if relative and verify_seed_child(run_dir / relative, encoder, seed) is not None:
                        valid_seed_runs[str(seed)] = relative
                # Adopt a child that finished just before the queue process was interrupted.
                for seed in declared_seeds:
                    if str(seed) in valid_seed_runs:
                        continue
                    candidates = sorted(
                        path / "manifest.json"
                        for path in run_dir.iterdir()
                        if path.is_dir()
                        and (path.name == f"seed-{seed}" or path.name.startswith(f"seed-{seed}-attempt-"))
                    )
                    complete_children = [
                        path for path in candidates
                        if verify_seed_child(path, encoder, seed) is not None
                    ]
                    if len(complete_children) > 1:
                        raise RuntimeError(f"multiple completed child runs exist for {encoder} seed {seed}; refusing to pool")
                    if complete_children:
                        valid_seed_runs[str(seed)] = complete_children[0].relative_to(run_dir).as_posix()
                parent_manifest["seed_runs"] = valid_seed_runs
                parent_manifest["seed_metrics"] = [
                    verify_seed_child(run_dir / valid_seed_runs[str(seed)], encoder, seed)
                    for seed in declared_seeds
                    if str(seed) in valid_seed_runs
                ]
                prior_failures = parent_manifest.get("failures", [])
                if prior_failures:
                    parent_manifest.setdefault("seed_attempt_history", []).append(
                        {"recorded_at": _now(), "prior_failures": prior_failures}
                    )
                parent_manifest["failures"] = []
                parent_manifest["status"] = "running"
                parent_manifest["completed_at"] = None
                parent_manifest["runner_pid"] = None
                parent_manifest["worker_started_at"] = None
                parent_manifest["execution"].setdefault("cache_materialization_by_seed", {})
                write_metrics(run_dir, parent_manifest["seed_metrics"])
                write_parent(run_dir, parent_manifest)
                print(
                    f"Resuming {encoder} attempt {run_dir.name}; preserving and skipping "
                    f"{len(valid_seed_runs)} verified seed run(s).",
                    flush=True,
                )
            else:
                run_dir = _next_attempt_dir(runs_dir, encoder)
                run_dir.mkdir(parents=True, exist_ok=False)
                created_at = _now()
                parent_manifest = {
                    "schema_version": 1,
                    "run_id": run_dir.name,
                    "status": "running",
                    "created_at": created_at,
                    "runner_pid": None,
                    "worker_started_at": None,
                    "experiment_protocol_id": CACHE_PROCESS_PROTOCOL_ID,
                    "purpose": "reference_baseline_reproduction",
                    "dataset": "hoasa",
                    "train_dataset": "hoasa",
                    "test_dataset": "hoasa",
                    "architecture": "emcgcn",
                    "encoder": encoder,
                    "seeds": list(declared_seeds),
                    "execution": {
                        "mode": "fresh_process_per_declared_seed",
                        "seed_order": list(declared_seeds),
                        "cache_source_run_id": cache_sources[encoder]["run_id"],
                        "cache_source_relative_path": cache_sources[encoder]["relative_path"],
                        "cache_source_sha256": cache_sources[encoder]["sha256"],
                        "cache_materialization_by_seed": {},
                    },
                    "model": {"repository": source_spec["model_id"], "revision": source_spec["model_revision"]},
                    "requested_config": source_spec["settings"],
                    "resolved_config": None,
                    "seed_metrics": [],
                    "seed_runs": {},
                    "failures": [],
                    "artifact_sha256": {},
                }
                parent_manifest_path = run_dir / "manifest.json"
                _atomic_json(parent_manifest_path, parent_manifest)
                _atomic_json(run_dir / "reference_metrics.json", {
                    "run_id": run_dir.name,
                    "metric": "PyABSA MetricVisualizer Max-Test-F1",
                    "unit": "percent",
                    "selection_boundary": "test set used during training for checkpoint selection; test-leaked reference only",
                    "execution_protocol_id": CACHE_PROCESS_PROTOCOL_ID,
                    "seeds": [],
                })

            for seed in declared_seeds:
                if str(seed) in parent_manifest["seed_runs"]:
                    continue
                child_dir = run_dir / f"seed-{seed}"
                if child_dir.exists():
                    attempt_index = 2
                    while (run_dir / f"seed-{seed}-attempt-{attempt_index:03d}").exists():
                        attempt_index += 1
                    child_dir = run_dir / f"seed-{seed}-attempt-{attempt_index:03d}"
                child_dir.mkdir(parents=True, exist_ok=False)
                cache_source = cache_sources[encoder]
                source_cache = cache_source["path"]
                expected_hash = cache_source["sha256"]
                destination_cache = child_dir / cache_source["relative_path"]
                materialization = "hardlink"
                try:
                    os.link(source_cache, destination_cache)
                except OSError:
                    materialization = "copy"
                    shutil.copy2(source_cache, destination_cache)
                if _sha256(destination_cache) != expected_hash:
                    raise RuntimeError(f"dataset cache materialization checksum mismatch for seed {seed} for {encoder}")
                parent_manifest["execution"]["cache_materialization_by_seed"][str(seed)] = materialization

                child_log = child_dir / "training.log"
                current.update({"current_encoder": encoder, "current_seed": seed, "worker_pid": None, "worker_started_at": None, "attempt": run_dir.name})
                write("running", phase="launching_seed_source_process")
                env = os.environ.copy()
                env.update({
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "HF_DATASETS_OFFLINE": "1",
                    "MPLBACKEND": "Agg",
                    "PYTHONUNBUFFERED": "1",
                    "PYTHONIOENCODING": "utf-8",
                    "ASTE_PINNED_DATASET_DIR": str(
                        cache_source["path"].parent / "datasets" / "aste"
                    ),
                })
                command = [
                    str(python_exe), str(runner),
                    "--encoder", encoder,
                    "--run-dir", str(child_dir),
                    "--reference-repo", str(reference_repo),
                    "--protocol-id", CACHE_PROCESS_PROTOCOL_ID,
                    "--seed", str(seed),
                ]
                print(
                    f"Launching unchanged {source_spec['path']} in a fresh process for seed {seed} "
                    f"({encoder}) into {child_dir.relative_to(run_dir).as_posix()}.", flush=True
                )
                with child_log.open("wb") as log_stream:
                    worker = subprocess.Popen(
                        command,
                        cwd=child_dir,
                        env=env,
                        stdout=log_stream,
                        stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
                    )
                    current.update({"worker_pid": worker.pid, "worker_started_at": _now()})
                    parent_manifest["runner_pid"] = worker.pid
                    parent_manifest["worker_started_at"] = current["worker_started_at"]
                    write_parent(run_dir, parent_manifest)
                    write("running", phase="seed_source_process_training")
                    last_write = time.monotonic()
                    while worker.poll() is None:
                        time.sleep(10)
                        if time.monotonic() - last_write >= 30:
                            write(
                                "running",
                                phase="seed_source_process_training",
                                worker_log_bytes=child_log.stat().st_size if child_log.exists() else 0,
                                last_training_progress=_latest_training_progress(child_log),
                            )
                            last_write = time.monotonic()
                    return_code = worker.wait()

                child_manifest_path = child_dir / "manifest.json"
                child_manifest = _read_json(child_manifest_path) if child_manifest_path.is_file() else None
                if child_manifest is not None:
                    child_manifest = _finalize_artifact_hashes(child_dir)
                child_protocol = {**source_protocol, "id": CACHE_PROCESS_PROTOCOL_ID, "seeds": [seed]}
                if return_code == 0 and child_manifest and child_manifest.get("status") == "complete":
                    verified_seed_metrics = _verify_reference_artifacts(
                        {"path": child_manifest_path, "manifest": child_manifest}, child_protocol
                    )
                    if len(verified_seed_metrics) != 1 or int(verified_seed_metrics[0]["seed"]) != seed:
                        raise RuntimeError(f"verified seed metrics do not match requested seed {seed} for {encoder}")
                    seed_metric = verified_seed_metrics[0]
                    parent_manifest["seed_runs"][str(seed)] = (
                        f"{child_dir.name}/manifest.json"
                    )
                    parent_manifest["seed_metrics"].append(seed_metric)
                    cache_record = child_manifest.get("reference", {}).get("dataset_cache_evidence", {})
                    if cache_record.get("cache_hit_required") is not True:
                        raise RuntimeError(f"seed {seed} did not record the required shared cache hit for {encoder}")
                    if seed == declared_seeds[0]:
                        parent_manifest["code"] = child_manifest.get("code")
                        parent_manifest["source_files"] = child_manifest.get("source_files")
                        parent_manifest["resolved_config"] = child_manifest.get("resolved_config")
                        parent_manifest["reference"] = child_manifest.get("reference")
                        parent_manifest["environment"] = child_manifest.get("environment")
                    current.update({"worker_pid": None, "worker_started_at": None})
                    write_metrics(run_dir, parent_manifest["seed_metrics"])
                    parent_manifest["runner_pid"] = None
                    parent_manifest["worker_started_at"] = None
                    write_parent(run_dir, parent_manifest)
                    _verify_isolated_reference_attempt(
                        {"path": parent_manifest_path, "manifest": parent_manifest}, source_protocol, protocol
                    )
                    write("running", phase="refreshing_report_after_seed")
                    _refresh_report(report_path, matrix_path, runs_dir)
                    print(f"Verified {encoder} seed {seed}: {seed_metric['max_test_f1_pct']:.6f} Max-Test-F1.", flush=True)
                else:
                    error = (child_manifest or {}).get("failure") or f"source worker exited with status {return_code}"
                    failure = {"encoder": encoder, "seed": seed, "attempt": run_dir.name, "status": (child_manifest or {}).get("status", "missing_manifest"), "return_code": return_code, "error": str(error)[:1000]}
                    failures.append(failure)
                    parent_manifest["failures"].append(failure)
                    current.update({"worker_pid": None, "worker_started_at": None})
                    write_metrics(run_dir, parent_manifest["seed_metrics"])
                    parent_manifest["runner_pid"] = None
                    parent_manifest["worker_started_at"] = None
                    write_parent(run_dir, parent_manifest)
                    _refresh_report(report_path, matrix_path, runs_dir)
                    print(f"Preserved {encoder} seed {seed} as {failure['status']}; continuing without selecting a substitute seed.", flush=True)

            parent_manifest["completed_at"] = _now()
            parent_manifest["status"] = (
                "complete"
                if set(parent_manifest["seed_runs"]) == {str(value) for value in declared_seeds} and not parent_manifest["failures"]
                else "complete_with_failures" if parent_manifest["seed_runs"] else "failed"
            )
            write_metrics(run_dir, parent_manifest["seed_metrics"])
            write_parent(run_dir, parent_manifest)
            if parent_manifest["status"] == "complete":
                _verify_isolated_reference_attempt(
                    {"path": parent_manifest_path, "manifest": parent_manifest}, source_protocol, protocol
                )
                completed_encoders.append(encoder)
                print(f"Completed and verified all declared cache-process seeds for {encoder}.", flush=True)
            current.update({"current_encoder": encoder, "current_seed": None, "worker_pid": None, "worker_started_at": None})
            write("running", phase="refreshing_report_after_encoder")
            _refresh_report(report_path, matrix_path, runs_dir)

        complete_count = sum(
            1 for encoder in selected_encoders
            if any(m.get("status") == "complete" for _p, m in _reference_manifests(runs_dir, CACHE_PROCESS_PROTOCOL_ID, encoder))
        )
        outcome_count = sum(
            len(m.get("seed_metrics", []))
            for encoder in selected_encoders
            for _p, m in _reference_manifests(runs_dir, CACHE_PROCESS_PROTOCOL_ID, encoder)
        )
        status = "complete" if complete_count == len(selected_encoders) and not failures else "complete_with_failures"
        write(
            status,
            phase="finished",
            completed_at=_now(),
            complete_scripts=complete_count,
            seed_outcomes=outcome_count,
        )
        _refresh_report(report_path, matrix_path, runs_dir)
        print(
            f"Cache-process reference queue {status}: {complete_count}/{len(selected_encoders)} source cells, "
            f"{outcome_count}/{len(selected_encoders) * len(declared_seeds)} seed outcomes; {len(failures)} failure(s).",
            flush=True,
        )
        return _read_json(marker)
    except KeyboardInterrupt:
        write(
            "interrupted",
            phase="interrupted",
            completed_at=_now(),
            failure_type="KeyboardInterrupt",
            failure="notebook cache-process queue interrupted; completed and partial seed artifacts are preserved",
        )
        raise
    except BaseException as error:
        write(
            "failed",
            phase="failed",
            completed_at=_now(),
            failure_type=type(error).__name__,
            failure=str(error)[:2000],
        )
        raise
