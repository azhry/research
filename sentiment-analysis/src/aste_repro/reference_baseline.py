"""Run one unchanged local PyABSA EMCGCN source script with full provenance.

This protocol reproduces the source scripts' test-selected Max-Test-F1 metric.
It is intentionally kept separate from the clean, validation-selected v3 runs.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import runpy
import shutil
import subprocess
import sys
import tempfile
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "configs" / "experiment-matrix.json"
CANONICAL_DATA = ROOT / "data" / "sources" / "hoasa"
PROTOCOL_ID = "pyabsa-2.4.3-emcgcn-reference-max-test-v2-seed-cache-compatible"
CACHE_PROCESS_PROTOCOL_ID = "pyabsa-2.4.3-emcgcn-reference-max-test-v3-fresh-seed-process-cache-hit"
SEEDS = [1, 2, 3]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _parsed_rows(path: Path) -> list[tuple[str, list[tuple[tuple[int, ...], tuple[int, ...], str]]]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, raw in enumerate(stream, 1):
            line = raw.rstrip("\r\n")
            if line.count("####") != 1:
                raise ValueError(f"{path}:{line_number}: expected one #### delimiter")
            sentence, raw_annotations = line.split("####", maxsplit=1)
            annotations = ast.literal_eval(raw_annotations.strip())
            normalized = []
            for annotation in annotations:
                if not isinstance(annotation, (list, tuple)) or len(annotation) != 3:
                    raise ValueError(f"{path}:{line_number}: invalid ASTE annotation")
                aspect, opinion, polarity = annotation
                normalized.append((tuple(aspect), tuple(opinion), str(polarity)))
            rows.append((sentence.strip(), normalized))
    return rows


def _git_value(cwd: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _expected_scripts(matrix: dict[str, Any]) -> dict[str, dict[str, Any]]:
    settings = matrix.get("reference_baseline_protocol", {})
    scripts = settings.get("source_scripts", {})
    if settings.get("id") != PROTOCOL_ID:
        raise ValueError("matrix reference baseline protocol ID does not match this runner")
    if set(scripts) != set(matrix["encoders"]):
        raise ValueError("matrix source-script mapping must cover all four encoders")
    return scripts


def _package_versions() -> dict[str, str | None]:
    versions = {}
    for name in ("pyabsa", "transformers", "torch", "metric-visualizer", "huggingface-hub"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _snapshot_workspace_source(code_files: dict[str, Path], run_dir: Path) -> dict[str, str]:
    """Keep the exact runner/config source that produced the manifest beside the run."""
    snapshot_root = run_dir / "provenance" / "workspace-source"
    hashes: dict[str, str] = {}
    for relative, source in code_files.items():
        if not source.is_file():
            continue
        expected_hash = _sha256(source)
        destination = snapshot_root / Path(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if _sha256(destination) != expected_hash:
            raise RuntimeError(f"workspace source snapshot checksum mismatch for {relative}")
        hashes[relative] = expected_hash
    return hashes


def _resolve_log_step_sentinel(
    trainer: Any,
    expected: dict[str, Any],
    observed: Any,
    log_path: Path,
    run_dir: Path,
    expected_seeds: list[int] | None = None,
) -> dict[str, Any]:
    """Record PyABSA's evidenced -1 => train-loader-length runtime normalization."""
    if expected.get("settings", {}).get("log_step") != -1:
        raise RuntimeError("runtime log_step normalization was requested for a non-sentinel setting")
    if not isinstance(observed, int) or isinstance(observed, bool) or observed <= 0:
        raise RuntimeError(f"PyABSA resolved negative log_step to an invalid value: {observed!r}")

    instructor_class = getattr(trainer, "training_instructor", None)
    source_name = inspect.getsourcefile(instructor_class) if instructor_class is not None else None
    if not source_name:
        raise RuntimeError("could not locate PyABSA instructor source for log_step normalization")
    source_path = Path(source_name).resolve()
    source_text = source_path.read_text(encoding="utf-8")
    if not re.search(
        r"if\s+self\.config\.log_step\s*<\s*0\s*:\s*"
        r"self\.config\.log_step\s*=\s*\(\s*"
        r"len\(self\.train_dataloaders\[0\]\)",
        source_text,
        re.DOTALL,
    ):
        raise RuntimeError("installed PyABSA instructor lacks the expected negative log_step normalization")

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    log_steps = [int(value) for value in re.findall(r"\blog_step:(-?\d+)\s+-->\s+Calling Count", log_text)]
    seeds = [int(value) for value in re.findall(r"\bseed:(\d+)\s+-->\s+Calling Count", log_text)]
    num_steps = [int(value) for value in re.findall(r"\bNum steps\s*=\s*(\d+)", log_text)]
    epochs = int(expected["settings"]["num_epoch"])
    expected_seeds = SEEDS if expected_seeds is None else expected_seeds
    # A fresh process logs the requested -1 once. PyABSA normalizes its
    # in-memory config after constructing the dataloader; the normalized value
    # is durable in the saved checkpoint args, not in another startup log line.
    expected_log_steps = [-1] if len(expected_seeds) == 1 else [-1, observed, observed]
    if log_steps != expected_log_steps:
        raise RuntimeError(f"training log does not prove expected log_step request: {log_steps!r}")
    if seeds != expected_seeds:
        raise RuntimeError(f"training log does not prove expected seed starts {expected_seeds!r}: {seeds!r}")
    if num_steps != [observed * epochs] * len(expected_seeds):
        raise RuntimeError(f"training log step counts disagree with resolved log_step: {num_steps!r}")
    checkpoint_metrics = _checkpoint_max_test_metrics(
        run_dir, expected_seeds, str(expected.get("model_id", ""))
    )
    checkpoint_log_steps = {
        int(item["seed"]): int(item["log_step"])
        for item in checkpoint_metrics
        if item.get("log_step") is not None
    }
    if checkpoint_log_steps != {seed: observed for seed in expected_seeds}:
        raise RuntimeError(
            "saved checkpoint args do not prove the expected normalized log_step: "
            f"{checkpoint_log_steps!r}"
        )

    snapshot = run_dir / "provenance" / "runtime" / "pyabsa_instructor.py"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, snapshot)
    source_hash = _sha256(source_path)
    if _sha256(snapshot) != source_hash:
        raise RuntimeError("PyABSA instructor source snapshot checksum mismatch")
    return {
        "requested": -1,
        "resolved": observed,
        "resolution_rule": "PyABSA 2.4.3 replaces negative log_step with len(train_dataloaders[0])",
        "runtime_source_path": str(source_path),
        "runtime_source_sha256": source_hash,
        "runtime_source_snapshot": snapshot.relative_to(run_dir).as_posix(),
        "training_log_path": "training.log",
        "logged_seed_order": seeds,
        "logged_log_step_values": log_steps,
        "logged_num_steps": num_steps,
        "checkpoint_log_step_by_seed": {str(seed): step for seed, step in checkpoint_log_steps.items()},
        "num_epoch": epochs,
        "evidence_verified": True,
    }


def _install_seed_cache_compatibility(run_dir: Path) -> tuple[dict[str, Any], Any]:
    """Keep a PyABSA dataset-cache hit from replacing the seed selected by Trainer._run."""
    from pyabsa.framework.instructor_class.instructor_template import BaseTrainingInstructor
    from pyabsa.tasks.AspectSentimentTripletExtraction.instructor.instructor import (
        ASTETrainingInstructor,
    )

    original = BaseTrainingInstructor.load_cache_dataset
    original_aste_prepare = ASTETrainingInstructor._load_dataset_and_prepare_dataloader
    source_name = inspect.getsourcefile(original)
    if not source_name:
        raise RuntimeError("could not locate PyABSA dataset-cache loader source")
    source_path = Path(source_name).resolve()
    source_snapshot = run_dir / "provenance" / "runtime" / "pyabsa_instructor_template.py"
    source_snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, source_snapshot)
    source_hash = _sha256(source_path)
    if _sha256(source_snapshot) != source_hash:
        raise RuntimeError("PyABSA dataset-cache loader snapshot checksum mismatch")

    aste_source_name = inspect.getsourcefile(original_aste_prepare)
    if not aste_source_name:
        raise RuntimeError("could not locate PyABSA ASTE dataset preparation source")
    aste_source_path = Path(aste_source_name).resolve()
    aste_source_snapshot = run_dir / "provenance" / "runtime" / "pyabsa_aste_instructor.py"
    aste_source_snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(aste_source_path, aste_source_snapshot)
    aste_source_hash = _sha256(aste_source_path)
    if _sha256(aste_source_snapshot) != aste_source_hash:
        raise RuntimeError("PyABSA ASTE instructor source snapshot checksum mismatch")

    dataset_cache_key_events: list[dict[str, Any]] = []
    dataset_restore_events: list[dict[str, Any]] = []
    pinned_dataset_text = os.environ.get("ASTE_PINNED_DATASET_DIR")
    pinned_dataset_dir = Path(pinned_dataset_text).resolve() if pinned_dataset_text else None
    child_dataset_dir = (run_dir / "datasets" / "aste").resolve()
    preloaded_cache_files = {
        path.name: _sha256(path)
        for path in run_dir.glob("*.dataset.*.cache")
        if path.is_file()
    }

    def config_dataset(config: Any) -> Any:
        if config is None:
            return None
        args = getattr(config, "args", None)
        if isinstance(args, dict) and "dataset" in args:
            return args["dataset"]
        return getattr(config, "dataset", None)

    def set_config_dataset(config: Any, value: str) -> Any:
        if config is None:
            return None
        previous = config_dataset(config)
        setattr(config, "dataset", value)
        args = getattr(config, "args", None)
        if isinstance(args, dict):
            args["dataset"] = value
        return previous

    if pinned_dataset_dir is not None:
        if not pinned_dataset_dir.is_dir():
            raise FileNotFoundError(f"pinned dataset-cache source directory is missing: {pinned_dataset_dir}")
        if len(preloaded_cache_files) != 1:
            raise RuntimeError("fresh seed process must contain exactly one preloaded dataset cache")
        data_hashes = {}
        for split in ("train", "dev", "test"):
            child_path = child_dataset_dir / f"{split}.txt"
            pinned_path = pinned_dataset_dir / f"{split}.txt"
            if not child_path.is_file() or not pinned_path.is_file():
                raise FileNotFoundError(f"cache parity dataset split is missing for {split}")
            child_hash = _sha256(child_path)
            pinned_hash = _sha256(pinned_path)
            if child_hash != pinned_hash:
                raise RuntimeError(f"pinned cache dataset differs from child source data for {split}")
            data_hashes[split] = child_hash

        expected_cache_name = next(iter(preloaded_cache_files))
        expected_cache_sha256 = preloaded_cache_files[expected_cache_name]
        if expected_cache_sha256 == "":
            raise RuntimeError("preloaded dataset cache has no checksum")
    else:
        data_hashes = {}
        expected_cache_name = None
        expected_cache_sha256 = None

    corrections: list[dict[str, Any]] = []
    post_load_corrections: list[dict[str, Any]] = []
    runtime_handle_corrections: list[dict[str, Any]] = []
    post_load_runtime_handle_corrections: list[dict[str, Any]] = []

    def runtime_handles(config: Any) -> dict[str, Any]:
        return {
            name: getattr(config, name, None)
            for name in ("logger", "MV")
            if config is not None and getattr(config, name, None) is not None
        }

    def restore_runtime_handles(config: Any, handles: dict[str, Any]) -> list[str]:
        restored = []
        if config is None:
            return restored
        for name, value in handles.items():
            if getattr(config, name, None) is not value:
                setattr(config, name, value)
                restored.append(name)
        return restored

    def restore_config_seed(config: Any, requested_seed: int) -> tuple[bool, bool]:
        restored_attribute = getattr(config, "seed", None) != requested_seed
        if restored_attribute:
            config.seed = requested_seed
        config_args = getattr(config, "args", None)
        restored_args = isinstance(config_args, dict) and config_args.get("seed") != requested_seed
        if restored_args:
            config_args["seed"] = requested_seed
        return restored_attribute, restored_args

    def preserve_requested_seed(instructor: Any, *args: Any, **kwargs: Any) -> Any:
        requested_seed = getattr(instructor.config, "seed", None)
        requested_dataset = config_dataset(instructor.config)
        if pinned_dataset_dir is not None and Path(str(requested_dataset)).resolve() != child_dataset_dir:
            raise RuntimeError(
                "PyABSA cache-loader dataset path differs from the child-local data directory: "
                f"{requested_dataset!r} != {child_dataset_dir}"
            )
        requested_dataset = str(Path(str(requested_dataset)).resolve()) if requested_dataset else None
        instructor_handles = runtime_handles(instructor.config)
        external_config = kwargs.get("config")
        external_requested_dataset = config_dataset(external_config)
        external_handles = runtime_handles(external_config)
        external_requested_seed = getattr(external_config, "seed", None)
        external_args = getattr(external_config, "args", None)
        external_requested_args_seed = (
            external_args.get("seed") if isinstance(external_args, dict) else None
        )
        if pinned_dataset_dir is not None:
            set_config_dataset(instructor.config, str(pinned_dataset_dir))
        try:
            cache_path = original(instructor, *args, **kwargs)
        finally:
            loaded_dataset = config_dataset(instructor.config)
            if requested_dataset is not None:
                set_config_dataset(instructor.config, requested_dataset)
            if external_config is not None and external_requested_dataset is not None:
                set_config_dataset(external_config, str(Path(str(external_requested_dataset)).resolve()))
        resolved_cache_path = (
            (Path.cwd() / cache_path).resolve()
            if cache_path is not None and not Path(str(cache_path)).is_absolute()
            else Path(str(cache_path)).resolve()
            if cache_path is not None
            else None
        )
        if pinned_dataset_dir is not None:
            expected_cache_path = (Path.cwd() / str(expected_cache_name)).resolve()
            if (
                resolved_cache_path != expected_cache_path
                or not expected_cache_path.is_file()
                or _sha256(expected_cache_path) != expected_cache_sha256
                or getattr(instructor.config, "overwrite_cache", False)
            ):
                raise RuntimeError(
                    "PyABSA cache-key normalization did not resolve to the preloaded pinned cache"
                )
            dataset_cache_key_events.append(
                {
                    "requested_dataset_directory": requested_dataset,
                    "cache_key_dataset_directory": str(pinned_dataset_dir),
                    "dataset_directory_after_cache_load": config_dataset(instructor.config),
                    "external_config_dataset_after_cache_load": config_dataset(external_config),
                    "external_config_present": external_config is not None,
                    "cache_path": str(resolved_cache_path),
                    "cache_sha256": _sha256(expected_cache_path),
                    "cache_hit_confirmed_by_preloaded_file": True,
                    "split_sha256": dict(data_hashes),
                }
            )
        if requested_dataset is not None:
            dataset_restore_events.append(
                {
                    "method": "BaseTrainingInstructor.load_cache_dataset",
                    "requested_dataset_directory": requested_dataset,
                    "loaded_config_dataset_directory": loaded_dataset,
                    "restored_dataset_directory": config_dataset(instructor.config),
                }
            )
        cached_seed = getattr(instructor.config, "seed", None)
        cached_args = getattr(instructor.config, "args", None)
        cached_args_seed = cached_args.get("seed") if isinstance(cached_args, dict) else None
        if isinstance(requested_seed, int) and not isinstance(requested_seed, bool):
            restored_instructor_attribute, restored_instructor_args = restore_config_seed(
                instructor.config, requested_seed
            )
            restored_external_attribute = False
            restored_external_args = False
            if external_config is not None:
                # Some PyABSA call paths pass a separate ConfigManager through
                # ``config``; keep both its attribute and args mapping aligned.
                restored_external_attribute, restored_external_args = restore_config_seed(
                    external_config, requested_seed
                )
            restored_instructor_handles = restore_runtime_handles(instructor.config, instructor_handles)
            restored_external_handles = restore_runtime_handles(external_config, external_handles)
            if any(
                (
                    restored_instructor_attribute,
                    restored_instructor_args,
                    restored_external_attribute,
                    restored_external_args,
                )
            ):
                corrections.append(
                    {
                        "requested_seed": requested_seed,
                        "cached_config_seed": cached_seed,
                        "cached_config_args_seed": cached_args_seed,
                        "external_config_seed_before_restore": external_requested_seed,
                        "external_config_args_seed_before_restore": external_requested_args_seed,
                        "restored_seed": getattr(instructor.config, "seed", None),
                        "restored_config_args_seed": (
                            instructor.config.args.get("seed")
                            if isinstance(getattr(instructor.config, "args", None), dict)
                            else None
                        ),
                        "restored_external_config_seed": getattr(external_config, "seed", None),
                        "restored_external_config_args_seed": (
                            external_config.args.get("seed")
                            if isinstance(getattr(external_config, "args", None), dict)
                            else None
                        ),
                        "cache_path": str(cache_path) if cache_path is not None else None,
                    }
                )
            if restored_instructor_handles or restored_external_handles:
                runtime_handle_corrections.append(
                    {
                        "requested_seed": requested_seed,
                        "restored_instructor_handles": restored_instructor_handles,
                        "restored_external_handles": restored_external_handles,
                        "cache_path": str(cache_path) if cache_path is not None else None,
                    }
                )
        return cache_path

    def preserve_requested_seed_after_dataset_load(
        instructor: Any, *args: Any, **kwargs: Any
    ) -> Any:
        # ASTETrainingInstructor reloads the cache a second time after
        # load_cache_dataset returns. Capture the seed before that second load
        # and restore it after the cached ConfigManager has replaced self.config.
        requested_seed = getattr(instructor.config, "seed", None)
        requested_dataset = config_dataset(instructor.config)
        requested_dataset = str(Path(str(requested_dataset)).resolve()) if requested_dataset else None
        requested_handles = runtime_handles(instructor.config)
        result = original_aste_prepare(instructor, *args, **kwargs)
        loaded_dataset = config_dataset(instructor.config)
        if requested_dataset is not None:
            set_config_dataset(instructor.config, requested_dataset)
            dataset_restore_events.append(
                {
                    "method": "ASTETrainingInstructor._load_dataset_and_prepare_dataloader",
                    "requested_dataset_directory": requested_dataset,
                    "loaded_config_dataset_directory": loaded_dataset,
                    "restored_dataset_directory": config_dataset(instructor.config),
                }
            )
        loaded_seed = getattr(instructor.config, "seed", None)
        loaded_args = getattr(instructor.config, "args", None)
        loaded_args_seed = loaded_args.get("seed") if isinstance(loaded_args, dict) else None
        if isinstance(requested_seed, int) and not isinstance(requested_seed, bool):
            restored_attribute, restored_args = restore_config_seed(
                instructor.config, requested_seed
            )
            restored_handles = restore_runtime_handles(instructor.config, requested_handles)
            if restored_attribute or restored_args:
                post_load_corrections.append(
                    {
                        "requested_seed": requested_seed,
                        "reloaded_config_seed": loaded_seed,
                        "reloaded_config_args_seed": loaded_args_seed,
                        "restored_seed": getattr(instructor.config, "seed", None),
                        "restored_config_args_seed": (
                            instructor.config.args.get("seed")
                            if isinstance(getattr(instructor.config, "args", None), dict)
                            else None
                        ),
                    }
                )
            if restored_handles:
                post_load_runtime_handle_corrections.append(
                    {
                        "requested_seed": requested_seed,
                        "restored_handles": restored_handles,
                    }
                )
        return result

    preserve_requested_seed.__name__ = original.__name__
    preserve_requested_seed.__doc__ = original.__doc__
    BaseTrainingInstructor.load_cache_dataset = preserve_requested_seed
    preserve_requested_seed_after_dataset_load.__name__ = original_aste_prepare.__name__
    preserve_requested_seed_after_dataset_load.__doc__ = original_aste_prepare.__doc__
    ASTETrainingInstructor._load_dataset_and_prepare_dataloader = (
        preserve_requested_seed_after_dataset_load
    )

    def restore_runtime_method() -> None:
        if BaseTrainingInstructor.load_cache_dataset is preserve_requested_seed:
            BaseTrainingInstructor.load_cache_dataset = original
        if (
            ASTETrainingInstructor._load_dataset_and_prepare_dataloader
            is preserve_requested_seed_after_dataset_load
        ):
            ASTETrainingInstructor._load_dataset_and_prepare_dataloader = original_aste_prepare

    evidence = {
        "method": "BaseTrainingInstructor.load_cache_dataset",
        "runtime_source_path": str(source_path),
        "runtime_source_sha256": source_hash,
        "runtime_source_snapshot": source_snapshot.relative_to(run_dir).as_posix(),
        "secondary_method": "ASTETrainingInstructor._load_dataset_and_prepare_dataloader",
        "secondary_runtime_source_path": str(aste_source_path),
        "secondary_runtime_source_sha256": aste_source_hash,
        "secondary_runtime_source_snapshot": aste_source_snapshot.relative_to(run_dir).as_posix(),
        "behavior": (
            "PyABSA 2.4.3 omits seed from its dataset-cache key and reloads cached ConfigManager state in both "
            "BaseTrainingInstructor.load_cache_dataset and ASTETrainingInstructor._load_dataset_and_prepare_dataloader; "
            "the runtime wrapper normalizes only the dataset path while computing the cache filename, then restores "
            "the seed-local dataset path, requested seed, and fresh-process logger/MetricVisualizer after each load. "
            "The cache-key dataset and child-local training data are byte-verified identical."
        ),
        "seed_restore_events": corrections,
        "post_dataset_load_seed_restore_events": post_load_corrections,
        "runtime_handle_restore_events": runtime_handle_corrections,
        "post_dataset_load_runtime_handle_restore_events": post_load_runtime_handle_corrections,
        "dataset_cache_key_normalization": {
            "enabled": pinned_dataset_dir is not None,
            "pinned_dataset_directory": str(pinned_dataset_dir) if pinned_dataset_dir else None,
            "child_dataset_directory": str(child_dataset_dir),
            "cache_file_name": expected_cache_name,
            "cache_file_sha256": expected_cache_sha256,
            "split_sha256": data_hashes if pinned_dataset_dir is not None else {},
            "events": dataset_cache_key_events,
            "dataset_restore_events": dataset_restore_events,
        },
    }
    return evidence, restore_runtime_method


def _install_single_seed_training(run_dir: Path, selected_seed: int) -> tuple[dict[str, Any], Any]:
    """Run one declared source seed in a fresh process, as the historical logs do."""
    from pyabsa.framework.trainer_class.trainer_template import Trainer

    original = Trainer._run
    source_name = inspect.getsourcefile(original)
    if not source_name:
        raise RuntimeError("could not locate PyABSA Trainer._run source")
    source_path = Path(source_name).resolve()
    snapshot = run_dir / "provenance" / "runtime" / "pyabsa_trainer_template.py"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, snapshot)
    source_hash = _sha256(source_path)
    if _sha256(snapshot) != source_hash:
        raise RuntimeError("PyABSA Trainer._run source snapshot checksum mismatch")

    events: list[dict[str, Any]] = []

    def run_one_seed(instructor: Any, *args: Any, **kwargs: Any) -> Any:
        original_seeds = getattr(instructor.config, "seed", None)
        declared = list(original_seeds) if isinstance(original_seeds, (list, tuple)) else [original_seeds]
        if selected_seed not in declared:
            raise RuntimeError(
                f"selected seed {selected_seed} is not in unchanged source declaration {declared!r}"
            )
        instructor.config.seed = [selected_seed]
        events.append({"declared_seeds": declared, "executed_seed": selected_seed})
        try:
            return original(instructor, *args, **kwargs)
        finally:
            instructor.config.seed = original_seeds

    run_one_seed.__name__ = original.__name__
    run_one_seed.__doc__ = original.__doc__
    Trainer._run = run_one_seed

    def restore() -> None:
        if Trainer._run is run_one_seed:
            Trainer._run = original

    evidence = {
        "mode": "fresh_process_per_declared_seed",
        "method": "pyabsa.framework.trainer_class.trainer_template.Trainer._run",
        "selected_seed": selected_seed,
        "runtime_source_path": str(source_path),
        "runtime_source_sha256": source_hash,
        "runtime_source_snapshot": snapshot.relative_to(run_dir).as_posix(),
        "events": events,
        "behavior": (
            "The unchanged source script declares seeds [1, 2, 3]. The historical local logs show one seed start and one metric record per fresh process. "
            "This runtime wrapper narrows only the Trainer._run seed loop to the currently declared seed; each seed runs in a separate process."
        ),
    }
    return evidence, restore


def _get_score_series(metric_visualizer: Any, expected_seeds: list[int] | None = None) -> list[float]:
    expected_seeds = SEEDS if expected_seeds is None else expected_seeds
    metrics = getattr(metric_visualizer, "metrics", None)
    if not isinstance(metrics, dict):
        raise RuntimeError("MetricVisualizer did not expose a metrics mapping")
    max_test = metrics.get("Max-Test-F1")
    if not isinstance(max_test, dict):
        raise RuntimeError("source script did not record MetricVisualizer Max-Test-F1")
    trial = max_test.get("trial0")
    if not isinstance(trial, (list, tuple)) or len(trial) != len(expected_seeds):
        raise RuntimeError(
            f"expected one Max-Test-F1 outcome for each of seeds {expected_seeds}; observed {trial!r}"
        )
    values = []
    for item in trial:
        value = float(item)
        if not math.isfinite(value):
            raise RuntimeError("MetricVisualizer returned a non-finite Max-Test-F1")
        # PyABSA releases have emitted both fractions and percentage values.
        values.append(value * 100.0 if value <= 1.0 else value)
    return values


def _checkpoint_max_test_metrics(
    run_dir: Path,
    expected_seeds: list[int],
    expected_model_id: str,
) -> list[dict[str, Any]]:
    """Recover per-seed Max-Test-F1 from PyABSA's saved best-checkpoint args.

    PyABSA can finish training and save a best checkpoint for every seed while
    MetricVisualizer leaves its in-memory ``trial0`` series empty. Checkpoint
    ``max_test_metrics`` is the same source metric used to name/select each
    legacy checkpoint, so it is durable evidence for that run's test-selected
    score.
    """
    candidates: dict[int, list[dict[str, Any]]] = {seed: [] for seed in expected_seeds}
    for args_path in sorted(run_dir.glob("checkpoints/**/emcgcn.args.txt")):
        fields: dict[str, str] = {}
        for line in args_path.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip() in {
                "model", "seed", "pretrained_bert", "num_epoch",
                "log_step", "metrics_of_this_checkpoint", "max_test_metrics",
            }:
                fields[key.strip()] = value.strip()
        try:
            seed = int(ast.literal_eval(fields["seed"]))
            model_id = fields["pretrained_bert"].strip()
            model_name = fields["model"]
            epoch_count = int(ast.literal_eval(fields["num_epoch"]))
            checkpoint_log_step = int(ast.literal_eval(fields["log_step"]))
            checkpoint_metrics = ast.literal_eval(fields["metrics_of_this_checkpoint"])
            max_test_metrics = ast.literal_eval(fields["max_test_metrics"])
            raw_score = float(max_test_metrics["max_apc_test_f1"])
        except (KeyError, TypeError, ValueError, SyntaxError) as error:
            raise RuntimeError(f"could not parse checkpoint metric evidence at {args_path}") from error
        if seed not in candidates:
            continue
        if model_id != expected_model_id or "EMCGCN" not in model_name:
            raise RuntimeError(f"checkpoint identity mismatch at {args_path}")
        if epoch_count != 10:
            raise RuntimeError(f"checkpoint does not record the full 10-epoch setting at {args_path}")
        if not math.isfinite(raw_score) or not 0.0 <= raw_score <= 1.0:
            raise RuntimeError(f"checkpoint Max-Test-F1 is invalid at {args_path}")
        reported_f1 = float(checkpoint_metrics["f1"])
        if not math.isclose(raw_score, reported_f1, rel_tol=0, abs_tol=1e-12):
            raise RuntimeError(f"checkpoint score and Max-Test-F1 disagree at {args_path}")
        state_dict = args_path.with_name("emcgcn.state_dict")
        if not state_dict.is_file():
            raise RuntimeError(f"checkpoint state dict is missing beside {args_path}")
        label_match = re.search(r"_f1_([0-9]+(?:\.[0-9]+)?)$", args_path.parent.name)
        score_pct = raw_score * 100.0
        if label_match and not math.isclose(
            float(label_match.group(1)), score_pct, rel_tol=0, abs_tol=0.011
        ):
            raise RuntimeError(f"checkpoint directory score label disagrees at {args_path}")
        candidates[seed].append(
            {
                "seed": seed,
                "max_test_f1_fraction": raw_score,
                "max_test_f1_pct": score_pct,
                "log_step": checkpoint_log_step,
                "checkpoint_directory": args_path.parent.relative_to(run_dir).as_posix(),
                "artifact_sha256": {
                    args_path.relative_to(run_dir).as_posix(): _sha256(args_path),
                    state_dict.relative_to(run_dir).as_posix(): _sha256(state_dict),
                },
            }
        )

    recovered = []
    for seed in expected_seeds:
        seed_candidates = candidates[seed]
        if not seed_candidates:
            raise RuntimeError(f"no saved EMCGCN Max-Test-F1 checkpoint found for seed {seed}")
        # PyABSA may retain multiple improving checkpoints for one seed. Its
        # reported value is the highest test-scoring checkpoint for that seed.
        selected = max(seed_candidates, key=lambda item: item["max_test_f1_fraction"])
        recovered.append(
            {
                **selected,
                "candidate_checkpoints": seed_candidates,
                "selection": "highest saved max_test_metrics.max_apc_test_f1 for this seed",
            }
        )
    return recovered


def recover_failed_reference_run(run_dir: Path) -> dict[str, Any]:
    """Write a separate recovery record without changing a failed run manifest."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_failure = "expected one Max-Test-F1 outcome for each of seeds"
    if (
        manifest.get("experiment_protocol_id") != PROTOCOL_ID
        or manifest.get("status") != "failed"
        or manifest.get("failure_type") != "RuntimeError"
        or expected_failure not in manifest.get("failure", "")
    ):
        raise RuntimeError("run is not a reference attempt with only MetricVisualizer score extraction failure")
    metrics = _checkpoint_max_test_metrics(
        run_dir,
        list(manifest.get("seeds", SEEDS)),
        manifest["model"]["repository"],
    )
    recovered_metrics = {
        "run_id": manifest["run_id"],
        "metric": "PyABSA EMCGCN checkpoint max_test_metrics.max_apc_test_f1",
        "unit": "percent",
        "selection_boundary": "test set used during training for checkpoint selection; test-leaked reference only",
        "score_source": "saved per-seed best-checkpoint emcgcn.args.txt metadata",
        "seeds": [
            {
                "seed": item["seed"],
                "max_test_f1_pct": item["max_test_f1_pct"],
                "checkpoint_directory": item["checkpoint_directory"],
            }
            for item in metrics
        ],
    }
    recovery = {
        "schema_version": 1,
        "recovery_status": "checkpoint_scores_recovered; original run manifest remains failed",
        "recovered_at": _now(),
        "run_id": manifest["run_id"],
        "original_manifest_sha256": _sha256(manifest_path),
        "original_manifest_status": manifest["status"],
        "original_manifest_failure": manifest["failure"],
        "experiment_protocol_id": manifest["experiment_protocol_id"],
        "encoder": manifest["encoder"],
        "model": manifest["model"],
        "execution_backend": manifest.get("execution_backend"),
        "execution_backend_metadata": manifest.get("execution_backend_metadata"),
        "seed_metrics": metrics,
        "score_recovery_method": "parse saved PyABSA best-checkpoint max_test_metrics.max_apc_test_f1; require exact seed coverage, matching model identity, 10-epoch config, checkpoint state dict, and agreement with metrics_of_this_checkpoint.f1",
    }
    metrics_path = run_dir / "reference-metrics-recovered.json"
    recovery_path = run_dir / "checkpoint-metric-recovery.json"
    if metrics_path.exists() or recovery_path.exists():
        raise FileExistsError("checkpoint-metric recovery artifacts already exist; refusing to overwrite them")
    _atomic_json(metrics_path, recovered_metrics)
    _atomic_json(recovery_path, recovery)
    return recovery


def _resolved_config(
    namespace: dict[str, Any],
    expected: dict[str, Any],
    log_path: Path,
    run_dir: Path,
    expected_seeds: list[int] | None = None,
) -> dict[str, Any]:
    config = namespace.get("config")
    trainer = namespace.get("trainer")
    if config is None or trainer is None:
        raise RuntimeError("source script returned without its config and trainer objects")

    actual = {
        "model": str(getattr(config, "model", "")),
        "pretrained_bert": str(getattr(config, "pretrained_bert", "")),
        "batch_size": getattr(config, "batch_size", None),
        "patience": getattr(config, "patience", None),
        "log_step": getattr(config, "log_step", None),
        "seed": list(getattr(config, "seed", [])),
        "verbose": getattr(config, "verbose", None),
        "notice": getattr(config, "notice", None),
        "overwrite_cache": getattr(config, "overwrite_cache", False),
        "gcn_dropout": getattr(config, "gcn_dropout", None),
        "num_epoch": getattr(config, "num_epoch", None),
        "max_seq_len": getattr(config, "max_seq_len", None),
        "learning_rate": getattr(config, "learning_rate", None),
        "load_aug": True,
        "auto_device": "PyABSA DeviceTypeOption.AUTO",
        "checkpoint_save_mode": "PyABSA ModelSaveOption.SAVE_MODEL_STATE_DICT",
    }
    expected_config = expected["settings"]
    log_step_resolution = None
    for key, expected_value in expected_config.items():
        observed = actual.get(key)
        if key == "log_step" and expected_value == -1 and isinstance(observed, int) and observed > 0:
            log_step_resolution = _resolve_log_step_sentinel(
                trainer, expected, observed, log_path, run_dir, expected_seeds
            )
            continue
        if isinstance(expected_value, float) and observed is not None:
            matches = math.isclose(float(observed), expected_value, rel_tol=0, abs_tol=1e-12)
        else:
            matches = observed == expected_value
        if not matches:
            raise RuntimeError(
                f"source config mismatch for {key}: expected {expected_value!r}, observed {observed!r}"
            )
    if log_step_resolution is not None:
        actual["log_step_resolution"] = log_step_resolution
    return actual


def run_reference_baseline(
    encoder: str,
    run_dir: Path,
    reference_repo: Path,
    *,
    seed: int | None = None,
    protocol_id: str = PROTOCOL_ID,
) -> dict[str, Any]:
    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    scripts = _expected_scripts(matrix)
    if encoder not in scripts:
        raise ValueError(f"unknown encoder {encoder!r}")
    if protocol_id not in {PROTOCOL_ID, CACHE_PROCESS_PROTOCOL_ID}:
        raise ValueError(f"unknown source-script protocol {protocol_id!r}")
    if protocol_id == CACHE_PROCESS_PROTOCOL_ID and (seed not in SEEDS):
        raise ValueError(f"isolated source-script run needs one declared seed from {SEEDS!r}")
    if protocol_id == PROTOCOL_ID and seed is not None:
        raise ValueError("single-seed execution requires the cache-process protocol")
    execution_seeds = SEEDS if seed is None else [int(seed)]
    preloaded_cache_hashes: dict[str, str] = {}
    if run_dir.exists():
        preloaded_caches = {
            path.name: _sha256(path)
            for path in run_dir.glob("*.dataset.*.cache")
            if path.is_file()
        }
        preloaded_cache_hashes = preloaded_caches
        unexpected = [
            path.name
            for path in run_dir.iterdir()
            if path.name != "training.log" and path.name not in preloaded_caches
        ]
        if unexpected:
            raise FileExistsError(f"refusing to overwrite non-empty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    spec = scripts[encoder]
    source_path = (reference_repo / spec["path"]).resolve()
    canonical_paths = {split: CANONICAL_DATA / f"{split}.txt" for split in ("train", "dev", "test")}
    source_paths = {split: reference_repo / "datasets" / "aste" / f"{split}.txt" for split in canonical_paths}
    run_data_dir = run_dir / "datasets" / "aste"
    copied_paths = {split: run_data_dir / f"{split}.txt" for split in canonical_paths}

    workspace_status = _git_value(ROOT, "status", "--porcelain") or ""
    reference_revision = _git_value(reference_repo, "rev-parse", "HEAD")
    reference_status = _git_value(reference_repo, "status", "--porcelain")
    code_files = {
        "configs/experiment-matrix.json": MATRIX_PATH,
        "src/aste_repro/reference_baseline.py": Path(__file__).resolve(),
        "src/aste_repro/reference_baseline_queue.py": ROOT / "src" / "aste_repro" / "reference_baseline_queue.py",
        "src/aste_repro/report.py": ROOT / "src" / "aste_repro" / "report.py",
        "src/aste_repro/source_data.py": ROOT / "src" / "aste_repro" / "source_data.py",
    }
    initial = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "status": "running",
        "created_at": _now(),
        "runner_pid": os.getpid(),
        "experiment_protocol_id": protocol_id,
        "execution": {
            "mode": "fresh_process_per_declared_seed" if seed is not None else "single_source_process_all_declared_seeds",
            "selected_seed": seed,
            "declared_source_seeds": list(SEEDS),
        },
        "purpose": "reference_baseline_reproduction",
        "dataset": "hoasa",
        "train_dataset": "hoasa",
        "test_dataset": "hoasa",
        "architecture": "emcgcn",
        "encoder": encoder,
        "seeds": execution_seeds,
        "code": {
            "revision": _git_value(ROOT, "rev-parse", "HEAD"),
            "dirty": bool(workspace_status),
            "dirty_paths": [line[3:] for line in workspace_status.splitlines() if len(line) > 3],
            "file_sha256": {name: _sha256(path) for name, path in code_files.items() if path.is_file()},
            "snapshot_directory": "provenance/workspace-source",
        },
        "reference": {
            "metric": "PyABSA MetricVisualizer Max-Test-F1",
            "checkpoint_selection": "best test-scoring checkpoint during training; legacy test-selected, test-leaked source-script metric",
            "test_selected_checkpoint": True,
            "clean_heldout_estimate": False,
            "invocation": (
                "runpy.run_path(source_script, run_name='__main__') with the PyABSA Trainer._run loop limited by runtime wrapper to the selected declared seed"
                if seed is not None
                else "runpy.run_path(source_script, run_name='__main__') with source training statements unchanged"
            ),
            "repository": str(reference_repo.resolve()),
            "repository_revision": reference_revision,
            "repository_status_available": reference_status is not None,
            "repository_dirty": bool(reference_status) if reference_status is not None else None,
            "repository_dirty_paths": (
                [line[3:] for line in reference_status.splitlines() if len(line) > 3]
                if reference_status is not None
                else None
            ),
            "script": spec["path"],
            "script_path": str(source_path),
            "script_sha256": None,
            "dataset_directory": str(run_data_dir.resolve()),
            "run_dataset_copy_directory": str(run_data_dir.resolve()),
            "source_script_copy": "provenance/source_script.py",
            "source_script_sha256": None,
            "runner_log": "training.log",
            "model_cache": None,
        },
        "model": {"repository": spec["model_id"], "revision": spec["model_revision"]},
        "source_files": {},
        "requested_config": spec["settings"],
        "resolved_config": None,
        "seed_metrics": [],
        "artifact_sha256": {},
    }
    initial["code"]["file_sha256"] = _snapshot_workspace_source(code_files, run_dir)
    runner_working_directory = str(Path.cwd().resolve())
    _atomic_json(manifest_path, initial)

    try:
        if not source_path.is_file():
            raise FileNotFoundError(f"source script not found: {source_path}")
        actual_script_sha = _sha256(source_path)
        if actual_script_sha != spec["sha256"]:
            raise RuntimeError(
                f"source script hash mismatch for {spec['path']}: {actual_script_sha}"
            )
        provenance = run_dir / "provenance"
        # The workspace-source snapshot is recorded before the PyABSA script
        # copy, so this directory already exists in every new run.
        provenance.mkdir(parents=True, exist_ok=True)
        script_copy = provenance / "source_script.py"
        shutil.copyfile(source_path, script_copy)
        initial["reference"]["script_sha256"] = actual_script_sha
        initial["reference"]["source_script_sha256"] = _sha256(script_copy)

        run_data_dir.mkdir(parents=True, exist_ok=False)
        source_data_records: dict[str, Any] = {}
        for split in ("train", "dev", "test"):
            canonical = canonical_paths[split]
            source = source_paths[split]
            copied = copied_paths[split]
            if not canonical.is_file() or not source.is_file():
                raise FileNotFoundError(f"required {split} data source is missing")
            shutil.copyfile(source, copied)
            canonical_rows = _parsed_rows(canonical)
            reference_rows = _parsed_rows(source)
            run_rows = _parsed_rows(copied)
            source_data_records[split] = {
                "canonical_path": str(canonical.resolve()),
                "canonical_sha256": _sha256(canonical),
                "reference_path": str(source.resolve()),
                "reference_sha256": _sha256(source),
                "run_copy_path": str(copied.relative_to(run_dir).as_posix()),
                "run_copy_sha256": _sha256(copied),
                "parsed_lines_equal": canonical_rows == reference_rows == run_rows,
                "rows": len(run_rows),
            }
            if not source_data_records[split]["parsed_lines_equal"]:
                raise RuntimeError(f"canonical and source-script {split} rows differ")
        initial["source_files"] = source_data_records

        try:
            from huggingface_hub import snapshot_download
        except ImportError as error:
            raise RuntimeError("huggingface_hub is unavailable in the source runtime") from error
        snapshot = Path(
            snapshot_download(
                repo_id=spec["model_id"],
                revision=spec["model_revision"],
                local_files_only=True,
            )
        ).resolve()
        if snapshot.name != spec["model_revision"]:
            raise RuntimeError("resolved model cache directory does not match the pinned revision")
        small_model_files = {}
        for path in sorted(item for item in snapshot.rglob("*") if item.is_file()):
            rel = path.relative_to(snapshot).as_posix()
            if path.name in {
                "config.json", "tokenizer.json", "tokenizer_config.json",
                "special_tokens_map.json", "added_tokens.json", "vocab.txt",
                "vocab.json", "merges.txt", "spiece.model", "sentencepiece.bpe.model",
            }:
                small_model_files[rel] = _sha256(path)
        if not small_model_files:
            raise RuntimeError("pinned model cache has no tokenizer/config provenance files")
        initial["reference"]["model_cache"] = {
            "snapshot_path": str(snapshot),
            "revision_directory": snapshot.name,
            "small_file_sha256": small_model_files,
        }
        initial["resolved_config"] = {
            "python": platform.python_version(),
            "packages": _package_versions(),
            "requested_settings": spec["settings"],
            "actual_settings": None,
        }
        _atomic_json(manifest_path, initial)

        # A cache hit can replace the current config with a serialized config
        # whose seed belongs to the previous training pass. Preserve the seed
        # selected by PyABSA's unchanged Trainer._run loop, and record the
        # installed cache-loader source plus every correction in the manifest.
        runtime_seed_cache, restore_runtime_method = _install_seed_cache_compatibility(run_dir)
        initial["reference"]["runtime_seed_cache_compatibility"] = runtime_seed_cache
        single_seed_runtime = None
        restore_single_seed_runtime = None
        if seed is not None:
            single_seed_runtime, restore_single_seed_runtime = _install_single_seed_training(run_dir, seed)
            initial["reference"]["runtime_single_seed_execution"] = single_seed_runtime
        _atomic_json(manifest_path, initial)

        # The source uses PyABSA's date-based logger directory. A stable label
        # avoids dated output names without changing training or checkpoint settings.
        from pyabsa.utils.logger import logger

        logger.today = "reference-baseline"
        try:
            namespace = runpy.run_path(str(source_path), run_name="__main__")
        finally:
            initial["reference"]["runtime_seed_cache_compatibility"] = runtime_seed_cache
            restore_runtime_method()
            if restore_single_seed_runtime is not None:
                restore_single_seed_runtime()
        actual_config = _resolved_config(
            namespace, spec, run_dir / "training.log", run_dir, execution_seeds
        )
        if single_seed_runtime is not None:
            single_seed_runtime["events"] = list(single_seed_runtime.get("events", []))
        cache_loads = re.findall(
            r"Load cache dataset from ([^\r\n]+\.cache)",
            (run_dir / "training.log").read_text(encoding="utf-8", errors="replace"),
        )
        cache_saves = re.findall(
            r"Save cache dataset to ([^\r\n]+\.cache)",
            (run_dir / "training.log").read_text(encoding="utf-8", errors="replace"),
        )
        dataset_cache_files = sorted(
            {
                path.name: _sha256(path)
                for path in run_dir.glob("*.dataset.*.cache")
                if path.is_file()
            }.items()
        )
        initial["reference"]["dataset_cache_evidence"] = {
            "cache_key_excludes_seed": True,
            "cache_load_events": cache_loads,
            "cache_save_events": cache_saves,
            "cache_file_sha256": dict(dataset_cache_files),
            "preloaded_cache_files": preloaded_cache_hashes if seed is not None else {},
            "cache_hit_required": bool(
                seed is not None
                and preloaded_cache_hashes
                and not spec["settings"].get("overwrite_cache", False)
            ),
        }
        for cache_name, expected_hash in preloaded_cache_hashes.items():
            cache_path = run_dir / cache_name
            if not cache_path.is_file() or _sha256(cache_path) != expected_hash:
                raise RuntimeError(f"preloaded dataset cache changed during source execution: {cache_name}")
        if initial["reference"]["dataset_cache_evidence"]["cache_hit_required"]:
            loaded_names = {Path(value.strip()).name for value in cache_loads}
            preloaded_names = set(
                initial["reference"]["dataset_cache_evidence"]["preloaded_cache_files"]
            )
            if not preloaded_names or not (loaded_names & preloaded_names):
                raise RuntimeError(
                    "isolated seed process did not load a predeclared cache file shared from seed 1"
                )
        metric_visualizer = namespace.get("mv")
        try:
            values = _get_score_series(metric_visualizer, execution_seeds)
        except RuntimeError as metric_visualizer_error:
            checkpoint_metrics = _checkpoint_max_test_metrics(
                run_dir,
                execution_seeds,
                spec["model_id"],
            )
            values = [item["max_test_f1_pct"] for item in checkpoint_metrics]
            initial["reference"]["metric_score_source"] = {
                "method": "saved_per_seed_best_checkpoint_args",
                "metric": "max_test_metrics.max_apc_test_f1",
                "metric_visualizer_error": str(metric_visualizer_error),
                "seed_metrics": checkpoint_metrics,
            }
        seed_metrics = [
            {"seed": seed, "max_test_f1_pct": score}
            for seed, score in zip(execution_seeds, values)
        ]
        metrics = {
            "metric": "PyABSA MetricVisualizer Max-Test-F1",
            "unit": "percent",
            "selection_boundary": "test set used during training for checkpoint selection; test-leaked reference only",
            "score_source": initial["reference"].get("metric_score_source", {"method": "MetricVisualizer.metrics['Max-Test-F1']['trial0']"}),
            "seeds": seed_metrics,
        }
        _atomic_json(run_dir / "reference_metrics.json", metrics)
        initial["resolved_config"]["actual_settings"] = actual_config
        initial["seed_metrics"] = seed_metrics
        initial["completed_at"] = _now()
        initial["status"] = "complete"
        initial["environment"] = {
            "python": platform.python_version(),
            "packages": _package_versions(),
            "platform": platform.platform(),
            "working_directory": runner_working_directory,
            "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
            "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE"),
            "matplotlib_backend": os.environ.get("MPLBACKEND"),
        }
        artifact_hashes = {}
        for path in sorted(item for item in run_dir.rglob("*") if item.is_file() and item != manifest_path):
            artifact_hashes[path.relative_to(run_dir).as_posix()] = _sha256(path)
        initial["artifact_sha256"] = artifact_hashes
        _atomic_json(manifest_path, initial)
        return initial
    except BaseException as error:
        initial.update(
            {
                "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                "completed_at": _now(),
                "failure_type": type(error).__name__,
                "failure": str(error)[:2000] or repr(error)[:2000],
            }
        )
        try:
            _atomic_json(manifest_path, initial)
        except Exception:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reference-repo", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--protocol-id", default=PROTOCOL_ID)
    args = parser.parse_args()
    try:
        run_reference_baseline(
            args.encoder,
            args.run_dir.resolve(),
            args.reference_repo.resolve(),
            seed=args.seed,
            protocol_id=args.protocol_id,
        )
    except KeyboardInterrupt:
        print("reference source run interrupted; partial evidence preserved", file=sys.stderr, flush=True)
        return 130
    except Exception as error:
        print(f"reference source run failed: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
