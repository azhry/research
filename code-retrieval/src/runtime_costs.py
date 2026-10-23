"""Measurement helpers for reproducible local experiment runtime costs."""

from __future__ import annotations

import ctypes
import importlib.util
import os
import platform
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping


def _process_rss_bytes() -> int | None:
    """Return the current process resident working set without adding a dependency."""

    if os.name == "nt":
        try:
            from ctypes import wintypes

            class ProcessMemoryCountersEx(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCountersEx()
            counters.cb = ctypes.sizeof(counters)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            process = kernel32.GetCurrentProcess()
            get_info = psapi.GetProcessMemoryInfo
            get_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(ProcessMemoryCountersEx), wintypes.DWORD]
            get_info.restype = wintypes.BOOL
            if get_info(process, ctypes.byref(counters), counters.cb):
                return int(counters.WorkingSetSize)
        except (AttributeError, OSError, TypeError, ValueError):
            return None
        return None

    statm = "/proc/self/statm"
    if os.path.exists(statm):
        try:
            resident_pages = int(Path(statm).read_text(encoding="ascii").split()[1])
            return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
        except (OSError, IndexError, ValueError):
            return None

    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except (ImportError, OSError, ValueError):
        pass

    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports KiB; macOS and other BSDs report bytes.
        return value if platform.system() == "Darwin" else value * 1024
    except (ImportError, OSError, ValueError):
        return None


def _process_memory_method() -> str:
    if os.name == "nt":
        return "sampled process working set"
    if os.path.exists("/proc/self/statm"):
        return "sampled process resident set size from procfs"
    if importlib.util.find_spec("psutil") is not None:
        return "sampled process resident set size via psutil"
    return "process lifetime peak resident set size via resource"


def _host_memory_total_bytes() -> int | None:
    if os.name == "nt":
        try:
            from ctypes import wintypes

            class MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", wintypes.DWORD),
                    ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatusEx()
            status.dwLength = ctypes.sizeof(status)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            get_status = kernel32.GlobalMemoryStatusEx
            get_status.argtypes = [ctypes.POINTER(MemoryStatusEx)]
            get_status.restype = wintypes.BOOL
            if get_status(ctypes.byref(status)):
                return int(status.ullTotalPhys)
        except (AttributeError, OSError, TypeError, ValueError):
            return None
        return None
    try:
        return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return None


def synchronize_device(device: str) -> None:
    """Synchronize CUDA work around a timed call; CPU calls remain synchronous."""

    if str(device).startswith("cuda"):
        import torch

        torch.cuda.synchronize(device)


@contextmanager
def observe_module_batches(
    module: Any,
    *,
    device: str,
    stage: str,
    observer: Any,
    unit: str,
    fallback_batch_size: int,
) -> Iterator[None]:
    """Observe model forward batches without changing encode/predict batching."""

    if observer is None:
        yield
        return

    starts: list[tuple[float, int]] = []

    def batch_size_from(args: Any, kwargs: Any) -> int:
        features = args[0] if args else kwargs
        shape = getattr(features, "shape", None)
        if shape is not None and len(shape) > 0:
            return int(shape[0])
        if isinstance(features, Mapping):
            for value in features.values():
                shape = getattr(value, "shape", None)
                if shape is not None and len(shape) > 0:
                    return int(shape[0])
        return int(fallback_batch_size)

    def before_forward(_module: Any, args: Any, kwargs: Any) -> None:
        synchronize_device(device)
        starts.append((time.perf_counter(), batch_size_from(args, kwargs)))

    def after_forward(_module: Any, _args: Any, _kwargs: Any, output: Any) -> Any:
        synchronize_device(device)
        if starts:
            started, count = starts.pop()
            observer(
                stage=stage,
                seconds=time.perf_counter() - started,
                unit_count=count,
                unit=unit,
            )
        return output

    before_handle = module.register_forward_pre_hook(before_forward, with_kwargs=True)
    after_handle = module.register_forward_hook(after_forward, with_kwargs=True)
    try:
        yield
    finally:
        before_handle.remove()
        after_handle.remove()


class RuntimeCostRecorder:
    """Collect actual stage timings, model-batch timings, and sampled process RSS."""

    def __init__(self, *, sample_interval_seconds: float = 0.05) -> None:
        if sample_interval_seconds <= 0:
            raise ValueError("sample_interval_seconds must be positive")
        self.sample_interval_seconds = sample_interval_seconds
        self.started_at: float | None = None
        self.stopped_at: float | None = None
        self.start_rss_bytes: int | None = None
        self.end_rss_bytes: int | None = None
        self.peak_rss_bytes: int | None = None
        self.memory_sample_count = 0
        self.phase_timings: list[dict[str, Any]] = []
        self.cache_stages: list[dict[str, Any]] = []
        self.batch_observations: list[dict[str, Any]] = []
        self.gpu_peak_allocated_bytes: int | None = None
        self.gpu_peak_reserved_bytes: int | None = None
        self.gpu_memory_reason: str | None = "CUDA was not selected"
        self._stop_event = threading.Event()
        self._sampler_thread: threading.Thread | None = None

    def __enter__(self) -> "RuntimeCostRecorder":
        self.started_at = time.perf_counter()
        self.start_rss_bytes = _process_rss_bytes()
        self.peak_rss_bytes = self.start_rss_bytes
        self._sampler_thread = threading.Thread(target=self._sample_memory, daemon=True)
        self._sampler_thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop()

    def _sample_memory(self) -> None:
        while not self._stop_event.wait(self.sample_interval_seconds):
            rss = _process_rss_bytes()
            if rss is not None:
                self.memory_sample_count += 1
                if self.peak_rss_bytes is None or rss > self.peak_rss_bytes:
                    self.peak_rss_bytes = rss

    def stop(self) -> None:
        if self.stopped_at is not None:
            return
        self.stopped_at = time.perf_counter()
        self._stop_event.set()
        if self._sampler_thread is not None:
            self._sampler_thread.join(timeout=max(1.0, self.sample_interval_seconds * 4))
        self.end_rss_bytes = _process_rss_bytes()
        if self.end_rss_bytes is not None:
            self.memory_sample_count += 1
            if self.peak_rss_bytes is None or self.end_rss_bytes > self.peak_rss_bytes:
                self.peak_rss_bytes = self.end_rss_bytes

    @contextmanager
    def phase(self, name: str, category: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.phase_timings.append(
                {"name": name, "category": category, "seconds": time.perf_counter() - started}
            )

    def record_cache_stage(
        self,
        *,
        name: str,
        category: str,
        cache_hit: bool,
        load_seconds: float,
        compute_seconds: float,
        save_seconds: float,
        item_count: int,
    ) -> None:
        self.cache_stages.append(
            {
                "name": name,
                "category": category,
                "cache_status": "hit" if cache_hit else "miss",
                "cache_load_seconds": float(load_seconds),
                "compute_seconds": float(compute_seconds),
                "cache_save_seconds": float(save_seconds),
                "total_seconds": float(load_seconds + compute_seconds + save_seconds),
                "item_count": int(item_count),
            }
        )

    def observe_batch(
        self,
        *,
        stage: str,
        seconds: float,
        unit_count: int,
        unit: str,
    ) -> None:
        if seconds < 0 or unit_count < 0:
            raise ValueError("batch timing and unit count must be non-negative")
        self.batch_observations.append(
            {
                "stage": stage,
                "seconds": float(seconds),
                "unit_count": int(unit_count),
                "unit": unit,
            }
        )

    def set_cuda_device(self, torch_module: Any, device: str) -> None:
        if not str(device).startswith("cuda"):
            self.gpu_memory_reason = "CUDA was not selected"
            return
        try:
            torch_module.cuda.reset_peak_memory_stats(device)
            self.gpu_memory_reason = None
        except (AttributeError, RuntimeError, TypeError):
            self.gpu_memory_reason = "CUDA peak-memory counters were unavailable"

    def _gpu_memory(self, environment: Mapping[str, Any]) -> None:
        if not str(environment.get("device") or "").startswith("cuda"):
            self.gpu_memory_reason = "Measured model device was CPU"
            return
        if not environment.get("cuda_available"):
            self.gpu_memory_reason = "CUDA was not available in the measured environment"
            return
        try:
            import torch

            if torch.cuda.is_available():
                device = str(environment.get("device") or "cuda")
                self.gpu_peak_allocated_bytes = int(torch.cuda.max_memory_allocated(device))
                self.gpu_peak_reserved_bytes = int(torch.cuda.max_memory_reserved(device))
                self.gpu_memory_reason = None
        except (ImportError, RuntimeError, TypeError):
            self.gpu_memory_reason = "CUDA peak-memory counters could not be read"

    def summary(
        self,
        *,
        run_id: str,
        cache_identity: str,
        status: str,
        run_mode: str,
        environment: Mapping[str, Any],
        config: Any,
        query_count: int,
        corpus_count: int,
        reranked_pair_count: int,
        model_precision: Mapping[str, str | None],
        offline_index_build_seconds: float,
    ) -> dict[str, Any]:
        self.stop()
        self._gpu_memory(environment)
        elapsed = max(0.0, float(self.stopped_at - (self.started_at or self.stopped_at)))
        by_stage: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in self.batch_observations:
            by_stage[item["stage"]].append(item)
        latency_summaries: dict[str, dict[str, Any]] = {}
        for stage, observations in by_stage.items():
            values = sorted(item["seconds"] for item in observations)
            total_units = sum(item["unit_count"] for item in observations)
            total_seconds = sum(values)
            latency_summaries[stage] = {
                "sample_count": len(values),
                "unit": observations[0]["unit"],
                "unit_count": total_units,
                "min_seconds": values[0],
                "p50_seconds": _quantile(values, 0.50),
                "p95_seconds": _quantile(values, 0.95),
                "max_seconds": values[-1],
                "throughput_units_per_second": (
                    total_units / total_seconds if total_seconds > 0 else None
                ),
                "distribution_scope": "observed model or retrieval call batches; sample count is reported",
            }

        online_categories = {"online_query", "query_search", "query_generation", "query_reranking"}
        online_seconds = sum(
            item["total_seconds"]
            for item in self.cache_stages
            if item["category"] in online_categories
        )
        online_seconds += sum(
            item["seconds"]
            for item in self.phase_timings
            if item["category"] in online_categories
        )
        # The cache-stage wall time for the first uncached ranking includes the
        # lazy index build. That work is reported as offline preprocessing.
        online_seconds = max(0.0, online_seconds - float(offline_index_build_seconds))
        total_query_units = int(query_count)
        process_method = _process_memory_method()
        evidence_level = (
            "smoke"
            if run_mode == "smoke"
            else status
            if status in {"cache_replay", "blocked"}
            else run_mode
        )
        return {
            "schema_version": 1,
            "run_id": run_id,
            "cache_identity": cache_identity,
            "status": status,
            "evidence_level": evidence_level,
            "measurement_controls": {
                "hardware": {
                    "platform": environment.get("platform"),
                    "processor": environment.get("processor"),
                    "logical_cpu_count": os.cpu_count(),
                    "host_total_memory_bytes": _host_memory_total_bytes(),
                },
                "device": environment.get("device"),
                "cuda_device_name": environment.get("cuda_device_name"),
                "precision": dict(model_precision),
                "batch_size": int(config.batch_size),
                "reranker_batch_size": int(config.reranker_batch_size),
                "torch_num_threads": environment.get("torch_num_threads"),
                "torch_default_dtype": environment.get("torch_default_dtype"),
                "torch_float32_matmul_precision": environment.get("torch_float32_matmul_precision"),
                "candidate_depth": int(config.candidate_depth),
                "candidate_count": int(reranked_pair_count),
                "candidate_count_definition": "total query-candidate pairs passed to the cross-encoder",
                "query_count": int(query_count),
                "corpus_count": int(corpus_count),
                "max_seq_length": int(config.max_seq_length),
                "reranker_max_length": int(config.reranker_max_length),
                "experiment_settings": config.as_dict(),
                "package_versions": dict(environment.get("packages", {})),
                "git_commit": environment.get("git_commit"),
            },
            "measurement_protocol": {
                "clock": "time.perf_counter",
                "warmup_runs": 0,
                "measured_pipeline_runs": 1,
                "repetition_policy": "one execution of each uncached pipeline stage; actual model batches are individual latency samples",
                "timing_synchronization": "CPU calls are synchronous; torch.cuda.synchronize is used around timed model batches when the model device is CUDA",
                "memory_sample_interval_ms": int(self.sample_interval_seconds * 1000),
                "cache_policy": "cache load, actual compute, and cache save are measured separately; cache hits are not counted as inference",
                "cold_warm_definition": "model initialization is reported separately; first uncached inference is measured without an added warm-up; cache-backed replay reports cache-load time",
            },
            "pipeline_wall_seconds": elapsed,
            "phase_timings": list(self.phase_timings),
            "cache_stages": list(self.cache_stages),
            "batch_observations": list(self.batch_observations),
            "latency_summaries": latency_summaries,
            "offline_costs": {
                "corpus_encoding_seconds": sum(
                    item["compute_seconds"]
                    for item in self.cache_stages
                    if item["category"] == "offline_corpus_encoding"
                ),
                "corpus_cache_load_seconds": sum(
                    item["cache_load_seconds"]
                    for item in self.cache_stages
                    if item["category"] == "offline_corpus_encoding"
                ),
                "corpus_cache_save_seconds": sum(
                    item["cache_save_seconds"]
                    for item in self.cache_stages
                    if item["category"] == "offline_corpus_encoding"
                ),
                "faiss_index_build_seconds": float(offline_index_build_seconds),
                "index_build_count": sum(
                    1 for item in self.phase_timings if item["name"] == "faiss_index_build"
                ),
                "classification": "offline preprocessing; no persistent Faiss index is saved by this experiment",
            },
            "end_to_end_query_cost": {
                "online_stage_seconds": float(online_seconds),
                "online_seconds_per_query": (
                    float(online_seconds / total_query_units) if total_query_units else None
                ),
                "online_queries_per_second": (
                    float(total_query_units / online_seconds) if online_seconds > 0 else None
                ),
                "full_pipeline_seconds_per_query": (
                    float(elapsed / total_query_units) if total_query_units else None
                ),
                "query_count": total_query_units,
                "definition": "online stage sum excludes corpus embedding and Faiss index construction; full-pipeline value includes startup, offline preprocessing, evaluation, and primary artifact writes",
            },
            "memory": {
                "process_rss_start_bytes": self.start_rss_bytes,
                "process_peak_rss_bytes": self.peak_rss_bytes,
                "process_rss_end_bytes": self.end_rss_bytes,
                "peak_delta_from_run_start_bytes": (
                    max(0, self.peak_rss_bytes - self.start_rss_bytes)
                    if self.peak_rss_bytes is not None and self.start_rss_bytes is not None
                    else None
                ),
                "sample_count": int(self.memory_sample_count),
                "method": process_method,
                "limitation": "process memory only; sampled peak can miss transients shorter than the sample interval and is not total host utilization",
                "gpu_peak_allocated_bytes": self.gpu_peak_allocated_bytes,
                "gpu_peak_reserved_bytes": self.gpu_peak_reserved_bytes,
                "gpu_memory_reason": self.gpu_memory_reason,
            },
        }


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile needs at least one value")
    index = max(0, min(len(values) - 1, int((len(values) - 1) * probability + 0.5)))
    return float(values[index])
