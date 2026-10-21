"""Optional Modal entrypoint for the existing ASTE experiment runner."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import modal


REMOTE_PROJECT_ROOT = PurePosixPath("/opt/aste")
REMOTE_RUNS_ROOT = PurePosixPath("/mnt/aste-research-runs")


def _resolve_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent

    remote_root = Path(str(REMOTE_PROJECT_ROOT))
    if (remote_root / "pyproject.toml").is_file():
        return remote_root

    raise RuntimeError("Could not locate the ASTE project root")


PROJECT_ROOT = _resolve_project_root()
REPOSITORY_ROOT = PROJECT_ROOT.parent
RUNS_ROOT = PROJECT_ROOT / "runs"
RUNS_VOLUME_NAME = "aste-research-runs"
CORE_CODE_FILES = (
    "__init__.py",
    "training.py",
    "modal_runner.py",
    "hoasa.py",
    "metrics.py",
    "pyabsa_compat.py",
    "source_data.py",
    "ssgcn.py",
)
RUNS_VOLUME = modal.Volume.from_name(RUNS_VOLUME_NAME, create_if_missing=True)
app = modal.App("aste-reproduction")


def _required_source_files() -> list[Path]:
    relative_paths = [
        "data/sources/hoasa/train.txt",
        "data/sources/hoasa/dev.txt",
        "data/sources/hoasa/test.txt",
        "data/sources/casa/train_preprocess.csv",
        "data/sources/casa/valid_preprocess.csv",
        "data/sources/casa/test_preprocess.csv",
        "docs/thesis/data/labelstudio_import/casa_train.json",
        "docs/thesis/data/labelstudio_import/casa_valid.json",
        "docs/thesis/data/labelstudio_import/casa_test.json",
    ]
    return [PROJECT_ROOT / relative for relative in relative_paths]


def _parser_dir() -> Path:
    matrix_path = PROJECT_ROOT / "configs" / "experiment-matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    parser_version = matrix["parser"]["version"]
    return PROJECT_ROOT / "data" / "parser" / f"stanza-{parser_version}"


def _build_image() -> modal.Image:
    image = (
        modal.Image.debian_slim(python_version="3.10")
        .apt_install("git")
        .uv_sync(
            uv_project_dir=str(PROJECT_ROOT),
            extras=["modal"],
            frozen=True,
            uv_version="0.10.10",
        )
        # Keep the project lock and local runner CPU-only; replace only the
        # Modal image's Torch wheel with the same release built for CUDA 12.1.
        .uv_pip_install(
            "torch==2.5.1+cu121",
            index_url="https://download.pytorch.org/whl/cu121",
            extra_index_url="https://pypi.org/simple",
            extra_options="--index-strategy unsafe-best-match",
            uv_version="0.10.10",
        )
        .env({"PYTHONPATH": f"{REMOTE_PROJECT_ROOT}/src"})
    )

    for filename in CORE_CODE_FILES:
        local_path = PROJECT_ROOT / "src" / "aste_repro" / filename
        if local_path.is_file():
            remote_path = REMOTE_PROJECT_ROOT / "src" / "aste_repro" / filename
            image = image.add_local_file(str(local_path), remote_path=str(remote_path))

    project_files = [
        (PROJECT_ROOT / "pyproject.toml", REMOTE_PROJECT_ROOT / "pyproject.toml"),
        (PROJECT_ROOT / "uv.lock", REMOTE_PROJECT_ROOT / "uv.lock"),
        (
            PROJECT_ROOT / "configs" / "experiment-matrix.json",
            REMOTE_PROJECT_ROOT / "configs" / "experiment-matrix.json",
        ),
    ]
    for local_path, remote_path in project_files:
        if local_path.is_file():
            image = image.add_local_file(str(local_path), remote_path=str(remote_path))

    for local_path in _required_source_files():
        if local_path.is_file():
            remote_path = REMOTE_PROJECT_ROOT / local_path.relative_to(PROJECT_ROOT).as_posix()
            image = image.add_local_file(str(local_path), remote_path=str(remote_path))

    parser_dir = _parser_dir()
    if parser_dir.is_dir():
        image = image.add_local_dir(
            str(parser_dir),
            remote_path=str(REMOTE_PROJECT_ROOT / "data" / "parser" / parser_dir.name),
        )
    return image


def _recover_run_attempt(
    *,
    requested_run_id: str,
    dataset: str,
    architecture: str,
    encoder: str,
    seed: int,
    purpose: str,
) -> tuple[str, str | None, bool]:
    """Return a safe run ID when Modal retries a preempted function input.

    Modal can restart a preempted Function with the same arguments. The first
    attempt may already have created its run directory, so retries must never
    reuse that directory. Completed prior attempts are returned as-is; partial
    attempts are marked interrupted and kept, and a fresh unique ID is used.
    """
    from aste_repro.training import create_run_id

    runs_root = Path(str(REMOTE_RUNS_ROOT))
    expected = {
        "dataset": dataset,
        "architecture": architecture,
        "encoder": encoder,
        "seed": int(seed),
        "purpose": purpose,
    }
    matching: list[tuple[Path, dict[str, Any] | None]] = []
    for candidate in runs_root.iterdir():
        if not candidate.is_dir():
            continue
        manifest_path = candidate / "manifest.json"
        if not manifest_path.is_file():
            if candidate.name == requested_run_id:
                matching.append((candidate, None))
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            if candidate.name == requested_run_id:
                matching.append((candidate, None))
            continue
        metadata = manifest.get("execution_backend_metadata", {})
        if (
            candidate.name != requested_run_id
            and metadata.get("requested_run_id") != requested_run_id
        ):
            continue
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise RuntimeError(
                f"existing Modal run {candidate.name} does not match the requested experiment"
            )
        matching.append((candidate, manifest))

    for candidate, manifest in matching:
        if manifest is not None and manifest.get("status") == "complete":
            return candidate.name, None, True

    recovered_from: str | None = None
    for candidate, manifest in matching:
        if manifest is None:
            recovered_from = recovered_from or candidate.name
            continue
        status = manifest.get("status")
        if status in {"running", "interrupted"}:
            if status == "running":
                manifest.update(
                    {
                        "status": "interrupted",
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                        "failure_type": "ModalPreemptionRetry",
                        "failure": (
                            "Modal restarted this function input after container preemption; "
                            "the partial attempt is preserved and a fresh run ID will be used."
                        ),
                    }
                )
                (candidate / "manifest.json").write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            recovered_from = recovered_from or candidate.name
            continue
        raise RuntimeError(
            f"refusing to retry Modal run {candidate.name} with status {status!r}"
        )

    if matching:
        RUNS_VOLUME.commit()
        fresh_run_id = create_run_id(
            dataset=dataset,
            architecture=architecture,
            encoder=encoder,
            seed=seed,
            purpose=purpose,
        )
        while (runs_root / fresh_run_id).exists():
            fresh_run_id = create_run_id(
                dataset=dataset,
                architecture=architecture,
                encoder=encoder,
                seed=seed,
                purpose=purpose,
            )
        return fresh_run_id, recovered_from, False
    return requested_run_id, None, False


image = _build_image()


def _local_code_state() -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=REPOSITORY_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}
    return {"revision": revision, "dirty": dirty}


def _validate_local_inputs(download_parser: bool) -> None:
    matrix_path = PROJECT_ROOT / "configs" / "experiment-matrix.json"
    if not matrix_path.is_file():
        raise FileNotFoundError(f"missing Modal project input: {matrix_path}")
    required_project_files = [
        PROJECT_ROOT / "pyproject.toml",
        PROJECT_ROOT / "uv.lock",
        matrix_path,
        *(PROJECT_ROOT / "src" / "aste_repro" / filename for filename in CORE_CODE_FILES),
    ]
    missing = [
        path
        for path in [*required_project_files, *_required_source_files()]
        if not path.is_file()
    ]
    parser_dir = _parser_dir()
    if not download_parser and not parser_dir.is_dir():
        missing.append(parser_dir)
    if missing:
        paths = ", ".join(str(path.relative_to(PROJECT_ROOT)) for path in missing)
        raise FileNotFoundError(
            "Modal run is missing local experiment inputs: "
            f"{paths}. Fetch the pinned sources and Label Studio exports, and install the "
            "pinned Indonesian parser before starting a remote run."
        )


def _download_run(run_id: str, *, summary_only: bool = False) -> Path:
    destination = RUNS_ROOT / run_id
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing run artifacts: {destination}")

    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{run_id}-download-", dir=RUNS_ROOT))
    remote_only_files: list[str] = []
    try:
        entries = list(RUNS_VOLUME.iterdir(run_id, recursive=True))
        if not entries:
            raise FileNotFoundError(f"Modal Volume has no artifacts for run {run_id}")

        for entry in entries:
            entry_type = getattr(entry.type, "name", str(entry.type))
            path_parts = PurePosixPath(str(entry.path).lstrip("/")).parts
            if run_id in path_parts:
                run_position = path_parts.index(run_id)
                relative = PurePosixPath(*path_parts[run_position + 1 :])
            else:
                relative = PurePosixPath(*path_parts)
            if ".." in relative.parts:
                raise ValueError(f"unexpected Modal Volume path for run {run_id}: {entry.path}")
            if not relative.parts and entry_type == "DIRECTORY":
                continue
            if not relative.parts:
                raise ValueError(
                    f"unexpected Modal Volume file path for run {run_id}: {entry.path}"
                )

            local_path = staging.joinpath(*relative.parts)
            if entry_type == "DIRECTORY":
                local_path.mkdir(parents=True, exist_ok=True)
                continue
            if entry_type != "FILE":
                raise ValueError(f"unsupported Modal Volume artifact type: {entry_type}")

            if summary_only and relative.parts[0] == "checkpoints":
                remote_only_files.append(relative.as_posix())
                continue

            local_path.parent.mkdir(parents=True, exist_ok=True)
            with local_path.open("wb") as stream:
                for chunk in RUNS_VOLUME.read_file(str(entry.path)):
                    stream.write(chunk)

        manifest_path = staging / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Modal run {run_id} did not persist manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("run_id") != run_id:
            raise ValueError(f"Modal manifest run_id does not match requested run {run_id}")
        if summary_only:
            storage_record = {
                "run_id": run_id,
                "local_artifacts_complete": False,
                "local_artifacts": sorted(
                    path.relative_to(staging).as_posix()
                    for path in staging.rglob("*")
                    if path.is_file()
                ),
                "remote_volume": RUNS_VOLUME_NAME,
                "remote_run_prefix": run_id,
                "remote_only_artifacts": sorted(remote_only_files),
                "remote_artifact_sha256": manifest.get("artifact_sha256", {}),
            }
            (staging / "modal-storage.json").write_text(
                json.dumps(storage_record, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite existing run artifacts: {destination}")
        staging.replace(destination)
        return destination
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


@app.function(
    image=image,
    gpu="T4",
    cpu=4,
    memory=16_384,
    timeout=4 * 60 * 60,
    volumes={str(REMOTE_RUNS_ROOT): RUNS_VOLUME},
)
def run_remote(
    *,
    dataset: str,
    architecture: str,
    encoder: str,
    seed: int,
    purpose: str,
    smoke_examples: int,
    download_parser: bool,
    run_id: str,
    code_state: dict[str, Any],
) -> dict[str, Any]:
    from aste_repro.training import run_experiment

    requested_run_id = run_id
    run_id, recovered_from, already_complete = _recover_run_attempt(
        requested_run_id=requested_run_id,
        dataset=dataset,
        architecture=architecture,
        encoder=encoder,
        seed=seed,
        purpose=purpose,
    )
    if already_complete:
        RUNS_VOLUME.commit()
        return {"run_id": run_id, "status": "complete"}

    backend_metadata = {
        "modal_sdk_version": importlib.metadata.version("modal"),
        "worker_python": platform.python_version(),
        "image_python": "3.10",
        "accelerator": "nvidia-t4",
        "gpu_type": "T4",
        "gpu_required": True,
        "torch_cuda_build": "2.5.1+cu121",
        "cpu_request": 4,
        "memory_mib_request": 16_384,
        "function_timeout_seconds": 4 * 60 * 60,
        "requested_run_id": requested_run_id,
    }
    if recovered_from is not None:
        backend_metadata["recovered_from_run_id"] = recovered_from
    try:
        result = run_experiment(
            dataset=dataset,
            architecture=architecture,
            encoder=encoder,
            seed=seed,
            purpose=purpose,
            smoke_examples=smoke_examples,
            download_parser=download_parser,
            run_id=run_id,
            runs_root=Path(str(REMOTE_RUNS_ROOT)),
            execution_backend="modal",
            code_state=code_state,
            backend_metadata=backend_metadata,
        )
        return {"run_id": result["run_id"], "status": result["status"]}
    finally:
        RUNS_VOLUME.commit()


@app.local_entrypoint()
def main(
    dataset: str,
    architecture: str,
    encoder: str,
    seed: int,
    purpose: str = "smoke",
    smoke_examples: int = 24,
    download_parser: bool = False,
    artifact_mode: str = "full",
) -> None:
    if purpose not in {"smoke", "benchmark"}:
        raise ValueError("purpose must be 'smoke' or 'benchmark'")
    if artifact_mode not in {"full", "summary"}:
        raise ValueError("artifact_mode must be 'full' or 'summary'")
    _validate_local_inputs(download_parser)

    from aste_repro.training import create_run_id

    run_id = create_run_id(
        dataset=dataset,
        architecture=architecture,
        encoder=encoder,
        seed=seed,
        purpose=purpose,
    )
    code_state = _local_code_state()
    try:
        result = run_remote.remote(
            dataset=dataset,
            architecture=architecture,
            encoder=encoder,
            seed=seed,
            purpose=purpose,
            smoke_examples=smoke_examples,
            download_parser=download_parser,
            run_id=run_id,
            code_state=code_state,
        )
    except Exception:
        try:
            artifact_path = _download_run(run_id, summary_only=artifact_mode == "summary")
        except Exception:
            pass
        else:
            print(
                f"Partial or failed Modal artifacts saved at {artifact_path}",
                file=sys.stderr,
            )
        raise

    artifact_path = _download_run(
        result["run_id"], summary_only=artifact_mode == "summary"
    )
    print(
        json.dumps(
            {
                "run_id": result["run_id"],
                "status": result["status"],
                "execution_backend": "modal",
                "artifact_mode": artifact_mode,
                "artifact_path": str(artifact_path),
                "remote_artifact_location": f"{RUNS_VOLUME_NAME}/{result['run_id']}",
            },
            sort_keys=True,
        )
    )
