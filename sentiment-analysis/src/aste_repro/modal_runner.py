"""Optional Modal entrypoint for the existing ASTE experiment runner."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import tempfile
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


def _download_run(run_id: str) -> Path:
    destination = RUNS_ROOT / run_id
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing run artifacts: {destination}")

    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{run_id}-download-", dir=RUNS_ROOT))
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
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite existing run artifacts: {destination}")
        staging.replace(destination)
        return destination
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


@app.function(
    image=image,
    cpu=4,
    memory=16_384,
    timeout=60 * 60,
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

    backend_metadata = {
        "modal_sdk_version": importlib.metadata.version("modal"),
        "worker_python": platform.python_version(),
        "image_python": "3.10",
        "accelerator": "cpu",
        "cpu_request": 4,
        "memory_mib_request": 16_384,
    }
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
) -> None:
    if purpose not in {"smoke", "benchmark"}:
        raise ValueError("purpose must be 'smoke' or 'benchmark'")
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
            artifact_path = _download_run(run_id)
        except Exception:
            pass
        else:
            print(
                f"Partial or failed Modal artifacts saved at {artifact_path}",
                file=sys.stderr,
            )
        raise

    artifact_path = _download_run(result["run_id"])
    print(
        json.dumps(
            {
                "run_id": result["run_id"],
                "status": result["status"],
                "execution_backend": "modal",
                "artifact_path": str(artifact_path),
            },
            sort_keys=True,
        )
    )
