"""Execute a notebook with this project's pinned Python environment.

Jupyter's default ``python3`` kernelspec can silently select a global Python
installation instead of the interpreter that launched ``nbconvert``. This
runner creates a temporary kernelspec that points at the current interpreter
and rejects environments that do not match ``.python-version`` and
``requirements.txt``.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    output_dir = _project_path(args.output_dir, must_exist=False)
    output_name = args.output or f"{notebook.stem}_executed.ipynb"
    if Path(output_name).name != output_name or not output_name.lower().endswith(".ipynb"):
        raise SystemExit("--output must be a plain .ipynb filename")

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
        command = [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            str(notebook),
            "--output",
            output_name,
            "--output-dir",
            str(output_dir),
            "--ExecutePreprocessor.timeout=0",
            f"--ExecutePreprocessor.kernel_name={kernel_name}",
        ]
        return subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
