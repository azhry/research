"""DeepSeek V4.1 Flash pre-annotation for HoASA and CASA into Label Studio import files.

Design constraints, in line with the workspace research workflow:

- every annotation is a real one-shot generation through the local OpenCode CLI;
- raw responses are cached per batch so runs resume without recomputation;
- spans are resolved to exact character offsets and validated against the
  sentence text; anything unresolvable is recorded, never guessed;
- exports are machine *predictions* in Label Studio JSON; the merged CASA files
  preserve the existing human annotations verbatim and add predictions;
- LLM output is auxiliary pre-annotation and is never treated as benchmark gold.

Subcommands
-----------

``run``       annotate one or more dataset/split pairs with the model;
``export``    rebuild Label Studio import files, sidecars, and the manifest;
``verify``    check the emitted files for schema, span, and relation invariants;
``diagnose``  measure agreement of the pre-annotations against HoASA gold.

The model calls default to ``opencode/deepseek-v4.1-flash-free#max`` which is
the user's configured model and effort. On this machine the OpenChamber-bundled
OpenCode v2 CLI is preferred over ``opencode`` on ``PATH`` because the managed
configuration requires the newer binary; pass ``--opencode-bin`` or set
``OPENCODE_BIN`` to override.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .hoasa import load_hoasa

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_SPLITS = {"hoasa": ("train", "dev", "test"), "casa": ("train", "valid", "test")}

DEFAULT_SOURCES_DIR = PROJECT_ROOT / "data" / "sources"
DEFAULT_LABELSTUDIO_DIR = PROJECT_ROOT / "docs" / "thesis" / "data" / "labelstudio_import"
DEFAULT_OUTPUT_DIR = DEFAULT_LABELSTUDIO_DIR / "deepseek_preannotations"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "llm_annotations"
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "results" / "llm-annotation" / "azh-538-summary.md"

MODEL_PROVIDER = "opencode"
MODEL_ID = "deepseek-v4.1-flash-free"
MODEL_VARIANT = "max"
PROMPT_VERSION = "v1"

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
POLARITY_ALIASES = {
    "POS": "POS",
    "POSITIVE": "POS",
    "P": "POS",
    "NEG": "NEG",
    "NEGATIVE": "NEG",
    "N": "NEG",
    "NEU": "NEU",
    "NEUTRAL": "NEU",
    "NEUT": "NEU",
}

PROMPT_TEMPLATE = """You are labeling Indonesian review sentences for aspect-based sentiment \
triplet extraction (ASTE). The sentences come from car reviews (CASA) or hotel reviews (HoASA).

For every numbered sentence, find each aspect term, each opinion word or phrase that expresses \
sentiment toward it, and the polarity toward that aspect. Report an implied aspect (no explicit \
aspect word in the sentence) with aspect null. If the opinion is implied and no opinion word \
exists, use opinion null.

Rules:
- aspect and opinion must be copied EXACTLY from the sentence (same characters, same spacing, \
same punctuation). Never translate, fix, shorten, reorder, or otherwise rewrite them.
- polarity is POS (positive), NEG (negative), or NEU (neutral), from the writer's perspective \
toward that aspect.
- emit one triplet per (aspect, opinion, polarity) combination; if one opinion applies to \
several aspects, emit one triplet per aspect.
- if a sentence expresses no aspect-opinion sentiment, emit an empty triplets list for it.

Example format (this example is not part of the batch):
[{"id": 1, "triplets": [{"aspect": "Kopi nya", "opinion": "enak", "polarity": "POS"}, \
{"aspect": null, "opinion": "bagus", "polarity": "POS"}]}]

Answer with exactly one JSON array, no explanation and no code fences.
[{"id": <sentence number>, "triplets": [{"aspect": "<exact substring or null>", "opinion": \
"<exact substring or null>", "polarity": "POS|NEG|NEU"}, ...]}, ...]

Sentences (numbered; return one object per sentence, in order):
__SENTENCES__
"""


def prompt_sha256() -> str:
    return hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest()


def build_prompt(texts: Sequence[str]) -> str:
    block = "\n".join(f"{position}. {text}" for position, text in enumerate(texts, start=1))
    return PROMPT_TEMPLATE.replace("__SENTENCES__", block)


def model_version_label(provider: str, model: str, variant: str) -> str:
    return f"{provider}/{model}#{variant}/prompt-{PROMPT_VERSION}"


def run_key(provider: str, model: str, variant: str) -> str:
    slug = f"{provider}_{model}-{variant}-{prompt_sha256()[:8]}"
    return re.sub(r"[^A-Za-z0-9._-]+", "-", slug)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def write_json(path: Path, payload: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    write_text(tmp, json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------
# Sentence loading
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Sentence:
    dataset: str
    split: str
    index: int
    text: str


def load_sentences(dataset: str, split: str, sources_dir: Path) -> list[Sentence]:
    if dataset == "hoasa":
        path = sources_dir / "hoasa" / f"{split}.txt"
        records = load_hoasa(path)
        sentences = [Sentence(dataset, split, record.row_id, record.sentence) for record in records]
    elif dataset == "casa":
        path = sources_dir / "casa" / f"{split}_preprocess.csv"
        sentences = []
        with path.open("r", encoding="utf-8", newline="") as stream:
            for index, row in enumerate(csv.DictReader(stream)):
                sentences.append(Sentence(dataset, split, index, row["sentence"]))
    else:
        raise ValueError(f"unknown dataset: {dataset}")
    for sentence in sentences:
        if "\n" in sentence.text or "\r" in sentence.text:
            raise ValueError(f"{dataset}/{split}:{sentence.index}: sentence contains a newline")
        if sentence.text != sentence.text.strip():
            raise ValueError(f"{dataset}/{split}:{sentence.index}: sentence is not stripped")
    return sentences


def resolve_pairs(dataset: str, split: str) -> list[tuple[str, str]]:
    datasets = ("hoasa", "casa") if dataset == "all" else (dataset,)
    pairs: list[tuple[str, str]] = []
    for name in datasets:
        splits = DATASET_SPLITS[name]
        if split == "all":
            pairs.extend((name, candidate) for candidate in splits)
        elif split in splits:
            pairs.append((name, split))
        else:
            raise SystemExit(f"dataset {name} has no split {split!r} (expected one of {splits})")
    return pairs


def hoasa_gold_spans(sources_dir: Path, split: str) -> dict[int, list[tuple[Any, Any, str]]]:
    """Gold HoASA triplets with character spans; used only for agreement diagnostics."""
    records = load_hoasa(sources_dir / "hoasa" / f"{split}.txt")
    gold: dict[int, list[tuple[Any, Any, str]]] = {}
    for record in records:
        tokens = [(match.start(), match.end()) for match in re.finditer(r"\S+", record.sentence)]
        triplets: list[tuple[Any, Any, str]] = []
        for aspect, opinion, sentiment in record.triplets:
            aspect_span = None if aspect == (-1,) else [tokens[aspect[0]][0], tokens[aspect[-1]][1]]
            opinion_span = [tokens[opinion[0]][0], tokens[opinion[-1]][1]]
            triplets.append((aspect_span, opinion_span, sentiment))
        gold[record.row_id] = triplets
    return gold


# --------------------------------------------------------------------------------------
# Model calls
# --------------------------------------------------------------------------------------

OPENCHAMBER_BUNDLED_CANDIDATES = (
    Path(os.environ.get("LOCALAPPDATA", ""))
    / "Programs"
    / "@openchamberelectron"
    / "resources"
    / "opencode-cli"
    / "opencode.exe",
)


def find_opencode_bin(explicit: str | None) -> str:
    if explicit:
        return explicit
    env = os.environ.get("OPENCODE_BIN")
    if env:
        return env
    for candidate in OPENCHAMBER_BUNDLED_CANDIDATES:
        if candidate.is_file():
            return str(candidate)
    found = shutil.which("opencode")
    if found:
        return found
    raise SystemExit("opencode binary not found; pass --opencode-bin or set OPENCODE_BIN")


def opencode_version(binary: str) -> str:
    try:
        completed = subprocess.run(
            [binary, "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"unknown ({error.__class__.__name__})"
    return (completed.stdout or completed.stderr or "unknown").strip().splitlines()[0]


def call_opencode(
    binary: str, model_ref: str, prompt: str, timeout: float, workdir: Path
) -> tuple[int, str, str, float]:
    started = time.monotonic()
    completed = subprocess.run(
        [binary, "run", "--standalone", "--model", model_ref, prompt],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        cwd=str(workdir),
    )
    duration = time.monotonic() - started
    return completed.returncode, ANSI_RE.sub("", completed.stdout or ""), completed.stderr or "", duration


def extract_triplet_array(raw: str) -> list[dict]:
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON array found in the response")
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON: {error.msg} at position {error.pos}") from error
    if not isinstance(payload, list):
        raise ValueError("response JSON is not an array")
    return payload


def resolve_span(text: str, needle: str) -> tuple[tuple[int, int] | None, int, str | None]:
    """Resolve a copied span to character offsets, preferring whole-token matches."""
    candidate = needle
    issue = None
    if candidate != candidate.strip():
        candidate = candidate.strip()
        issue = "trimmed surrounding whitespace"
    if not candidate:
        return None, 0, "empty span"
    starts: list[int] = []
    cursor = text.find(candidate)
    while cursor != -1:
        starts.append(cursor)
        cursor = text.find(candidate, cursor + 1)
    if not starts:
        return None, 0, "not an exact substring of the sentence"
    for start in starts:
        end = start + len(candidate)
        if (start == 0 or text[start - 1] == " ") and (end == len(text) or text[end] == " "):
            return (start, end), len(starts), issue
    return (starts[0], starts[0] + len(candidate)), len(starts), issue


def resolve_batch(
    sentences: Sequence[Sentence], payload: list[dict]
) -> tuple[list[dict], list[str]]:
    """Validate a parsed batch against its sentences; returns (records, errors)."""
    errors: list[str] = []
    expected_ids = list(range(1, len(sentences) + 1))
    by_id: dict[int, dict] = {}
    for item in payload:
        if not isinstance(item, dict):
            errors.append("payload item is not an object")
            continue
        item_id = item.get("id")
        if not isinstance(item_id, int) or item_id not in expected_ids:
            errors.append(f"unexpected id {item_id!r}")
            continue
        if item_id in by_id:
            errors.append(f"duplicate id {item_id}")
            continue
        by_id[item_id] = item
    if sorted(by_id) != expected_ids:
        missing = sorted(set(expected_ids) - set(by_id))
        errors.append(f"missing ids {missing}")
        return [], errors

    records: list[dict] = []
    for position, sentence in enumerate(sentences, start=1):
        item = by_id[position]
        raw_triplets = item.get("triplets")
        if not isinstance(raw_triplets, list):
            errors.append(f"id {position}: triplets is not a list")
            raw_triplets = []
        resolved: list[dict] = []
        seen: set[tuple[Any, Any, str]] = set()
        sentence_issues: list[str] = []
        for raw_triplet in raw_triplets:
            if not isinstance(raw_triplet, dict):
                sentence_issues.append("triplet is not an object")
                continue
            polarity = POLARITY_ALIASES.get(str(raw_triplet.get("polarity", "")).strip().upper())
            if polarity is None:
                sentence_issues.append(f"unknown polarity {raw_triplet.get('polarity')!r}")
                continue
            aspect_value = raw_triplet.get("aspect")
            opinion_value = raw_triplet.get("opinion")
            aspect_text = aspect_value if isinstance(aspect_value, str) and aspect_value.strip() else None
            opinion_text = (
                opinion_value if isinstance(opinion_value, str) and opinion_value.strip() else None
            )
            if aspect_text is None and opinion_text is None:
                sentence_issues.append("triplet has neither aspect nor opinion text")
                continue
            record = {
                "aspect": aspect_text,
                "opinion": opinion_text,
                "polarity": polarity,
                "aspect_span": None,
                "opinion_span": None,
                "aspect_occurrences": 0,
                "opinion_occurrences": 0,
                "issues": [],
            }
            if aspect_text is not None:
                span, occurrences, issue = resolve_span(sentence.text, aspect_text)
                record["aspect_occurrences"] = occurrences
                if span is not None:
                    record["aspect_span"] = [span[0], span[1]]
                if issue:
                    record["issues"].append(f"aspect: {issue}")
                if span is None:
                    record["issues"].append("aspect: unresolved")
            if opinion_text is not None:
                span, occurrences, issue = resolve_span(sentence.text, opinion_text)
                record["opinion_occurrences"] = occurrences
                if span is not None:
                    record["opinion_span"] = [span[0], span[1]]
                if issue:
                    record["issues"].append(f"opinion: {issue}")
                if span is None:
                    record["issues"].append("opinion: unresolved")
            key = (
                tuple(record["aspect_span"]) if record["aspect_span"] else None,
                tuple(record["opinion_span"]) if record["opinion_span"] else None,
                polarity,
            )
            if key in seen:
                continue
            seen.add(key)
            resolved.append(record)
        records.append({"index": sentence.index, "triplets": resolved, "issues": sentence_issues})
    return records, errors


def batch_problems(records: Sequence[dict], errors: Sequence[str]) -> list[str]:
    problems = list(errors)
    for record in records:
        for triplet in record.get("triplets", []):
            if "aspect: unresolved" in triplet["issues"] or "opinion: unresolved" in triplet["issues"]:
                problems.append(f"sentence {record['index']}: unresolved span")
    return problems


# --------------------------------------------------------------------------------------
# Annotation runner
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BatchSpec:
    dataset: str
    split: str
    index: int
    sentences: tuple[Sentence, ...]


def batch_path(cache_root: Path, spec: BatchSpec) -> Path:
    return cache_root / f"{spec.dataset}_{spec.split}" / f"batch_{spec.index:05d}.json"


def chunk_specs(dataset: str, split: str, sentences: Sequence[Sentence], size: int) -> list[BatchSpec]:
    return [
        BatchSpec(dataset, split, index, tuple(sentences[offset : offset + size]))
        for index, offset in enumerate(range(0, len(sentences), size))
    ]


def batch_state(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return str(read_json(path).get("status"))
    except (OSError, ValueError):
        return "corrupt"


def annotate_batch(
    spec: BatchSpec,
    binary: str,
    version: str,
    model_ref: str,
    attempts: int,
    timeout: float,
    workdir: Path,
) -> dict:
    prompt = build_prompt([sentence.text for sentence in spec.sentences])
    attempt_records: list[dict] = []
    attempts_store: list[dict] = []
    record: dict = {
        "batch_key": f"{spec.dataset}/{spec.split}/{spec.index:05d}",
        "dataset": spec.dataset,
        "split": spec.split,
        "batch_index": spec.index,
        "sentence_indices": [sentence.index for sentence in spec.sentences],
        "model_ref": model_ref,
        "provider": model_ref.split("/", 1)[0],
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_sha256(),
        "opencode_bin": binary,
        "opencode_version": version,
        "cwd": str(workdir),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    final_status = "failed"
    total_duration = 0.0
    for attempt in range(1, attempts + 1):
        attempt_prompt = prompt
        if attempt > 1 and attempt_records:
            feedback = attempt_records[-1].get("error") or "the previous answer was rejected"
            attempt_prompt = (
                prompt
                + f"\n\nYour previous answer was rejected: {feedback}\n"
                + "Return the full corrected JSON array again, now."
            )
        attempt_record: dict = {"attempt": attempt}
        try:
            code, stdout, stderr, duration = call_opencode(
                binary, model_ref, attempt_prompt, timeout, workdir
            )
        except subprocess.TimeoutExpired:
            attempt_record.update({"error": f"timeout after {timeout:.0f}s", "duration_s": timeout})
            attempts_store.append(attempt_record)
            total_duration += timeout
            continue
        except OSError as error:
            attempt_record.update({"error": f"OSError: {error}", "duration_s": 0.0})
            attempts_store.append(attempt_record)
            break
        total_duration += duration
        attempt_record.update(
            {
                "returncode": code,
                "duration_s": round(duration, 2),
                "raw_response": stdout[:100_000],
                "stderr_tail": stderr[-2_000:],
            }
        )
        if code != 0:
            attempt_record["error"] = f"exit code {code}"
            attempts_store.append(attempt_record)
            continue
        try:
            payload = extract_triplet_array(stdout)
        except ValueError as error:
            attempt_record["error"] = f"parse: {error}"
            attempts_store.append(attempt_record)
            continue
        records, errors = resolve_batch(spec.sentences, payload)
        problems = batch_problems(records, errors)
        if problems:
            attempt_record["error"] = "validation: " + "; ".join(problems[:8])
            attempts_store.append(attempt_record)
            continue
        attempt_records = records
        attempts_store.append(attempt_record)
        final_status = "ok"
        break

    record["status"] = final_status
    record["duration_s"] = round(total_duration, 2)
    record["attempts"] = attempts_store
    record["records"] = attempt_records
    if final_status != "ok":
        record["error"] = attempts_store[-1].get("error") if attempts_store else "no attempts recorded"
    return record


def run_pairs(pairs: Sequence[tuple[str, str]], args: argparse.Namespace) -> int:
    binary = find_opencode_bin(args.opencode_bin)
    version = opencode_version(binary)
    model_ref = f"{args.provider}/{args.model}#{args.variant}"
    cache_root = Path(args.cache_dir) / run_key(args.provider, args.model, args.variant)
    workdir = Path(tempfile.gettempdir()) / "aste-annotate" / run_key(args.provider, args.model, args.variant)
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"model: {model_ref} via {binary} ({version})")
    print(f"cache: {cache_root}")
    print(f"model cwd: {workdir}")
    exit_code = 0
    for dataset, split in pairs:
        sentences = load_sentences(dataset, split, args.sources_dir)
        if args.limit:
            sentences = sentences[: args.limit]
        specs = chunk_specs(dataset, split, sentences, args.batch_size)
        pending: list[BatchSpec] = []
        for spec in specs:
            state = batch_state(batch_path(cache_root, spec))
            if state is None or state == "corrupt" or (state == "failed" and args.retry_failed):
                pending.append(spec)
        print(
            f"{dataset}/{split}: {len(sentences)} sentences, {len(specs)} batches, "
            f"{len(specs) - len(pending)} already cached, {len(pending)} to run"
        )
        if not pending:
            continue
        failures = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    annotate_batch,
                    spec,
                    binary,
                    version,
                    model_ref,
                    args.attempts,
                    args.timeout,
                    workdir,
                ): spec
                for spec in pending
            }
            for done, future in enumerate(as_completed(futures), start=1):
                spec = futures[future]
                record = future.result()
                write_json(batch_path(cache_root, spec), record)
                status = record["status"]
                if status != "ok":
                    failures += 1
                print(
                    f"  [{done}/{len(pending)}] {dataset}/{split} batch {spec.index:05d}: "
                    f"{status} ({record['duration_s']:.1f}s)"
                )
        if failures:
            print(f"{dataset}/{split}: {failures} batches failed; rerun with --retry-failed")
            exit_code = 1
    return exit_code


# --------------------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------------------


@dataclass
class SplitRecords:
    sentences: list[Sentence]
    records: list[dict | None]
    batches: int
    failed_batches: list[int]
    missing_sentences: list[int]
    unresolved: int
    triplets: int
    implicit_triplets: int


def load_split_records(dataset: str, split: str, args: argparse.Namespace) -> SplitRecords:
    sentences = load_sentences(dataset, split, args.sources_dir)
    cache_split = Path(args.cache_dir) / run_key(args.provider, args.model, args.variant) / f"{dataset}_{split}"
    by_index: dict[int, dict] = {}
    batches = 0
    failed: list[int] = []
    for path in sorted(cache_split.glob("batch_*.json")):
        payload = read_json(path)
        batches += 1
        if payload.get("status") != "ok":
            failed.append(int(payload.get("batch_index", -1)))
            continue
        for record in payload.get("records", []):
            by_index[int(record["index"])] = record
    missing = [sentence.index for sentence in sentences if sentence.index not in by_index]
    records = [by_index.get(sentence.index) for sentence in sentences]
    unresolved = 0
    triplets = 0
    implicit = 0
    for record in records:
        if not record:
            continue
        for triplet in record.get("triplets", []):
            triplets += 1
            if triplet.get("aspect_span") is None or triplet.get("opinion_span") is None:
                implicit += 1
            if "aspect: unresolved" in triplet.get("issues", []) or "opinion: unresolved" in triplet.get("issues", []):
                unresolved += 1
    return SplitRecords(sentences, records, batches, failed, missing, unresolved, triplets, implicit)


def prediction_result(triplets: Sequence[dict], text: str) -> list[dict]:
    result: list[dict] = []
    relation_index = 0
    for triplet in triplets:
        aspect_span = triplet.get("aspect_span")
        opinion_span = triplet.get("opinion_span")
        if not aspect_span or not opinion_span:
            continue
        aspect_id = f"aspect_{relation_index}"
        opinion_id = f"opinion_{relation_index}"
        result.append(
            {
                "id": aspect_id,
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                "value": {
                    "start": aspect_span[0],
                    "end": aspect_span[1],
                    "text": text[aspect_span[0] : aspect_span[1]],
                    "labels": ["Aspect"],
                },
            }
        )
        result.append(
            {
                "id": opinion_id,
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                "value": {
                    "start": opinion_span[0],
                    "end": opinion_span[1],
                    "text": text[opinion_span[0] : opinion_span[1]],
                    "labels": ["Opinion"],
                },
            }
        )
        result.append(
            {
                "from_id": aspect_id,
                "to_id": opinion_id,
                "type": "relation",
                "direction": "right",
                "labels": [triplet["polarity"]],
            }
        )
        relation_index += 1
    return result


def sidecar_line(sentence: Sentence, record: dict | None) -> dict:
    return {
        "index": sentence.index,
        "text_sha256_12": sha256_text(sentence.text)[:12],
        "triplets": (record or {}).get("triplets", []),
        "issues": (record or {}).get("issues", []),
        "missing": record is None,
    }


def export_pair(
    dataset: str,
    split: str,
    args: argparse.Namespace,
    data: SplitRecords,
    manifest: dict,
) -> list[Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_version = model_version_label(args.provider, args.model, args.variant)
    predictions: list[dict] = []
    for sentence, record in zip(data.sentences, data.records):
        result = prediction_result((record or {}).get("triplets", []), sentence.text)
        predictions.append(
            {
                "data": {"text": sentence.text},
                "predictions": [{"model_version": model_version, "result": result}],
            }
        )
    written: list[Path] = []
    sidecar = "\n".join(
        json.dumps(sidecar_line(sentence, record), ensure_ascii=False)
        for sentence, record in zip(data.sentences, data.records)
    )
    sidecar_path = output_dir / f"{dataset}_{split}_triplets.jsonl"
    write_text(sidecar_path, sidecar + "\n")
    written.append(sidecar_path)

    if dataset == "casa":
        predictions_path = output_dir / f"casa_{split}_predictions.json"
        write_text(predictions_path, json.dumps(predictions, ensure_ascii=False, indent=1) + "\n")
        written.append(predictions_path)
        original_path = Path(args.labelstudio_dir) / f"casa_{split}.json"
        original_tasks = read_json(original_path)
        if len(original_tasks) != len(predictions):
            raise SystemExit(
                f"{original_path}: {len(original_tasks)} tasks do not match {len(predictions)} sentences"
            )
        merged = []
        for prediction_task, original_task in zip(predictions, original_tasks):
            merged.append(
                {
                    "data": original_task.get("data", prediction_task["data"]),
                    "annotations": original_task.get("annotations", []),
                    "predictions": prediction_task["predictions"],
                }
            )
        merged_path = output_dir / f"casa_{split}_merged.json"
        write_text(merged_path, json.dumps(merged, ensure_ascii=False, indent=1) + "\n")
        written.append(merged_path)
        inputs = [original_path]
    else:
        predictions_path = output_dir / f"hoasa_{split}.json"
        write_text(predictions_path, json.dumps(predictions, ensure_ascii=False, indent=1) + "\n")
        written.append(predictions_path)
        inputs = [Path(args.sources_dir) / "hoasa" / f"{split}.txt"]

    for path in written:
        key = path.name
        if path == sidecar_path:
            kind = "sidecar"
        elif path.name.endswith("_merged.json"):
            kind = "merged"
        else:
            kind = "predictions"
        manifest["outputs"][key] = {
            "file": str(path),
            "sha256": sha256_file(path),
            "kind": kind,
            "tasks": len(predictions),
            "predictions": sum(1 for task in predictions if task["predictions"]),
            "relations": sum(
                sum(1 for item in task["predictions"][0]["result"] if item["type"] == "relation")
                for task in predictions
            ),
            "missing_sentences": len(data.missing_sentences),
            "failed_batches": data.failed_batches,
            "unresolved": data.unresolved,
            "triplets": data.triplets,
            "implicit_triplets": data.implicit_triplets,
        }
    for path in inputs:
        manifest["inputs"][path.name] = {"file": str(path), "sha256": sha256_file(path)}
    return written


def export_all(pairs: Sequence[tuple[str, str]], args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        manifest.setdefault("outputs", {})
        manifest.setdefault("inputs", {})
    else:
        manifest = {"outputs": {}, "inputs": {}}
    manifest.update(
        {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "run_key": run_key(args.provider, args.model, args.variant),
            "model": {
                "provider": args.provider,
                "id": args.model,
                "variant": args.variant,
                "version_label": model_version_label(args.provider, args.model, args.variant),
            },
            "prompt": {"version": PROMPT_VERSION, "sha256": prompt_sha256()},
            "sources_dir": str(args.sources_dir),
            "cache_dir": str(Path(args.cache_dir) / run_key(args.provider, args.model, args.variant)),
            "output_dir": str(output_dir),
        }
    )
    problems = 0
    for dataset, split in pairs:
        data = load_split_records(dataset, split, args)
        if data.missing_sentences or data.failed_batches:
            problems += 1
            print(
                f"{dataset}/{split}: WARNING missing {len(data.missing_sentences)} sentences; "
                f"failed batches {data.failed_batches}"
            )
        written = export_pair(dataset, split, args, data, manifest)
        for path in written:
            print(f"wrote {path}")
    write_json(manifest_path, manifest)
    print(f"wrote {manifest_path}")
    return 1 if problems else 0


# --------------------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------------------

IMPORT_FILE_RE = re.compile(r"^(hoasa|casa)_(train|dev|test|valid)(?:_(predictions|merged))?\.json$")


def verify_file(path: Path, labelstudio_dir: Path | None, errors: list[str]) -> dict[str, int]:
    match = IMPORT_FILE_RE.match(path.name)
    if not match:
        errors.append(f"{path.name}: unexpected file name")
        return {}
    dataset, split, kind = match.groups()
    tasks = read_json(path)
    if not isinstance(tasks, list):
        errors.append(f"{path.name}: top level is not a list")
        return {}
    counters = {"tasks": len(tasks), "predictions": 0, "relations": 0, "spans": 0}
    for task_index, task in enumerate(tasks):
        text = (task.get("data") or {}).get("text")
        if not isinstance(text, str):
            errors.append(f"{path.name}: task {task_index} has no text")
            continue
        predictions = task.get("predictions") or []
        if len(predictions) != 1:
            errors.append(f"{path.name}: task {task_index} has {len(predictions)} predictions")
        for prediction in predictions:
            counters["predictions"] += 1
            ids: dict[str, str] = {}
            for item in prediction.get("result", []):
                item_type = item.get("type")
                if item_type == "labels":
                    counters["spans"] += 1
                    labels = (item.get("value") or {}).get("labels") or []
                    if len(labels) != 1 or labels[0] not in ("Aspect", "Opinion"):
                        errors.append(f"{path.name}: task {task_index}: bad labels {labels}")
                        continue
                    value = item["value"]
                    start, end = value.get("start"), value.get("end")
                    if not isinstance(start, int) or not isinstance(end, int) or not (0 <= start < end <= len(text)):
                        errors.append(f"{path.name}: task {task_index}: bad span offsets")
                        continue
                    if text[start:end] != value.get("text"):
                        errors.append(f"{path.name}: task {task_index}: span text mismatch")
                    if item.get("from_name") != "label" or item.get("to_name") != "text":
                        errors.append(f"{path.name}: task {task_index}: bad from/to names")
                    item_id = item.get("id")
                    if not isinstance(item_id, str) or item_id in ids:
                        errors.append(f"{path.name}: task {task_index}: duplicate or missing id")
                        continue
                    ids[item_id] = labels[0]
                elif item_type == "relation":
                    counters["relations"] += 1
                    from_id, to_id = item.get("from_id"), item.get("to_id")
                    labels = item.get("labels") or []
                    if len(labels) != 1 or labels[0] not in ("POS", "NEG", "NEU"):
                        errors.append(f"{path.name}: task {task_index}: bad relation labels {labels}")
                    if ids.get(from_id) != "Aspect" or ids.get(to_id) != "Opinion":
                        errors.append(f"{path.name}: task {task_index}: dangling relation ids")
                    if item.get("direction") != "right":
                        errors.append(f"{path.name}: task {task_index}: bad relation direction")
                else:
                    errors.append(f"{path.name}: task {task_index}: unknown result type {item_type!r}")
    if dataset == "casa" and kind == "merged":
        if labelstudio_dir is None:
            errors.append(f"{path.name}: pass --labelstudio-dir to compare against the originals")
        else:
            original_path = labelstudio_dir / f"casa_{split}.json"
            original_tasks = read_json(original_path)
            if len(original_tasks) != len(tasks):
                errors.append(f"{path.name}: task count differs from {original_path.name}")
            else:
                for task_index, (task, original) in enumerate(zip(tasks, original_tasks)):
                    if task.get("annotations") != original.get("annotations"):
                        errors.append(f"{path.name}: task {task_index}: human annotations changed")
                        break
                    if task.get("data") != original.get("data"):
                        errors.append(f"{path.name}: task {task_index}: task data changed")
                        break
    return counters


def verify_directory(directory: Path, labelstudio_dir: Path | None) -> int:
    errors: list[str] = []
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        errors.append("manifest.json is missing")
        manifest = None
    else:
        manifest = read_json(manifest_path)
    files = sorted(
        path
        for path in directory.glob("*.json")
        if path.name != "manifest.json"
    )
    if not files:
        errors.append("no import files found")
    total = {"tasks": 0, "predictions": 0, "relations": 0, "spans": 0}
    for path in files:
        counters = verify_file(path, labelstudio_dir, errors)
        for key, value in counters.items():
            total[key] = total.get(key, 0) + value
        print(
            f"{path.name}: tasks={counters.get('tasks', 0)} predictions={counters.get('predictions', 0)} "
            f"spans={counters.get('spans', 0)} relations={counters.get('relations', 0)}"
        )
    sidecars = sorted(directory.glob("*_triplets.jsonl"))
    for path in sidecars:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        expected = None
        match = re.match(r"^(hoasa|casa)_(train|dev|test|valid)_triplets\.jsonl$", path.name)
        if not match:
            errors.append(f"{path.name}: unexpected sidecar name")
            continue
        dataset, split = match.groups()
        if dataset == "casa":
            expected = len(read_json(directory / f"casa_{split}_predictions.json"))
        else:
            expected = len(read_json(directory / f"hoasa_{split}.json"))
        if len(lines) != expected:
            errors.append(f"{path.name}: {len(lines)} lines do not match {expected} tasks")
        missing = 0
        for line in lines:
            payload = json.loads(line)
            for triplet in payload.get("triplets", []):
                issues = triplet.get("issues", [])
                if "aspect: unresolved" in issues or "opinion: unresolved" in issues:
                    errors.append(f"{path.name}: unresolved span at index {payload.get('index')}")
                    break
            if payload.get("missing"):
                missing += 1
        if missing:
            errors.append(f"{path.name}: {missing} sentences have no model record")
        print(f"{path.name}: lines={len(lines)} missing={missing}")
    if manifest:
        for name, entry in sorted((manifest.get("outputs") or {}).items()):
            path = directory / name
            if not path.exists():
                errors.append(f"manifest entry {name} has no file")
                continue
            if entry.get("sha256") != sha256_file(path):
                errors.append(f"{name}: sha256 differs from manifest")
        if not isinstance(manifest.get("prompt", {}).get("sha256"), str):
            errors.append("manifest has no prompt sha256")
        if not isinstance(manifest.get("model", {}).get("id"), str):
            errors.append("manifest has no model id")
    if errors:
        for error in errors[:50]:
            print(f"ERROR: {error}")
        print(f"verify: FAILED ({len(errors)} problems)")
        return 1
    print(
        f"verify: OK (tasks={total['tasks']} predictions={total['predictions']} "
        f"spans={total['spans']} relations={total['relations']})"
    )
    return 0


# --------------------------------------------------------------------------------------
# HoASA agreement diagnostics
# --------------------------------------------------------------------------------------


def triplet_counter(triplets: Iterable[tuple[Any, Any, str]]) -> dict[tuple[Any, Any, str], int]:
    counter: dict[tuple[Any, Any, str], int] = {}
    for aspect_span, opinion_span, polarity in triplets:
        key = (
            tuple(aspect_span) if aspect_span else None,
            tuple(opinion_span) if opinion_span else None,
            polarity,
        )
        counter[key] = counter.get(key, 0) + 1
    return counter


def prf(matched: int, predicted: int, gold: int) -> tuple[float, float, float]:
    precision = matched / predicted if predicted else 0.0
    recall = matched / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def diagnose(args: argparse.Namespace) -> tuple[str, int]:
    lines: list[str] = []
    lines.append("# AZH-538 DeepSeek V4.1 Flash pre-annotation summary")
    lines.append("")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M %z')}")
    lines.append(
        f"Model: `{args.provider}/{args.model}#{args.variant}` "
        f"(prompt {PROMPT_VERSION}, sha256 {prompt_sha256()[:12]})"
    )
    lines.append("")
    lines.append(
        "These are machine pre-annotations for human review. They are auxiliary data, "
        "not benchmark gold; the numbers below are a machine-vs-gold diagnostic."
    )
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    lines.append(
        "| dataset | split | sentences | missing sentences | triplets | implicit triplets | unresolved spans | failed batches |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    coverage: dict[tuple[str, str], SplitRecords] = {}
    for dataset, split in resolve_pairs(args.dataset, args.split):
        data = load_split_records(dataset, split, args)
        coverage[(dataset, split)] = data
        lines.append(
            f"| {dataset} | {split} | {len(data.sentences)} | {len(data.missing_sentences)} | "
            f"{data.triplets} | {data.implicit_triplets} | {data.unresolved} | {len(data.failed_batches)} |"
        )
    lines.append("")
    lines.append("## HoASA agreement against gold (exact triplet match)")
    lines.append("")
    lines.append(
        "| split | gold | predicted | matched | P | R | F1 | explicit-only P | R | F1 | gold implicit | predicted implicit |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for dataset, split in sorted(coverage):
        if dataset != "hoasa":
            continue
        data = coverage[(dataset, split)]
        gold = hoasa_gold_spans(Path(args.sources_dir), split)
        gold_all = [triplet for row in gold.values() for triplet in row]
        gold_explicit = [triplet for triplet in gold_all if triplet[0] is not None]
        predicted_all: list[tuple[Any, Any, str]] = []
        predicted_explicit: list[tuple[Any, Any, str]] = []
        for record in data.records:
            for triplet in (record or {}).get("triplets", []):
                item = (triplet.get("aspect_span"), triplet.get("opinion_span"), triplet.get("polarity"))
                predicted_all.append(item)
                if item[0] is not None:
                    predicted_explicit.append(item)
        gold_counter_all = triplet_counter(gold_all)
        predicted_counter_all = triplet_counter(predicted_all)
        matched_all = sum(
            min(count, predicted_counter_all.get(key, 0))
            for key, count in gold_counter_all.items()
        )
        gold_counter_explicit = triplet_counter(gold_explicit)
        predicted_counter_explicit = triplet_counter(predicted_explicit)
        matched_explicit = sum(
            min(count, predicted_counter_explicit.get(key, 0))
            for key, count in gold_counter_explicit.items()
        )
        precision, recall, f1 = prf(matched_all, len(predicted_all), len(gold_all))
        precision_x, recall_x, f1_x = prf(
            matched_explicit, len(predicted_explicit), len(gold_explicit)
        )
        gold_implicit = len(gold_all) - len(gold_explicit)
        predicted_implicit = len(predicted_all) - len(predicted_explicit)
        lines.append(
            f"| {split} | {len(gold_all)} | {len(predicted_all)} | {matched_all} | "
            f"{precision:.3f} | {recall:.3f} | {f1:.3f} | {precision_x:.3f} | {recall_x:.3f} | "
            f"{f1_x:.3f} | {gold_implicit} | {predicted_implicit} |"
        )
    lines.append("")
    lines.append(
        "`predicted implicit` counts triplets with a missing aspect span (implicit aspect) or a "
        "missing opinion span; only triplets with both spans become Label Studio relations, the "
        "rest remain in the sidecar files."
    )
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append(f"Run key: `{run_key(args.provider, args.model, args.variant)}`")
    lines.append("")
    lines.append("| input | sha256 |")
    lines.append("| --- | --- |")
    seen_inputs: set[str] = set()
    for dataset, split in sorted(coverage):
        if dataset == "hoasa":
            paths = [Path(args.sources_dir) / "hoasa" / f"{split}.txt"]
        else:
            paths = [
                Path(args.sources_dir) / "casa" / f"{split}_preprocess.csv",
                Path(args.labelstudio_dir) / f"casa_{split}.json",
            ]
        for path in paths:
            if str(path) in seen_inputs:
                continue
            seen_inputs.add(str(path))
            digest = sha256_file(path) if path.is_file() else "missing"
            lines.append(f"| {path} | {digest} |")
    lines.append("")
    exit_code = 1 if any(
        data.failed_batches or data.missing_sentences for data in coverage.values()
    ) else 0
    return "\n".join(lines) + "\n", exit_code


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", choices=("hoasa", "casa", "all"), default="all")
    parser.add_argument(
        "--split",
        choices=("train", "dev", "test", "valid", "all"),
        default="all",
        help="dataset split; 'dev' is HoASA-only, 'valid' is CASA-only",
    )
    parser.add_argument("--sources-dir", type=Path, default=DEFAULT_SOURCES_DIR)
    parser.add_argument("--labelstudio-dir", type=Path, default=DEFAULT_LABELSTUDIO_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--provider", default=MODEL_PROVIDER)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--variant", default=MODEL_VARIANT)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aste-annotate", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="annotate dataset/split pairs with the model")
    add_common(run_parser)
    run_parser.add_argument("--opencode-bin", default=None)
    run_parser.add_argument("--batch-size", type=int, default=10)
    run_parser.add_argument("--workers", type=int, default=4)
    run_parser.add_argument("--attempts", type=int, default=3)
    run_parser.add_argument("--timeout", type=float, default=600.0)
    run_parser.add_argument("--limit", type=int, default=0, help="pilot: only the first N sentences per split")
    run_parser.add_argument("--retry-failed", action="store_true")

    export_parser = subparsers.add_parser("export", help="rebuild import files and the manifest from cache")
    add_common(export_parser)

    verify_parser = subparsers.add_parser("verify", help="check import files for schema/span invariants")
    verify_parser.add_argument("--directory", type=Path, required=True)
    verify_parser.add_argument("--labelstudio-dir", type=Path, default=None)

    diagnose_parser = subparsers.add_parser("diagnose", help="measure agreement against HoASA gold")
    add_common(diagnose_parser)
    diagnose_parser.add_argument("--write", type=Path, default=None)

    args = parser.parse_args(argv)
    if args.command == "run":
        return run_pairs(resolve_pairs(args.dataset, args.split), args)
    if args.command == "export":
        return export_all(resolve_pairs(args.dataset, args.split), args)
    if args.command == "verify":
        return verify_directory(args.directory, args.labelstudio_dir)
    if args.command == "diagnose":
        report, exit_code = diagnose(args)
        if args.write:
            write_text(Path(args.write), report)
            print(f"wrote {args.write}")
        else:
            print(report)
        return exit_code
    raise SystemExit(f"unknown command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
