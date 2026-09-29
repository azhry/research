"""Pinned source recovery and integrity audits for HoASA and CASA."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "sources"
ANNOTATION_ROOT = ROOT / "docs" / "thesis" / "data" / "labelstudio_import"
HOASA_REVISION = "0bb3880fed6caf28e05b4eccf329b642e064fd0a"
CASA_REVISION = "ce728f6926a36174b9923dfe49d6a6839b6e9bb7"
HOASA_REPOSITORY = "https://github.com/rdyzakya/IndoLEGO-ABSA"
CASA_REPOSITORY = "https://github.com/indobenchmark/indonlu"
CASA_CATEGORIES = ("fuel", "machine", "others", "part", "price", "service")
CASA_POLARITIES = {"positive", "negative", "neutral"}
ASTE_POLARITIES = {"POS", "NEG", "NEU"}

HOASA_SPLITS = {
    "train": {
        "sentences": 3000,
        "triplets": 7551,
        "implicit_aspects": 1294,
        "sha256": "33714cfdafa40a5c581cb744845268aeea18450265af67ec572623afa7c20b6d",
    },
    "dev": {
        "sentences": 1000,
        "triplets": 2458,
        "implicit_aspects": 469,
        "sha256": "a00d16dcb0bcbc6cde1c9650323145c5d22be33620eff7c812fafe36ec06c690",
    },
    "test": {
        "sentences": 1000,
        "triplets": 2709,
        "implicit_aspects": 418,
        "sha256": "3f3883f8c25706d79b1c6bbce253bb548b38c4eb4a786f4b14506d3a28a749a0",
    },
}
CASA_SPLITS = {
    "train": {
        "filename": "train_preprocess.csv",
        "rows": 810,
        "sha256": "ffd2a88edf5e270cea79ad84d2ca4170c9a2fd71a38540280d5eb3b95d261f76",
    },
    "valid": {
        "filename": "valid_preprocess.csv",
        "rows": 90,
        "sha256": "4ea114d060796e59944b1cf7f0ad7950bd0532024348a17d0f7c6b6464328424",
    },
    "test": {
        "filename": "test_preprocess.csv",
        "rows": 180,
        "sha256": "843564f24c10d8360fe63395a821f94eb46985abddb94dba6865fcd0a29c70c7",
    },
}
LABELSTUDIO_SPLITS = {
    "train": {"filename": "casa_train.json", "tasks": 810, "annotated": 436, "relations": 970},
    "valid": {"filename": "casa_valid.json", "tasks": 90, "annotated": 48, "relations": 92},
    "test": {"filename": "casa_test.json", "tasks": 180, "annotated": 93, "relations": 170},
}


class AuditError(ValueError):
    """Raised when a pinned source or dataset invariant does not hold."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        actual = sha256_file(destination)
        if actual != expected_sha256:
            raise AuditError(f"refusing to replace existing file with unexpected hash: {destination}")
        return

    request = Request(url, headers={"User-Agent": "AZH-521-reproduction/0.1"})
    with urlopen(request, timeout=60) as response:
        body = response.read()
    actual = hashlib.sha256(body).hexdigest()
    if actual != expected_sha256:
        raise AuditError(f"download hash mismatch for {destination.name}: {actual}")

    fd, temporary_name = tempfile.mkstemp(prefix=f"{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(body)
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def fetch_pinned_sources(data_root: Path = DATA_ROOT) -> None:
    """Download only absent pinned source files; preserve any unexpected file."""
    for split, expected in HOASA_SPLITS.items():
        url = f"https://raw.githubusercontent.com/rdyzakya/IndoLEGO-ABSA/{HOASA_REVISION}/data/{split}.txt"
        _download(url, data_root / "hoasa" / f"{split}.txt", expected["sha256"])

    for split, expected in CASA_SPLITS.items():
        url = (
            f"https://raw.githubusercontent.com/indobenchmark/indonlu/{CASA_REVISION}/"
            f"dataset/casa_absa-prosa/{expected['filename']}"
        )
        _download(url, data_root / "casa" / expected["filename"], expected["sha256"])


def _parse_hoasa(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    sentence_count = triplet_count = implicit_count = explicit_count = 0
    duplicates = Counter()
    polarity_counts = Counter()
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            line = raw_line.rstrip("\r\n")
            if line.count("####") != 1:
                raise AuditError(f"{path.name}:{line_number}: expected one #### delimiter")
            sentence, raw_annotations = line.split("####", maxsplit=1)
            tokens = sentence.strip().split()
            if not tokens:
                raise AuditError(f"{path.name}:{line_number}: empty token sequence")
            try:
                annotations = ast.literal_eval(raw_annotations.strip())
            except (SyntaxError, ValueError) as error:
                raise AuditError(f"{path.name}:{line_number}: invalid annotation literal") from error
            if not isinstance(annotations, list):
                raise AuditError(f"{path.name}:{line_number}: annotations must be a list")

            sentence_count += 1
            duplicates[sentence.strip()] += 1
            for annotation in annotations:
                if not isinstance(annotation, (tuple, list)) or len(annotation) != 3:
                    raise AuditError(f"{path.name}:{line_number}: expected a triplet tuple")
                aspect, opinion, polarity = annotation
                if not isinstance(aspect, list) or not isinstance(opinion, list):
                    raise AuditError(f"{path.name}:{line_number}: aspect/opinion indices must be lists")
                if polarity not in ASTE_POLARITIES:
                    raise AuditError(f"{path.name}:{line_number}: unknown polarity label")
                if not opinion or any(type(index) is not int or index < 0 or index >= len(tokens) for index in opinion):
                    raise AuditError(f"{path.name}:{line_number}: opinion index outside token range")
                if aspect == [-1]:
                    implicit_count += 1
                elif not aspect or any(type(index) is not int or index < 0 or index >= len(tokens) for index in aspect):
                    raise AuditError(f"{path.name}:{line_number}: aspect index outside token range")
                else:
                    explicit_count += 1
                triplet_count += 1
                polarity_counts[polarity] += 1

    digest = sha256_file(path)
    observed = {
        "sentences": sentence_count,
        "triplets": triplet_count,
        "implicit_aspects": implicit_count,
        "sha256": digest,
    }
    for field in ("sentences", "triplets", "implicit_aspects", "sha256"):
        if observed[field] != expected[field]:
            raise AuditError(
                f"{path.name}: {field} mismatch, expected {expected[field]}, observed {observed[field]}"
            )
    return {
        **observed,
        "explicit_aspects": explicit_count,
        "duplicate_sentence_rows": sum(count - 1 for count in duplicates.values()),
        "polarity_triplets": dict(sorted(polarity_counts.items())),
    }


def _parse_casa_csv(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    rows = 0
    vectors: dict[str, tuple[str, ...]] = {}
    duplicate_rows = 0
    category_counts = {category: Counter() for category in CASA_CATEGORIES}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"sentence", *CASA_CATEGORIES}.issubset(reader.fieldnames):
            raise AuditError(f"{path.name}: required CASA columns are missing")
        for line_number, row in enumerate(reader, 2):
            sentence = (row.get("sentence") or "").strip()
            if not sentence:
                raise AuditError(f"{path.name}:{line_number}: empty sentence")
            vector = tuple((row.get(category) or "").strip().lower() for category in CASA_CATEGORIES)
            if any(value not in CASA_POLARITIES for value in vector):
                raise AuditError(f"{path.name}:{line_number}: invalid or missing category polarity")
            previous = vectors.get(sentence)
            if previous is not None:
                duplicate_rows += 1
                if previous != vector:
                    raise AuditError(f"{path.name}:{line_number}: duplicate sentence has conflicting labels")
            else:
                vectors[sentence] = vector
            for category, value in zip(CASA_CATEGORIES, vector):
                category_counts[category][value] += 1
            rows += 1

    digest = sha256_file(path)
    if rows != expected["rows"]:
        raise AuditError(f"{path.name}: row count mismatch, expected {expected['rows']}, observed {rows}")
    if digest != expected["sha256"]:
        raise AuditError(f"{path.name}: SHA-256 mismatch")
    return {
        "rows": rows,
        "unique_sentences": len(vectors),
        "duplicate_rows": duplicate_rows,
        "sha256": digest,
        "category_polarities": {
            category: dict(sorted(counts.items())) for category, counts in category_counts.items()
        },
        "sentences": set(vectors),
        "label_vectors": vectors,
    }


def _labelstudio_coverage(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    try:
        tasks = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AuditError(f"cannot read Label Studio export: {path.name}") from error
    if not isinstance(tasks, list):
        raise AuditError(f"{path.name}: expected task array")

    texts: set[str] = set()
    annotated_tasks = 0
    relation_count = aspect_tasks = opinion_tasks = 0
    for task in tasks:
        text = str((task.get("data") or {}).get("text", "")).strip()
        if not text:
            raise AuditError(f"{path.name}: task missing review text")
        texts.add(text)
        results = [
            result
            for annotation in task.get("annotations", [])
            for result in annotation.get("result", [])
        ]
        has_aspect = any(
            result.get("type") == "labels" and "Aspect" in result.get("value", {}).get("labels", [])
            for result in results
        )
        has_opinion = any(
            result.get("type") == "labels" and "Opinion" in result.get("value", {}).get("labels", [])
            for result in results
        )
        relations = [result for result in results if result.get("type") == "relation"]
        if has_aspect:
            aspect_tasks += 1
        if has_opinion:
            opinion_tasks += 1
        if has_aspect or has_opinion or relations:
            annotated_tasks += 1
        relation_count += len(relations)

    if len(tasks) != expected["tasks"]:
        raise AuditError(f"{path.name}: task count mismatch")
    if annotated_tasks != expected["annotated"] or relation_count != expected["relations"]:
        raise AuditError(f"{path.name}: Label Studio annotation coverage changed")
    return {
        "tasks": len(tasks),
        "unique_texts": len(texts),
        "annotated_tasks": annotated_tasks,
        "aspect_tasks": aspect_tasks,
        "opinion_tasks": opinion_tasks,
        "relations": relation_count,
        "texts": texts,
    }


def audit_sources(data_root: Path = DATA_ROOT, annotation_root: Path = ANNOTATION_ROOT) -> dict[str, Any]:
    """Validate pinned files and report annotation readiness without review text."""
    hoasa = {
        split: _parse_hoasa(data_root / "hoasa" / f"{split}.txt", expected)
        for split, expected in HOASA_SPLITS.items()
    }
    casa = {
        split: _parse_casa_csv(data_root / "casa" / expected["filename"], expected)
        for split, expected in CASA_SPLITS.items()
    }
    annotations = {
        split: _labelstudio_coverage(annotation_root / expected["filename"], expected)
        for split, expected in LABELSTUDIO_SPLITS.items()
    }
    alignment = {
        split: {
            "all_reviews_match_source": annotations[split]["texts"] == casa[split]["sentences"],
            "source_rows": casa[split]["rows"],
            "unique_source_reviews": casa[split]["unique_sentences"],
            "annotated_tasks": annotations[split]["annotated_tasks"],
            "total_tasks": annotations[split]["tasks"],
            "relations": annotations[split]["relations"],
        }
        for split in CASA_SPLITS
    }
    for split, values in alignment.items():
        if not values["all_reviews_match_source"]:
            raise AuditError(f"CASA {split}: Label Studio texts do not match pinned source rows")

    summary = {
        "schema_version": 1,
        "hoasa": {"repository": HOASA_REPOSITORY, "revision": HOASA_REVISION, "splits": hoasa},
        "casa": {
            "repository": CASA_REPOSITORY,
            "revision": CASA_REVISION,
            "categories": list(CASA_CATEGORIES),
            "splits": {
                split: {key: value for key, value in detail.items() if key not in {"sentences", "label_vectors"}}
                for split, detail in casa.items()
            },
            "labelstudio": {
                split: {key: value for key, value in detail.items() if key != "texts"}
                for split, detail in annotations.items()
            },
            "alignment": alignment,
            "aste_gold_ready": False,
            "readiness_blockers": [
                "Label Studio task coverage is partial in all splits.",
                "Relations do not contain a human-adjudicated sentiment label.",
                "No approved aspect-to-category mapping artifact is present.",
                "Duplicate train review rows require an explicit split-preserving policy.",
            ],
        },
    }
    return summary
