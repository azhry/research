"""Canonical HoASA loading and a guarded explicit-span PyABSA data view."""

from __future__ import annotations

import ast
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from .metrics import Triplet, normalize_triplet
from .source_data import ASTE_POLARITIES


@dataclass(frozen=True)
class HoasaRecord:
    row_id: int
    sentence: str
    triplets: tuple[Triplet, ...]


def load_hoasa(path: Path) -> list[HoasaRecord]:
    """Load a source split without changing text, indices, order, or implicit labels."""
    records = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for row_id, raw_line in enumerate(stream):
            line = raw_line.rstrip("\r\n")
            if line.count("####") != 1:
                raise ValueError(f"{path.name}:{row_id + 1}: expected one #### delimiter")
            sentence, literal = line.split("####", maxsplit=1)
            sentence = sentence.strip()
            if not sentence:
                raise ValueError(f"{path.name}:{row_id + 1}: empty sentence")
            annotations = ast.literal_eval(literal.strip())
            if not isinstance(annotations, list):
                raise ValueError(f"{path.name}:{row_id + 1}: annotations must be a list")
            triplets = []
            token_count = len(sentence.split())
            for value in annotations:
                triplet = normalize_triplet(value)
                aspect, opinion, sentiment = triplet
                if sentiment not in ASTE_POLARITIES:
                    raise ValueError(f"{path.name}:{row_id + 1}: invalid sentiment")
                if opinion[0] < 0 or opinion[-1] >= token_count:
                    raise ValueError(f"{path.name}:{row_id + 1}: opinion index outside sentence")
                if aspect != (-1,) and (aspect[0] < 0 or aspect[-1] >= token_count):
                    raise ValueError(f"{path.name}:{row_id + 1}: aspect index outside sentence")
                if aspect != (-1,) and aspect != tuple(range(aspect[0], aspect[-1] + 1)):
                    raise ValueError(f"{path.name}:{row_id + 1}: noncontiguous aspect span")
                if opinion != tuple(range(opinion[0], opinion[-1] + 1)):
                    raise ValueError(f"{path.name}:{row_id + 1}: noncontiguous opinion span")
                triplets.append(triplet)
            records.append(HoasaRecord(row_id, sentence, tuple(triplets)))
    return records


def _pyabsa_sentence_map(sentence: str) -> tuple[str, list[int]]:
    """Mirror PyABSA 2.4.3's hyphen rewrite and retain reversible token offsets."""
    if "placeholder" in sentence:
        raise ValueError("PyABSA 2.4.3 rewrites the literal word 'placeholder'; refusing lossy input")
    original_tokens = sentence.split()
    normalized = sentence.replace(" - ", " placeholder ").replace("-", " ")
    model_tokens = normalized.split()
    source_to_model: list[list[int]] = []
    mapped_tokens: list[str] = []
    model_to_source: list[int] = []
    cursor = 0
    for source_index, token in enumerate(original_tokens):
        if token == "-" and 0 < source_index < len(original_tokens) - 1:
            pieces = ["placeholder"]
        else:
            pieces = token.replace("-", " ").split()
        indices = list(range(cursor, cursor + len(pieces)))
        source_to_model.append(indices)
        mapped_tokens.extend(pieces)
        model_to_source.extend([source_index] * len(pieces))
        cursor += len(pieces)
    if len(model_tokens) != cursor or mapped_tokens != model_tokens:
        raise ValueError("PyABSA sentence tokenization does not have a reversible source mapping")
    return normalized, model_to_source


def _map_span(span: tuple[int, ...], source_to_model: list[list[int]]) -> list[int]:
    mapped = [index for source_index in span for index in source_to_model[source_index]]
    if not mapped or mapped != list(range(mapped[0], mapped[-1] + 1)):
        raise ValueError("span cannot be represented as one contiguous PyABSA span")
    return mapped


def write_pyabsa_explicit_view(
    records: list[HoasaRecord],
    output_path: Path,
    map_path: Path,
) -> dict[str, int]:
    """Write explicit-only PyABSA rows and a text-free reversible offset map.

    Implicit-aspect triplets remain in the canonical source and are counted in
    the sidecar. They are omitted from PyABSA input because version 2.4.3 treats
    index ``-1`` as the last token and would silently relabel them.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    map_path.parent.mkdir(parents=True, exist_ok=True)
    implicit_count = filtered_rows = empty_rows = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as data_stream, map_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as map_stream:
        for record in records:
            normalized, model_to_source = _pyabsa_sentence_map(record.sentence)
            source_to_model: list[list[int]] = [[] for _ in record.sentence.split()]
            for model_index, source_index in enumerate(model_to_source):
                source_to_model[source_index].append(model_index)
            explicit = []
            row_implicit = 0
            for aspect, opinion, sentiment in record.triplets:
                if aspect == (-1,):
                    implicit_count += 1
                    row_implicit += 1
                    continue
                explicit.append(
                    (
                        _map_span(aspect, source_to_model),
                        _map_span(opinion, source_to_model),
                        sentiment,
                    )
                )
            if len(explicit) != len(record.triplets):
                filtered_rows += 1
            if not explicit:
                empty_rows += 1
            serialized = repr([(list(a), list(o), s) for a, o, s in explicit])
            data_stream.write(f"{normalized}####{serialized}\n")
            map_stream.write(
                json.dumps(
                    {
                        "row_id": record.row_id,
                        "source_token_count": len(record.sentence.split()),
                        "model_token_to_source": model_to_source,
                        "implicit_triplets_excluded": row_implicit,
                        "explicit_triplets_retained": len(explicit),
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
    return {
        "sentences": len(records),
        "source_triplets": sum(len(record.triplets) for record in records),
        "implicit_triplets_excluded": implicit_count,
        "rows_with_implicit_triplets": filtered_rows,
        "rows_with_no_explicit_triplets": empty_rows,
        "explicit_triplets_retained": sum(len(record.triplets) for record in records) - implicit_count,
    }


def map_model_span_to_source(
    start: int, end: int, model_token_to_source: list[int]
) -> tuple[int, ...] | None:
    """Map model-token boundaries only when the prediction covers full source tokens."""
    if start < 0 or end < start or end >= len(model_token_to_source):
        return None
    source_positions = model_token_to_source[start : end + 1]
    source_span = tuple(range(source_positions[0], source_positions[-1] + 1))
    expected_model_positions = [
        model_index
        for model_index, source_index in enumerate(model_token_to_source)
        if source_index in source_span
    ]
    if expected_model_positions != list(range(start, end + 1)):
        return None
    return source_span


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare reversible explicit-only PyABSA HoASA views.")
    parser.add_argument("--data", type=Path, default=Path("data/sources/hoasa"))
    parser.add_argument("--output", type=Path, default=Path("data/prepared/hoasa"))
    args = parser.parse_args()
    summary: dict[str, dict[str, int]] = {}
    for split in ("train", "dev", "test"):
        records = load_hoasa(args.data / f"{split}.txt")
        summary[split] = write_pyabsa_explicit_view(
            records,
            args.output / f"{split}.explicit.txt",
            args.output / f"{split}.token-map.jsonl",
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
