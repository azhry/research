"""Exact token-index triplet and term metrics with implicit aspects retained."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


Span = tuple[int, ...]
Triplet = tuple[Span, Span, str]


@dataclass(frozen=True)
class Counts:
    true_positive: int
    predicted: int
    gold: int

    @property
    def precision(self) -> float:
        return self.true_positive / self.predicted if self.predicted else 0.0

    @property
    def recall(self) -> float:
        return self.true_positive / self.gold if self.gold else 0.0

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        return 2 * self.precision * self.recall / denominator if denominator else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "true_positive": self.true_positive,
            "predicted": self.predicted,
            "gold": self.gold,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


def _span(value: Iterable[int]) -> Span:
    result = tuple(int(index) for index in value)
    if not result:
        raise ValueError("span cannot be empty")
    return result


def normalize_triplet(value: Sequence[object]) -> Triplet:
    if len(value) != 3:
        raise ValueError("triplet must contain aspect, opinion, and sentiment")
    aspect, opinion, sentiment = value
    label = str(sentiment).upper()
    if label not in {"POS", "NEG", "NEU"}:
        raise ValueError(f"unsupported sentiment label: {label}")
    aspect_span = _span(aspect)  # type: ignore[arg-type]
    opinion_span = _span(opinion)  # type: ignore[arg-type]
    if aspect_span == (-1,) and any(index < 0 for index in opinion_span):
        raise ValueError("only the aspect may use the implicit [-1] marker")
    return aspect_span, opinion_span, label


def _counts(gold: set[object], predicted: set[object]) -> Counts:
    return Counts(
        true_positive=len(gold & predicted),
        predicted=len(predicted),
        gold=len(gold),
    )


def evaluate_triplets(
    gold_by_sentence: Sequence[Sequence[Sequence[object]]],
    predicted_by_sentence: Sequence[Sequence[Sequence[object]]],
) -> dict[str, object]:
    """Compute exact micro metrics; sentence identity is part of every match.

    Duplicate annotations within a sentence are counted once, following set-based
    exact-match ASTE evaluation. The `[-1]` implicit-aspect tuple stays distinct
    and remains in the primary gold denominator.
    """
    if len(gold_by_sentence) != len(predicted_by_sentence):
        raise ValueError("gold and prediction sentence counts differ")

    gold_triplets: set[tuple[int, Triplet]] = set()
    predicted_triplets: set[tuple[int, Triplet]] = set()
    gold_aspects: set[tuple[int, Span]] = set()
    predicted_aspects: set[tuple[int, Span]] = set()
    gold_opinions: set[tuple[int, Span]] = set()
    predicted_opinions: set[tuple[int, Span]] = set()
    per_polarity_gold: dict[str, set[tuple[int, Triplet]]] = {label: set() for label in ("POS", "NEG", "NEU")}
    per_polarity_predicted: dict[str, set[tuple[int, Triplet]]] = {label: set() for label in ("POS", "NEG", "NEU")}

    for sentence_id, (gold_row, predicted_row) in enumerate(zip(gold_by_sentence, predicted_by_sentence)):
        for raw in gold_row:
            triplet = normalize_triplet(raw)
            aspect, opinion, sentiment = triplet
            key = (sentence_id, triplet)
            gold_triplets.add(key)
            if aspect != (-1,):
                gold_aspects.add((sentence_id, aspect))
            gold_opinions.add((sentence_id, opinion))
            per_polarity_gold[sentiment].add(key)
        for raw in predicted_row:
            triplet = normalize_triplet(raw)
            aspect, opinion, sentiment = triplet
            key = (sentence_id, triplet)
            predicted_triplets.add(key)
            if aspect != (-1,):
                predicted_aspects.add((sentence_id, aspect))
            predicted_opinions.add((sentence_id, opinion))
            per_polarity_predicted[sentiment].add(key)

    by_polarity: dict[str, dict[str, float | int]] = {}
    for label in ("POS", "NEG", "NEU"):
        by_polarity[label] = _counts(
            per_polarity_gold[label], per_polarity_predicted[label]
        ).as_dict()

    implicit_gold = sum(1 for _, (aspect, _, _) in gold_triplets if aspect == (-1,))
    implicit_predicted = sum(1 for _, (aspect, _, _) in predicted_triplets if aspect == (-1,))
    explicit_gold = {key for key in gold_triplets if key[1][0] != (-1,)}
    explicit_predicted = {key for key in predicted_triplets if key[1][0] != (-1,)}

    return {
        "triplet": _counts(gold_triplets, predicted_triplets).as_dict(),
        "aspect_term": _counts(gold_aspects, predicted_aspects).as_dict(),
        "opinion_term": _counts(gold_opinions, predicted_opinions).as_dict(),
        "by_polarity": by_polarity,
        "explicit_only_triplet": _counts(explicit_gold, explicit_predicted).as_dict(),
        "implicit_aspects": {
            "gold": implicit_gold,
            "predicted": implicit_predicted,
            "true_positive": len(gold_triplets & predicted_triplets & {
                key for key in gold_triplets if key[1][0] == (-1,)
            }),
        },
        "sentence_count": len(gold_by_sentence),
    }
