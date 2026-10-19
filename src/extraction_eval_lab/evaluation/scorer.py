"""Scoring an extraction against the gold booking, per field and per document.

`score(predicted, gold)` is a pure function from one prediction to a number in [0, 1], so it can also serve as a
reward for training. `FieldReport` aggregates many documents into per-field precision, recall and F1.
"""

from dataclasses import dataclass
from typing import Any

from pydantic import TypeAdapter, ValidationError

from extraction_eval_lab.data.schema import FIELDS, Booking

_ADAPTERS = {name: TypeAdapter(info.annotation) for name, info in Booking.model_fields.items()}


def _normalize(name: str, value: Any) -> Any:
    """The comparable form of one predicted value, or a sentinel when it cannot be read as the field's type.

    Strings compare case-insensitively with whitespace collapsed; dates, amounts and counts compare as typed values,
    so "2027-03-04" equals date(2027, 3, 4) and "1700.00" equals Decimal("1700"). Nothing else is forgiven: a
    currency symbol inside an amount, or a date in another format, is wrong.
    """
    if value is None:
        return None
    try:
        typed = _ADAPTERS[name].validate_python(value)
    except ValidationError:
        return _UNREADABLE
    return " ".join(typed.split()).casefold() if isinstance(typed, str) else typed


_UNREADABLE = object()


def field_correct(predicted: dict | None, gold: Booking) -> dict[str, bool]:
    """Per field: does the prediction hold the gold value? An absent optional field is correct when gold is absent."""
    predicted = predicted or {}
    expected = gold.model_dump()
    return {name: _normalize(name, predicted.get(name)) == _normalize(name, expected[name]) for name in FIELDS}


def score(predicted: dict | None, gold: Booking) -> float:
    """The share of fields extracted correctly, in [0, 1]. A missing or unparseable prediction scores 0."""
    if not predicted:
        return 0.0
    correct = field_correct(predicted, gold)
    return sum(correct.values()) / len(correct)


def is_valid(predicted: dict | None) -> bool:
    """Whether the prediction passes the strict schema as a whole: right keys, right types, no extras."""
    try:
        Booking.model_validate(predicted)
        return True
    except ValidationError:
        return False


@dataclass(frozen=True)
class FieldScores:
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class FieldReport:
    """Extraction metrics over many documents.

    Per field, a prediction is a true positive when it states the gold value, a false positive when it states any
    other value (or one where gold has none), and a false negative when gold has a value the prediction misses or
    gets wrong. A wrong value is therefore both a false positive and a false negative.
    """

    fields: dict[str, FieldScores]
    exact_match: float  # share of documents with every field right
    valid: float  # share of predictions that pass the strict schema
    per_document: list[float]  # score() for each document, for paired comparisons

    @classmethod
    def from_pairs(cls, predictions: list[dict | None], golds: list[Booking]) -> "FieldReport":
        if len(predictions) != len(golds):
            raise ValueError(f"{len(predictions)} predictions for {len(golds)} documents")
        counts = {name: [0, 0, 0] for name in FIELDS}  # tp, fp, fn
        for predicted, gold in zip(predictions, golds):
            expected, given = gold.model_dump(), predicted or {}
            for name, right in field_correct(predicted, gold).items():
                stated = _normalize(name, given.get(name)) is not None
                if right and expected[name] is not None:
                    counts[name][0] += 1
                elif not right:
                    counts[name][1] += stated
                    counts[name][2] += expected[name] is not None
        fields = {}
        for name, (tp, fp, fn) in counts.items():
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0
            fields[name] = FieldScores(precision, recall, f1)
        per_document = [score(p, g) for p, g in zip(predictions, golds)]
        return cls(fields=fields, exact_match=sum(s == 1.0 for s in per_document) / len(golds),
                   valid=sum(map(is_valid, predictions)) / len(golds), per_document=per_document)
