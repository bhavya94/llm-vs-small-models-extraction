"""Adversarial documents: a seen-layout document with one targeted change, and its gold adjusted to match.

Each category tests one way extraction goes wrong in real documents. The change is made through the generator's
field spans, so the edit lands exactly on the value it targets.
"""

from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal

import numpy as np

from extraction_eval_lab.data.generator import CATERERS, BookingRecipe, Document, _money_forms
from extraction_eval_lab.data.schema import Booking

CATEGORIES = {
    "revised_count": ("guest_count", "a later line revises the headcount; the revised number is right"),
    "decoy_amount": ("deposit_amount", "a refundable damage deposit is mentioned before the booking deposit"),
    "numeric_date": ("event_date", "the event date is written as dd/mm/yyyy, and the text says so"),
    "negated_caterer": ("caterer_name", "a caterer is named but not booked; the right value is null"),
    "yearless_due": ("balance_due", "the balance due date has a weekday but no year; the year follows from context"),
}


def _replace_span(doc: Document, field: str, surface: str) -> str:
    start, end = doc.spans[field]
    return doc.text[:start] + surface + doc.text[end:]


def _revised_count(doc: Document, rng) -> tuple[str, dict]:
    new = doc.gold.guest_count + int(rng.integers(5, 40))
    note = f"\nUpdate: the headcount has changed from {doc.gold.guest_count} to {new} guests.\n"
    return doc.text.rstrip("\n") + note, {"guest_count": new}


def _decoy_amount(doc: Document, rng) -> tuple[str, dict]:
    decoy = rng.choice([250, 500, 750])
    note = (f"A refundable damage deposit of {_money_forms(Decimal(int(decoy)), doc.gold.currency)[0]} "
            f"is held on arrival and returned after the event. ")
    start = doc.spans["deposit_amount"][0]
    sentence_start = max(doc.text.rfind(". ", 0, start) + 2, doc.text.rfind("\n", 0, start) + 1)
    return doc.text[:sentence_start] + note + doc.text[sentence_start:], {}


def _numeric_date(doc: Document, rng) -> tuple[str, dict]:
    # Use a day of 12 or less, so day and month could swap and only the stated convention settles it. Moving a later
    # day into the next month keeps the event after both payment dates.
    event = doc.gold.event_date
    if event.day > 12:
        event = (event.replace(day=1) + timedelta(days=32)).replace(day=int(rng.integers(1, 13)))
    text = _replace_span(doc, "event_date", event.strftime("%d/%m/%Y"))
    return "All dates in this message are day/month/year.\n" + text, {"event_date": event}


def _negated_caterer(doc: Document, rng) -> tuple[str, dict]:
    name = CATERERS[int(rng.integers(len(CATERERS)))]
    note = f"\n{name} was considered for catering but is not booked; the client is bringing their own food.\n"
    text = doc.text if doc.gold.caterer_name is None else _replace_span(doc, "caterer_name", "the client")
    return text.rstrip("\n") + note, {"caterer_name": None}


def _yearless_due(doc: Document, rng) -> tuple[str, dict]:
    return _replace_span(doc, "balance_due", doc.gold.balance_due.strftime("%A, %B %-d")), {}


_MAKERS = {"revised_count": _revised_count, "decoy_amount": _decoy_amount, "numeric_date": _numeric_date,
           "negated_caterer": _negated_caterer, "yearless_due": _yearless_due}


@dataclass(frozen=True)
class AdversarialRecipe:
    """`n` documents per category, each built from a fresh seen-layout document."""

    n: int = 40
    seed: int = 5

    def sample(self) -> dict[str, list[Document]]:
        rng = np.random.default_rng(self.seed)
        base = BookingRecipe(n=self.n * len(CATEGORIES), seed=self.seed, layouts="seen").sample()
        out = {}
        for k, category in enumerate(CATEGORIES):
            docs = []
            for doc in base[k * self.n:(k + 1) * self.n]:
                text, changes = _MAKERS[category](doc, rng)
                gold = Booking.model_validate(doc.gold.model_dump() | changes)
                docs.append(Document(id=f"{category}-{len(docs)}", text=text, gold=gold, spans={}))
            out[category] = docs
        return out
