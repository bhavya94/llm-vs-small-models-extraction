"""Fictional venue booking confirmations, with the gold JSON and the character span of every field value.

All names, venues and emails are invented. Each document is rendered from one of several layouts, and dates and
amounts take one of several surface forms, so an extractor must normalize, not copy. The spans let a token tagger
learn BIO labels from the same documents.

Layouts come in two groups. The three "seen" layouts are for training and tuning. The two "unseen" layouts use
other wording and another order of facts (the balance before the deposit, for one), so a test on them measures
whether an extractor learned the task or memorized the seen layouts.
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal

import numpy as np

from extraction_eval_lab.data.schema import Booking

FIRST = ["Amara", "Bastien", "Chiara", "Dmitri", "Elif", "Farid", "Greta", "Hiroshi", "Ines", "Jonah", "Kalani",
         "Lucia", "Mateo", "Nadia", "Oskar", "Priya", "Quentin", "Rosa", "Soren", "Talia"]
LAST = ["Abernathy", "Brightwater", "Castellane", "Dunmore", "Everhart", "Fairweather", "Galloway", "Holloway",
        "Ivanova", "Kingsley", "Lindqvist", "Marchetti", "Northcott", "Okafor", "Pemberton", "Quill", "Rowntree"]
VENUES = ["The Copper Lantern", "Willowmere Hall", "Harbor Light Loft", "The Glasshouse", "Old Mill Ballroom",
          "Juniper Terrace", "The Foundry Rooms", "Maple & Stone", "Saltmarsh Pavilion", "The Orchard Barn"]
CITIES = ["Port Ellery", "Northbridge", "Calder Falls", "Westmere", "Ashford Bay", "Riverton", "Lowhaven"]
CATERERS = ["Bramble Kitchen", "Silver Spoon Catering", "Fennel & Fig", "The Hungry Heron", "Copperpot Events"]
CURRENCIES = {"USD": "$", "CAD": "C$", "EUR": "€", "GBP": "£"}


def _date_forms(value: date) -> list[str]:
    """Unambiguous forms only: each states the year and spells the month or orders it year-first."""
    return [value.strftime("%B %-d, %Y"), value.isoformat(), value.strftime("%-d %b %Y"),
            value.strftime("%A, %B %-d, %Y")]


def _money_forms(amount: Decimal, currency: str) -> list[str]:
    plain = f"{amount:,.2f}"
    return [f"{CURRENCIES[currency]}{plain}", f"{currency} {plain}", f"{plain} {currency}"]


@dataclass
class _Writer:
    """Builds text piece by piece and remembers where each field's value landed."""

    text: str = ""
    spans: dict[str, tuple[int, int]] = field(default_factory=dict)

    def add(self, piece: str, field_name: str | None = None) -> "_Writer":
        if field_name:
            self.spans[field_name] = (len(self.text), len(self.text) + len(piece))
        self.text += piece
        return self


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    gold: Booking
    spans: dict[str, tuple[int, int]]  # field -> [start, end) in `text`; only fields the text states


@dataclass(frozen=True)
class BookingRecipe:
    """A recipe for `n` documents. The same seed always gives the same documents."""

    n: int = 300
    seed: int = 0
    optional_rate: float = 0.5  # chance each optional field is present
    layouts: str = "seen"  # "seen", "unseen", or "all"

    def __post_init__(self):
        if self.layouts not in LAYOUTS:
            raise ValueError(f"layouts must be one of {sorted(LAYOUTS)}, got {self.layouts!r}")

    def sample(self) -> list[Document]:
        rng = np.random.default_rng(self.seed)
        return [self._one(rng, i) for i in range(self.n)]

    def _one(self, rng: np.random.Generator, i: int) -> Document:
        def pick(options):
            return options[int(rng.integers(len(options)))]

        first, last = pick(FIRST), pick(LAST)
        event = date(2027, 1, 1) + timedelta(days=int(rng.integers(0, 365)))
        currency = pick(list(CURRENCIES))
        total = Decimal(int(rng.integers(20, 400)) * 50)
        deposit = (total * Decimal("0.25")).quantize(Decimal("0.01"))
        gold = Booking(
            booker_name=f"{first} {last}", booker_email=f"{first}.{last}@example.com".lower(),
            venue_name=pick(VENUES), venue_city=pick(CITIES), event_date=event,
            guest_count=int(rng.integers(12, 300)), currency=currency,
            deposit_amount=deposit, deposit_due=event - timedelta(days=int(rng.integers(30, 120))),
            balance_amount=(total - deposit).quantize(Decimal("0.01")),
            balance_due=event - timedelta(days=int(rng.integers(3, 21))),
            caterer_name=pick(CATERERS) if rng.random() < self.optional_rate else None,
            cancellation_days=int(rng.choice([7, 14, 30, 60])) if rng.random() < self.optional_rate else None)
        form = {"date": lambda d: pick(_date_forms(d)), "money": lambda m: pick(_money_forms(m, currency))}
        layout = pick(LAYOUTS[self.layouts])
        writer = layout(gold, form)
        return Document(id=f"doc-{self.seed}-{i}", text=writer.text, gold=gold, spans=writer.spans)


def _email_layout(g: Booking, form) -> _Writer:
    w = _Writer().add("Subject: Your booking at ").add(g.venue_name, "venue_name").add(" is confirmed\n\nDear ")
    w.add(g.booker_name, "booker_name").add(",\n\nThank you for choosing ").add(g.venue_name).add(" in ")
    w.add(g.venue_city, "venue_city").add(". We are delighted to host your event on ")
    w.add(form["date"](g.event_date), "event_date").add(" for ").add(str(g.guest_count), "guest_count")
    w.add(" guests.\n\nTo hold the date, a deposit of ").add(form["money"](g.deposit_amount), "deposit_amount")
    w.add(" is due by ").add(form["date"](g.deposit_due), "deposit_due").add(". The remaining balance of ")
    w.add(form["money"](g.balance_amount), "balance_amount").add(" is due on ")
    w.add(form["date"](g.balance_due), "balance_due").add(". All amounts are in ").add(g.currency, "currency")
    w.add(".\n")
    if g.caterer_name:
        w.add("Your caterer, ").add(g.caterer_name, "caterer_name").add(", will contact you about the menu.\n")
    if g.cancellation_days is not None:
        w.add("Cancellations require ").add(str(g.cancellation_days), "cancellation_days").add(" days' notice.\n")
    w.add("\nWe will send all updates to ").add(g.booker_email, "booker_email")
    return w.add(".\n\nWarm regards,\nEvents team\n")


def _invoice_layout(g: Booking, form) -> _Writer:
    w = _Writer().add("BOOKING CONFIRMATION\n").add("Venue: ").add(g.venue_name, "venue_name").add(", ")
    w.add(g.venue_city, "venue_city").add("\nClient: ").add(g.booker_name, "booker_name").add(" <")
    w.add(g.booker_email, "booker_email").add(">\nEvent date: ").add(form["date"](g.event_date), "event_date")
    w.add("\nHeadcount: ").add(str(g.guest_count), "guest_count").add("\nCurrency: ").add(g.currency, "currency")
    w.add("\n\nDeposit ........ ").add(form["money"](g.deposit_amount), "deposit_amount").add("  due ")
    w.add(form["date"](g.deposit_due), "deposit_due").add("\nBalance ........ ")
    w.add(form["money"](g.balance_amount), "balance_amount").add("  due ")
    w.add(form["date"](g.balance_due), "balance_due")
    if g.caterer_name:
        w.add("\nCatering: ").add(g.caterer_name, "caterer_name")
    if g.cancellation_days is not None:
        w.add("\nCancellation notice: ").add(str(g.cancellation_days), "cancellation_days").add(" days")
    return w.add("\n")


def _note_layout(g: Booking, form) -> _Writer:
    w = _Writer().add("Hi ").add(g.booker_name, "booker_name").add(" - quick note to confirm ")
    w.add(g.venue_name, "venue_name").add(" (").add(g.venue_city, "venue_city").add(") is yours on ")
    w.add(form["date"](g.event_date), "event_date").add(", ").add(str(g.guest_count), "guest_count")
    w.add(" people. Deposit ").add(form["money"](g.deposit_amount), "deposit_amount").add(" by ")
    w.add(form["date"](g.deposit_due), "deposit_due").add(", then ")
    w.add(form["money"](g.balance_amount), "balance_amount")
    w.add(" by ").add(form["date"](g.balance_due), "balance_due").add(" (").add(g.currency, "currency").add(").")
    if g.caterer_name:
        w.add(" Food is by ").add(g.caterer_name, "caterer_name").add(".")
    if g.cancellation_days is not None:
        w.add(" Need ").add(str(g.cancellation_days), "cancellation_days").add(" days' notice to cancel.")
    return w.add(" Replies go to ").add(g.booker_email, "booker_email").add(".\n")


def _chat_layout(g: Booking, form) -> _Writer:
    w = _Writer().add("[events desk] booking locked in for ").add(g.booker_name, "booker_name")
    w.add("\n[events desk] place: ").add(g.venue_name, "venue_name").add(" / ").add(g.venue_city, "venue_city")
    w.add("\n[events desk] balance of ").add(form["money"](g.balance_amount), "balance_amount").add(" lands on ")
    w.add(form["date"](g.balance_due), "balance_due").add(", but first the deposit, ")
    w.add(form["money"](g.deposit_amount), "deposit_amount").add(", before ")
    w.add(form["date"](g.deposit_due), "deposit_due")
    w.add("\n[events desk] party size ").add(str(g.guest_count), "guest_count").add(", the big day is ")
    w.add(form["date"](g.event_date), "event_date").add(", billing in ").add(g.currency, "currency")
    if g.cancellation_days is not None:
        w.add("\n[events desk] cancel at least ").add(str(g.cancellation_days), "cancellation_days").add(" days ahead")
    if g.caterer_name:
        w.add("\n[events desk] kitchen: ").add(g.caterer_name, "caterer_name")
    return w.add("\n[events desk] cc ").add(g.booker_email, "booker_email").add("\n")


def _letter_layout(g: Booking, form) -> _Writer:
    w = _Writer().add("Re: reservation for ").add(g.booker_name, "booker_name").add(" (")
    w.add(g.booker_email, "booker_email").add(")\n\nThis letter confirms that ").add(g.venue_name, "venue_name")
    w.add(", of ").add(g.venue_city, "venue_city").add(", has reserved its space for a party of ")
    w.add(str(g.guest_count), "guest_count").add(". The final payment, ")
    w.add(form["money"](g.balance_amount), "balance_amount")
    w.add(", must reach us by ").add(form["date"](g.balance_due), "balance_due")
    w.add("; the event itself takes place on ")
    w.add(form["date"](g.event_date), "event_date").add(". A retainer of ")
    w.add(form["money"](g.deposit_amount), "deposit_amount")
    w.add(" was requested for ").add(form["date"](g.deposit_due), "deposit_due").add(". Figures are quoted in ")
    w.add(g.currency, "currency").add(".")
    if g.caterer_name:
        w.add(" Catering will be provided by ").add(g.caterer_name, "caterer_name").add(".")
    if g.cancellation_days is not None:
        w.add(" The reservation may be released with ").add(str(g.cancellation_days), "cancellation_days")
        w.add(" days of written notice.")
    return w.add("\n\nYours faithfully,\nReservations\n")


LAYOUTS = {"seen": [_email_layout, _invoice_layout, _note_layout], "unseen": [_chat_layout, _letter_layout]}
LAYOUTS["all"] = LAYOUTS["seen"] + LAYOUTS["unseen"]
