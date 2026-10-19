"""Turning a surface string into a schema value. Shared by every extractor that finds text spans first."""

import re
from datetime import datetime

_WEEKDAY = re.compile(r"^[A-Z][a-z]+day, ")
_DATE_FORMS = ("%Y-%m-%d", "%B %d, %Y", "%d %b %Y")
_AMOUNT = re.compile(r"\d[\d,]*\.\d{2}")


def parse_date(surface: str) -> str | None:
    """ISO date from "2027-03-04", "March 4, 2027", "4 Mar 2027" or "Thursday, March 4, 2027"; else None."""
    surface = _WEEKDAY.sub("", surface.strip())
    for form in _DATE_FORMS:
        try:
            return datetime.strptime(surface, form).date().isoformat()
        except ValueError:
            pass
    return None


def parse_amount(surface: str) -> str | None:
    """The number in "$1,700.00", "CAD 1,700.00" or "1,700.00 CAD", without symbol or commas; else None."""
    match = _AMOUNT.search(surface)
    return match.group(0).replace(",", "") if match else None


def parse_int(surface: str) -> int | None:
    match = re.search(r"\d+", surface)
    return int(match.group(0)) if match else None


def parse_field(name: str, surface: str):
    """The value for field `name` from its surface text, typed the way the schema expects."""
    if name.endswith(("_date", "_due")):
        return parse_date(surface)
    if name.endswith("_amount"):
        return parse_amount(surface)
    if name in ("guest_count", "cancellation_days"):
        return parse_int(surface)
    return surface.strip() or None
