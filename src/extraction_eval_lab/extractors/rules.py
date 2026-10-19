"""A rule-based extractor: regular expressions written against the generator's three layouts.

It is the free baseline every model must beat. It knows the layouts' wording and their order of facts (event date,
then deposit, then balance), so it should be near perfect on them and break on anything else.
"""

import re

from extraction_eval_lab.extractors.parsing import parse_amount, parse_date

_DATE = re.compile(r"\d{4}-\d{2}-\d{2}"                                    # 2027-03-04
                   r"|(?:[A-Z][a-z]+day, )?[A-Z][a-z]+ \d{1,2}, \d{4}"       # (Thursday, )March 4, 2027
                   r"|\d{1,2} [A-Z][a-z]{2} \d{4}")                         # 4 Mar 2027
_MONEY = re.compile(r"\d[\d,]*\.\d{2}")
_PATTERNS = {
    "booker_email": r"[\w.+-]+@[\w-]+\.[\w.]+\w",
    "currency": r"\b(USD|CAD|EUR|GBP)\b",
    "booker_name": r"Dear (.+?),|Client: (.+?) <|^Hi (.+?) - ",
    "venue_name": r"booking at (.+?) is confirmed|Venue: (.+?), |confirm (.+?) \(",
    "venue_city": r" in ([^.]+)\. We are|Venue: [^,\n]+, (.+)|\(([^)]+)\) is yours",
    "guest_count": r"(\d+) (?:guests|people)|Headcount: (\d+)",
    "caterer_name": r"Your caterer, (.+?), will|Catering: (.+)|Food is by (.+?)\.",
    "cancellation_days": r"(\d+) days' notice|Cancellation notice: (\d+)",
}


def _first(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        return None
    groups = [g for g in match.groups() if g] if match.groups() else []
    return groups[0] if groups else match.group(0)


def extract(text: str) -> dict:
    """A prediction dict in the schema's shape. Fields the rules cannot find are None."""
    found = {name: _first(pattern, text) for name, pattern in _PATTERNS.items()}
    dates = [parse_date(d) for d in _DATE.findall(text)]
    amounts = [parse_amount(m) for m in _MONEY.findall(text)]
    found["event_date"], found["deposit_due"], found["balance_due"] = (dates + [None] * 3)[:3]
    found["deposit_amount"], found["balance_amount"] = (amounts + [None] * 2)[:2]
    for name in ("guest_count", "cancellation_days"):
        found[name] = int(found[name]) if found[name] else None
    return found
