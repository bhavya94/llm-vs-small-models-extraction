"""The extraction target: a venue booking confirmation as strict, typed JSON."""

from datetime import date
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, EmailStr, Field

# Every field an extractor can be scored on, in schema order. Optional fields may be absent from a document.
FIELDS = ["booker_name", "booker_email", "venue_name", "venue_city", "event_date", "guest_count", "currency",
          "deposit_amount", "deposit_due", "balance_amount", "balance_due", "caterer_name", "cancellation_days"]
OPTIONAL = {"caterer_name", "cancellation_days"}


class Booking(BaseModel):
    """One booking. `extra="forbid"` rejects any key the schema does not name, so a sloppy extractor fails loudly."""

    model_config = ConfigDict(extra="forbid")

    booker_name: str
    booker_email: EmailStr
    venue_name: str
    venue_city: str
    event_date: date
    guest_count: int = Field(gt=0)
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    deposit_amount: Decimal = Field(ge=0, decimal_places=2)
    deposit_due: date
    balance_amount: Decimal = Field(ge=0, decimal_places=2)
    balance_due: date
    caterer_name: str | None = None
    cancellation_days: int | None = Field(default=None, ge=0)
