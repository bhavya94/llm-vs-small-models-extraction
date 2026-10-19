"""An LLM extractor with schema validation and a repair loop.

1. Ask the model for the booking as JSON, with the schema in the system prompt.
2. Validate the reply against the strict schema. If it fails, send the validation errors back and ask for a
   corrected object, up to `repairs` times.

Calls go through `CachedLLM`, which caches replies and enforces the budget.
"""

import json
from dataclasses import dataclass, field

from pydantic import ValidationError

from extraction_eval_lab.data.schema import Booking
from extraction_eval_lab.extractors.parsing import parse_reply
from extraction_eval_lab.llm.client import CachedLLM, Meter

SYSTEM = ("You extract venue bookings from documents. Reply with one JSON object that matches this JSON schema, "
          "and nothing else. Use null for a field the document does not state.\n\n"
          + json.dumps(Booking.model_json_schema()))


@dataclass
class LLMExtractor:
    llm: CachedLLM
    repairs: int = 1
    repaired: int = field(default=0, init=False)  # documents that needed at least one repair

    def extract(self, text: str) -> dict | None:
        messages = [{"role": "user", "content": text}]
        reply = self._ask(messages)
        for attempt in range(self.repairs + 1):
            predicted = parse_reply(reply)
            error = _validation_error(predicted)
            if error is None or attempt == self.repairs:
                return predicted
            if attempt == 0:
                self.repaired += 1
            fix = f"That JSON is invalid: {error}\nReply with the corrected JSON only."
            messages += [{"role": "assistant", "content": reply}, {"role": "user", "content": fix}]
            reply = self._ask(messages)
        return predicted

    def _ask(self, messages: list[dict]) -> str:
        return self.llm.ask(SYSTEM, messages)


def _validation_error(predicted: dict | None) -> str | None:
    if predicted is None:
        return "the reply holds no JSON object"
    try:
        Booking.model_validate(predicted)
        return None
    except ValidationError as error:
        return "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors())


def estimate(texts: list[str], meter: Meter, max_tokens: int, repair_share: float = 0.2) -> float:
    """A rough upper estimate of dollars to extract `texts`: input at 1 token per 3 characters, full output length,
    and a repair call for `repair_share` of documents."""
    per_call = [(len(SYSTEM) + len(text)) / 3 * meter.input_price / 1e6 + max_tokens * meter.output_price / 1e6
                for text in texts]
    return sum(per_call) * (1 + repair_share)
