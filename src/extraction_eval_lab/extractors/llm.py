"""An LLM extractor with schema validation, a repair loop, a response cache and a hard budget.

1. Ask the model for the booking as JSON, with the schema in the system prompt.
2. Validate the reply against the strict schema. If it fails, send the validation errors back and ask for a
   corrected object, up to `repairs` times.
3. Cache every reply on disk by a hash of the request, so a rerun costs nothing.
4. Count tokens and dollars; refuse any call that could take spend past the budget.

The client is passed in, so tests run against a fake one and never call the API.
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from extraction_eval_lab.data.schema import Booking
from extraction_eval_lab.extractors.parsing import parse_reply

SYSTEM = ("You extract venue bookings from documents. Reply with one JSON object that matches this JSON schema, "
          "and nothing else. Use null for a field the document does not state.\n\n"
          + json.dumps(Booking.model_json_schema()))


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Meter:
    """Spend so far, at the given prices in dollars per million tokens, against a hard cap."""

    input_price: float
    output_price: float
    cap: float
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def spent(self) -> float:
        return (self.input_tokens * self.input_price + self.output_tokens * self.output_price) / 1e6

    def check(self, max_output_tokens: int, input_tokens_estimate: int) -> None:
        """Refuse a call whose worst case (estimated input, every output token used) would pass the cap."""
        worst = (input_tokens_estimate * self.input_price + max_output_tokens * self.output_price) / 1e6
        if self.spent + worst > self.cap:
            raise BudgetExceeded(f"spent ${self.spent:.4f}; the next call could cost ${worst:.4f}; cap ${self.cap:.2f}")

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens


@dataclass
class LLMExtractor:
    client: Any  # anything with .messages.create(...) shaped like the Anthropic SDK's
    model: str
    meter: Meter
    cache: Path = Path(".cache") / "llm"
    repairs: int = 1
    max_tokens: int = 400
    calls: int = field(default=0, init=False)  # API calls actually made, cache hits excluded
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
        request = {"model": self.model, "system": SYSTEM, "messages": messages, "max_tokens": self.max_tokens}
        key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        path = self.cache / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text())["text"]
        self.meter.check(self.max_tokens, input_tokens_estimate=len(json.dumps(request)) // 3)
        response = self.client.messages.create(**request)
        self.calls += 1
        self.meter.add(response.usage.input_tokens, response.usage.output_tokens)
        text = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"text": text, "usage": [response.usage.input_tokens,
                                                            response.usage.output_tokens]}))
        return text


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
