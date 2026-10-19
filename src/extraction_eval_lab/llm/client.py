"""One cached, budget-capped way to call a model. Shared by the LLM extractor and the judge.

Every reply is cached on disk by a hash of the request, so a rerun costs nothing, and a hard cap refuses any call
that could take spend past the budget. The client is passed in, so tests use a fake one and never call the API.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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
class CachedLLM:
    client: Any  # anything with .messages.create(...) shaped like the Anthropic SDK's
    model: str
    meter: Meter
    cache: Path = Path(".cache") / "llm"
    max_tokens: int = 400
    calls: int = field(default=0, init=False)  # API calls actually made, cache hits excluded
    # Tokens and seconds of every reply, cache hits included, so cost and latency per document survive a cached
    # rerun. Seconds are None for replies cached before latency was recorded.
    replies: list[dict] = field(default_factory=list, init=False)

    def ask(self, system: str, messages: list[dict]) -> str:
        request = {"model": self.model, "system": system, "messages": messages, "max_tokens": self.max_tokens}
        key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        path = self.cache / f"{key}.json"
        if path.exists():
            cached = json.loads(path.read_text())
            self._log(*cached["usage"], cached.get("seconds"))
            return cached["text"]
        self.meter.check(self.max_tokens, input_tokens_estimate=len(json.dumps(request)) // 3)
        start = time.perf_counter()
        response = self.client.messages.create(**request)
        seconds = time.perf_counter() - start
        self.calls += 1
        self.meter.add(response.usage.input_tokens, response.usage.output_tokens)
        text = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
        path.parent.mkdir(parents=True, exist_ok=True)
        usage = [response.usage.input_tokens, response.usage.output_tokens]
        path.write_text(json.dumps({"text": text, "usage": usage, "seconds": seconds}))
        self._log(*usage, seconds)
        return text

    def _log(self, input_tokens: int, output_tokens: int, seconds: float | None) -> None:
        self.replies.append({"input_tokens": input_tokens, "output_tokens": output_tokens, "seconds": seconds})

    def dollars_uncached(self) -> float:
        """What every reply so far would cost with an empty cache."""
        return sum(r["input_tokens"] * self.meter.input_price + r["output_tokens"] * self.meter.output_price
                   for r in self.replies) / 1e6
