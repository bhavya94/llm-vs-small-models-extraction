from types import SimpleNamespace

import pytest

from extraction_eval_lab.data.generator import BookingRecipe
from extraction_eval_lab.extractors.llm import LLMExtractor, estimate
from extraction_eval_lab.llm.client import BudgetExceeded, CachedLLM, Meter

DOC = BookingRecipe(n=1, seed=0).sample()[0]
GOOD = DOC.gold.model_dump_json()
BAD = GOOD.replace('"currency":"CAD"', '"currency":"dollars"')


class FakeClient:
    """Replies with the queued texts in order and records every request."""

    def __init__(self, *replies):
        self.replies, self.requests = list(replies), []
        self.messages = self

    def create(self, **request):
        self.requests.append(request)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.replies.pop(0))],
                               usage=SimpleNamespace(input_tokens=1000, output_tokens=100))


def _extractor(client, tmp_path, cap=1.0, repairs=1):
    meter = Meter(input_price=1.0, output_price=5.0, cap=cap)
    return LLMExtractor(CachedLLM(client, "test-model", meter, cache=tmp_path), repairs=repairs)


def test_a_valid_reply_is_returned_after_one_call(tmp_path):
    client = FakeClient(GOOD)
    extractor = _extractor(client, tmp_path)
    assert extractor.extract(DOC.text) == DOC.gold.model_dump(mode="json")
    assert extractor.llm.calls == 1 and extractor.llm.meter.spent == pytest.approx((1000 * 1 + 100 * 5) / 1e6)


def test_an_invalid_reply_gets_one_repair_with_the_validation_error(tmp_path):
    client = FakeClient(BAD, GOOD)
    extractor = _extractor(client, tmp_path)
    assert extractor.extract(DOC.text)["currency"] == "CAD"
    repair = client.requests[1]["messages"]
    assert [m["role"] for m in repair] == ["user", "assistant", "user"] and "currency" in repair[2]["content"]
    assert extractor.repaired == 1


def test_repairs_stop_at_the_limit_and_return_the_last_attempt(tmp_path):
    client = FakeClient(BAD, BAD)
    extractor = _extractor(client, tmp_path)
    assert extractor.extract(DOC.text)["currency"] == "dollars" and extractor.llm.calls == 2


def test_the_cache_makes_a_rerun_free(tmp_path):
    first = _extractor(FakeClient(GOOD), tmp_path)
    first.extract(DOC.text)
    second = _extractor(FakeClient(), tmp_path)  # no replies queued: any API call would fail
    assert second.extract(DOC.text) == DOC.gold.model_dump(mode="json") and second.llm.calls == 0


def test_the_budget_refuses_a_call_that_could_pass_the_cap(tmp_path):
    extractor = _extractor(FakeClient(GOOD), tmp_path, cap=0.001)
    with pytest.raises(BudgetExceeded, match="cap"):
        extractor.extract(DOC.text)
    assert extractor.llm.calls == 0


def test_estimate_scales_with_documents_and_repairs():
    meter = Meter(input_price=1.0, output_price=5.0, cap=10)
    one = estimate([DOC.text], meter, max_tokens=400, repair_share=0)
    assert estimate([DOC.text] * 10, meter, max_tokens=400, repair_share=0) == pytest.approx(10 * one)
    assert estimate([DOC.text], meter, max_tokens=400, repair_share=0.5) == pytest.approx(1.5 * one)


def test_cost_and_latency_survive_a_cached_rerun(tmp_path):
    first = _extractor(FakeClient(GOOD), tmp_path)
    first.extract(DOC.text)
    second = _extractor(FakeClient(), tmp_path)
    second.extract(DOC.text)
    assert second.llm.meter.spent == 0
    assert second.llm.dollars_uncached() == pytest.approx(first.llm.meter.spent)
    assert second.llm.replies[0]["seconds"] == first.llm.replies[0]["seconds"] is not None
