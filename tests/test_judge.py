import json
from types import SimpleNamespace

import pytest

from extraction_eval_lab.data.generator import BookingRecipe
from extraction_eval_lab.data.schema import FIELDS
from extraction_eval_lab.evaluation.judge import Agreement, RubricJudge
from extraction_eval_lab.llm.client import CachedLLM, Meter

DOC = BookingRecipe(n=1, seed=0).sample()[0]
ALL_TRUE = {name: True for name in FIELDS}


class FakeClient:
    def __init__(self, reply):
        self.reply, self.messages, self.seen = reply, self, []

    def create(self, **request):
        self.seen.append(request)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.reply)],
                               usage=SimpleNamespace(input_tokens=800, output_tokens=120))


def _judge(reply, tmp_path):
    client = FakeClient(reply)
    return RubricJudge(CachedLLM(client, "m", Meter(1.0, 5.0, cap=1.0), cache=tmp_path)), client


def test_verdicts_parse_a_full_true_false_object_and_never_show_gold(tmp_path):
    judge, client = _judge("Here you go: " + json.dumps(ALL_TRUE), tmp_path)
    assert judge.verdicts(DOC.text, {"guest_count": 16}) == ALL_TRUE
    sent = client.seen[0]["messages"][0]["content"]
    assert DOC.text in sent and '"guest_count": 16' in sent
    assert DOC.gold.booker_email not in sent.split("Extraction:")[1]  # the judge sees the prediction, not the gold


@pytest.mark.parametrize("reply", ["no json", json.dumps({"guest_count": True}),
                                   json.dumps({**ALL_TRUE, "guest_count": "yes"})])
def test_unusable_replies_give_none(reply, tmp_path):
    judge, _ = _judge(reply, tmp_path)
    assert judge.verdicts(DOC.text, {}) is None


def test_agreement_counts_tpr_tnr_and_skips_unparsed():
    truth = [{**ALL_TRUE, "guest_count": False}, ALL_TRUE]
    judged = [ALL_TRUE, None]  # accepts every field, including the wrong guest count; second reply unusable
    agreement = Agreement.of(truth, judged)
    assert agreement.pairs == 13 and agreement.unparsed == 1
    assert agreement.tpr == 1.0 and agreement.tnr == 0.0 and agreement.kappa == pytest.approx(0.0)
