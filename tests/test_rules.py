from extraction_eval_lab.data.generator import BookingRecipe
from extraction_eval_lab.evaluation.scorer import FieldReport
from extraction_eval_lab.extractors.rules import extract


def _report(layouts):
    docs = BookingRecipe(n=150, seed=7, layouts=layouts).sample()
    return FieldReport.from_pairs([extract(d.text) for d in docs], [d.gold for d in docs])


def test_rules_are_perfect_on_the_layouts_they_were_written_for():
    assert _report("seen").exact_match == 1.0


def test_rules_break_on_unseen_layouts():
    assert _report("unseen").exact_match < 0.5
