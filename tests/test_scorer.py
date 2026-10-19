from datetime import date

import pytest

from extraction_eval_lab.data.generator import BookingRecipe
from extraction_eval_lab.evaluation.scorer import FieldReport, field_correct, is_valid, score
from extraction_eval_lab.data.schema import FIELDS

GOLD = BookingRecipe(n=1, seed=0).sample()[0].gold  # caterer and cancellation are absent in this one
PERFECT = GOLD.model_dump(mode="json")


def test_a_perfect_prediction_scores_one_and_is_valid():
    assert score(PERFECT, GOLD) == 1.0 and is_valid(PERFECT)


def test_values_compare_as_typed_values_not_strings():
    loose = {**PERFECT, "booker_name": "  rosa   LINDQVIST ", "deposit_amount": "1700",
             "event_date": date(2027, 7, 6), "guest_count": "16"}
    assert all(field_correct(loose, GOLD).values())


def test_a_wrong_or_unreadable_value_costs_its_field_only():
    wrong = {**PERFECT, "deposit_amount": "CAD 1,700.00", "event_date": "06/07/2027"}
    correct = field_correct(wrong, GOLD)
    assert [name for name, ok in correct.items() if not ok] == ["event_date", "deposit_amount"]
    assert score(wrong, GOLD) == pytest.approx(11 / 13)


def test_inventing_an_optional_field_is_wrong_and_omitting_it_is_right():
    assert not field_correct({**PERFECT, "caterer_name": "Bramble Kitchen"}, GOLD)["caterer_name"]
    assert field_correct({k: v for k, v in PERFECT.items() if k != "caterer_name"}, GOLD)["caterer_name"]


def test_missing_prediction_scores_zero_and_extra_keys_fail_validity_only():
    assert score(None, GOLD) == 0.0 and score({}, GOLD) == 0.0
    extra = {**PERFECT, "notes": "late arrival"}
    assert score(extra, GOLD) == 1.0 and not is_valid(extra)


def test_report_counts_a_wrong_value_as_both_false_positive_and_false_negative():
    wrong_name = {**PERFECT, "venue_name": "Somewhere Else"}
    report = FieldReport.from_pairs([PERFECT, wrong_name], [GOLD, GOLD])
    venue = report.fields["venue_name"]
    assert (venue.precision, venue.recall) == (0.5, 0.5)
    assert report.exact_match == 0.5 and report.per_document == [1.0, pytest.approx(12 / 13)]
    assert report.fields["caterer_name"].f1 == 1.0  # never present, never predicted: nothing to get wrong


def test_report_needs_one_prediction_per_document():
    with pytest.raises(ValueError, match="1 predictions for 2"):
        FieldReport.from_pairs([PERFECT], [GOLD, GOLD])


def test_fields_list_matches_the_schema():
    assert set(FIELDS) == set(GOLD.model_dump())
