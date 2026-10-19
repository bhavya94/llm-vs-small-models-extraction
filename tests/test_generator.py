import pytest
from pydantic import ValidationError

from extraction_eval_lab.data.generator import BookingRecipe, _date_forms, _money_forms
from extraction_eval_lab.data.schema import FIELDS, OPTIONAL, Booking

DOCS = BookingRecipe(n=200, seed=0, layouts="all").sample()


def test_same_seed_same_documents_and_new_seed_new_ones():
    again = BookingRecipe(n=200, seed=0, layouts="all").sample()
    assert [d.text for d in again] == [d.text for d in DOCS]
    assert BookingRecipe(n=5, seed=1).sample()[0].text != DOCS[0].text


def test_every_span_holds_a_surface_form_of_its_gold_value():
    for doc in DOCS:
        gold = doc.gold.model_dump()
        for name, (start, end) in doc.spans.items():
            surface, value = doc.text[start:end], gold[name]
            if name.endswith(("_date", "_due")):
                assert surface in _date_forms(value), (doc.id, name, surface)
            elif name.endswith("_amount"):
                assert surface in _money_forms(value, doc.gold.currency), (doc.id, name, surface)
            else:
                assert surface == str(value), (doc.id, name, surface)


def test_required_fields_always_appear_and_optional_ones_only_when_present():
    for doc in DOCS:
        gold = doc.gold.model_dump()
        for name in FIELDS:
            assert (name in doc.spans) == (gold[name] is not None), (doc.id, name)
    assert all(name in doc.spans for doc in DOCS for name in set(FIELDS) - OPTIONAL)


def test_gold_round_trips_through_json():
    for doc in DOCS[:20]:
        assert Booking.model_validate_json(doc.gold.model_dump_json()) == doc.gold


def test_schema_rejects_unknown_keys_and_bad_values():
    good = DOCS[0].gold.model_dump(mode="json")
    with pytest.raises(ValidationError, match="Extra inputs"):
        Booking.model_validate({**good, "notes": "x"})
    with pytest.raises(ValidationError):
        Booking.model_validate({**good, "currency": "dollars"})
    with pytest.raises(ValidationError):
        Booking.model_validate({**good, "guest_count": 0})


def test_seen_and_unseen_layouts_share_no_layout():
    def first_words(docs):
        return {doc.text.split()[0] for doc in docs}

    seen = first_words(BookingRecipe(n=60, seed=1, layouts="seen").sample())
    unseen = first_words(BookingRecipe(n=60, seed=1, layouts="unseen").sample())
    assert len(seen) == 3 and len(unseen) == 2 and not seen & unseen


def test_unknown_layout_group_is_rejected():
    with pytest.raises(ValueError, match="layouts"):
        BookingRecipe(layouts="some")
