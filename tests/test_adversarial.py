from extraction_eval_lab.data.adversarial import CATEGORIES, AdversarialRecipe
from extraction_eval_lab.data.generator import _money_forms
from extraction_eval_lab.data.schema import Booking

SETS = AdversarialRecipe(n=30).sample()


def test_every_category_has_n_documents_with_valid_gold():
    assert set(SETS) == set(CATEGORIES)
    assert all(len(docs) == 30 for docs in SETS.values())
    assert all(Booking.model_validate(doc.gold.model_dump()) == doc.gold for docs in SETS.values() for doc in docs)


def test_revised_count_states_both_numbers_and_gold_is_the_new_one():
    for doc in SETS["revised_count"]:
        assert f"to {doc.gold.guest_count} guests" in doc.text


def test_decoy_amount_comes_before_the_real_deposit():
    for doc in SETS["decoy_amount"]:
        real = min(doc.text.find(form) for form in _money_forms(doc.gold.deposit_amount, doc.gold.currency)
                   if form in doc.text)
        assert 0 <= doc.text.find("damage deposit") < real


def test_numeric_dates_are_swappable_stated_and_before_payments_end():
    for doc in SETS["numeric_date"]:
        event = doc.gold.event_date
        assert event.day <= 12 and event.strftime("%d/%m/%Y") in doc.text and "day/month/year" in doc.text
        assert doc.gold.balance_due < event and doc.gold.deposit_due < event


def test_negated_caterer_names_a_caterer_but_gold_has_none():
    for doc in SETS["negated_caterer"]:
        assert "is not booked" in doc.text and doc.gold.caterer_name is None


def test_yearless_due_dates_lose_the_year_only_there():
    for doc in SETS["yearless_due"]:
        surface = doc.gold.balance_due.strftime("%A, %B %-d")
        assert surface in doc.text and f"{surface}, {doc.gold.balance_due.year}" not in doc.text
