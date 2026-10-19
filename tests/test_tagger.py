import pytest
from transformers import AutoTokenizer

from extraction_eval_lab.data.generator import BookingRecipe
from extraction_eval_lab.extractors.tagger import LABELS, Tagger, TaggerTraining, labels_from_spans, spans_from_labels

DOCS = BookingRecipe(n=60, seed=1, layouts="all").sample()


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("distilbert-base-cased")


def test_spans_survive_the_round_trip_through_token_labels(tokenizer):
    for doc in DOCS:
        offsets = tokenizer(doc.text, return_offsets_mapping=True)["offset_mapping"]
        assert spans_from_labels(offsets, labels_from_spans(offsets, doc.spans)) == doc.spans, doc.id


def test_each_span_starts_with_b_and_continues_with_i(tokenizer):
    doc = DOCS[0]
    offsets = tokenizer(doc.text, return_offsets_mapping=True)["offset_mapping"]
    names = [LABELS[i] for i in labels_from_spans(offsets, doc.spans) if i != -100]
    for field in doc.spans:
        tagged = [label for label in names if label.endswith(f"-{field}")]
        assert tagged[0] == f"B-{field}" and all(label == f"I-{field}" for label in tagged[1:])


def test_only_the_first_span_of_a_field_is_kept():
    offsets = [(0, 0), (0, 4), (5, 9), (10, 14), (0, 0)]
    labels = [-100, LABELS.index("B-venue_city"), LABELS.index("O"), LABELS.index("B-venue_city"), -100]
    assert spans_from_labels(offsets, labels) == {"venue_city": (0, 4)}


def test_a_short_training_run_saves_a_model_that_extracts_and_is_reused(tmp_path):
    recipe = TaggerTraining(epochs=1, batch_size=4, directory=tmp_path)
    path = recipe.run(DOCS[:8])
    prediction = Tagger(path, device="cpu").extract(DOCS[0].text)
    assert set(prediction) == set(DOCS[0].gold.model_dump())
    modified = (path / "config.json").stat().st_mtime
    assert recipe.run(DOCS[:8]) == path and (path / "config.json").stat().st_mtime == modified
