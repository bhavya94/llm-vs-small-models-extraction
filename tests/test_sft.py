import pytest
from transformers import AutoTokenizer

from extraction_eval_lab.data.generator import BookingRecipe
from extraction_eval_lab.extractors.parsing import parse_reply
from extraction_eval_lab.extractors.sft import BASE, IGNORE, SFTExtractor, SFTTraining, target

DOCS = BookingRecipe(n=4, seed=1).sample()


def test_parse_reply_finds_the_first_json_object_or_none():
    assert parse_reply('Sure! {"guest_count": 12} Hope that helps.') == {"guest_count": 12}
    assert parse_reply('{"deposit_amount": C$3}') is None
    assert parse_reply("no json here") is None and parse_reply("[1, 2]") is None


def test_loss_covers_the_answer_only():
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    ids, labels = SFTTraining()._encode(tokenizer, DOCS[0])
    answer_start = labels.index(next(label for label in labels if label != IGNORE))
    assert all(label == IGNORE for label in labels[:answer_start])
    assert labels[answer_start:] == ids[answer_start:]
    assert tokenizer.decode(ids[answer_start:], skip_special_tokens=True) == target(DOCS[0])


def test_a_short_run_saves_an_adapter_that_loads_and_is_reused(tmp_path):
    recipe = SFTTraining(epochs=1, batch_size=2, directory=tmp_path)
    path = recipe.run(DOCS[:2])
    assert (path / "adapter_config.json").exists()
    SFTExtractor(adapter=path, max_new_tokens=8).reply(DOCS[0].text)  # loads and generates
    modified = (path / "adapter_config.json").stat().st_mtime
    assert recipe.run(DOCS[:2]) == path and (path / "adapter_config.json").stat().st_mtime == modified


def test_no_training_pair_is_cut_and_an_overlong_one_raises():
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    for doc in BookingRecipe(n=60, seed=1).sample():
        ids, labels = SFTTraining()._encode(tokenizer, doc)
        assert tokenizer.decode([i for i, label in zip(ids, labels) if label != IGNORE],
                                skip_special_tokens=True) == target(doc)
    with pytest.raises(ValueError, match="over max_length"):
        SFTTraining(max_length=100)._encode(tokenizer, DOCS[0])
