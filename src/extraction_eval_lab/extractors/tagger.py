"""A token tagger: a small encoder labels each token B-field, I-field or O, and spans become field values.

This is the classic named-entity approach to extraction. Training needs only the generator's character spans,
turned into one BIO label per token. At prediction time the first tagged span of each field is parsed into the
schema's type with the same parsing the rules use.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from extraction_eval_lab.data.generator import Document
from extraction_eval_lab.data.schema import FIELDS
from extraction_eval_lab.extractors.parsing import parse_field

LABELS = ["O"] + [f"{prefix}-{name}" for name in FIELDS for prefix in ("B", "I")]
LABEL_ID = {label: i for i, label in enumerate(LABELS)}
IGNORE = -100  # special tokens get no label; the loss skips them


def default_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def labels_from_spans(offsets: list[tuple[int, int]], spans: dict[str, tuple[int, int]]) -> list[int]:
    """One label id per token: B for the first token that overlaps a field's span, I for the next ones, else O."""
    labels, previous = [], None
    for start, end in offsets:
        if start == end:  # special tokens such as [CLS] have an empty offset
            labels.append(IGNORE)
            continue
        name = next((n for n, (span_start, span_end) in spans.items() if start < span_end and span_start < end), None)
        labels.append(LABEL_ID[f"{'I' if name == previous else 'B'}-{name}"] if name else LABEL_ID["O"])
        previous = name
    return labels


def spans_from_labels(offsets: list[tuple[int, int]], labels: list[int]) -> dict[str, tuple[int, int]]:
    """The first span tagged for each field: from a B (or a stray I) through the I tokens that follow it."""
    spans: dict[str, tuple[int, int]] = {}
    current, start, end = None, 0, 0
    for (token_start, token_end), label_id in zip(offsets, labels):
        label = LABELS[label_id] if label_id != IGNORE else "O"
        prefix, _, name = label.partition("-")
        if current and prefix == "I" and name == current:
            end = token_end
            continue
        if current and current not in spans:
            spans[current] = (start, end)
        current = name if prefix in ("B", "I") else None
        start, end = token_start, token_end
    if current and current not in spans:
        spans[current] = (start, end)
    return spans


@dataclass(frozen=True)
class TaggerTraining:
    """A recipe for one training run. The trained model is saved under a name hashed from the recipe and data."""

    base_model: str = "distilbert-base-cased"
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 5e-5
    # Every batch is padded to this one length. On MPS each new tensor shape compiles and caches a new graph, and
    # per-batch padding grew one run past 5 GB. The longest generated document is 154 tokens.
    max_length: int = 192
    seed: int = 0
    # On an 8 GB M1 with other apps open, MPS training thrashed (resident memory fell to a few MB). The CPU, with
    # 4 cores, measured 1.35 s per step and finishes 3 epochs of 600 documents in about 2.5 minutes.
    device: str = "cpu"
    directory: Path = Path(".cache") / "models"

    def output(self, documents: list[Document]) -> Path:
        recipe = {k: str(v) for k, v in asdict(self).items() if k not in ("directory", "device")}
        digest = hashlib.sha256(json.dumps([recipe, [d.text for d in documents]]).encode()).hexdigest()[:8]
        return self.directory / f"tagger-{self.base_model.split('/')[-1]}-{digest}"

    def run(self, documents: list[Document]) -> Path:
        """Train on `documents`, or reuse the saved model trained on exactly these documents with this recipe."""
        output = self.output(documents)
        if (output / "config.json").exists():
            return output
        torch.manual_seed(self.seed)
        device = self.device
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        model = AutoModelForTokenClassification.from_pretrained(
            self.base_model, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=LABEL_ID).to(device)
        encoded = [self._encode(tokenizer, doc) for doc in documents]
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        rng = np.random.default_rng(self.seed)
        model.train()
        for _ in range(self.epochs):
            for batch in np.array_split(rng.permutation(len(encoded)), max(1, len(encoded) // self.batch_size)):
                inputs = tokenizer.pad([{k: encoded[i][k] for k in ("input_ids", "attention_mask")} for i in batch],
                                       padding="max_length", max_length=self.max_length, return_tensors="pt")
                width = inputs["input_ids"].shape[1]
                labels = torch.tensor([encoded[i]["labels"] + [IGNORE] * (width - len(encoded[i]["labels"]))
                                       for i in batch])
                loss = model(**inputs.to(device), labels=labels.to(device)).loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        model.save_pretrained(output)
        tokenizer.save_pretrained(output)
        return output

    def _encode(self, tokenizer, doc: Document) -> dict:
        tokens = tokenizer(doc.text, truncation=True, max_length=self.max_length, return_offsets_mapping=True)
        return {"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"],
                "labels": labels_from_spans(tokens["offset_mapping"], doc.spans)}


@dataclass
class Tagger:
    """A trained tagger, loaded once, that extracts one document at a time."""

    path: Path
    max_length: int = 192
    device: str = default_device()

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        self.model = AutoModelForTokenClassification.from_pretrained(self.path).to(self.device).eval()

    def extract(self, text: str) -> dict:
        tokens = self.tokenizer(text, truncation=True, max_length=self.max_length, return_offsets_mapping=True,
                                return_tensors="pt")
        offsets = [tuple(pair) for pair in tokens.pop("offset_mapping")[0].tolist()]
        with torch.no_grad():
            logits = self.model(**tokens.to(self.device)).logits[0]
        labels = [IGNORE if start == end else int(i) for (start, end), i in zip(offsets, logits.argmax(-1).tolist())]
        spans = spans_from_labels(offsets, labels)
        return {name: parse_field(name, text[slice(*spans[name])]) if name in spans else None for name in FIELDS}
