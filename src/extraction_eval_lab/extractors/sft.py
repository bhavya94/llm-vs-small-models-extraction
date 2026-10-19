"""A small instruct model taught by supervised fine-tuning (SFT) to write the booking JSON.

SmolLM2-135M-Instruct gets LoRA adapters on its attention layers: small low-rank matrices are trained while the
base weights stay frozen, so memory stays low on a laptop. The loss covers only the answer tokens, the gold JSON;
the prompt tokens are context, not targets. The same `extract` runs on the base model too, for a before-and-after
comparison on the same documents.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from extraction_eval_lab.data.generator import Document
from extraction_eval_lab.data.schema import FIELDS
from extraction_eval_lab.extractors.parsing import parse_reply

BASE = "HuggingFaceTB/SmolLM2-135M-Instruct"
INSTRUCTION = (f"Extract the venue booking from the document as one JSON object with exactly these keys: "
               f"{', '.join(FIELDS)}. Dates are YYYY-MM-DD, amounts are numbers with two decimals, currency is the "
               f"3-letter code, and an absent field is null. Reply with the JSON only.")
IGNORE = -100


def messages(text: str) -> list[dict]:
    return [{"role": "system", "content": INSTRUCTION}, {"role": "user", "content": text}]


def target(doc: Document) -> str:
    return doc.gold.model_dump_json()


@dataclass(frozen=True)
class SFTTraining:
    """A recipe for one LoRA fine-tuning run. The adapter is saved under a name hashed from the recipe and data."""

    base_model: str = BASE
    epochs: int = 2
    batch_size: int = 4
    learning_rate: float = 2e-4
    lora_rank: int = 16
    max_length: int = 512  # prompt plus answer reaches about 450 tokens; a lower limit cut off answers
    seed: int = 0
    directory: Path = Path(".cache") / "models"

    def output(self, documents: list[Document]) -> Path:
        recipe = {k: str(v) for k, v in asdict(self).items() if k != "directory"}
        digest = hashlib.sha256(json.dumps([recipe, [d.text for d in documents]]).encode()).hexdigest()[:8]
        return self.directory / f"sft-{self.base_model.split('/')[-1]}-{digest}"

    def run(self, documents: list[Document]) -> Path:
        """Train on `documents`, or reuse the adapter trained on exactly these documents with this recipe."""
        output = self.output(documents)
        if (output / "adapter_config.json").exists():
            return output
        torch.manual_seed(self.seed)
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        # SmolLM2's weights are stored in bfloat16. The M1's CPU has no fast bfloat16 math, so loading them as
        # stored made one training step take about 230 seconds; float32 is the fast path here.
        model = get_peft_model(AutoModelForCausalLM.from_pretrained(self.base_model, dtype=torch.float32),
                               LoraConfig(r=self.lora_rank, lora_alpha=2 * self.lora_rank, lora_dropout=0.05,
                                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                                          task_type="CAUSAL_LM"))
        examples = [self._encode(tokenizer, doc) for doc in documents]
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=self.learning_rate)
        rng = np.random.default_rng(self.seed)
        model.train()
        for _ in range(self.epochs):
            for batch in np.array_split(rng.permutation(len(examples)), max(1, len(examples) // self.batch_size)):
                width = max(len(examples[i][0]) for i in batch)
                pad = tokenizer.pad_token_id
                ids = torch.tensor([examples[i][0] + [pad] * (width - len(examples[i][0])) for i in batch])
                labels = torch.tensor([examples[i][1] + [IGNORE] * (width - len(examples[i][1])) for i in batch])
                mask = torch.tensor([[1] * len(examples[i][0]) + [0] * (width - len(examples[i][0])) for i in batch])
                loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        model.save_pretrained(output)
        return output

    def _encode(self, tokenizer, doc: Document) -> tuple[list[int], list[int]]:
        """Token ids for prompt + answer, and labels that are IGNORE on the prompt and the ids on the answer.

        Raises if the pair is longer than `max_length`: truncation would cut off the answer the model must learn.
        """
        prompt = tokenizer.apply_chat_template(messages(doc.text), add_generation_prompt=True, return_dict=True)
        prompt = list(prompt["input_ids"])
        answer = tokenizer(target(doc) + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
        if len(prompt) + len(answer) > self.max_length:
            raise ValueError(f"{doc.id}: {len(prompt) + len(answer)} tokens, over max_length {self.max_length}")
        return prompt + answer, [IGNORE] * len(prompt) + answer


@dataclass
class SFTExtractor:
    """The base model, or the base model with a trained adapter at `adapter`, extracting by greedy decoding."""

    adapter: Path | None = None
    base_model: str = BASE
    max_new_tokens: int = 220

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        model = AutoModelForCausalLM.from_pretrained(self.base_model, dtype=torch.float32)  # float32: see SFTTraining.run
        self.model = (PeftModel.from_pretrained(model, self.adapter) if self.adapter else model).eval()

    def reply(self, text: str) -> str:
        prompt = self.tokenizer.apply_chat_template(messages(text), add_generation_prompt=True, return_tensors="pt",
                                                    return_dict=True)
        with torch.no_grad():
            out = self.model.generate(**prompt, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(out[0, prompt["input_ids"].shape[1]:], skip_special_tokens=True)

    def extract(self, text: str) -> dict | None:
        return parse_reply(self.reply(text))
