# llm-vs-small-models-extraction

Can a small model replace an LLM for structured extraction? Four extractors read the same fictional booking
documents and fill the same strict 13-field JSON schema. The scorer checks every field on three sets: layouts the
models trained on, layouts they never saw, and documents written to trick them.

**The task.** A seeded generator writes each document and its correct JSON (the gold), so every score is exact.
Three layouts (email, invoice, note) are "seen": the trained models learn from them. Two layouts (chat, letter)
are "unseen": they reword every label and put the balance before the deposit. A model that learned where values
sit, not what they mean, fails on them. The generator invents every name, venue and email.

```console
$ uv run eel sample --n 1
--- doc-0-0
Hi Rosa Lindqvist - quick note to confirm The Copper Lantern (Port Ellery) is yours on 6 Jul 2027, 16 people. Deposit CAD 1,700.00 by 22 May 2027, then CAD 5,100.00 by Saturday, June 19, 2027 (CAD). Replies go to rosa.lindqvist@example.com.
gold: {"booker_name":"Rosa Lindqvist","booker_email":"rosa.lindqvist@example.com","venue_name":"The Copper Lantern","venue_city":"Port Ellery","event_date":"2027-07-06","guest_count":16,"currency":"CAD","deposit_amount":"1700.00","deposit_due":"2027-05-22","balance_amount":"5100.00","balance_due":"2027-06-19","caterer_name":null,"cancellation_days":null}
```

**The extractors.**
| Extractor | What it is | Trained on |
|---|---|---|
| Rules | Regular expressions written for the seen layouts' labels | nothing |
| Tagger | `distilbert-base-cased` labels each token as the start, inside or outside of a field (BIO) | 600 seen-layout documents |
| SFT | `SmolLM2-135M-Instruct` with LoRA adapters, taught to write the JSON | 600 seen-layout documents |
| LLM | Claude Haiku 4.5, schema in the prompt, validation and one repair round | nothing (prompt only) |

**The scores.**
- **Field F1:** per field, over 100 test documents. A stated wrong value counts against precision and recall.
- **Exact match:** the share of documents with all 13 fields right.
- **Valid JSON:** the share of outputs that pass the strict schema: right keys, types and formats.

## Evidence

**Only the LLM survives unseen layouts.** Rules, the tagger and the SFT model are perfect or nearly perfect on
seen layouts. On unseen layouts, all three score 0% exact match; the LLM scores 100%. The trained models keep
part of the names and cities, because those words look alike in any layout. They lose the dates and amounts,
because they learned each value's position and label text, and the unseen layouts change both.

```console
$ uv run eel rules
rule-based extractor, 100 documents per layout group; per-field F1
  field                seen  unseen
  booker_name         1.000   0.000
  booker_email        1.000   1.000
  venue_name          1.000   0.000
  venue_city          1.000   0.000
  event_date          1.000   0.000
  guest_count         1.000   0.000
  currency            1.000   1.000
  deposit_amount      1.000   0.000
  deposit_due         1.000   0.600
  balance_amount      1.000   0.000
  balance_due         1.000   0.000
  caterer_name        1.000   0.000
  cancellation_days   1.000   0.000
  exact match         1.000   0.000
  valid JSON          1.000   0.000
  seconds per document: seen 0.0001, unseen 0.0001
```

```console
$ uv run eel tagger
distilbert-base-cased token tagger, trained on 600 seen-layout documents for 3 epochs; per-field F1 on 100 test documents per layout group
  field                seen  unseen
  booker_name         1.000   0.938
  booker_email        1.000   0.560
  venue_name          1.000   0.854
  venue_city          1.000   0.940
  event_date          1.000   0.000
  guest_count         1.000   0.739
  currency            1.000   1.000
  deposit_amount      1.000   0.000
  deposit_due         1.000   0.145
  balance_amount      1.000   0.000
  balance_due         1.000   0.000
  caterer_name        1.000   1.000
  cancellation_days   1.000   0.718
  exact match         1.000   0.000
  valid JSON          1.000   0.000
  seconds per document: seen 0.0188, unseen 0.0153
```

**Fine-tuning buys the format, not the task.** The base 135M model never writes valid JSON for this schema. After
LoRA SFT on 600 documents it writes valid JSON for 98% of seen documents and gets 94% fully right. On unseen
layouts it still writes valid JSON 75% of the time, but every document has at least one wrong field.

```console
$ uv run eel sft
SmolLM2-135M-Instruct, base vs LoRA SFT on 600 seen-layout documents for 2 epochs; per-field F1 on 100 test documents per layout group
  field               base seen  base unseen  SFT seen  SFT unseen
  booker_name             0.020        0.000     1.000       0.810
  booker_email            0.000        0.000     1.000       0.718
  venue_name              0.346        0.000     1.000       0.882
  venue_city              0.387        0.020     1.000       0.974
  event_date              0.189        0.000     0.990       0.154
  guest_count             0.413        0.020     1.000       0.728
  currency                0.409        0.020     1.000       0.974
  deposit_amount          0.157        0.000     0.990       0.431
  deposit_due             0.000        0.000     0.960       0.523
  balance_amount          0.110        0.000     1.000       0.503
  balance_due             0.000        0.000     0.980       0.041
  caterer_name            0.000        0.000     1.000       0.824
  cancellation_days       0.136        0.000     1.000       0.957
  exact match             0.000        0.000     0.940       0.000
  valid JSON              0.000        0.000     0.980       0.750
  seconds per document: base seen 5.4532, base unseen 5.4266, tuned seen 4.5173, tuned unseen 4.4310
```

```console
$ uv run eel llm
claude-haiku-4-5-20251001 on 200 documents: estimated at most $0.65 before caching; cap $2.00
per-field F1 on 100 test documents per layout group; 0 API calls, 0 documents repaired, $0.00 spent
  field                seen  unseen
  booker_name         1.000   1.000
  booker_email        1.000   1.000
  venue_name          1.000   1.000
  venue_city          1.000   1.000
  event_date          1.000   1.000
  guest_count         1.000   1.000
  currency            1.000   1.000
  deposit_amount      1.000   1.000
  deposit_due         1.000   1.000
  balance_amount      1.000   1.000
  balance_due         1.000   1.000
  caterer_name        1.000   1.000
  cancellation_days   1.000   1.000
  exact match         1.000   1.000
  valid JSON          1.000   1.000
  seconds per document: seen 0.0002, unseen 0.0002
  uncached cost per document $0.00161; median call latency not recorded (all replies cached)
```
The cost per document comes from the token counts stored with each cached reply, so a rerun reports it for free.

**Adversarial documents.** Each category edits one field of a seen-layout document; the table shows that field's
accuracy, then the share of documents with every field right. The trained models fail where the right value
depends on context: a revised headcount, a decoy deposit, a caterer named but not booked. The LLM gets every
document right, so this set no longer separates it from anything: it needs harder cases.

```console
$ uv run eel adversarial --llm
claude-haiku-4-5-20251001 on 200 documents: estimated at most $0.65 before caching; cap $2.00
40 adversarial documents per category, built from seen layouts; targeted field right / whole document right
  category                  rules        tagger           SFT           LLM
  revised_count       0.00 / 0.00   0.00 / 0.00   0.15 / 0.10   1.00 / 1.00
  decoy_amount        0.00 / 0.00   0.00 / 0.00   0.03 / 0.03   1.00 / 1.00
  numeric_date        0.00 / 0.00   0.00 / 0.00   0.57 / 0.53   1.00 / 1.00
  negated_caterer     0.42 / 0.42   0.10 / 0.10   0.12 / 0.12   1.00 / 1.00
  yearless_due        0.00 / 0.00   0.00 / 0.00   0.72 / 0.70   1.00 / 1.00
LLM spend: $0.00 (cached replies are free)
LLM median call latency: 1.70 s over 40 live calls
```

**Cost, speed and accuracy.**
| Extractor | API $ per 1k docs | Seconds per doc | Exact match, unseen | Adversarial, all fields right |
|---|---|---|---|---|
| Rules | 0.00 | 0.0001 | 0% | 8% |
| Tagger | 0.00 | 0.02 | 0% | 2% |
| SFT | 0.00 | 4.43 | 0% | 30% |
| LLM | 1.61 | 1.70 | 100% | 100% |

The tagger runs on the M1's GPU (MPS), the SFT model on its CPU in float32. The LLM's seconds are the median API
round trip over the adversarial run's live calls. Local models cost no API money, but they need labeled data
and training time: about 20 minutes for the SFT model on this laptop.

**A judge can stand in for the gold, here.** Real documents have no gold JSON. A rubric judge sees only the
document and the prediction and says true or false per field. The exact scorer knows the truth, so it scores the
judge. TPR is the share of right fields the judge accepts; TNR is the share of wrong fields it catches; Cohen's
kappa is agreement beyond chance.

```console
$ uv run eel judge
claude-haiku-4-5-20251001 as a rubric judge, one true/false per field, against the exact scorer
  predictions               fields  right    TPR    TNR  kappa  unparsed
  tagger, unseen layouts      1300   0.52  0.999  0.986  0.985         0
  rules, adversarial          2600   0.88  0.994  0.950  0.948         0
judge spend: $0.00 over 0 calls (cached replies are free)
```

## What is here
- **Data** (`data/`): the booking schema, the seeded generator with field spans, and the adversarial set.
- **LLM client** (`llm/`): one cached, budget-capped way to call a model, shared by the extractor and the judge.
- **Extractors** (`extractors/`): rules, the BIO tagger, the SFT model, the LLM extractor and shared parsers.
- **Evaluation** (`evaluation/`): the per-field scorer, with a pure `score(predicted, gold)` usable as a training
  reward, and the rubric judge.

Cohen's kappa comes from [metrics-from-scratch](https://github.com/bhavya94/metrics-from-scratch).

## Run it
```console
uv sync
uv run pytest
uv run eel --help
```
`eel tagger` and `eel sft` train once and cache the model in `.cache/models/`. The SFT model takes about 20
minutes to train and about 30 minutes to score on an M1 CPU. Paid commands need `ANTHROPIC_API_KEY`. They print
an estimate first, refuse any call that could pass `--cap`, and cache every reply in `.cache/llm/`, so a rerun
is free. The runs above cost $0.88 in total.

## Charts
`docs/views/index.html` explains the task and draws each result above. Every command saves its numbers with
`uv run eel --results-dir docs/results <command>`; `uv run python scripts/build_views.py` rebuilds the page from
those files.

## Layers
A package never imports a package listed after it, and only `cli.py` prints. `tests/test_layers.py` enforces both.

1. `data`: schema, generator and adversarial set.
2. `llm`: the cached, capped client.
3. `extractors`: rules, tagger, SFT and LLM.
4. `evaluation`: scorer and judge.
5. `cli.py`: the `eel` command.

## Limits
- **Synthetic, clean documents.** Real documents add OCR noise, tables and many more layouts.
- **One seed per trained model.** Small differences between the tagger and the SFT model are not significant.
- **The adversarial set is too easy for the LLM.** It scores 100%, so the set cannot rank stronger models.
- **The judge is stricter than the scorer.** The rubric asks for values exactly as stated; the scorer ignores case
  and accepts `1700` for `1700.00`. Some judge "errors" may come from this gap.
- **Same model family.** The judge and the LLM extractor are both Haiku 4.5. A different judge could disagree
  differently.
- **Default temperature.** API calls use the default temperature, and the cache hides run-to-run variation.
- **Rough edits.** The negated-caterer edit can read "Your caterer, the client", and the day/month header sits
  above dates written in other forms.
- **Model caches key on training texts.** A generator change that keeps the texts but moves spans would reuse a
  stale model; clear `.cache/models/` after changing the generator.
