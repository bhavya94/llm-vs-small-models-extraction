# extraction-eval-lab

Structured extraction from fictional documents into a strict JSON schema: an LLM, a token tagger and a small fine-tuned model, scored per field.

## Evidence

## What is here

## Run it
```console
uv sync
uv run pytest
uv run eel --help
```

## Charts
`docs/views/index.html` draws the results above from the same seeded runs. Rebuild it with
`uv run python scripts/build_views.py`.

## Layers
A package never imports a package listed after it, and only `cli.py` prints. `tests/test_layers.py` enforces both.

## Limits
