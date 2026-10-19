"""Structured extraction experiments on fictional booking confirmations."""

import os

# The tokenizers library's Rust thread pool can deadlock MPS graph compilation during training.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
