"""Vocalis: offline voice AI workstation (dictation, quick reader, document reader, OCR)."""
import os

# Force Hugging Face Hub / transformers offline mode before ANY submodule
# of this package can be imported. This must live here rather than in a
# leaf module (e.g. vocalis.engines.parakeet) because huggingface_hub
# reads HF_HUB_OFFLINE only once, at its OWN import time, baking the
# value into a module-level constant (huggingface_hub.constants.
# HF_HUB_OFFLINE) — a later os.environ.setdefault() call does not change
# that already-frozen constant, only os.environ itself. Whichever vocalis
# submodule some caller imports first (vocalis.engines.parakeet,
# vocalis.engines.qwen3_tts, etc.), Python always runs this file's body
# first as part of resolving `vocalis.<submodule>`, so this guard fires
# before huggingface_hub — however it eventually gets pulled in
# transitively (onnx_asr, transformers, ...) — has a chance to import and
# freeze its offline flag. That's what makes this fix import-order-
# independent, unlike setting the env vars inside a single leaf module.
#
# Without this, a missing/incomplete local model directory can fall back
# to a live Hugging Face Hub network lookup, which violates this
# project's fully-offline design and can hang for many minutes on
# network retries (see tests/test_offline_guard.py for the regression
# test proving this).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

__version__ = "0.1.0"
