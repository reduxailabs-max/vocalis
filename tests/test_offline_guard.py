"""Proves HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE are set before any code that
could trigger a network fallback, regardless of which vocalis submodule
gets imported first — see the Phase 3 Task 2 review finding this guards
against (an import-order-dependent version of this fix silently regressed
to a 16-minute network hang).

Checking os.environ alone is NOT a sufficient regression guard here:
huggingface_hub reads HF_HUB_OFFLINE once, at its OWN import time, into a
module-level constant (huggingface_hub.constants.HF_HUB_OFFLINE). A later
os.environ.setdefault() call still leaves os.environ looking correct even
when huggingface_hub's cached constant was already frozen to False — so
these tests assert on that internal constant, which is what actually
governs whether onnx_asr/huggingface_hub attempt a live network call.

Must run in a FRESH subprocess per case: pytest has already imported
everything by the time any test body runs, so an in-process assertion
proves nothing about import order.
"""
import subprocess
import sys

_MARKER = "VOCALIS_OFFLINE_GUARD_RESULT"


def _offline_state_after_importing(module_import_order: list[str]) -> tuple[str, str, str]:
    imports = "; ".join(f"import {m}" for m in module_import_order)
    code = (
        f"{imports}; "
        "import os, huggingface_hub.constants as _hf_const; "
        f"print({_MARKER!r}, "
        "os.environ.get('HF_HUB_OFFLINE', ''), "
        "os.environ.get('TRANSFORMERS_OFFLINE', ''), "
        "_hf_const.HF_HUB_OFFLINE)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30, check=True
    )
    # qwen_tts prints a banner to stdout on import, so pick out our marker
    # line rather than assuming the whole capture is just our print().
    for line in result.stdout.splitlines():
        if line.startswith(_MARKER):
            _, hub_env, transformers_env, hub_const = line.split()
            return hub_env, transformers_env, hub_const
    raise AssertionError(f"marker line not found in subprocess stdout: {result.stdout!r}")


def test_offline_guard_active_when_qwen3_tts_imported_first() -> None:
    hub_env, transformers_env, hub_const = _offline_state_after_importing(
        ["vocalis.engines.qwen3_tts", "vocalis.engines.parakeet"]
    )
    assert hub_env == "1"
    assert transformers_env == "1"
    assert hub_const == "True"


def test_offline_guard_active_when_parakeet_imported_first() -> None:
    hub_env, transformers_env, hub_const = _offline_state_after_importing(
        ["vocalis.engines.parakeet", "vocalis.engines.qwen3_tts"]
    )
    assert hub_env == "1"
    assert transformers_env == "1"
    assert hub_const == "True"
