"""Non-fatal checks run once at startup, each reported as a desktop
notification if something looks wrong.

Every check is independent: if one is missing, the others still run and
Vocalis still starts. A missing piece only means the ONE feature that
needs it will fail per-request later, exactly as it does today -- these
checks exist purely to tell the user that up front instead of leaving
them to discover it the first time they use that feature.

Checks run in a fixed order: the Parakeet model, then Qwen3-TTS, then
the Kokoro voice, then ydotoold, then the capture tools, then the
onnxruntime graphics-card check, then a stale ~/.asoundrc.
"""
import importlib
import logging
import os
import shutil
from pathlib import Path
from typing import Callable

from vocalis.config import VocalisConfig

logger = logging.getLogger("vocalis")

PARAKEET_MODEL_FILES = ("config.json", "encoder-model.int8.onnx", "decoder_joint-model.int8.onnx", "vocab.txt")
QWEN3_TTS_MODEL_FILES = ("config.json", "model.safetensors")
KOKORO_MODEL_FILES = ("kokoro-v1.0.onnx", "voices-v1.0.bin")
_CAPTURE_TOOLS = ("tesseract", "xclip")


def check_parakeet_model(model_dir: Path) -> str | None:
    """Whether the dictation (Parakeet) model files are all present.

    Returns:
        None if every expected file exists; otherwise a message naming
        the model and the folder it was expected in.
    """
    if all((model_dir / name).exists() for name in PARAKEET_MODEL_FILES):
        return None
    return (
        f"The Parakeet dictation model is missing or incomplete at {model_dir}. "
        "Dictation (NUMLOCK) will not work until it is installed."
    )


def check_qwen3_tts_model(model_dir: Path) -> str | None:
    """Whether the Qwen3-TTS voice model files are present.

    Returns:
        None if the expected files exist; otherwise a message naming the
        folder. Reading aloud still works via the Piper fallback voice.
    """
    if all((model_dir / name).exists() for name in QWEN3_TTS_MODEL_FILES):
        return None
    return (
        f"The primary voice model is missing at {model_dir}. "
        "Vocalis will use the backup voice until it is installed."
    )


def check_kokoro_model(model_path: Path, voices_path: Path) -> str | None:
    """Whether the Kokoro voice files are present.

    Returns:
        None if both files exist; otherwise a message naming the folder.
    """
    if model_path.exists() and voices_path.exists():
        return None
    return (
        f"The light Kokoro voice is missing at {model_path.parent}. "
        "Vocalis will use another voice until it is installed."
    )


def check_ydotoold(socket_path: Path) -> str | None:
    """Whether ydotoold's socket exists, so dictation can type at the cursor.

    Returns:
        None if the socket exists; otherwise a message explaining that
        typed text will fall back to the clipboard.
    """
    if socket_path.exists():
        return None
    return (
        f"ydotoold is not running ({socket_path} not found). "
        "Dictation will copy text to the clipboard instead of typing it."
    )


def check_capture_tools(which: Callable[[str], str | None] = shutil.which) -> str | None:
    """Whether the OCR Reader's text tools (tesseract, xclip) are installed.

    Returns:
        None if both are found; otherwise a message naming only the
        missing ones, grammatically correct for either one missing tool
        ("...needs xclip, which is not installed.") or more than one
        ("...needs tesseract and xclip, which are not installed.").
    """
    missing = [name for name in _CAPTURE_TOOLS if which(name) is None]
    if not missing:
        return None
    if len(missing) == 1:
        return f"The OCR Reader needs {missing[0]}, which is not installed."
    names = " and ".join(missing)
    return f"The OCR Reader needs {names}, which are not installed."


def _gpu_present() -> bool:
    """Whether a CUDA graphics card is available (asks PyTorch)."""
    from vocalis.hardware import query_gpu_with_torch

    return query_gpu_with_torch() is not None


def _onnx_providers() -> list[str]:
    """The execution providers the installed onnxruntime offers."""
    return list(importlib.import_module("onnxruntime").get_available_providers())


def check_onnx_gpu(has_gpu: Callable[[], bool], providers: Callable[[], list[str]]) -> str | None:
    """Warn if a graphics card exists but onnxruntime cannot use it.

    A CPU-only onnxruntime installed next to the GPU one silently removes
    the graphics card from dictation; this reports that instead of letting
    dictation just feel slow.
    """
    if has_gpu() and "CUDAExecutionProvider" not in providers():
        return (
            "Your graphics card is not being used for dictation, so dictation may be slow. "
            "Re-running the Vocalis installer usually fixes this."
        )
    return None


def check_stale_asoundrc(path: Path) -> str | None:
    """Whether an old Voxtype ~/.asoundrc is still overriding the system's sound.

    Returns:
        None if the file is absent, unreadable, or not from Voxtype; otherwise
        a plain message saying what it does and how to retire it.
    """
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if "voxtype" not in content.lower():
        return None
    return (
        f"An old sound settings file from Voxtype is still at {path}. "
        "It can make the microphone silent in other apps. Vocalis works around it, "
        f"but you can rename it to {path.name}.old to fix it everywhere."
    )


def run_startup_checks(config: VocalisConfig) -> list[str]:
    """Run every startup check and collect the real problems, in a fixed order.

    A check that itself raises is logged as a warning and treated as
    passing, so one broken check can never stop the others or crash
    startup.

    Args:
        config: Loaded, validated vocalis configuration.

    Returns:
        Zero or more human-readable messages, Parakeet first, then
        Qwen3-TTS, then the Kokoro voice, then ydotoold, then the capture
        tools, then the onnxruntime graphics-card check, then a stale
        ~/.asoundrc.
    """
    runtime_dir = Path(os.environ.get("XDG_RUNTIME_DIR", "/run/user/0"))
    checks: "list[Callable[[], str | None]]" = [
        lambda: check_parakeet_model(Path(config.engines.parakeet_model_dir)),
        lambda: check_qwen3_tts_model(Path(config.engines.qwen3_tts_model_dir)),
        lambda: check_kokoro_model(Path(config.engines.kokoro_model_path), Path(config.engines.kokoro_voices_path)),
        lambda: check_ydotoold(runtime_dir / ".ydotool_socket"),
        # Pass shutil.which explicitly (looked up fresh here, at call time)
        # rather than relying on check_capture_tools' own default parameter,
        # which binds to shutil.which once at module-import time and would
        # not observe a test's `monkeypatch.setattr("...shutil.which", ...)`.
        lambda: check_capture_tools(shutil.which),
        lambda: check_onnx_gpu(_gpu_present, _onnx_providers),
        lambda: check_stale_asoundrc(Path(os.environ.get("HOME", str(Path.home()))) / ".asoundrc"),
    ]
    messages: "list[str]" = []
    for check in checks:
        try:
            result = check()
        except Exception:
            logger.warning("A startup check failed to run", exc_info=True)
            continue
        if result is not None:
            messages.append(result)
    return messages
