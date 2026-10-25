"""Qwen3-TTS text-to-speech engine — Quick Reader's primary voice.

Wraps the qwen-tts package's CustomVoice model (a fixed set of 9 preset
voices, selected by name — no reference audio or voice-cloning step
needed, unlike Qwen3-TTS's separate "Base" variant). Implements the
ModelEngine protocol so ModelManager's single-GPU-residency policy
applies to it exactly like Parakeet — this is the only Quick Reader
engine that touches the GPU; PiperTTS (the fallback) does not and is
never routed through ModelManager.

torch's newer "native ops" dispatch path JIT-compiles some CUDA kernels
(used internally by qwen-tts's rotary embedding computation) via gcc at
runtime, which requires the Python development headers (Python.h) for
the exact CPython version the venv uses. Dev machines that only have a
different Python version's headers installed (or none at all) hit a
gcc compile failure the first time such a kernel runs. Disabling native
JIT falls back to torch's standard eager kernels — a pure performance
trade-off, not a functional one — so we set this unconditionally before
torch is imported, rather than requiring every caller to know about it.
"""
import os

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

import gc
import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from qwen_tts import Qwen3TTSModel

from vocalis.audio.pace import clamp_speed, paced_audio

logger = logging.getLogger("vocalis")

_TEMPERATURE = 0.65
_TOP_K = 20
_TOKENS_PER_CHAR = 6
_MIN_NEW_TOKENS = 200
_MAX_NEW_TOKENS = 2048
_STRETCH_MIN_SPEED = 1.0
_STRETCH_MAX_SPEED = 1.2
_BYTES_PER_MB = 1024 * 1024


_default_num_threads: "int | None" = None


def _torch_default_num_threads() -> int:
    """torch's own default CPU thread count, captured once on first use.

    torch.set_num_threads() is a process-global setting with no per-call
    "unset" -- the only way back to torch's own default, once a cap has
    been applied, is to set it explicitly. Caching the value returned by
    torch.get_num_threads() the first time it is needed (before any
    Qwen3TTS instance has had a chance to apply a cap of its own) lets
    load() restore it precisely whenever the live cap later resolves back
    to None, instead of leaving whatever cap was last applied in place.
    """
    global _default_num_threads
    if _default_num_threads is None:
        _default_num_threads = torch.get_num_threads()
    return _default_num_threads


def _max_new_tokens_for(text: str) -> int:
    """How many codec tokens to let Qwen3-TTS generate for `text`.

    Scaled to the text's length instead of always requesting the
    library's full default budget (2048): a short chunk then generates
    faster and is less likely to wander or repeat near the end. A floor
    keeps very short text from being cut off; a ceiling never asks for
    more than the library's own tuned default.
    """
    return max(_MIN_NEW_TOKENS, min(_MAX_NEW_TOKENS, len(text) * _TOKENS_PER_CHAR))


class Qwen3TTSLoadError(Exception):
    """Raised when the Qwen3-TTS model fails to load."""


class Qwen3TTS:
    """Text-to-speech engine backed by the local Qwen3-TTS CustomVoice model."""

    def __init__(
        self,
        model_dir: Path,
        speaker: str | Callable[[], str] = "Ryan",
        language: str = "English",
        use_gpu: bool = True,
        speed: float | Callable[[], float] = 1.0,
        max_cpu_threads: "int | None | Callable[[], int | None]" = None,
        gpu_vram_ceiling_mb: "int | None | Callable[[], int | None]" = None,
    ) -> None:
        """Create a Qwen3TTS engine (does not load the model yet).

        Args:
            model_dir: Local directory the CustomVoice model files were
                downloaded into (see this task's one-time download step).
            speaker: Preset voice name (one of the 9 CustomVoice presets),
                or a function returning it, re-read on every synthesis so a
                change in Settings applies to the next reading.
            language: Spoken language passed to generate_custom_voice().
            use_gpu: If True, load onto CUDA (the only supported device
                for Qwen3-TTS); False is only useful in tests that never
                call load().
            speed: Desired speaking pace, or a function returning it,
                re-read on every synthesis. Qwen3-TTS has no native speed
                control, so this is approximated with a mild (1.0-1.2x)
                time-stretch of the finished audio -- see
                _capped_speed(): a bigger stretch, in either direction,
                audibly hurts this voice, so anything outside that range
                is simply played at the nearest end of it instead.
            max_cpu_threads: Cap for torch.set_num_threads(), or a function
                returning it, re-read at load time. None means no cap.
            gpu_vram_ceiling_mb: Cap for torch.cuda.set_per_process_memory_fraction(),
                or a function returning it, re-read at load time. None means
                no cap; ignored when use_gpu is False. Best-effort: a failed
                GPU query never blocks loading.
        """
        self._model_dir = model_dir
        self._speaker = speaker
        self._language = language
        self._use_gpu = use_gpu
        self._speed = speed
        self._max_cpu_threads = max_cpu_threads
        self._gpu_vram_ceiling_mb = gpu_vram_ceiling_mb
        self._model: Qwen3TTSModel | None = None

    @property
    def is_loaded(self) -> bool:
        """Whether the model is currently loaded."""
        return self._model is not None

    def _current_max_cpu_threads(self) -> "int | None":
        """The CPU thread cap to apply right now (re-read at every load)."""
        return self._max_cpu_threads() if callable(self._max_cpu_threads) else self._max_cpu_threads

    def _current_gpu_vram_ceiling_mb(self) -> "int | None":
        """The GPU VRAM ceiling to apply right now (re-read at every load)."""
        return self._gpu_vram_ceiling_mb() if callable(self._gpu_vram_ceiling_mb) else self._gpu_vram_ceiling_mb

    def _apply_vram_ceiling(self) -> None:
        """Cap this process's GPU memory fraction. Best-effort: never raises."""
        ceiling_mb = self._current_gpu_vram_ceiling_mb()
        if ceiling_mb is None:
            return
        try:
            _, total_bytes = torch.cuda.mem_get_info(0)
            fraction = min(1.0, (ceiling_mb * _BYTES_PER_MB) / total_bytes)
            torch.cuda.set_per_process_memory_fraction(fraction)
        except Exception:
            logger.warning("Could not apply the GPU VRAM ceiling", exc_info=True)

    def load(self) -> None:
        """Load the model onto the GPU.

        Raises:
            Qwen3TTSLoadError: If loading fails for any reason (missing
                files, CUDA unavailable, out-of-memory, etc.).
        """
        max_threads = self._current_max_cpu_threads()
        torch.set_num_threads(max_threads if max_threads is not None else _torch_default_num_threads())
        if self._use_gpu:
            self._apply_vram_ceiling()
        try:
            self._model = Qwen3TTSModel.from_pretrained(
                str(self._model_dir),
                device_map="cuda:0" if self._use_gpu else "cpu",
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
        except Exception as exc:
            raise Qwen3TTSLoadError(
                f"Failed to load Qwen3-TTS model from {self._model_dir}: {exc}"
            ) from exc

    def unload(self) -> None:
        """Release the model reference and, if it was loaded onto the GPU,
        return its VRAM to the OS. Safe to call even if never loaded.

        torch's caching allocator does not release CUDA memory back to the
        OS just because Python's reference count for the model hit zero —
        only an explicit torch.cuda.empty_cache() call does, and without it
        ModelManager's whole idle-unload policy would not actually free VRAM
        for this engine while the process keeps running.

        gc.collect() runs here, before empty_cache(), rather than after: a
        Python reference cycle holding the model (or one of its tensors)
        alive would keep its CUDA memory allocated even after self._model is
        set to None, so empty_cache() alone would have nothing real to hand
        back to the OS/driver. Breaking the cycle first lets the tensor's
        refcount actually reach zero, so the empty_cache() call that follows
        empties a cache that is genuinely free -- making the combination of
        this gc.collect() and ModelManager's own (a separate, additional
        safety net run after unload() returns) effective.
        """
        self._model = None
        if self._use_gpu:
            gc.collect()
            torch.cuda.empty_cache()

    def _current_speaker(self) -> str:
        """The voice to use right now (re-read on every synthesis so a change in Settings applies to the next reading)."""
        return self._speaker() if callable(self._speaker) else self._speaker

    def _current_speed(self) -> float:
        """The requested speaking pace right now (re-read on every synthesis)."""
        return self._speed() if callable(self._speed) else self._speed

    def _capped_speed(self) -> float:
        """The speed actually applied: clamped to the supported range, then
        narrowed to 1.0-1.2x (see __init__'s `speed` docstring)."""
        speed = clamp_speed(self._current_speed())
        return min(max(speed, _STRETCH_MIN_SPEED), _STRETCH_MAX_SPEED)

    def synthesize(self, text: str) -> "tuple[npt.NDArray[np.float32], int]":
        """Generate speech audio for the given text using the preset voice.

        Args:
            text: The text to speak.

        Returns:
            (audio, sample_rate) — a 1-D float32 numpy array and its rate.

        Raises:
            RuntimeError: If called before load().
        """
        if self._model is None:
            raise RuntimeError("Qwen3TTS.synthesize() called before load()")
        wavs, sample_rate = self._model.generate_custom_voice(
            text=text,
            language=self._language,
            speaker=self._current_speaker(),
            temperature=_TEMPERATURE,
            top_k=_TOP_K,
            max_new_tokens=_max_new_tokens_for(text),
        )
        audio = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
        sample_rate = int(sample_rate)
        return paced_audio(audio, sample_rate, self._capped_speed), sample_rate
