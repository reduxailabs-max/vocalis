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

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from qwen_tts import Qwen3TTSModel

logger = logging.getLogger("vocalis")


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
        """
        self._model_dir = model_dir
        self._speaker = speaker
        self._language = language
        self._use_gpu = use_gpu
        self._model: Qwen3TTSModel | None = None

    @property
    def is_loaded(self) -> bool:
        """Whether the model is currently loaded."""
        return self._model is not None

    def load(self) -> None:
        """Load the model onto the GPU.

        Raises:
            Qwen3TTSLoadError: If loading fails for any reason (missing
                files, CUDA unavailable, out-of-memory, etc.).
        """
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
        """
        self._model = None
        if self._use_gpu:
            torch.cuda.empty_cache()

    def _current_speaker(self) -> str:
        """The voice to use right now (re-read on every synthesis so a change in Settings applies to the next reading)."""
        return self._speaker() if callable(self._speaker) else self._speaker

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
            text=text, language=self._language, speaker=self._current_speaker()
        )
        audio = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
        return audio, int(sample_rate)
