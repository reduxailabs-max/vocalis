"""Kokoro text-to-speech engine: small, natural-sounding, fast on a normal
processor and faster on a graphics card. Wraps the `kokoro-onnx` library.

`kokoro_onnx` and `onnxruntime` are imported only inside the two small
default builder functions below (via importlib), so importing this module
is cheap and tests can replace both with fakes.
"""
import importlib
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import numpy.typing as npt

logger = logging.getLogger("vocalis")


class KokoroLoadError(Exception):
    """Raised when the Kokoro model cannot be loaded."""


class KokoroSynthesisError(Exception):
    """Raised when Kokoro cannot turn text into speech."""


def _language_for(voice: str) -> str:
    """Kokoro voices that start with 'b' speak British English, the rest American."""
    return "en-gb" if voice.startswith("b") else "en-us"


def _default_session_builder(model_path: Path, use_gpu: bool) -> Any:
    """Create the ONNX session, on the graphics card only if asked and available."""
    ort = importlib.import_module("onnxruntime")
    providers = ["CPUExecutionProvider"]
    if use_gpu and "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(str(model_path), providers=providers)


def _default_kokoro_factory(session: Any, voices_path: Path) -> Any:
    """Wrap an ONNX session in the kokoro_onnx.Kokoro synthesizer."""
    module = importlib.import_module("kokoro_onnx")
    return module.Kokoro.from_session(session, str(voices_path))


class KokoroTTS:
    """Text-to-speech engine backed by the Kokoro ONNX model."""

    def __init__(
        self,
        model_path: Path,
        voices_path: Path,
        voice: "str | Callable[[], str]" = "af_heart",
        use_gpu: "bool | Callable[[], bool]" = False,
        session_builder: Callable[[Path, bool], Any] = _default_session_builder,
        kokoro_factory: Callable[[Any, Path], Any] = _default_kokoro_factory,
    ) -> None:
        """Create a KokoroTTS engine (does not load the model yet).

        Args:
            model_path: Path to kokoro-v1.0.onnx.
            voices_path: Path to voices-v1.0.bin.
            voice: Voice name, or a function returning it (re-read on every
                synthesis so a Settings change applies to the next reading).
            use_gpu: Whether to run on the graphics card, or a function
                answering that at load time.
            session_builder: Builds the ONNX session; injectable for tests.
            kokoro_factory: Wraps a session into a synthesizer; injectable for tests.
        """
        self._model_path = model_path
        self._voices_path = voices_path
        self._voice = voice
        self._use_gpu = use_gpu
        self._session_builder = session_builder
        self._kokoro_factory = kokoro_factory
        self._kokoro: Any | None = None

    @property
    def is_loaded(self) -> bool:
        """Whether the model is currently loaded."""
        return self._kokoro is not None

    def load(self) -> None:
        """Load the model.

        Raises:
            KokoroLoadError: If a model file is missing or loading fails.
        """
        for path in (self._model_path, self._voices_path):
            if not path.exists():
                raise KokoroLoadError(f"Kokoro file not found: {path}")
        use_gpu = self._use_gpu() if callable(self._use_gpu) else self._use_gpu
        try:
            session = self._session_builder(self._model_path, use_gpu)
            self._kokoro = self._kokoro_factory(session, self._voices_path)
        except Exception as exc:
            self._kokoro = None
            raise KokoroLoadError(f"Could not load Kokoro: {exc}") from exc
        logger.info("Kokoro loaded (graphics card: %s)", use_gpu)

    def unload(self) -> None:
        """Release the model."""
        self._kokoro = None

    def _current_voice(self) -> str:
        """The voice to use right now."""
        return self._voice() if callable(self._voice) else self._voice

    def synthesize(self, text: str) -> "tuple[npt.NDArray[np.float32], int]":
        """Turn text into speech.

        Raises:
            KokoroSynthesisError: If the engine is not loaded, the voice does
                not exist, or Kokoro fails.
        """
        if self._kokoro is None:
            raise KokoroSynthesisError("Kokoro is not loaded")
        voice = self._current_voice()
        try:
            samples, sample_rate = self._kokoro.create(text, voice=voice, speed=1.0, lang=_language_for(voice))
        except Exception as exc:
            raise KokoroSynthesisError(f"Kokoro could not speak this text: {exc}") from exc
        return np.asarray(samples, dtype=np.float32).reshape(-1), int(sample_rate)
