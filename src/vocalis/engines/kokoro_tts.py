"""Kokoro text-to-speech engine: small, natural-sounding, fast on a normal
processor and faster on a graphics card. Wraps the `kokoro-onnx` library.

`kokoro_onnx` and `onnxruntime` are imported only inside the two small
default builder functions below (via importlib), so importing this module
is cheap and tests can replace both with fakes.
"""
import importlib
import logging
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np
import numpy.typing as npt

logger = logging.getLogger("vocalis")


class KokoroLoadError(Exception):
    """Raised when the Kokoro model cannot be loaded."""


class KokoroSynthesisError(Exception):
    """Raised when Kokoro cannot turn text into speech."""


_MAX_PIECE_CHARS = 250
_GAP_SECONDS = 0.06


def _split_long_part(part: str, max_chars: int) -> list[str]:
    """Split one over-long part at whitespace; hard-cut any single word that is still too long."""
    pieces: list[str] = []
    current = ""
    for word in part.split():
        while len(word) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.append(word[:max_chars])
            word = word[max_chars:]
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= max_chars:
            current = f"{current} {word}"
        else:
            pieces.append(current)
            current = word
    if current:
        pieces.append(current)
    return pieces


def _pieces_for_kokoro(text: str, max_chars: int = _MAX_PIECE_CHARS) -> list[str]:
    """Cut text into pieces of at most max_chars so Kokoro never silently truncates a long unpunctuated stretch.

    Splits at sentence/clause punctuation first, packs neighbours together
    while they fit, and splits any still-too-long part at spaces (or hard-cuts
    a single enormous word). Text that already fits is returned unchanged.
    """
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return [text]
    parts: list[str] = []
    for part in re.split(r"(?<=[.!?;:,])\s+", stripped):
        parts.extend(_split_long_part(part, max_chars) if len(part) > max_chars else [part])
    pieces: list[str] = []
    current = ""
    for part in parts:
        if not part:
            continue
        if not current:
            current = part
        elif len(current) + 1 + len(part) <= max_chars:
            current = f"{current} {part}"
        else:
            pieces.append(current)
            current = part
    if current:
        pieces.append(current)
    return pieces or [text]


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
        try:
            use_gpu = self._use_gpu() if callable(self._use_gpu) else self._use_gpu
            session = self._session_builder(self._model_path, use_gpu)
            self._kokoro = self._kokoro_factory(session, self._voices_path)
        except Exception as exc:
            self._kokoro = None
            raise KokoroLoadError(f"Could not load Kokoro: {exc}") from exc
        providers: list[str] = getattr(session, "get_providers", lambda: [])()
        logger.info("Kokoro loaded (graphics card requested: %s, providers: %s)", use_gpu, providers)

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
        kokoro = self._kokoro
        if kokoro is None:
            raise KokoroSynthesisError("Kokoro is not loaded")
        try:
            voice = self._current_voice()
            lang = _language_for(voice)
            clips: list[npt.NDArray[np.float32]] = []
            sample_rate = 0
            for piece in _pieces_for_kokoro(text):
                samples, rate = kokoro.create(piece, voice=voice, speed=1.0, lang=lang)
                sample_rate = int(rate)
                if clips:
                    clips.append(np.zeros(int(sample_rate * _GAP_SECONDS), dtype=np.float32))
                clips.append(np.asarray(samples, dtype=np.float32).reshape(-1))
        except Exception as exc:
            raise KokoroSynthesisError(f"Kokoro could not speak this text: {exc}") from exc
        return np.concatenate(clips), sample_rate
