"""The shared shapes of a text-to-speech engine."""
from typing import Protocol

import numpy as np
import numpy.typing as npt

from vocalis.model_manager import ModelEngine


class Synthesizer(Protocol):
    """Anything that can turn text into (audio, sample_rate)."""

    def synthesize(self, text: str) -> "tuple[npt.NDArray[np.float32], int]":
        """Speak `text`; returns mono float32 audio and its sample rate."""
        ...


class TTSEngine(ModelEngine, Synthesizer, Protocol):
    """A text-to-speech engine that ModelManager can load and unload."""
