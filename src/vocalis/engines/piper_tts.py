"""Piper text-to-speech engine — Quick Reader's fallback voice.

Implements the ModelEngine protocol for interface consistency with every
other engine, but is never actually routed through ModelManager: Piper
runs entirely on CPU via the existing local `piper` binary (already
installed, proven working — see old/read-selected.sh), so it never
competes for GPU residency the way Qwen3-TTS does. load()/unload() exist
to fail fast if the binary or voice model files go missing, and to match
the shape every other engine wrapper uses; there is no persistent process
to actually hold — each synthesize() call runs a fresh subprocess, the
same pattern TextOutput/Notifier already use for ydotool/xclip/notify-send.
"""
import logging
import subprocess
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

logger = logging.getLogger("vocalis")


class CommandRunner(Protocol):
    """Runs a command, optionally piping bytes to its stdin."""

    def __call__(
        self, argv: list[str], input_bytes: bytes | None = None
    ) -> "subprocess.CompletedProcess[bytes]": ...


class PiperLoadError(Exception):
    """Raised when Piper's binary or voice model files are missing."""


class PiperSynthesisError(Exception):
    """Raised when Piper fails to synthesize speech."""


class PiperTTS:
    """Text-to-speech engine backed by the local Piper binary (CPU-only)."""

    def __init__(
        self,
        binary_path: Path,
        model_path: Path,
        sample_rate: int = 22050,
        command_runner: CommandRunner | None = None,
    ) -> None:
        """Create a PiperTTS engine (does not check files yet).

        Args:
            binary_path: Path to the piper executable.
            model_path: Path to the .onnx voice model file.
            sample_rate: Sample rate of the voice model's audio output, in
                Hz (22050 for en_US-lessac-medium, per its .onnx.json).
            command_runner: Callable(argv, input_bytes=None) -> CompletedProcess,
                used to run piper. Defaults to subprocess.run with a
                30-second timeout. Injectable for tests.
        """
        self._binary_path = binary_path
        self._model_path = model_path
        self._sample_rate = sample_rate
        self._run: CommandRunner = command_runner or _default_command_runner
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Whether load() has succeeded."""
        return self._loaded

    def load(self) -> None:
        """Verify the piper binary and voice model files exist.

        Raises:
            PiperLoadError: If either file is missing.
        """
        if not self._binary_path.exists():
            logger.error("Piper binary not found at %s", self._binary_path)
            raise PiperLoadError(f"Piper binary not found at {self._binary_path}")
        if not self._model_path.exists():
            logger.error("Piper voice model not found at %s", self._model_path)
            raise PiperLoadError(f"Piper voice model not found at {self._model_path}")
        self._loaded = True

    def unload(self) -> None:
        """Clear the loaded flag. Safe to call even if never loaded."""
        self._loaded = False

    def synthesize(self, text: str) -> "tuple[npt.NDArray[np.float32], int]":
        """Generate speech audio for the given text.

        Args:
            text: The text to speak.

        Returns:
            (audio, sample_rate) — a 1-D float32 numpy array and its rate.

        Raises:
            RuntimeError: If called before load().
            PiperSynthesisError: If the piper subprocess fails or cannot
                be run at all.
        """
        if not self._loaded:
            raise RuntimeError("PiperTTS.synthesize() called before load()")
        try:
            result = self._run(
                [str(self._binary_path), "--model", str(self._model_path), "--output_raw"],
                input_bytes=text.encode("utf-8"),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.error("Failed to run piper: %s", exc)
            raise PiperSynthesisError(f"Failed to run piper: {exc}") from exc
        if result.returncode != 0:
            logger.error("piper exited with code %d", result.returncode)
            raise PiperSynthesisError(f"piper exited with code {result.returncode}")
        samples = np.frombuffer(result.stdout, dtype="<i2").astype(np.float32) / 32768.0
        return samples, self._sample_rate


def _default_command_runner(
    argv: list[str], input_bytes: bytes | None = None
) -> "subprocess.CompletedProcess[bytes]":
    """Default CommandRunner: subprocess.run with a 30-second timeout."""
    return subprocess.run(argv, input=input_bytes, timeout=30.0, capture_output=True)
