"""Microphone audio capture for dictation.

Records float32 mono audio at a fixed sample rate while the caller holds
the stream open (e.g. NUMLOCK held down), with a hard safety cutoff so a
stuck hotkey can never record forever. The safety cutoff only stops the
underlying stream — it does not discard already-captured audio, so a
later stop() call still returns everything captured up to the cutoff.
"""
import logging
import threading
from typing import Any, Callable

import numpy as np
import numpy.typing as npt
import sounddevice as sd

logger = logging.getLogger("vocalis")

DEFAULT_SAMPLE_RATE = 16000
_PREFERRED_INPUT_NAMES = ("pipewire", "pulse")


def pick_input_device(
    query_devices: Callable[[], Any] = sd.query_devices,
) -> int | None:
    """Pick the microphone path that talks to the desktop sound server.

    The plain ALSA "default" input can be silently routed nowhere by an old
    ~/.asoundrc, so the stream opens but no sound ever arrives. The
    "pipewire" and "pulse" inputs go straight to the sound server instead.

    Args:
        query_devices: Returns the device list (injectable for tests).

    Returns:
        The index of the "pipewire" input, else the "pulse" input, else None
        (meaning: use the system default).
    """
    try:
        devices = list(query_devices())
    except Exception:
        logger.exception("Could not list audio input devices; using system default")
        return None
    for wanted in _PREFERRED_INPUT_NAMES:
        for index, device in enumerate(devices):
            if device["name"] == wanted and device["max_input_channels"] > 0:
                return index
    return None


class RecordingAlreadyActiveError(Exception):
    """Raised when start() is called while a recording is already active."""


class MicRecorder:
    """Captures microphone audio into an in-memory float32 buffer."""

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        max_duration_seconds: float = 60.0,
        stream_factory: Callable[..., Any] | None = None,
        device: int | None = None,
    ) -> None:
        """Create a MicRecorder.

        Args:
            sample_rate: Capture sample rate in Hz.
            max_duration_seconds: Hard safety cutoff — recording auto-stops
                the audio stream after this many seconds even if stop() is
                never called.
            stream_factory: Callable(**kwargs) -> stream object with
                start()/stop()/close() methods, used to construct the audio
                stream. Defaults to sounddevice.InputStream; injectable for
                tests.
            device: Index of the input device to record from. None keeps
                the system default.
        """
        self._device = device
        self._sample_rate = sample_rate
        self._max_duration_seconds = max_duration_seconds
        self._stream_factory = stream_factory or sd.InputStream
        self._chunks: list[npt.NDArray[np.float32]] = []
        self._stream: Any | None = None
        self._lock = threading.Lock()
        self._safety_timer: threading.Timer | None = None

    @property
    def is_recording(self) -> bool:
        """Whether the underlying audio stream is currently open."""
        return self._stream is not None

    def start(self) -> None:
        """Begin recording from the default microphone.

        Raises:
            RecordingAlreadyActiveError: If a recording is already active.
        """
        if self.is_recording:
            raise RecordingAlreadyActiveError("MicRecorder.start() called while already recording")

        self._chunks = []
        stream_kwargs: dict[str, Any] = {
            "samplerate": self._sample_rate,
            "channels": 1,
            "dtype": "float32",
            "callback": self._on_audio,
        }
        if self._device is not None:
            stream_kwargs["device"] = self._device
        stream = self._stream_factory(**stream_kwargs)
        stream.start()
        self._stream = stream

        self._safety_timer = threading.Timer(self._max_duration_seconds, self._stop_stream_only)
        self._safety_timer.daemon = True
        self._safety_timer.start()
        logger.info("Recording started (sample_rate=%d, device=%s)", self._sample_rate, self._device)

    def stop(self) -> npt.NDArray[np.float32]:
        """Stop recording and return everything captured.

        Returns:
            A 1-D float32 numpy array of all captured samples (empty array
            if nothing was captured).
        """
        if self._safety_timer is not None:
            self._safety_timer.cancel()
            self._safety_timer = None

        self._stop_stream_only()

        with self._lock:
            chunks = self._chunks
            self._chunks = []

        if not chunks:
            logger.info("Recording stopped: no audio captured")
            return np.zeros(0, dtype=np.float32)

        audio = np.concatenate(chunks).astype(np.float32).reshape(-1)
        logger.info("Recording stopped: %.2fs captured", len(audio) / self._sample_rate)
        return audio

    def _stop_stream_only(self) -> None:
        """Stop and close the underlying stream without clearing captured
        chunks, so a later stop() call can still return them. Safe to call
        multiple times (e.g. once from the safety timer, once from stop()).
        """
        with self._lock:
            stream, self._stream = self._stream, None
        if stream is not None:
            stream.stop()
            stream.close()
            logger.warning(
                "Recording stream stopped (safety cutoff or explicit stop)"
            )

    def _on_audio(
        self, indata: npt.NDArray[np.float32], frames: int, time_info: object, status: object
    ) -> None:
        """sounddevice callback: append each incoming audio block."""
        if status:
            logger.warning("Audio input status: %s", status)
        with self._lock:
            self._chunks.append(indata.copy().reshape(-1))
