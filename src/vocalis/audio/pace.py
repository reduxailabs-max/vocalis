"""Changes the speaking pace of finished speech audio.

Qwen3-TTS has no native speed control of its own, so the pace is applied
to its finished audio with a pitch-preserving time stretch (librosa's
phase vocoder). (Kokoro and Piper gained their own native speed controls
in Tasks 2-3 of Stage 6c and no longer need this module.) librosa is
imported inside the function so merely importing Vocalis does not pay its
start-up cost, and nothing here touches the network.
"""
import logging
from typing import Callable

import numpy as np
import numpy.typing as npt

logger = logging.getLogger("vocalis")

MIN_SPEED = 0.7
MAX_SPEED = 1.6
_NEUTRAL_TOLERANCE = 0.01
_MIN_SAMPLES = 4096


def clamp_speed(speed: float) -> float:
    """Limit a speed factor to the supported range (NaN counts as 1.0)."""
    if speed != speed:  # NaN
        return 1.0
    return max(MIN_SPEED, min(MAX_SPEED, speed))


def apply_speed(
    audio: npt.NDArray[np.float32], sample_rate: int, speed: float
) -> npt.NDArray[np.float32]:
    """Return `audio` played `speed` times faster (>1) or slower (<1).

    Pitch is preserved. The speed is clamped to MIN_SPEED..MAX_SPEED.
    Audio that is empty, too short to stretch, or whose speed is within 1%
    of 1.0 is returned unchanged (the same object).

    Args:
        audio: 1-D float32 samples.
        sample_rate: Sample rate in Hz (kept for the signature's clarity;
            the stretch itself is rate-independent).
        speed: Desired speed factor.
    """
    speed = clamp_speed(speed)
    if abs(speed - 1.0) < _NEUTRAL_TOLERANCE or audio.size < _MIN_SAMPLES:
        return audio
    import librosa

    stretched = librosa.effects.time_stretch(audio.astype(np.float32, copy=False), rate=speed)
    return np.asarray(stretched, dtype=np.float32)


def paced_audio(
    audio: npt.NDArray[np.float32],
    sample_rate: int,
    provider: "Callable[[], float] | None",
) -> npt.NDArray[np.float32]:
    """Apply the caller's current speaking pace to finished audio.

    Since Stage 6c this has exactly one caller, `Qwen3TTS.synthesize()`
    (Kokoro and Piper now use their own native speed controls instead).
    The try/except below still guards against a pace failure breaking a
    reading. Returns `audio` unchanged when `provider` is None. Otherwise calls
    `provider()` to get the desired speed and applies it with
    `apply_speed()`; if either call raises, the exception is logged as a
    warning (with traceback) and the original `audio` is returned
    unchanged.

    Args:
        audio: 1-D float32 samples, already synthesized.
        sample_rate: Sample rate in Hz.
        provider: Callable returning the desired speed factor, or None
            if no pacing should be applied.
    """
    if provider is None:
        return audio
    try:
        speed = provider()
        return apply_speed(audio, sample_rate, speed)
    except Exception:
        logger.warning("Failed to apply speaking pace; using original audio", exc_info=True)
        return audio
