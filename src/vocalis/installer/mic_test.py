"""Microphone test for the installer wizard.

Records a couple of seconds from the microphone and says whether Vocalis
could hear anything. It never raises: every problem becomes a result the
wizard can show as a gentle warning (it must never block Finish).
"""
import logging
import time
from dataclasses import dataclass
from typing import Callable, Literal

from vocalis.audio.capture import MicRecorder, pick_input_device
from vocalis.features.dictation import NO_AUDIO_MESSAGE

logger = logging.getLogger("vocalis")

HEARD_PEAK = 0.02
DEFAULT_SECONDS = 2.0
HEARD_MESSAGE = "Great – Vocalis can hear you."
NO_DEVICE_MESSAGE = (
    "Vocalis could not find a microphone. You can still finish setup and fix "
    "this later in your sound settings."
)


@dataclass(frozen=True)
class MicResult:
    """Outcome of one microphone test."""

    verdict: Literal["heard", "quiet", "no_device"]
    peak: float
    message: str


def _default_recorder_factory(device: int | None) -> MicRecorder:
    """Build a real recorder for the chosen input device."""
    return MicRecorder(device=device)


def run_mic_test(
    seconds: float = DEFAULT_SECONDS,
    recorder_factory: Callable[[int | None], MicRecorder] = _default_recorder_factory,
    device_picker: Callable[[], int | None] = pick_input_device,
    sleep: Callable[[float], None] = time.sleep,
) -> MicResult:
    """Record for `seconds` and report whether sound was heard.

    Args:
        seconds: How long to listen.
        recorder_factory: Builds a recorder for a device index (injectable).
        device_picker: Chooses the input device (injectable).
        sleep: Waits while recording (injectable).

    Returns:
        A MicResult. Never raises.
    """
    try:
        recorder = recorder_factory(device_picker())
        recorder.start()
    except Exception:
        logger.exception("Microphone test could not start recording")
        return MicResult("no_device", 0.0, NO_DEVICE_MESSAGE)

    try:
        sleep(seconds)
    except Exception:
        logger.exception("Microphone test was interrupted while waiting")

    try:
        audio = recorder.stop()
    except Exception:
        logger.exception("Microphone test could not stop recording")
        return MicResult("no_device", 0.0, NO_DEVICE_MESSAGE)

    peak = float(abs(audio).max()) if len(audio) else 0.0
    if peak >= HEARD_PEAK:
        return MicResult("heard", peak, HEARD_MESSAGE)
    return MicResult("quiet", peak, NO_AUDIO_MESSAGE)
