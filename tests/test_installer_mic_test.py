"""Tests for the installer microphone test."""
from typing import Any

import numpy as np

from vocalis.features.dictation import NO_AUDIO_MESSAGE
from vocalis.installer.mic_test import MicResult, run_mic_test


class FakeRecorder:
    """Stands in for MicRecorder."""

    def __init__(self, audio: Any = None, start_error: Exception | None = None) -> None:
        self.audio = np.zeros(0, dtype=np.float32) if audio is None else audio
        self.start_error = start_error
        self.started = False
        self.stopped = False

    def start(self) -> None:
        if self.start_error:
            raise self.start_error
        self.started = True

    def stop(self) -> Any:
        self.stopped = True
        return self.audio


def _run(recorder: FakeRecorder, **kwargs: Any) -> MicResult:
    return run_mic_test(
        recorder_factory=lambda device: recorder,  # type: ignore[arg-type,return-value]
        device_picker=lambda: None,
        sleep=lambda s: None,
        **kwargs,
    )


def test_loud_audio_is_heard() -> None:
    result = _run(FakeRecorder(np.array([0.0, -0.5, 0.1], dtype=np.float32)))
    assert result.verdict == "heard"
    assert result.peak == 0.5
    assert "can hear you" in result.message


def test_silence_is_quiet_and_uses_the_existing_message() -> None:
    result = _run(FakeRecorder(np.zeros(100, dtype=np.float32)))
    assert result.verdict == "quiet"
    assert result.message == NO_AUDIO_MESSAGE


def test_empty_capture_is_quiet() -> None:
    result = _run(FakeRecorder())
    assert result.verdict == "quiet"
    assert result.peak == 0.0


def test_start_failure_is_no_device() -> None:
    result = _run(FakeRecorder(start_error=RuntimeError("no mic")))
    assert result.verdict == "no_device"
    assert "could not find a microphone" in result.message


def test_recorder_is_stopped_after_a_successful_start() -> None:
    recorder = FakeRecorder(np.ones(3, dtype=np.float32))
    _run(recorder)
    assert recorder.started and recorder.stopped


def test_uses_the_picked_device() -> None:
    seen: list[int | None] = []

    def factory(device: int | None) -> FakeRecorder:
        seen.append(device)
        return FakeRecorder()

    run_mic_test(
        recorder_factory=factory,  # type: ignore[arg-type]
        device_picker=lambda: 7,
        sleep=lambda s: None,
    )
    assert seen == [7]


def test_waits_for_the_requested_seconds() -> None:
    waits: list[float] = []
    run_mic_test(
        recorder_factory=lambda device: FakeRecorder(),  # type: ignore[arg-type,return-value]
        device_picker=lambda: None,
        sleep=waits.append,
    )
    assert waits == [2.0]


class RaisingStopRecorder(FakeRecorder):
    def stop(self) -> Any:
        self.stopped = True
        raise RuntimeError("stop failed")


def test_stop_returning_none_is_quiet() -> None:
    result = _run(FakeRecorder(audio=None))
    recorder = FakeRecorder()
    recorder.stop = lambda: None  # type: ignore[method-assign,assignment]
    assert _run(recorder).verdict == "quiet"
    assert result.verdict == "quiet"


def test_stop_raising_is_quiet_and_does_not_escape() -> None:
    result = _run(RaisingStopRecorder())
    assert result.verdict == "quiet"
    assert result.message == NO_AUDIO_MESSAGE


def test_sleep_raising_still_stops_the_recorder() -> None:
    recorder = FakeRecorder(np.ones(2, dtype=np.float32))

    def bad_sleep(seconds: float) -> None:
        raise RuntimeError("interrupted")

    result = run_mic_test(
        recorder_factory=lambda device: recorder,  # type: ignore[arg-type,return-value]
        device_picker=lambda: None,
        sleep=bad_sleep,
    )
    assert recorder.stopped
    assert result.verdict == "heard"


def test_keyboard_interrupt_in_sleep_still_stops_the_recorder() -> None:
    recorder = FakeRecorder()

    def bad_sleep(seconds: float) -> None:
        raise KeyboardInterrupt

    try:
        run_mic_test(
            recorder_factory=lambda device: recorder,  # type: ignore[arg-type,return-value]
            device_picker=lambda: None,
            sleep=bad_sleep,
        )
    except KeyboardInterrupt:
        pass
    assert recorder.stopped


def test_peak_exactly_at_threshold_is_heard() -> None:
    assert _run(FakeRecorder(np.array([0.02], dtype=np.float64))).verdict == "heard"


def test_peak_just_below_threshold_is_quiet() -> None:
    assert _run(FakeRecorder(np.array([0.019], dtype=np.float32))).verdict == "quiet"


def test_nan_samples_are_quiet() -> None:
    result = _run(FakeRecorder(np.array([np.nan, np.nan], dtype=np.float32)))
    assert result.verdict == "quiet"
    assert result.peak == 0.0
