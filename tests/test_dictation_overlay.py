"""The floating dictation animation, its levels and its wiring."""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QRectF
from PySide6.QtWidgets import QApplication

from vocalis.audio.capture import MicRecorder
from vocalis.features.dictation import (
    PHASE_DONE,
    PHASE_FAILED,
    PHASE_LISTENING,
    PHASE_TRANSCRIBING,
    DictationFeature,
)
from vocalis.ui import dictation_overlay as overlay_module
from vocalis.ui.dictation_overlay import DictationOverlay, overlay_position, smooth_level
from vocalis.ui.status_hub import StatusHub


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])  # type: ignore[return-value]


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def make_overlay(clock: Clock, enabled: bool = True) -> DictationOverlay:
    return DictationOverlay(lambda: enabled, clock)


# --- pure maths -----------------------------------------------------------------
def test_level_rises_fast_and_falls_slowly() -> None:
    rising = smooth_level(0.0, 0.1)
    falling = smooth_level(1.0, 0.0)
    assert rising > 0.2
    assert 0.85 < falling < 1.0


def test_level_is_clamped_to_one() -> None:
    assert smooth_level(0.0, 50.0) <= 0.56


def test_overlay_sits_centred_above_the_bottom_edge() -> None:
    point = overlay_position(QRectF(0, 0, 1920, 1080), 480, 120)
    assert point.x() == 720
    assert point.y() == 1080 - 120 - overlay_module.BOTTOM_MARGIN


# --- states ---------------------------------------------------------------------
def test_listening_shows_and_animates(qapp: QApplication) -> None:
    overlay = make_overlay(Clock())
    overlay.set_phase(PHASE_LISTENING)
    assert overlay.isVisible() and overlay.is_animating
    assert overlay.phase == PHASE_LISTENING
    overlay.set_phase(PHASE_DONE)
    overlay._finish()


def test_disabled_overlay_stays_hidden(qapp: QApplication) -> None:
    overlay = make_overlay(Clock(), enabled=False)
    overlay.set_phase(PHASE_LISTENING)
    assert not overlay.isVisible() and not overlay.is_animating


def test_phases_before_listening_are_ignored(qapp: QApplication) -> None:
    overlay = make_overlay(Clock())
    overlay.set_phase(PHASE_TRANSCRIBING)
    assert not overlay.isVisible() and overlay.phase == ""


def test_done_fades_out_then_stops_the_timer(qapp: QApplication) -> None:
    clock = Clock()
    overlay = make_overlay(clock)
    overlay.set_phase(PHASE_LISTENING)
    overlay.set_phase(PHASE_TRANSCRIBING)
    assert overlay.phase == PHASE_TRANSCRIBING
    overlay.set_phase(PHASE_DONE)
    clock.now += overlay_module.DONE_HOLD_SECONDS + 0.1
    overlay._tick()
    clock.now += 0.1
    overlay._tick()
    assert overlay.isVisible() and overlay.windowOpacity() < 1.0
    clock.now += overlay_module.FADE_SECONDS + 0.1
    overlay._tick()
    assert not overlay.isVisible() and not overlay.is_animating and overlay.phase == ""


def test_failed_stays_longer_than_done(qapp: QApplication) -> None:
    clock = Clock()
    overlay = make_overlay(clock)
    overlay.set_phase(PHASE_LISTENING)
    overlay.set_phase(PHASE_FAILED)
    clock.now += overlay_module.DONE_HOLD_SECONDS + 0.1
    overlay._tick()
    assert overlay.isVisible() and overlay.windowOpacity() == 1.0
    overlay._finish()


def test_waves_follow_the_microphone_level(qapp: QApplication) -> None:
    overlay = make_overlay(Clock())
    overlay.set_phase(PHASE_LISTENING)
    overlay.set_level(0.2)
    for _ in range(20):
        overlay._tick()
    loud = overlay._level
    overlay.set_level(0.0)
    for _ in range(60):
        overlay._tick()
    assert loud > 0.5 and overlay._level < loud / 2
    overlay._finish()


@pytest.mark.parametrize("phase", [PHASE_LISTENING, PHASE_TRANSCRIBING, PHASE_FAILED])
def test_every_phase_paints_something(qapp: QApplication, phase: str) -> None:
    overlay = make_overlay(Clock())
    overlay.set_phase(PHASE_LISTENING)
    overlay.set_phase(phase)
    image = overlay.grab().toImage()
    assert image.width() == overlay_module.OVERLAY_WIDTH
    assert any(image.pixelColor(x, 60).alpha() > 0 for x in range(0, image.width(), 20))
    overlay._finish()


# --- the hub signals --------------------------------------------------------------
def test_hub_forwards_phase_and_level(qapp: QApplication) -> None:
    hub = StatusHub()
    phases: list[str] = []
    levels: list[float] = []
    hub.dictation_phase_changed.connect(phases.append)
    hub.level_changed.connect(levels.append)
    hub.set_dictation_phase(PHASE_LISTENING)
    hub.set_level(0.25)
    assert phases == [PHASE_LISTENING] and levels == [0.25]


# --- microphone level --------------------------------------------------------------
class FakeStream:
    def __init__(self, **_: object) -> None:
        pass

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...


def test_recorder_reports_block_loudness() -> None:
    levels: list[float] = []
    recorder = MicRecorder(stream_factory=FakeStream, on_level=levels.append)
    recorder._on_audio(np.full((100, 1), 0.5, dtype=np.float32), 100, None, None)
    assert levels == [pytest.approx(0.5)]


def test_a_failing_level_hook_never_breaks_recording() -> None:
    def broken(_: float) -> None:
        raise RuntimeError("boom")

    recorder = MicRecorder(stream_factory=FakeStream, on_level=broken)
    recorder._on_audio(np.ones((10, 1), dtype=np.float32), 10, None, None)
    assert len(recorder._chunks) == 1


def test_empty_blocks_report_no_level() -> None:
    levels: list[float] = []
    recorder = MicRecorder(stream_factory=FakeStream, on_level=levels.append)
    recorder._report_level(np.zeros(0, dtype=np.float32))
    assert levels == []


# --- dictation phases and notifications ------------------------------------------------
class Notes:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def notify(self, title: str, message: str) -> None:
        self.messages.append(message)


class Rec:
    def __init__(self, audio: np.ndarray) -> None:
        self.audio = audio

    def start(self) -> None: ...

    def stop(self) -> np.ndarray:
        return self.audio


class Manager:
    def __init__(self, text: str) -> None:
        self.text = text

    def acquire(self, _id: str, _factory: object) -> object:
        text = self.text
        return type("E", (), {"transcribe": lambda self, audio, sample_rate: text})()

    def release(self, _id: str) -> None: ...


class Output:
    def __init__(self, method: str = "typed") -> None:
        self.method = method

    def deliver(self, _text: str) -> object:
        return type("R", (), {"method": self.method})()


def run_dictation(
    text: str = "hello", audio: np.ndarray | None = None, method: str = "typed", quiet: bool = True
) -> tuple[list[str], list[str]]:
    notes = Notes()
    phases: list[str] = []
    feature = DictationFeature(
        Manager(text),  # type: ignore[arg-type]
        lambda: None,  # type: ignore[arg-type,return-value]
        Rec(np.ones(10, dtype=np.float32) if audio is None else audio),
        Output(method),  # type: ignore[arg-type]
        notes,  # type: ignore[arg-type]
        on_phase=phases.append,
        quiet_notices=lambda: quiet,
    )
    feature.on_press()
    feature.on_release()
    return notes.messages, phases


def test_success_reports_phases_and_stays_quiet() -> None:
    messages, phases = run_dictation()
    assert phases == [PHASE_LISTENING, PHASE_TRANSCRIBING, PHASE_DONE]
    assert messages == []


def test_without_the_animation_notifications_still_appear() -> None:
    messages, _ = run_dictation(quiet=False)
    assert messages == ["Listening...", "Transcribing...", "hello", "Done"]


def test_errors_still_notify_when_quiet() -> None:
    messages, phases = run_dictation(audio=np.zeros(0, dtype=np.float32))
    assert len(messages) == 1 and "No sound" in messages[0]
    assert phases[-1] == PHASE_FAILED


def test_no_speech_fails_the_animation() -> None:
    _, phases = run_dictation(text="  ")
    assert phases[-1] == PHASE_FAILED


def test_clipboard_fallback_notifies_and_fails_the_animation() -> None:
    messages, phases = run_dictation(method="clipboard")
    assert any("copied to the clipboard" in message for message in messages)
    assert phases[-1] == PHASE_FAILED


def test_failing_hooks_never_break_dictation() -> None:
    feature = DictationFeature(
        Manager("x"),  # type: ignore[arg-type]
        lambda: None,  # type: ignore[arg-type,return-value]
        Rec(np.ones(4, dtype=np.float32)),
        Output(),  # type: ignore[arg-type]
        Notes(),  # type: ignore[arg-type]
        on_phase=lambda _p: (_ for _ in ()).throw(RuntimeError("x")),
        quiet_notices=lambda: (_ for _ in ()).throw(RuntimeError("y")),
    )
    feature.on_press()
    feature.on_release()


def test_a_recording_that_cannot_start_fails_the_animation() -> None:
    class Broken(Rec):
        def start(self) -> None:
            raise RuntimeError("no mic")

    phases: list[str] = []
    feature = DictationFeature(
        Manager("x"),  # type: ignore[arg-type]
        lambda: None,  # type: ignore[arg-type,return-value]
        Broken(np.ones(1, dtype=np.float32)),
        Output(),  # type: ignore[arg-type]
        Notes(),  # type: ignore[arg-type]
        on_phase=phases.append,
    )
    feature.on_press()
    assert phases == [PHASE_FAILED]
