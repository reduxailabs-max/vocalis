"""Tests for vocalis.features.quick_reader.QuickReaderFeature.

Uses fakes for every collaborator — this suite verifies orchestration
logic only, the same approach test_dictation.py already established.
"""
import logging
import subprocess
import threading
import time
from typing import Any, Callable

import numpy as np
import numpy.typing as npt
import pytest

from tests.fake_output import FakeAudioPlayer, OutputRecorder, pump_stream_factory
from vocalis.audio.playback import AudioPlayer
from vocalis.engines.chain import make_downshift_handler
from vocalis.features.chunked_speaker import SpeakOutcome
from vocalis.features.quick_reader import QuickReaderFeature
from vocalis.features.reading_coordinator import ReadingCoordinator
from vocalis.model_manager import ModelManager
from vocalis.output.clipboard import ClipboardReader


class FakeSynthesizer:
    def __init__(
        self,
        audio: npt.NDArray[np.float32],
        sample_rate: int = 22050,
        raise_exc: Exception | None = None,
        delay_seconds: float = 0.0,
    ) -> None:
        self.audio = audio
        self.sample_rate = sample_rate
        self.raise_exc = raise_exc
        self.delay_seconds = delay_seconds
        self.calls: list[str] = []
        self._loaded = False

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def synthesize(self, text: str):  # type: ignore[no-untyped-def]
        self.calls.append(text)
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.audio, self.sample_rate


class FakeClipboardReader:
    def __init__(self, text: str) -> None:
        self.text = text

    def read(self) -> str:
        return self.text


class ResettablePlayer(FakeAudioPlayer):
    """A real AudioPlayer on a fake sound output.

    `delay_seconds > 0` means slow, real-time output (a clip really keeps
    playing for its whole length, so a stop can land mid-playback); 0 means
    fast output that plays a whole clip almost at once. reset() starts a
    fresh recording of what was played.
    """

    def __init__(self, delay_seconds: float = 0.0) -> None:
        """Pick slow (real-time) or fast output."""
        self._speedup = 1.0 if delay_seconds > 0 else 50.0
        super().__init__(speedup=self._speedup)

    def reset(self) -> None:
        """Forget everything played so far and record from scratch."""
        self.output = OutputRecorder()
        self._stream_factory = pump_stream_factory(self.output, self._speedup)


class FakeNotifier:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def notify(self, title: str, message: str = "") -> None:
        self.messages.append((title, message))


def make_feature(
    primary: FakeSynthesizer,
    fallback: FakeSynthesizer,
    clipboard_text: str,
    player: AudioPlayer | None = None,
    reading_coordinator: "ReadingCoordinator | None" = None,
    on_reading_changed: "Callable[[bool], None] | None" = None,
    speed_provider: "Callable[[], float] | None" = None,
) -> "tuple[QuickReaderFeature, FakeSynthesizer, FakeSynthesizer, Any, FakeNotifier, ModelManager]":
    manager = ModelManager()
    notifier = FakeNotifier()
    player = player or FakeAudioPlayer()
    feature = QuickReaderFeature(
        model_manager=manager,
        primary_engine_factory=lambda: primary,
        fallback_engine=fallback,
        clipboard_reader=FakeClipboardReader(clipboard_text),  # type: ignore[arg-type]
        player=player,
        notifier=notifier,  # type: ignore[arg-type]
        reading_coordinator=reading_coordinator,
        on_reading_changed=on_reading_changed,
        speed_provider=speed_provider,
    )
    return feature, primary, fallback, player, notifier, manager


def sine(freq: float = 440.0, seconds: float = 2.0, rate: int = 24000) -> npt.NDArray[np.float32]:
    """Make a quiet sine wave long enough for a real time-stretch."""
    t = np.arange(int(rate * seconds)) / rate
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def wait_until(predicate, timeout: float = 5.0) -> None:  # type: ignore[no-untyped-def]
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("Condition was not met in time")


def test_trigger_with_empty_clipboard_notifies_and_does_nothing() -> None:
    primary = FakeSynthesizer(audio=np.zeros(1, dtype=np.float32))
    fallback = FakeSynthesizer(audio=np.zeros(1, dtype=np.float32))
    feature, _, _, player, notifier, _ = make_feature(primary, fallback, clipboard_text="")

    feature.on_trigger()
    wait_until(lambda: any("Nothing copied" in msg for _, msg in notifier.messages))

    assert player.output.opened == 0
    assert primary.calls == []


def test_trigger_reads_speaks_and_notifies_done() -> None:
    audio = np.full(2400, 0.1, dtype=np.float32)
    primary = FakeSynthesizer(audio=audio, sample_rate=24000)
    fallback = FakeSynthesizer(audio=np.zeros(1, dtype=np.float32))
    feature, _, _, player, notifier, _ = make_feature(primary, fallback, clipboard_text="hello there")

    feature.on_trigger()
    wait_until(lambda: any(msg == "Done" for _, msg in notifier.messages))

    assert primary.calls == ["hello there"]
    assert player.output.opened == 1
    assert player.output.kwargs[0]["samplerate"] == 24000
    played = player.output.played()
    assert abs(len(played) - len(audio)) < 0.1 * len(audio)  # only the ~8 ms fades are lost
    assert float(played.max()) == pytest.approx(0.1)
    assert any(msg == "Reading..." for _, msg in notifier.messages)
    assert any(msg == "hello there" for _, msg in notifier.messages)


def test_primary_failure_falls_back_to_secondary_engine() -> None:
    primary = FakeSynthesizer(audio=np.zeros(1, dtype=np.float32), raise_exc=RuntimeError("cuda oom"))
    fallback_audio = np.full(2205, 0.4, dtype=np.float32)
    fallback = FakeSynthesizer(audio=fallback_audio, sample_rate=22050)
    feature, _, _, player, notifier, _ = make_feature(primary, fallback, clipboard_text="fallback please")

    feature.on_trigger()
    wait_until(lambda: any(msg == "Done" for _, msg in notifier.messages))

    assert fallback.calls == ["fallback please"]
    assert player.output.kwargs[0]["samplerate"] == 22050
    assert float(player.output.played().max()) == pytest.approx(0.4)
    assert any("backup voice" in msg.lower() for _, msg in notifier.messages)


def test_both_engines_failing_notifies_failure_and_does_not_crash() -> None:
    primary = FakeSynthesizer(audio=np.zeros(1, dtype=np.float32), raise_exc=RuntimeError("cuda oom"))
    fallback = FakeSynthesizer(audio=np.zeros(1, dtype=np.float32), raise_exc=RuntimeError("piper missing"))
    feature, _, _, player, notifier, _ = make_feature(primary, fallback, clipboard_text="uh oh")

    feature.on_trigger()  # must not raise
    wait_until(lambda: any("Could not generate speech" in msg for _, msg in notifier.messages))

    assert player.output.opened == 0


def test_second_trigger_while_playing_stops_and_notifies_stopped() -> None:
    primary = FakeSynthesizer(audio=np.full(72000, 0.1, dtype=np.float32), sample_rate=24000)
    fallback = FakeSynthesizer(audio=np.zeros(1, dtype=np.float32))
    player = ResettablePlayer(delay_seconds=2.0)  # real-time output: 3 s of sound is still playing at the 2nd press
    feature, _, _, _, notifier, _ = make_feature(primary, fallback, clipboard_text="a long passage", player=player)

    feature.on_trigger()
    wait_until(lambda: player.output.opened == 1)
    feature.on_trigger()  # the interrupting press

    wait_until(lambda: any(msg == "Stopped" for _, msg in notifier.messages))
    wait_until(lambda: player.output.closed >= 1)
    assert not any(msg == "Done" for _, msg in notifier.messages)


def test_second_trigger_while_synthesizing_cancels_before_playback_starts() -> None:
    primary = FakeSynthesizer(audio=np.zeros(1, dtype=np.float32), delay_seconds=0.3)
    fallback = FakeSynthesizer(audio=np.zeros(1, dtype=np.float32))
    feature, _, _, player, notifier, _ = make_feature(primary, fallback, clipboard_text="slow one")

    feature.on_trigger()
    wait_until(lambda: any(msg == "Reading..." for _, msg in notifier.messages))
    feature.on_trigger()  # interrupt while still synthesizing

    wait_until(lambda: any(msg == "Stopped" for _, msg in notifier.messages))
    time.sleep(0.5)  # let the background synthesis finish
    assert player.output.opened == 0  # playback must never start after being cancelled


class TriggeringNotifier:
    """A notifier whose notify() re-enters feature.on_trigger() the
    moment it sees the live-text notification -- simulating a real
    Ctrl+Alt+R press landing while the copied text is being announced."""

    def __init__(self, trigger_on_message: str) -> None:
        self._trigger_on_message = trigger_on_message
        self.messages: list[tuple[str, str]] = []
        self.feature: QuickReaderFeature | None = None

    def notify(self, title: str, message: str = "") -> None:
        self.messages.append((title, message))
        if message == self._trigger_on_message and self.feature is not None:
            self.feature.on_trigger()


def test_stop_landing_during_live_text_notification_stops_actual_playback() -> None:
    """A press landing while the copied text is being announced (which now
    happens on a side thread while the sound is already starting) must stop
    the sound well before the clip could finish, and no 'Done' may follow."""
    primary = FakeSynthesizer(audio=np.full(72000, 0.1, dtype=np.float32), sample_rate=24000)  # a 3 s clip
    fallback = FakeSynthesizer(audio=np.zeros(1, dtype=np.float32))
    player = FakeAudioPlayer(speedup=1.0)  # real-time output
    notifier = TriggeringNotifier(trigger_on_message="hello there")
    feature = QuickReaderFeature(
        model_manager=ModelManager(),
        primary_engine_factory=lambda: primary,
        fallback_engine=fallback,
        clipboard_reader=FakeClipboardReader("hello there"),  # type: ignore[arg-type]
        player=player,
        notifier=notifier,  # type: ignore[arg-type]
    )
    notifier.feature = feature

    feature.on_trigger()
    wait_until(lambda: any(msg == "Stopped" for _, msg in notifier.messages))
    time.sleep(0.5)  # the stop may land before the stream opens (then none ever opens) or just after

    assert player.output.closed == player.output.opened  # no stream left open
    assert player.output.frames_pulled < 4800  # far less than the 3 s clip
    assert player.is_playing is False
    assert not any(msg == "Done" for _, msg in notifier.messages)


def test_third_trigger_after_stop_starts_a_fresh_reading() -> None:
    primary = FakeSynthesizer(audio=np.full(72000, 0.1, dtype=np.float32), sample_rate=24000)
    fallback = FakeSynthesizer(audio=np.zeros(1, dtype=np.float32))
    player = ResettablePlayer(delay_seconds=3.0)
    feature, _, _, _, _notifier, _ = make_feature(primary, fallback, clipboard_text="again", player=player)

    feature.on_trigger()
    wait_until(lambda: player.output.opened == 1)
    feature.on_trigger()  # stop
    wait_until(lambda: player.output.closed >= 1)
    feature.on_trigger()  # fresh start
    wait_until(lambda: player.output.opened == 2)
    feature.stop()


def test_starting_a_reading_claims_coordinator_ownership() -> None:
    coordinator = ReadingCoordinator()
    primary = FakeSynthesizer(audio=np.zeros(1, dtype=np.float32))
    fallback = FakeSynthesizer(audio=np.zeros(1, dtype=np.float32))
    feature, _, _, _, _, _ = make_feature(
        primary, fallback, clipboard_text="hello", reading_coordinator=coordinator
    )

    feature.on_trigger()
    wait_until(lambda: coordinator._owner is feature)  # noqa: SLF001


def test_a_different_owner_starting_stops_the_quick_reader() -> None:
    coordinator = ReadingCoordinator()
    primary = FakeSynthesizer(audio=np.full(72000, 0.1, dtype=np.float32), sample_rate=24000)
    fallback = FakeSynthesizer(audio=np.zeros(1, dtype=np.float32))
    player = ResettablePlayer(delay_seconds=2.0)  # real-time output, still playing when preempted
    feature, _, _, _, notifier, _ = make_feature(
        primary,
        fallback,
        clipboard_text="a long passage",
        player=player,
        reading_coordinator=coordinator,
    )

    feature.on_trigger()
    wait_until(lambda: player.output.opened == 1)
    coordinator.start_reading(object(), lambda: None)  # a different owner preempts it

    wait_until(lambda: any(msg == "Stopped" for _, msg in notifier.messages))
    wait_until(lambda: player.output.closed >= 1)


class GatedClipboardRunner:
    """Command runner (ClipboardReader's injectable seam) whose call blocks
    until the test opens the gate, then returns the given text as xclip
    output. Plugs into a real ClipboardReader, so no cast is needed."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.entered = threading.Event()
        self.gate = threading.Event()

    def __call__(self, argv: "list[str]") -> "subprocess.CompletedProcess[bytes]":
        """Block until the gate opens, then return the text as stdout."""
        self.entered.set()
        self.gate.wait(timeout=5.0)
        return subprocess.CompletedProcess(argv, 0, stdout=self.text.encode(), stderr=b"")


def test_preempting_an_idle_quick_reader_is_a_noop_and_next_press_reads() -> None:
    audio = np.full(2400, 0.1, dtype=np.float32)
    primary = FakeSynthesizer(audio=audio)
    fallback = FakeSynthesizer(audio=np.zeros(1, dtype=np.float32))
    coordinator = ReadingCoordinator()
    feature, _, _, player, notifier, _ = make_feature(
        primary, fallback, clipboard_text="hello", reading_coordinator=coordinator
    )
    gated = GatedClipboardRunner("hello")
    feature._clipboard_reader = ClipboardReader(command_runner=gated)  # noqa: SLF001

    # Start, then cancel while still reading the clipboard (mid-prepare).
    feature.on_trigger()
    assert gated.entered.wait(timeout=2.0)
    feature.on_trigger()
    wait_until(lambda: sum(1 for _, m in notifier.messages if m == "Stopped") == 1)

    # Another feature now preempts; the stale claim still names Quick Reader.
    coordinator.start_reading(object(), lambda: None)
    time.sleep(0.1)  # let any (wrongly) spawned stop thread run

    assert feature._active is False  # noqa: SLF001
    assert sum(1 for _, m in notifier.messages if m == "Stopped") == 1

    # The stale worker finishes and bails out; the next press must read.
    gated.gate.set()
    feature.on_trigger()
    wait_until(lambda: player.output.opened == 1)
    wait_until(lambda: len(player.output.played()) > 0)
    assert player.output.kwargs[0]["samplerate"] == 22050


def _quiet_feature(
    changes: "list[bool]", player: ResettablePlayer | None = None
) -> "tuple[QuickReaderFeature, ResettablePlayer, FakeNotifier]":
    """A feature whose slow (3 s) sound is still playing until stopped; reports go to `changes`."""
    primary = FakeSynthesizer(audio=np.full(72000, 0.1, dtype=np.float32), sample_rate=24000)
    fallback = FakeSynthesizer(audio=np.zeros(1, dtype=np.float32))
    feature, _, _, player, notifier, _ = make_feature(
        primary, fallback, "some text", player=player or ResettablePlayer(delay_seconds=3.0),
        on_reading_changed=changes.append,
    )
    return feature, player, notifier


def test_reading_changes_are_reported_start_and_stop_synchronously() -> None:
    changes: list[bool] = []
    feature, player, _ = _quiet_feature(changes)

    feature.on_trigger()
    assert changes == [True]
    wait_until(lambda: player.output.opened == 1)
    feature.on_trigger()  # second press: False is reported before returning
    assert changes == [True, False]
    wait_until(lambda: player.output.closed >= 1)
    assert changes == [True, False]


def test_press_stop_press_quickly_ends_with_active_true() -> None:
    changes: list[bool] = []
    feature, player, notifier = _quiet_feature(changes)

    feature.on_trigger()
    wait_until(lambda: player.output.opened == 1)
    feature.on_trigger()
    wait_until(lambda: any(msg == "Stopped" for _, msg in notifier.messages))
    feature.on_trigger()
    wait_until(lambda: player.output.opened == 2)
    time.sleep(0.3)  # the first reading's worker has long finished by now

    assert changes == [True, False, True]
    assert player.output.closed == 1  # only the first stream; the stale worker did not touch the second
    feature.stop()  # cleanup so the worker thread ends


def test_public_stop_ends_an_active_reading_and_is_a_noop_when_idle() -> None:
    changes: list[bool] = []
    feature, player, _ = _quiet_feature(changes)

    feature.stop()
    assert changes == []
    assert player.output.closed == 0

    feature.on_trigger()
    wait_until(lambda: player.output.opened == 1)
    feature.stop()
    assert changes == [True, False]
    wait_until(lambda: player.output.closed >= 1)


def test_natural_end_reports_false_once() -> None:
    changes: list[bool] = []
    feature, _player, notifier = _quiet_feature(changes, player=ResettablePlayer())

    feature.on_trigger()
    wait_until(lambda: any(msg == "Done" for _, msg in notifier.messages))

    assert changes == [True, False]


def test_raising_reading_hook_does_not_break_reading() -> None:
    def broken(active: bool) -> None:
        raise RuntimeError("ui bug")

    primary = FakeSynthesizer(audio=np.zeros(3, dtype=np.float32))
    fallback = FakeSynthesizer(audio=np.zeros(1, dtype=np.float32))
    feature, _, _, player, notifier, _ = make_feature(
        primary, fallback, "some text", player=ResettablePlayer(), on_reading_changed=broken
    )

    feature.on_trigger()
    wait_until(lambda: any(msg == "Done" for _, msg in notifier.messages))

    assert player.output.opened == 1


def test_late_end_report_after_a_new_press_leaves_reported_state_true() -> None:
    changes: list[bool] = []
    feature, _player, _ = _quiet_feature(changes)
    feature._active = True  # noqa: SLF001  (a reading is running)
    feature._report_reading_state()  # noqa: SLF001
    assert changes == [True]

    # natural end clears the flag, but before its report goes out a new press lands
    feature._active = False  # noqa: SLF001
    feature._active = True  # noqa: SLF001
    feature._report_reading_state()  # the late report from the ended reading  # noqa: SLF001

    assert changes == [True]
    assert feature._last_reported is True  # noqa: SLF001

    # the state still drives the hook: when the new reading really ends, False follows
    feature._active = False  # noqa: SLF001
    feature._report_reading_state()  # noqa: SLF001
    assert changes == [True, False]


def test_widget_stop_between_flip_and_true_report_leaves_no_stuck_true() -> None:
    changes: list[bool] = []
    feature, _player, _ = _quiet_feature(changes)
    feature._active = True  # noqa: SLF001  (flipped, True not yet reported)
    feature._active = False  # noqa: SLF001  (a widget stop() lands here)
    feature._report_reading_state()  # noqa: SLF001
    assert changes == []
    assert feature._last_reported is False  # noqa: SLF001


def test_reporting_twice_without_a_state_change_calls_the_hook_once() -> None:
    changes: list[bool] = []
    feature, _player, _ = _quiet_feature(changes)
    feature._active = True  # noqa: SLF001
    feature._report_reading_state()  # noqa: SLF001
    feature._report_reading_state()  # noqa: SLF001
    assert changes == [True]


def test_a_stale_reading_finishing_does_not_release_the_newer_readings_claim() -> None:
    coordinator = ReadingCoordinator()
    primary = FakeSynthesizer(audio=np.zeros(3, dtype=np.float32))
    fallback = FakeSynthesizer(audio=np.zeros(1, dtype=np.float32))
    feature, _, _, _player, _, _ = make_feature(
        primary, fallback, "some text", player=ResettablePlayer(delay_seconds=3.0),
        reading_coordinator=coordinator,
    )
    feature.on_trigger()  # reading A
    first_generation = feature._generation  # noqa: SLF001
    feature.on_trigger()  # stop A
    feature.on_trigger()  # reading B
    assert feature._generation == first_generation + 1  # noqa: SLF001

    feature._deactivate(first_generation)  # A's stale worker finishes late  # noqa: SLF001

    document_stops: list[str] = []
    coordinator.start_reading(object(), lambda: document_stops.append("doc"))  # a Document reading starts
    wait_until(lambda: not feature._active)  # noqa: SLF001
    assert feature._active is False  # noqa: SLF001  # B was stopped by the coordinator
    assert document_stops == []


class _LockSpyCoordinator(ReadingCoordinator):
    """A coordinator that records whether the feature's own lock was held
    each time finish_reading() was called."""

    def __init__(self) -> None:
        super().__init__()
        self.feature: "QuickReaderFeature | None" = None
        self.lock_held_at_finish: "list[bool]" = []

    def finish_reading(self, owner: object) -> None:
        assert self.feature is not None
        self.lock_held_at_finish.append(self.feature._lock.locked())  # noqa: SLF001
        super().finish_reading(owner)


def test_the_reading_claim_is_released_while_holding_the_feature_lock() -> None:
    coordinator = _LockSpyCoordinator()
    feature, _, _, player, _, _ = make_feature(
        FakeSynthesizer(audio=np.zeros(1, dtype=np.float32)),
        FakeSynthesizer(audio=np.zeros(1, dtype=np.float32)),
        clipboard_text="hello",
        reading_coordinator=coordinator,
    )
    coordinator.feature = feature

    feature.on_trigger()
    wait_until(lambda: bool(coordinator.lock_held_at_finish))

    assert coordinator.lock_held_at_finish == [True]


def _played_length(player: Any) -> int:
    """How many audible samples reached the (fake) sound card."""
    return len(player.output.played())


def test_no_speed_provider_leaves_audio_unchanged() -> None:
    """Without a speed_provider, the clip's length is unchanged (only the ~8 ms fades are lost)."""
    audio = sine()
    primary = FakeSynthesizer(audio=audio, sample_rate=24000)
    fallback = FakeSynthesizer(audio=np.zeros(1, dtype=np.float32))
    feature, _, _, player, notifier, _ = make_feature(primary, fallback, clipboard_text="pace me")

    feature.on_trigger()
    wait_until(lambda: any(msg == "Done" for _, msg in notifier.messages))

    assert player.output.opened == 1
    assert player.output.kwargs[0]["samplerate"] == 24000
    assert abs(_played_length(player) - len(audio)) < 0.03 * len(audio)


def test_speed_provider_returning_one_leaves_audio_unchanged() -> None:
    """A neutral speed (1.0) also leaves the length unchanged."""
    audio = sine()
    primary = FakeSynthesizer(audio=audio, sample_rate=24000)
    fallback = FakeSynthesizer(audio=np.zeros(1, dtype=np.float32))
    feature, _, _, player, notifier, _ = make_feature(
        primary, fallback, clipboard_text="pace me", speed_provider=lambda: 1.0
    )

    feature.on_trigger()
    wait_until(lambda: any(msg == "Done" for _, msg in notifier.messages))

    assert player.output.opened == 1
    assert abs(_played_length(player) - len(audio)) < 0.03 * len(audio)


def test_speed_provider_shortens_the_audio_handed_to_the_player() -> None:
    """A speed_provider returning 1.5 shortens the played audio by ~1/1.5."""
    audio = sine()
    primary = FakeSynthesizer(audio=audio, sample_rate=24000)
    fallback = FakeSynthesizer(audio=np.zeros(1, dtype=np.float32))
    feature, _, _, player, notifier, _ = make_feature(
        primary, fallback, clipboard_text="pace me", speed_provider=lambda: 1.5
    )

    feature.on_trigger()
    wait_until(lambda: any(msg == "Done" for _, msg in notifier.messages))

    assert player.output.opened == 1
    assert player.output.kwargs[0]["samplerate"] == 24000
    assert abs(_played_length(player) - len(audio) / 1.5) < 0.03 * len(audio)


def test_a_raising_speed_provider_still_plays_the_original_audio(
    caplog: "pytest.LogCaptureFixture",
) -> None:
    """A speed_provider that raises must never break the reading: the
    original audio still plays and a warning is logged."""
    audio = sine()
    primary = FakeSynthesizer(audio=audio, sample_rate=24000)
    fallback = FakeSynthesizer(audio=np.zeros(1, dtype=np.float32))

    def exploding_provider() -> float:
        raise RuntimeError("settings unavailable")

    feature, _, _, player, notifier, _ = make_feature(
        primary, fallback, clipboard_text="pace me", speed_provider=exploding_provider
    )

    with caplog.at_level(logging.WARNING, logger="vocalis"):
        feature.on_trigger()
        wait_until(lambda: any(msg == "Done" for _, msg in notifier.messages))

    assert player.output.opened == 1
    assert abs(_played_length(player) - len(audio)) < 0.03 * len(audio)
    assert any(record.levelno == logging.WARNING for record in caplog.records)


def _build_with_selector(
    preference: str,
    factories: "dict[str, Callable[[], object]]",
    fallback: FakeSynthesizer,
) -> "tuple[QuickReaderFeature, FakeNotifier]":
    """Build a feature whose engines are chosen by a selector (no real hardware)."""
    from vocalis.engines.registry import EngineSelector
    from vocalis.hardware import HardwareInfo

    notifier = FakeNotifier()
    selector = EngineSelector(
        preference=lambda: preference,
        hardware_probe=lambda: HardwareInfo(None, None, None, None, None),
    )
    feature = QuickReaderFeature(
        model_manager=ModelManager(),
        primary_engine_factory=lambda: pytest.fail("the old primary factory must not be used"),
        fallback_engine=fallback,
        clipboard_reader=FakeClipboardReader("Hello."),  # type: ignore[arg-type]
        player=FakeAudioPlayer(),
        notifier=notifier,  # type: ignore[arg-type]
        engine_selector=selector,
        engine_factories=factories,  # type: ignore[arg-type]
    )
    return feature, notifier


def test_a_selector_chooses_the_engine_for_each_reading() -> None:
    """With a selector, the planned engine is used and the primary factory is ignored."""
    kokoro = FakeSynthesizer(audio=np.ones(3, dtype=np.float32), sample_rate=24000)
    fallback = FakeSynthesizer(audio=np.zeros(1, dtype=np.float32))
    feature, notifier = _build_with_selector("kokoro", {"kokoro": lambda: kokoro}, fallback)

    source = feature._new_source()  # noqa: SLF001
    audio, rate = source.synthesize("Hello.")
    source.close()

    assert kokoro.calls == ["Hello."]
    assert fallback.calls == []
    assert rate == 24000
    assert len(audio) == 3
    assert notifier.messages == []


def test_a_failing_planned_engine_notifies_the_user_and_uses_the_next() -> None:
    """The downshift text is unchanged: 'Primary voice unavailable — using backup voice'."""
    fallback = FakeSynthesizer(audio=np.ones(2, dtype=np.float32), sample_rate=22050)

    def broken_kokoro() -> object:
        raise RuntimeError("kokoro will not load")

    feature, notifier = _build_with_selector("kokoro", {"kokoro": broken_kokoro}, fallback)

    source = feature._new_source()  # noqa: SLF001
    _audio, rate = source.synthesize("Hello.")
    source.close()

    assert fallback.calls == ["Hello."]
    assert rate == 22050
    assert [msg for _title, msg in notifier.messages] == ["Primary voice unavailable — using backup voice"]


def test_a_failing_backup_voice_sends_no_downshift_notification() -> None:
    """When Piper itself fails there is no backup left, so nothing is announced."""
    notifier = FakeNotifier()
    handler = make_downshift_handler(lambda message: notifier.notify("t", message))

    handler("piper", None)

    assert notifier.messages == []


def test_a_long_clipboard_is_read_in_several_chunks_on_one_stream() -> None:
    """Long text is made in several pieces yet heard on one continuous stream, nothing lost or repeated."""
    sentences = " ".join(f"This is sentence number {n} of a rather long clipboard text." for n in range(30))
    primary = FakeSynthesizer(audio=np.full(2400, 0.1, dtype=np.float32), sample_rate=24000)
    feature, _, _, player, notifier, _ = make_feature(
        primary, FakeSynthesizer(audio=np.zeros(1, dtype=np.float32)), clipboard_text=sentences
    )

    feature.on_trigger()
    wait_until(lambda: any(msg == "Done" for _, msg in notifier.messages), timeout=10.0)

    assert len(primary.calls) > 1  # chunked, not one giant call
    assert player.output.opened == 1  # but one continuous stream
    assert " ".join(primary.calls).split() == sentences.split()  # nothing lost, nothing repeated


def test_stop_pressed_while_the_first_piece_is_still_being_made_never_makes_sound() -> None:
    """A stop during the very first synthesis means no sound is ever opened."""
    primary = FakeSynthesizer(audio=np.full(2400, 0.1, dtype=np.float32), delay_seconds=0.4)
    feature, _, _, player, notifier, _ = make_feature(
        primary, FakeSynthesizer(audio=np.zeros(1, dtype=np.float32)), clipboard_text="One sentence here."
    )

    feature.on_trigger()
    wait_until(lambda: len(primary.calls) == 1)  # synthesis has begun
    feature.on_trigger()  # stop while the very first piece is still being made

    wait_until(lambda: any(msg == "Stopped" for _, msg in notifier.messages))
    threading.Event().wait(0.7)
    assert player.output.opened == 0
    assert not any(msg == "Done" for _, msg in notifier.messages)


def test_a_broken_sound_device_reports_playback_failed() -> None:
    """A sound device that refuses to start is reported, never 'Done'."""

    class RefusingStream:
        """A stream whose start() always fails."""

        def __init__(self, **kwargs: Any) -> None:
            """Accept and ignore the stream arguments."""

        def start(self) -> None:
            """Refuse to start."""
            raise RuntimeError("device busy")

        def stop(self) -> None:
            """Nothing to stop."""

        def close(self) -> None:
            """Nothing to close."""

    primary = FakeSynthesizer(audio=np.full(2400, 0.1, dtype=np.float32), sample_rate=24000)
    changes: "list[bool]" = []
    feature, _, _, _, notifier, _ = make_feature(
        primary,
        FakeSynthesizer(audio=np.zeros(1, dtype=np.float32)),
        clipboard_text="Hello there.",
        player=AudioPlayer(stream_factory=lambda **kw: RefusingStream(**kw)),
        on_reading_changed=changes.append,
    )

    feature.on_trigger()

    wait_until(lambda: any(msg == "Playback failed" for _, msg in notifier.messages))
    wait_until(lambda: changes == [True, False])
    assert not any(msg == "Done" for _, msg in notifier.messages)


class _BlockingNotifier(FakeNotifier):
    """A notifier whose notify() of one chosen message waits for the test."""

    def __init__(self, block_on: str) -> None:
        """Remember which message blocks."""
        super().__init__()
        self._block_on = block_on
        self.entered = threading.Event()
        self.gate = threading.Event()

    def notify(self, title: str, message: str = "") -> None:
        """Record the message; hold the caller if it is the chosen one."""
        super().notify(title, message)
        if message == self._block_on:
            self.entered.set()
            self.gate.wait(timeout=5.0)


def test_stop_during_the_reading_notification_never_publishes_a_speaker_or_makes_sound() -> None:
    """A stop landing before the speaker exists is honoured: the worker bails out, no voice is even asked."""
    primary = FakeSynthesizer(audio=np.full(2400, 0.1, dtype=np.float32))
    notifier = _BlockingNotifier("Reading...")
    player = FakeAudioPlayer()
    feature = QuickReaderFeature(
        model_manager=ModelManager(),
        primary_engine_factory=lambda: primary,
        fallback_engine=FakeSynthesizer(audio=np.zeros(1, dtype=np.float32)),
        clipboard_reader=FakeClipboardReader("Some text."),  # type: ignore[arg-type]
        player=player,
        notifier=notifier,  # type: ignore[arg-type]
    )

    feature.on_trigger()
    assert notifier.entered.wait(timeout=2.0)
    feature.on_trigger()  # stop while "Reading..." is being announced
    notifier.gate.set()
    wait_until(lambda: any(msg == "Stopped" for _, msg in notifier.messages))
    time.sleep(0.3)

    assert feature._speaker is None  # noqa: SLF001
    assert primary.calls == []
    assert player.output.opened == 0
    assert not any(msg == "Done" for _, msg in notifier.messages)


def test_each_reading_ends_exactly_once_even_when_stopped() -> None:
    """After a stop, the reading worker finishes quietly: one False report, one 'Stopped', no 'Done'."""
    changes: "list[bool]" = []
    threads_before = set(threading.enumerate())
    feature, player, notifier = _quiet_feature(changes)

    feature.on_trigger()
    wait_until(lambda: player.output.opened == 1)
    feature.on_trigger()
    for thread in set(threading.enumerate()) - threads_before:
        thread.join(timeout=5.0)  # bounded: a worker that never ends fails the checks below

    assert changes == [True, False]
    assert [msg for _, msg in notifier.messages].count("Stopped") == 1
    assert not any(msg == "Done" for _, msg in notifier.messages)
    assert feature._speaker is None  # noqa: SLF001
    assert player.output.closed == 1


def test_a_stopped_reading_never_gets_a_closing_done_notice() -> None:
    """A reading ended by a stop is not ended again by its worker, so no "Done" follows."""
    primary = FakeSynthesizer(audio=np.zeros(1, dtype=np.float32))
    fallback = FakeSynthesizer(audio=np.zeros(1, dtype=np.float32))
    feature, _, _, _, notifier, _ = make_feature(primary, fallback, clipboard_text="hello")
    feature._active = True  # noqa: SLF001
    feature._generation = 1  # noqa: SLF001
    feature._active = False  # noqa: SLF001  # a stop already ended generation 1

    feature._finish_reading(1, SpeakOutcome.DONE, threading.Thread())  # noqa: SLF001

    assert notifier.messages == []


def test_deactivate_reports_true_only_for_the_call_that_ended_the_reading() -> None:
    """The first _deactivate ends the reading; a second one (or a stale one) does not."""
    primary = FakeSynthesizer(audio=np.zeros(1, dtype=np.float32))
    fallback = FakeSynthesizer(audio=np.zeros(1, dtype=np.float32))
    feature, _, _, _, _, _ = make_feature(primary, fallback, clipboard_text="hello")
    feature._active = True  # noqa: SLF001
    feature._generation = 2  # noqa: SLF001

    assert feature._deactivate(1) is False  # noqa: SLF001  # stale generation
    assert feature._deactivate(2) is True  # noqa: SLF001
    assert feature._deactivate(2) is False  # noqa: SLF001  # already inactive


def test_a_finished_reading_still_announces_done_once() -> None:
    """A reading that ends on its own is announced exactly once."""
    primary = FakeSynthesizer(audio=np.zeros(1, dtype=np.float32))
    fallback = FakeSynthesizer(audio=np.zeros(1, dtype=np.float32))
    feature, _, _, _, notifier, _ = make_feature(primary, fallback, clipboard_text="hello")
    feature._active = True  # noqa: SLF001
    feature._generation = 1  # noqa: SLF001
    notice = threading.Thread(target=lambda: None)
    notice.start()

    feature._finish_reading(1, SpeakOutcome.DONE, notice)  # noqa: SLF001

    assert [msg for _, msg in notifier.messages] == ["Done"]
