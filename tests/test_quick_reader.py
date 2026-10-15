"""Tests for vocalis.features.quick_reader.QuickReaderFeature.

Uses fakes for every collaborator — this suite verifies orchestration
logic only, the same approach test_dictation.py already established.
"""
import logging
import subprocess
import threading
import time
from typing import Callable, cast

import numpy as np
import numpy.typing as npt
import pytest

from vocalis.audio.playback import AudioPlayer
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


class FakePlayer:
    def __init__(self, delay_seconds: float = 0.0, raise_on_play: Exception | None = None) -> None:
        self.delay_seconds = delay_seconds
        self.raise_on_play = raise_on_play
        self.play_calls: list[tuple[object, int]] = []
        self.stop_calls = 0
        self._stopped = threading.Event()

    def play(self, audio, sample_rate: int) -> None:  # type: ignore[no-untyped-def]
        self.play_calls.append((audio, sample_rate))
        if self.raise_on_play is not None:
            raise self.raise_on_play

    def wait_until_done(self) -> bool:
        if self._stopped.wait(timeout=self.delay_seconds or 0.05):
            return False
        return True

    def stop(self) -> None:
        self.stop_calls += 1
        self._stopped.set()


class FakeNotifier:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def notify(self, title: str, message: str = "") -> None:
        self.messages.append((title, message))


def make_feature(
    primary: FakeSynthesizer,
    fallback: FakeSynthesizer,
    clipboard_text: str,
    player: FakePlayer | None = None,
    reading_coordinator: "ReadingCoordinator | None" = None,
    on_reading_changed: "Callable[[bool], None] | None" = None,
    speed_provider: "Callable[[], float] | None" = None,
) -> "tuple[QuickReaderFeature, FakeSynthesizer, FakeSynthesizer, FakePlayer, FakeNotifier, ModelManager]":
    manager = ModelManager()
    notifier = FakeNotifier()
    player = player or FakePlayer()
    feature = QuickReaderFeature(
        model_manager=manager,
        primary_engine_factory=lambda: primary,
        fallback_engine=fallback,
        clipboard_reader=FakeClipboardReader(clipboard_text),  # type: ignore[arg-type]
        player=player,  # type: ignore[arg-type]
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


def wait_until(predicate, timeout: float = 1.0) -> None:  # type: ignore[no-untyped-def]
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

    assert player.play_calls == []
    assert primary.calls == []


def test_trigger_reads_speaks_and_notifies_done() -> None:
    audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    primary = FakeSynthesizer(audio=audio, sample_rate=24000)
    fallback = FakeSynthesizer(audio=np.zeros(1, dtype=np.float32))
    feature, _, _, player, notifier, _ = make_feature(primary, fallback, clipboard_text="hello there")

    feature.on_trigger()
    wait_until(lambda: any(msg == "Done" for _, msg in notifier.messages))

    assert primary.calls == ["hello there"]
    assert player.play_calls == [(audio, 24000)]
    assert any(msg == "Reading..." for _, msg in notifier.messages)
    assert any(msg == "hello there" for _, msg in notifier.messages)


def test_primary_failure_falls_back_to_secondary_engine() -> None:
    primary = FakeSynthesizer(audio=np.zeros(1, dtype=np.float32), raise_exc=RuntimeError("cuda oom"))
    fallback_audio = np.array([0.4, 0.5], dtype=np.float32)
    fallback = FakeSynthesizer(audio=fallback_audio, sample_rate=22050)
    feature, _, _, player, notifier, _ = make_feature(primary, fallback, clipboard_text="fallback please")

    feature.on_trigger()
    wait_until(lambda: any(msg == "Done" for _, msg in notifier.messages))

    assert fallback.calls == ["fallback please"]
    assert player.play_calls == [(fallback_audio, 22050)]
    assert any("backup voice" in msg.lower() for _, msg in notifier.messages)


def test_both_engines_failing_notifies_failure_and_does_not_crash() -> None:
    primary = FakeSynthesizer(audio=np.zeros(1, dtype=np.float32), raise_exc=RuntimeError("cuda oom"))
    fallback = FakeSynthesizer(audio=np.zeros(1, dtype=np.float32), raise_exc=RuntimeError("piper missing"))
    feature, _, _, player, notifier, _ = make_feature(primary, fallback, clipboard_text="uh oh")

    feature.on_trigger()  # must not raise
    wait_until(lambda: any("Could not generate speech" in msg for _, msg in notifier.messages))

    assert player.play_calls == []


def test_second_trigger_while_playing_stops_and_notifies_stopped() -> None:
    audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    primary = FakeSynthesizer(audio=audio)
    fallback = FakeSynthesizer(audio=np.zeros(1, dtype=np.float32))
    player = FakePlayer(delay_seconds=2.0)  # wait_until_done() would block a long time unless stopped
    feature, _, _, _, notifier, _ = make_feature(primary, fallback, clipboard_text="a long passage", player=player)

    feature.on_trigger()
    wait_until(lambda: len(player.play_calls) == 1)
    feature.on_trigger()  # the interrupting press

    wait_until(lambda: any(msg == "Stopped" for _, msg in notifier.messages))
    assert player.stop_calls == 1
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
    assert player.play_calls == []  # playback must never start after being cancelled


class FakeStream:
    """Minimal sounddevice.OutputStream-alike for the real AudioPlayer."""

    def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True


class TriggeringNotifier:
    """A notifier whose notify() re-enters feature.on_trigger() the
    moment it sees the live-text notification -- simulating a real
    Ctrl+Alt+R press landing in the narrow window between the
    post-synthesis liveness check and the actual play() call."""

    def __init__(self, trigger_on_message: str) -> None:
        self._trigger_on_message = trigger_on_message
        self.messages: list[tuple[str, str]] = []
        self.feature: QuickReaderFeature | None = None

    def notify(self, title: str, message: str = "") -> None:
        self.messages.append((title, message))
        if message == self._trigger_on_message and self.feature is not None:
            self.feature.on_trigger()


def test_stop_landing_during_live_text_notification_stops_actual_playback() -> None:
    """Regression test: a press landing during the notifier.notify(text)
    call (after the last liveness check but before play() starts) must
    not be silently dropped. Uses a real AudioPlayer (with a fake stream)
    so the actual play()/stop() interaction is exercised, not a fake that
    could paper over the race.
    """
    audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    primary = FakeSynthesizer(audio=audio, sample_rate=22050)
    fallback = FakeSynthesizer(audio=np.zeros(1, dtype=np.float32))

    streams: list[FakeStream] = []

    def factory(**kwargs):  # type: ignore[no-untyped-def]
        stream = FakeStream(**kwargs)
        streams.append(stream)
        return stream

    player = AudioPlayer(stream_factory=factory)
    notifier = TriggeringNotifier(trigger_on_message="hello there")
    manager = ModelManager()
    feature = QuickReaderFeature(
        model_manager=manager,
        primary_engine_factory=lambda: primary,
        fallback_engine=fallback,
        clipboard_reader=FakeClipboardReader("hello there"),  # type: ignore[arg-type]
        player=player,
        notifier=notifier,  # type: ignore[arg-type]
    )
    notifier.feature = feature

    feature.on_trigger()
    wait_until(lambda: any(msg == "Stopped" for _, msg in notifier.messages))
    wait_until(lambda: len(streams) == 1)

    assert streams[0].stopped is True
    assert streams[0].closed is True
    assert player.is_playing is False
    assert not any(msg == "Done" for _, msg in notifier.messages)


def test_third_trigger_after_stop_starts_a_fresh_reading() -> None:
    audio = np.array([0.1], dtype=np.float32)
    primary = FakeSynthesizer(audio=audio)
    fallback = FakeSynthesizer(audio=np.zeros(1, dtype=np.float32))
    feature, _, _, player, notifier, _ = make_feature(primary, fallback, clipboard_text="again")

    feature.on_trigger()
    wait_until(lambda: len(player.play_calls) == 1)
    feature.on_trigger()  # stop
    wait_until(lambda: player.stop_calls == 1)
    feature.on_trigger()  # fresh start
    wait_until(lambda: len(player.play_calls) == 2)


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
    primary = FakeSynthesizer(audio=np.zeros(1, dtype=np.float32))
    fallback = FakeSynthesizer(audio=np.zeros(1, dtype=np.float32))
    player = FakePlayer(delay_seconds=2.0)  # wait_until_done() would block a long time unless stopped
    feature, _, _, _, notifier, _ = make_feature(
        primary,
        fallback,
        clipboard_text="a long passage",
        player=player,
        reading_coordinator=coordinator,
    )

    feature.on_trigger()
    wait_until(lambda: len(player.play_calls) == 1)
    coordinator.start_reading(object(), lambda: None)  # a different owner preempts it

    wait_until(lambda: any(msg == "Stopped" for _, msg in notifier.messages))
    assert player.stop_calls == 1


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
    audio = np.array([0.1], dtype=np.float32)
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
    wait_until(lambda: len(player.play_calls) == 1)
    assert player.play_calls == [(audio, 22050)]


class ResettablePlayer(FakePlayer):
    """A player whose stop() only affects the playback in progress (play()
    re-arms it), so a second reading really keeps playing after a stop."""

    def play(self, audio, sample_rate: int) -> None:  # type: ignore[no-untyped-def]
        self._stopped.clear()
        super().play(audio, sample_rate)


def _quiet_feature(
    changes: "list[bool]", player: FakePlayer | None = None
) -> "tuple[QuickReaderFeature, FakePlayer, FakeNotifier]":
    primary = FakeSynthesizer(audio=np.zeros(3, dtype=np.float32))
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
    wait_until(lambda: len(player.play_calls) == 1)
    feature.on_trigger()  # second press: False is reported before returning
    assert changes == [True, False]
    wait_until(lambda: player.stop_calls == 1)
    assert changes == [True, False]


def test_press_stop_press_quickly_ends_with_active_true() -> None:
    changes: list[bool] = []
    feature, player, notifier = _quiet_feature(changes)

    feature.on_trigger()
    wait_until(lambda: len(player.play_calls) == 1)
    feature.on_trigger()
    wait_until(lambda: any(msg == "Stopped" for _, msg in notifier.messages))
    feature.on_trigger()
    wait_until(lambda: len(player.play_calls) == 2)

    assert changes == [True, False, True]
    feature.stop()  # cleanup so the worker thread ends


def test_public_stop_ends_an_active_reading_and_is_a_noop_when_idle() -> None:
    changes: list[bool] = []
    feature, player, _ = _quiet_feature(changes)

    feature.stop()
    assert changes == []
    assert player.stop_calls == 0

    feature.on_trigger()
    wait_until(lambda: len(player.play_calls) == 1)
    feature.stop()
    assert changes == [True, False]
    wait_until(lambda: player.stop_calls == 1)


def test_natural_end_reports_false_once() -> None:
    changes: list[bool] = []
    feature, _player, notifier = _quiet_feature(changes, player=FakePlayer())

    feature.on_trigger()
    wait_until(lambda: any(msg == "Done" for _, msg in notifier.messages))

    assert changes == [True, False]


def test_raising_reading_hook_does_not_break_reading() -> None:
    def broken(active: bool) -> None:
        raise RuntimeError("ui bug")

    primary = FakeSynthesizer(audio=np.zeros(3, dtype=np.float32))
    fallback = FakeSynthesizer(audio=np.zeros(1, dtype=np.float32))
    feature, _, _, player, notifier, _ = make_feature(
        primary, fallback, "some text", player=FakePlayer(), on_reading_changed=broken
    )

    feature.on_trigger()
    wait_until(lambda: any(msg == "Done" for _, msg in notifier.messages))

    assert len(player.play_calls) == 1


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


def test_no_speed_provider_leaves_audio_unchanged() -> None:
    """Without a speed_provider, the exact same audio object reaches the player."""
    audio = sine()
    primary = FakeSynthesizer(audio=audio, sample_rate=24000)
    fallback = FakeSynthesizer(audio=np.zeros(1, dtype=np.float32))
    feature, _, _, player, notifier, _ = make_feature(primary, fallback, clipboard_text="pace me")

    feature.on_trigger()
    wait_until(lambda: any(msg == "Done" for _, msg in notifier.messages))

    assert len(player.play_calls) == 1
    played_audio, played_rate = player.play_calls[0]
    assert played_audio is audio
    assert played_rate == 24000


def test_speed_provider_returning_one_leaves_audio_unchanged() -> None:
    """A neutral speed (1.0) also hands back the same object."""
    audio = sine()
    primary = FakeSynthesizer(audio=audio, sample_rate=24000)
    fallback = FakeSynthesizer(audio=np.zeros(1, dtype=np.float32))
    feature, _, _, player, notifier, _ = make_feature(
        primary, fallback, clipboard_text="pace me", speed_provider=lambda: 1.0
    )

    feature.on_trigger()
    wait_until(lambda: any(msg == "Done" for _, msg in notifier.messages))

    assert len(player.play_calls) == 1
    played_audio, _ = player.play_calls[0]
    assert played_audio is audio


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

    assert len(player.play_calls) == 1
    played_audio, played_rate = player.play_calls[0]
    assert played_rate == 24000
    assert abs(len(cast(npt.NDArray[np.float32], played_audio)) - len(audio) / 1.5) < 0.03 * len(audio)


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

    assert len(player.play_calls) == 1
    played_audio, _ = player.play_calls[0]
    assert played_audio is audio
    assert any(record.levelno == logging.WARNING for record in caplog.records)
