"""Tests for vocalis.features.doc_reader.DocumentReaderFeature.

Uses fakes for every collaborator, the same approach test_quick_reader.py
already established, plus a real temporary file for read_file() so
extraction genuinely runs (extractors.py itself is unit-tested
separately in test_extractors.py).
"""
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import numpy.typing as npt
import pytest

from vocalis.audio.playback import AudioPlayer
from vocalis.features.doc_reader import DocumentReaderFeature
from vocalis.features.reading_coordinator import ReadingCoordinator
from vocalis.model_manager import TTS_ENGINE_ID, ModelManager
from vocalis.text.extractors import (
    DocumentOpenError,
    DocumentTooLargeError,
    NoReadableTextError,
)


class FakeSynthesizer:
    def __init__(
        self,
        sample_rate: int = 22050,
        raise_exc: Exception | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.raise_exc = raise_exc
        self.calls: "list[str]" = []
        self._loaded = False

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def synthesize(self, text: str) -> "tuple[npt.NDArray[np.float32], int]":
        self.calls.append(text)
        if self.raise_exc is not None:
            raise self.raise_exc
        return np.zeros(100, dtype=np.float32), self.sample_rate


class GatedSynthesizer:
    """A synthesizer whose synthesize() blocks until release() is called,
    so a test can pause() while synthesis of a chunk is genuinely still in
    flight -- deterministic, unlike a fixed sleep() that merely hopes to
    land inside the right window."""

    def __init__(self, sample_rate: int = 22050) -> None:
        self.sample_rate = sample_rate
        self.calls: "list[str]" = []
        self._entered = threading.Event()
        self._release = threading.Event()

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    @property
    def is_loaded(self) -> bool:
        return True

    def wait_until_entered(self, timeout: float = 5.0) -> None:
        """Block the calling (test) thread until synthesize() has been
        entered by the worker thread."""
        assert self._entered.wait(timeout=timeout), "synthesize() was never entered"

    def release(self) -> None:
        """Let a blocked synthesize() call return."""
        self._release.set()

    def synthesize(self, text: str) -> "tuple[npt.NDArray[np.float32], int]":
        self.calls.append(text)
        self._entered.set()
        assert self._release.wait(timeout=5.0), "release() was never called"
        return np.zeros(100, dtype=np.float32), self.sample_rate


class FakeStream:
    def __init__(self, **kwargs: Any) -> None:
        self._finished_callback = kwargs["finished_callback"]

    def start(self) -> None:
        threading.Thread(target=self._finish_soon, daemon=True).start()

    def _finish_soon(self) -> None:
        time.sleep(0.02)
        self._finished_callback()

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


def _fake_stream_factory(**kwargs: Any) -> FakeStream:
    return FakeStream(**kwargs)


class StreamStartRecorder:
    """Records when (and how many times) a stream's start() actually ran,
    so tests can assert "playback has/hasn't started" from real signal
    instead of guessing from a fixed sleep duration."""

    def __init__(self) -> None:
        self.count = 0
        self.started = threading.Event()

    def record(self) -> None:
        self.count += 1
        self.started.set()


def _recording_stream_factory(recorder: StreamStartRecorder) -> "Callable[..., FakeStream]":
    """Build a stream_factory whose streams call `recorder.record()` the
    moment start() is invoked, i.e. the moment AudioPlayer actually begins
    playback -- not merely the moment play() was called."""

    class RecordingFakeStream(FakeStream):
        def start(self) -> None:
            recorder.record()
            super().start()

    def factory(**kwargs: Any) -> RecordingFakeStream:
        return RecordingFakeStream(**kwargs)

    return factory


def _endless_stream_factory(recorder: StreamStartRecorder) -> "Callable[..., Any]":
    """Build a stream_factory whose streams NEVER finish by themselves:
    playback only ends when AudioPlayer.stop() is called. A test using it
    is guaranteed the reading is still in progress when it acts, so a stop
    that silently did nothing would leave the reading unfinished and the
    test would fail instead of passing vacuously."""

    class EndlessFakeStream(FakeStream):
        def start(self) -> None:
            recorder.record()

    def factory(**kwargs: Any) -> EndlessFakeStream:
        return EndlessFakeStream(**kwargs)

    return factory


class _SignalingEvent(threading.Event):
    """An Event that reports when some thread starts waiting on it."""

    def __init__(self) -> None:
        super().__init__()
        self.waiting = threading.Event()

    def wait(self, timeout: "float | None" = None) -> bool:
        self.waiting.set()
        return super().wait(timeout)


class RecordingNotifier:
    def __init__(self) -> None:
        self.messages: "list[str]" = []

    def notify(self, title: str, message: str = "") -> None:
        self.messages.append(message)


def wait_until(predicate: "Callable[[], bool]", timeout: float = 5.0) -> None:
    """Block until `predicate()` is true; FAIL the test if `timeout`
    seconds pass first (never silently falls through)."""
    deadline = time.monotonic() + timeout
    pause = threading.Event()
    while not predicate():
        if time.monotonic() >= deadline:
            pytest.fail(f"condition not met within {timeout} s")
        pause.wait(0.005)


def _make_feature(
    fallback: FakeSynthesizer,
    primary: "FakeSynthesizer | GatedSynthesizer",
    notifier: RecordingNotifier,
    coordinator: ReadingCoordinator | None = None,
    stream_factory: "Callable[..., Any]" = _fake_stream_factory,
    on_started: "Callable[[], None] | None" = None,
    speed_provider: "Callable[[], float] | None" = None,
) -> "tuple[DocumentReaderFeature, list[str], list[tuple[int, int]], list[None]]":
    statuses: "list[str]" = []
    progresses: "list[tuple[int, int]]" = []
    finishes: "list[None]" = []
    feature = DocumentReaderFeature(
        model_manager=ModelManager(),
        primary_engine_factory=lambda: primary,
        fallback_engine=fallback,
        player=AudioPlayer(stream_factory=stream_factory),
        notifier=notifier,  # type: ignore[arg-type]
        reading_coordinator=coordinator or ReadingCoordinator(),
        on_status=statuses.append,
        on_progress=lambda current, total: progresses.append((current, total)),
        on_finished=lambda: finishes.append(None),
        on_started=on_started,
        speed_provider=speed_provider,
    )
    return feature, statuses, progresses, finishes


class SineSynthesizer:
    """Returns a real ~2 s 24 kHz sine every call -- long enough for an
    actual time-stretch to measurably change its length (unlike
    FakeSynthesizer's 100-sample zeros)."""

    def __init__(self, sample_rate: int = 24000) -> None:
        self.sample_rate = sample_rate
        self.calls: "list[str]" = []

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    @property
    def is_loaded(self) -> bool:
        return True

    def synthesize(self, text: str) -> "tuple[npt.NDArray[np.float32], int]":
        self.calls.append(text)
        t = np.arange(int(self.sample_rate * 2.0)) / self.sample_rate
        audio = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        return audio, self.sample_rate


class LengthRecordingPlayer:
    """A minimal player that records the length of every played chunk
    and finishes each one naturally right away."""

    def __init__(self) -> None:
        self.play_lengths: "list[int]" = []

    def play(self, audio: Any, sample_rate: int) -> None:
        self.play_lengths.append(len(audio))

    def wait_until_done(self) -> bool:
        return True

    def stop(self) -> None:
        pass


def test_read_text_speaks_every_chunk_and_reports_done() -> None:
    primary = FakeSynthesizer()
    feature, statuses, progresses, finishes = _make_feature(
        fallback=FakeSynthesizer(), primary=primary, notifier=RecordingNotifier()
    )

    feature.read_text("First sentence. Second sentence.")
    wait_until(lambda: bool(finishes))

    assert len(primary.calls) == 1  # short text fits in one chunk
    assert progresses[0] == (1, 1)
    assert "Done" in statuses
    assert finishes == [None]
    assert feature.is_reading is False


def test_read_file_extracts_and_speaks_a_real_file(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("Some file content.", encoding="utf-8")
    primary = FakeSynthesizer()
    feature, _statuses, _progresses, finishes = _make_feature(
        fallback=FakeSynthesizer(), primary=primary, notifier=RecordingNotifier()
    )

    feature.read_file(path)
    wait_until(lambda: bool(finishes))

    assert primary.calls == ["Some file content."]
    assert finishes == [None]
    assert feature.is_reading is False


def test_unsupported_file_reports_status_and_finishes(tmp_path: Path) -> None:
    path = tmp_path / "sample.rtf"
    path.write_text("nope", encoding="utf-8")
    feature, statuses, _progresses, finishes = _make_feature(
        fallback=FakeSynthesizer(), primary=FakeSynthesizer(), notifier=RecordingNotifier()
    )

    feature.read_file(path)
    wait_until(lambda: bool(finishes))

    assert any("Unsupported file type" in status for status in statuses)
    assert feature.is_reading is False


def test_primary_failure_falls_back_to_piper_once_per_document() -> None:
    primary = FakeSynthesizer(raise_exc=RuntimeError("gpu unavailable"))
    fallback = FakeSynthesizer()
    notifier = RecordingNotifier()
    feature, _statuses, _progresses, finishes = _make_feature(
        fallback=fallback, primary=primary, notifier=notifier
    )

    feature.read_text("One. Two. Three. Four. Five. Six. Seven. Eight.")
    wait_until(lambda: bool(finishes))

    assert fallback.calls  # every chunk after the first failure used Piper
    assert notifier.messages.count("Primary voice unavailable — using backup voice") == 1


def test_pause_stops_audio_immediately_and_resume_replays_current_chunk() -> None:
    primary = FakeSynthesizer()
    recorder = StreamStartRecorder()
    feature, statuses, _progresses, finishes = _make_feature(
        fallback=FakeSynthesizer(),
        primary=primary,
        notifier=RecordingNotifier(),
        stream_factory=_recording_stream_factory(recorder),
    )

    feature.read_text("Only one short sentence here to read aloud slowly.")
    # Deterministic: wait for playback to actually start (rather than
    # guessing with a fixed sleep) before pausing, so this test reliably
    # exercises the "stop an already-playing chunk" path, not the
    # "paused during synthesis" path covered by
    # test_pause_during_synthesis_blocks_playback_until_resume below.
    assert recorder.started.wait(timeout=5.0)
    feature.pause()
    assert "Paused" in statuses
    assert feature.is_reading is True  # paused, not stopped

    feature.resume()
    wait_until(lambda: bool(finishes))
    assert "Done" in statuses


def test_pause_during_synthesis_blocks_playback_until_resume() -> None:
    """Regression guard: pausing while a chunk is still synthesizing must
    actually block before playback starts for that chunk, not just set a
    status that the worker thread then ignores. pause() has no stream to
    stop() yet at that point (play() hasn't been called), so the worker
    itself must wait on the resume event before ever calling play()."""
    primary = GatedSynthesizer()
    recorder = StreamStartRecorder()
    feature, statuses, _progresses, finishes = _make_feature(
        fallback=FakeSynthesizer(),
        primary=primary,
        notifier=RecordingNotifier(),
        stream_factory=_recording_stream_factory(recorder),
    )

    gate = _SignalingEvent()
    gate.set()
    feature._resume_event = gate  # noqa: SLF001 -- lets the test see the worker arrive

    feature.read_text("Only one short sentence here to read aloud slowly.")
    primary.wait_until_entered()  # synthesis of the first chunk is in flight
    feature.pause()
    assert "Paused" in statuses
    primary.release()  # let synthesis finish while still paused

    # Wait until the worker has reached the pause gate: if the bug were
    # present it would have called play() instead of arriving here.
    assert gate.waiting.wait(timeout=5.0)
    assert recorder.count == 0  # playback must not have started while paused
    assert finishes == []

    feature.resume()
    wait_until(lambda: bool(finishes))
    assert recorder.count == 1  # playback started only after resume()
    assert "Done" in statuses


def test_stop_while_paused_actually_stops() -> None:
    """Regression guard: stopping while paused must not leave the worker
    thread blocked forever waiting on a resume that will never come."""
    primary = FakeSynthesizer()
    recorder = StreamStartRecorder()
    feature, statuses, _progresses, finishes = _make_feature(
        fallback=FakeSynthesizer(),
        primary=primary,
        notifier=RecordingNotifier(),
        stream_factory=_endless_stream_factory(recorder),
    )

    gate = _CountingEvent()
    gate.set()
    feature._resume_event = gate  # noqa: SLF001 -- lets the test see the worker park

    feature.read_text("Only one short sentence here to read aloud slowly.")
    assert recorder.started.wait(timeout=5.0)  # playback has really begun
    waits_before = gate.wait_calls
    feature.pause()
    wait_until(lambda: gate.wait_calls > waits_before)  # worker is parked, waiting for resume
    feature.stop()

    wait_until(lambda: bool(finishes))
    assert finishes == [None]
    assert feature.is_reading is False
    _join_workers()  # a worker stuck waiting for a resume would fail here
    assert "Stopped" in statuses
    assert "Done" not in statuses


def test_a_second_reader_preempts_this_one_via_the_coordinator() -> None:
    coordinator = ReadingCoordinator()
    primary = FakeSynthesizer()
    recorder = StreamStartRecorder()
    feature, statuses, _progresses, finishes = _make_feature(
        fallback=FakeSynthesizer(),
        primary=primary,
        notifier=RecordingNotifier(),
        coordinator=coordinator,
        stream_factory=_endless_stream_factory(recorder),
    )

    feature.read_text("Only one short sentence here to read aloud slowly.")
    assert recorder.started.wait(timeout=5.0)  # playback has really begun
    coordinator.start_reading(object(), lambda: None)  # a different owner preempts it

    wait_until(lambda: bool(finishes))
    assert finishes == [None]
    assert feature.is_reading is False
    _join_workers()  # a worker stuck waiting for a resume would fail here
    assert "Stopped" in statuses
    assert "Done" not in statuses


def test_empty_pasted_text_reports_nothing_to_read() -> None:
    feature, statuses, _progresses, finishes = _make_feature(
        fallback=FakeSynthesizer(), primary=FakeSynthesizer(), notifier=RecordingNotifier()
    )

    feature.read_text("   ")
    wait_until(lambda: bool(finishes))

    assert "Nothing to read" in statuses


class BlockingNotifier:
    """A notifier whose control-message notify() calls (Paused / Resuming /
    Stopped) block until release() is called, like a slow notify-send."""

    _CONTROL = {"Paused", "Resuming...", "Stopped"}

    def __init__(self) -> None:
        self.messages: "list[str]" = []
        self._gate = threading.Event()
        self._lock = threading.Lock()
        self._delivered = threading.Event()

    def release(self) -> None:
        self._gate.set()

    def notify(self, title: str, message: str = "") -> None:
        if message in self._CONTROL:
            assert self._gate.wait(timeout=5.0), "release() was never called"
        with self._lock:
            self.messages.append(message)
            if {"Paused", "Resuming...", "Stopped"} <= set(self.messages):
                self._delivered.set()

    def wait_all_control_messages(self) -> bool:
        return self._delivered.wait(timeout=5.0)


def _returns_promptly(call: "Callable[[], None]") -> bool:
    """Run `call` on a helper thread; True if it returned within 2 s."""
    thread = threading.Thread(target=call, daemon=True)
    thread.start()
    thread.join(timeout=2.0)
    return not thread.is_alive()


def test_pause_resume_stop_do_not_block_on_a_slow_notifier() -> None:
    primary = GatedSynthesizer()
    notifier = BlockingNotifier()
    feature, statuses, _progresses, finishes = _make_feature(
        fallback=FakeSynthesizer(),
        primary=primary,
        notifier=notifier,  # type: ignore[arg-type]
    )

    feature.read_text("Only one short sentence here to read aloud slowly.")
    primary.wait_until_entered()  # reading is active, synthesis in flight

    assert _returns_promptly(feature.pause)
    assert _returns_promptly(feature.resume)
    assert _returns_promptly(feature.stop)
    assert statuses[-1] == "Stopped"
    assert finishes == [None]

    notifier.release()
    primary.release()
    assert notifier.wait_all_control_messages()
    assert {"Paused", "Resuming...", "Stopped"} <= set(notifier.messages)



class FirstMessageBlockedNotifier:
    """A notifier whose FIRST control message ("Paused") blocks until
    release() is called; every later message is delivered at once. A
    delivery design that does not preserve order would therefore deliver
    "Resuming..." and "Stopped" while "Paused" is still stuck."""

    def __init__(self) -> None:
        self.messages: "list[str]" = []
        self.first_entered = threading.Event()
        self.all_delivered = threading.Event()
        self._gate = threading.Event()
        self._lock = threading.Lock()

    def release(self) -> None:
        self._gate.set()

    def notify(self, title: str, message: str = "") -> None:
        if message == "Paused":
            self.first_entered.set()
            assert self._gate.wait(timeout=5.0), "release() was never called"
        with self._lock:
            self.messages.append(message)
            if message == "Stopped":
                self.all_delivered.set()


def test_control_notifications_are_delivered_in_call_order() -> None:
    """With only the first notify() ("Paused") stuck, pause/resume/stop
    still return promptly and, once it is released, the notifier sees
    Paused, Resuming..., Stopped in exactly the order they were called."""
    primary = GatedSynthesizer()
    notifier = FirstMessageBlockedNotifier()
    feature, _statuses, _progresses, _finishes = _make_feature(
        fallback=FakeSynthesizer(),
        primary=primary,
        notifier=notifier,  # type: ignore[arg-type]
    )

    feature.read_text("Only one short sentence here to read aloud slowly.")
    primary.wait_until_entered()
    assert _returns_promptly(feature.pause)
    assert notifier.first_entered.wait(timeout=5.0)  # "Paused" is now stuck inside notify()
    assert _returns_promptly(feature.resume)
    assert _returns_promptly(feature.stop)
    # A reordering design would already have delivered the later messages.
    with notifier._lock:  # noqa: SLF001
        early = [m for m in notifier.messages if m in BlockingNotifier._CONTROL]
    assert early == []

    notifier.release()
    primary.release()
    assert notifier.all_delivered.wait(timeout=5.0)
    control = [m for m in notifier.messages if m in BlockingNotifier._CONTROL]
    assert control == ["Paused", "Resuming...", "Stopped"]


class FailingOnceNotifier:
    """Raises on its first control message, then records the rest."""

    def __init__(self) -> None:
        self.messages: "list[str]" = []
        self._failed = False
        self.got_stopped = threading.Event()

    def notify(self, title: str, message: str = "") -> None:
        if message == "Paused" and not self._failed:
            self._failed = True
            raise RuntimeError("notify-send missing")
        self.messages.append(message)
        if message == "Stopped":
            self.got_stopped.set()


def test_a_notifier_that_raises_is_survived_and_later_messages_still_arrive() -> None:
    primary = GatedSynthesizer()
    notifier = FailingOnceNotifier()
    feature, _statuses, _progresses, _finishes = _make_feature(
        fallback=FakeSynthesizer(),
        primary=primary,
        notifier=notifier,  # type: ignore[arg-type]
    )

    feature.read_text("Only one short sentence here to read aloud slowly.")
    primary.wait_until_entered()
    feature.pause()
    feature.resume()
    feature.stop()
    primary.release()

    assert notifier.got_stopped.wait(timeout=5.0)
    # "Paused" raised and was dropped; the consumer survived and delivered the rest in order.
    assert "Paused" not in notifier.messages
    assert [m for m in notifier.messages if m in BlockingNotifier._CONTROL] == [
        "Resuming...",
        "Stopped",
    ]


class RacingPlayer:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.stop_calls = 0
        self.play_calls = 0

    def play(self, audio: Any, sample_rate: int) -> None:
        self.play_calls += 1
        self.entered.set()
        assert self.release.wait(timeout=5.0), "release was never set"

    def wait_until_done(self) -> bool:
        return True

    def stop(self) -> None:
        self.stop_calls += 1


def test_stop_landing_just_before_play_starts_still_stops_that_chunk() -> None:
    finishes: "list[None]" = []
    player = RacingPlayer()
    feature = DocumentReaderFeature(
        model_manager=ModelManager(),
        primary_engine_factory=lambda: FakeSynthesizer(),
        fallback_engine=FakeSynthesizer(),
        player=player,  # type: ignore[arg-type]
        notifier=RecordingNotifier(),  # type: ignore[arg-type]
        reading_coordinator=ReadingCoordinator(),
        on_status=lambda message: None,
        on_progress=lambda current, total: None,
        on_finished=lambda: finishes.append(None),
    )

    feature.read_text("Only one short sentence here to read aloud slowly.")
    assert player.entered.wait(timeout=5.0)  # worker is inside play()
    feature.stop()  # player.stop() here is a no-op in real life: no stream yet
    stops_before_release = player.stop_calls
    player.release.set()  # play() now returns
    wait_until(lambda: player.stop_calls > stops_before_release)

    assert player.stop_calls == stops_before_release + 1  # worker re-stopped it
    assert finishes == [None]


class _CountingEvent(threading.Event):
    """An Event that counts how many times wait() has been entered."""

    def __init__(self) -> None:
        super().__init__()
        self.wait_calls = 0

    def wait(self, timeout: "float | None" = None) -> bool:
        self.wait_calls += 1
        return super().wait(timeout)


class WindowPlayer:
    """A player that behaves like the real AudioPlayer in the one respect
    that matters here: stop() only has an effect once play() has returned
    (before that there is no stream to stop). The first play() blocks until
    release is set, so a test can call pause()/stop() in exactly the window
    between "play() was called" and "a stream exists"."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.play_calls = 0
        self.effective_stops = 0
        self.returned = threading.Event()
        self._streaming = False
        self._stopped = False

    def play(self, audio: Any, sample_rate: int) -> None:
        self.play_calls += 1
        self._stopped = False
        if self.play_calls == 1:
            self.entered.set()
            assert self.release.wait(timeout=5.0), "release was never set"
        self._streaming = True

    def wait_until_done(self) -> bool:
        self._streaming = False
        self.returned.set()
        return not self._stopped

    def stop(self) -> None:
        if self._streaming:
            self._stopped = True
            self.effective_stops += 1


def _join_workers() -> None:
    """Wait (bounded) for every reading worker thread to finish, so a test
    can assert that nothing more fires after the reading ended."""
    for thread in threading.enumerate():
        if "_prepare_and_read" in thread.name:
            thread.join(timeout=5.0)
            assert not thread.is_alive(), "worker thread did not finish"


def _window_feature(
    player: WindowPlayer,
) -> "tuple[DocumentReaderFeature, list[str], list[None]]":
    """Build a feature around `player`; returns it with its status and
    on_finished records."""
    statuses: "list[str]" = []
    finishes: "list[None]" = []
    feature = DocumentReaderFeature(
        model_manager=ModelManager(),
        primary_engine_factory=lambda: FakeSynthesizer(),
        fallback_engine=FakeSynthesizer(),
        player=player,  # type: ignore[arg-type]
        notifier=RecordingNotifier(),  # type: ignore[arg-type]
        reading_coordinator=ReadingCoordinator(),
        on_status=statuses.append,
        on_progress=lambda current, total: None,
        on_finished=lambda: finishes.append(None),
    )
    return feature, statuses, finishes


def test_pause_landing_just_before_play_starts_still_pauses_that_chunk() -> None:
    player = WindowPlayer()
    feature, statuses, finishes = _window_feature(player)
    gate = _CountingEvent()
    gate.set()
    feature._resume_event = gate  # noqa: SLF001 -- lets the test see the worker arrive

    feature.read_text("Only one short sentence here to read aloud slowly.")
    assert player.entered.wait(timeout=5.0)  # worker is inside play(), no stream yet
    feature.pause()  # player.stop() is a no-op: there is no stream to stop
    assert player.effective_stops == 0
    waits_before = gate.wait_calls
    player.release.set()  # play() now returns

    wait_until(lambda: player.effective_stops == 1)  # worker stopped the chunk itself
    wait_until(lambda: gate.wait_calls > waits_before)  # and is parked, waiting for resume
    assert statuses[-1] == "Paused"
    assert player.play_calls == 1
    assert finishes == []

    feature.resume()
    wait_until(lambda: bool(finishes))
    _join_workers()
    assert player.play_calls == 2  # the same chunk was replayed
    assert statuses[-1] == "Done"
    assert finishes == [None]


def test_pause_then_stop_around_play_start_ends_the_reading_once() -> None:
    player = WindowPlayer()
    feature, statuses, finishes = _window_feature(player)

    feature.read_text("Only one short sentence here to read aloud slowly.")
    assert player.entered.wait(timeout=5.0)
    feature.pause()
    feature.stop()
    player.release.set()

    assert player.returned.wait(timeout=5.0)
    _join_workers()
    assert statuses[-1] == "Stopped"
    assert "Done" not in statuses
    assert finishes == [None]
    assert player.play_calls == 1
    assert feature.is_reading is False


@pytest.mark.parametrize(
    "error",
    [
        DocumentOpenError("Could not open that document"),
        NoReadableTextError("No readable text in that document"),
        DocumentTooLargeError("That document is too large to read"),
    ],
    ids=["open", "no-text", "too-large"],
)
def test_extraction_errors_reach_status_and_notification_and_finish_once(
    monkeypatch: pytest.MonkeyPatch, error: Exception, tmp_path: Path
) -> None:
    def raise_error(path: Path) -> str:
        raise error

    monkeypatch.setattr("vocalis.features.doc_reader.extract_text", raise_error)
    notifier = RecordingNotifier()
    feature, statuses, _progresses, finishes = _make_feature(
        fallback=FakeSynthesizer(), primary=FakeSynthesizer(), notifier=notifier
    )

    feature.read_file(tmp_path / "anything.txt")
    wait_until(lambda: bool(finishes))

    assert str(error) in statuses
    assert str(error) in notifier.messages
    assert finishes == [None]
    assert feature.is_reading is False


def test_both_engines_failing_reports_could_not_generate_speech() -> None:
    notifier = RecordingNotifier()
    feature, statuses, _progresses, finishes = _make_feature(
        fallback=FakeSynthesizer(raise_exc=RuntimeError("piper broke")),
        primary=FakeSynthesizer(raise_exc=RuntimeError("gpu broke")),
        notifier=notifier,
    )

    feature.read_text("Only one short sentence here to read aloud slowly.")
    wait_until(lambda: bool(finishes))

    assert "Could not generate speech" in statuses
    assert "Could not generate speech" in notifier.messages
    assert finishes == [None]
    assert feature.is_reading is False


def test_playback_raising_reports_playback_failed() -> None:
    def broken_stream_factory(**kwargs: Any) -> FakeStream:
        raise RuntimeError("audio device busy")

    notifier = RecordingNotifier()
    feature, statuses, _progresses, finishes = _make_feature(
        fallback=FakeSynthesizer(),
        primary=FakeSynthesizer(),
        notifier=notifier,
        stream_factory=broken_stream_factory,
    )

    feature.read_text("Only one short sentence here to read aloud slowly.")
    wait_until(lambda: bool(finishes))

    assert "Playback failed" in statuses
    assert "Playback failed" in notifier.messages
    assert finishes == [None]
    assert feature.is_reading is False


def test_after_one_failure_later_chunks_go_straight_to_piper() -> None:
    primary = FakeSynthesizer(raise_exc=RuntimeError("gpu unavailable"))
    fallback = FakeSynthesizer()
    feature, _statuses, progresses, finishes = _make_feature(
        fallback=fallback, primary=primary, notifier=RecordingNotifier()
    )

    feature.read_text(("A" * 200 + ". ") + ("B" * 200 + "."))  # two chunks
    wait_until(lambda: bool(finishes))

    assert [total for _, total in progresses] == [2, 2]
    assert len(primary.calls) == 1  # primary tried once, never again
    assert len(fallback.calls) == 2


def test_an_already_loaded_primary_engine_is_reused_without_fallback() -> None:
    primary = FakeSynthesizer()
    fallback = FakeSynthesizer()
    manager = ModelManager()
    manager.acquire(TTS_ENGINE_ID, lambda: primary)  # primary is now loaded
    manager.release(TTS_ENGINE_ID)
    factory_calls: "list[None]" = []

    def factory() -> FakeSynthesizer:
        factory_calls.append(None)
        return FakeSynthesizer()

    finishes: "list[None]" = []
    feature = DocumentReaderFeature(
        model_manager=manager,
        primary_engine_factory=factory,
        fallback_engine=fallback,
        player=AudioPlayer(stream_factory=_fake_stream_factory),
        notifier=RecordingNotifier(),  # type: ignore[arg-type]
        reading_coordinator=ReadingCoordinator(),
        on_status=lambda message: None,
        on_progress=lambda current, total: None,
        on_finished=lambda: finishes.append(None),
    )

    feature.read_text("Only one short sentence here to read aloud slowly.")
    wait_until(lambda: bool(finishes))

    assert primary.calls == ["Only one short sentence here to read aloud slowly."]
    assert factory_calls == []  # the loaded engine was reused, not rebuilt
    assert fallback.calls == []


def test_generation_increases_by_one_for_each_reading() -> None:
    feature, _statuses, _progresses, finishes = _make_feature(
        FakeSynthesizer(), FakeSynthesizer(), RecordingNotifier()
    )
    first = feature.generation

    feature.read_text("Hello there.")
    assert feature.generation == first + 1
    wait_until(lambda: len(finishes) == 1)

    feature.read_text("Hello again.")
    assert feature.generation == first + 2
    wait_until(lambda: len(finishes) == 2)


def test_ocr_reading_replaces_a_document_reading_in_progress() -> None:
    from vocalis.features.ocr_reader import OcrReaderFeature

    class Capture:
        def capture(self) -> bytes | None:
            return b"PNG"

    class Ocr:
        def recognize(self, png_bytes: bytes) -> str:
            return "Text read from the screen."

    class Clipboard:
        def write(self, text: str) -> bool:
            return True

    recorder = StreamStartRecorder()
    notifier = RecordingNotifier()
    reader, statuses, _progresses, finishes = _make_feature(
        fallback=FakeSynthesizer(),
        primary=FakeSynthesizer(),
        notifier=notifier,
        stream_factory=_endless_stream_factory(recorder),
    )
    ocr_feature = OcrReaderFeature(
        capture=Capture(), ocr=Ocr(), clipboard=Clipboard(), reader=reader,
        notifier=notifier,  # type: ignore[arg-type]
    )
    try:
        reader.read_text("First document text.")
        assert recorder.started.wait(5.0)
        recorder.started.clear()

        ocr_feature.on_trigger()
        wait_until(lambda: not ocr_feature.is_busy)
        assert recorder.started.wait(5.0), "the OCR text never started playing"

        assert recorder.count == 2
        assert not any("Playback failed" in m for m in notifier.messages)
        assert not any("Playback failed" in s for s in statuses)
        assert len(finishes) >= 1  # the old reading was stopped
        assert reader.is_reading is True
    finally:
        reader.stop()


def test_on_started_is_called_once_per_reading(tmp_path: Path) -> None:
    started: "list[None]" = []
    feature, _s, _p, finishes = _make_feature(
        FakeSynthesizer(), FakeSynthesizer(), RecordingNotifier(), on_started=lambda: started.append(None)
    )
    feature.read_text("Hello there.")
    wait_until(lambda: len(finishes) == 1)
    assert started == [None]

    path = tmp_path / "a.txt"
    path.write_text("File text.", encoding="utf-8")
    feature.read_file(path)
    wait_until(lambda: len(finishes) == 2)
    assert started == [None, None]


def test_raising_on_started_hook_does_not_break_reading() -> None:
    def broken() -> None:
        raise RuntimeError("ui bug")

    primary = FakeSynthesizer()
    feature, _s, _p, finishes = _make_feature(
        FakeSynthesizer(), primary, RecordingNotifier(), on_started=broken
    )
    feature.read_text("Hello there.")
    wait_until(lambda: len(finishes) == 1)
    assert primary.calls == ["Hello there."]


class _GatedDoneNotifier:
    """A notifier whose "Done" notification blocks until released, like a
    slow notify-send; records that it was reached."""

    def __init__(self) -> None:
        self.reached = threading.Event()
        self.gate = threading.Event()

    def notify(self, title: str, message: str = "") -> None:
        if message == "Done":
            self.reached.set()
            assert self.gate.wait(timeout=5.0), "gate was never released"


def test_reading_end_is_reported_before_the_done_notification_is_sent() -> None:
    notifier = _GatedDoneNotifier()
    feature, _statuses, _progresses, finishes = _make_feature(
        fallback=FakeSynthesizer(),
        primary=FakeSynthesizer(),
        notifier=notifier,  # type: ignore[arg-type]
    )
    try:
        feature.read_text("First sentence. Second sentence.")
        assert notifier.reached.wait(timeout=5.0)

        # The "Done" notification is still blocked: the end must already
        # have been reported, or a new reading could be wiped out by it.
        ended_while_notify_blocked = bool(finishes)
    finally:
        notifier.gate.set()
    assert ended_while_notify_blocked


class _LockSpyCoordinator(ReadingCoordinator):
    """A coordinator that records whether the feature's own lock was held
    each time finish_reading() was called."""

    def __init__(self) -> None:
        super().__init__()
        self.feature: "DocumentReaderFeature | None" = None
        self.lock_held_at_finish: "list[bool]" = []

    def finish_reading(self, owner: object) -> None:
        assert self.feature is not None
        self.lock_held_at_finish.append(self.feature._lock.locked())  # noqa: SLF001
        super().finish_reading(owner)


def test_the_reading_claim_is_released_while_holding_the_feature_lock() -> None:
    coordinator = _LockSpyCoordinator()
    feature, _statuses, _progresses, finishes = _make_feature(
        fallback=FakeSynthesizer(),
        primary=FakeSynthesizer(),
        notifier=RecordingNotifier(),
        coordinator=coordinator,
    )
    coordinator.feature = feature

    feature.read_text("First sentence. Second sentence.")
    wait_until(lambda: bool(finishes))

    assert coordinator.lock_held_at_finish == [True]


def _two_chunk_text() -> str:
    """Two sentences that chunk_text() is guaranteed to split into two
    separate chunks (the first alone already exceeds the default
    max_chars, forcing the second sentence into its own chunk)."""
    long_sentence = "A" * 320 + "."
    short_sentence = "B" * 20 + "."
    return f"{long_sentence} {short_sentence}"


def test_no_speed_provider_leaves_chunk_audio_unchanged() -> None:
    primary = SineSynthesizer()
    player = LengthRecordingPlayer()
    finishes: "list[None]" = []
    feature = DocumentReaderFeature(
        model_manager=ModelManager(),
        primary_engine_factory=lambda: primary,
        fallback_engine=FakeSynthesizer(),
        player=player,  # type: ignore[arg-type]
        notifier=RecordingNotifier(),  # type: ignore[arg-type]
        reading_coordinator=ReadingCoordinator(),
        on_status=lambda message: None,
        on_progress=lambda current, total: None,
        on_finished=lambda: finishes.append(None),
    )

    feature.read_text("Only one short sentence here to read aloud slowly.")
    wait_until(lambda: bool(finishes))

    assert player.play_lengths == [int(primary.sample_rate * 2.0)]


def test_speed_provider_is_read_per_chunk_and_produces_two_lengths() -> None:
    """Changing the speed between two chunks of the same document
    produces two differently-sized played chunks."""
    primary = SineSynthesizer()
    player = LengthRecordingPlayer()
    speeds = iter([1.2, 1.5])
    finishes: "list[None]" = []
    feature = DocumentReaderFeature(
        model_manager=ModelManager(),
        primary_engine_factory=lambda: primary,
        fallback_engine=FakeSynthesizer(),
        player=player,  # type: ignore[arg-type]
        notifier=RecordingNotifier(),  # type: ignore[arg-type]
        reading_coordinator=ReadingCoordinator(),
        on_status=lambda message: None,
        on_progress=lambda current, total: None,
        on_finished=lambda: finishes.append(None),
        speed_provider=lambda: next(speeds),
    )

    feature.read_text(_two_chunk_text())
    wait_until(lambda: bool(finishes))

    assert len(primary.calls) == 2  # really two chunks
    assert len(player.play_lengths) == 2
    original_length = int(primary.sample_rate * 2.0)
    assert abs(player.play_lengths[0] - original_length / 1.2) < 0.03 * original_length
    assert abs(player.play_lengths[1] - original_length / 1.5) < 0.03 * original_length
    assert player.play_lengths[0] != player.play_lengths[1]


class PacedWindowPlayer:
    """Like WindowPlayer (stop() only has an effect once play() has
    returned -- before that there is no stream to stop) but also records
    each played chunk's length, so a test can confirm a resumed chunk
    replays the exact same (already-paced) audio rather than a
    re-synthesized/re-paced one."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.play_lengths: "list[int]" = []
        self.effective_stops = 0
        self._streaming = False
        self._stopped = False

    def play(self, audio: Any, sample_rate: int) -> None:
        self.play_lengths.append(len(audio))
        self._stopped = False
        if len(self.play_lengths) == 1:
            self.entered.set()
            assert self.release.wait(timeout=5.0), "release was never set"
        self._streaming = True

    def wait_until_done(self) -> bool:
        self._streaming = False
        return not self._stopped

    def stop(self) -> None:
        if self._streaming:
            self._stopped = True
            self.effective_stops += 1


def test_pause_then_resume_replays_the_same_already_paced_chunk() -> None:
    player = PacedWindowPlayer()
    primary = SineSynthesizer()
    statuses: "list[str]" = []
    finishes: "list[None]" = []
    feature = DocumentReaderFeature(
        model_manager=ModelManager(),
        primary_engine_factory=lambda: primary,
        fallback_engine=FakeSynthesizer(),
        player=player,  # type: ignore[arg-type]
        notifier=RecordingNotifier(),  # type: ignore[arg-type]
        reading_coordinator=ReadingCoordinator(),
        on_status=statuses.append,
        on_progress=lambda current, total: None,
        on_finished=lambda: finishes.append(None),
        speed_provider=lambda: 1.5,
    )
    gate = _CountingEvent()
    gate.set()
    feature._resume_event = gate  # noqa: SLF001 -- lets the test see the worker arrive

    feature.read_text("Only one short sentence here to read aloud slowly.")
    assert player.entered.wait(timeout=5.0)  # worker is inside play(), no stream yet
    feature.pause()  # player.stop() is a no-op: there is no stream to stop
    assert player.effective_stops == 0
    waits_before = gate.wait_calls
    player.release.set()  # play() now returns

    wait_until(lambda: player.effective_stops == 1)  # worker stopped the chunk itself
    wait_until(lambda: gate.wait_calls > waits_before)  # and is parked, waiting for resume
    assert statuses[-1] == "Paused"
    assert len(player.play_lengths) == 1

    feature.resume()
    wait_until(lambda: bool(finishes))
    _join_workers()

    assert len(player.play_lengths) == 2  # the same chunk was replayed
    assert player.play_lengths[0] == player.play_lengths[1]  # same already-paced audio
    assert primary.calls == ["Only one short sentence here to read aloud slowly."]  # synthesized once
    assert statuses[-1] == "Done"
    assert finishes == [None]


def test_a_raising_speed_provider_still_plays_the_original_chunk(
    caplog: "pytest.LogCaptureFixture",
) -> None:
    """A speed_provider that raises must never break the reading: the
    original chunk still plays and a warning is logged."""
    primary = SineSynthesizer()
    player = LengthRecordingPlayer()
    finishes: "list[None]" = []

    def exploding_provider() -> float:
        raise RuntimeError("settings unavailable")

    feature = DocumentReaderFeature(
        model_manager=ModelManager(),
        primary_engine_factory=lambda: primary,
        fallback_engine=FakeSynthesizer(),
        player=player,  # type: ignore[arg-type]
        notifier=RecordingNotifier(),  # type: ignore[arg-type]
        reading_coordinator=ReadingCoordinator(),
        on_status=lambda message: None,
        on_progress=lambda current, total: None,
        on_finished=lambda: finishes.append(None),
        speed_provider=exploding_provider,
    )

    with caplog.at_level(logging.WARNING, logger="vocalis"):
        feature.read_text("Only one short sentence here to read aloud slowly.")
        wait_until(lambda: bool(finishes))

    assert player.play_lengths == [int(primary.sample_rate * 2.0)]  # unchanged
    assert any(record.levelno == logging.WARNING for record in caplog.records)


def _make_selector_feature(
    preference: str,
    factories: "dict[str, Callable[[], Any]]",
    fallback: FakeSynthesizer,
    notifier: RecordingNotifier,
) -> "tuple[DocumentReaderFeature, list[str], list[None]]":
    """Build a feature whose engines are chosen by a selector (no real hardware)."""
    from vocalis.engines.registry import EngineSelector
    from vocalis.hardware import HardwareInfo

    statuses: "list[str]" = []
    finishes: "list[None]" = []
    selector = EngineSelector(
        preference=lambda: preference,
        hardware_probe=lambda: HardwareInfo(None, None, None, None, None),
    )
    feature = DocumentReaderFeature(
        model_manager=ModelManager(),
        primary_engine_factory=lambda: pytest.fail("the old primary factory must not be used"),
        fallback_engine=fallback,
        player=AudioPlayer(stream_factory=_fake_stream_factory),
        notifier=notifier,  # type: ignore[arg-type]
        reading_coordinator=ReadingCoordinator(),
        on_status=statuses.append,
        on_progress=lambda current, total: None,
        on_finished=lambda: finishes.append(None),
        engine_selector=selector,
        engine_factories=factories,
    )
    return feature, statuses, finishes


def test_a_selector_planned_engine_speaks_the_document() -> None:
    kokoro = FakeSynthesizer()
    fallback = FakeSynthesizer()
    notifier = RecordingNotifier()
    feature, _statuses, finishes = _make_selector_feature(
        "kokoro", {"kokoro": lambda: kokoro}, fallback, notifier
    )

    feature.read_text("One. Two. Three. Four. Five. Six. Seven. Eight.")
    wait_until(lambda: bool(finishes))

    assert kokoro.calls
    assert fallback.calls == []
    assert "Primary voice unavailable — using backup voice" not in notifier.messages


def test_a_failed_planned_engine_is_not_retried_and_is_announced_once() -> None:
    fallback = FakeSynthesizer()
    notifier = RecordingNotifier()
    factory_calls: "list[int]" = []

    def broken_kokoro() -> Any:
        factory_calls.append(1)
        raise RuntimeError("kokoro will not load")

    feature, statuses, finishes = _make_selector_feature(
        "kokoro", {"kokoro": broken_kokoro}, fallback, notifier
    )

    feature.read_text("One. Two. Three. Four. Five. Six. Seven. Eight.")
    wait_until(lambda: bool(finishes))

    assert fallback.calls
    assert len(factory_calls) == 1  # never retried for later chunks
    assert notifier.messages.count("Primary voice unavailable — using backup voice") == 1
    assert statuses.count("Using backup voice") == 1


def test_a_failing_backup_voice_sends_no_downshift_notification() -> None:
    notifier = RecordingNotifier()
    feature, statuses, _finishes = _make_selector_feature(
        "kokoro", {}, FakeSynthesizer(), notifier
    )

    feature._notify_downshift("piper", None)  # noqa: SLF001

    assert notifier.messages == []
    assert "Using backup voice" not in statuses
