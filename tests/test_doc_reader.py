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

from tests.fake_output import OutputRecorder, manual_stream_factory, pump_stream_factory
from vocalis.audio.playback import AudioPlayer
from vocalis.engines.chain import make_downshift_handler
from vocalis.features.doc_reader import DocumentReaderFeature
from vocalis.features.reading_coordinator import ReadingCoordinator
from vocalis.model_manager import TTS_ENGINE_ID, ModelManager
from vocalis.text.chunker import chunk_text
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
        level: float = 0.0,
    ) -> None:
        """Create a fake voice; `level` is the constant value of its audio (0 = silence)."""
        self.sample_rate = sample_rate
        self.raise_exc = raise_exc
        self.level = level
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
        return np.full(100, self.level, dtype=np.float32), self.sample_rate


class GatedSynthesizer:
    """A synthesizer whose synthesize() blocks until release() is called,
    so a test can pause() while synthesis of a chunk is genuinely still in
    flight -- deterministic, unlike a fixed sleep() that merely hopes to
    land inside the right window."""

    def __init__(self, sample_rate: int = 22050, level: float = 0.0) -> None:
        """Create a gated fake voice; `level` is the constant value of its audio."""
        self.sample_rate = sample_rate
        self.level = level
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
        return np.full(100, self.level, dtype=np.float32), self.sample_rate


StreamStartRecorder = OutputRecorder  # compatibility alias for tests/test_reading_hooks_integration.py


def _endless_stream_factory(recorder: OutputRecorder) -> "Callable[..., Any]":
    """A stream factory whose streams nobody pulls: a reading that never finishes by itself."""
    return manual_stream_factory(recorder)


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
    stream_factory: "Callable[..., Any] | None" = None,
    on_started: "Callable[[], None] | None" = None,
) -> "tuple[DocumentReaderFeature, list[str], list[tuple[int, int]], list[None]]":
    statuses: "list[str]" = []
    progresses: "list[tuple[int, int]]" = []
    finishes: "list[None]" = []
    feature = DocumentReaderFeature(
        model_manager=ModelManager(),
        primary_engine_factory=lambda: primary,
        fallback_engine=fallback,
        player=AudioPlayer(
            stream_factory=stream_factory or pump_stream_factory(OutputRecorder())
        ),
        notifier=notifier,  # type: ignore[arg-type]
        reading_coordinator=coordinator or ReadingCoordinator(),
        on_status=statuses.append,
        on_progress=lambda current, total: progresses.append((current, total)),
        on_finished=lambda: finishes.append(None),
        on_started=on_started,
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


def test_read_text_speaks_every_chunk_and_reports_done() -> None:
    primary = FakeSynthesizer()
    feature, statuses, progresses, finishes = _make_feature(
        fallback=FakeSynthesizer(), primary=primary, notifier=RecordingNotifier()
    )

    feature.read_text("First sentence. Second sentence.")
    wait_until(lambda: bool(finishes))

    assert len(primary.calls) == 1  # short text fits in one chunk
    wait_until(lambda: progresses[:1] == [(1, 1)])
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


def _wait_queued(feature: DocumentReaderFeature, clips: int) -> None:
    """Wait until the feature's speaker has `clips` chunks queued in its stream."""
    wait_until(lambda: feature._speaker is not None and feature._speaker.queued_clips == clips)  # noqa: SLF001


def _drain(recorder: OutputRecorder, finishes: "list[None]") -> "list[Any]":
    """Pull the last stream until the reading reports it has finished; returns the pulls."""
    stream = recorder.streams[-1]
    pulls: "list[Any]" = []
    deadline = time.monotonic() + 5.0
    while not finishes:
        assert time.monotonic() < deadline, "the reading never finished"
        pulls.append(stream.pull(2048))
        time.sleep(0.002)
    return pulls


def _reading_threads() -> "set[threading.Thread]":
    """Snapshot of all live threads (taken before a reading starts)."""
    return set(threading.enumerate())


def _join_reading_workers(before: "set[threading.Thread]") -> None:
    """Wait (bounded) for every reading worker started since `before` to finish.

    A worker that kept running after stop() could still send a second "Done"
    or finished report, so the stop tests call this before their final asserts.
    """
    for thread in threading.enumerate():
        if thread not in before and "_prepare_and_read" in thread.name:
            thread.join(timeout=5.0)
            assert not thread.is_alive(), "reading worker did not finish"


def test_pause_stops_audio_immediately_and_resume_replays_current_chunk() -> None:
    """Pause goes silent at once; resume replays the heard chunk from its start without re-making it."""
    primary = FakeSynthesizer(level=0.5)
    recorder = OutputRecorder()
    feature, statuses, _progresses, finishes = _make_feature(
        fallback=FakeSynthesizer(),
        primary=primary,
        notifier=RecordingNotifier(),
        stream_factory=manual_stream_factory(recorder),
    )

    feature.read_text("A. B.")
    _wait_queued(feature, 1)
    stream = recorder.streams[-1]
    heard = stream.pull(10)
    assert heard.any()

    feature.pause()
    assert not stream.pull(10).any()
    assert "Paused" in statuses
    assert feature.is_reading is True  # paused, not stopped

    feature.resume()
    replay = stream.pull(100)
    np.testing.assert_allclose(replay[:10], heard)  # the same chunk, from its start
    _drain(recorder, finishes)
    assert "Done" in statuses
    assert len(primary.calls) == 1  # the pause did not re-make anything


def test_pause_during_synthesis_blocks_playback_until_resume() -> None:
    """A pause that lands while a chunk is still being made keeps it silent until resume."""
    primary = GatedSynthesizer(level=0.5)
    recorder = OutputRecorder()
    feature, statuses, _progresses, finishes = _make_feature(
        fallback=FakeSynthesizer(),
        primary=primary,
        notifier=RecordingNotifier(),
        stream_factory=manual_stream_factory(recorder),
    )

    feature.read_text("Only one short sentence here to read aloud slowly.")
    primary.wait_until_entered()  # synthesis of the first chunk is in flight
    feature.pause()
    assert "Paused" in statuses
    primary.release()
    _wait_queued(feature, 1)
    stream = recorder.streams[-1]

    assert not stream.pull(50).any()  # paused: not a sound
    assert finishes == []

    feature.resume()
    assert stream.pull(50).any()
    _drain(recorder, finishes)
    assert "Done" in statuses


def test_stop_while_paused_actually_stops() -> None:
    """Stopping while paused ends the reading once and closes the sound stream."""
    primary = FakeSynthesizer(level=0.5)
    recorder = OutputRecorder()
    feature, statuses, _progresses, finishes = _make_feature(
        fallback=FakeSynthesizer(),
        primary=primary,
        notifier=RecordingNotifier(),
        stream_factory=manual_stream_factory(recorder),
    )

    before = _reading_threads()
    feature.read_text("Only one short sentence here to read aloud slowly.")
    _wait_queued(feature, 1)
    feature.pause()
    feature.stop()

    wait_until(lambda: recorder.closed == 1)
    _join_reading_workers(before)
    assert finishes == [None]
    assert feature.is_reading is False
    assert "Stopped" in statuses
    assert "Done" not in statuses


def test_a_second_reader_preempts_this_one_via_the_coordinator() -> None:
    """Another reading claiming the coordinator stops this one cleanly."""
    coordinator = ReadingCoordinator()
    recorder = OutputRecorder()
    feature, statuses, _progresses, finishes = _make_feature(
        fallback=FakeSynthesizer(),
        primary=FakeSynthesizer(level=0.5),
        notifier=RecordingNotifier(),
        coordinator=coordinator,
        stream_factory=_endless_stream_factory(recorder),
    )

    before = _reading_threads()
    feature.read_text("Only one short sentence here to read aloud slowly.")
    wait_until(lambda: recorder.opened == 1)  # playback has really begun
    coordinator.start_reading(object(), lambda: None)  # a different owner preempts it

    wait_until(lambda: bool(finishes))
    _join_reading_workers(before)
    assert finishes == [None]
    assert feature.is_reading is False
    wait_until(lambda: recorder.closed == 1)
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


class _RefusingStream:
    """A sound stream that cannot be started."""

    def __init__(self, **kwargs: Any) -> None:
        """Accept and ignore the stream arguments."""

    def start(self) -> None:
        """Refuse to start."""
        raise RuntimeError("device busy")

    def stop(self) -> None:
        """Nothing to stop."""

    def close(self) -> None:
        """Nothing to close."""


def test_playback_raising_reports_playback_failed() -> None:
    notifier = RecordingNotifier()
    feature, statuses, _progresses, finishes = _make_feature(
        fallback=FakeSynthesizer(),
        primary=FakeSynthesizer(),
        notifier=notifier,
        stream_factory=lambda **kwargs: _RefusingStream(**kwargs),
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
        player=AudioPlayer(stream_factory=pump_stream_factory(OutputRecorder())),
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

    recorder = OutputRecorder()
    notifier = RecordingNotifier()
    reader, statuses, _progresses, finishes = _make_feature(
        fallback=FakeSynthesizer(),
        primary=FakeSynthesizer(level=0.5),
        notifier=notifier,
        stream_factory=_endless_stream_factory(recorder),
    )
    ocr_feature = OcrReaderFeature(
        capture=Capture(), ocr=Ocr(), clipboard=Clipboard(), reader=reader,
        notifier=notifier,  # type: ignore[arg-type]
    )
    try:
        reader.read_text("First document text.")
        wait_until(lambda: recorder.opened == 1)

        ocr_feature.on_trigger()
        wait_until(lambda: not ocr_feature.is_busy)
        wait_until(lambda: recorder.opened == 2)  # the OCR text started playing

        assert recorder.closed >= 1  # the first reading's sound stream was shut
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
        player=AudioPlayer(stream_factory=pump_stream_factory(OutputRecorder())),
        notifier=notifier,  # type: ignore[arg-type]
        reading_coordinator=ReadingCoordinator(),
        on_status=statuses.append,
        on_progress=lambda current, total: None,
        on_finished=lambda: finishes.append(None),
        engine_selector=selector,
        engine_factories=factories,
    )
    return feature, statuses, finishes


def _several_chunk_text() -> str:
    """Three long sentences that chunk_text() splits into at least three chunks."""
    return " ".join(letter * 200 + "." for letter in "ABC")


def test_a_selector_planned_engine_speaks_the_document() -> None:
    kokoro = FakeSynthesizer()
    fallback = FakeSynthesizer()
    notifier = RecordingNotifier()
    feature, _statuses, finishes = _make_selector_feature(
        "kokoro", {"kokoro": lambda: kokoro}, fallback, notifier
    )

    text = _several_chunk_text()
    chunk_count = len(chunk_text(text))
    assert chunk_count >= 3
    feature.read_text(text)
    wait_until(lambda: bool(finishes))

    assert len(kokoro.calls) == chunk_count
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

    text = _several_chunk_text()
    chunk_count = len(chunk_text(text))
    assert chunk_count >= 3
    feature.read_text(text)
    wait_until(lambda: bool(finishes))

    assert len(fallback.calls) == chunk_count  # Piper spoke every chunk
    assert len(factory_calls) == 1  # never retried for later chunks
    assert notifier.messages.count("Primary voice unavailable — using backup voice") == 1
    assert statuses.count("Using backup voice") == 1


def test_a_failing_backup_voice_sends_no_downshift_notification() -> None:
    notifier = RecordingNotifier()
    statuses: "list[str]" = []
    handler = make_downshift_handler(lambda message: notifier.notify("t", message), statuses.append)

    handler("piper", None)

    assert notifier.messages == []
    assert "Using backup voice" not in statuses


def test_progress_is_reported_when_each_piece_starts_to_be_heard() -> None:
    """Progress (N, M) arrives when chunk N starts to be heard, not when it is made."""
    recorder = OutputRecorder()
    primary = FakeSynthesizer(level=0.5)
    feature, _statuses, progresses, _finishes = _make_feature(
        fallback=FakeSynthesizer(), primary=primary, notifier=RecordingNotifier(),
        stream_factory=manual_stream_factory(recorder),
    )
    # 350 characters each, and capitalised so the chunker sees a sentence start
    # after every full stop: each sentence becomes its own chunk.
    sentence = " ".join(["Word"] + ["word"] * 69) + "."
    feature.read_text(" ".join([sentence] * 3))

    wait_until(lambda: feature._speaker is not None and feature._speaker.queued_clips == 3)  # noqa: SLF001
    assert progresses == []  # all three are made, none has been heard yet

    stream = recorder.streams[-1]
    stream.pull(10)
    wait_until(lambda: progresses == [(1, 3)])
    stream.pull(10_000)
    wait_until(lambda: progresses == [(1, 3), (2, 3), (3, 3)])


class _NoopTimer:
    """A timer that does nothing when cancelled."""

    def cancel(self) -> None:
        """Nothing to cancel."""


class _CountingTimers:
    """A ModelManager timer factory that only counts how many idle timers were started."""

    def __init__(self) -> None:
        """Start with no timers scheduled."""
        self.scheduled = 0

    def __call__(self, delay: float, callback: Callable[[], None]) -> "_NoopTimer":
        """Count one scheduled idle timer."""
        self.scheduled += 1
        return _NoopTimer()


def test_the_voice_stays_loaded_for_the_whole_reading_and_the_idle_timer_starts_after_playback() -> None:
    """The voice is held until playback ends; only then does the idle-unload timer start."""
    timers = _CountingTimers()
    recorder = OutputRecorder()
    primary = FakeSynthesizer(level=0.5)
    finishes: "list[None]" = []
    feature = DocumentReaderFeature(
        model_manager=ModelManager(idle_timeout_seconds=30.0, timer_factory=timers),
        primary_engine_factory=lambda: primary,
        fallback_engine=FakeSynthesizer(),
        player=AudioPlayer(stream_factory=manual_stream_factory(recorder)),
        notifier=RecordingNotifier(),  # type: ignore[arg-type]
        reading_coordinator=ReadingCoordinator(),
        on_status=lambda message: None,
        on_progress=lambda current, total: None,
        on_finished=lambda: finishes.append(None),
    )

    feature.read_text("One short sentence. Another short sentence.")
    wait_until(lambda: feature._speaker is not None and feature._speaker.queued_clips >= 1)  # noqa: SLF001
    assert timers.scheduled == 0  # every piece is made, none has been heard: the voice must stay loaded

    _drain(recorder, finishes)
    wait_until(lambda: timers.scheduled == 1)  # the idle timer starts only now that playback has ended


def test_stop_before_the_first_sound_exists_never_makes_sound() -> None:
    """A stop that lands while the first chunk is still being made never lets a sound out."""
    recorder = OutputRecorder()
    primary = GatedSynthesizer(level=0.5)
    feature, _statuses, _progresses, finishes = _make_feature(
        fallback=FakeSynthesizer(), primary=primary, notifier=RecordingNotifier(),
        stream_factory=manual_stream_factory(recorder),
    )

    before = _reading_threads()
    feature.read_text("Only one short sentence here.")
    primary.wait_until_entered()
    feature.stop()
    primary.release()
    wait_until(lambda: bool(finishes))
    _join_reading_workers(before)

    assert finishes == [None]
    assert recorder.opened == 0
    assert feature.is_reading is False


def _gate_chunking(monkeypatch: pytest.MonkeyPatch) -> "tuple[threading.Event, threading.Event]":
    """Hold the reading worker inside the chunking step; returns (in_chunking, let_go) events."""
    in_chunking = threading.Event()
    let_go = threading.Event()
    real_chunk_text = chunk_text

    def gated(text: str) -> "list[str]":
        """Wait for the test, then chunk normally."""
        in_chunking.set()
        assert let_go.wait(timeout=5.0), "the chunking gate was never opened"
        return real_chunk_text(text)

    monkeypatch.setattr("vocalis.features.doc_reader.chunk_text", gated)
    return in_chunking, let_go


def test_pause_before_the_speaker_exists_starts_the_reading_paused(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pause that lands before the speaker exists silences the reading from its first sound."""
    recorder = OutputRecorder()
    feature, statuses, _progresses, finishes = _make_feature(
        fallback=FakeSynthesizer(), primary=FakeSynthesizer(level=0.5), notifier=RecordingNotifier(),
        stream_factory=manual_stream_factory(recorder),
    )
    in_chunking, let_go = _gate_chunking(monkeypatch)

    feature.read_text("Only one short sentence here.")
    assert in_chunking.wait(timeout=5.0)
    assert feature._speaker is None  # noqa: SLF001 -- really before the speaker exists
    feature.pause()
    let_go.set()
    _wait_queued(feature, 1)
    stream = recorder.streams[-1]

    assert not stream.pull(50).any()
    feature.resume()
    assert stream.pull(50).any()
    _drain(recorder, finishes)
    assert "Paused" in statuses and "Done" in statuses


def test_a_resume_in_the_publish_window_is_not_lost(monkeypatch: pytest.MonkeyPatch) -> None:
    """A resume() racing the worker that publishes a speaker for a pre-paused reading still gets sound."""
    from vocalis.features.chunked_speaker import ChunkedSpeaker

    recorder = OutputRecorder()
    feature, _statuses, _progresses, finishes = _make_feature(
        fallback=FakeSynthesizer(), primary=FakeSynthesizer(level=0.5), notifier=RecordingNotifier(),
        stream_factory=manual_stream_factory(recorder),
    )
    in_chunking, let_go = _gate_chunking(monkeypatch)
    pausing = threading.Event()
    real_pause = ChunkedSpeaker.pause

    def slow_pause(self: ChunkedSpeaker) -> None:
        """Pause, but only after the test's resume() has had time to race it."""
        pausing.set()
        threading.Event().wait(0.2)
        real_pause(self)

    monkeypatch.setattr(ChunkedSpeaker, "pause", slow_pause)

    feature.read_text("Only one short sentence here.")
    assert in_chunking.wait(timeout=5.0)
    feature.pause()
    let_go.set()
    assert pausing.wait(timeout=5.0)  # the worker is applying the pause to its new speaker
    resumer = threading.Thread(target=feature.resume, daemon=True)
    resumer.start()
    resumer.join(timeout=5.0)
    _wait_queued(feature, 1)
    stream = recorder.streams[-1]

    assert stream.pull(50).any()  # the resume was not swallowed by the late pause
    _drain(recorder, finishes)


def test_a_repeat_read_stops_the_reading_in_progress_properly() -> None:
    """Reading again while a reading is under way (voice preview) ends the old one once and plays the new."""
    recorder = OutputRecorder()
    failed: "list[int | None]" = []
    statuses: "list[str]" = []
    finishes: "list[None]" = []
    feature = DocumentReaderFeature(
        model_manager=ModelManager(),
        primary_engine_factory=lambda: FakeSynthesizer(level=0.5),
        fallback_engine=FakeSynthesizer(),
        player=AudioPlayer(stream_factory=manual_stream_factory(recorder), on_device_failed=failed.append),
        notifier=RecordingNotifier(),  # type: ignore[arg-type]
        reading_coordinator=ReadingCoordinator(),
        on_status=statuses.append,
        on_progress=lambda current, total: None,
        on_finished=lambda: finishes.append(None),
    )
    sentence = " ".join(["Word"] + ["word"] * 69) + "."
    feature.read_text(" ".join([sentence] * 5))
    _wait_queued(feature, 3)
    old_speaker = feature._speaker  # noqa: SLF001
    assert old_speaker is not None

    feature.read_text("Preview.")

    assert old_speaker._stop.is_set()  # noqa: SLF001 -- the old reading really was stopped
    assert finishes == [None] and statuses.count("Stopped") == 1
    wait_until(lambda: recorder.opened == 2)
    wait_until(lambda: feature._speaker is not None and feature._speaker.queued_clips == 1)  # noqa: SLF001
    assert recorder.streams[-1].pull(50).any()  # the new reading plays
    stream = recorder.streams[-1]
    deadline = time.monotonic() + 5.0
    while len(finishes) < 2:
        assert time.monotonic() < deadline, "the new reading never finished"
        stream.pull(2048)
        time.sleep(0.002)
    wait_until(lambda: "Done" in statuses)
    assert len(finishes) == 2 and statuses.count("Stopped") == 1
    assert failed == []  # the working sound device was never blamed
