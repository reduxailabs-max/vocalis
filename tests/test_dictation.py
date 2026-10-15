"""Tests for vocalis.features.dictation.DictationFeature.

Uses fakes for every collaborator (engine, recorder, text output,
notifier) — this suite verifies orchestration logic only. Each
collaborator's own real-world correctness is tested in its own test file
(test_parakeet.py, test_capture.py, test_typer.py, test_notify.py).
"""
import time
from typing import Callable

import numpy as np
import numpy.typing as npt

from vocalis.features.dictation import DictationFeature
from vocalis.model_manager import ModelManager


class FakeEngine:
    def __init__(self, transcript: str = "hello world", raise_on_transcribe: Exception | None = None) -> None:
        self.transcript = transcript
        self.raise_on_transcribe = raise_on_transcribe
        self._loaded = False
        self.load_calls = 0
        self.unload_calls = 0
        self.transcribe_calls: list[tuple[npt.NDArray[np.float32], int]] = []

    def load(self) -> None:
        self.load_calls += 1
        self._loaded = True

    def unload(self) -> None:
        self.unload_calls += 1
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def transcribe(self, audio: npt.NDArray[np.float32], sample_rate: int) -> str:
        self.transcribe_calls.append((audio, sample_rate))
        if self.raise_on_transcribe is not None:
            raise self.raise_on_transcribe
        return self.transcript


class FakeRecorder:
    def __init__(
        self,
        audio_to_return: npt.NDArray[np.float32],
        raise_on_start: Exception | None = None,
        raise_on_stop: Exception | None = None,
    ) -> None:
        self.audio_to_return = audio_to_return
        self.raise_on_start = raise_on_start
        self.raise_on_stop = raise_on_stop
        self.start_calls = 0
        self.stop_calls = 0

    def start(self) -> None:
        self.start_calls += 1
        if self.raise_on_start is not None:
            raise self.raise_on_start

    def stop(self) -> npt.NDArray[np.float32]:
        self.stop_calls += 1
        if self.raise_on_stop is not None:
            raise self.raise_on_stop
        return self.audio_to_return


class FakeOutputResult:
    def __init__(self, method: str, text: str) -> None:
        self.method = method
        self.text = text


class FakeTextOutput:
    def __init__(self, method: str = "typed") -> None:
        self.method = method
        self.delivered: list[str] = []

    def deliver(self, text: str) -> FakeOutputResult:
        self.delivered.append(text)
        return FakeOutputResult(method=self.method, text=text)


class FakeNotifier:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def notify(self, title: str, message: str = "") -> None:
        self.messages.append((title, message))


def make_feature(
    engine: FakeEngine,
    audio: npt.NDArray[np.float32],
    output_method: str = "typed",
    on_listening_changed: "Callable[[bool], None] | None" = None,
    recorder: "FakeRecorder | None" = None,
) -> tuple[DictationFeature, FakeRecorder, FakeTextOutput, FakeNotifier, ModelManager]:
    # A short real idle-timeout (matching the convention used in
    # test_model_manager_integration.py) — ModelManager.release() only
    # *schedules* an unload via a real threading.Timer, it never clears
    # current_engine_id synchronously, so tests that check current_engine_id
    # after on_release() need a real, but short, timeout plus a sleep past
    # it (see test_release_transcribes_and_delivers_result and
    # test_transcription_failure_notifies_and_does_not_crash below).
    manager = ModelManager(idle_timeout_seconds=0.05)
    recorder = recorder or FakeRecorder(audio_to_return=audio)
    text_output = FakeTextOutput(method=output_method)
    notifier = FakeNotifier()
    feature = DictationFeature(
        model_manager=manager,
        engine_factory=lambda: engine,
        recorder=recorder,
        text_output=text_output,  # type: ignore[arg-type]
        notifier=notifier,  # type: ignore[arg-type]
        on_listening_changed=on_listening_changed,
    )
    return feature, recorder, text_output, notifier, manager


def test_press_starts_recording_and_notifies() -> None:
    engine = FakeEngine()
    feature, recorder, _, notifier, _ = make_feature(engine, np.zeros(0, dtype=np.float32))
    feature.on_press()
    assert recorder.start_calls == 1
    assert any("Listening" in message for _, message in notifier.messages)


def test_release_transcribes_and_delivers_result() -> None:
    engine = FakeEngine(transcript="hello world")
    audio = np.ones(1600, dtype=np.float32)
    feature, recorder, text_output, notifier, manager = make_feature(engine, audio)
    feature.on_press()
    feature.on_release()
    assert recorder.stop_calls == 1
    assert len(engine.transcribe_calls) == 1
    called_audio, called_sample_rate = engine.transcribe_calls[0]
    assert called_sample_rate == 16000
    assert np.array_equal(called_audio, audio)
    assert text_output.delivered == ["hello world"]
    time.sleep(0.2)  # let the real idle-timeout unload after release()
    assert manager.current_engine_id is None  # released back after use
    assert any(message == "Done" for _, message in notifier.messages)


def test_empty_audio_skips_transcription() -> None:
    engine = FakeEngine()
    feature, recorder, text_output, notifier, _ = make_feature(engine, np.zeros(0, dtype=np.float32))
    feature.on_press()
    feature.on_release()
    assert engine.transcribe_calls == []
    assert text_output.delivered == []
    assert any("No audio" in message for _, message in notifier.messages)


def test_blank_transcript_skips_output() -> None:
    engine = FakeEngine(transcript="   ")
    audio = np.ones(1600, dtype=np.float32)
    feature, _, text_output, notifier, _ = make_feature(engine, audio)
    feature.on_press()
    feature.on_release()
    assert text_output.delivered == []
    assert any("no speech" in message.lower() for _, message in notifier.messages)


def test_transcription_failure_notifies_and_does_not_crash() -> None:
    engine = FakeEngine(raise_on_transcribe=RuntimeError("boom"))
    audio = np.ones(1600, dtype=np.float32)
    feature, _, text_output, notifier, manager = make_feature(engine, audio)
    feature.on_press()
    feature.on_release()  # must not raise
    assert text_output.delivered == []
    assert any("Transcription failed" in message for _, message in notifier.messages)
    time.sleep(0.2)  # let the real idle-timeout unload after release()
    assert manager.current_engine_id is None  # still released despite failure


def test_clipboard_fallback_notifies_accordingly() -> None:
    engine = FakeEngine(transcript="hello")
    audio = np.ones(1600, dtype=np.float32)
    feature, _, _, notifier, _ = make_feature(engine, audio, output_method="clipboard")
    feature.on_press()
    feature.on_release()
    assert any("clipboard" in message.lower() for _, message in notifier.messages)


def test_recorder_stop_failure_notifies_and_does_not_crash() -> None:
    engine = FakeEngine()
    manager = ModelManager()
    recorder = FakeRecorder(
        audio_to_return=np.zeros(0, dtype=np.float32), raise_on_stop=RuntimeError("mic error")
    )
    text_output = FakeTextOutput()
    notifier = FakeNotifier()
    feature = DictationFeature(
        model_manager=manager,
        engine_factory=lambda: engine,
        recorder=recorder,
        text_output=text_output,  # type: ignore[arg-type]
        notifier=notifier,  # type: ignore[arg-type]
    )
    feature.on_press()
    feature.on_release()  # must not raise
    assert any("Recording failed" in message for _, message in notifier.messages)


def test_duplicate_press_while_recording_is_ignored() -> None:
    engine = FakeEngine()
    feature, recorder, _, notifier, _ = make_feature(engine, np.zeros(0, dtype=np.float32))
    feature.on_press()
    feature.on_press()  # duplicate — must be ignored
    assert recorder.start_calls == 1
    assert sum(1 for _, message in notifier.messages if "Listening" in message) == 1


def test_duplicate_release_while_not_recording_is_ignored() -> None:
    engine = FakeEngine()
    feature, recorder, _, _, _ = make_feature(engine, np.zeros(0, dtype=np.float32))
    feature.on_release()  # never pressed — must be ignored, not call stop()
    assert recorder.stop_calls == 0


def test_recorder_start_failure_notifies_and_does_not_crash() -> None:
    engine = FakeEngine()
    manager = ModelManager()
    recorder = FakeRecorder(
        audio_to_return=np.zeros(0, dtype=np.float32), raise_on_start=RuntimeError("mic busy")
    )
    text_output = FakeTextOutput()
    notifier = FakeNotifier()
    feature = DictationFeature(
        model_manager=manager,
        engine_factory=lambda: engine,
        recorder=recorder,
        text_output=text_output,  # type: ignore[arg-type]
        notifier=notifier,  # type: ignore[arg-type]
    )
    feature.on_press()  # must not raise
    assert any("Could not start recording" in message for _, message in notifier.messages)


def test_listening_is_reported_true_on_press_and_false_on_release() -> None:
    changes: list[bool] = []
    feature, _, _, _, _ = make_feature(
        FakeEngine(), np.ones(1600, dtype=np.float32), on_listening_changed=changes.append
    )
    feature.on_press()
    assert changes == [True]
    feature.on_release()
    assert changes == [True, False]


def test_listening_not_reported_true_when_recording_fails_to_start() -> None:
    changes: list[bool] = []
    recorder = FakeRecorder(np.zeros(0, dtype=np.float32), raise_on_start=RuntimeError("no mic"))
    feature, _, _, _, _ = make_feature(
        FakeEngine(), np.zeros(0, dtype=np.float32), on_listening_changed=changes.append, recorder=recorder
    )
    feature.on_press()
    assert changes == []
    feature.on_release()  # not recording any more: ignored
    assert changes == []


def test_duplicate_press_does_not_report_twice() -> None:
    changes: list[bool] = []
    feature, _, _, _, _ = make_feature(
        FakeEngine(), np.ones(1600, dtype=np.float32), on_listening_changed=changes.append
    )
    feature.on_press()
    feature.on_press()
    assert changes == [True]


def test_raising_listening_hook_does_not_break_dictation() -> None:
    def broken(active: bool) -> None:
        raise RuntimeError("ui bug")

    engine = FakeEngine(transcript="hello world")
    feature, recorder, text_output, _, _ = make_feature(
        engine, np.ones(1600, dtype=np.float32), on_listening_changed=broken
    )
    feature.on_press()
    feature.on_release()
    assert recorder.stop_calls == 1
    assert text_output.delivered == ["hello world"]
