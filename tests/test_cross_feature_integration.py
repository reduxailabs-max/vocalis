"""Whole-system tests: do multiple Vocalis features behave correctly
TOGETHER in one process? Every other test file in this project
exercises one feature at a time, heavily faked. These tests reuse
cli.py's own private builder functions (never reinvented wiring) against
one shared ModelManager/ReadingCoordinator, with every external
dependency (microphone, speaker, clipboard, screen capture, OCR, the two
AI engine classes, typed output, desktop notifications) faked the same
way tests/test_cli.py already fakes them -- never real hardware, a real
GPU model, or the real network.
"""
import subprocess
import threading
from collections.abc import Iterator
from pathlib import Path
from types import MethodType
from typing import cast
from unittest.mock import patch

import numpy as np
import numpy.typing as npt
import pytest

from tests.fake_output import FakeAudioPlayer
from vocalis.cli import build_dictation, build_doc_reader, build_ocr_reader, build_quick_reader
from vocalis.config import VocalisConfig
from vocalis.features.reading_coordinator import ReadingCoordinator
from vocalis.model_manager import ModelManager, TTS_ENGINE_ID
from vocalis.output.typer import OutputResult
from vocalis.settings import LiveSettings


class _FakeEngine:
    """Fake ModelEngine (STT or TTS); records load/unload calls."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.load_count = 0
        self.unload_count = 0
        self._loaded = False

    def load(self) -> None:
        """Mark this fake engine loaded and count the call."""
        self.load_count += 1
        self._loaded = True

    def unload(self) -> None:
        """Mark this fake engine unloaded and count the call."""
        self.unload_count += 1
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Whether load() has run more recently than unload()."""
        return self._loaded

    def transcribe(self, audio: "npt.NDArray[np.float32]", sample_rate: int) -> str:
        """Fake transcription result, ignoring the input."""
        return "fake transcription"

    def synthesize(self, text: str) -> "tuple[npt.NDArray[np.float32], int]":
        """Fake synthesis result: a fixed 10-sample silent clip at 24kHz."""
        return np.zeros(10, dtype=np.float32), 24000


class _FakeRecorder:
    """Fake MicRecorder: start() does nothing, stop() returns fixed audio."""

    def start(self) -> None:
        """No-op: no real microphone is ever opened."""

    def stop(self) -> "npt.NDArray[np.float32]":
        """Return a fixed 100-sample buffer, never real audio."""
        return np.ones(100, dtype=np.float32)


class _FakePlayer:
    """Fake AudioPlayer: records play() calls and signals completion via
    a real threading.Event, so a test can wait() on it deterministically
    instead of sleeping -- on_trigger()/read_text() each run on a
    background thread, so this is the only reliable way to know the
    fake "playback" has actually finished."""

    def __init__(self) -> None:
        self.done_event = threading.Event()
        self.played: "list[tuple[npt.NDArray[np.float32], int]]" = []

    def play(self, audio: "npt.NDArray[np.float32]", sample_rate: int) -> None:
        """Record what was "played"."""
        self.played.append((audio, sample_rate))

    def wait_until_done(self) -> bool:
        """Signal done_event, then report a normal (non-stopped) finish."""
        self.done_event.set()
        return True

    def stop(self) -> None:
        """No-op: nothing real is playing to stop."""


class _FakeClipboardReader:
    """Fake ClipboardReader: always returns the same fixed text."""

    def read(self) -> str:
        """Return fixed clipboard text, never touching a real clipboard."""
        return "clipboard text"


class _FakePiper:
    """Fake PiperTTS: never checks for a real binary/voice file."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Accept any constructor arguments; store nothing real."""

    def load(self) -> None:
        """No-op: no real file-existence check."""

    def synthesize(self, text: str) -> "tuple[npt.NDArray[np.float32], int]":
        """Fake synthesis result, used only if the primary engine fails."""
        return np.zeros(5, dtype=np.float32), 22050


class _FakeTextOutput:
    """Fake TextOutput: never shells out to ydotool/xclip."""

    def deliver(self, text: str) -> "OutputResult":
        """Pretend the text was typed, without touching anything real."""
        return OutputResult(method="typed", text=text)


def _quiet_notify(self: object, title: str, message: str = "") -> None:
    """Stand-in for Notifier.notify: drops the notification, never runs notify-send."""


_FORBIDDEN_PROGRAMS = ("ydotool", "xclip", "notify-send")
_THREAD_JOIN_TIMEOUT_SECONDS = 2.0


def _finish_background_threads(started_before: "set[threading.Thread]") -> "list[threading.Thread]":
    """Wait for every thread started after `started_before` was taken to
    finish, and return any that are still running after the timeout.

    The readers send their final "Done" notification on a background
    thread AFTER the fake player reports playback finished -- so
    done_event.wait() alone does NOT mean the feature is finished. A test
    must call this before leaving the `with` block that fakes
    Notifier.notify, or that last notification escapes the fake.

    threading.Timer threads are skipped: they are ModelManager's idle-
    unload timers, which these tests set to 9999s so they never fire
    (and which only ever call a fake engine's unload() if they did).
    """

    def is_feature_worker(thread: threading.Thread) -> bool:
        return (
            thread not in started_before
            and thread is not threading.current_thread()
            and not isinstance(thread, threading.Timer)
        )

    for thread in threading.enumerate():
        if is_feature_worker(thread):
            thread.join(timeout=_THREAD_JOIN_TIMEOUT_SECONDS)
    return [t for t in threading.enumerate() if is_feature_worker(t) and t.is_alive()]


@pytest.fixture(autouse=True)
def _strong_idle_pc() -> Iterator[None]:
    """Every builder sees a big, idle PC, so Auto picks Qwen3-TTS and no real hardware is read."""
    from vocalis.hardware import GpuInfo, HardwareInfo

    info = HardwareInfo(GpuInfo("Test GPU", 8188, 7000), 16000, 10000, 8, 1.0)
    with patch("vocalis.cli.probe_hardware", return_value=info):
        yield


@pytest.fixture(autouse=True)
def _never_touch_real_hardware() -> "Iterator[None]":
    """Safety net for this whole file: no test here may ever shell out
    to a real ydotool/xclip/notify-send, even if a future test forgets
    to patch the class that would call it. A regression here would mean
    this test suite silently types real text and fires real desktop
    notifications on whoever's machine runs it -- this happened for
    real during this stage's own development, caught only by the final
    whole-branch review, not any per-task review. Raises loudly instead
    of touching real hardware.
    """
    real_run = subprocess.run

    def guarded_run(argv, *args, **kwargs):  # type: ignore[no-untyped-def]
        program = str(argv[0]) if isinstance(argv, (list, tuple)) and argv else str(argv)
        if any(name in program for name in _FORBIDDEN_PROGRAMS):
            raise AssertionError(
                f"tests/test_cross_feature_integration.py attempted a REAL subprocess "
                f"call to {program!r} -- a test here forgot to fake the class that "
                f"calls it. Patch vocalis.cli.TextOutput / vocalis.cli.Notifier.notify "
                f"(or whatever else this needs) instead of letting this call through."
            )
        return real_run(argv, *args, **kwargs)

    threads_before = set(threading.enumerate())
    with patch("subprocess.run", guarded_run), patch("vocalis.cli.pick_input_device", return_value=None):
        yield
        # Keep the guard active until every background thread this test
        # started has finished: a straggler thread finishing AFTER the
        # guard is removed could otherwise still reach real notify-send.
        stragglers = _finish_background_threads(threads_before)
    if stragglers:
        pytest.fail(f"background thread(s) outlived the test: {stragglers!r}")


def test_one_gpu_model_at_a_time_holds_across_features(qapp) -> None:  # type: ignore[no-untyped-def]
    """Dictate, then Quick Reader. ModelManager must unload the STT
    engine and load the TTS engine exactly once each -- proving the
    single-GPU-residency rule holds ACROSS features, not just within
    one feature's own already-tested local wiring."""
    config = VocalisConfig()
    manager = ModelManager(idle_timeout_seconds=9999.0)
    coordinator = ReadingCoordinator()
    parakeet_engine = _FakeEngine("parakeet")
    qwen_engine = _FakeEngine("qwen")

    threads_before = set(threading.enumerate())
    with (
        patch("vocalis.cli.ParakeetSTT", return_value=parakeet_engine),
        patch("vocalis.cli.MicRecorder", return_value=_FakeRecorder()),
        patch("vocalis.cli.Qwen3TTS", return_value=qwen_engine),
        patch("vocalis.cli.PiperTTS", _FakePiper),
        patch("vocalis.cli.AudioPlayer") as mock_player_cls,
        patch("vocalis.cli.ClipboardReader", return_value=_FakeClipboardReader()),
        patch("vocalis.cli.TextOutput", return_value=_FakeTextOutput()),
        patch("vocalis.cli.Notifier.notify", _quiet_notify),
    ):
        fake_player = _FakePlayer()
        mock_player_cls.return_value = fake_player

        dictation, _dictation_listener = build_dictation(config=config, model_manager=manager)
        quick_reader, _quick_listener = build_quick_reader(
            config=config, model_manager=manager, reading_coordinator=coordinator
        )

        dictation.on_press()
        dictation.on_release()
        assert manager.current_engine_id == "parakeet"
        assert parakeet_engine.load_count == 1
        assert parakeet_engine.unload_count == 0

        quick_reader.on_trigger()
        assert fake_player.done_event.wait(timeout=2.0)
        assert not _finish_background_threads(threads_before)
        assert manager.current_engine_id == TTS_ENGINE_ID
        assert parakeet_engine.unload_count == 1
        assert qwen_engine.load_count == 1


def test_switching_between_quick_reader_and_document_reader_reuses_the_loaded_engine(qapp) -> None:  # type: ignore[no-untyped-def]
    """Quick Reader and Document Reader share TTS_ENGINE_ID on purpose
    (see model_manager.py's own comment) -- switching between them must
    NOT reload the engine. This is the one case in Scenario 1 where "no
    reload happened" is the correct, desired behavior, not a bug."""
    config = VocalisConfig()
    manager = ModelManager(idle_timeout_seconds=9999.0)
    coordinator = ReadingCoordinator()
    qwen_engine = _FakeEngine("qwen")

    threads_before = set(threading.enumerate())
    with (
        patch("vocalis.cli.Qwen3TTS", return_value=qwen_engine),
        patch("vocalis.cli.PiperTTS", _FakePiper),
        patch("vocalis.cli.AudioPlayer") as mock_player_cls,
        patch("vocalis.cli.ClipboardReader", return_value=_FakeClipboardReader()),
        patch("vocalis.cli.Notifier.notify", _quiet_notify),
    ):
        first_player = _FakePlayer()
        mock_player_cls.return_value = first_player
        quick_reader, _quick_listener = build_quick_reader(
            config=config, model_manager=manager, reading_coordinator=coordinator
        )
        quick_reader.on_trigger()
        assert first_player.done_event.wait(timeout=2.0)
        assert qwen_engine.load_count == 1

        second_player = FakeAudioPlayer()
        mock_player_cls.return_value = second_player
        doc_reader, _popup, _doc_listener = build_doc_reader(
            config=config, model_manager=manager, reading_coordinator=coordinator
        )
        doc_reader.read_text("some document text")
        assert second_player.done_event.wait(timeout=2.0)
        assert not _finish_background_threads(threads_before)

    assert manager.current_engine_id == TTS_ENGINE_ID
    assert qwen_engine.load_count == 1  # NOT reloaded -- same engine id, same cached instance
    assert qwen_engine.unload_count == 0


def test_auto_picks_the_same_engine_for_both_readers_so_nothing_reloads(qapp) -> None:  # type: ignore[no-untyped-def]
    """With both readers on Auto they resolve to the same engine id, so
    switching from Quick Reader to Document Reader never reloads it."""
    config = VocalisConfig()
    assert config.engines.quick_reader_engine == config.engines.document_reader_engine == "auto"
    manager = ModelManager(idle_timeout_seconds=9999.0)
    coordinator = ReadingCoordinator()
    qwen_engine = _FakeEngine("qwen")

    threads_before = set(threading.enumerate())
    with (
        patch("vocalis.cli.Qwen3TTS", return_value=qwen_engine),
        patch("vocalis.cli.PiperTTS", _FakePiper),
        patch("vocalis.cli.AudioPlayer") as mock_player_cls,
        patch("vocalis.cli.ClipboardReader", return_value=_FakeClipboardReader()),
        patch("vocalis.cli.Notifier.notify", _quiet_notify),
    ):
        first_player = _FakePlayer()
        mock_player_cls.return_value = first_player
        quick_reader, _quick_listener = build_quick_reader(
            config=config, model_manager=manager, reading_coordinator=coordinator
        )
        quick_reader.on_trigger()
        assert first_player.done_event.wait(timeout=2.0)
        assert manager.current_engine_id == "qwen3_tts"

        second_player = FakeAudioPlayer()
        mock_player_cls.return_value = second_player
        doc_reader, _popup, _doc_listener = build_doc_reader(
            config=config, model_manager=manager, reading_coordinator=coordinator
        )
        doc_reader.read_text("some document text")
        assert second_player.done_event.wait(timeout=2.0)
        assert not _finish_background_threads(threads_before)

    assert manager.current_engine_id == "qwen3_tts"
    assert qwen_engine.load_count == 1
    assert qwen_engine.unload_count == 0


def test_hotkeys_never_cross_trigger_another_features_handler(qapp) -> None:  # type: ignore[no-untyped-def]
    """All four shortcut listeners, built together against one shared
    ModelManager, must each be bound to their OWN feature's trigger
    method -- never another feature's."""
    config = VocalisConfig()
    manager = ModelManager(idle_timeout_seconds=9999.0)
    coordinator = ReadingCoordinator()

    with (
        patch("vocalis.cli.ParakeetSTT", return_value=_FakeEngine("p")),
        patch("vocalis.cli.MicRecorder", return_value=_FakeRecorder()),
        patch("vocalis.cli.Qwen3TTS", return_value=_FakeEngine("q")),
        patch("vocalis.cli.PiperTTS", _FakePiper),
        patch("vocalis.cli.AudioPlayer", return_value=_FakePlayer()),
        patch("vocalis.cli.ClipboardReader", return_value=_FakeClipboardReader()),
        patch("vocalis.cli.ClipboardWriter"),
        patch("vocalis.cli.ScreenRegionCapture"),
        patch("vocalis.cli.TesseractOCR"),
        patch("vocalis.cli.list_numlock_capable_devices", return_value=[]),
        patch("vocalis.cli.list_hotkey_capable_devices", return_value=[]),
        patch("vocalis.cli.list_devices_with_key", return_value=[]),
        patch("vocalis.cli.Notifier.notify", _quiet_notify),
    ):
        dictation, dictation_listener = build_dictation(config=config, model_manager=manager)
        quick_reader, quick_listener = build_quick_reader(
            config=config, model_manager=manager, reading_coordinator=coordinator
        )
        doc_reader, popup, doc_listener = build_doc_reader(
            config=config, model_manager=manager, reading_coordinator=coordinator
        )
        ocr_reader, ocr_listener = build_ocr_reader(
            config=config, document_reader=doc_reader, popup=popup
        )

        assert dictation_listener._on_press == dictation.on_press  # noqa: SLF001
        assert dictation_listener._on_release == dictation.on_release  # noqa: SLF001
        assert quick_listener._on_trigger == quick_reader.on_trigger  # noqa: SLF001
        assert doc_listener._on_trigger == popup.hotkey_pressed.emit  # noqa: SLF001
        assert ocr_listener._on_trigger == ocr_reader.on_trigger  # noqa: SLF001

        bound_callback_owners = [
            cast(MethodType, dictation_listener._on_press).__self__,  # noqa: SLF001
            cast(MethodType, quick_listener._on_trigger).__self__,  # noqa: SLF001
            cast(MethodType, ocr_listener._on_trigger).__self__,  # noqa: SLF001
        ]
        assert len(set(id(owner) for owner in bound_callback_owners)) == 3

        # Each listener must also watch its OWN key: two features sharing
        # one key code would cross-trigger even with correct callbacks.
        listener_key_codes = [
            dictation_listener._key_code,  # noqa: SLF001
            quick_listener._key_code,  # noqa: SLF001
            doc_listener._key_code,  # noqa: SLF001
            ocr_listener._key_code,  # noqa: SLF001
        ]
        assert listener_key_codes == [69, 19, 32, 24]  # NUMLOCK, R, D, O -- cli.py's real defaults
        assert len(set(listener_key_codes)) == 4


def test_a_settings_change_is_picked_up_by_an_already_built_readers_next_reading(  # type: ignore[no-untyped-def]
    qapp, tmp_path: Path
) -> None:
    """Build Document Reader once, change speaking_speed via
    LiveSettings.update() (the same path the real Settings window
    uses), then read TWICE with the SAME already-built reader instance
    -- proving a live setting change reaches a reader that already
    exists, not just one built freshly after the change (which earlier
    test code accidentally always did, masking this exact kind of
    regression)."""
    config = VocalisConfig()
    manager = ModelManager(idle_timeout_seconds=9999.0)
    coordinator = ReadingCoordinator()
    settings = LiveSettings(config, tmp_path / "config.toml", saver=lambda cfg, path: None)

    fixed_audio_length = 24000  # 1 second at 24kHz, long enough to stretch measurably
    qwen_engine = _FakeEngine("qwen")

    def fixed_length_synthesize(text: str) -> "tuple[npt.NDArray[np.float32], int]":
        return np.full(fixed_audio_length, 0.5, dtype=np.float32), 24000

    qwen_engine.synthesize = fixed_length_synthesize  # type: ignore[method-assign]

    threads_before = set(threading.enumerate())
    with (
        patch("vocalis.cli.Qwen3TTS", return_value=qwen_engine),
        patch("vocalis.cli.PiperTTS", _FakePiper),
        patch("vocalis.cli.AudioPlayer") as mock_player_cls,
        patch("vocalis.cli.Notifier.notify", _quiet_notify),
    ):
        fake_player = FakeAudioPlayer()
        mock_player_cls.return_value = fake_player

        settings.update(config.model_copy(update={"speaking_speed": 1.6}))
        doc_reader, _popup, _listener = build_doc_reader(
            config=config, model_manager=manager, reading_coordinator=coordinator, settings=settings
        )

        doc_reader.read_text("some document text")
        assert fake_player.done_event.wait(timeout=2.0)
        assert not _finish_background_threads(threads_before)  # first reading fully over
        fast_length = len(fake_player.output.played())
        fake_player.done_event.clear()

        settings.update(config.model_copy(update={"speaking_speed": 0.7}))
        doc_reader.read_text("some document text")  # SAME reader instance, not rebuilt
        assert fake_player.done_event.wait(timeout=2.0)
        assert not _finish_background_threads(threads_before)

    assert mock_player_cls.call_count == 1  # proves the reader really was built only once
    assert fake_player.output.opened == 2  # one sound stream per reading
    slow_length = len(fake_player.output.played()) - fast_length
    assert fast_length < fixed_audio_length < slow_length


def test_notifications_from_two_different_features_both_reach_the_shared_notifier(  # type: ignore[no-untyped-def]
    qapp,
) -> None:
    """Dictate, then Quick Reader, in sequence. Every notification from
    BOTH features must reach the shared desktop-notification path
    (Notifier.notify -- each builder makes its own Notifier instance, but
    they all send through this one method), in order, un-garbled by the
    other feature's notifications."""
    config = VocalisConfig()
    manager = ModelManager(idle_timeout_seconds=9999.0)
    coordinator = ReadingCoordinator()
    sent: "list[tuple[str, str]]" = []

    def fake_notify(self: object, title: str, message: str = "") -> None:  # noqa: ANN001
        sent.append((title, message))

    threads_before = set(threading.enumerate())
    with (
        patch("vocalis.cli.ParakeetSTT", return_value=_FakeEngine("p")),
        patch("vocalis.cli.MicRecorder", return_value=_FakeRecorder()),
        patch("vocalis.cli.Qwen3TTS", return_value=_FakeEngine("q")),
        patch("vocalis.cli.PiperTTS", _FakePiper),
        patch("vocalis.cli.AudioPlayer") as mock_player_cls,
        patch("vocalis.cli.ClipboardReader", return_value=_FakeClipboardReader()),
        patch("vocalis.cli.Notifier.notify", fake_notify),
        patch("vocalis.cli.TextOutput", return_value=_FakeTextOutput()),
    ):
        fake_player = _FakePlayer()
        mock_player_cls.return_value = fake_player
        dictation, _dictation_listener = build_dictation(config=config, model_manager=manager)
        quick_reader, _quick_listener = build_quick_reader(
            config=config, model_manager=manager, reading_coordinator=coordinator
        )

        dictation.on_press()
        dictation.on_release()
        quick_reader.on_trigger()
        assert fake_player.done_event.wait(timeout=2.0)
        assert not _finish_background_threads(threads_before)

    # dictation.py's _TITLE == "Vocalis — Dictation", quick_reader.py's
    # _TITLE == "Vocalis — Quick Reader". The exact, complete sequence:
    # dictation's four notifications all first, then quick reader's three.
    assert sent == [
        ("Vocalis — Dictation", "Listening..."),
        ("Vocalis — Dictation", "Transcribing..."),
        ("Vocalis — Dictation", "fake transcription"),
        ("Vocalis — Dictation", "Done"),
        ("Vocalis — Quick Reader", "Reading..."),
        ("Vocalis — Quick Reader", "clipboard text"),
        ("Vocalis — Quick Reader", "Done"),
    ]


def test_full_lifecycle_two_features_then_clean_shutdown(qapp) -> None:  # type: ignore[no-untyped-def]
    """Build dictation and quick reader together (mirroring main()'s own
    construction order), have each handle one request, then shut the
    shared ModelManager down -- the same call main()'s daemon.stop()
    makes on the way out. The engine must end up unloaded and
    current_engine_id cleared, regardless of which feature used it last."""
    config = VocalisConfig()
    manager = ModelManager(idle_timeout_seconds=9999.0)
    coordinator = ReadingCoordinator()
    parakeet_engine = _FakeEngine("parakeet")
    qwen_engine = _FakeEngine("qwen")

    threads_before = set(threading.enumerate())
    with (
        patch("vocalis.cli.ParakeetSTT", return_value=parakeet_engine),
        patch("vocalis.cli.MicRecorder", return_value=_FakeRecorder()),
        patch("vocalis.cli.Qwen3TTS", return_value=qwen_engine),
        patch("vocalis.cli.PiperTTS", _FakePiper),
        patch("vocalis.cli.AudioPlayer") as mock_player_cls,
        patch("vocalis.cli.ClipboardReader", return_value=_FakeClipboardReader()),
        patch("vocalis.cli.TextOutput", return_value=_FakeTextOutput()),
        patch("vocalis.cli.Notifier.notify", _quiet_notify),
    ):
        fake_player = _FakePlayer()
        mock_player_cls.return_value = fake_player
        dictation, _dictation_listener = build_dictation(config=config, model_manager=manager)
        quick_reader, _quick_listener = build_quick_reader(
            config=config, model_manager=manager, reading_coordinator=coordinator
        )

        dictation.on_press()
        dictation.on_release()
        quick_reader.on_trigger()
        assert fake_player.done_event.wait(timeout=2.0)
        assert not _finish_background_threads(threads_before)

    assert manager.current_engine_id == TTS_ENGINE_ID  # still resident, nothing has released it yet

    manager.shutdown()

    assert manager.current_engine_id is None
    assert qwen_engine.unload_count == 1
