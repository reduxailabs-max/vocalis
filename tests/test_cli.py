"""Tests for vocalis.cli wiring: config -> logging -> ModelManager -> daemon
-> dictation feature."""
import logging
import signal
import threading
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vocalis.cli import (
    _SettingsBridge,
    build_daemon,
    build_dictation,
    build_doc_reader,
    build_ocr_reader,
    build_quick_reader,
    main,
)
from vocalis.config import VocalisConfig
from vocalis.daemon import DaemonAlreadyRunningError
from vocalis.remote import SHOW_SETTINGS
from vocalis.features.reading_coordinator import ReadingCoordinator
from vocalis.model_manager import ModelManager


@pytest.fixture(autouse=True)
def _restore_quit_on_last_window_closed():  # type: ignore[no-untyped-def]
    """main() switches quitOnLastWindowClosed off; put it back so tests stay order-independent."""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    original = QApplication.quitOnLastWindowClosed() if app is not None else None
    yield
    if QApplication.instance() is not None and original is not None:
        QApplication.setQuitOnLastWindowClosed(original)


@pytest.fixture(autouse=True)
def _no_real_startup_checks():  # type: ignore[no-untyped-def]
    """Keep every main()-calling test immune to real model/ydotoold/PATH
    state: run_startup_checks is patched to report nothing by default.
    Tests that want to exercise real messages still patch it themselves
    with their own `with patch(...)` block, which overrides this one for
    the duration of that `with`.
    """
    with patch("vocalis.cli.run_startup_checks", return_value=[]):
        yield


def test_build_daemon_uses_the_given_model_manager(tmp_path: Path) -> None:
    manager = ModelManager(idle_timeout_seconds=12.5)
    socket_path = tmp_path / "vocalis.sock"

    daemon = build_daemon(model_manager=manager, socket_path=socket_path)

    assert daemon._model_manager is manager  # noqa: SLF001
    assert daemon._socket_path == socket_path  # noqa: SLF001


@pytest.fixture(autouse=True)
def _no_real_audio_device_listing() -> Iterator[None]:
    with patch("vocalis.cli.pick_input_device", return_value=None):
        yield


@pytest.fixture(autouse=True)
def _strong_idle_pc() -> Iterator[None]:
    """Every builder sees a big, idle PC, so Auto picks Qwen3-TTS and no real hardware is read."""
    from vocalis.hardware import GpuInfo, HardwareInfo

    info = HardwareInfo(GpuInfo("Test GPU", 8188, 7000), 16000, 10000, 8, 1.0)
    with patch("vocalis.cli.probe_hardware", return_value=info):
        yield


@pytest.fixture(autouse=True)
def _no_real_hardware_probe() -> Iterator[None]:
    with patch("vocalis.cli.log_hardware_summary"):
        yield


def test_build_dictation_wires_engine_factory_to_configured_model_dir() -> None:
    config = VocalisConfig()
    manager = ModelManager()

    feature, listener = build_dictation(config=config, model_manager=manager)

    engine = feature._engine_factory()  # noqa: SLF001
    assert str(engine._model_dir) == config.engines.parakeet_model_dir  # type: ignore[attr-defined]  # noqa: SLF001
    assert listener._on_press == feature.on_press  # noqa: SLF001
    assert listener._on_release == feature.on_release  # noqa: SLF001


def test_main_starts_and_stops_daemon_on_keyboard_interrupt(tmp_path: Path, qapp) -> None:  # type: ignore[no-untyped-def]
    socket_path = tmp_path / "vocalis.sock"

    with (
        patch("vocalis.cli.default_config_path", return_value=tmp_path / "config.toml"),
        patch("vocalis.cli._socket_path", return_value=socket_path),
        patch("vocalis.cli.list_numlock_capable_devices", return_value=[]),
        patch("vocalis.cli.list_hotkey_capable_devices", return_value=[]),
        patch("vocalis.cli.list_devices_with_key", return_value=[]),
        patch("vocalis.cli.QApplication.exec", side_effect=KeyboardInterrupt),
    ):
        from vocalis.cli import main

        main()

    assert not socket_path.exists()


def test_main_exits_quietly_when_another_instance_is_already_running(  # type: ignore[no-untyped-def]
    qapp, caplog
) -> None:
    """A refused second launch must log at INFO and return normally -- no
    traceback, no non-zero exit -- instead of letting the exception
    propagate (which would count as a crash against systemd's
    StartLimitBurst for no good reason)."""
    mock_daemon = MagicMock()
    mock_daemon.start.side_effect = DaemonAlreadyRunningError(
        "Another vocalis instance is already running on /tmp/vocalis.sock"
    )

    with (
        patch("vocalis.cli.list_numlock_capable_devices", return_value=[]),
        patch("vocalis.cli.list_hotkey_capable_devices", return_value=[]),
        patch("vocalis.cli.list_devices_with_key", return_value=[]),
        patch("vocalis.cli.load_config", return_value=VocalisConfig()),
        patch("vocalis.cli.VocalisDaemon", return_value=mock_daemon),
        caplog.at_level(logging.INFO, logger="vocalis"),
    ):
        main()  # must return normally; a propagating exception fails this test

    assert "already running" in caplog.text.lower()
    mock_daemon.stop.assert_called_once()


def test_build_quick_reader_returns_feature_and_listener() -> None:
    with patch("vocalis.cli.list_hotkey_capable_devices", return_value=[]):
        config = VocalisConfig()
        manager = ModelManager()
        feature, listener = build_quick_reader(
            config=config, model_manager=manager, reading_coordinator=ReadingCoordinator()
        )
        assert feature is not None
        assert listener is not None


def test_main_starts_and_stops_quick_reader_listener(qapp) -> None:  # type: ignore[no-untyped-def]
    with (
        patch("vocalis.cli.list_numlock_capable_devices", return_value=[]),
        patch("vocalis.cli.list_hotkey_capable_devices", return_value=[]),
        patch("vocalis.cli.load_config", return_value=VocalisConfig()),
        patch("vocalis.cli.VocalisDaemon") as mock_daemon_cls,
        patch("vocalis.cli.list_devices_with_key", return_value=[]),
        patch("vocalis.cli.QApplication.exec", side_effect=KeyboardInterrupt),
    ):
        main()
        mock_daemon_cls.return_value.start.assert_called_once()
        mock_daemon_cls.return_value.stop.assert_called_once()


def test_build_quick_reader_accepts_a_shared_reading_coordinator() -> None:
    config = VocalisConfig()
    manager = ModelManager()
    coordinator = ReadingCoordinator()

    feature, _listener = build_quick_reader(
        config=config, model_manager=manager, reading_coordinator=coordinator
    )

    assert feature._reading_coordinator is coordinator  # noqa: SLF001


class _StopRecorder:
    """Fake feature that is always 'reading' and records which thread stop() ran on."""

    is_reading = True

    def __init__(self) -> None:
        self.stop_threads: list[object] = []

    def stop(self) -> None:
        self.stop_threads.append(threading.current_thread())


def test_build_doc_reader_wires_engines_popup_and_listener(qtbot) -> None:  # type: ignore[no-untyped-def]
    config = VocalisConfig()
    manager = ModelManager()
    coordinator = ReadingCoordinator()

    feature, popup, listener = build_doc_reader(
        config=config, model_manager=manager, reading_coordinator=coordinator
    )
    qtbot.addWidget(popup)

    assert feature._reading_coordinator is coordinator  # noqa: SLF001

    # The hotkey callback must reach the popup's slot (behavioural check:
    # a signal's .emit is a fresh object per access, so it is not compared).
    popup._feature = _StopRecorder()  # type: ignore[assignment]  # noqa: SLF001
    listener._on_trigger()  # noqa: SLF001
    qtbot.waitUntil(lambda: popup._feature.stop_threads != [])  # type: ignore[attr-defined]  # noqa: SLF001

    # The on_status/on_progress/on_finished callbacks wired to the popup's
    # signals must actually reach the popup's widgets, proving the
    # construction-order workaround (feature built before popup exists,
    # popup referenced by closure) really works end-to-end.
    feature._on_status("Reading...")  # noqa: SLF001
    assert popup._status_label.text() == "Reading..."  # noqa: SLF001


def test_main_installs_sigint_handler_that_quits_qt_and_restores_it(qapp) -> None:  # type: ignore[no-untyped-def]
    original_handler = signal.getsignal(signal.SIGINT)
    seen: dict[str, object] = {}

    def fake_exec(_self: object) -> int:
        handler = signal.getsignal(signal.SIGINT)
        seen["handler"] = handler
        assert callable(handler)
        assert handler is not original_handler
        handler(signal.SIGINT, None)
        return 0

    with (
        patch("vocalis.cli.list_numlock_capable_devices", return_value=[]),
        patch("vocalis.cli.list_hotkey_capable_devices", return_value=[]),
        patch("vocalis.cli.list_devices_with_key", return_value=[]),
        patch("vocalis.cli.load_config", return_value=VocalisConfig()),
        patch("vocalis.cli.VocalisDaemon"),
        patch("vocalis.cli.QApplication.exec", fake_exec),
        patch("vocalis.cli.QApplication.quit") as mock_quit,
    ):
        main()

    assert "handler" in seen
    mock_quit.assert_called_once()
    assert signal.getsignal(signal.SIGINT) is original_handler


def test_main_installs_sigterm_handler_that_quits_qt_and_restores_it(qapp) -> None:  # type: ignore[no-untyped-def]
    original_handler = signal.getsignal(signal.SIGTERM)
    seen: dict[str, object] = {}

    def fake_exec(_self: object) -> int:
        handler = signal.getsignal(signal.SIGTERM)
        seen["handler"] = handler
        assert callable(handler)
        assert handler is not original_handler
        handler(signal.SIGTERM, None)
        return 0

    with (
        patch("vocalis.cli.list_numlock_capable_devices", return_value=[]),
        patch("vocalis.cli.list_hotkey_capable_devices", return_value=[]),
        patch("vocalis.cli.list_devices_with_key", return_value=[]),
        patch("vocalis.cli.load_config", return_value=VocalisConfig()),
        patch("vocalis.cli.VocalisDaemon"),
        patch("vocalis.cli.QApplication.exec", fake_exec),
        patch("vocalis.cli.QApplication.quit") as mock_quit,
    ):
        main()

    assert "handler" in seen
    mock_quit.assert_called_once()
    assert signal.getsignal(signal.SIGTERM) is original_handler


def test_hotkey_emitted_from_another_thread_runs_on_the_gui_thread(qtbot) -> None:  # type: ignore[no-untyped-def]
    with patch("vocalis.cli.list_devices_with_key", return_value=[]):
        _feature, popup, listener = build_doc_reader(
            config=VocalisConfig(),
            model_manager=ModelManager(),
            reading_coordinator=ReadingCoordinator(),
        )
    qtbot.addWidget(popup)
    recorder = _StopRecorder()
    popup._feature = recorder  # type: ignore[assignment]  # noqa: SLF001

    emitter = threading.Thread(target=listener._on_trigger)  # noqa: SLF001
    emitter.start()
    emitter.join()
    qtbot.waitUntil(lambda: recorder.stop_threads != [])

    assert recorder.stop_threads == [threading.main_thread()]
    assert emitter is not threading.main_thread()


def test_main_restores_sigint_handler_when_the_keepalive_timer_fails(qapp) -> None:  # type: ignore[no-untyped-def]
    original_handler = signal.getsignal(signal.SIGINT)
    failing_timer = MagicMock()
    failing_timer.start.side_effect = RuntimeError("timer broke")

    with (
        patch("vocalis.cli.list_numlock_capable_devices", return_value=[]),
        patch("vocalis.cli.list_hotkey_capable_devices", return_value=[]),
        patch("vocalis.cli.list_devices_with_key", return_value=[]),
        patch("vocalis.cli.load_config", return_value=VocalisConfig()),
        patch("vocalis.cli.VocalisDaemon") as mock_daemon_cls,
        patch("vocalis.cli.QTimer", return_value=failing_timer),
        pytest.raises(RuntimeError, match="timer broke"),
    ):
        main()

    assert signal.getsignal(signal.SIGINT) is original_handler
    mock_daemon_cls.return_value.stop.assert_called_once()


def test_main_starts_all_four_listeners_then_stops_them_in_reverse_order(qapp) -> None:  # type: ignore[no-untyped-def]
    calls = MagicMock()
    dictation, quick, doc, ocr = calls.dictation, calls.quick, calls.doc, calls.ocr

    with (
        patch("vocalis.cli.load_config", return_value=VocalisConfig()),
        patch("vocalis.cli.VocalisDaemon", return_value=calls.daemon),
        patch("vocalis.cli.build_dictation", return_value=(MagicMock(), dictation)),
        patch("vocalis.cli.build_quick_reader", return_value=(MagicMock(), quick)),
        patch("vocalis.cli.build_doc_reader", return_value=(MagicMock(), MagicMock(), doc)),
        patch("vocalis.cli.build_ocr_reader", return_value=(MagicMock(), ocr)),
        patch("vocalis.cli.QApplication.exec", side_effect=KeyboardInterrupt),
    ):
        main()

    names = [c[0] for c in calls.mock_calls if c[0] in {
        "daemon.start", "dictation.start", "quick.start", "doc.start", "ocr.start",
        "ocr.stop", "doc.stop", "quick.stop", "dictation.stop", "daemon.stop",
    }]
    assert names == [
        "daemon.start", "dictation.start", "quick.start", "doc.start", "ocr.start",
        "ocr.stop", "doc.stop", "quick.stop", "dictation.stop", "daemon.stop",
    ]


def test_main_attempts_every_stop_in_order_even_if_a_listener_stop_raises(qapp) -> None:  # type: ignore[no-untyped-def]
    calls = MagicMock()
    dictation, quick, doc, ocr = calls.dictation, calls.quick, calls.doc, calls.ocr
    quick.stop.side_effect = RuntimeError("quick reader stop failed")

    with (
        patch("vocalis.cli.load_config", return_value=VocalisConfig()),
        patch("vocalis.cli.VocalisDaemon", return_value=calls.daemon),
        patch("vocalis.cli.build_dictation", return_value=(MagicMock(), dictation)),
        patch("vocalis.cli.build_quick_reader", return_value=(MagicMock(), quick)),
        patch("vocalis.cli.build_doc_reader", return_value=(MagicMock(), MagicMock(), doc)),
        patch("vocalis.cli.build_ocr_reader", return_value=(MagicMock(), ocr)),
        patch("vocalis.cli.QApplication.exec", side_effect=KeyboardInterrupt),
    ):
        main()  # must not raise

    stops = [c[0] for c in calls.mock_calls if c[0].endswith(".stop")]
    assert stops == ["ocr.stop", "doc.stop", "quick.stop", "dictation.stop", "daemon.stop"]


def test_build_ocr_reader_shares_the_document_reader_and_listens_for_ctrl_alt_o(qtbot) -> None:  # type: ignore[no-untyped-def]
    config = VocalisConfig()
    manager = ModelManager()
    coordinator = ReadingCoordinator()
    doc_feature, popup, _doc_listener = build_doc_reader(
        config=config, model_manager=manager, reading_coordinator=coordinator
    )
    qtbot.addWidget(popup)

    feature, listener = build_ocr_reader(config=config, document_reader=doc_feature, popup=popup)

    assert feature._reader is doc_feature  # noqa: SLF001
    assert listener._key_code == 24  # noqa: SLF001  # evdev KEY_O

    # The "reading is starting" hook must reach the popup so Pause/Stop work.
    hook = feature._on_reading_starting  # noqa: SLF001
    assert hook is not None
    hook("Hello from the screen")
    assert popup._text_box.toPlainText() == "Hello from the screen"  # noqa: SLF001
    assert popup._stop_button.isEnabled() is True  # noqa: SLF001


def test_main_starts_and_stops_the_ocr_listener(tmp_path: Path, qapp) -> None:  # type: ignore[no-untyped-def]
    with (
        patch("vocalis.cli.default_config_path", return_value=tmp_path / "config.toml"),
        patch("vocalis.cli._socket_path", return_value=tmp_path / "vocalis.sock"),
        patch("vocalis.cli.list_numlock_capable_devices", return_value=[]),
        patch("vocalis.cli.list_hotkey_capable_devices", return_value=[]),
        patch("vocalis.cli.list_devices_with_key", return_value=[]),
        patch("vocalis.cli.build_ocr_reader") as mock_build_ocr,
        patch("vocalis.cli.QApplication.exec", side_effect=KeyboardInterrupt),
    ):
        listener = MagicMock()
        mock_build_ocr.return_value = (MagicMock(), listener)
        main()

    listener.start.assert_called_once_with()
    listener.stop.assert_called_once_with()


def test_main_connects_the_bridges_show_settings_signal_to_settings_window_present(  # type: ignore[no-untyped-def]
    tmp_path: Path, qapp
):
    """If main() ever drops `bridge.show_settings.connect(settings_window.present)`,
    this must fail: it captures the real _SettingsBridge main() builds, emits its
    show_settings signal (direct call, same thread, no event loop needed), and
    checks SettingsWindow.present actually ran."""
    captured_bridges: list[_SettingsBridge] = []

    class CapturingBridge(_SettingsBridge):
        def __init__(self) -> None:
            super().__init__()
            captured_bridges.append(self)

    def exec_side_effect() -> None:
        captured_bridges[0].show_settings.emit()
        raise KeyboardInterrupt

    with (
        patch("vocalis.cli.default_config_path", return_value=tmp_path / "config.toml"),
        patch("vocalis.cli._socket_path", return_value=tmp_path / "vocalis.sock"),
        patch("vocalis.cli.list_numlock_capable_devices", return_value=[]),
        patch("vocalis.cli.list_hotkey_capable_devices", return_value=[]),
        patch("vocalis.cli.list_devices_with_key", return_value=[]),
        patch("vocalis.cli._SettingsBridge", CapturingBridge),
        patch("vocalis.cli.SettingsWindow.present") as mock_present,
        patch("vocalis.cli.QApplication.exec", side_effect=exec_side_effect),
    ):
        main()

    mock_present.assert_called_once_with()


@pytest.mark.parametrize("exec_side_effect", [None, KeyboardInterrupt])
def test_main_stops_and_deletes_the_keepalive_timer_on_exit(  # type: ignore[no-untyped-def]
    tmp_path: Path, qapp, exec_side_effect
) -> None:
    with (
        patch("vocalis.cli.default_config_path", return_value=tmp_path / "config.toml"),
        patch("vocalis.cli._socket_path", return_value=tmp_path / "vocalis.sock"),
        patch("vocalis.cli.list_numlock_capable_devices", return_value=[]),
        patch("vocalis.cli.list_hotkey_capable_devices", return_value=[]),
        patch("vocalis.cli.list_devices_with_key", return_value=[]),
        patch("vocalis.cli.QApplication.exec", side_effect=exec_side_effect),
        patch("vocalis.cli.QTimer") as mock_timer_cls,
    ):
        main()

    timer = mock_timer_cls.return_value
    timer.start.assert_called_once_with(200)
    timer.stop.assert_called_once_with()
    timer.deleteLater.assert_called_once_with()


# ---- Task 9: configurable shortcuts and live settings ---------------------

import logging  # noqa: E402
from typing import Callable, cast  # noqa: E402

from vocalis.config import ShortcutsConfig  # noqa: E402
from vocalis.cli import resolve_shortcuts  # noqa: E402
from vocalis.engines.qwen3_tts import Qwen3TTS  # noqa: E402
from vocalis.engines.kokoro_tts import KokoroTTS  # noqa: E402
from vocalis.hotkeys.shortcut_parser import DEFAULT_SHORTCUTS  # noqa: E402
from vocalis.settings import LiveSettings  # noqa: E402
from vocalis.ui.status_hub import StatusHub  # noqa: E402


def _live_settings(tmp_path: Path) -> LiveSettings:
    """LiveSettings whose saver does nothing and whose path is under tmp_path."""
    return LiveSettings(VocalisConfig(), tmp_path / "config.toml", saver=lambda _c, _p: None)


class _HubRecorder:
    """Records every signal a StatusHub emits."""

    def __init__(self, hub: StatusHub) -> None:
        self.events: list[tuple[object, ...]] = []
        hub.reading_started.connect(lambda s, p: self.events.append(("started", s, p)))
        hub.reading_finished.connect(lambda: self.events.append(("finished",)))
        hub.paused_changed.connect(lambda p: self.events.append(("paused", p)))
        hub.progress_changed.connect(lambda c, t: self.events.append(("progress", c, t)))
        hub.status_changed.connect(lambda m: self.events.append(("status", m)))


def test_resolve_shortcuts_returns_defaults_for_a_bad_config(caplog) -> None:  # type: ignore[no-untyped-def]
    bad = VocalisConfig(shortcuts=ShortcutsConfig(ocr_reader="Ctrl+Alt+Banana"))
    with caplog.at_level(logging.ERROR, logger="vocalis"):
        result = resolve_shortcuts(bad)
    assert result == DEFAULT_SHORTCUTS
    assert len([r for r in caplog.records if r.levelno == logging.ERROR]) == 1

    good = VocalisConfig(shortcuts=ShortcutsConfig(ocr_reader="Ctrl+Alt+P"))
    assert resolve_shortcuts(good).ocr_reader == 25


def test_build_dictation_uses_live_notifications_and_the_given_key(tmp_path: Path) -> None:
    settings = _live_settings(tmp_path)
    feature, listener = build_dictation(
        config=settings.config, model_manager=ModelManager(), settings=settings, key_code=70
    )
    enabled = cast(Callable[[], bool], feature._notifier._enabled)  # noqa: SLF001
    assert enabled() is True
    settings.update(VocalisConfig(notifications_enabled=False))
    assert enabled() is False
    assert listener._key_code == 70  # noqa: SLF001


def test_primary_engine_reads_the_live_voice(tmp_path: Path, qtbot) -> None:  # type: ignore[no-untyped-def]
    settings = _live_settings(tmp_path)
    manager = ModelManager()
    with patch("vocalis.cli.list_hotkey_capable_devices", return_value=[]):
        quick, _ = build_quick_reader(
            config=settings.config, model_manager=manager,
            reading_coordinator=ReadingCoordinator(), settings=settings,
        )
    doc, popup, _ = build_doc_reader(
        config=settings.config, model_manager=manager,
        reading_coordinator=ReadingCoordinator(), settings=settings,
    )
    qtbot.addWidget(popup)
    settings.update(
        VocalisConfig(engines=settings.config.engines.model_copy(update={"qwen3_tts_speaker": "Dylan"}))
    )
    for feature in (quick, doc):
        engine = feature._primary_engine_factory()  # noqa: SLF001
        assert isinstance(engine, Qwen3TTS)
        assert engine._current_speaker() == "Dylan"  # noqa: SLF001


def test_quick_reader_reports_to_the_hub(qapp) -> None:  # type: ignore[no-untyped-def]
    hub = StatusHub()
    seen = _HubRecorder(hub)
    with patch("vocalis.cli.list_hotkey_capable_devices", return_value=[]):
        feature, _ = build_quick_reader(
            config=VocalisConfig(), model_manager=ModelManager(),
            reading_coordinator=ReadingCoordinator(), hub=hub,
        )
    hook = feature._on_reading_changed  # noqa: SLF001
    assert hook is not None
    hook(True)
    assert seen.events == [("started", "Quick read", False)]
    assert hub.is_reading
    hook(False)
    assert not hub.is_reading
    assert seen.events[-1] == ("finished",)


def test_doc_reader_reports_status_progress_finish_and_pause_to_the_hub(qtbot) -> None:  # type: ignore[no-untyped-def]
    hub = StatusHub()
    seen = _HubRecorder(hub)
    feature, popup, _ = build_doc_reader(
        config=VocalisConfig(), model_manager=ModelManager(),
        reading_coordinator=ReadingCoordinator(), hub=hub,
    )
    qtbot.addWidget(popup)

    feature._on_started()  # type: ignore[misc]  # noqa: SLF001
    feature._on_status("Paused")  # noqa: SLF001
    feature._on_progress(2, 5)  # noqa: SLF001
    feature._on_status("Resuming...")  # noqa: SLF001
    feature._on_finished()  # noqa: SLF001

    assert seen.events[0] == ("started", "Document", True)
    assert ("paused", True) in seen.events
    assert ("progress", 2, 5) in seen.events
    assert seen.events.index(("paused", True)) < seen.events.index(("paused", False))
    assert seen.events[-1] == ("finished",)
    assert popup._status_label.text() == "Resuming..."  # noqa: SLF001
    assert popup._progress_label.text() != ""  # noqa: SLF001


def test_doc_reader_start_enables_popup_buttons_even_without_a_hub(qtbot) -> None:  # type: ignore[no-untyped-def]
    feature, popup, _ = build_doc_reader(
        config=VocalisConfig(), model_manager=ModelManager(), reading_coordinator=ReadingCoordinator(),
    )
    qtbot.addWidget(popup)
    assert popup._pause_button.isEnabled() is False  # noqa: SLF001

    feature._on_started()  # type: ignore[misc]  # noqa: SLF001

    assert popup._pause_button.isEnabled() is True  # noqa: SLF001
    assert popup._stop_button.isEnabled() is True  # noqa: SLF001


def test_ocr_shows_the_widget_not_the_popup_when_the_widget_is_enabled(tmp_path: Path, qtbot) -> None:  # type: ignore[no-untyped-def]
    hub = StatusHub()
    seen = _HubRecorder(hub)
    settings = _live_settings(tmp_path)
    doc, popup, _ = build_doc_reader(
        config=settings.config, model_manager=ModelManager(),
        reading_coordinator=ReadingCoordinator(), settings=settings, hub=hub,
    )
    qtbot.addWidget(popup)
    feature, _ = build_ocr_reader(
        config=settings.config, document_reader=doc, popup=popup, settings=settings, hub=hub
    )

    feature._on_reading_starting("hi")  # type: ignore[misc]  # noqa: SLF001
    assert popup._text_box.toPlainText() == ""  # noqa: SLF001
    doc._on_started()  # type: ignore[misc]  # noqa: SLF001
    assert seen.events[0] == ("started", "Screen text", True)
    doc._on_finished()  # noqa: SLF001

    settings.update(VocalisConfig(floating_widget_enabled=False))
    feature._on_reading_starting("shown text")  # type: ignore[misc]  # noqa: SLF001
    assert popup._text_box.toPlainText() == "shown text"  # noqa: SLF001


# ---- Task 10: assemble the UI in main() -----------------------------------

from vocalis.cli import build_ui  # noqa: E402
from vocalis.ui.doc_popup import DocumentReaderPopup  # noqa: E402


def _safe_output_device_chooser(*args: object, **kwargs: object) -> "OutputDeviceChooser":
    """Build a real OutputDeviceChooser, but with query_devices faked to return none.

    Used in place of the real `vocalis.cli.OutputDeviceChooser` in main()
    tests, so SettingsWindow's real `output_names()` call during
    construction never reaches the real `sounddevice.query_devices()`.
    Still returns a genuine `OutputDeviceChooser` instance (not a
    MagicMock), so `isinstance(..., OutputDeviceChooser)` checks keep
    working.
    """
    kwargs.setdefault("query_devices", lambda: [])
    return OutputDeviceChooser(*args, **kwargs)  # type: ignore[arg-type]


def _main_patches(tmp_path: Path) -> "list[object]":
    """The usual patches that keep main() away from real devices and the real config."""
    return [
        patch("vocalis.cli.default_config_path", return_value=tmp_path / "config.toml"),
        patch("vocalis.cli._socket_path", return_value=tmp_path / "vocalis.sock"),
        patch("vocalis.cli.list_numlock_capable_devices", return_value=[]),
        patch("vocalis.cli.list_hotkey_capable_devices", return_value=[]),
        patch("vocalis.cli.list_devices_with_key", return_value=[]),
        patch("vocalis.cli.OutputDeviceChooser", side_effect=_safe_output_device_chooser),
    ]


def test_build_ui_wires_tray_widget_and_settings(qtbot, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    settings = LiveSettings(VocalisConfig(), tmp_path / "c.toml", saver=lambda _c, _p: None)
    hub = StatusHub()
    popup = DocumentReaderPopup(_StopRecorder())  # type: ignore[arg-type]
    qtbot.addWidget(popup)
    quits: list[str] = []

    widget, settings_window, tray = build_ui(settings, hub, popup, lambda: quits.append("quit"))
    qtbot.addWidget(widget)
    qtbot.addWidget(settings_window)

    tray._quit_action.trigger()  # noqa: SLF001
    assert quits == ["quit"]
    settings_window._notifications_box.setChecked(False)  # noqa: SLF001
    tray._settings_action.trigger()  # noqa: SLF001
    assert settings_window.isVisible()
    assert settings_window._notifications_box.isChecked() is True  # noqa: SLF001  # reloaded
    tray._docs_action.trigger()  # noqa: SLF001
    assert popup.isVisible()

    from vocalis.ui.status_hub import ReadingControls
    hub.begin_reading("Quick read", ReadingControls(stop=lambda: None))
    assert widget.isVisible()
    settings_window._widget_box.setChecked(False)  # noqa: SLF001
    settings_window._save_button.click()  # noqa: SLF001
    assert not widget.isVisible()
    tray.close()


def test_main_does_not_quit_when_the_last_window_closes(tmp_path: Path, qapp) -> None:  # type: ignore[no-untyped-def]
    from PySide6.QtWidgets import QApplication

    seen: list[bool] = []

    def fake_exec(*_args: object) -> int:
        seen.append(QApplication.instance().quitOnLastWindowClosed())  # type: ignore[union-attr]
        return 0

    patches = _main_patches(tmp_path)
    with (
        patches[0], patches[1], patches[2], patches[3], patches[4], patches[5],  # type: ignore[attr-defined]
        patch("vocalis.cli.QApplication.exec", side_effect=fake_exec),
    ):
        main()

    assert seen == [False]


def test_main_closes_the_tray_on_exit_and_a_failing_close_stops_nothing_else(tmp_path: Path, qapp) -> None:  # type: ignore[no-untyped-def]
    calls = MagicMock()
    calls.tray.close.side_effect = RuntimeError("tray close failed")
    patches = _main_patches(tmp_path)
    with (
        patches[0], patches[1], patches[2], patches[3], patches[4], patches[5],  # type: ignore[attr-defined]
        patch("vocalis.cli.VocalisDaemon", return_value=calls.daemon),
        patch("vocalis.cli.build_dictation", return_value=(MagicMock(), calls.dictation)),
        patch("vocalis.cli.build_quick_reader", return_value=(MagicMock(), calls.quick)),
        patch("vocalis.cli.build_doc_reader", return_value=(MagicMock(), MagicMock(), calls.doc)),
        patch("vocalis.cli.build_ocr_reader", return_value=(MagicMock(), calls.ocr)),
        patch("vocalis.cli.build_ui", return_value=(calls.widget, MagicMock(), calls.tray)),
        patch("vocalis.cli.QApplication.exec", side_effect=KeyboardInterrupt),
    ):
        main()  # must not raise

    calls.tray.show.assert_called_once_with()
    calls.tray.close.assert_called_once_with()
    calls.widget.hide.assert_called_once_with()
    for listener in (calls.dictation, calls.quick, calls.doc, calls.ocr):
        listener.stop.assert_called_once_with()
    calls.daemon.stop.assert_called_once_with()


def test_main_quit_action_closes_the_tray_before_quitting(tmp_path: Path, qapp) -> None:  # type: ignore[no-untyped-def]
    order: list[str] = []
    captured: dict[str, object] = {}
    patches = _main_patches(tmp_path)

    def fake_tray(hub, on_open_document_reader, on_open_settings, on_quit):  # type: ignore[no-untyped-def]
        captured["quit"] = on_quit
        tray = MagicMock()
        tray.close.side_effect = lambda: order.append("close")
        return tray

    def fake_exec(*_args: object) -> int:
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        app.quit = lambda: order.append("quit")  # type: ignore[union-attr,method-assign]
        captured["quit"]()  # type: ignore[operator]
        del app.quit  # type: ignore[union-attr]
        return 0

    with (
        patches[0], patches[1], patches[2], patches[3], patches[4], patches[5],  # type: ignore[attr-defined]
        patch("vocalis.cli.TrayIcon", side_effect=fake_tray),
        patch("vocalis.cli.QApplication.exec", side_effect=fake_exec),
    ):
        main()

    assert order[:2] == ["close", "quit"]


# ---- Task 9: sound output, speaking pace and the new Settings callbacks ---

from vocalis.audio.output_device import OutputDeviceChooser  # noqa: E402
from vocalis.audio.playback import PlaybackTimeoutError  # noqa: E402
from vocalis.cli import PREVIEW_SENTENCE  # noqa: E402


class _FakeChooser:
    """A fake OutputDeviceChooser: records mark_failed() calls, returns a fixed device."""

    def __init__(self, device: "int | None" = 3) -> None:
        self.device = device
        self.marked_failed: "list[int | None]" = []

    def current(self) -> "int | None":
        return self.device

    def mark_failed(self, device: "int | None") -> None:
        self.marked_failed.append(device)

    def output_names(self) -> "list[str]":
        return ["pulse", "pipewire"]


class _FakeNotifier:
    """A fake Notifier that just records every notify() call."""

    def __init__(self) -> None:
        self.calls: "list[tuple[str, str]]" = []

    def notify(self, title: str, message: str = "") -> None:
        self.calls.append((title, message))


def test_build_quick_reader_wires_device_provider_and_failure_hook_to_chooser() -> None:
    chooser = _FakeChooser(device=5)
    with patch("vocalis.cli.list_hotkey_capable_devices", return_value=[]):
        feature, _ = build_quick_reader(
            config=VocalisConfig(), model_manager=ModelManager(),
            reading_coordinator=ReadingCoordinator(), output_chooser=cast(OutputDeviceChooser, chooser),
        )
    player = feature._player  # noqa: SLF001
    assert player._device_provider == chooser.current  # noqa: SLF001
    failure_hook = player._on_device_failed  # noqa: SLF001
    assert failure_hook is not None
    failure_hook(5)
    failure_hook(5)
    assert chooser.marked_failed == [5, 5]


def test_build_doc_reader_wires_device_provider_and_failure_hook_to_chooser(qtbot) -> None:  # type: ignore[no-untyped-def]
    chooser = _FakeChooser(device=7)
    feature, popup, _ = build_doc_reader(
        config=VocalisConfig(), model_manager=ModelManager(),
        reading_coordinator=ReadingCoordinator(), output_chooser=cast(OutputDeviceChooser, chooser),
    )
    qtbot.addWidget(popup)
    player = feature._player  # noqa: SLF001
    assert player._device_provider == chooser.current  # noqa: SLF001
    assert player._on_device_failed is not None  # noqa: SLF001


def test_device_failure_hook_notifies_once_per_device_but_again_on_switch() -> None:
    """Fix round 1, ruling #3: notify again whenever the failing device changes.

    hook(2) -> notify (device 2); hook(2) again -> no new notification;
    hook(5) -> notify again (different device); hook(5) again -> no new
    notification; hook(None) -> notify again (None is a distinct "device").
    """
    chooser = _FakeChooser(device=2)
    notifier = _FakeNotifier()
    with patch("vocalis.cli._make_notifier", return_value=notifier), \
         patch("vocalis.cli.list_hotkey_capable_devices", return_value=[]):
        feature, _ = build_quick_reader(
            config=VocalisConfig(), model_manager=ModelManager(),
            reading_coordinator=ReadingCoordinator(), output_chooser=cast(OutputDeviceChooser, chooser),
        )
    hook = feature._player._on_device_failed  # noqa: SLF001
    assert hook is not None
    hook(2)
    hook(2)
    hook(5)
    hook(5)
    hook(None)
    hook(None)
    assert chooser.marked_failed == [2, 2, 5, 5, None, None]
    sound_notification = ("Vocalis — Sound", "Sound output changed because it stopped working")
    assert notifier.calls == [sound_notification, sound_notification, sound_notification]


def test_device_failure_hook_marks_failed_and_notifies_for_doc_reader(qtbot) -> None:  # type: ignore[no-untyped-def]
    """Item 5: the Document Reader's own hook marks failed and notifies once."""
    chooser = _FakeChooser(device=7)
    notifier = _FakeNotifier()
    with patch("vocalis.cli._make_notifier", return_value=notifier):
        feature, popup, _ = build_doc_reader(
            config=VocalisConfig(), model_manager=ModelManager(),
            reading_coordinator=ReadingCoordinator(), output_chooser=cast(OutputDeviceChooser, chooser),
        )
    qtbot.addWidget(popup)
    hook = feature._player._on_device_failed  # noqa: SLF001
    assert hook is not None

    hook(7)

    assert chooser.marked_failed == [7]
    assert notifier.calls == [("Vocalis — Sound", "Sound output changed because it stopped working")]


def test_device_failure_hooks_of_quick_and_doc_reader_notify_independently(qtbot) -> None:  # type: ignore[no-untyped-def]
    """Item 5: triggering one reader's hook must not suppress the other's."""
    quick_chooser = _FakeChooser(device=2)
    doc_chooser = _FakeChooser(device=7)
    quick_notifier = _FakeNotifier()
    doc_notifier = _FakeNotifier()

    notifiers = iter([quick_notifier, doc_notifier])
    with patch("vocalis.cli._make_notifier", side_effect=lambda *_a, **_kw: next(notifiers)), \
         patch("vocalis.cli.list_hotkey_capable_devices", return_value=[]):
        quick_feature, _ = build_quick_reader(
            config=VocalisConfig(), model_manager=ModelManager(),
            reading_coordinator=ReadingCoordinator(), output_chooser=cast(OutputDeviceChooser, quick_chooser),
        )
        doc_feature, popup, _ = build_doc_reader(
            config=VocalisConfig(), model_manager=ModelManager(),
            reading_coordinator=ReadingCoordinator(), output_chooser=cast(OutputDeviceChooser, doc_chooser),
        )
    qtbot.addWidget(popup)
    quick_hook = quick_feature._player._on_device_failed  # noqa: SLF001
    doc_hook = doc_feature._player._on_device_failed  # noqa: SLF001
    assert quick_hook is not None
    assert doc_hook is not None

    quick_hook(2)
    doc_hook(7)

    assert quick_notifier.calls == [("Vocalis — Sound", "Sound output changed because it stopped working")]
    assert doc_notifier.calls == [("Vocalis — Sound", "Sound output changed because it stopped working")]


def test_build_quick_reader_without_output_chooser_keeps_a_plain_audio_player() -> None:
    with patch("vocalis.cli.list_hotkey_capable_devices", return_value=[]):
        feature, _ = build_quick_reader(
            config=VocalisConfig(), model_manager=ModelManager(),
            reading_coordinator=ReadingCoordinator(),
        )
    player = feature._player  # noqa: SLF001
    assert player._device_provider is None  # noqa: SLF001
    assert player._on_device_failed is None  # noqa: SLF001


def test_build_doc_reader_without_output_chooser_keeps_a_plain_audio_player(qtbot) -> None:  # type: ignore[no-untyped-def]
    feature, popup, _ = build_doc_reader(
        config=VocalisConfig(), model_manager=ModelManager(), reading_coordinator=ReadingCoordinator(),
    )
    qtbot.addWidget(popup)
    player = feature._player  # noqa: SLF001
    assert player._device_provider is None  # noqa: SLF001
    assert player._on_device_failed is None  # noqa: SLF001


def test_primary_engine_reads_the_live_speed(tmp_path: Path, qtbot) -> None:  # type: ignore[no-untyped-def]
    settings = _live_settings(tmp_path)
    manager = ModelManager()
    with patch("vocalis.cli.list_hotkey_capable_devices", return_value=[]):
        quick, _ = build_quick_reader(
            config=settings.config, model_manager=manager,
            reading_coordinator=ReadingCoordinator(), settings=settings,
        )
    doc, popup, _ = build_doc_reader(
        config=settings.config, model_manager=manager,
        reading_coordinator=ReadingCoordinator(), settings=settings,
    )
    qtbot.addWidget(popup)
    settings.update(settings.config.model_copy(update={"speaking_speed": 1.4}))
    for feature in (quick, doc):
        engine = feature._primary_engine_factory()  # noqa: SLF001
        assert isinstance(engine, Qwen3TTS)
        assert engine._current_speed() == 1.4  # noqa: SLF001


def test_kokoro_factory_reads_the_live_speed(tmp_path: Path) -> None:
    settings = _live_settings(tmp_path)
    with patch("vocalis.cli.list_hotkey_capable_devices", return_value=[]):
        feature, _ = build_quick_reader(
            config=settings.config, model_manager=ModelManager(),
            reading_coordinator=ReadingCoordinator(), settings=settings,
        )
    kokoro = (feature._engine_factories or {})["kokoro"]()  # noqa: SLF001
    assert isinstance(kokoro, KokoroTTS)
    assert kokoro._current_speed() == settings.speaking_speed()  # noqa: SLF001
    settings.update(settings.config.model_copy(update={"speaking_speed": 1.3}))
    assert kokoro._current_speed() == 1.3  # noqa: SLF001


def test_fallback_engine_reads_the_live_speed(tmp_path: Path) -> None:
    settings = _live_settings(tmp_path)
    with patch("vocalis.cli.list_hotkey_capable_devices", return_value=[]):
        feature, _ = build_quick_reader(
            config=settings.config, model_manager=ModelManager(),
            reading_coordinator=ReadingCoordinator(), settings=settings,
        )
    piper = feature._fallback_engine  # noqa: SLF001
    assert piper._current_speed() == settings.speaking_speed()  # noqa: SLF001
    settings.update(settings.config.model_copy(update={"speaking_speed": 0.9}))
    assert piper._current_speed() == 0.9  # noqa: SLF001


def test_build_quick_reader_without_settings_uses_the_configured_speed() -> None:
    with patch("vocalis.cli.list_hotkey_capable_devices", return_value=[]):
        feature, _ = build_quick_reader(
            config=VocalisConfig(), model_manager=ModelManager(), reading_coordinator=ReadingCoordinator(),
        )
    engine = feature._primary_engine_factory()  # noqa: SLF001
    assert engine._current_speed() == VocalisConfig().speaking_speed  # noqa: SLF001


def test_main_builds_one_shared_chooser_and_passes_it_to_both_builders(tmp_path: Path, qapp) -> None:  # type: ignore[no-untyped-def]
    seen: dict[str, object] = {}

    def fake_quick_reader(**kwargs):  # type: ignore[no-untyped-def]
        seen["quick_chooser"] = kwargs.get("output_chooser")
        return MagicMock(), MagicMock()

    def fake_doc_reader(**kwargs):  # type: ignore[no-untyped-def]
        seen["doc_chooser"] = kwargs.get("output_chooser")
        return MagicMock(), DocumentReaderPopup(_StopRecorder()), MagicMock()  # type: ignore[arg-type]

    patches = _main_patches(tmp_path)
    with (
        patches[0], patches[1], patches[2], patches[3], patches[4], patches[5],  # type: ignore[attr-defined]
        patch("vocalis.cli.build_quick_reader", side_effect=fake_quick_reader),
        patch("vocalis.cli.build_doc_reader", side_effect=fake_doc_reader),
        patch("vocalis.cli.QApplication.exec", side_effect=KeyboardInterrupt),
    ):
        main()

    assert seen["quick_chooser"] is not None
    assert seen["quick_chooser"] is seen["doc_chooser"]
    assert isinstance(seen["quick_chooser"], OutputDeviceChooser)


class _FakeDocumentReader:
    """A fake Document Reader used only to test the Settings window callbacks."""

    def __init__(self, is_reading: bool = False) -> None:
        self.is_reading = is_reading
        self.read_calls: "list[str]" = []

    def read_text(self, text: str) -> None:
        self.read_calls.append(text)


def test_on_preview_voice_reads_the_preview_sentence() -> None:
    from vocalis.cli import _make_on_preview_voice

    document_reader = _FakeDocumentReader()
    on_preview_voice = _make_on_preview_voice(document_reader)  # type: ignore[arg-type]

    on_preview_voice()

    assert document_reader.read_calls == [PREVIEW_SENTENCE]


def test_on_test_sound_returns_busy_message_without_opening_a_stream() -> None:
    from vocalis.cli import _make_on_test_sound

    document_reader = _FakeDocumentReader(is_reading=True)
    chooser = _FakeChooser(device=1)
    on_test_sound = _make_on_test_sound(document_reader, cast(OutputDeviceChooser, chooser))  # type: ignore[arg-type]

    with patch("vocalis.cli.AudioPlayer") as mock_player_cls:
        message = on_test_sound("auto")

    assert message == "A reading is playing right now — try again when it has finished."
    mock_player_cls.assert_not_called()


def test_on_test_sound_returns_ok_when_the_player_plays_successfully() -> None:
    from vocalis.cli import _make_on_test_sound

    document_reader = _FakeDocumentReader(is_reading=False)
    chooser = _FakeChooser(device=4)
    on_test_sound = _make_on_test_sound(document_reader, cast(OutputDeviceChooser, chooser))  # type: ignore[arg-type]

    fake_player = MagicMock()
    fake_player.wait_until_done.return_value = True
    with (
        patch("vocalis.cli.AudioPlayer", return_value=fake_player) as mock_player_cls,
        patch("vocalis.cli.sd.query_devices", return_value={"name": "Fake Device 4"}) as mock_query,
    ):
        message = on_test_sound("auto")

    assert message == "Sound test OK (Fake Device 4)"
    mock_query.assert_called_once_with(4)
    fake_player.play.assert_called_once()
    fake_player.wait_until_done.assert_called_once_with()
    _, kwargs = mock_player_cls.call_args
    assert kwargs["device_provider"]() == 4


def test_on_test_sound_auto_resolving_to_none_is_a_failure() -> None:
    """Fix round 1, #1: device None for "auto" is a failure, not a silent success.

    Must be checked before any AudioPlayer is constructed, so a machine
    whose system default silently stalls never opens the throwaway stream
    (which is what used to cause the ~10s GUI freeze).
    """
    from vocalis.cli import _make_on_test_sound

    document_reader = _FakeDocumentReader(is_reading=False)
    chooser = _FakeChooser(device=None)
    on_test_sound = _make_on_test_sound(document_reader, cast(OutputDeviceChooser, chooser))  # type: ignore[arg-type]

    with patch("vocalis.cli.AudioPlayer") as mock_player_cls:
        message = on_test_sound("auto")

    assert message == "Sound test failed: no working output found"
    mock_player_cls.assert_not_called()


def test_on_test_sound_named_selection_not_found_is_a_failure() -> None:
    """Fix round 1, #1: a named selection whose device cannot be resolved at
    all (no matching device on the system) fails with the same generic
    message as "auto" finding nothing, and never opens a stream."""
    from vocalis.cli import _make_on_test_sound

    document_reader = _FakeDocumentReader(is_reading=False)
    shared_chooser = _FakeChooser(device=4)
    probe_chooser = _FakeProbeChooser(devices=[(0, "default")])
    on_test_sound = _make_on_test_sound(document_reader, cast(OutputDeviceChooser, shared_chooser))  # type: ignore[arg-type]

    with (
        patch("vocalis.cli.OutputDeviceChooser", return_value=probe_chooser),
        patch("vocalis.cli.AudioPlayer") as mock_player_cls,
    ):
        message = on_test_sound("pulse")

    assert message == "Sound test failed: no working output found"
    mock_player_cls.assert_not_called()
    assert probe_chooser.probed == []


def test_on_test_sound_uses_a_short_timeout_margin_for_the_throwaway_player() -> None:
    """Fix round 1, #1: the throwaway test-tone player must fail fast, not
    wait the production ~10s margin, so a silently stalling device reports
    a clear failure quickly instead of freezing the Settings window."""
    from vocalis.cli import _make_on_test_sound

    document_reader = _FakeDocumentReader(is_reading=False)
    chooser = _FakeChooser(device=4)
    on_test_sound = _make_on_test_sound(document_reader, cast(OutputDeviceChooser, chooser))  # type: ignore[arg-type]

    fake_player = MagicMock()
    fake_player.wait_until_done.return_value = True
    with (
        patch("vocalis.cli.AudioPlayer", return_value=fake_player) as mock_player_cls,
        patch("vocalis.cli.sd.query_devices", return_value={"name": "pulse"}),
    ):
        on_test_sound("auto")

    _, kwargs = mock_player_cls.call_args
    assert kwargs["timeout_margin_seconds"] <= 3.0


class _FakeProbeChooser:
    """A fake one-off OutputDeviceChooser used to test probing a named device.

    Mirrors the two methods `_make_on_test_sound` actually calls on it:
    `_output_devices()` to look up the index for a chosen name, and
    `probe(index)` to test exactly that one device -- never the chooser's
    own pulse/pipewire/default/other fallback order.
    """

    def __init__(self, devices: "list[tuple[int, str]]", probe_result: bool = True) -> None:
        self._devices = devices
        self._probe_result = probe_result
        self.probed: "list[int]" = []

    def _output_devices(self) -> "list[tuple[int, str]]":
        return self._devices

    def probe(self, index: int) -> bool:
        self.probed.append(index)
        return self._probe_result


def test_on_test_sound_named_selection_plays_on_that_device_only() -> None:
    """Fix round 1, #2: a named selection plays on that exact device and
    names it correctly, probing only that one device (no fallback)."""
    from vocalis.cli import _make_on_test_sound

    document_reader = _FakeDocumentReader(is_reading=False)
    shared_chooser = _FakeChooser(device=4)
    probe_chooser = _FakeProbeChooser(devices=[(0, "default"), (9, "pulse")], probe_result=True)
    on_test_sound = _make_on_test_sound(document_reader, cast(OutputDeviceChooser, shared_chooser))  # type: ignore[arg-type]

    fake_player = MagicMock()
    fake_player.wait_until_done.return_value = True
    with (
        patch("vocalis.cli.OutputDeviceChooser", return_value=probe_chooser) as mock_chooser_cls,
        patch("vocalis.cli.AudioPlayer", return_value=fake_player) as mock_player_cls,
        patch("vocalis.cli.sd.query_devices", return_value={"name": "pulse"}),
    ):
        message = on_test_sound("pulse")

    assert message == "Sound test OK (pulse)"
    mock_chooser_cls.assert_called_once()
    _, kwargs = mock_chooser_cls.call_args
    assert kwargs["preference"]() == "pulse"
    assert probe_chooser.probed == [9]
    _, player_kwargs = mock_player_cls.call_args
    assert player_kwargs["device_provider"]() == 9


def test_on_test_sound_named_selection_that_fails_probes_only_that_device() -> None:
    """Fix round 1, #2: a named selection whose one device is bad fails
    clearly and never falls through to try a different, working device."""
    from vocalis.cli import _make_on_test_sound

    document_reader = _FakeDocumentReader(is_reading=False)
    shared_chooser = _FakeChooser(device=4)
    probe_chooser = _FakeProbeChooser(devices=[(0, "default"), (9, "pulse")], probe_result=False)
    on_test_sound = _make_on_test_sound(document_reader, cast(OutputDeviceChooser, shared_chooser))  # type: ignore[arg-type]

    with (
        patch("vocalis.cli.OutputDeviceChooser", return_value=probe_chooser),
        patch("vocalis.cli.AudioPlayer") as mock_player_cls,
    ):
        message = on_test_sound("pulse")

    assert message == "Sound test failed: pulse did not respond"
    mock_player_cls.assert_not_called()
    assert probe_chooser.probed == [9]


def test_on_test_sound_auto_still_uses_the_normal_fallback_order() -> None:
    """Fix round 1, #2: "auto" keeps using the chooser's normal fallback
    order (via its own current()), naming whichever device actually worked."""
    from vocalis.cli import _make_on_test_sound

    document_reader = _FakeDocumentReader(is_reading=False)
    chooser = _FakeChooser(device=11)
    on_test_sound = _make_on_test_sound(document_reader, cast(OutputDeviceChooser, chooser))  # type: ignore[arg-type]

    fake_player = MagicMock()
    fake_player.wait_until_done.return_value = True
    with (
        patch("vocalis.cli.AudioPlayer", return_value=fake_player),
        patch("vocalis.cli.sd.query_devices", return_value={"name": "pipewire"}) as mock_query,
    ):
        message = on_test_sound("auto")

    assert message == "Sound test OK (pipewire)"
    mock_query.assert_called_once_with(11)


def test_on_test_sound_reports_failure_when_playback_raises() -> None:
    from vocalis.cli import _make_on_test_sound

    document_reader = _FakeDocumentReader(is_reading=False)
    chooser = _FakeChooser(device=4)
    on_test_sound = _make_on_test_sound(document_reader, cast(OutputDeviceChooser, chooser))  # type: ignore[arg-type]

    fake_player = MagicMock()
    fake_player.wait_until_done.side_effect = PlaybackTimeoutError("timed out")
    with patch("vocalis.cli.AudioPlayer", return_value=fake_player):
        message = on_test_sound("auto")

    assert message == "Sound test failed: timed out"


def test_build_ui_without_output_chooser_keeps_the_old_settings_window_defaults(qtbot, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    settings = LiveSettings(VocalisConfig(), tmp_path / "c.toml", saver=lambda _c, _p: None)
    hub = StatusHub()
    popup = DocumentReaderPopup(_StopRecorder())  # type: ignore[arg-type]
    qtbot.addWidget(popup)

    widget, settings_window, tray = build_ui(settings, hub, popup, lambda: None)
    qtbot.addWidget(widget)
    qtbot.addWidget(settings_window)

    assert settings_window._output_names() == []  # noqa: SLF001
    tray.close()


def test_build_ui_with_output_chooser_fills_output_names_and_wires_test_sound(qtbot, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    settings = LiveSettings(VocalisConfig(), tmp_path / "c.toml", saver=lambda _c, _p: None)
    hub = StatusHub()
    popup = DocumentReaderPopup(_FakeDocumentReader())  # type: ignore[arg-type]
    qtbot.addWidget(popup)
    chooser = _FakeChooser(device=3)

    widget, settings_window, tray = build_ui(
        settings, hub, popup, lambda: None, cast(OutputDeviceChooser, chooser)
    )
    qtbot.addWidget(widget)
    qtbot.addWidget(settings_window)

    assert settings_window._output_names() == ["pulse", "pipewire"]  # noqa: SLF001
    with (
        patch("vocalis.cli.AudioPlayer") as mock_player_cls,
        patch("vocalis.cli.sd.query_devices", return_value={"name": "pulse"}),
    ):
        mock_player_cls.return_value.wait_until_done.return_value = True
        message = settings_window._on_test_sound("auto")  # noqa: SLF001
    assert message.startswith("Sound test OK")

    settings_window._on_preview_voice()  # noqa: SLF001
    fake_document_reader = cast(_FakeDocumentReader, popup._feature)  # noqa: SLF001
    assert fake_document_reader.read_calls == [PREVIEW_SENTENCE]
    tray.close()


def test_warm_up_pace_changer_applies_speed_1_1_to_one_second_of_silence() -> None:
    import numpy as np

    from vocalis.cli import _warm_up_pace_changer

    with patch("vocalis.cli.apply_speed") as mock_apply_speed:
        _warm_up_pace_changer()

    mock_apply_speed.assert_called_once()
    args, _kwargs = mock_apply_speed.call_args
    audio, sample_rate, speed = args
    assert isinstance(audio, np.ndarray)
    assert audio.shape == (24000,)
    assert audio.dtype == np.float32
    assert np.all(audio == 0.0)
    assert sample_rate == 24000
    assert speed == 1.1


def test_warm_up_pace_changer_logs_a_warning_and_never_raises_when_apply_speed_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from vocalis.cli import _warm_up_pace_changer

    with patch("vocalis.cli.apply_speed", side_effect=RuntimeError("boom")):
        with caplog.at_level(logging.WARNING, logger="vocalis"):
            _warm_up_pace_changer()  # must not raise

    assert any(
        record.levelno == logging.WARNING and "warm up" in record.getMessage().lower()
        for record in caplog.records
    )


def test_main_starts_the_pace_warm_up_as_a_background_daemon_thread(tmp_path: Path, qapp) -> None:  # type: ignore[no-untyped-def]
    socket_path = tmp_path / "vocalis.sock"

    with (
        patch("vocalis.cli.default_config_path", return_value=tmp_path / "config.toml"),
        patch("vocalis.cli._socket_path", return_value=socket_path),
        patch("vocalis.cli.list_numlock_capable_devices", return_value=[]),
        patch("vocalis.cli.list_hotkey_capable_devices", return_value=[]),
        patch("vocalis.cli.list_devices_with_key", return_value=[]),
        patch("vocalis.cli.QApplication.exec", side_effect=KeyboardInterrupt),
        patch("vocalis.cli._warm_up_pace_changer") as mock_warm_up,
        patch("vocalis.cli.threading.Thread", wraps=threading.Thread) as mock_thread_cls,
    ):
        from vocalis.cli import main

        main()

    # Proves the warm-up runs on a background thread rather than being
    # called synchronously inline in main() (which would block app.exec()).
    mock_thread_cls.assert_any_call(target=mock_warm_up, daemon=True)


def test_main_notifies_about_every_real_startup_problem(tmp_path: Path, qapp) -> None:  # type: ignore[no-untyped-def]
    sent: list[tuple[str, str]] = []

    def fake_notify(self: object, title: str, message: str = "") -> None:  # noqa: ANN001
        sent.append((title, message))

    with (
        patch("vocalis.cli.default_config_path", return_value=tmp_path / "config.toml"),
        patch("vocalis.cli._socket_path", return_value=tmp_path / "vocalis.sock"),
        patch("vocalis.cli.list_numlock_capable_devices", return_value=[]),
        patch("vocalis.cli.list_hotkey_capable_devices", return_value=[]),
        patch("vocalis.cli.list_devices_with_key", return_value=[]),
        patch("vocalis.cli.run_startup_checks", return_value=["Problem A", "Problem B"]),
        patch("vocalis.cli.Notifier.notify", fake_notify),
        patch("vocalis.cli.QApplication.exec", side_effect=KeyboardInterrupt),
    ):
        main()

    assert ("Vocalis", "Problem A") in sent
    assert ("Vocalis", "Problem B") in sent


def test_main_sends_no_startup_notification_when_everything_is_fine(tmp_path: Path, qapp) -> None:  # type: ignore[no-untyped-def]
    sent: list[tuple[str, str]] = []

    def fake_notify(self: object, title: str, message: str = "") -> None:  # noqa: ANN001
        sent.append((title, message))

    with (
        patch("vocalis.cli.default_config_path", return_value=tmp_path / "config.toml"),
        patch("vocalis.cli._socket_path", return_value=tmp_path / "vocalis.sock"),
        patch("vocalis.cli.list_numlock_capable_devices", return_value=[]),
        patch("vocalis.cli.list_hotkey_capable_devices", return_value=[]),
        patch("vocalis.cli.list_devices_with_key", return_value=[]),
        patch("vocalis.cli.run_startup_checks", return_value=[]),
        patch("vocalis.cli.Notifier.notify", fake_notify),
        patch("vocalis.cli.QApplication.exec", side_effect=KeyboardInterrupt),
    ):
        main()

    assert sent == []


def test_main_never_attempts_a_real_startup_notification_without_patching_it(  # type: ignore[no-untyped-def]
    tmp_path: Path, qapp, monkeypatch
) -> None:
    """Fix round 1 regression test: reproduces the reviewer's exact finding.

    This test deliberately does NOT patch `vocalis.cli.run_startup_checks`
    itself -- it relies only on the autouse `_no_real_startup_checks`
    fixture above (the same fixture every other test in this file now
    relies on) to keep main() from running the real checks.

    `XDG_RUNTIME_DIR` is pointed at a fresh, empty `tmp_path`, so a REAL
    unpatched `check_ydotoold()` would find no `.ydotool_socket` there and
    report a problem -- exactly the realistic state (fresh login, CI, a
    different machine) the reviewer identified. `notifications_enabled`
    defaults to True, so if that real problem reached `Notifier.notify()`
    it would try to run a real `notify-send` subprocess.

    `vocalis.notify._default_command_runner` is replaced with a recorder
    that raises if it is ever called, instead of ever touching
    `subprocess.run` for real -- this both proves no real notify-send was
    attempted and guarantees this test can never cause one itself, even
    if the autouse fixture's protection were ever removed.

    Before this fix, this exact test failed on a CI-like machine (no
    ydotoold socket under the pointed-at XDG_RUNTIME_DIR): removing the
    autouse `_no_real_startup_checks` fixture reproduces that failure --
    see the "Fix round 1" section of task-4-report.md for the RED output.
    """
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    command_calls: list[list[str]] = []

    def fake_command_runner(argv: list[str]) -> None:
        """Record any attempted command and fail loudly; never run it."""
        command_calls.append(argv)
        raise AssertionError(f"a real command was attempted: {argv!r}")

    with (
        patch("vocalis.cli.default_config_path", return_value=tmp_path / "config.toml"),
        patch("vocalis.cli._socket_path", return_value=tmp_path / "vocalis.sock"),
        patch("vocalis.cli.list_numlock_capable_devices", return_value=[]),
        patch("vocalis.cli.list_hotkey_capable_devices", return_value=[]),
        patch("vocalis.cli.list_devices_with_key", return_value=[]),
        patch("vocalis.notify._default_command_runner", fake_command_runner),
        patch("vocalis.cli.QApplication.exec", side_effect=KeyboardInterrupt),
    ):
        main()

    assert command_calls == []


def test_build_daemon_passes_the_command_handler(tmp_path: Path) -> None:
    handler = MagicMock()
    daemon = build_daemon(model_manager=ModelManager(), socket_path=tmp_path / "s.sock", command_handler=handler)
    assert daemon._command_handler is handler  # noqa: SLF001


def test_settings_bridge_emits_show_settings_for_that_command(qtbot) -> None:  # type: ignore[no-untyped-def]
    bridge = _SettingsBridge()
    with qtbot.waitSignal(bridge.show_settings, timeout=1000):
        bridge.handle_command(SHOW_SETTINGS)


def test_settings_bridge_ignores_other_commands(qtbot) -> None:  # type: ignore[no-untyped-def]
    bridge = _SettingsBridge()
    received: list[int] = []
    bridge.show_settings.connect(lambda: received.append(1))
    bridge.handle_command("something-else")
    assert received == []


def test_settings_bridge_delivers_a_background_threads_command_to_the_main_thread(qtbot) -> None:  # type: ignore[no-untyped-def]
    """The daemon calls handle_command() on its own socket thread; the
    show_settings signal must still be delivered to a slot on the Qt main
    thread -- this is the cross-thread risk the bridge exists to cover."""
    bridge = _SettingsBridge()
    seen_threads: list[threading.Thread] = []
    bridge.show_settings.connect(lambda: seen_threads.append(threading.current_thread()))

    with qtbot.waitSignal(bridge.show_settings, timeout=2000):
        worker = threading.Thread(target=bridge.handle_command, args=(SHOW_SETTINGS,))
        worker.start()
        worker.join()

    assert seen_threads == [threading.main_thread()]


def test_both_readers_use_the_selector_and_default_to_auto(tmp_path: Path, qtbot) -> None:  # type: ignore[no-untyped-def]
    """Built readers carry an engine selector whose preference starts as Automatic."""
    settings = _live_settings(tmp_path)
    manager = ModelManager()
    with patch("vocalis.cli.list_hotkey_capable_devices", return_value=[]):
        quick, _ = build_quick_reader(
            config=settings.config, model_manager=manager, reading_coordinator=ReadingCoordinator(), settings=settings
        )
    assert quick._engine_selector is not None  # noqa: SLF001
    assert quick._engine_selector.plan().chain[0] == "qwen3_tts"  # noqa: SLF001
    assert set(quick._engine_factories or {}) == {"qwen3_tts", "kokoro"}  # noqa: SLF001


def test_choosing_kokoro_in_settings_changes_the_next_plan(tmp_path: Path) -> None:
    """The selector reads the live Settings choice, so a change applies to the next reading."""
    settings = _live_settings(tmp_path)
    with patch("vocalis.cli.list_hotkey_capable_devices", return_value=[]):
        quick, _ = build_quick_reader(
            config=settings.config, model_manager=ModelManager(),
            reading_coordinator=ReadingCoordinator(), settings=settings,
        )
    settings.update(settings.config.model_copy(update={
        "engines": settings.config.engines.model_copy(update={"quick_reader_engine": "kokoro"})}))
    assert quick._engine_selector.plan().chain == ("kokoro", "piper")  # type: ignore[union-attr]  # noqa: SLF001


def test_the_kokoro_factory_builds_a_kokoro_engine_with_live_voice(tmp_path: Path) -> None:
    """The Kokoro factory reads the voice live and decides graphics-card use at load time."""
    from vocalis.cli import _make_kokoro_factory
    from vocalis.engines.kokoro_tts import KokoroTTS

    settings = _live_settings(tmp_path)
    selector = type("S", (), {"kokoro_use_gpu": staticmethod(lambda: True)})()
    engine = _make_kokoro_factory(settings.config, settings, selector)()
    assert isinstance(engine, KokoroTTS)
    assert engine._current_voice() == "af_heart"  # noqa: SLF001
