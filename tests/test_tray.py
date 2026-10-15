"""Tests for vocalis.ui.tray.TrayIcon."""
import logging
from unittest.mock import patch

from vocalis.ui.status_hub import ReadingControls, StatusHub
from vocalis.ui.tray import TrayIcon


def make(qtbot: object) -> "tuple[StatusHub, TrayIcon, list[str]]":
    """Build a real hub and a tray icon whose callbacks record into a list."""
    hub = StatusHub()
    calls: "list[str]" = []
    tray = TrayIcon(hub, on_open_document_reader=lambda: calls.append("docs"),
                    on_open_settings=lambda: calls.append("settings"), on_quit=lambda: calls.append("quit"))
    return hub, tray, calls


def test_idle_menu_state(qtbot) -> None:  # type: ignore[no-untyped-def]
    """An idle tray shows the idle text and has Pause and Stop disabled."""
    _hub, tray, _calls = make(qtbot)
    assert not tray._status_action.isEnabled()  # noqa: SLF001
    assert not tray._pause_action.isEnabled() and not tray._stop_action.isEnabled()  # noqa: SLF001
    assert tray._status_action.text() == "Vocalis — idle"  # noqa: SLF001
    assert tray._tray.toolTip() == "Vocalis — idle"  # noqa: SLF001


def test_reading_enables_pause_and_stop_and_names_the_source(qtbot) -> None:  # type: ignore[no-untyped-def]
    """A pausable reading enables both actions, names the source and resets afterwards."""
    hub, tray, _calls = make(qtbot)
    hub.begin_reading("Screen text", ReadingControls(stop=lambda: None, pause=lambda: None, resume=lambda: None))
    assert tray._tray.toolTip() == "Vocalis — reading: Screen text"  # noqa: SLF001
    assert tray._pause_action.isEnabled() and tray._stop_action.isEnabled()  # noqa: SLF001
    hub.set_paused(True)
    assert tray._pause_action.text() == "Resume"  # noqa: SLF001
    hub.end_reading()
    assert not tray._pause_action.isEnabled() and not tray._stop_action.isEnabled()  # noqa: SLF001
    assert tray._pause_action.text() == "Pause"  # noqa: SLF001


def test_stop_only_reading_leaves_pause_disabled(qtbot) -> None:  # type: ignore[no-untyped-def]
    """A reading that cannot pause leaves Pause disabled but enables Stop."""
    hub, tray, _calls = make(qtbot)
    hub.begin_reading("Quick read", ReadingControls(stop=lambda: None))
    assert not tray._pause_action.isEnabled() and tray._stop_action.isEnabled()  # noqa: SLF001


def test_listening_changes_the_text_and_the_icon_colour(qtbot) -> None:  # type: ignore[no-untyped-def]
    """The icon dot is grey idle, red listening, blue reading."""
    hub, tray, _calls = make(qtbot)
    idle_icon = tray._tray.icon().pixmap(22, 22).toImage().pixelColor(11, 11).name()  # noqa: SLF001
    hub.set_listening(True)
    assert tray._tray.toolTip() == "Vocalis — listening"  # noqa: SLF001
    listening = tray._tray.icon().pixmap(22, 22).toImage().pixelColor(11, 11).name()  # noqa: SLF001
    assert listening == "#e53935" and idle_icon == "#9e9e9e"
    hub.begin_reading("Document", ReadingControls(stop=lambda: None))
    assert tray._tray.icon().pixmap(22, 22).toImage().pixelColor(11, 11).name() == "#1e88e5"  # noqa: SLF001


def test_menu_actions_call_the_callbacks_and_controls(qtbot) -> None:  # type: ignore[no-untyped-def]
    """Menu actions call the hub controls and the injected callbacks."""
    hub, tray, calls = make(qtbot)
    log: "list[str]" = []
    hub.begin_reading("Document", ReadingControls(stop=lambda: log.append("stop"),
                                                  pause=lambda: log.append("pause"), resume=lambda: log.append("resume")))
    tray._pause_action.trigger()  # noqa: SLF001
    tray._stop_action.trigger()  # noqa: SLF001
    tray._docs_action.trigger()  # noqa: SLF001
    tray._settings_action.trigger()  # noqa: SLF001
    tray._quit_action.trigger()  # noqa: SLF001
    assert log == ["pause", "stop"]
    assert calls == ["docs", "settings", "quit"]


def test_show_without_a_system_tray_warns_once_and_does_not_crash(qtbot, caplog) -> None:  # type: ignore[no-untyped-def]
    """Without a system tray show() warns once and reports unavailable."""
    _hub, tray, _calls = make(qtbot)
    with patch("vocalis.ui.tray.QSystemTrayIcon.isSystemTrayAvailable", return_value=False):
        with caplog.at_level(logging.WARNING, logger="vocalis"):
            tray.show()
            tray.show()
    assert [r.getMessage() for r in caplog.records].count("No system tray available; continuing without the tray icon") == 1
    assert tray.available is False


def test_show_with_a_system_tray_shows_the_icon(qtbot) -> None:  # type: ignore[no-untyped-def]
    """With a system tray show() marks the tray available."""
    _hub, tray, _calls = make(qtbot)
    try:
        with patch("vocalis.ui.tray.QSystemTrayIcon.isSystemTrayAvailable", return_value=True):
            tray.show()
        assert tray.available is True
    finally:
        tray.hide()


def test_close_is_idempotent_and_safe_without_show(qtbot) -> None:  # type: ignore[no-untyped-def]
    """close() works before any show() and a second call does nothing and does not raise."""
    _hub, tray, _calls = make(qtbot)
    tray.close()
    tray.close()
    assert tray._tray.contextMenu() is None  # noqa: SLF001


def test_close_after_show_hides_the_icon(qtbot) -> None:  # type: ignore[no-untyped-def]
    """close() after show() hides the icon and detaches the menu."""
    _hub, tray, _calls = make(qtbot)
    with patch("vocalis.ui.tray.QSystemTrayIcon.isSystemTrayAvailable", return_value=True):
        tray.show()
    tray.close()
    tray.close()
    assert not tray._tray.isVisible()  # noqa: SLF001


def test_a_closed_tray_ignores_hub_signals(qtbot, capsys) -> None:  # type: ignore[no-untyped-def]
    """After close() (and the menu's deferred delete) hub signals do nothing and raise nothing."""
    from PySide6.QtCore import SIGNAL, QEvent
    from PySide6.QtWidgets import QApplication

    signatures = [
        "state_changed(QString)",
        "reading_started(QString,bool)",
        "reading_finished()",
        "paused_changed(bool)",
    ]
    hub, tray, _calls = make(qtbot)
    for signature in signatures:
        assert hub.receivers(SIGNAL(signature)) == 1, signature
    tray.close()
    for signature in signatures:
        assert hub.receivers(SIGNAL(signature)) == 0, signature
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    hub.begin_reading("Document", ReadingControls(stop=lambda: None, pause=lambda: None, resume=lambda: None))
    hub.set_paused(True)
    hub.set_listening(True)
    hub.end_reading()
    assert "RuntimeError" not in capsys.readouterr().err
