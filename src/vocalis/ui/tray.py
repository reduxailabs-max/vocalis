"""The system tray icon: shows what Vocalis is doing and offers a small menu."""
import logging
from typing import Callable

from PySide6.QtCore import QObject, Qt
from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QMenu, QSystemTrayIcon

from vocalis.ui.status_hub import IDLE, LISTENING, READING, StatusHub

logger = logging.getLogger("vocalis")

ICON_SIZE = 22
COLOURS = {IDLE: "#9e9e9e", LISTENING: "#e53935", READING: "#1e88e5"}
NO_TRAY_WARNING = "No system tray available; continuing without the tray icon"


def _make_icon(colour: str) -> QIcon:
    """Draw a 22x22 icon: a filled circle of the given colour on a transparent background."""
    pixmap = QPixmap(ICON_SIZE, ICON_SIZE)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QColor(colour))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(3, 3, 16, 16)
    painter.end()
    return QIcon(pixmap)


class TrayIcon(QObject):
    """Tray icon with a status line, Pause/Stop and shortcuts to the app windows."""

    def __init__(
        self,
        hub: StatusHub,
        on_open_document_reader: Callable[[], None],
        on_open_settings: Callable[[], None],
        on_quit: Callable[[], None],
    ) -> None:
        """Build the icon and its menu once; later changes come from the hub's signals."""
        super().__init__()
        self._hub = hub
        self._available = False
        self._warned = False
        self._closed = False
        self._tray = QSystemTrayIcon(self)
        self._menu = QMenu()
        self._status_action = QAction("Vocalis — idle", self._menu)
        self._status_action.setEnabled(False)
        self._pause_action = QAction("Pause", self._menu)
        self._stop_action = QAction("Stop", self._menu)
        self._docs_action = QAction("Open Document Reader", self._menu)
        self._settings_action = QAction("Settings…", self._menu)
        self._quit_action = QAction("Quit", self._menu)
        self._menu.addAction(self._status_action)
        self._menu.addAction(self._pause_action)
        self._menu.addAction(self._stop_action)
        self._menu.addSeparator()
        self._menu.addAction(self._docs_action)
        self._menu.addAction(self._settings_action)
        self._menu.addSeparator()
        self._menu.addAction(self._quit_action)
        self._tray.setContextMenu(self._menu)
        self._pause_action.triggered.connect(lambda _checked=False: hub.toggle_pause())
        self._stop_action.triggered.connect(lambda _checked=False: hub.stop())
        self._docs_action.triggered.connect(lambda _checked=False: on_open_document_reader())
        self._settings_action.triggered.connect(lambda _checked=False: on_open_settings())
        self._quit_action.triggered.connect(lambda _checked=False: on_quit())
        hub.state_changed.connect(self._on_state_changed)
        hub.reading_started.connect(self._on_reading_started)
        hub.reading_finished.connect(self._on_reading_finished)
        hub.paused_changed.connect(self._on_paused_changed)
        self._set_actions_enabled(False, False)
        self._refresh()

    @property
    def available(self) -> bool:
        """Whether a system tray existed when show() was last called (not reset by hide()/close())."""
        return self._available

    def show(self) -> None:
        """Show the icon if a system tray exists; otherwise warn once and carry on."""
        if not QSystemTrayIcon.isSystemTrayAvailable():
            self._available = False
            if not self._warned:
                self._warned = True
                logger.warning(NO_TRAY_WARNING)
            return
        self._available = True
        self._tray.show()

    def hide(self) -> None:
        """Hide the tray icon."""
        self._tray.hide()

    def close(self) -> None:
        """Tear the icon and its menu down safely; call while QApplication is alive.

        Idempotent, and safe to call without a prior show().
        """
        if self._closed:
            return
        self._closed = True
        for signal, slot in (
            (self._hub.state_changed, self._on_state_changed),
            (self._hub.reading_started, self._on_reading_started),
            (self._hub.reading_finished, self._on_reading_finished),
            (self._hub.paused_changed, self._on_paused_changed),
        ):
            try:
                signal.disconnect(slot)
            except (RuntimeError, TypeError):
                logger.debug("hub signal was already disconnected")
        self._tray.hide()
        self._tray.setContextMenu(None)  # type: ignore[arg-type]
        self._menu.deleteLater()

    def _set_actions_enabled(self, pause: bool, stop: bool) -> None:
        """Enable or disable the Pause and Stop entries."""
        self._pause_action.setEnabled(pause)
        self._stop_action.setEnabled(stop)

    def _refresh(self) -> None:
        """Update the tooltip, status line and icon colour from the hub's current state."""
        state = self._hub.state
        if state == READING:
            text = f"Vocalis — reading: {self._hub.source}"
        elif state == LISTENING:
            text = "Vocalis — listening"
        else:
            text = "Vocalis — idle"
        self._status_action.setText(text)
        self._tray.setToolTip(text)
        self._tray.setIcon(_make_icon(COLOURS.get(state, COLOURS[IDLE])))

    def _on_state_changed(self, _state: str) -> None:
        """The overall state changed."""
        if self._closed:
            return
        self._refresh()

    def _on_reading_started(self, _source: str, can_pause: bool) -> None:
        """A reading began: enable Stop, and Pause when supported."""
        if self._closed:
            return
        self._set_actions_enabled(can_pause, True)
        self._pause_action.setText("Pause")
        self._refresh()

    def _on_reading_finished(self) -> None:
        """The reading ended: disable Pause and Stop."""
        if self._closed:
            return
        self._set_actions_enabled(False, False)
        self._pause_action.setText("Pause")

    def _on_paused_changed(self, paused: bool) -> None:
        """Relabel the Pause entry as Resume while paused."""
        if self._closed:
            return
        self._pause_action.setText("Resume" if paused else "Pause")
