"""The floating mini-widget: shown only while something is read aloud."""
from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from vocalis.ui.status_hub import StatusHub

_MARGIN = 16


class FloatingWidget(QWidget):
    """Small always-on-top box shown only while something is read aloud."""

    def __init__(self, hub: StatusHub, enabled: Callable[[], bool]) -> None:
        """Build the widget (hidden) and connect it to the status hub."""
        super().__init__(
            None,
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowDoesNotAcceptFocus,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self._hub = hub
        self._enabled = enabled
        self._source_label = QLabel("")
        self._status_label = QLabel("")
        self._progress_label = QLabel("")
        self._pause_button = QPushButton("Pause")
        self._stop_button = QPushButton("Stop")
        for button in (self._pause_button, self._stop_button):
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        row = QHBoxLayout()
        row.addWidget(self._pause_button)
        row.addWidget(self._stop_button)
        layout = QVBoxLayout()
        layout.addWidget(self._source_label)
        layout.addWidget(self._status_label)
        layout.addWidget(self._progress_label)
        layout.addLayout(row)
        self.setLayout(layout)
        hub.reading_started.connect(self._on_reading_started)
        hub.reading_finished.connect(self._on_reading_finished)
        hub.status_changed.connect(self._status_label.setText)
        hub.progress_changed.connect(self._on_progress)
        hub.paused_changed.connect(self._on_paused_changed)
        self._pause_button.clicked.connect(hub.toggle_pause)
        self._stop_button.clicked.connect(hub.stop)

    def _on_reading_started(self, source: str, can_pause: bool) -> None:
        """Show the new reading's name and, if switched on, the widget."""
        self._source_label.setText(source)
        self._status_label.setText("")
        self._progress_label.setText("")
        self._pause_button.setText("Pause")
        self._pause_button.setVisible(can_pause)
        if self._enabled():
            self.adjustSize()
            self._place_bottom_right()
            self.show()

    def _on_reading_finished(self) -> None:
        """Hide the widget and clear its labels."""
        self.hide()
        self._source_label.setText("")
        self._status_label.setText("")
        self._progress_label.setText("")

    def _on_progress(self, current: int, total: int) -> None:
        """Show which piece of the text is being read."""
        self._progress_label.setText(f"piece {current} of {total}")

    def _on_paused_changed(self, paused: bool) -> None:
        """Make the button say what it will do next."""
        self._pause_button.setText("Resume" if paused else "Pause")

    def apply_enabled(self) -> None:
        """Hide the widget right away if it is visible but now switched off."""
        if self.isVisible() and not self._enabled():
            self.hide()

    def _place_bottom_right(self) -> None:
        """Move the widget to the bottom-right corner of the primary screen."""
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        area = screen.availableGeometry()
        self.move(
            area.right() + 1 - self.width() - _MARGIN,
            area.bottom() + 1 - self.height() - _MARGIN,
        )
