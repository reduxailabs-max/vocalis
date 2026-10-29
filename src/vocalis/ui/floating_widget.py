"""The floating mini-widget: a small draggable card shown only while
something is read aloud."""
from typing import Callable

from PySide6.QtCore import QPoint, QRectF, Qt
from PySide6.QtGui import QColor, QGuiApplication, QMouseEvent, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout, QWidget

from vocalis.ui import theme
from vocalis.ui.status_hub import StatusHub

_MARGIN = 16
_CORNER_RADIUS = 8.0
_PADDING_H = 14
_PADDING_V = 12


class FloatingWidget(QWidget):
    """Small always-on-top card shown only while something is read aloud.

    It can be dragged with the mouse. A dragged position is kept until
    Vocalis quits; until the first drag each reading opens bottom-right.
    """

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
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setObjectName(theme.FLOATING_CARD_NAME)
        self._hub = hub
        self._enabled = enabled
        self._drag_offset: QPoint | None = None
        self._user_position: QPoint | None = None
        self._source_label = QLabel("")
        source_font = self._source_label.font()
        source_font.setBold(True)
        self._source_label.setFont(source_font)
        self._status_label = QLabel("")
        self._status_label.setStyleSheet(f"color: {theme.TEXT_MUTED};")
        self._progress_bar = QProgressBar()
        self._progress_bar.setTextVisible(False)
        self._reset_progress_bar()
        self._progress_label = QLabel("")
        self._pause_button = QPushButton("Pause")
        self._stop_button = QPushButton("Stop")
        for button in (self._pause_button, self._stop_button):
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        row = QHBoxLayout()
        row.addWidget(self._pause_button)
        row.addWidget(self._stop_button)
        layout = QVBoxLayout()
        layout.setContentsMargins(_PADDING_H, _PADDING_V, _PADDING_H, _PADDING_V)
        layout.addWidget(self._source_label)
        layout.addWidget(self._status_label)
        layout.addWidget(self._progress_bar)
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

    def paintEvent(self, _event: QPaintEvent) -> None:
        """Draw the rounded dark card behind the labels and buttons."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor(theme.BORDER), 1.0))
        painter.setBrush(QColor(theme.SURFACE))
        painter.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), _CORNER_RADIUS, _CORNER_RADIUS)
        painter.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Start dragging when the left button goes down on the card."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.pos()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Follow the mouse while the left button is held after a press."""
        if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            self._user_position = self.pos()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Stop dragging."""
        self._drag_offset = None
        super().mouseReleaseEvent(event)

    def _reset_progress_bar(self) -> None:
        """Show an empty bar (nothing read yet)."""
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(0)

    def _on_reading_started(self, source: str, can_pause: bool) -> None:
        """Show the new reading's name and, if switched on, the widget."""
        self._source_label.setText(source)
        self._status_label.setText("")
        self._progress_label.setText("")
        self._reset_progress_bar()
        self._pause_button.setText("Pause")
        self._pause_button.setVisible(can_pause)
        if self._enabled():
            self.adjustSize()
            if self._user_position is None:
                self._place_bottom_right()
            else:
                self.move(self._user_position)
            self.show()

    def _on_reading_finished(self) -> None:
        """Hide the widget and clear its labels."""
        self.hide()
        self._drag_offset = None
        self._source_label.setText("")
        self._status_label.setText("")
        self._progress_label.setText("")

    def _on_progress(self, current: int, total: int) -> None:
        """Show which piece of the text is being read, as text and as a bar."""
        self._progress_label.setText(f"piece {current} of {total}")
        self._progress_bar.setRange(0, max(total, 1))
        self._progress_bar.setValue(min(max(current, 0), max(total, 1)))

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
