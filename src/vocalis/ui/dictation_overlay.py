"""The floating dictation animation shown at the bottom of the screen.

While the dictation key is held it shows a glowing microphone in pulsing
red rings with sound waves on both sides; the waves follow the live
microphone volume. After the key is released it shows "Transcribing…",
then fades away. It never takes keyboard focus and ignores mouse clicks.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import (
    QColor,
    QFont,
    QGuiApplication,
    QPainter,
    QPainterPath,
    QPen,
    QRadialGradient,
)
from PySide6.QtWidgets import QWidget

from vocalis.features.dictation import PHASE_DONE, PHASE_FAILED, PHASE_LISTENING, PHASE_TRANSCRIBING
from vocalis.ui import theme

OVERLAY_WIDTH = 480
OVERLAY_HEIGHT = 120
BOTTOM_MARGIN = 48
FRAME_MS = 16  # about 60 frames a second
ATTACK = 0.55  # how fast the waves rise with the voice (0-1 per frame)
RELEASE = 0.08  # how slowly they fall back
LEVEL_GAIN = 6.0  # microphone volume is small; scale it up for the picture
FADE_SECONDS = 0.35
DONE_HOLD_SECONDS = 0.25
FAILED_HOLD_SECONDS = 1.2
WAVE_LINES = 5
WAVE_SPAN = 150.0
_LABELS = {
    PHASE_TRANSCRIBING: "Transcribing…",
    PHASE_FAILED: "Didn't catch that",
}

_FILL = QColor(13, 13, 15, 225)
_BORDER = QColor(142, 14, 20, 200)


def smooth_level(previous: float, target: float) -> float:
    """Move the shown volume towards the real volume: fast up, slow down."""
    target = max(0.0, min(1.0, target * LEVEL_GAIN))
    rate = ATTACK if target > previous else RELEASE
    return previous + (target - previous) * rate


def overlay_position(available: QRectF, width: int = OVERLAY_WIDTH, height: int = OVERLAY_HEIGHT) -> QPointF:
    """Top-left corner that centres the overlay near the bottom of ``available``."""
    x = available.x() + (available.width() - width) / 2
    y = available.y() + available.height() - height - BOTTOM_MARGIN
    return QPointF(x, y)


class DictationOverlay(QWidget):
    """A frameless, click-through pill that animates while you dictate."""

    def __init__(
        self,
        enabled: Callable[[], bool] = lambda: True,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create the (hidden) overlay.

        Args:
            enabled: Asked each time dictation starts; False keeps it hidden.
            clock: Seconds clock, replaceable in tests.
        """
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus
            | Qt.WindowType.WindowTransparentForInput,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setFixedSize(OVERLAY_WIDTH, OVERLAY_HEIGHT)
        self._enabled = enabled
        self._clock = clock
        self._phase = ""
        self._level = 0.0
        self._target = 0.0
        self._started = clock()
        self._fade_from: float | None = None
        self._hide_at: float | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(FRAME_MS)
        self._timer.timeout.connect(self._tick)

    # -- inputs (call from the GUI thread, e.g. through Qt signals) -------
    @property
    def phase(self) -> str:
        """The current phase ("" when hidden)."""
        return self._phase

    @property
    def is_animating(self) -> bool:
        """Whether the animation timer is running."""
        return self._timer.isActive()

    def set_phase(self, phase: str) -> None:
        """React to a dictation phase: listening, transcribing, done or failed."""
        if phase == PHASE_LISTENING:
            if not self._enabled():
                return
            self._begin()
        elif not self.isVisible():
            return
        self._phase = phase
        if phase == PHASE_DONE:
            self._schedule_hide(DONE_HOLD_SECONDS)
        elif phase == PHASE_FAILED:
            self._schedule_hide(FAILED_HOLD_SECONDS)
        else:
            self._hide_at = None

    def set_level(self, level: float) -> None:
        """Remember the latest microphone volume (0 to about 1)."""
        self._target = level

    # -- internals ---------------------------------------------------------
    def _begin(self) -> None:
        """Show the overlay at the bottom of the screen and start animating."""
        self._level = 0.0
        self._target = 0.0
        self._fade_from = None
        self._hide_at = None
        self._started = self._clock()
        self._place()
        self.setWindowOpacity(1.0)
        self.show()
        self._timer.start()

    def _place(self) -> None:
        """Move to the bottom centre of the screen the mouse is on."""
        screen = QGuiApplication.screenAt(self.cursor().pos()) or QGuiApplication.primaryScreen()
        if screen is None:
            return
        position = overlay_position(QRectF(screen.availableGeometry()))
        self.move(int(position.x()), int(position.y()))

    def _schedule_hide(self, hold_seconds: float) -> None:
        """Fade out after ``hold_seconds``."""
        self._hide_at = self._clock() + hold_seconds
        self._fade_from = None

    def _tick(self) -> None:
        """Advance one animation frame."""
        self._level = smooth_level(self._level, self._target if self._phase == PHASE_LISTENING else 0.0)
        now = self._clock()
        if self._hide_at is not None and now >= self._hide_at:
            if self._fade_from is None:
                self._fade_from = now
            progress = (now - self._fade_from) / FADE_SECONDS
            if progress >= 1.0:
                self._finish()
                return
            self.setWindowOpacity(1.0 - progress)
        self.update()

    def _finish(self) -> None:
        """Hide completely and stop the timer so nothing runs while idle."""
        self._timer.stop()
        self.hide()
        self._phase = ""
        self._fade_from = None
        self._hide_at = None

    # -- drawing -----------------------------------------------------------
    def paintEvent(self, event: object) -> None:  # noqa: N802 - Qt API
        """Draw the pill, rings, mic and waves."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        seconds = self._clock() - self._started
        self._draw_pill(painter)
        centre = QPointF(self.width() / 2, 52.0)
        self._draw_waves(painter, centre, seconds)
        self._draw_rings(painter, centre, seconds)
        self._draw_mic(painter, centre)
        self._draw_label(painter)
        painter.end()

    def _draw_pill(self, painter: QPainter) -> None:
        """The dark rounded background."""
        rect = QRectF(4, 4, self.width() - 8, self.height() - 8)
        painter.setPen(QPen(_BORDER, 2))
        painter.setBrush(_FILL)
        painter.drawRoundedRect(rect, rect.height() / 2, rect.height() / 2)

    def _draw_rings(self, painter: QPainter, centre: QPointF, seconds: float) -> None:
        """Three red rings that pulse with the voice (or spin-pulse while transcribing)."""
        glow = QRadialGradient(centre, 46)
        glow.setColorAt(0.0, QColor(229, 57, 53, int(110 + 100 * self._level)))
        glow.setColorAt(1.0, QColor(142, 14, 20, 0))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(glow)
        painter.drawEllipse(centre, 46, 46)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        busy = self._phase == PHASE_TRANSCRIBING
        for index, base in enumerate((22.0, 30.0, 38.0)):
            pulse = (0.5 + 0.5 * math.sin(seconds * 4.0 - index * 0.8)) if busy else self._level
            radius = base + 3.0 * pulse * (index + 1)
            alpha = int(90 + 110 * pulse) if index else int(160 + 80 * pulse)
            colour = QColor(theme.ACCENT_BRIGHT if index == 0 else theme.ACCENT)
            colour.setAlpha(min(255, alpha))
            painter.setPen(QPen(colour, 2.2 - index * 0.3))
            painter.drawEllipse(centre, radius, radius)

    def _draw_mic(self, painter: QPainter, centre: QPointF) -> None:
        """A small microphone in the middle."""
        colour = QColor(theme.ACCENT_BRIGHT)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(colour)
        painter.drawRoundedRect(QRectF(centre.x() - 5, centre.y() - 13, 10, 18), 5, 5)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(colour, 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        arc = QPainterPath()
        arc.moveTo(centre.x() - 9, centre.y() - 2)
        arc.arcTo(QRectF(centre.x() - 9, centre.y() - 11, 18, 18), 180, 180)
        painter.drawPath(arc)
        painter.drawLine(QPointF(centre.x(), centre.y() + 7), QPointF(centre.x(), centre.y() + 12))
        painter.drawLine(QPointF(centre.x() - 5, centre.y() + 12), QPointF(centre.x() + 5, centre.y() + 12))

    def _draw_waves(self, painter: QPainter, centre: QPointF, seconds: float) -> None:
        """Layered red sine waves left and right of the mic; height follows the voice."""
        resting = 0.06
        height = 8.0 + 34.0 * max(resting, self._level)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for side in (-1, 1):
            for line in range(WAVE_LINES):
                path = QPainterPath()
                phase = seconds * 5.0 + line * 0.55
                steps = 48
                for step in range(steps + 1):
                    t = step / steps
                    distance = 52.0 + t * WAVE_SPAN
                    envelope = math.sin(math.pi * t) ** 1.4
                    wobble = math.sin(t * 9.0 * (1 + line * 0.07) + phase) * height * envelope
                    point = QPointF(centre.x() + side * distance, centre.y() + wobble * (1 - 0.12 * line))
                    if step == 0:
                        path.moveTo(point)
                    else:
                        path.lineTo(point)
                mix = line / max(1, WAVE_LINES - 1)
                colour = QColor(theme.ACCENT_BRIGHT if mix < 0.5 else theme.ACCENT)
                colour.setAlpha(int(230 - 140 * mix))
                painter.setPen(QPen(colour, 2.0 - 0.8 * mix))
                painter.drawPath(path)

    def _draw_label(self, painter: QPainter) -> None:
        """Small text under the mic for the non-listening phases."""
        text = _LABELS.get(self._phase)
        if not text:
            return
        font = QFont(self.font())
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(theme.WHITE))
        painter.drawText(
            QRectF(0, self.height() - 34, self.width(), 24), Qt.AlignmentFlag.AlignCenter, text
        )
