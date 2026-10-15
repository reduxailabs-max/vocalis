"""The status board that the tray icon and the floating widget listen to.

Features (dictation, Quick Reader, Document Reader, OCR Reader) report what
they are doing; the tray and widget connect to the signals and never reach
into the features. All reporting methods are safe to call from any thread:
state changes happen under a short lock, and signals are emitted after the
lock is released (Qt delivers them queued to widgets living on the GUI
thread).
"""
import threading
from dataclasses import dataclass
from typing import Callable

from PySide6.QtCore import QObject, Signal

IDLE = "idle"
LISTENING = "listening"
READING = "reading"


@dataclass(frozen=True)
class ReadingControls:
    """What the widget/tray may do to the active reading."""

    stop: Callable[[], None]
    pause: Callable[[], None] | None = None
    resume: Callable[[], None] | None = None

    @property
    def can_pause(self) -> bool:
        """Whether this reading supports pause and resume."""
        return self.pause is not None and self.resume is not None


class StatusHub(QObject):
    """Thread-safe status board with Qt signals.

    Threading contract: every method may be called from any thread, but
    each feature must call begin_reading()/end_reading() for its own
    readings from a consistent thread (or otherwise in order), so its
    begin always comes before its matching end. Signals are emitted
    outside the lock; Qt delivers them to the GUI thread.
    """

    state_changed = Signal(str)
    status_changed = Signal(str)
    progress_changed = Signal(int, int)
    reading_started = Signal(str, bool)
    reading_finished = Signal()
    paused_changed = Signal(bool)

    def __init__(self, parent: QObject | None = None) -> None:
        """Create an idle hub."""
        super().__init__(parent)
        self._lock = threading.Lock()
        self._listening = False
        self._controls: ReadingControls | None = None
        self._source = ""
        self._next_source: str | None = None
        self._paused = False
        self._last_state = IDLE

    # -- queries -----------------------------------------------------------
    @property
    def state(self) -> str:
        """"reading", else "listening", else "idle"."""
        with self._lock:
            return self._compute_state_locked()

    @property
    def is_reading(self) -> bool:
        """Whether a reading is active (including while paused)."""
        with self._lock:
            return self._controls is not None

    @property
    def is_paused(self) -> bool:
        """Whether the active reading is paused."""
        with self._lock:
            return self._paused

    @property
    def source(self) -> str:
        """Label of the active reading ("" when none)."""
        with self._lock:
            return self._source

    def _compute_state_locked(self) -> str:
        """State derived from the fields; caller holds the lock."""
        if self._controls is not None:
            return READING
        return LISTENING if self._listening else IDLE

    def _take_state_change_locked(self) -> str | None:
        """If the derived state changed, remember and return it (else None); caller holds the lock."""
        state = self._compute_state_locked()
        if state == self._last_state:
            return None
        self._last_state = state
        return state

    # -- reporting (any thread) -------------------------------------------
    def set_listening(self, active: bool) -> None:
        """Dictation started/stopped listening to the microphone."""
        with self._lock:
            self._listening = active
            changed = self._take_state_change_locked()
        if changed is not None:
            self.state_changed.emit(changed)

    def set_next_source(self, label: str) -> None:
        """Name the next reading differently from the default (used once)."""
        with self._lock:
            self._next_source = label

    def begin_reading(self, source: str, controls: ReadingControls) -> None:
        """A reading started. Ignored if one is already being shown."""
        with self._lock:
            if self._controls is not None:
                self._next_source = None  # must not leak onto a later reading
                return
            self._controls = controls
            self._source = self._next_source or source
            self._next_source = None
            self._paused = False
            label = self._source
            changed = self._take_state_change_locked()
        if changed is not None:
            self.state_changed.emit(changed)
        self.reading_started.emit(label, controls.can_pause)

    def end_reading(self) -> None:
        """The reading ended for any reason. Ignored if none is active."""
        with self._lock:
            if self._controls is None:
                return
            was_paused = self._paused
            self._controls = None
            self._source = ""
            self._paused = False
            changed = self._take_state_change_locked()
        if was_paused:
            self.paused_changed.emit(False)
        self.reading_finished.emit()
        if changed is not None:
            self.state_changed.emit(changed)

    def report_status(self, text: str) -> None:
        """A human-readable status line (e.g. "Reading...")."""
        self.status_changed.emit(text)

    def report_progress(self, current: int, total: int) -> None:
        """Progress through the text: piece `current` of `total`."""
        self.progress_changed.emit(current, total)

    def set_paused(self, paused: bool) -> None:
        """The reader confirms it paused/resumed."""
        with self._lock:
            if self._controls is None or self._paused == paused:
                return
            self._paused = paused
        self.paused_changed.emit(paused)

    # -- commands (GUI thread) ---------------------------------------------
    def pause(self) -> None:
        """Ask the active reading to pause (no-op if none or not pausable)."""
        with self._lock:
            controls = self._controls
        if controls is not None and controls.pause is not None and controls.can_pause:
            controls.pause()

    def resume(self) -> None:
        """Ask the active reading to resume (no-op if none or not pausable)."""
        with self._lock:
            controls = self._controls
        if controls is not None and controls.resume is not None and controls.can_pause:
            controls.resume()

    def toggle_pause(self) -> None:
        """Pause if playing, resume if paused."""
        if self.is_paused:
            self.resume()
        else:
            self.pause()

    def stop(self) -> None:
        """Ask the active reading to stop (no-op if none)."""
        with self._lock:
            controls = self._controls
        if controls is not None:
            controls.stop()
