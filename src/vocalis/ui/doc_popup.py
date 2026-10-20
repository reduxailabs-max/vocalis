"""The Document Reader popup window: Open file / paste text / Read /
Pause-Resume / Stop, with a progress label.

Talks to a DocumentReaderFeature only through its public methods and the
three callables it takes at construction (on_status/on_progress/
on_finished, wired in cli.py's build_doc_reader() to this popup's *.emit
methods) -- this module is the only place PySide6 is imported for
Document Reader, keeping doc_reader.py itself independently testable
without a display.

Cross-thread signal emission: DocumentReaderFeature calls on_status/
on_progress/on_finished from its own background worker thread, never
from the Qt UI thread. Each is wired to emit a Qt signal here rather than
touch a widget directly -- Qt auto-queues a signal's delivery onto the
receiving object's own thread when sender and receiver live on different
threads, which is what makes this safe without any locking of our own.

The Ctrl+Alt+D hotkey thread (CtrlAltKeyListener's evdev watcher) reaches
the popup the same way: only through the `hotkey_pressed` signal, never by
calling a widget method directly.
"""
from pathlib import Path
from typing import Protocol

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

_SUPPORTED_FILE_FILTER = "Documents (*.txt *.md *.pdf *.docx *.html *.htm);;All files (*)"


class DocumentReaderLike(Protocol):
    """The subset of DocumentReaderFeature this popup depends on."""

    @property
    def is_reading(self) -> bool: ...
    def read_file(self, path: Path) -> None: ...
    def read_text(self, text: str) -> None: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...
    def stop(self) -> None: ...


class DocumentReaderPopup(QWidget):
    """The Ctrl+Alt+D popup window.

    The hotkey listener's evdev thread must never call on_hotkey_trigger()
    directly (it touches widgets); it emits `hotkey_pressed` instead, and
    Qt queues the slot call onto this popup's own (GUI) thread.
    """

    status_changed = Signal(str)
    progress_changed = Signal(int, int)
    reading_finished = Signal()
    reading_started = Signal()
    hotkey_pressed = Signal()
    external_reading_started = Signal(str)

    def __init__(self, feature: DocumentReaderLike) -> None:
        """Create the popup, wired to the given feature.

        Args:
            feature: The DocumentReaderFeature this popup controls.
        """
        super().__init__()
        self._feature = feature
        self._paused = False
        self.setWindowTitle("Vocalis — Document Reader")

        self._text_box = QPlainTextEdit()
        self._text_box.setPlaceholderText("Paste text here, or open a file below...")

        self._open_button = QPushButton("Open file...")
        self._read_button = QPushButton("Read")
        self._pause_button = QPushButton("Pause")
        self._stop_button = QPushButton("Stop")
        self._status_label = QLabel("Idle")
        self._progress_label = QLabel("")

        self._open_button.clicked.connect(self._on_open_clicked)
        self._read_button.clicked.connect(self._on_read_clicked)
        self._pause_button.clicked.connect(self._on_pause_clicked)
        self._stop_button.clicked.connect(self._on_stop_clicked)

        self.status_changed.connect(self._status_label.setText)
        self.progress_changed.connect(self._update_progress_label)
        self.reading_finished.connect(self._on_reading_finished)
        self.reading_started.connect(lambda: self._set_reading_controls_enabled(True))
        self.hotkey_pressed.connect(self.on_hotkey_trigger)
        self.external_reading_started.connect(self.on_external_reading_started)

        button_row = QHBoxLayout()
        button_row.addWidget(self._open_button)
        button_row.addWidget(self._read_button)
        button_row.addWidget(self._pause_button)
        button_row.addWidget(self._stop_button)

        layout = QVBoxLayout()
        layout.addWidget(self._text_box)
        layout.addLayout(button_row)
        layout.addWidget(self._status_label)
        layout.addWidget(self._progress_label)
        self.setLayout(layout)

        self._set_reading_controls_enabled(False)

    def on_hotkey_trigger(self) -> None:
        """Call when Ctrl+Alt+D is pressed: stop an in-progress reading,
        or raise/focus this window if idle."""
        if self._feature.is_reading:
            self._feature.stop()
            return
        self.show()
        self.raise_()
        self.activateWindow()

    def on_external_reading_started(self, text: str) -> None:
        """Show a reading that another feature (the OCR Reader) is about to
        start on the shared DocumentReaderFeature: put its text in the box
        and enable Pause/Stop so the user can control it.

        Does not call the feature (the other feature starts the reading)
        and does not take keyboard focus, so it never steals focus from
        the window the user is working in.
        """
        self._text_box.setPlainText(text)
        self._reset_pause_state()
        self._set_reading_controls_enabled(True)
        self.show()
        self.raise_()

    def _on_open_clicked(self) -> None:
        """Handle Open file button: prompt for a file path and read it via the feature."""
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Open document", "", _SUPPORTED_FILE_FILTER
        )
        if not path_str:
            return
        self._reset_pause_state()
        self._set_reading_controls_enabled(True)
        self._feature.read_file(Path(path_str))

    def _on_read_clicked(self) -> None:
        """Handle Read button: send the text box content to the feature for reading."""
        text = self._text_box.toPlainText()
        if not text.strip():
            self._status_label.setText("Nothing to read")
            return
        self._reset_pause_state()
        self._set_reading_controls_enabled(True)
        self._feature.read_text(text)

    def _on_pause_clicked(self) -> None:
        """Handle Pause button: toggle between pause and resume, tracked by
        an explicit flag. Ignored (and the state reset) if nothing is reading."""
        if not self._feature.is_reading:
            self._reset_pause_state()
            return
        if self._paused:
            self._feature.resume()
            self._paused = False
            self._pause_button.setText("Pause")
        else:
            self._feature.pause()
            self._paused = True
            self._pause_button.setText("Resume")

    def _reset_pause_state(self) -> None:
        """Mark the popup as not paused and show the "Pause" label."""
        self._paused = False
        self._pause_button.setText("Pause")

    def _on_stop_clicked(self) -> None:
        """Handle Stop button: stop the current reading via the feature."""
        self._feature.stop()

    def _on_reading_finished(self) -> None:
        """Handle reading_finished signal: disable controls and reset pause button label."""
        self._set_reading_controls_enabled(False)
        self._reset_pause_state()

    def _update_progress_label(self, current: int, total: int) -> None:
        """Update progress label with current and total piece counts."""
        self._progress_label.setText(f"piece {current} of {total}")

    def _set_reading_controls_enabled(self, reading: bool) -> None:
        """Enable/disable read and open buttons when idle; enable pause and stop when reading."""
        self._open_button.setEnabled(not reading)
        self._read_button.setEnabled(not reading)
        self._pause_button.setEnabled(reading)
        self._stop_button.setEnabled(reading)
