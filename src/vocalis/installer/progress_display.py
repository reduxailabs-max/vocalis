"""Turns installer output lines into a progress bar and a plain-English status.

Shared by the setup wizard and the Settings window's Qwen3-TTS dialog, so a
download looks the same everywhere: a smooth bar, "x of y", the speed and the
time left.
"""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QPlainTextEdit, QProgressBar

from vocalis.installer.download_progress import format_bytes, format_eta, format_speed
from vocalis.installer.progress import ProgressEvent, parse_line

_BAR_STEPS = 1000
_PERCENT_STEPS = 100


class ProgressDisplay:
    """Drives one bar, one status label and one log pane from installer lines."""

    def __init__(self, bar: QProgressBar, status: QLabel, log_pane: QPlainTextEdit) -> None:
        self._bar, self._status, self._log_pane = bar, status, log_pane
        self._step_text = ""

    def start(self) -> None:
        """Show 'busy' until the first STEP line arrives."""
        self._step_text = ""
        self._bar.setRange(0, 0)
        self._status.setText("Starting…")

    def handle(self, line: str) -> ProgressEvent | None:
        """Show one line. Returns what it was (None for a plain line).

        STEP lines move the bar and set the status text, PROGRESS lines show
        a download (and are not added to the log pane), anything else goes to
        the log pane.
        """
        event = parse_line(line)
        if event is not None and event.kind == "progress":
            self._show_download(event)
        elif event is not None and event.kind == "step":
            self._step_text = event.text
            self._status.setText(event.text)
            self._bar.setRange(0, _PERCENT_STEPS)
            self._bar.setValue((event.index - 1) * _PERCENT_STEPS // event.total)
        else:
            self._log_pane.appendPlainText(event.text if event is not None else line)
        return event

    def finish(self, succeeded: bool, text: str = "") -> None:
        """Show the end state: a full bar and the text on success, an empty bar otherwise."""
        self._bar.setRange(0, _PERCENT_STEPS)
        self._bar.setValue(_PERCENT_STEPS if succeeded else 0)
        if succeeded:
            self._status.setText(text)

    def _show_download(self, event: ProgressEvent) -> None:
        """Smooth bar plus 'x of y · speed · time left'."""
        self._bar.setRange(0, _BAR_STEPS)
        self._bar.setValue(event.done * _BAR_STEPS // event.total)
        parts = [f"{format_bytes(event.done)} of {format_bytes(event.total)}"]
        if event.speed > 0:
            parts.append(format_speed(event.speed))
            eta = format_eta(None if event.eta < 0 else event.eta)
            if eta:
                parts.append(eta)
        self._status.setText(f"{self._step_text} — {' · '.join(parts)}")
