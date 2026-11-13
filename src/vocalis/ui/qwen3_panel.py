"""Settings: add or remove the optional Qwen3-TTS voice.

Adding or removing runs the kept copy of the installer (``install.sh --phase
qwen3-add`` / ``qwen3-remove``) and shows its progress in a small dialog,
exactly like the setup window does.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from vocalis.config import VocalisConfig
from vocalis.installer.progress_display import ProgressDisplay
from vocalis.installer.runner import PhaseRunner
from vocalis.installer.wizard_pages import FAILURE_MESSAGE, PhaseThread, RunnerFactory, track_thread
from vocalis.qwen3_addon import qwen3_model_present, qwen3_packages_installed, runtime_installer_dir
from vocalis.ui.theme import mark_primary

ADD_PHASE = "qwen3-add"
REMOVE_PHASE = "qwen3-remove"
STATUS_INSTALLED = "Installed."
STATUS_MISSING = "Not installed."
STATUS_PARTIAL = "Only partly installed. Press “Download and install” to finish."
STATUS_NO_INSTALLER = (
    "The installer files were not found, so this cannot be changed from here. "
    "Run the Vocalis installer again to repair it."
)
NOTE_TEXT = (
    "An extra, optional voice. It needs a strong NVIDIA graphics card and about 8 GB of space. "
    "Kokoro already sounds natural without it."
)
REMOVE_QUESTION = (
    "Remove Qwen3-TTS? This deletes its voice files and helper programs (about 8 GB). "
    "You can add it again later."
)
STOPPED_TEXT = "Stopped. Press the button again to continue where it stopped."
ADDED_TEXT = "Qwen3-TTS is ready. Choose it in the voice engine lists below."
REMOVED_TEXT = "Qwen3-TTS was removed. Restart Vocalis to free its memory if it was in use."


def _default_runner(
    installer_dir: Path, phase: str, on_line: Callable[[str], None], extra_args: Sequence[str]
) -> PhaseRunner:
    """Build the real runner for one installer phase."""
    return PhaseRunner(installer_dir, phase, on_line, extra_args=extra_args)


class PhaseDialog(QDialog):
    """Runs one installer phase and shows its progress; Stop keeps partial downloads."""

    outcome = Signal(bool)

    def __init__(
        self,
        title: str,
        phase: str,
        installer_dir: Path,
        factory: RunnerFactory,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(480)
        self._phase, self._installer_dir, self._factory = phase, installer_dir, factory
        self._thread: PhaseThread | None = None
        self._running = self._stopped = False
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.log_pane = QPlainTextEdit()
        self.log_pane.setReadOnly(True)
        self.button = QPushButton("Stop")
        self.button.clicked.connect(self._on_button)
        self._display = ProgressDisplay(self.bar, self.status, self.log_pane)
        layout = QVBoxLayout(self)
        for widget in (self.status, self.bar, self.log_pane):
            layout.addWidget(widget)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self.button)
        layout.addLayout(row)

    @property
    def is_running(self) -> bool:
        """Whether the phase is still going."""
        return self._running

    def start(self) -> None:
        """Run the phase on a helper thread."""
        self._display.start()
        thread = PhaseThread(self._factory, self._installer_dir, self._phase)
        thread.line.connect(self._on_line)
        thread.finished_with.connect(self._on_finished)
        self._thread, self._running = thread, True
        track_thread(thread)
        thread.start()

    def _on_line(self, line: str) -> None:
        """Show one installer line (a Qt slot, so it runs on the window thread)."""
        self._display.handle(line)

    def _on_button(self) -> None:
        """Stop while running, Close afterwards."""
        if self._running:
            self._stop()
        else:
            self.accept()

    def _stop(self) -> None:
        """Ask the running phase to stop (partial downloads are kept)."""
        self._stopped = True
        self.button.setEnabled(False)
        self.status.setText("Stopping…")
        if self._thread is not None:
            self._thread.runner.terminate()

    def reject(self) -> None:
        """Closing the dialog while it runs means Stop; it closes once the phase has ended."""
        if self._running:
            self._stop()
            return
        super().reject()

    def _on_finished(self, code: int) -> None:
        """Show the end state and tell the panel how it went."""
        self._running = False
        succeeded = code == 0 and not self._stopped
        self._display.finish(succeeded, "Done.")
        if self._stopped:
            self.status.setText(STOPPED_TEXT)
        elif not succeeded:
            self.status.setText(FAILURE_MESSAGE)
        self.button.setText("Close")
        self.button.setEnabled(True)
        self.outcome.emit(succeeded)


class Qwen3Panel(QGroupBox):
    """Shows whether Qwen3-TTS is installed, with buttons to add or remove it."""

    changed = Signal()

    def __init__(
        self,
        config: Callable[[], VocalisConfig],
        installer_dir: Callable[[], Path] = runtime_installer_dir,
        runner_factory: RunnerFactory = _default_runner,
        packages_installed: Callable[[], bool] = qwen3_packages_installed,
        model_present: Callable[[Path], bool] = qwen3_model_present,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__("Qwen3-TTS (optional extra voice)", parent)
        self._config, self._installer_dir, self._runner_factory = config, installer_dir, runner_factory
        self._packages_installed, self._model_present = packages_installed, model_present
        self._dialog: PhaseDialog | None = None
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.note = QLabel(NOTE_TEXT)
        self.note.setWordWrap(True)
        self.add_button = QPushButton("Download and install")
        self.remove_button = QPushButton("Remove")
        mark_primary(self.add_button)
        self.add_button.clicked.connect(self._on_add)
        self.remove_button.clicked.connect(self._on_remove)
        layout = QVBoxLayout(self)
        layout.addWidget(self.status)
        row = QHBoxLayout()
        row.addWidget(self.add_button)
        row.addWidget(self.remove_button)
        row.addStretch(1)
        layout.addLayout(row)
        layout.addWidget(self.note)
        self.refresh()

    def _parts(self) -> tuple[bool, bool]:
        """(helper programs installed, voice model present)."""
        model_dir = Path(self._config().engines.qwen3_tts_model_dir)
        return self._packages_installed(), self._model_present(model_dir)

    def is_installed(self) -> bool:
        """Whether Qwen3-TTS can be used right now (helper programs and voice model)."""
        return all(self._parts())

    def refresh(self) -> None:
        """Update the status text and which buttons can be pressed."""
        packages, model = self._parts()
        fully, nothing = packages and model, not packages and not model
        has_installer = (self._installer_dir() / "install.sh").is_file()
        if not has_installer:
            self.status.setText(STATUS_NO_INSTALLER)
        else:
            self.status.setText(STATUS_INSTALLED if fully else STATUS_MISSING if nothing else STATUS_PARTIAL)
        busy = self._dialog is not None and self._dialog.is_running
        self.add_button.setEnabled(has_installer and not fully and not busy)
        self.remove_button.setEnabled(has_installer and not nothing and not busy)

    def _confirm_remove(self) -> bool:
        """Ask before deleting (tests replace this)."""
        answer = QMessageBox.question(self, "Remove Qwen3-TTS", REMOVE_QUESTION)
        return answer == QMessageBox.StandardButton.Yes

    def _on_add(self) -> None:
        """Download and install the voice."""
        self._run(ADD_PHASE, "Adding Qwen3-TTS", ADDED_TEXT)

    def _on_remove(self) -> None:
        """Remove the voice after a confirmation."""
        if self._confirm_remove():
            self._run(REMOVE_PHASE, "Removing Qwen3-TTS", REMOVED_TEXT)

    def _run(self, phase: str, title: str, done_text: str) -> None:
        """Open the progress dialog and run the installer phase in it."""
        dialog = PhaseDialog(title, phase, self._installer_dir(), self._runner_factory, self)
        dialog.outcome.connect(lambda ok: self._on_outcome(ok, done_text))
        self._dialog = dialog
        dialog.open()
        dialog.start()
        self.refresh()

    def _on_outcome(self, succeeded: bool, done_text: str) -> None:
        """A phase ended: refresh, and tell the window when something changed."""
        if succeeded and self._dialog is not None:
            self._dialog.status.setText(done_text)
        self.refresh()
        self.changed.emit()
