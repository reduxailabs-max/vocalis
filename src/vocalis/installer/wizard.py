"""The Vocalis setup wizard: five friendly pages that run the installer phases."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from PySide6.QtWidgets import QMessageBox, QWidget, QWizard

from vocalis.installer.mic_test import MicResult, run_mic_test
from vocalis.installer.plan import InstallPlan, make_plan, read_gpu
from vocalis.installer.runner import PhaseRunner
from vocalis.installer.wizard_pages import (
    DID_NOT_FINISH_MESSAGE,
    RunnerFactory,
    _CallThread,
    _FoundPage,
    _MicrophonePage,
    _PhasePage,
    _TextPage,
    log_hint,
    track_thread,
)

STOPPED_MESSAGE = "Setup was stopped. Run the installer again to finish."
SETTINGS_HINT = "Vocalis is running — open it from the tray icon or the applications menu."
OPENING_TEXT = "Opening Settings… this can take up to a minute"
SETTINGS_TRIES = 3
SETTINGS_RETRY_SECONDS = 1.0


def _pause(seconds: float) -> None:
    """Wait between settings attempts (tests replace this)."""
    time.sleep(seconds)


def _default_plan() -> InstallPlan:
    """Look at the graphics card and decide what to install."""
    return make_plan(read_gpu())


def _default_runner(installer_dir: Path, phase: str, on_line: Callable[[str], None]) -> PhaseRunner:
    """Build the real runner for one installer phase."""
    return PhaseRunner(installer_dir, phase, on_line)


def _default_settings_opener() -> int:
    """Ask the running Vocalis to open its Settings window (0 = opened)."""
    from vocalis.remote import default_socket_path, open_settings

    return open_settings(default_socket_path())



class SetupWizard(QWizard):
    """Welcome, found, installing, microphone and done pages for first-time setup."""

    def __init__(
        self,
        installer_dir: Path,
        plan_reader: Callable[[], InstallPlan] = _default_plan,
        runner_factory: RunnerFactory = _default_runner,
        mic_tester: Callable[[], MicResult] = run_mic_test,
        settings_opener: Callable[[], int] = _default_settings_opener,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Vocalis setup")
        self._settings_opener = settings_opener
        self._succeeded = False
        self._opening = False
        self._closed = False
        self._failure_message = ""
        self._settings_thread: _CallThread | None = None
        self.installing_page = _PhasePage(
            "installing", "Installing", "models", "All set.", installer_dir, runner_factory)
        self.microphone_page = _MicrophonePage(mic_tester)
        self.done_page = _PhasePage(
            "done", "Finishing up", "finish", "Vocalis is running", installer_dir, runner_factory)
        self.addPage(_TextPage(
            "welcome", "Welcome to Vocalis",
            "Vocalis reads text aloud and types what you say. Press Start and we will set "
            "everything up for you."))
        self.addPage(_FoundPage(plan_reader))
        self.addPage(self.installing_page)
        self.addPage(self.microphone_page)
        self.addPage(self.done_page)
        self.setButtonLayout([
            QWizard.WizardButton.Stretch, QWizard.WizardButton.NextButton,
            QWizard.WizardButton.FinishButton, QWizard.WizardButton.CancelButton])
        self.currentIdChanged.connect(self._update_next_text)
        self._update_next_text(0)

    @property
    def exit_code(self) -> int:
        """0 after a finished setup, 1 if it was stopped or failed."""
        return 0 if self._succeeded else 1

    @property
    def final_message(self) -> str:
        """What to tell the person when the exit code is not 0."""
        return f"{self._failure_message or STOPPED_MESSAGE}\n{log_hint()}"

    def phase_succeeded(self, phase: str) -> None:
        """Called by a page when its phase worked; the finish phase ends setup."""
        self._failure_message = ""
        if phase == "finish":
            self._succeeded = True

    def phase_failed(self, message: str) -> None:
        """Called by a page when its phase failed; remembered for the exit message."""
        self._failure_message = message

    def _update_next_text(self, page_id: int) -> None:
        """The first page's button says Start."""
        self.setButtonText(QWizard.WizardButton.NextButton, "Start" if page_id == 0 else "Next")

    def _show_stopped_message(self) -> None:
        """Tell the person setup was stopped (tests replace this)."""
        QMessageBox.information(self, "Vocalis setup", f"{STOPPED_MESSAGE}\n{log_hint()}")

    def _show_settings_hint(self) -> None:
        """Explain how to open Vocalis when Settings could not be opened (tests replace this)."""
        QMessageBox.information(self, "Vocalis setup", SETTINGS_HINT)

    def reject(self) -> None:
        """Closing or Cancel: stop any running phase and say so."""
        self._closed = True
        if not self._succeeded:
            self.installing_page.stop()
            self.done_page.stop()
            self.microphone_page.stop()
            self._show_stopped_message()
        super().reject()

    def accept(self) -> None:
        """Finish: open Settings on a helper thread (it can take a while), then close."""
        if self._opening:
            return
        self._opening = True
        self.done_page.status.setText(OPENING_TEXT)
        finish = self.button(QWizard.WizardButton.FinishButton)
        if finish is not None:
            finish.setEnabled(False)
        thread = _CallThread(self._open_settings_with_retries, False)
        thread.result.connect(self._on_settings_done)
        self._settings_thread = thread
        track_thread(thread)
        thread.start()

    def _open_settings_with_retries(self) -> bool:
        """Try up to SETTINGS_TRIES times, a second apart; True once Settings opened."""
        for attempt in range(SETTINGS_TRIES):
            try:
                if self._settings_opener() == 0:
                    return True
            except Exception:  # noqa: BLE001 - keep trying, never crash
                pass
            if attempt < SETTINGS_TRIES - 1:
                _pause(SETTINGS_RETRY_SECONDS)
        return False

    def _on_settings_done(self, opened: bool) -> None:
        """Explain if Settings did not open, then close the wizard anyway."""
        if self._closed:
            return
        if not opened:
            self._show_settings_hint()
        super().accept()
