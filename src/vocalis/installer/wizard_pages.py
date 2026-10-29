"""The building blocks of the setup wizard: worker threads and the five pages."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWizardPage,
)

from vocalis.installer.mic_test import MicResult
from vocalis.installer.plan import InstallPlan
from vocalis.installer.progress import parse_line
from vocalis.installer.runner import PhaseRunner
from vocalis.ui.theme import mark_primary

if TYPE_CHECKING:
    from vocalis.installer.wizard import SetupWizard

INSTALL_LOG_PATH = Path.home() / ".local" / "share" / "vocalis" / "install.log"
FAILURE_MESSAGE = "Something went wrong. Check your internet connection, then press Try again."
_THREAD_WAIT_MS = 5000
_ERROR_STYLE = "color: #ff6b6b;"

RunnerFactory = Callable[[Path, str, Callable[[str], None]], PhaseRunner]


def _append_log(line: str) -> None:
    """Add one line to the install log; a log problem must never stop the install."""
    try:
        INSTALL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with INSTALL_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except (OSError, ValueError):
        pass



class _PhaseThread(QThread):
    """Runs one installer phase off the window thread; only emits signals."""

    line = Signal(str)
    finished_with = Signal(int)

    def __init__(self, factory: RunnerFactory, installer_dir: Path, phase: str) -> None:
        super().__init__()
        self.runner = factory(installer_dir, phase, self.line.emit)

    def run(self) -> None:
        """Run the phase and report its exit code."""
        try:
            code = self.runner.run()
        except Exception as error:  # noqa: BLE001 - any failure must reach the window
            self.line.emit(f"Installer error: {error}")
            code = 1
        self.finished_with.emit(code)


class _CallThread(QThread):
    """Runs one function off the window thread and emits what it returned."""

    result = Signal(object)

    def __init__(self, function: Callable[[], Any], fallback: Any) -> None:
        super().__init__()
        self._function = function
        self._fallback = fallback

    def run(self) -> None:
        """Call the function; on any error emit the fallback instead."""
        try:
            value = self._function()
        except Exception:  # noqa: BLE001 - the window must never crash on a helper
            value = self._fallback
        self.result.emit(value)


class _TextPage(QWizardPage):
    """A page with a title and one block of plain text."""

    def __init__(self, name: str, title: str, text: str) -> None:
        super().__init__()
        self.setObjectName(name)
        self.setTitle(title)
        self.label = QLabel(text)
        self.label.setObjectName(f"{name}_text")
        self.label.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)


class _FoundPage(_TextPage):
    """Shows what was found in this PC and what will be installed."""

    def __init__(self, plan_reader: Callable[[], InstallPlan]) -> None:
        super().__init__("found", "What we found", "Looking at your PC…")
        self._plan_reader = plan_reader

    def initializePage(self) -> None:  # noqa: N802 - Qt name
        """Read the plan and list it in plain English."""
        try:
            self.label.setText("\n".join(self._plan_reader().summary_lines()))
        except Exception:  # noqa: BLE001
            self.label.setText("Could not look at your PC, but setup can still continue.")


class _PhasePage(QWizardPage):
    """Runs one installer phase, shows progress, and offers Try again on failure."""

    def __init__(self, name: str, title: str, phase: str, success_text: str,
                 installer_dir: Path, factory: RunnerFactory) -> None:
        super().__init__()
        self.setObjectName(name)
        self.setTitle(title)
        self._phase, self._success_text = phase, success_text
        self._installer_dir, self._factory = installer_dir, factory
        self._thread: _PhaseThread | None = None
        self._running = self._succeeded = self._cancelled = False
        self.status = QLabel("")
        self.status.setObjectName(f"{name}_status")
        self.status.setWordWrap(True)
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.log_pane = QPlainTextEdit()
        self.log_pane.setReadOnly(True)
        self.error = QLabel(FAILURE_MESSAGE)
        self.error.setObjectName(f"{name}_error")
        self.error.setStyleSheet(_ERROR_STYLE)
        self.error.setWordWrap(True)
        self.retry = QPushButton("Try again")
        self.retry.setObjectName(f"{name}_retry")
        mark_primary(self.retry)
        self.retry.clicked.connect(self._start)
        layout = QVBoxLayout(self)
        for widget in (self.status, self.bar, self.log_pane, self.error, self.retry):
            layout.addWidget(widget)
        self.error.hide()
        self.retry.hide()

    @property
    def phase(self) -> str:
        """The installer phase this page runs."""
        return self._phase

    def initializePage(self) -> None:  # noqa: N802 - Qt name
        """Start the phase as soon as the page appears."""
        if not self._running and not self._succeeded:
            self._start()

    def isComplete(self) -> bool:  # noqa: N802 - Qt name
        """Next/Finish unlock only after the phase succeeded."""
        return self._succeeded

    def _owner(self) -> SetupWizard:
        """The wizard this page lives in."""
        return cast("SetupWizard", self.wizard())

    def _start(self) -> None:
        """Run the phase with a brand-new runner (runners are single-use)."""
        self.error.hide()
        self.retry.hide()
        self.bar.setValue(0)
        self.status.setText("Starting…")
        thread = _PhaseThread(self._factory, self._installer_dir, self._phase)
        thread.line.connect(self._on_line)
        thread.finished_with.connect(self._on_finished)
        self._thread, self._running = thread, True
        thread.start()

    def _on_line(self, line: str) -> None:
        """Log every line; steps move the bar, other lines go to the log pane."""
        _append_log(line)
        event = parse_line(line)
        if event is not None and event.kind == "step":
            self.status.setText(event.text)
            self.bar.setValue((event.index - 1) * 100 // event.total)
            return
        self.log_pane.appendPlainText(event.text if event is not None else line)

    def _on_finished(self, code: int) -> None:
        """Show success, or the failure message with Try again."""
        self._running = False
        if self._cancelled:
            return
        if code == 0:
            self._succeeded = True
            self.bar.setValue(100)
            self.status.setText(self._success_text)
            self._owner().phase_succeeded(self._phase)
        else:
            self.error.show()
            self.retry.show()
            self._owner().phase_failed(FAILURE_MESSAGE)
        self.completeChanged.emit()

    def stop(self) -> None:
        """Stop a running phase (window closed or Cancel)."""
        self._cancelled = True
        if self._running and self._thread is not None:
            self._thread.runner.terminate()
            self._thread.wait(_THREAD_WAIT_MS)
            self._running = False


class _MicrophonePage(QWizardPage):
    """Lets the person try the microphone; a warning never blocks Next."""

    def __init__(self, mic_tester: Callable[[], MicResult]) -> None:
        super().__init__()
        self.setObjectName("microphone")
        self.setTitle("Check your microphone")
        self._mic_tester = mic_tester
        self.thread_: _CallThread | None = None
        self.intro = QLabel("Press the button and say a few words. You can skip this.")
        self.intro.setWordWrap(True)
        self.button = QPushButton("Test my microphone")
        self.button.setObjectName("mic_test_button")
        self.button.clicked.connect(self._start)
        self.result_label = QLabel("")
        self.result_label.setObjectName("mic_result")
        self.result_label.setWordWrap(True)
        layout = QVBoxLayout(self)
        for widget in (self.intro, self.button, self.result_label):
            layout.addWidget(widget)

    def _start(self) -> None:
        """Listen on a helper thread so the window stays responsive."""
        self.button.setEnabled(False)
        self.result_label.setText("Listening…")
        fallback = MicResult("no_device", 0.0, "The microphone test could not run. You can continue.")
        thread = _CallThread(self._mic_tester, fallback)
        thread.result.connect(self._on_result)
        self.thread_ = thread
        thread.start()

    def _on_result(self, result: MicResult) -> None:
        """Show the plain-English outcome."""
        self.result_label.setText(result.message)
        self.button.setEnabled(True)

    def stop(self) -> None:
        """Wait briefly for a running test when the window closes."""
        if self.thread_ is not None:
            self.thread_.wait(_THREAD_WAIT_MS)
