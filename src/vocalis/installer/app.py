"""The ``vocalis-setup`` program: opens the setup wizard window.

Exit codes (install.sh relies on them): 0 = finished, 1 = stopped or an
install step failed, 3 = a window could not be opened.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
from collections.abc import Callable
from pathlib import Path
from types import FrameType

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from vocalis.installer.wizard import SetupWizard
from vocalis.installer.wizard_pages import running_threads
from vocalis.ui.theme import apply_theme

EXIT_OK = 0
EXIT_STOPPED = 1
EXIT_NO_WINDOW = 3
NO_WINDOW_MESSAGE = "Vocalis setup could not open a window (no screen was found)."


def _has_display() -> bool:
    """True when Qt has somewhere to draw (a screen, or an explicit platform choice)."""
    return bool(
        os.environ.get("QT_QPA_PLATFORM")
        or os.environ.get("DISPLAY")
        or os.environ.get("WAYLAND_DISPLAY")
    )


def _make_app(argv: list[str]) -> QApplication:
    """Create (or reuse) the Qt application."""
    existing = QApplication.instance()
    if isinstance(existing, QApplication):
        return existing
    return QApplication(argv)


_STOP_SIGNALS = (signal.SIGINT, signal.SIGHUP, signal.SIGTERM)
_SIGNAL_TICK_MS = 200


def install_signal_handlers(
    app: QApplication, wizard: SetupWizard
) -> Callable[[int, FrameType | None], None]:
    """Make Ctrl+C, a closed terminal or a kill stop the running install cleanly.

    The handler stops the running phase (its whole process group), closes the
    window without a dialog and ends the program, so no download is left
    running unseen. A small timer wakes Python so the handler can run.
    """

    def handler(signum: int, frame: FrameType | None) -> None:
        wizard.stop_for_signal()
        app.quit()

    for signum in _STOP_SIGNALS:
        signal.signal(signum, handler)
    timer = QTimer(app)
    timer.timeout.connect(lambda: None)
    timer.start(_SIGNAL_TICK_MS)
    return handler


def _leave(code: int) -> int:
    """Return the exit code; if a worker is still running, exit hard so Qt cannot abort."""
    if running_threads():
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    return code


def main(argv: list[str] | None = None) -> int:
    """Open the wizard and return the process exit code."""
    parser = argparse.ArgumentParser(prog="vocalis-setup")
    parser.add_argument("--installer-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if not _has_display():
        print(NO_WINDOW_MESSAGE, file=sys.stderr)
        return EXIT_NO_WINDOW
    try:
        app = _make_app(sys.argv[:1])
        apply_theme(app)
        wizard = SetupWizard(args.installer_dir)
        wizard.show()
        install_signal_handlers(app, wizard)
    except Exception as error:  # noqa: BLE001 - any start-up problem means "no window"
        print(f"{NO_WINDOW_MESSAGE} ({error})", file=sys.stderr)
        return EXIT_NO_WINDOW
    app.exec()
    if wizard.exit_code != EXIT_OK:
        print(wizard.final_message, file=sys.stderr)
        return _leave(EXIT_STOPPED)
    return _leave(EXIT_OK)
