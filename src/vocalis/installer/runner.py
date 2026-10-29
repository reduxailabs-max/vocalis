"""Runs one phase of the Vocalis installer script and relays its output."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

_COULD_NOT_START_STATUS = 127
_TERMINATED_STATUS = 143
_KILL_GRACE_SECONDS = 3.0


class PhaseRunner:
    """Runs `install.sh --phase <phase>`; terminate() is safe from another thread.

    A runner is single-use: build a new one for each attempt.
    """

    def __init__(
        self,
        installer_dir: Path,
        phase: str,
        on_line: Callable[[str], None],
        popen: Callable[..., Any] = subprocess.Popen,
    ) -> None:
        self._installer_dir = installer_dir
        self._phase = phase
        self._on_line = on_line
        self._popen = popen
        self._lock = threading.Lock()
        self._child: Any = None
        self._terminated = False

    def run(self) -> int:
        """Run the phase to the end and return its exit code (non-zero if stopped)."""
        command = ["bash", str(self._installer_dir / "install.sh"), "--phase", self._phase]
        with self._lock:
            if self._terminated:
                return _TERMINATED_STATUS
            try:
                child = self._popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    errors="replace",
                    start_new_session=True,
                )
            except OSError as error:
                self._on_line(f"Could not start the installer: {error}")
                return _COULD_NOT_START_STATUS
            self._child = child
        try:
            for raw in child.stdout or []:
                line = raw.strip()
                if line:
                    self._on_line(line)
        except BaseException:
            self.terminate()
            child.wait()
            close = getattr(child.stdout, "close", None)
            if close is not None:
                with contextlib.suppress(OSError):
                    close()
            raise
        status = int(child.wait())
        with self._lock:
            was_terminated = self._terminated
        if was_terminated and status == 0:
            return _TERMINATED_STATUS
        return status

    def terminate(self) -> None:
        """Stop the installer and everything it started. Safe to call twice or early."""
        with self._lock:
            already = self._terminated
            self._terminated = True
            child = self._child
        if already or child is None or child.poll() is not None:
            return
        pid = getattr(child, "pid", None)
        if isinstance(pid, int) and pid > 1:
            self._signal_group(pid, signal.SIGTERM)
            timer = threading.Timer(
                _KILL_GRACE_SECONDS, self._signal_group, args=(pid, signal.SIGKILL)
            )
            timer.daemon = True
            timer.start()
        else:
            with contextlib.suppress(OSError):
                child.terminate()

    @staticmethod
    def _signal_group(pid: int, sig: signal.Signals) -> None:
        """Send a signal to the child's whole process group, ignoring if gone."""
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(pid, sig)
