"""Runs one phase of the Vocalis installer script and relays its output."""

from __future__ import annotations

import subprocess
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

_COULD_NOT_START_STATUS = 127
_TERMINATED_STATUS = 143


class PhaseRunner:
    """Runs `install.sh --phase <phase>`; terminate() is safe from another thread."""

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
                )
            except OSError as error:
                self._on_line(f"Could not start the installer: {error}")
                return _COULD_NOT_START_STATUS
            self._child = child
        for raw in child.stdout or []:
            line = raw.strip()
            if line:
                self._on_line(line)
        status = int(child.wait())
        with self._lock:
            was_terminated = self._terminated
        if was_terminated and status == 0:
            return _TERMINATED_STATUS
        return status

    def terminate(self) -> None:
        """Stop the running installer, if any. Safe to call twice or before run()."""
        with self._lock:
            self._terminated = True
            child = self._child
        if child is not None and child.poll() is None:
            try:
                child.terminate()
            except OSError:
                pass
