"""Reads clipboard text for Quick Reader — the read-side counterpart to
TextOutput's clipboard-write fallback, using the same xclip-based
CommandRunner seam. ClipboardWriter is the write-side counterpart used by the OCR Reader.
"""
import logging
import subprocess
from typing import Protocol

logger = logging.getLogger("vocalis")


class CommandRunner(Protocol):
    """Runs a command and captures its output."""

    def __call__(self, argv: list[str]) -> "subprocess.CompletedProcess[bytes]": ...


class ClipboardReader:
    """Reads the current clipboard contents as text."""

    def __init__(self, command_runner: CommandRunner | None = None) -> None:
        """Create a ClipboardReader.

        Args:
            command_runner: Callable(argv) -> CompletedProcess, used to
                run xclip. Defaults to subprocess.run with a 5-second
                timeout. Injectable for tests.
        """
        self._run: CommandRunner = command_runner or _default_command_runner

    def read(self) -> str:
        """Read the current clipboard contents.

        Returns:
            The clipboard's text contents, or "" if the clipboard is
            empty or xclip fails for any reason. Never raises.
        """
        try:
            result = self._run(["xclip", "-o", "-selection", "clipboard"])
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning("Failed to read clipboard: %s", exc)
            return ""
        if result.returncode != 0:
            return ""
        return result.stdout.decode("utf-8", errors="replace")


class InputCommandRunner(Protocol):
    """Runs a command, optionally piping bytes to its stdin."""

    def __call__(
        self, argv: list[str], input_bytes: bytes | None = None
    ) -> "subprocess.CompletedProcess[bytes]": ...


class ClipboardWriter:
    """Puts text on the clipboard via xclip."""

    def __init__(self, command_runner: InputCommandRunner | None = None) -> None:
        """Create a ClipboardWriter.

        Args:
            command_runner: Callable(argv, input_bytes) -> CompletedProcess,
                used to run xclip. Defaults to subprocess.run with a
                5-second timeout. Injectable for tests.
        """
        self._run: InputCommandRunner = command_runner or _default_input_runner

    def write(self, text: str) -> bool:
        """Copy `text` to the clipboard.

        Returns:
            True if xclip accepted the text; False on any failure
            (missing binary, timeout, non-zero exit). Never raises.
        """
        try:
            result = self._run(["xclip", "-selection", "clipboard"], input_bytes=text.encode("utf-8"))
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning("Failed to write clipboard: %s", exc)
            return False
        if result.returncode != 0:
            logger.warning("xclip exited with code %d", result.returncode)
            return False
        return True


def _default_command_runner(argv: list[str]) -> "subprocess.CompletedProcess[bytes]":
    """Default CommandRunner: subprocess.run with a 5-second timeout."""
    return subprocess.run(argv, timeout=5.0, capture_output=True)


def _default_input_runner(
    argv: list[str], input_bytes: bytes | None = None
) -> "subprocess.CompletedProcess[bytes]":
    """Default InputCommandRunner: subprocess.run with a 5-second timeout.

    stdout/stderr go to DEVNULL (not pipes) because xclip stays alive in
    the background to serve the selection; holding its output pipes open
    would make subprocess.run wait for it until the timeout.
    """
    return subprocess.run(
        argv,
        input=input_bytes,
        timeout=5.0,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
