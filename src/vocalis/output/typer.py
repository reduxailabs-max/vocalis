"""Delivers transcribed text at the cursor via ydotool, falling back to the
clipboard (via xclip) if typing fails for any reason (ydotoold not running,
binary missing, timeout, non-zero exit).
"""
import logging
import subprocess
from typing import Protocol

logger = logging.getLogger("vocalis")


class CommandRunner(Protocol):
    """Runs a command, optionally piping bytes to its stdin."""

    def __call__(
        self, argv: list[str], input_bytes: bytes | None = None
    ) -> "subprocess.CompletedProcess[bytes]": ...


class OutputResult:
    """Outcome of attempting to deliver transcribed text to the user."""

    def __init__(self, method: str, text: str) -> None:
        """Create an OutputResult.

        Args:
            method: "typed" or "clipboard" — how the text was delivered.
            text: The text that was delivered.
        """
        self.method = method
        self.text = text


class TextOutput:
    """Delivers text via ydotool typing, falling back to clipboard."""

    def __init__(self, command_runner: CommandRunner | None = None) -> None:
        """Create a TextOutput.

        Args:
            command_runner: Callable(argv, input_bytes=None) -> CompletedProcess,
                used to run ydotool/xclip. Defaults to subprocess.run with a
                5-second timeout. Injectable for tests.
        """
        self._run: CommandRunner = command_runner or _default_command_runner

    def deliver(self, text: str) -> OutputResult:
        """Type text at the cursor via ydotool; fall back to clipboard on failure.

        Args:
            text: The text to deliver. Empty text is a no-op.

        Returns:
            An OutputResult describing how the text was delivered.
        """
        if not text:
            return OutputResult(method="typed", text="")

        try:
            result = self._run(["ydotool", "type", "--", text])
            if result.returncode == 0:
                return OutputResult(method="typed", text=text)
            logger.warning(
                "ydotool exited with code %d; falling back to clipboard", result.returncode
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning("ydotool failed (%s); falling back to clipboard", exc)

        try:
            clip_result = self._run(
                ["xclip", "-selection", "clipboard"], input_bytes=text.encode("utf-8")
            )
            if clip_result.returncode != 0:
                logger.error("xclip exited with code %d", clip_result.returncode)
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.error("Clipboard fallback also failed: %s", exc)

        return OutputResult(method="clipboard", text=text)


def _default_command_runner(
    argv: list[str], input_bytes: bytes | None = None
) -> "subprocess.CompletedProcess[bytes]":
    """Default CommandRunner: subprocess.run with a 5-second timeout."""
    return subprocess.run(argv, input=input_bytes, timeout=5.0, capture_output=True)
