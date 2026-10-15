"""Desktop notifications for vocalis features, via notify-send. Respects
VocalisConfig.notifications_enabled — when disabled, notify() is a no-op.
"""
import logging
import subprocess
from typing import Callable

logger = logging.getLogger("vocalis")

CommandRunner = Callable[[list[str]], "subprocess.CompletedProcess[bytes]"]


class Notifier:
    """Sends desktop notifications, or does nothing if disabled."""

    def __init__(self, enabled: bool | Callable[[], bool], command_runner: CommandRunner | None = None) -> None:
        """Create a Notifier.

        Args:
            enabled: Whether to send notifications, or a function returning
                that, checked on every notification so a Settings change
                applies immediately.
            command_runner: Callable(argv) -> CompletedProcess, used to run
                notify-send. Defaults to subprocess.run with a 5-second
                timeout. Injectable for tests.
        """
        self._enabled = enabled
        self._run: CommandRunner = command_runner or _default_command_runner

    def notify(self, title: str, message: str = "") -> None:
        """Send a desktop notification, if enabled.

        A failure to send never raises — a missing notification is not
        worth crashing a feature over.

        Args:
            title: Notification title.
            message: Notification body (optional).
        """
        try:
            enabled = self._enabled() if callable(self._enabled) else self._enabled
        except Exception:
            logger.exception("Could not read the notifications setting; skipping notification")
            return
        if not enabled:
            return
        argv = ["notify-send", "--app-name=Vocalis", title]
        if message:
            argv.append(message)
        try:
            self._run(argv)
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning("notify-send failed: %s", exc)


def _default_command_runner(argv: list[str]) -> "subprocess.CompletedProcess[bytes]":
    """Default CommandRunner: subprocess.run with a 5-second timeout."""
    return subprocess.run(argv, timeout=5.0, capture_output=True)
