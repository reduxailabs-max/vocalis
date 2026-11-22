"""Delivers transcribed text to the user.

Short plain-English text is typed with ydotool. Long text, or text with
accents / non-Latin letters (which ydotool cannot type), is pasted: it is put
on the clipboard and Ctrl+V (Ctrl+Shift+V in a terminal) is pressed. If
anything fails, the text is always left on the clipboard. The user can also
choose "clipboard only" mode, which never presses any key.
"""
import logging
import re
import subprocess
from collections.abc import Callable
from typing import Protocol

from vocalis.output.clipboard import ClipboardWriter

logger = logging.getLogger("vocalis")

MODE_TYPE = "type"
MODE_CLIPBOARD = "clipboard"
MODE_BOTH = "both"  # type/paste it AND keep a copy on the clipboard

METHOD_TYPED = "typed"
METHOD_PASTED = "pasted"
METHOD_COPIED = "copied"  # user chose clipboard-only mode
METHOD_CLIPBOARD = "clipboard"  # typing/pasting failed, text is on the clipboard
METHOD_FAILED = "failed"  # nothing worked, not even the clipboard

DEFAULT_TIMEOUT_SECONDS = 5.0
TYPE_MAX_CHARS = 80  # longer plain text is pasted: typing is ~40 ms per character
TYPE_SECONDS_PER_CHAR = 0.08
PASTE_SETTLE_SECONDS = 0.15  # let the clipboard owner register before Ctrl+V

# Linux key codes used by `ydotool key`
_KEY_CTRL, _KEY_SHIFT, _KEY_V = 29, 42, 47
_PASTE_KEYS = [f"{_KEY_CTRL}:1", f"{_KEY_V}:1", f"{_KEY_V}:0", f"{_KEY_CTRL}:0"]
_TERMINAL_PASTE_KEYS = [
    f"{_KEY_CTRL}:1", f"{_KEY_SHIFT}:1", f"{_KEY_V}:1",
    f"{_KEY_V}:0", f"{_KEY_SHIFT}:0", f"{_KEY_CTRL}:0",
]

_TERMINAL_CLASSES = (
    "konsole", "yakuake", "xterm", "uxterm", "urxvt", "rxvt", "gnome-terminal",
    "xfce4-terminal", "terminator", "tilix", "alacritty", "kitty", "wezterm",
    "foot", "lxterminal", "qterminal", "terminology", "guake", "st-256color",
    "mate-terminal", "deepin-terminal", "cool-retro-term", "tabby", "hyper",
    "ptyxis", "kgx", "blackbox", "contour", "ghostty",
)
_WINDOW_ID_PATTERN = re.compile(r"0x[0-9a-fA-F]+")


class CommandRunner(Protocol):
    """Runs a command, optionally piping bytes to its stdin."""

    def __call__(
        self,
        argv: list[str],
        input_bytes: bytes | None = None,
        timeout: float | None = None,
    ) -> "subprocess.CompletedProcess[bytes]": ...


class OutputResult:
    """Outcome of attempting to deliver transcribed text to the user."""

    def __init__(self, method: str, text: str, also_copied: bool = False) -> None:
        """Create an OutputResult.

        Args:
            method: One of "typed" (typed at the cursor), "pasted" (pasted
                with Ctrl+V), "copied" (clipboard-only mode), "clipboard"
                (typing failed, text is on the clipboard) or "failed"
                (not even the clipboard worked).
            text: The text that was delivered.
            also_copied: True when the user asked for "type and also copy"
                and the copy succeeded.
        """
        self.method = method
        self.text = text
        self.also_copied = also_copied


class TextOutput:
    """Types, pastes or copies text, always falling back to the clipboard."""

    def __init__(
        self,
        command_runner: CommandRunner | None = None,
        mode: Callable[[], str] | None = None,
    ) -> None:
        """Create a TextOutput.

        Args:
            command_runner: Callable(argv, input_bytes=None, timeout=None) ->
                CompletedProcess, used to run ydotool / xclip / wl-copy /
                xprop. Defaults to subprocess.run. Injectable for tests.
            mode: Returns "type" (default), "clipboard" or "both", read on every
                delivery so Settings changes apply immediately.
        """
        self._run: CommandRunner = command_runner or _default_command_runner
        self._mode: Callable[[], str] = mode or (lambda: MODE_TYPE)
        self._clipboard = ClipboardWriter(command_runner=self._run)

    def deliver(self, text: str) -> OutputResult:
        """Deliver text according to the chosen mode; never loses the text.

        Args:
            text: The text to deliver. Empty text is a no-op.

        Returns:
            An OutputResult describing how the text was delivered.
        """
        if not text:
            return OutputResult(method=METHOD_TYPED, text="")

        mode = self._mode()
        if mode == MODE_CLIPBOARD:
            return OutputResult(METHOD_COPIED if self._copy(text) else METHOD_FAILED, text)

        keep_copy = mode == MODE_BOTH

        if _is_simple_text(text):
            copied = keep_copy and self._copy(text)
            if self._type(text):
                return OutputResult(METHOD_TYPED, text, also_copied=copied)
            return OutputResult(METHOD_CLIPBOARD if copied or self._copy(text) else METHOD_FAILED, text)

        if not self._copy(text):
            return OutputResult(METHOD_FAILED, text)
        if self._press_paste():
            return OutputResult(METHOD_PASTED, text, also_copied=keep_copy)
        return OutputResult(METHOD_CLIPBOARD, text)

    def _type(self, text: str) -> bool:
        """Type plain text with ydotool; True on success."""
        timeout = DEFAULT_TIMEOUT_SECONDS + TYPE_SECONDS_PER_CHAR * len(text)
        return self._succeeds(["ydotool", "type", "--", text], timeout=timeout)

    def _press_paste(self) -> bool:
        """Press Ctrl+V (Ctrl+Shift+V in a terminal); True on success."""
        keys = _TERMINAL_PASTE_KEYS if self._active_window_is_terminal() else _PASTE_KEYS
        self._sleep(PASTE_SETTLE_SECONDS)
        return self._succeeds(["ydotool", "key", *keys])

    def _copy(self, text: str) -> bool:
        """Put text on the clipboard (xclip, then wl-copy); True on success."""
        if self._clipboard.write(text):
            return True
        return self._succeeds(["wl-copy"], input_bytes=text.encode("utf-8"))

    def _succeeds(
        self,
        argv: list[str],
        input_bytes: bytes | None = None,
        timeout: float | None = None,
    ) -> bool:
        """Run a command; True only if it started and exited with code 0."""
        try:
            result = self._run(argv, input_bytes=input_bytes, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning("%s failed: %s", argv[0], exc)
            return False
        if result.returncode != 0:
            logger.warning("%s exited with code %d", argv[0], result.returncode)
            return False
        return True

    def _active_window_is_terminal(self) -> bool:
        """Best-effort check (X11 only) of whether a terminal has focus."""
        try:
            root = self._run(["xprop", "-root", "_NET_ACTIVE_WINDOW"])
            window_id = next(
                (m for m in _WINDOW_ID_PATTERN.findall(root.stdout.decode(errors="replace"))
                 if int(m, 16) != 0),
                None,
            )
            if root.returncode != 0 or window_id is None:
                return False
            info = self._run(["xprop", "-id", window_id, "WM_CLASS"])
            if info.returncode != 0:
                return False
        except (OSError, subprocess.TimeoutExpired, AttributeError):
            return False
        window_class = info.stdout.decode(errors="replace").lower()
        return any(name in window_class for name in _TERMINAL_CLASSES)

    @staticmethod
    def _sleep(seconds: float) -> None:
        """Pause briefly (a method so tests can silence it)."""
        import time

        time.sleep(seconds)


def _is_simple_text(text: str) -> bool:
    """True for short plain-ASCII text that ydotool can type reliably."""
    return len(text) <= TYPE_MAX_CHARS and text.isascii() and text.isprintable()


def _default_command_runner(
    argv: list[str],
    input_bytes: bytes | None = None,
    timeout: float | None = None,
) -> "subprocess.CompletedProcess[bytes]":
    """Default CommandRunner: subprocess.run with a timeout.

    When text is piped in (xclip / wl-copy), stdout/stderr go to DEVNULL
    because those tools stay alive in the background to serve the clipboard;
    holding their output pipes would make subprocess.run wait for the timeout.
    """
    limit = DEFAULT_TIMEOUT_SECONDS if timeout is None else timeout
    if input_bytes is not None:
        return subprocess.run(
            argv, input=input_bytes, timeout=limit,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    return subprocess.run(argv, timeout=limit, capture_output=True)
