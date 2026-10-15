"""Lets the user drag a box on the screen and returns that region as a PNG.

Uses whichever of these tools is installed, in this order:
spectacle (the box can be resized by dragging its edges/corners before
confirming), maim (draw once), flameshot (draw once; tried last because it
hangs on some X11 setups), or slurp + grim (Wayland, draw once).
Spectacle uses a private temp directory that is deleted on completion.
The other tools keep the picture in memory (stdout), never written to disk.
"""
import logging
import os
import re
import shutil
import subprocess
import tempfile
from typing import Callable, Protocol

logger = logging.getLogger("vocalis")

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
# slurp prints "x,y WxH"; anything else is refused before it reaches grim.
_GEOMETRY_RE = re.compile(r"^[0-9]+,[0-9]+ [0-9]+x[0-9]+$")
# The user is dragging a box by hand, so allow plenty of time.
_CAPTURE_TIMEOUT_SECONDS = 120.0


class ScreenCaptureUnavailable(RuntimeError):
    """No supported screen-capture tool is installed (message is user-readable)."""


class CaptureRunner(Protocol):
    """Runs a command and captures its output."""

    def __call__(self, argv: list[str]) -> "subprocess.CompletedProcess[bytes]": ...


class ScreenRegionCapture:
    """Asks the user to drag a box on screen and returns the picture."""

    def __init__(
        self,
        command_runner: CaptureRunner | None = None,
        which: Callable[[str], str | None] = shutil.which,
    ) -> None:
        """Create a ScreenRegionCapture.

        Args:
            command_runner: Callable(argv) -> CompletedProcess. Defaults to
                subprocess.run with a 120-second timeout. Injectable for tests.
            which: Finds an executable on PATH (returns None if absent).
                Injectable for tests.
        """
        self._run: CaptureRunner = command_runner or _default_command_runner
        self._which = which

    def capture(self) -> bytes | None:
        """Let the user choose a screen region and return it.

        Returns:
            PNG bytes of the region, or None if the user cancelled (Escape,
            closed the tool, empty or non-PNG output, tool timed out or failed).

        Raises:
            ScreenCaptureUnavailable: None of the supported tools is installed.
        """
        if self._which("spectacle"):
            return self._spectacle()
        if self._which("maim"):
            return self._png_from(["maim", "-s", "-u"])
        if self._which("flameshot"):
            return self._png_from(["flameshot", "gui", "--raw"])
        if self._which("slurp") and self._which("grim"):
            return self._slurp_then_grim()
        raise ScreenCaptureUnavailable(
            "No screen-capture tool found. Install one of: spectacle, maim, flameshot, or slurp + grim"
        )

    def _spectacle(self) -> bytes | None:
        """Capture with spectacle, writing to a private temp directory."""
        with tempfile.TemporaryDirectory(prefix="vocalis-ocr-") as tmpdir:
            filepath = os.path.join(tmpdir, "region.png")
            result = self._run_or_none(["spectacle", "-r", "-b", "-n", "-o", filepath])
            if result is None or result.returncode != 0:
                return None
            # Read the file if it exists and is a valid PNG
            if not os.path.isfile(filepath):
                return None
            try:
                with open(filepath, "rb") as f:
                    data = f.read()
                if not data or not data.startswith(_PNG_MAGIC):
                    return None
                return data
            except OSError:
                return None

    def _slurp_then_grim(self) -> bytes | None:
        """Pick a region with slurp, then grab exactly that region with grim."""
        result = self._run_or_none(["slurp"])
        if result is None or result.returncode != 0:
            return None
        geometry = result.stdout.decode("utf-8", errors="replace").strip()
        if not _GEOMETRY_RE.match(geometry):
            logger.warning("slurp printed an unexpected region, ignoring it")
            return None
        return self._png_from(["grim", "-g", geometry, "-"])

    def _png_from(self, argv: list[str]) -> bytes | None:
        """Run a capture command and return its stdout if it is a PNG."""
        result = self._run_or_none(argv)
        if result is None or result.returncode != 0:
            return None
        if not result.stdout.startswith(_PNG_MAGIC):
            return None
        return result.stdout

    def _run_or_none(self, argv: list[str]) -> "subprocess.CompletedProcess[bytes] | None":
        """Run a command; None if it could not be run or timed out."""
        try:
            return self._run(argv)
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning("Screen capture command %s failed: %s", argv[0], exc)
            return None


def _default_command_runner(argv: list[str]) -> "subprocess.CompletedProcess[bytes]":
    """Default CaptureRunner: subprocess.run with a 120-second timeout."""
    return subprocess.run(argv, timeout=_CAPTURE_TIMEOUT_SECONDS, capture_output=True)
