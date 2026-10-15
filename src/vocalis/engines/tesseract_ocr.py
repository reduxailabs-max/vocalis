"""Tesseract OCR engine — reads text out of a picture for the OCR Reader.

Runs the system `tesseract` binary on the CPU, feeding the picture on
stdin and reading the text from stdout. It never touches the GPU, so it
is deliberately NOT routed through ModelManager (single-residency only
applies to GPU engines).
"""
import logging
import subprocess
from typing import Protocol

logger = logging.getLogger("vocalis")

_OCR_TIMEOUT_SECONDS = 60.0


class TesseractError(RuntimeError):
    """Tesseract could not be run or could not read the picture.

    The message is written for the end user (it is shown in a
    notification as-is).
    """


class OcrCommandRunner(Protocol):
    """Runs a command with bytes piped to its stdin."""

    def __call__(self, argv: list[str], input_bytes: bytes) -> "subprocess.CompletedProcess[bytes]": ...


class TesseractOCR:
    """Recognises text in a PNG picture using the tesseract binary."""

    def __init__(
        self,
        binary: str = "tesseract",
        language: str = "eng",
        command_runner: OcrCommandRunner | None = None,
    ) -> None:
        """Create a TesseractOCR.

        Args:
            binary: Name or path of the tesseract executable.
            language: Tesseract language pack to use (only "eng" is installed).
            command_runner: Callable(argv, input_bytes) -> CompletedProcess.
                Defaults to subprocess.run with a 60-second timeout.
                Injectable for tests.
        """
        self._binary = binary
        self._language = language
        self._run: OcrCommandRunner = command_runner or _default_command_runner

    def recognize(self, png_bytes: bytes) -> str:
        """Read the text in a picture.

        Args:
            png_bytes: The picture, PNG-encoded.

        Returns:
            The recognised text with all runs of whitespace (including line
            breaks) collapsed to single spaces; "" if nothing was found.

        Raises:
            TesseractError: Tesseract is missing, timed out, or failed.
        """
        argv = [self._binary, "stdin", "stdout", "-l", self._language]
        try:
            result = self._run(argv, png_bytes)
        except FileNotFoundError as exc:
            logger.error("tesseract binary not found: %s", exc)
            raise TesseractError(
                "Tesseract is not installed. Install it with: sudo apt install tesseract-ocr tesseract-ocr-eng"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            logger.error("tesseract timed out")
            raise TesseractError("Tesseract took too long to read the picture") from exc
        except OSError as exc:
            logger.error("Failed to run tesseract: %s", exc)
            raise TesseractError(f"Could not run Tesseract: {exc}") from exc
        if result.returncode != 0:
            logger.error("tesseract exited with code %d: %r", result.returncode, result.stderr)
            raise TesseractError("Tesseract could not read the picture")
        return " ".join(result.stdout.decode("utf-8", errors="replace").split())


def _default_command_runner(argv: list[str], input_bytes: bytes) -> "subprocess.CompletedProcess[bytes]":
    """Default OcrCommandRunner: subprocess.run with a 60-second timeout."""
    return subprocess.run(argv, input=input_bytes, timeout=_OCR_TIMEOUT_SECONDS, capture_output=True)
