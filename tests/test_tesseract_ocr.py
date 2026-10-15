"""Tests for vocalis.engines.tesseract_ocr.TesseractOCR (fake runner, no real tesseract)."""
import subprocess

import pytest

from vocalis.engines.tesseract_ocr import TesseractError, TesseractOCR


class FakeRunner:
    def __init__(
        self,
        stdout: bytes = b"",
        returncode: int = 0,
        stderr: bytes = b"",
        raise_exc: Exception | None = None,
    ) -> None:
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr
        self.raise_exc = raise_exc
        self.calls: "list[tuple[list[str], bytes]]" = []

    def __call__(self, argv: list[str], input_bytes: bytes) -> "subprocess.CompletedProcess[bytes]":
        self.calls.append((argv, input_bytes))
        if self.raise_exc is not None:
            raise self.raise_exc
        return subprocess.CompletedProcess(
            args=argv, returncode=self.returncode, stdout=self.stdout, stderr=self.stderr
        )


def test_recognize_pipes_the_picture_to_tesseract_and_returns_text() -> None:
    runner = FakeRunner(stdout=b"Hello world\n")
    text = TesseractOCR(command_runner=runner).recognize(b"PNGDATA")
    assert text == "Hello world"
    assert runner.calls == [(["tesseract", "stdin", "stdout", "-l", "eng"], b"PNGDATA")]


def test_recognize_collapses_line_breaks_and_extra_spaces() -> None:
    runner = FakeRunner(stdout=b"  one  two\n\nthree\t four \n")
    assert TesseractOCR(command_runner=runner).recognize(b"x") == "one two three four"


def test_recognize_returns_empty_string_when_nothing_is_found() -> None:
    assert TesseractOCR(command_runner=FakeRunner(stdout=b" \n\n")).recognize(b"x") == ""


def test_missing_binary_gives_an_install_hint() -> None:
    runner = FakeRunner(raise_exc=FileNotFoundError("tesseract"))
    with pytest.raises(TesseractError, match="not installed"):
        TesseractOCR(command_runner=runner).recognize(b"x")


def test_timeout_is_reported() -> None:
    runner = FakeRunner(raise_exc=subprocess.TimeoutExpired(cmd="tesseract", timeout=60.0))
    with pytest.raises(TesseractError, match="too long"):
        TesseractOCR(command_runner=runner).recognize(b"x")


def test_other_os_error_is_reported() -> None:
    runner = FakeRunner(raise_exc=PermissionError("denied"))
    with pytest.raises(TesseractError, match="Could not run Tesseract"):
        TesseractOCR(command_runner=runner).recognize(b"x")


def test_nonzero_exit_is_reported() -> None:
    runner = FakeRunner(returncode=1, stderr=b"Error: bad image")
    with pytest.raises(TesseractError, match="could not read"):
        TesseractOCR(command_runner=runner).recognize(b"x")


def test_custom_binary_and_language_are_used() -> None:
    runner = FakeRunner(stdout=b"hi")
    TesseractOCR(binary="/opt/tess", language="deu", command_runner=runner).recognize(b"x")
    assert runner.calls[0][0] == ["/opt/tess", "stdin", "stdout", "-l", "deu"]
