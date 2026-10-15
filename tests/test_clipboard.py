"""Tests for vocalis.output.clipboard.ClipboardReader.

Uses a FakeCommandRunner instead of a real xclip subprocess.
"""
import subprocess

from vocalis.output.clipboard import ClipboardReader, ClipboardWriter, _default_input_runner


class FakeCommandRunner:
    def __init__(self, returncode: int = 0, stdout: bytes = b"", raise_exc: Exception | None = None) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.raise_exc = raise_exc
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str]) -> "subprocess.CompletedProcess[bytes]":
        self.calls.append(argv)
        if self.raise_exc is not None:
            raise self.raise_exc
        return subprocess.CompletedProcess(args=argv, returncode=self.returncode, stdout=self.stdout)


def test_read_returns_clipboard_text() -> None:
    runner = FakeCommandRunner(returncode=0, stdout=b"hello world")
    reader = ClipboardReader(command_runner=runner)
    assert reader.read() == "hello world"
    assert runner.calls == [["xclip", "-o", "-selection", "clipboard"]]


def test_read_returns_empty_string_on_nonzero_exit() -> None:
    # xclip exits nonzero when the clipboard is empty/unavailable.
    runner = FakeCommandRunner(returncode=1, stdout=b"")
    reader = ClipboardReader(command_runner=runner)
    assert reader.read() == ""


def test_read_returns_empty_string_when_runner_raises() -> None:
    runner = FakeCommandRunner(raise_exc=OSError("xclip not found"))
    reader = ClipboardReader(command_runner=runner)
    assert reader.read() == ""


def test_read_returns_empty_string_on_timeout() -> None:
    runner = FakeCommandRunner(raise_exc=subprocess.TimeoutExpired(cmd="xclip", timeout=5.0))
    reader = ClipboardReader(command_runner=runner)
    assert reader.read() == ""


def test_read_decodes_utf8_with_replacement_on_bad_bytes() -> None:
    runner = FakeCommandRunner(returncode=0, stdout=b"caf\xc3\xa9")
    reader = ClipboardReader(command_runner=runner)
    assert reader.read() == "café"


class FakeInputRunner:
    def __init__(self, returncode: int = 0, raise_exc: Exception | None = None) -> None:
        self.returncode = returncode
        self.raise_exc = raise_exc
        self.calls: "list[tuple[list[str], bytes | None]]" = []

    def __call__(
        self, argv: list[str], input_bytes: bytes | None = None
    ) -> "subprocess.CompletedProcess[bytes]":
        self.calls.append((argv, input_bytes))
        if self.raise_exc is not None:
            raise self.raise_exc
        return subprocess.CompletedProcess(args=argv, returncode=self.returncode)


def test_write_sends_text_to_xclip_and_returns_true() -> None:
    runner = FakeInputRunner()
    assert ClipboardWriter(command_runner=runner).write("héllo") is True
    assert runner.calls == [(["xclip", "-selection", "clipboard"], "héllo".encode("utf-8"))]


def test_write_returns_false_on_nonzero_exit() -> None:
    assert ClipboardWriter(command_runner=FakeInputRunner(returncode=1)).write("x") is False


def test_write_returns_false_when_xclip_is_missing() -> None:
    runner = FakeInputRunner(raise_exc=FileNotFoundError("xclip"))
    assert ClipboardWriter(command_runner=runner).write("x") is False


def test_write_returns_false_on_timeout() -> None:
    runner = FakeInputRunner(raise_exc=subprocess.TimeoutExpired(cmd="xclip", timeout=5.0))
    assert ClipboardWriter(command_runner=runner).write("x") is False


def test_default_input_runner_sends_xclip_output_to_devnull(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    seen: dict[str, object] = {}

    def recorder(argv: list[str], **kwargs: object) -> "subprocess.CompletedProcess[bytes]":
        seen["argv"] = argv
        seen["kwargs"] = kwargs
        return subprocess.CompletedProcess(args=argv, returncode=0)

    monkeypatch.setattr("vocalis.output.clipboard.subprocess.run", recorder)

    result = _default_input_runner(["xclip", "-selection", "clipboard"], input_bytes=b"x")

    assert result.returncode == 0
    assert seen["argv"] == ["xclip", "-selection", "clipboard"]
    assert seen["kwargs"] == {
        "input": b"x",
        "timeout": 5.0,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
