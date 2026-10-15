"""Tests for vocalis.output.typer.TextOutput."""
import subprocess

from vocalis.output.typer import TextOutput


class FakeRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], bytes | None]] = []
        self.ydotool_returncode = 0
        self.ydotool_raises: Exception | None = None
        self.xclip_returncode = 0

    def __call__(self, argv: list[str], input_bytes: bytes | None = None):  # type: ignore[no-untyped-def]
        self.calls.append((argv, input_bytes))
        if argv[0] == "ydotool":
            if self.ydotool_raises is not None:
                raise self.ydotool_raises
            return subprocess.CompletedProcess(argv, self.ydotool_returncode)
        if argv[0] == "xclip":
            return subprocess.CompletedProcess(argv, self.xclip_returncode)
        raise AssertionError(f"Unexpected command: {argv}")


def test_empty_text_is_a_noop() -> None:
    runner = FakeRunner()
    output = TextOutput(command_runner=runner)
    result = output.deliver("")
    assert result.method == "typed"
    assert result.text == ""
    assert runner.calls == []


def test_successful_typing_does_not_touch_clipboard() -> None:
    runner = FakeRunner()
    output = TextOutput(command_runner=runner)
    result = output.deliver("hello world")
    assert result.method == "typed"
    assert result.text == "hello world"
    assert len(runner.calls) == 1
    assert runner.calls[0][0][:2] == ["ydotool", "type"]


def test_ydotool_nonzero_exit_falls_back_to_clipboard() -> None:
    runner = FakeRunner()
    runner.ydotool_returncode = 1
    output = TextOutput(command_runner=runner)
    result = output.deliver("hello world")
    assert result.method == "clipboard"
    assert len(runner.calls) == 2
    assert runner.calls[1][0][0] == "xclip"
    assert runner.calls[1][1] == b"hello world"


def test_ydotool_missing_binary_falls_back_to_clipboard() -> None:
    runner = FakeRunner()
    runner.ydotool_raises = FileNotFoundError("ydotool not found")
    output = TextOutput(command_runner=runner)
    result = output.deliver("hello world")
    assert result.method == "clipboard"
    assert len(runner.calls) == 2


def test_ydotool_timeout_falls_back_to_clipboard() -> None:
    runner = FakeRunner()
    runner.ydotool_raises = subprocess.TimeoutExpired(cmd="ydotool", timeout=5.0)
    output = TextOutput(command_runner=runner)
    result = output.deliver("hello world")
    assert result.method == "clipboard"


def test_clipboard_also_failing_does_not_raise() -> None:
    runner = FakeRunner()
    runner.ydotool_returncode = 1
    runner.xclip_returncode = 1
    output = TextOutput(command_runner=runner)
    result = output.deliver("hello world")  # must not raise
    assert result.method == "clipboard"
