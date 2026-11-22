"""Tests for vocalis.output.typer.TextOutput."""
import subprocess

import pytest

from vocalis.output import typer
from vocalis.output.typer import TYPE_MAX_CHARS, TextOutput


_REAL_SLEEP = TextOutput._sleep


class FakeRunner:
    """Pretends to be ydotool / xclip / wl-copy / xprop and records every call."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], bytes | None, float | None]] = []
        self.returncodes: dict[str, int] = {}
        self.raises: dict[str, Exception] = {}
        self.window_class = 'WM_CLASS(STRING) = "firefox", "Firefox"'
        self.active_window = "_NET_ACTIVE_WINDOW(WINDOW): window id # 0x7400007, 0x0"

    def commands(self) -> list[str]:
        return [c[0][0] for c in self.calls]

    def __call__(  # type: ignore[no-untyped-def]
        self, argv: list[str], input_bytes: bytes | None = None, timeout: float | None = None
    ):
        self.calls.append((argv, input_bytes, timeout))
        name = argv[0]
        if name in self.raises:
            raise self.raises[name]
        code = self.returncodes.get(name, 0)
        if name == "xprop":
            out = self.active_window if "-root" in argv else self.window_class
            return subprocess.CompletedProcess(argv, code, stdout=out.encode(), stderr=b"")
        if name in ("ydotool", "xclip", "wl-copy"):
            return subprocess.CompletedProcess(argv, code, stdout=b"", stderr=b"")
        raise AssertionError(f"Unexpected command: {argv}")


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(TextOutput, "_sleep", staticmethod(lambda seconds: None))


def _output(runner: FakeRunner, mode: str = "type") -> TextOutput:
    return TextOutput(command_runner=runner, mode=lambda: mode)


def test_empty_text_is_a_noop() -> None:
    runner = FakeRunner()
    result = _output(runner).deliver("")
    assert result.method == "typed"
    assert result.text == ""
    assert runner.calls == []


def test_short_plain_text_is_typed_without_touching_clipboard() -> None:
    runner = FakeRunner()
    result = _output(runner).deliver("hello world")
    assert result.method == "typed"
    assert result.text == "hello world"
    assert runner.commands() == ["ydotool"]
    assert runner.calls[0][0][:2] == ["ydotool", "type"]


def test_typing_timeout_grows_with_text_length() -> None:
    runner = FakeRunner()
    _output(runner).deliver("a" * TYPE_MAX_CHARS)
    timeout = runner.calls[0][2]
    assert timeout is not None and timeout > 5.0 + 0.04 * TYPE_MAX_CHARS


def test_default_mode_is_typing() -> None:
    runner = FakeRunner()
    result = TextOutput(command_runner=runner).deliver("hello")
    assert result.method == "typed"


@pytest.mark.parametrize("failure", ["nonzero", "missing", "timeout"])
def test_typing_failure_falls_back_to_clipboard(failure: str) -> None:
    runner = FakeRunner()
    if failure == "nonzero":
        runner.returncodes["ydotool"] = 1
    elif failure == "missing":
        runner.raises["ydotool"] = FileNotFoundError("ydotool not found")
    else:
        runner.raises["ydotool"] = subprocess.TimeoutExpired(cmd="ydotool", timeout=5.0)
    result = _output(runner).deliver("hello world")
    assert result.method == "clipboard"
    assert runner.commands() == ["ydotool", "xclip"]
    assert runner.calls[1][1] == b"hello world"


def test_typing_and_clipboard_both_failing_reports_failed_without_raising() -> None:
    runner = FakeRunner()
    runner.returncodes.update({"ydotool": 1, "xclip": 1, "wl-copy": 1})
    result = _output(runner).deliver("hello world")
    assert result.method == "failed"
    assert result.text == "hello world"


def test_wl_copy_is_used_when_xclip_fails() -> None:
    runner = FakeRunner()
    runner.returncodes.update({"ydotool": 1, "xclip": 1})
    result = _output(runner).deliver("hello world")
    assert result.method == "clipboard"
    assert runner.commands() == ["ydotool", "xclip", "wl-copy"]
    assert runner.calls[2][1] == b"hello world"


def test_wl_copy_missing_is_handled() -> None:
    runner = FakeRunner()
    runner.returncodes.update({"ydotool": 1, "xclip": 1})
    runner.raises["wl-copy"] = FileNotFoundError("wl-copy not found")
    assert _output(runner).deliver("hi").method == "failed"


def test_long_text_is_pasted_not_typed() -> None:
    runner = FakeRunner()
    text = "word " * 100
    result = _output(runner).deliver(text)
    assert result.method == "pasted"
    assert result.text == text
    assert "type" not in [c[0][1] for c in runner.calls if c[0][0] == "ydotool"]
    assert runner.calls[0][0][0] == "xclip"
    assert runner.calls[0][1] == text.encode("utf-8")
    key_call = [c[0] for c in runner.calls if c[0][:2] == ["ydotool", "key"]][0]
    assert key_call == ["ydotool", "key", "29:1", "47:1", "47:0", "29:0"]


def test_non_english_text_is_pasted_not_typed() -> None:
    runner = FakeRunner()
    text = "Привет, как дела? Café naïve 你好"
    result = _output(runner).deliver(text)
    assert result.method == "pasted"
    assert runner.calls[0][1] == text.encode("utf-8")
    assert all(c[0][1] != "type" for c in runner.calls if c[0][0] == "ydotool")


def test_text_with_newlines_is_pasted() -> None:
    runner = FakeRunner()
    assert _output(runner).deliver("line one\nline two").method == "pasted"


def test_terminal_gets_ctrl_shift_v() -> None:
    runner = FakeRunner()
    runner.window_class = 'WM_CLASS(STRING) = "konsole", "konsole"'
    _output(runner).deliver("x" * 200)
    key_call = [c[0] for c in runner.calls if c[0][:2] == ["ydotool", "key"]][0]
    assert key_call == ["ydotool", "key", "29:1", "42:1", "47:1", "47:0", "42:0", "29:0"]


def test_unknown_active_window_gets_plain_ctrl_v() -> None:
    runner = FakeRunner()
    runner.active_window = "_NET_ACTIVE_WINDOW(WINDOW): window id # 0x0"
    _output(runner).deliver("x" * 200)
    key_call = [c[0] for c in runner.calls if c[0][:2] == ["ydotool", "key"]][0]
    assert key_call == ["ydotool", "key", "29:1", "47:1", "47:0", "29:0"]


@pytest.mark.parametrize("broken", ["missing", "nonzero"])
def test_missing_or_failing_xprop_means_plain_ctrl_v(broken: str) -> None:
    runner = FakeRunner()
    if broken == "missing":
        runner.raises["xprop"] = FileNotFoundError("xprop not found")
    else:
        runner.returncodes["xprop"] = 1
    assert _output(runner).deliver("x" * 200).method == "pasted"
    key_call = [c[0] for c in runner.calls if c[0][:2] == ["ydotool", "key"]][0]
    assert key_call == ["ydotool", "key", "29:1", "47:1", "47:0", "29:0"]


def test_window_class_lookup_failure_means_plain_ctrl_v() -> None:
    class HalfBrokenRunner(FakeRunner):
        def __call__(self, argv, input_bytes=None, timeout=None):  # type: ignore[no-untyped-def]
            if argv[:2] == ["xprop", "-id"]:
                return subprocess.CompletedProcess(argv, 1, stdout=b"", stderr=b"")
            return super().__call__(argv, input_bytes, timeout)

    runner = HalfBrokenRunner()
    assert _output(runner).deliver("x" * 200).method == "pasted"


def test_failed_paste_key_leaves_text_on_clipboard() -> None:
    runner = FakeRunner()
    runner.returncodes["ydotool"] = 1
    result = _output(runner).deliver("x" * 200)
    assert result.method == "clipboard"
    assert runner.calls[0][0][0] == "xclip"


def test_long_text_with_no_clipboard_at_all_reports_failed() -> None:
    runner = FakeRunner()
    runner.returncodes.update({"xclip": 1, "wl-copy": 1})
    result = _output(runner).deliver("x" * 200)
    assert result.method == "failed"
    assert "ydotool" not in runner.commands()


def test_clipboard_mode_only_copies_and_never_presses_keys() -> None:
    runner = FakeRunner()
    result = _output(runner, mode="clipboard").deliver("hello world")
    assert result.method == "copied"
    assert runner.commands() == ["xclip"]


def test_clipboard_mode_failure_reports_failed() -> None:
    runner = FakeRunner()
    runner.returncodes.update({"xclip": 1, "wl-copy": 1})
    assert _output(runner, mode="clipboard").deliver("hi").method == "failed"


def test_mode_is_read_on_every_delivery() -> None:
    runner = FakeRunner()
    mode = {"value": "type"}
    output = TextOutput(command_runner=runner, mode=lambda: mode["value"])
    assert output.deliver("hi").method == "typed"
    mode["value"] = "clipboard"
    assert output.deliver("hi").method == "copied"


def test_default_runner_pipes_text_without_capturing_output(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_run(argv, **kwargs):  # type: ignore[no-untyped-def]
        seen.update(kwargs)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    typer._default_command_runner(["xclip"], input_bytes=b"hi", timeout=7.0)
    assert seen["stdout"] is subprocess.DEVNULL
    assert seen["timeout"] == 7.0
    assert "capture_output" not in seen


def test_default_runner_captures_output_when_no_input(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_run(argv, **kwargs):  # type: ignore[no-untyped-def]
        seen.update(kwargs)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    typer._default_command_runner(["xprop"])
    assert seen["capture_output"] is True
    assert seen["timeout"] == typer.DEFAULT_TIMEOUT_SECONDS


def test_real_sleep_helper_runs() -> None:
    _REAL_SLEEP(0)


def test_both_mode_types_short_text_and_keeps_a_copy() -> None:
    runner = FakeRunner()
    result = _output(runner, mode="both").deliver("hello world")
    assert result.method == "typed"
    assert result.also_copied is True
    assert runner.commands() == ["xclip", "ydotool"]


def test_both_mode_pastes_long_text_and_reports_the_copy() -> None:
    runner = FakeRunner()
    result = _output(runner, mode="both").deliver("x" * 200)
    assert result.method == "pasted"
    assert result.also_copied is True


def test_both_mode_typing_failure_still_leaves_text_on_clipboard() -> None:
    runner = FakeRunner()
    runner.returncodes["ydotool"] = 1
    result = _output(runner, mode="both").deliver("hello world")
    assert result.method == "clipboard"
    assert runner.commands().count("xclip") == 1  # copied once, not twice


def test_both_mode_still_types_when_the_copy_fails() -> None:
    runner = FakeRunner()
    runner.returncodes.update({"xclip": 1, "wl-copy": 1})
    result = _output(runner, mode="both").deliver("hello world")
    assert result.method == "typed"
    assert result.also_copied is False


def test_plain_type_mode_does_not_claim_a_copy() -> None:
    assert _output(FakeRunner()).deliver("hello").also_copied is False
