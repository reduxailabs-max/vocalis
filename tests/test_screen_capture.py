"""Tests for vocalis.output.screen_capture.ScreenRegionCapture (fake tools, no screen)."""
import os
import subprocess
from typing import Callable

import pytest

from vocalis.output.screen_capture import ScreenCaptureUnavailable, ScreenRegionCapture

PNG = b"\x89PNG\r\n\x1a\n" + b"fakeimagebytes"


class FakeRunner:
    """Answers each argv from a dict keyed by the tool name (argv[0])."""

    def __init__(self, answers: "dict[str, subprocess.CompletedProcess[bytes] | Exception]") -> None:
        self.answers = answers
        self.calls: "list[list[str]]" = []

    def __call__(self, argv: list[str]) -> "subprocess.CompletedProcess[bytes]":
        self.calls.append(argv)
        answer = self.answers[argv[0]]
        if isinstance(answer, Exception):
            raise answer
        return answer


def done(stdout: bytes = b"", returncode: int = 0) -> "subprocess.CompletedProcess[bytes]":
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout)


def installed(*names: str) -> Callable[[str], str | None]:
    return lambda name: f"/usr/bin/{name}" if name in names else None


# ===== TOOL ORDER TESTS (pin exactly which tool runs when) =====


def test_spectacle_runs_when_all_tools_installed() -> None:
    """When all tools installed: only spectacle is tried."""
    def spectacle_runner(argv: list[str]) -> "subprocess.CompletedProcess[bytes]":
        if argv[0] == "spectacle" and "-o" in argv:
            idx = argv.index("-o")
            if idx + 1 < len(argv):
                with open(argv[idx + 1], "wb") as f:
                    f.write(PNG)
        return done()
    runner = FakeRunner({"spectacle": done(), "maim": done(PNG), "flameshot": done(PNG), "slurp": done(b"10,20 300x400\n"), "grim": done(PNG)})

    def combined_runner(argv: list[str]) -> "subprocess.CompletedProcess[bytes]":
        if argv[0] == "spectacle":
            return spectacle_runner(argv)
        return runner(argv)

    capture = ScreenRegionCapture(command_runner=combined_runner, which=installed("spectacle", "maim", "flameshot", "slurp", "grim"))
    result = capture.capture()
    assert result == PNG
    assert runner.calls == []  # No calls to maim/flameshot/slurp/grim


def test_maim_runs_when_spectacle_missing() -> None:
    """When only maim (and flameshot/slurp/grim) installed: only maim is tried."""
    runner = FakeRunner({"maim": done(PNG), "flameshot": done(PNG), "slurp": done(b"10,20 300x400\n"), "grim": done(PNG)})
    capture = ScreenRegionCapture(command_runner=runner, which=installed("maim", "flameshot", "slurp", "grim"))
    result = capture.capture()
    assert result == PNG
    assert runner.calls == [["maim", "-s", "-u"]]


def test_flameshot_runs_when_spectacle_maim_missing() -> None:
    """When spectacle/maim missing but flameshot/slurp/grim installed: only flameshot is tried."""
    runner = FakeRunner({"flameshot": done(PNG), "slurp": done(b"10,20 300x400\n"), "grim": done(PNG)})
    capture = ScreenRegionCapture(command_runner=runner, which=installed("flameshot", "slurp", "grim"))
    result = capture.capture()
    assert result == PNG
    assert runner.calls == [["flameshot", "gui", "--raw"]]


def test_slurp_then_grim_when_others_missing() -> None:
    """When only slurp/grim installed: slurp then grim are tried."""
    runner = FakeRunner({"slurp": done(b"10,20 300x400\n"), "grim": done(PNG)})
    capture = ScreenRegionCapture(command_runner=runner, which=installed("slurp", "grim"))
    result = capture.capture()
    assert result == PNG
    assert runner.calls == [["slurp"], ["grim", "-g", "10,20 300x400", "-"]]


# ===== SPECTACLE COMMAND AND CLEANUP TESTS =====


def test_spectacle_command_format() -> None:
    """Spectacle command is exactly ["spectacle","-r","-b","-n","-o",<path_to_region.png>]."""
    recorded_argv = None

    def spectacle_runner(argv: list[str]) -> "subprocess.CompletedProcess[bytes]":
        nonlocal recorded_argv
        recorded_argv = argv
        if argv[0] == "spectacle" and "-o" in argv:
            idx = argv.index("-o")
            if idx + 1 < len(argv):
                with open(argv[idx + 1], "wb") as f:
                    f.write(PNG)
        return done()

    capture = ScreenRegionCapture(command_runner=spectacle_runner, which=installed("spectacle"))
    result = capture.capture()

    assert result == PNG
    assert recorded_argv is not None
    assert recorded_argv[:5] == ["spectacle", "-r", "-b", "-n", "-o"]
    assert recorded_argv[5].endswith("region.png")
    assert "vocalis-ocr-" in recorded_argv[5]
    assert os.path.dirname(recorded_argv[5]) not in ["", "."]  # Has a directory


def test_spectacle_temp_dir_cleaned_up_on_success() -> None:
    """Spectacle's temp directory is deleted after successful capture."""
    temp_dir_path = None

    def spectacle_runner(argv: list[str]) -> "subprocess.CompletedProcess[bytes]":
        nonlocal temp_dir_path
        if argv[0] == "spectacle" and "-o" in argv:
            idx = argv.index("-o")
            if idx + 1 < len(argv):
                temp_dir_path = os.path.dirname(argv[idx + 1])
                with open(argv[idx + 1], "wb") as f:
                    f.write(PNG)
        return done()

    capture = ScreenRegionCapture(command_runner=spectacle_runner, which=installed("spectacle"))
    result = capture.capture()

    assert result == PNG
    assert temp_dir_path is not None
    assert not os.path.exists(temp_dir_path), f"Temp dir {temp_dir_path} should be deleted after capture"


def test_spectacle_temp_dir_cleaned_up_on_cancel() -> None:
    """Spectacle's temp directory is deleted when spectacle exits non-zero."""
    temp_dir_path = None

    def spectacle_runner(argv: list[str]) -> "subprocess.CompletedProcess[bytes]":
        nonlocal temp_dir_path
        if argv[0] == "spectacle" and "-o" in argv:
            idx = argv.index("-o")
            if idx + 1 < len(argv):
                temp_dir_path = os.path.dirname(argv[idx + 1])
        return done(b"", returncode=1)

    capture = ScreenRegionCapture(command_runner=spectacle_runner, which=installed("spectacle"))
    result = capture.capture()

    assert result is None
    assert temp_dir_path is not None
    assert not os.path.exists(temp_dir_path), f"Temp dir {temp_dir_path} should be deleted on cancel"


def test_spectacle_temp_dir_cleaned_up_on_no_file() -> None:
    """Spectacle's temp directory is deleted when no file was written."""
    temp_dir_path = None

    def spectacle_runner(argv: list[str]) -> "subprocess.CompletedProcess[bytes]":
        nonlocal temp_dir_path
        if argv[0] == "spectacle" and "-o" in argv:
            idx = argv.index("-o")
            if idx + 1 < len(argv):
                temp_dir_path = os.path.dirname(argv[idx + 1])
        return done()  # Success but file not written

    capture = ScreenRegionCapture(command_runner=spectacle_runner, which=installed("spectacle"))
    result = capture.capture()

    assert result is None
    assert temp_dir_path is not None
    assert not os.path.exists(temp_dir_path), f"Temp dir {temp_dir_path} should be deleted"


def test_spectacle_temp_dir_cleaned_up_on_non_png() -> None:
    """Spectacle's temp directory is deleted when file is not PNG."""
    temp_dir_path = None

    def spectacle_runner(argv: list[str]) -> "subprocess.CompletedProcess[bytes]":
        nonlocal temp_dir_path
        if argv[0] == "spectacle" and "-o" in argv:
            idx = argv.index("-o")
            if idx + 1 < len(argv):
                temp_dir_path = os.path.dirname(argv[idx + 1])
                with open(argv[idx + 1], "wb") as f:
                    f.write(b"not a png")
        return done()

    capture = ScreenRegionCapture(command_runner=spectacle_runner, which=installed("spectacle"))
    result = capture.capture()

    assert result is None
    assert temp_dir_path is not None
    assert not os.path.exists(temp_dir_path), f"Temp dir {temp_dir_path} should be deleted"


def test_spectacle_temp_dir_cleaned_up_on_empty_file() -> None:
    """Spectacle's temp directory is deleted when file is empty."""
    temp_dir_path = None

    def spectacle_runner(argv: list[str]) -> "subprocess.CompletedProcess[bytes]":
        nonlocal temp_dir_path
        if argv[0] == "spectacle" and "-o" in argv:
            idx = argv.index("-o")
            if idx + 1 < len(argv):
                temp_dir_path = os.path.dirname(argv[idx + 1])
                open(argv[idx + 1], "wb").close()
        return done()

    capture = ScreenRegionCapture(command_runner=spectacle_runner, which=installed("spectacle"))
    result = capture.capture()

    assert result is None
    assert temp_dir_path is not None
    assert not os.path.exists(temp_dir_path), f"Temp dir {temp_dir_path} should be deleted"


def test_spectacle_temp_dir_cleaned_up_on_timeout() -> None:
    """Spectacle's temp directory is cleaned up even on timeout."""
    temp_dir_path = None

    def spectacle_runner(argv: list[str]) -> "subprocess.CompletedProcess[bytes]":
        nonlocal temp_dir_path
        if argv[0] == "spectacle" and "-o" in argv:
            idx = argv.index("-o")
            if idx + 1 < len(argv):
                temp_dir_path = os.path.dirname(argv[idx + 1])
        raise subprocess.TimeoutExpired(cmd="spectacle", timeout=120.0)

    capture = ScreenRegionCapture(command_runner=spectacle_runner, which=installed("spectacle"))
    result = capture.capture()

    assert result is None
    assert temp_dir_path is not None
    assert not os.path.exists(temp_dir_path), f"Temp dir {temp_dir_path} should be deleted on timeout"


def test_spectacle_temp_dir_cleaned_up_on_os_error() -> None:
    """Spectacle's temp directory is cleaned up even on OSError."""
    temp_dir_path = None

    def spectacle_runner(argv: list[str]) -> "subprocess.CompletedProcess[bytes]":
        nonlocal temp_dir_path
        if argv[0] == "spectacle" and "-o" in argv:
            idx = argv.index("-o")
            if idx + 1 < len(argv):
                temp_dir_path = os.path.dirname(argv[idx + 1])
        raise PermissionError("nope")

    capture = ScreenRegionCapture(command_runner=spectacle_runner, which=installed("spectacle"))
    result = capture.capture()

    assert result is None
    assert temp_dir_path is not None
    assert not os.path.exists(temp_dir_path), f"Temp dir {temp_dir_path} should be deleted on OSError"


def test_spectacle_exit_1_with_valid_png_means_none() -> None:
    """Spectacle exits 1 but wrote a valid PNG: still returns None (exit code wins)."""
    temp_dir_path = None

    def spectacle_runner(argv: list[str]) -> "subprocess.CompletedProcess[bytes]":
        nonlocal temp_dir_path
        if argv[0] == "spectacle" and "-o" in argv:
            idx = argv.index("-o")
            if idx + 1 < len(argv):
                temp_dir_path = os.path.dirname(argv[idx + 1])
                with open(argv[idx + 1], "wb") as f:
                    f.write(PNG)
        return done(b"", returncode=1)

    capture = ScreenRegionCapture(command_runner=spectacle_runner, which=installed("spectacle"))
    result = capture.capture()

    assert result is None, "Non-zero exit should return None even if PNG was written"
    assert temp_dir_path is not None
    assert not os.path.exists(temp_dir_path), f"Temp dir {temp_dir_path} should be deleted even when exit code wins"


def test_spectacle_non_oserror_exception_propagates_and_cleans_up() -> None:
    """Non-OSError exception raised by runner propagates and temp directory is cleaned up."""
    temp_dir_path = None

    def spectacle_runner(argv: list[str]) -> "subprocess.CompletedProcess[bytes]":
        nonlocal temp_dir_path
        if argv[0] == "spectacle" and "-o" in argv:
            idx = argv.index("-o")
            if idx + 1 < len(argv):
                temp_dir_path = os.path.dirname(argv[idx + 1])
        raise RuntimeError("Unexpected error")

    capture = ScreenRegionCapture(command_runner=spectacle_runner, which=installed("spectacle"))

    with pytest.raises(RuntimeError, match="Unexpected error"):
        capture.capture()

    assert temp_dir_path is not None
    assert not os.path.exists(temp_dir_path), f"Temp dir {temp_dir_path} should be deleted even on unexpected exception"


# ===== STDOUT TOOLS (PNG FROM) TESTS =====


@pytest.mark.parametrize("tool_name,cmd", [("maim", ["maim", "-s", "-u"]), ("flameshot", ["flameshot", "gui", "--raw"])])
def test_stdout_tool_non_zero_exit_means_none(tool_name: str, cmd: list[str]) -> None:
    """Maim/flameshot with non-zero exit returns None, even if stdout is empty."""
    runner = FakeRunner({tool_name: done(b"", returncode=1)})
    capture = ScreenRegionCapture(command_runner=runner, which=installed(tool_name))
    assert capture.capture() is None
    assert runner.calls == [cmd]


@pytest.mark.parametrize("tool_name", ["maim", "flameshot"])
def test_stdout_tool_timeout_means_cancelled(tool_name: str) -> None:
    """Maim/flameshot timeout (user takes too long) means None."""
    runner = FakeRunner({tool_name: subprocess.TimeoutExpired(cmd=tool_name, timeout=120.0)})
    assert ScreenRegionCapture(command_runner=runner, which=installed(tool_name)).capture() is None


@pytest.mark.parametrize("tool_name", ["maim", "flameshot"])
def test_stdout_tool_os_error_means_cancelled(tool_name: str) -> None:
    """Maim/flameshot OSError (permission denied, etc.) means None."""
    runner = FakeRunner({tool_name: PermissionError("nope")})
    assert ScreenRegionCapture(command_runner=runner, which=installed(tool_name)).capture() is None


# ===== OTHER CANCELLATION TESTS =====


def test_slurp_alone_is_not_enough() -> None:
    """Slurp requires grim to complete the capture."""
    capture = ScreenRegionCapture(command_runner=FakeRunner({}), which=installed("slurp"))
    with pytest.raises(ScreenCaptureUnavailable):
        capture.capture()


def test_no_tool_at_all_raises_with_an_install_hint() -> None:
    """Error message lists all supported tools including spectacle."""
    capture = ScreenRegionCapture(command_runner=FakeRunner({}), which=installed())
    with pytest.raises(ScreenCaptureUnavailable, match="spectacle"):
        capture.capture()


def test_stdout_tool_non_png_means_cancelled() -> None:
    """Tool outputting non-PNG data returns None."""
    runner = FakeRunner({"maim": done(b"not a picture")})
    assert ScreenRegionCapture(command_runner=runner, which=installed("maim")).capture() is None


def test_slurp_cancel_never_starts_grim() -> None:
    """Slurp cancelling never starts grim."""
    runner = FakeRunner({"slurp": done(b"", returncode=1)})
    capture = ScreenRegionCapture(command_runner=runner, which=installed("slurp", "grim"))
    assert capture.capture() is None
    assert runner.calls == [["slurp"]]


@pytest.mark.parametrize("bad", ["; rm -rf ~", "10,20 300x400; id", "-o /tmp/x", "1,2 3x4 5", "abc"])
def test_slurp_output_that_is_not_a_plain_geometry_is_rejected(bad: str) -> None:
    """Slurp output that isn't plain geometry is rejected (injection protection)."""
    runner = FakeRunner({"slurp": done(bad.encode() + b"\n"), "grim": done(PNG)})
    capture = ScreenRegionCapture(command_runner=runner, which=installed("slurp", "grim"))
    assert capture.capture() is None
    assert runner.calls == [["slurp"]]


def test_slurp_output_with_non_ascii_digits_is_rejected() -> None:
    """Slurp output with non-ASCII digits is rejected."""
    arabic_indic = "١٠,٢٠ ٣x٤"
    runner = FakeRunner({"slurp": done(arabic_indic.encode() + b"\n"), "grim": done(PNG)})
    capture = ScreenRegionCapture(command_runner=runner, which=installed("slurp", "grim"))
    assert capture.capture() is None
    assert runner.calls == [["slurp"]]
