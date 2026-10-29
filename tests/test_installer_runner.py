import os
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from vocalis.installer import runner as runner_module
from vocalis.installer.runner import PhaseRunner


class FakeChild:
    def __init__(self, lines: list[str], status: int = 0) -> None:
        self.stdout = iter(lines)
        self._status = status
        self.running = True
        self.terminate_calls = 0

    def wait(self) -> int:
        self.running = False
        return self._status

    def poll(self) -> int | None:
        return None if self.running else self._status

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.running = False
        self._status = -15


def _runner(child: FakeChild, lines: list[str]) -> tuple[PhaseRunner, list[Any]]:
    calls: list[Any] = []

    def fake_popen(*args: Any, **kwargs: Any) -> FakeChild:
        calls.append((args, kwargs))
        return child

    return PhaseRunner(Path("/inst"), "base", lines.append, popen=fake_popen), calls


def test_run_forwards_each_line_and_returns_exit_code() -> None:
    got: list[str] = []
    runner, calls = _runner(FakeChild(["a\n", "\n", "  b  \n"], status=3), got)
    assert runner.run() == 3
    assert got == ["a", "b"]
    args, kwargs = calls[0]
    assert args[0] == ["bash", "/inst/install.sh", "--phase", "base"]
    assert kwargs["errors"] == "replace"


def test_run_survives_undecodable_bytes() -> None:
    got: list[str] = []
    runner, _ = _runner(FakeChild(["bad � byte\n"]), got)
    assert runner.run() == 0
    assert got == ["bad � byte"]


def test_run_reports_a_missing_bash_as_127() -> None:
    got: list[str] = []

    def failing_popen(*args: Any, **kwargs: Any) -> Any:
        raise FileNotFoundError("bash")

    runner = PhaseRunner(Path("/inst"), "base", got.append, popen=failing_popen)
    assert runner.run() == 127
    assert len(got) == 1


def test_terminate_stops_a_running_child_and_is_idempotent() -> None:
    child = FakeChild(["one\n", "two\n"])
    holder: list[PhaseRunner] = []

    def on_line(line: str) -> None:
        holder[0].terminate()
        holder[0].terminate()

    runner, _ = _runner(child, [])
    runner._on_line = on_line
    holder.append(runner)
    runner.run()
    assert child.terminate_calls == 1


def test_terminate_before_run_makes_run_return_143_without_starting() -> None:
    def popen(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("popen must not be called")

    runner = PhaseRunner(Path("/inst"), "base", lambda line: None, popen=popen)
    runner.terminate()
    assert runner.run() == 143


def test_run_stops_the_child_if_the_line_callback_raises() -> None:
    child = FakeChild(["a\n"])

    def boom(line: str) -> None:
        raise RuntimeError("boom")

    runner = PhaseRunner(Path("/inst"), "base", boom, popen=lambda *a, **k: child)
    with pytest.raises(RuntimeError):
        runner.run()
    assert child.terminate_calls == 1


def test_run_returns_non_zero_if_terminated_even_with_clean_exit() -> None:
    child = FakeChild([], status=0)
    runner, _ = _runner(child, [])
    original_wait = child.wait

    def wait_after_terminate() -> int:
        runner.terminate()
        child._status = 0
        original_wait()
        return 0

    child.wait = wait_after_terminate  # type: ignore[method-assign]
    assert runner.run() != 0


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        with open(f"/proc/{pid}/stat") as handle:
            return handle.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def _real_run(tmp_path: Path, script: str) -> tuple[int, float, int]:
    (tmp_path / "install.sh").write_text(script)
    pidfile = tmp_path / "pid"
    started = threading.Event()
    holder: list[PhaseRunner] = []

    def on_line(line: str) -> None:
        if line == "started" and not started.is_set():
            started.set()
            threading.Thread(target=holder[0].terminate).start()

    runner = PhaseRunner(tmp_path, "base", on_line)
    holder.append(runner)
    begin = time.monotonic()
    status = runner.run()
    elapsed = time.monotonic() - begin
    return status, elapsed, int(pidfile.read_text())


def _wait_gone(pid: int) -> bool:
    for _ in range(50):
        if not _alive(pid):
            return True
        time.sleep(0.1)
    return False


def test_terminate_kills_the_whole_process_group(tmp_path: Path) -> None:
    script = f"sleep 30 &\necho $! > {tmp_path}/pid\necho started\nwait\n"
    status, elapsed, pid = _real_run(tmp_path, script)
    assert status != 0
    assert elapsed < 10
    assert _wait_gone(pid)


def test_terminate_kills_a_child_that_ignores_sigterm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner_module, "_KILL_GRACE_SECONDS", 0.2)
    script = (
        "trap '' TERM\n"
        f"(trap '' TERM; exec sleep 30) &\necho $! > {tmp_path}/pid\n"
        "echo started\nwhile true; do sleep 0.1; done\n"
    )
    status, elapsed, pid = _real_run(tmp_path, script)
    assert status != 0
    assert elapsed < 10
    assert _wait_gone(pid)
