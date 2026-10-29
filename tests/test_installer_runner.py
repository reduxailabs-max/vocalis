from pathlib import Path
from typing import Any

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
    child = FakeChild([])
    runner, _ = _runner(child, [])
    runner._child = child  # simulate a run in progress
    runner.terminate()
    runner.terminate()
    assert child.terminate_calls == 1


def test_terminate_before_run_is_harmless() -> None:
    runner, _ = _runner(FakeChild([]), [])
    runner.terminate()


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
