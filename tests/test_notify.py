"""Tests for vocalis.notify.Notifier."""
import subprocess

from vocalis.notify import Notifier


class FakeRunner:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.raises: Exception | None = None

    def __call__(self, argv: list[str]):  # type: ignore[no-untyped-def]
        self.calls.append(argv)
        if self.raises is not None:
            raise self.raises
        return subprocess.CompletedProcess(argv, 0)


def test_disabled_notifier_sends_nothing() -> None:
    runner = FakeRunner()
    notifier = Notifier(enabled=False, command_runner=runner)
    notifier.notify("Vocalis", "Started")
    assert runner.calls == []


def test_enabled_notifier_sends_title_and_message() -> None:
    runner = FakeRunner()
    notifier = Notifier(enabled=True, command_runner=runner)
    notifier.notify("Vocalis", "Started")
    assert len(runner.calls) == 1
    assert "Vocalis" in runner.calls[0]
    assert "Started" in runner.calls[0]


def test_message_omitted_when_empty() -> None:
    runner = FakeRunner()
    notifier = Notifier(enabled=True, command_runner=runner)
    notifier.notify("Vocalis")
    assert len(runner.calls[0]) == 3  # notify-send, --app-name=Vocalis, title only


def test_notify_send_failure_does_not_raise() -> None:
    runner = FakeRunner()
    runner.raises = FileNotFoundError("notify-send not found")
    notifier = Notifier(enabled=True, command_runner=runner)
    notifier.notify("Vocalis", "Started")  # must not raise


def test_enabled_can_be_a_function_that_is_checked_on_every_notification() -> None:
    state = {"on": True}
    runner = FakeRunner()
    notifier = Notifier(enabled=lambda: state["on"], command_runner=runner)
    notifier.notify("T", "one")
    state["on"] = False
    notifier.notify("T", "two")
    state["on"] = True
    notifier.notify("T", "three")
    assert [call[-1] for call in runner.calls] == ["one", "three"]


def test_a_raising_enabled_callable_sends_nothing_and_does_not_raise() -> None:
    runner = FakeRunner()

    def broken() -> bool:
        raise RuntimeError("settings broke")

    Notifier(enabled=broken, command_runner=runner).notify("Vocalis", "Started")

    assert runner.calls == []
