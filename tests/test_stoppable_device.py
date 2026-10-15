"""Tests for vocalis.hotkeys.stoppable_device.StoppableEvdevDevice.

A real os.pipe() stands in for the evdev device file descriptor, so the
adapter's real select()-based waiting is exercised without hardware.
"""
import logging
import os
import select
import threading
from typing import Any, Iterator

import pytest

from vocalis.hotkeys.ctrl_alt_key import CtrlAltKeyListener
from vocalis.hotkeys.numlock import NumlockListener
from vocalis.hotkeys.stoppable_device import StoppableEvdevDevice

_EV_KEY = 1
_KEY_NUMLOCK = 69
_PROMPT = 1.0  # seconds a stop must finish within; a miss FAILS the test


class FakeEvent:
    """A minimal evdev-style event."""

    def __init__(self, type: int, code: int, value: int) -> None:
        self.type = type
        self.code = code
        self.value = value


class PipeDevice:
    """Fake evdev device: each byte written to the pipe is one key event."""

    def __init__(self) -> None:
        self._read_fd, self.write_fd = os.pipe()
        self.fd = self._read_fd
        self.name = "fake keyboard"
        self.close_calls = 0
        self.closed = threading.Event()

    def feed(self, code: int, value: int) -> None:
        """Queue one key event (code, value) on the fake device."""
        os.write(self.write_fd, bytes([code, value]))

    def read(self) -> Iterator[FakeEvent]:
        """Yield the queued events."""
        data = os.read(self._read_fd, 2)
        if data:
            yield FakeEvent(_EV_KEY, data[0], data[1])

    def capabilities(self) -> dict[int, list[int]]:
        """Report a fixed capability map."""
        return {_EV_KEY: [_KEY_NUMLOCK]}

    def close(self) -> None:
        """Close the pipe ends like evdev: close the fd, THEN set fd to -1."""
        self.close_calls += 1
        os.close(self._read_fd)
        os.close(self.write_fd)
        self.fd = -1
        self.closed.set()


class BlockingPipeDevice(PipeDevice):
    """Plain device wrapper reproducing the bug: close() cannot wake select()."""

    def read_loop(self) -> Iterator[FakeEvent]:
        """Block in select() on the device fd only, like evdev's read_loop."""
        while True:
            select.select([self.fd], [], [])
            yield from self.read()

    def close(self) -> None:
        """Close only the read end (the real-device behaviour that fails to wake)."""
        self.close_calls += 1
        os.close(self.fd)


class SelectProbe:
    """Counts how many times the adapter is about to block in select()."""

    def __init__(self) -> None:
        self._entered = threading.Semaphore(0)

    def entered(self) -> None:
        """Record one entry into select()."""
        self._entered.release()

    def wait_for(self, count: int) -> None:
        """Wait (bounded) until ``count`` select() entries happened, else fail."""
        for _ in range(count):
            assert self._entered.acquire(timeout=_PROMPT), "watcher never reached select()"


@pytest.fixture
def select_probe(monkeypatch: pytest.MonkeyPatch) -> SelectProbe:
    """Wrap the adapter module's select.select to signal just before it blocks."""
    import types

    from vocalis.hotkeys import stoppable_device

    probe = SelectProbe()
    real_select = select.select

    def probed_select(*args: Any) -> Any:
        """Signal the probe, then delegate to the real select.select."""
        probe.entered()
        return real_select(*args)

    monkeypatch.setattr(
        stoppable_device, "select", types.SimpleNamespace(select=probed_select)
    )
    return probe


def _run_loop(device: Any, sink: "list[FakeEvent]") -> threading.Thread:
    """Start a thread collecting events from device.read_loop() into sink."""

    def target() -> None:
        try:
            for event in device.read_loop():
                sink.append(event)
        except OSError:
            pass

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread


def test_events_written_to_the_device_come_out_of_read_loop() -> None:
    raw = PipeDevice()
    device = StoppableEvdevDevice(raw)
    got: "list[FakeEvent]" = []
    seen = threading.Event()

    def target() -> None:
        for event in device.read_loop():
            got.append(event)
            if len(got) == 2:
                seen.set()

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    raw.feed(69, 1)
    raw.feed(69, 0)
    assert seen.wait(_PROMPT)
    device.close()
    thread.join(_PROMPT)
    assert not thread.is_alive()
    assert [(e.code, e.value) for e in got] == [(69, 1), (69, 0)]


def test_close_from_another_thread_wakes_a_blocked_read_loop(
    select_probe: SelectProbe,
) -> None:
    raw = PipeDevice()
    device = StoppableEvdevDevice(raw)
    thread = _run_loop(device, [])
    select_probe.wait_for(1)
    device.close()
    thread.join(_PROMPT)
    assert not thread.is_alive()


def test_close_while_consumer_is_suspended_at_a_yield_ends_quietly() -> None:
    raw = PipeDevice()
    device = StoppableEvdevDevice(raw)
    raw.feed(69, 1)
    loop = device.read_loop()
    first = next(loop)  # generator is now suspended at the yield
    assert (first.code, first.value) == (69, 1)
    device.close()  # evdev-style: fd is now -1 while the generator is suspended
    assert list(loop) == []  # resuming ends quietly, no ValueError


def test_value_error_propagates_when_the_adapter_is_not_closed() -> None:
    raw = PipeDevice()
    device = StoppableEvdevDevice(raw)
    real_fd = raw.fd
    raw.fd = -1  # a genuine bug: bad fd while NOT closed must not be hidden
    with pytest.raises(ValueError):
        list(device.read_loop())
    raw.fd = real_fd
    device.close()


def _stop_while_callback_runs(
    listener: Any, raw: PipeDevice, in_callback: threading.Event, release: threading.Event
) -> None:
    """Stop the listener while its callback is blocked, then release the callback."""
    assert in_callback.wait(_PROMPT)
    done = threading.Event()

    def stop_then_signal() -> None:
        """Run stop() and report completion."""
        listener.stop()
        done.set()

    stopper = threading.Thread(target=stop_then_signal, daemon=True)
    stopper.start()
    assert raw.closed.wait(_PROMPT)  # the device is closed; the callback is still running
    release.set()
    assert done.wait(_PROMPT * 2)
    assert not listener.is_running


@pytest.mark.parametrize("kind", ["ctrl_alt", "numlock"])
def test_stopping_while_a_callback_runs_logs_no_error(
    kind: str, caplog: pytest.LogCaptureFixture
) -> None:
    raw = PipeDevice()
    devices: Any = [StoppableEvdevDevice(raw)]
    in_callback = threading.Event()
    release = threading.Event()

    def blocking_callback() -> None:
        """Signal that the callback started, then wait to be released."""
        in_callback.set()
        release.wait(_PROMPT * 2)

    listener: Any
    if kind == "ctrl_alt":
        listener = CtrlAltKeyListener(32, blocking_callback, lambda: devices)
    else:
        listener = NumlockListener(blocking_callback, lambda: None, lambda: devices)
    with caplog.at_level(logging.DEBUG, logger="vocalis"):
        listener.start()
        if kind == "ctrl_alt":
            raw.feed(29, 1)
            raw.feed(56, 1)
            raw.feed(32, 1)
        else:
            raw.feed(_KEY_NUMLOCK, 1)
        _stop_while_callback_runs(listener, raw, in_callback, release)
    bad = [r for r in caplog.records if r.levelno >= logging.WARNING or r.exc_info]
    assert bad == []


def test_plain_blocking_device_is_not_woken_by_close() -> None:
    """Documents the bug the adapter fixes: closing the fd leaves select() stuck."""
    raw = BlockingPipeDevice()
    write_fd = raw.write_fd
    thread = _run_loop(raw, [])
    raw.close()
    thread.join(0.3)
    stuck = thread.is_alive()
    os.write(write_fd, b"xx")  # cleanup: unblock the stuck thread
    thread.join(_PROMPT)
    os.close(write_fd)
    assert stuck


def test_close_twice_is_safe_and_closes_the_device_once(
    select_probe: SelectProbe,
) -> None:
    raw = PipeDevice()
    device = StoppableEvdevDevice(raw)
    thread = _run_loop(device, [])
    select_probe.wait_for(1)
    device.close()
    device.close()
    thread.join(_PROMPT)
    assert not thread.is_alive()
    assert raw.close_calls == 1


def test_close_without_a_running_reader_closes_everything_once() -> None:
    raw = PipeDevice()
    device = StoppableEvdevDevice(raw)
    device.close()
    device.close()
    assert raw.close_calls == 1


def test_read_loop_after_close_ends_cleanly() -> None:
    raw = PipeDevice()
    device = StoppableEvdevDevice(raw)
    device.close()
    assert list(device.read_loop()) == []


def test_other_attributes_are_delegated_to_the_device() -> None:
    raw = PipeDevice()
    device = StoppableEvdevDevice(raw)
    assert device.name == "fake keyboard"
    assert device.capabilities() == {_EV_KEY: [_KEY_NUMLOCK]}
    assert device.fd == raw.fd
    device.close()


def test_device_error_other_than_close_propagates() -> None:
    raw = PipeDevice()

    def broken() -> Iterator[FakeEvent]:
        raise OSError(19, "No such device")
        yield  # pragma: no cover

    raw.read = broken  # type: ignore[method-assign]
    device = StoppableEvdevDevice(raw)
    raw.feed(1, 1)
    with pytest.raises(OSError):
        list(device.read_loop())
    device.close()


def test_blocking_io_error_is_retried_without_crashing() -> None:
    raw = PipeDevice()
    calls = {"n": 0}
    original = raw.read

    def flaky() -> Iterator[FakeEvent]:
        calls["n"] += 1
        if calls["n"] == 1:
            os.read(raw.fd, 2)
            raise BlockingIOError()
        yield from original()

    raw.read = flaky  # type: ignore[method-assign]
    device = StoppableEvdevDevice(raw)
    got: "list[FakeEvent]" = []
    seen = threading.Event()

    def target() -> None:
        for event in device.read_loop():
            got.append(event)
            seen.set()

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    raw.feed(1, 1)  # swallowed by the first, failing read
    raw.feed(69, 1)
    assert seen.wait(_PROMPT)
    device.close()
    thread.join(_PROMPT)
    assert [(e.code, e.value) for e in got] == [(69, 1)]


@pytest.mark.parametrize("kind", ["ctrl_alt", "numlock"])
def test_listener_stop_is_prompt_with_stoppable_devices(
    kind: str, caplog: pytest.LogCaptureFixture, select_probe: SelectProbe
) -> None:
    raws = [PipeDevice(), PipeDevice()]
    devices: Any = [StoppableEvdevDevice(raw) for raw in raws]
    listener: Any
    if kind == "ctrl_alt":
        listener = CtrlAltKeyListener(32, lambda: None, lambda: devices)
    else:
        listener = NumlockListener(lambda: None, lambda: None, lambda: devices)
    listener.start()
    select_probe.wait_for(len(raws))  # every watcher is about to block in select()
    with caplog.at_level(logging.WARNING, logger="vocalis"):
        done = threading.Event()

        def stop_then_signal() -> None:
            """Run stop() and report completion."""
            listener.stop()
            done.set()

        stopper = threading.Thread(target=stop_then_signal, daemon=True)
        stopper.start()
        assert done.wait(_PROMPT * 2)
    assert "did not exit within timeout" not in caplog.text
    assert not listener.is_running
    assert all(raw.close_calls == 1 for raw in raws)


class _EnumDevice(PipeDevice):
    """Fake evdev.InputDevice for enumerator tests, keyed by path."""

    def __init__(self, name: str, keys: "list[int]") -> None:
        super().__init__()
        self.name = name
        self._keys = keys

    def capabilities(self) -> dict[int, list[int]]:
        """Report the configured key capabilities."""
        return {_EV_KEY: self._keys}


def _fake_evdev(devices: "dict[str, _EnumDevice]") -> Any:
    """Build a stand-in for the evdev module serving the given devices."""
    import types

    module = types.ModuleType("evdev")
    module.list_devices = lambda: list(devices)  # type: ignore[attr-defined]
    module.InputDevice = devices.__getitem__  # type: ignore[attr-defined]
    module.ecodes = types.SimpleNamespace(EV_KEY=_EV_KEY)  # type: ignore[attr-defined]
    return module


def _open_fd_count() -> int:
    """Number of file descriptors currently open in this process."""
    return len(os.listdir("/proc/self/fd"))


def test_enumerators_wrap_qualifying_devices_and_close_the_rest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vocalis.hotkeys.ctrl_alt_key import list_devices_with_key
    from vocalis.hotkeys.numlock import list_numlock_capable_devices

    def ctrl_alt_d() -> Any:
        """Enumerate Ctrl+Alt+D-capable devices."""
        return list_devices_with_key(32)

    enumerators: "list[Any]" = [ctrl_alt_d, list_numlock_capable_devices]
    fds_before = _open_fd_count()
    for enumerate_devices in enumerators:
        good = _EnumDevice("keyboard", [29, 56, 32, _KEY_NUMLOCK])
        no_keys = _EnumDevice("mouse", [])
        excluded = _EnumDevice("ydotoold virtual device", [29, 56, 32, _KEY_NUMLOCK])
        monkeypatch.setitem(
            __import__("sys").modules,
            "evdev",
            _fake_evdev({"/a": good, "/b": no_keys, "/c": excluded}),
        )
        result = enumerate_devices()
        assert len(result) == 1
        assert isinstance(result[0], StoppableEvdevDevice)
        assert result[0].name == "keyboard"
        assert good.close_calls == 0
        assert no_keys.close_calls == 1
        assert excluded.close_calls == 1
        result[0].close()
        assert good.close_calls == 1
    assert _open_fd_count() == fds_before  # no leaked pipe fds


@pytest.mark.parametrize("module_name", ["numlock", "ctrl_alt_key"])
def test_enumerator_closes_every_device_when_wrapping_fails_midway(
    module_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    module = importlib.import_module(f"vocalis.hotkeys.{module_name}")
    fds_before = _open_fd_count()
    first = _EnumDevice("keyboard one", [29, 56, 32, _KEY_NUMLOCK])
    second = _EnumDevice("keyboard two", [29, 56, 32, _KEY_NUMLOCK])
    third = _EnumDevice("keyboard three", [29, 56, 32, _KEY_NUMLOCK])
    monkeypatch.setitem(
        __import__("sys").modules,
        "evdev",
        _fake_evdev({"/a": first, "/b": second, "/c": third}),
    )
    wrapped: "list[StoppableEvdevDevice]" = []

    def flaky_wrapper(device: Any) -> StoppableEvdevDevice:
        """Wrap the first device, then fail like EMFILE on the second."""
        if wrapped:
            raise OSError(24, "Too many open files")
        wrapper = StoppableEvdevDevice(device)
        wrapped.append(wrapper)
        return wrapper

    monkeypatch.setattr(module, "StoppableEvdevDevice", flaky_wrapper)
    with pytest.raises(OSError):
        if module_name == "numlock":
            module.list_numlock_capable_devices()
        else:
            module.list_devices_with_key(32)
    assert first.close_calls == 1  # already wrapped, closed through its wrapper
    assert second.close_calls == 1  # the raw device that failed to wrap
    assert third.close_calls == 0  # never reached
    third.close()
    assert _open_fd_count() == fds_before


def test_getattr_does_not_recurse_when_the_device_is_not_set() -> None:
    import copy

    empty = StoppableEvdevDevice.__new__(StoppableEvdevDevice)
    with pytest.raises(AttributeError):
        empty.name
    raw = PipeDevice()
    device = StoppableEvdevDevice(raw)
    clone = copy.copy(device)  # copy builds the object without calling __init__
    assert clone.name == "fake keyboard"
    device.close()


class _UnpluggedDevice:
    """Fake device whose read_loop fails like an unplugged keyboard."""

    name = "unplugged keyboard"

    def read_loop(self) -> Iterator[FakeEvent]:
        """Fail at once with ENODEV."""
        raise OSError(19, "No such device")
        yield  # pragma: no cover

    def close(self) -> None:
        """Nothing to close."""


def _make_listener(kind: str, devices: Any) -> Any:
    """Build a NumlockListener or CtrlAltKeyListener over the given devices."""
    if kind == "ctrl_alt":
        return CtrlAltKeyListener(32, lambda: None, lambda: devices)
    return NumlockListener(lambda: None, lambda: None, lambda: devices)


@pytest.mark.parametrize("kind", ["ctrl_alt", "numlock"])
def test_unplugged_device_logs_exactly_one_warning(
    kind: str, caplog: pytest.LogCaptureFixture
) -> None:
    listener = _make_listener(kind, [_UnpluggedDevice()])
    with caplog.at_level(logging.DEBUG, logger="vocalis"):
        listener.start()
        for thread in listener._threads:
            thread.join(_PROMPT)
        assert not listener.is_running
        listener.stop()
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "unplugged keyboard" in warnings[0].getMessage()
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


@pytest.mark.parametrize("kind", ["ctrl_alt", "numlock"])
def test_normal_stop_logs_no_device_disappeared_warning(
    kind: str, caplog: pytest.LogCaptureFixture, select_probe: SelectProbe
) -> None:
    raw = PipeDevice()
    listener = _make_listener(kind, [StoppableEvdevDevice(raw)])
    with caplog.at_level(logging.DEBUG, logger="vocalis"):
        listener.start()
        select_probe.wait_for(1)
        listener.stop()
    assert "disappeared" not in caplog.text
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


@pytest.mark.parametrize("kind", ["ctrl_alt", "numlock"])
def test_oserror_from_a_plain_device_during_stop_stays_silent(
    kind: str, caplog: pytest.LogCaptureFixture
) -> None:
    entered = threading.Event()
    release = threading.Event()

    class ClosingDevice(_UnpluggedDevice):
        """Blocks in read_loop until closed, then raises like evdev does."""

        def read_loop(self) -> Iterator[FakeEvent]:
            """Wait for close(), then fail with EBADF."""
            entered.set()
            release.wait(_PROMPT * 2)
            raise OSError(9, "Bad file descriptor")
            yield  # pragma: no cover

        def close(self) -> None:
            """Release the blocked reader."""
            release.set()

    listener = _make_listener(kind, [ClosingDevice()])
    with caplog.at_level(logging.DEBUG, logger="vocalis"):
        listener.start()
        assert entered.wait(_PROMPT)
        listener.stop()
    assert "disappeared" not in caplog.text
