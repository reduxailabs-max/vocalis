"""Tests for vocalis.hotkeys.numlock.NumlockListener.

Uses a FakeDevice implementing the same read_loop()/close() pattern real
evdev.InputDevice objects use, so tests never touch real input hardware.
"""
import logging
import threading
import time
from typing import Iterator

import pytest

from vocalis.hotkeys.numlock import NumlockListener, _is_watchable

_EV_KEY = 1
_KEY_NUMLOCK = 69
_KEY_A = 30


class FakeEvent:
    def __init__(self, type_: int, code: int, value: int) -> None:
        self.type = type_
        self.code = code
        self.value = value


class FakeDevice:
    """Mimics evdev.InputDevice's read_loop()/close() contract:
    read_loop() blocks yielding queued events until close() is called,
    at which point it stops.
    """

    def __init__(self) -> None:
        self._pending: list[FakeEvent] = []
        self._closed = threading.Event()

    def queue_event(self, event: FakeEvent) -> None:
        self._pending.append(event)

    def read_loop(self) -> Iterator[FakeEvent]:
        while not self._closed.is_set():
            if self._pending:
                yield self._pending.pop(0)
            else:
                time.sleep(0.01)

    def close(self) -> None:
        self._closed.set()


def wait_for(flag: threading.Event, timeout: float = 1.0) -> None:
    assert flag.wait(timeout), "Expected callback was not invoked in time"


def test_no_devices_found_does_not_crash() -> None:
    press_called = threading.Event()
    listener = NumlockListener(
        on_press=press_called.set, on_release=lambda: None, device_enumerator=lambda: []
    )
    listener.start()
    listener.stop()
    assert not press_called.is_set()
    assert listener.is_running is False


def test_key_down_triggers_on_press() -> None:
    press_called = threading.Event()
    device = FakeDevice()
    device.queue_event(FakeEvent(_EV_KEY, _KEY_NUMLOCK, 1))
    listener = NumlockListener(
        on_press=press_called.set, on_release=lambda: None, device_enumerator=lambda: [device]
    )
    listener.start()
    try:
        wait_for(press_called)
    finally:
        listener.stop()


def test_key_up_triggers_on_release() -> None:
    release_called = threading.Event()
    device = FakeDevice()
    device.queue_event(FakeEvent(_EV_KEY, _KEY_NUMLOCK, 0))
    listener = NumlockListener(
        on_press=lambda: None, on_release=release_called.set, device_enumerator=lambda: [device]
    )
    listener.start()
    try:
        wait_for(release_called)
    finally:
        listener.stop()


def test_unrelated_key_is_ignored() -> None:
    press_called = threading.Event()
    device = FakeDevice()
    device.queue_event(FakeEvent(_EV_KEY, _KEY_A, 1))
    listener = NumlockListener(
        on_press=press_called.set, on_release=lambda: None, device_enumerator=lambda: [device]
    )
    listener.start()
    try:
        time.sleep(0.2)
        assert not press_called.is_set()
    finally:
        listener.stop()


def test_auto_repeat_value_two_is_ignored() -> None:
    press_called = threading.Event()
    device = FakeDevice()
    device.queue_event(FakeEvent(_EV_KEY, _KEY_NUMLOCK, 2))
    listener = NumlockListener(
        on_press=press_called.set, on_release=lambda: None, device_enumerator=lambda: [device]
    )
    listener.start()
    try:
        time.sleep(0.2)
        assert not press_called.is_set()
    finally:
        listener.stop()


def test_stop_terminates_background_threads() -> None:
    device = FakeDevice()
    listener = NumlockListener(
        on_press=lambda: None, on_release=lambda: None, device_enumerator=lambda: [device]
    )
    listener.start()
    assert listener.is_running is True
    listener.stop()
    assert listener.is_running is False


class StuckFakeDevice:
    """A fake device whose read_loop() never notices close() being called.

    Simulates a watcher thread that fails to exit within stop()'s join
    timeout, so stop()'s "thread did not exit in time" warning path can be
    exercised deterministically without waiting on real hardware.
    """

    def close(self) -> None:
        pass  # deliberately does not unblock read_loop()

    def read_loop(self) -> Iterator[FakeEvent]:
        while True:
            time.sleep(0.01)


def test_stop_logs_warning_when_thread_does_not_exit_within_timeout(
    caplog: pytest.LogCaptureFixture,
) -> None:
    device = StuckFakeDevice()
    listener = NumlockListener(
        on_press=lambda: None, on_release=lambda: None, device_enumerator=lambda: [device]
    )
    listener.start()
    assert listener.is_running is True
    with caplog.at_level(logging.WARNING, logger="vocalis"):
        listener.stop()
    assert any(
        "did not exit within timeout" in record.message for record in caplog.records
    )
    # Bookkeeping is still cleared even though the thread is still alive,
    # per the requested fix: observability, not blocking forever.
    assert listener.is_running is False


def test_is_watchable_true_for_a_real_numlock_capable_keyboard() -> None:
    assert _is_watchable("AT Translated Set 2 keyboard", [_KEY_NUMLOCK]) is True


def test_is_watchable_false_without_numlock_capability() -> None:
    assert _is_watchable("Some Other Device", [30]) is False  # KEY_A, no NUMLOCK


def test_is_watchable_false_for_vocalis_own_ydotoold_device() -> None:
    assert _is_watchable("ydotoold virtual device", [_KEY_NUMLOCK]) is False


def test_callback_exception_does_not_kill_watcher_thread() -> None:
    events_seen: list[str] = []

    def failing_on_press() -> None:
        events_seen.append("press")
        raise RuntimeError("boom")

    def on_release() -> None:
        events_seen.append("release")

    device = FakeDevice()
    device.queue_event(FakeEvent(_EV_KEY, _KEY_NUMLOCK, 1))
    device.queue_event(FakeEvent(_EV_KEY, _KEY_NUMLOCK, 0))
    listener = NumlockListener(
        on_press=failing_on_press, on_release=on_release, device_enumerator=lambda: [device]
    )
    listener.start()
    try:
        deadline = time.time() + 1.0
        while events_seen != ["press", "release"] and time.time() < deadline:
            time.sleep(0.01)
        assert events_seen == ["press", "release"]
        assert listener.is_running is True
    finally:
        listener.stop()


_KEY_SCROLLLOCK_TEST = 70
_KEY_UNRELATED_TEST = 30  # the "A" key: not watched by the listener


def test_listener_with_a_custom_key_code_reacts_to_that_key_only() -> None:
    pressed = threading.Event()
    released = threading.Event()
    device = FakeDevice()
    device.queue_event(FakeEvent(_EV_KEY, _KEY_NUMLOCK, 1))  # code 69: must be ignored
    device.queue_event(FakeEvent(_EV_KEY, _KEY_SCROLLLOCK_TEST, 1))
    device.queue_event(FakeEvent(_EV_KEY, _KEY_SCROLLLOCK_TEST, 0))
    count = {"presses": 0}

    def on_press() -> None:
        count["presses"] += 1
        pressed.set()

    listener = NumlockListener(
        on_press=on_press,
        on_release=released.set,
        device_enumerator=lambda: [device],
        key_code=_KEY_SCROLLLOCK_TEST,
    )
    listener.start()
    try:
        wait_for(pressed)
        wait_for(released)
        assert count["presses"] == 1
    finally:
        listener.stop()


def test_listener_default_key_code_still_reacts_to_numlock_only() -> None:
    pressed = threading.Event()
    device = FakeDevice()
    device.queue_event(FakeEvent(_EV_KEY, _KEY_SCROLLLOCK_TEST, 1))  # code 70: must be ignored
    device.queue_event(FakeEvent(_EV_KEY, _KEY_NUMLOCK, 1))  # code 69: the default key
    keys_pressed: list[int] = []

    def on_press() -> None:
        keys_pressed.append(len(keys_pressed))
        pressed.set()

    listener = NumlockListener(
        on_press=on_press, on_release=lambda: None, device_enumerator=lambda: [device]
    )
    listener.start()
    try:
        wait_for(pressed)
        device.queue_event(FakeEvent(_EV_KEY, _KEY_UNRELATED_TEST, 1))  # harmless sentinel
        deadline = time.monotonic() + 1.0
        while device._pending:  # sentinel popped => every earlier event was fully handled
            assert time.monotonic() < deadline, "Fake device did not consume all events in time"
            threading.Event().wait(0.005)
        assert keys_pressed == [0]  # exactly one press: the NUMLOCK one
    finally:
        listener.stop()


def test_is_watchable_follows_the_given_key_code() -> None:
    assert _is_watchable("kbd", {70}, 70) is True
    assert _is_watchable("kbd", {69}, 70) is False
    assert _is_watchable("ydotoold virtual device", {70}, 70) is False
