"""Tests for vocalis.hotkeys.ctrl_alt_r.CtrlAltRListener.

Uses a FakeDevice implementing the same read_loop()/close() pattern real
evdev.InputDevice objects use, so tests never touch real input hardware.
Mirrors test_numlock.py's FakeDevice/FakeEvent pattern exactly.
"""
import logging
import threading
import time
from typing import Iterator

import pytest

from vocalis.hotkeys.ctrl_alt_r import CtrlAltRListener

_EV_KEY = 1
_KEY_LEFTCTRL = 29
_KEY_RIGHTCTRL = 97
_KEY_LEFTALT = 56
_KEY_RIGHTALT = 100
_KEY_R = 19
_KEY_A = 30


class FakeEvent:
    def __init__(self, type_: int, code: int, value: int) -> None:
        self.type = type_
        self.code = code
        self.value = value


class FakeDevice:
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


def _press(code: int) -> FakeEvent:
    return FakeEvent(_EV_KEY, code, 1)


def _release(code: int) -> FakeEvent:
    return FakeEvent(_EV_KEY, code, 0)


def test_no_devices_found_does_not_crash() -> None:
    triggered = threading.Event()
    listener = CtrlAltRListener(on_trigger=triggered.set, device_enumerator=lambda: [])
    listener.start()
    listener.stop()
    assert not triggered.is_set()
    assert listener.is_running is False


def test_ctrl_alt_r_together_triggers() -> None:
    triggered = threading.Event()
    device = FakeDevice()
    device.queue_event(_press(_KEY_LEFTCTRL))
    device.queue_event(_press(_KEY_LEFTALT))
    device.queue_event(_press(_KEY_R))
    listener = CtrlAltRListener(on_trigger=triggered.set, device_enumerator=lambda: [device])
    listener.start()
    try:
        wait_for(triggered)
    finally:
        listener.stop()


def test_r_alone_does_not_trigger() -> None:
    triggered = threading.Event()
    device = FakeDevice()
    device.queue_event(_press(_KEY_R))
    listener = CtrlAltRListener(on_trigger=triggered.set, device_enumerator=lambda: [device])
    listener.start()
    try:
        time.sleep(0.2)
        assert not triggered.is_set()
    finally:
        listener.stop()


def test_ctrl_r_without_alt_does_not_trigger() -> None:
    triggered = threading.Event()
    device = FakeDevice()
    device.queue_event(_press(_KEY_LEFTCTRL))
    device.queue_event(_press(_KEY_R))
    listener = CtrlAltRListener(on_trigger=triggered.set, device_enumerator=lambda: [device])
    listener.start()
    try:
        time.sleep(0.2)
        assert not triggered.is_set()
    finally:
        listener.stop()


def test_releasing_ctrl_then_pressing_r_does_not_trigger() -> None:
    triggered = threading.Event()
    device = FakeDevice()
    device.queue_event(_press(_KEY_LEFTCTRL))
    device.queue_event(_press(_KEY_LEFTALT))
    device.queue_event(_release(_KEY_LEFTCTRL))
    device.queue_event(_press(_KEY_R))
    listener = CtrlAltRListener(on_trigger=triggered.set, device_enumerator=lambda: [device])
    listener.start()
    try:
        time.sleep(0.2)
        assert not triggered.is_set()
    finally:
        listener.stop()


def test_right_ctrl_and_right_alt_also_trigger() -> None:
    triggered = threading.Event()
    device = FakeDevice()
    device.queue_event(_press(_KEY_RIGHTCTRL))
    device.queue_event(_press(_KEY_RIGHTALT))
    device.queue_event(_press(_KEY_R))
    listener = CtrlAltRListener(on_trigger=triggered.set, device_enumerator=lambda: [device])
    listener.start()
    try:
        wait_for(triggered)
    finally:
        listener.stop()


def test_unrelated_key_is_ignored() -> None:
    triggered = threading.Event()
    device = FakeDevice()
    device.queue_event(_press(_KEY_LEFTCTRL))
    device.queue_event(_press(_KEY_LEFTALT))
    device.queue_event(_press(_KEY_A))
    listener = CtrlAltRListener(on_trigger=triggered.set, device_enumerator=lambda: [device])
    listener.start()
    try:
        time.sleep(0.2)
        assert not triggered.is_set()
    finally:
        listener.stop()


def test_stop_terminates_background_threads() -> None:
    device = FakeDevice()
    listener = CtrlAltRListener(on_trigger=lambda: None, device_enumerator=lambda: [device])
    listener.start()
    assert listener.is_running is True
    listener.stop()
    assert listener.is_running is False


def test_callback_exception_does_not_kill_watcher_thread() -> None:
    triggers_seen: list[int] = []

    def failing_on_trigger() -> None:
        triggers_seen.append(1)
        raise RuntimeError("boom")

    device = FakeDevice()
    device.queue_event(_press(_KEY_LEFTCTRL))
    device.queue_event(_press(_KEY_LEFTALT))
    device.queue_event(_press(_KEY_R))
    device.queue_event(_release(_KEY_R))
    device.queue_event(_press(_KEY_R))
    listener = CtrlAltRListener(on_trigger=failing_on_trigger, device_enumerator=lambda: [device])
    listener.start()
    try:
        deadline = time.time() + 1.0
        while len(triggers_seen) < 2 and time.time() < deadline:
            time.sleep(0.01)
        assert len(triggers_seen) == 2
        assert listener.is_running is True
    finally:
        listener.stop()
