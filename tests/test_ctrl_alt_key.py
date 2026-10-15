"""Tests for vocalis.hotkeys.ctrl_alt_key.CtrlAltKeyListener.

Mirrors test_ctrl_alt_r.py's approach with fake evdev devices, but
exercises the generalized key_code parameter directly (using the D key,
32, as a stand-in for a key other than R).
"""
import time
from typing import Iterator

from vocalis.hotkeys.ctrl_alt_key import CtrlAltKeyListener

_EV_KEY = 1
_KEY_LEFTCTRL = 29
_KEY_LEFTALT = 56
_KEY_D = 32


class FakeEvent:
    def __init__(self, type: int, code: int, value: int) -> None:
        self.type = type
        self.code = code
        self.value = value


class FakeDevice:
    def __init__(self, events: "list[FakeEvent]") -> None:
        self._events = events
        self._closed = False

    def read_loop(self) -> Iterator[FakeEvent]:
        for event in self._events:
            if self._closed:
                raise OSError("device closed")
            yield event
        while not self._closed:
            time.sleep(0.01)
        raise OSError("device closed")

    def close(self) -> None:
        self._closed = True


def test_fires_on_trigger_for_the_configured_key() -> None:
    triggers = []
    device = FakeDevice(
        [
            FakeEvent(_EV_KEY, _KEY_LEFTCTRL, 1),
            FakeEvent(_EV_KEY, 56, 1),  # left alt
            FakeEvent(_EV_KEY, _KEY_D, 1),
        ]
    )
    listener = CtrlAltKeyListener(
        key_code=_KEY_D, on_trigger=lambda: triggers.append(1), device_enumerator=lambda: [device]
    )
    listener.start()
    time.sleep(0.1)
    listener.stop()
    assert triggers == [1]


def test_does_not_fire_for_a_different_key() -> None:
    triggers = []
    device = FakeDevice(
        [
            FakeEvent(_EV_KEY, _KEY_LEFTCTRL, 1),
            FakeEvent(_EV_KEY, _KEY_LEFTALT, 1),
            FakeEvent(_EV_KEY, 19, 1),  # R, not the configured D
        ]
    )
    listener = CtrlAltKeyListener(
        key_code=_KEY_D, on_trigger=lambda: triggers.append(1), device_enumerator=lambda: [device]
    )
    listener.start()
    time.sleep(0.1)
    listener.stop()
    assert triggers == []


def test_a_raising_callback_does_not_kill_the_watcher_thread() -> None:
    triggers = []

    def flaky_callback() -> None:
        triggers.append(1)
        if len(triggers) == 1:
            raise RuntimeError("boom")

    device = FakeDevice(
        [
            FakeEvent(_EV_KEY, _KEY_LEFTCTRL, 1),
            FakeEvent(_EV_KEY, _KEY_LEFTALT, 1),
            FakeEvent(_EV_KEY, _KEY_D, 1),
            FakeEvent(_EV_KEY, _KEY_D, 1),
        ]
    )
    listener = CtrlAltKeyListener(
        key_code=_KEY_D, on_trigger=flaky_callback, device_enumerator=lambda: [device]
    )
    listener.start()
    time.sleep(0.1)
    listener.stop()
    assert triggers == [1, 1]
