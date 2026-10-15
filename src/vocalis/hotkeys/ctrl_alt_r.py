"""Ctrl+Alt+R push-to-trigger hotkey listener for Quick Reader.

Backward-compatible wrapper around the generalized CtrlAltKeyListener
(ctrl_alt_key.py), fixed to the R key (evdev code 19). Phase 4
generalized the underlying mechanism to also support Ctrl+Alt+D for
Document Reader; this module's public names and behavior are unchanged
from Phase 3 so existing call sites and tests keep working.
"""
from typing import Callable

from vocalis.hotkeys.ctrl_alt_key import (
    CtrlAltKeyListener,
    DeviceEnumerator,
    EvdevDevice,
    EvdevEvent,
    list_devices_with_key,
)

_KEY_R = 19

__all__ = [
    "CtrlAltRListener",
    "list_hotkey_capable_devices",
    "EvdevDevice",
    "EvdevEvent",
    "DeviceEnumerator",
]


class CtrlAltRListener(CtrlAltKeyListener):
    """Calls on_trigger() when Ctrl+Alt+R is pressed."""

    def __init__(self, on_trigger: Callable[[], None], device_enumerator: DeviceEnumerator) -> None:
        """Create a CtrlAltRListener.

        Args:
            on_trigger: Called when Ctrl+Alt+R is pressed.
            device_enumerator: Returns the list of evdev devices to watch.
                Injectable so tests can supply fake devices instead of
                real hardware.
        """
        super().__init__(key_code=_KEY_R, on_trigger=on_trigger, device_enumerator=device_enumerator)


def list_hotkey_capable_devices() -> "list[EvdevDevice]":
    """Enumerate real input devices that report Ctrl+Alt+R key capability."""
    return list_devices_with_key(_KEY_R)
