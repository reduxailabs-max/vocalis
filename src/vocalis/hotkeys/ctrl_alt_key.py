"""Generalized Ctrl+Alt+<key> push-to-trigger hotkey listener.

Built the same way as NumlockListener (Phase 2) and the original
CtrlAltRListener (Phase 3): watches raw evdev key events without
suppressing them, one background thread per keyboard-capable device,
firing on_trigger() when the configured key is pressed while both Ctrl
and Alt are currently held on that same device. Held-state for Ctrl/Alt
is tracked per-device (inside each device's own watcher thread) since a
real combo is normally pressed on a single physical keyboard.

CtrlAltRListener (ctrl_alt_r.py) is now a thin, backward-compatible
subclass of this class fixed to the R key, so Phase 3's existing call
sites and tests keep working unchanged after this generalization.

Two lessons from Phase 2 are applied here from the start: the
on_trigger() invocation is wrapped in its own try/except inside the read
loop (an exception must never permanently kill the watcher thread), and
Vocalis's own ydotoold virtual output device is excluded from
enumeration (the app must never watch the device it uses to type/inject
its own output).
"""
import logging
import threading
from typing import Any, Callable, Iterable, Protocol

from vocalis.hotkeys.stoppable_device import StoppableEvdevDevice

logger = logging.getLogger("vocalis")

_EV_KEY = 1
_KEY_LEFTCTRL = 29
_KEY_RIGHTCTRL = 97
_KEY_LEFTALT = 56
_KEY_RIGHTALT = 100
_EXCLUDED_DEVICE_NAMES = frozenset({"ydotoold virtual device"})


class EvdevEvent(Protocol):
    """The subset of an evdev InputEvent this module depends on."""

    type: int
    code: int
    value: int


class EvdevDevice(Protocol):
    """The subset of evdev.InputDevice this module depends on."""

    def read_loop(self) -> Iterable[EvdevEvent]:
        """Blocks, yielding events as they arrive, until close() is called."""
        ...

    def close(self) -> None:
        """Close the device, causing a blocked read_loop() to raise and exit."""
        ...


DeviceEnumerator = Callable[[], "list[EvdevDevice]"]


class CtrlAltKeyListener:
    """Calls on_trigger() when Ctrl+Alt+<key_code> is pressed."""

    def __init__(
        self,
        key_code: int,
        on_trigger: Callable[[], None],
        device_enumerator: DeviceEnumerator,
    ) -> None:
        """Create a CtrlAltKeyListener.

        Args:
            key_code: The evdev key code (e.g. 19 for R, 32 for D) that,
                combined with held Ctrl+Alt, fires on_trigger().
            on_trigger: Called when Ctrl+Alt+<key_code> is pressed.
            device_enumerator: Returns the list of evdev devices to watch.
                Injectable so tests can supply fake devices instead of
                real hardware.
        """
        self._key_code = key_code
        self._on_trigger = on_trigger
        self._device_enumerator = device_enumerator
        self._devices: "list[EvdevDevice]" = []
        self._threads: "list[threading.Thread]" = []
        self._stopping = threading.Event()

    @property
    def is_running(self) -> bool:
        """Whether any watcher thread is currently active."""
        return any(thread.is_alive() for thread in self._threads)

    def start(self) -> None:
        """Enumerate hotkey-capable devices and start watching each."""
        self._stopping.clear()
        self._devices = self._device_enumerator()
        if not self._devices:
            logger.warning("No keyboard-capable input devices found")
            return
        self._threads = [
            threading.Thread(target=self._watch_device, args=(device,), daemon=True)
            for device in self._devices
        ]
        for thread in self._threads:
            thread.start()
        logger.info(
            "Watching %d device(s) for Ctrl+Alt+%d", len(self._devices), self._key_code
        )

    def stop(self) -> None:
        """Close all watched devices and wait for their threads to exit."""
        self._stopping.set()
        for device in self._devices:
            try:
                device.close()
            except Exception:
                logger.exception("Error closing input device")
        for thread in self._threads:
            thread.join(timeout=1.0)
            if thread.is_alive():
                logger.warning("Device watcher thread did not exit within timeout")
        self._devices = []
        self._threads = []

    def _watch_device(self, device: EvdevDevice) -> None:
        """Read events from one device until it is closed."""
        ctrl_held = False
        alt_held = False
        try:
            for event in device.read_loop():
                if event.type != _EV_KEY:
                    continue
                try:
                    if event.code in (_KEY_LEFTCTRL, _KEY_RIGHTCTRL):
                        ctrl_held = event.value != 0
                    elif event.code in (_KEY_LEFTALT, _KEY_RIGHTALT):
                        alt_held = event.value != 0
                    elif (
                        event.code == self._key_code
                        and event.value == 1
                        and ctrl_held
                        and alt_held
                    ):
                        self._on_trigger()
                except Exception:
                    logger.exception(
                        "Error in Ctrl+Alt+%d callback; continuing to watch device",
                        self._key_code,
                    )
        except OSError:
            if not self._stopping.is_set():
                logger.warning(
                    "Input device %s disappeared; its hotkey watcher has stopped",
                    getattr(device, "name", "unknown"),
                )
            # while stopping, the OSError is the expected result of close()
        except Exception:
            logger.exception("Error reading input device events")


def _close_all(devices: "Iterable[Any]") -> None:
    """Close every device, logging (not raising) any failure, so none leak."""
    for device in devices:
        try:
            device.close()
        except Exception:
            logger.exception("Error closing input device")


def list_devices_with_key(key_code: int) -> "list[EvdevDevice]":
    """Enumerate real input devices that report Ctrl+Alt+<key_code> capability.

    Excludes Vocalis's own ydotoold virtual output device, closing any
    device that doesn't qualify immediately since it won't be watched.
    Qualifying devices are wrapped in StoppableEvdevDevice so stop() wakes
    their watcher threads at once. If wrapping fails, the device that failed
    and every device already wrapped are closed before the error propagates.

    Args:
        key_code: The evdev key code the caller wants to watch for,
            alongside Ctrl and Alt.
    """
    import evdev

    devices: "list[EvdevDevice]" = []
    needed = {_KEY_LEFTCTRL, _KEY_LEFTALT, key_code}
    for path in evdev.list_devices():
        device = evdev.InputDevice(path)
        capabilities = set(device.capabilities().get(evdev.ecodes.EV_KEY, []))
        if needed.issubset(capabilities) and device.name not in _EXCLUDED_DEVICE_NAMES:
            try:
                devices.append(StoppableEvdevDevice(device))
            except Exception:
                _close_all([device, *devices])
                raise
        else:
            device.close()
    return devices
