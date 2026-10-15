"""NUMLOCK push-to-talk hotkey listener using evdev.

Watches all keyboard-capable input devices for NUMLOCK key-down/key-up
events and invokes callbacks accordingly. Each device gets its own
background thread using evdev's read_loop()/close() idiom: read_loop()
blocks until the device is closed, at which point it raises OSError in the
reading thread — the standard, documented way to stop an evdev read loop
from another thread. Does not grab() or suppress the key, so the OS's own
NumLock LED/state toggling is unaffected (matching the previous Voxtype
behavior).
"""
import logging
import threading
from typing import Any, Callable, Iterable, Protocol

from vocalis.hotkeys.stoppable_device import StoppableEvdevDevice

logger = logging.getLogger("vocalis")

_EV_KEY = 1
_KEY_NUMLOCK = 69


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


class NumlockListener:
    """Calls on_press()/on_release() when NUMLOCK is held down/released."""

    def __init__(
        self,
        on_press: Callable[[], None],
        on_release: Callable[[], None],
        device_enumerator: DeviceEnumerator,
        key_code: int = _KEY_NUMLOCK,
    ) -> None:
        """Create a NumlockListener.

        Args:
            on_press: Called when NUMLOCK transitions to pressed.
            on_release: Called when NUMLOCK transitions to released.
            device_enumerator: Returns the list of evdev devices to watch.
                Injectable so tests can supply fake devices instead of
                real hardware.
            key_code: The evdev key code to watch (default NUMLOCK, 69);
                lets the push-to-talk key be configured.
        """
        self._key_code = key_code
        self._on_press = on_press
        self._on_release = on_release
        self._device_enumerator = device_enumerator
        self._devices: list[EvdevDevice] = []
        self._threads: list[threading.Thread] = []
        self._stopping = threading.Event()

    @property
    def is_running(self) -> bool:
        """Whether any watcher thread is currently active."""
        return any(thread.is_alive() for thread in self._threads)

    def start(self) -> None:
        """Enumerate NUMLOCK-capable devices and start watching each."""
        self._stopping.clear()
        self._devices = self._device_enumerator()
        if not self._devices:
            logger.warning("No NUMLOCK-capable input devices found")
            return

        self._threads = [
            threading.Thread(target=self._watch_device, args=(device,), daemon=True)
            for device in self._devices
        ]
        for thread in self._threads:
            thread.start()
        logger.info("Watching %d device(s) for NUMLOCK", len(self._devices))

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
        try:
            for event in device.read_loop():
                if event.type != _EV_KEY or event.code != self._key_code:
                    continue
                try:
                    if event.value == 1:
                        self._on_press()
                    elif event.value == 0:
                        self._on_release()
                    # value == 2 is auto-repeat; ignored
                except Exception:
                    logger.exception("Error in NUMLOCK callback; continuing to watch device")
        except OSError:
            if not self._stopping.is_set():
                logger.warning(
                    "Input device %s disappeared; its hotkey watcher has stopped",
                    getattr(device, "name", "unknown"),
                )
            # while stopping, the OSError is the expected result of close()
        except Exception:
            logger.exception("Error reading input device events")


_EXCLUDED_DEVICE_NAMES = frozenset({"ydotoold virtual device"})


def _is_watchable(name: str, capabilities: "Iterable[int]", key_code: int = _KEY_NUMLOCK) -> bool:
    """Whether a device should be watched for push-to-talk press/release:
    reports the `key_code` key capability (default NUMLOCK), and isn't Vocalis's own ydotoold
    virtual output device. Vocalis must never watch the very device it
    uses (via TextOutput/ydotool) to type text back out — confirmed as a
    real device on the target machine during the Phase 2 final review.
    """
    return key_code in capabilities and name not in _EXCLUDED_DEVICE_NAMES


def _close_all(devices: "Iterable[Any]") -> None:
    """Close every device, logging (not raising) any failure, so none leak."""
    for device in devices:
        try:
            device.close()
        except Exception:
            logger.exception("Error closing input device")


def list_numlock_capable_devices(key_code: int = _KEY_NUMLOCK) -> "list[EvdevDevice]":
    """Enumerate real input devices that should be watched for `key_code` (default NUMLOCK).

    Closes any device that isn't watchable immediately, since it won't
    be watched (see _is_watchable for what disqualifies a device).
    Watched devices are wrapped in StoppableEvdevDevice so stop() wakes
    their watcher threads at once. If wrapping fails, the device that failed
    and every device already wrapped are closed before the error propagates.
    """
    import evdev

    devices: list[EvdevDevice] = []
    for path in evdev.list_devices():
        device = evdev.InputDevice(path)
        capabilities = device.capabilities().get(evdev.ecodes.EV_KEY, [])
        if _is_watchable(device.name, capabilities, key_code):
            try:
                devices.append(StoppableEvdevDevice(device))
            except Exception:
                _close_all([device, *devices])
                raise
        else:
            device.close()
    return devices
