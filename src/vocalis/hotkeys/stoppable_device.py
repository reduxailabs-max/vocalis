"""A real evdev device wrapper whose read loop stops the moment it is closed.

evdev's own ``read_loop()`` blocks in ``select()`` on the device file
descriptor. On Linux, closing that descriptor from another thread does NOT
wake the blocked ``select()``, so a watcher thread never notices that
``stop()`` was called and every ``thread.join()`` runs into its timeout.

``StoppableEvdevDevice`` fixes that for REAL devices by also waiting on a
private "wake-up" pipe. ``close()`` writes one byte into that pipe, which
makes the waiting ``select()`` return at once, so the read loop ends cleanly.
Test fakes do not need this wrapper; it is only applied by the real device
enumerators.
"""
import errno
import logging
import os
import select
import threading
from typing import Any, Iterator, Protocol

logger = logging.getLogger("vocalis")


class _RawDevice(Protocol):
    """The subset of evdev.InputDevice that the wrapper uses."""

    fd: int

    def read(self) -> Iterator[Any]:
        """Yield the events currently waiting on the device."""
        ...

    def close(self) -> None:
        """Close the device."""
        ...


class StoppableEvdevDevice:
    """Wraps an evdev device so close() promptly ends read_loop()."""

    def __init__(self, device: _RawDevice) -> None:
        """Wrap ``device`` and create the wake-up pipe.

        Args:
            device: The real evdev device (already filtered/qualified).
        """
        self._device = device
        self._wake_read, self._wake_write = os.pipe2(os.O_NONBLOCK | os.O_CLOEXEC)
        self._lock = threading.Lock()
        self._closed = False
        self._reading = False
        self._wake_read_open = True

    def __getattr__(self, name: str) -> Any:
        """Delegate anything else (name, path, capabilities...) to the device.

        Raises:
            AttributeError: If the wrapped device is not set yet (copy, pickle
                or a failed __init__), instead of recursing forever.
        """
        if name == "_device":
            raise AttributeError(name)
        return getattr(self._device, name)

    def read_loop(self) -> Iterator[Any]:
        """Yield device events until close() is called (then end cleanly).

        Raises:
            OSError: If the device itself fails (for example it was unplugged).
        """
        with self._lock:
            if self._closed:
                return
            self._reading = True
        try:
            while not self._is_closed():
                readable, _, _ = select.select([self._device.fd, self._wake_read], [], [])
                if self._wake_read in readable or self._is_closed():
                    return
                try:
                    yield from self._device.read()
                except BlockingIOError:
                    continue
        except (OSError, ValueError) as error:
            # evdev closes the fd and then sets it to -1, so a reader that is
            # between select() calls sees EBADF (OSError) or -1 (ValueError).
            # Both are the expected result of close(); anything else is real.
            if self._is_closed() and (
                isinstance(error, ValueError) or error.errno == errno.EBADF
            ):
                return
            raise
        finally:
            self._finish_reading()

    def close(self) -> None:
        """Wake the reader, then close the device and the pipe. Idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            close_wake_read = not self._reading
            if close_wake_read:
                self._wake_read_open = False
        try:
            os.write(self._wake_write, b"x")
        except OSError:
            logger.debug("Wake-up pipe write failed during close", exc_info=True)
        try:
            self._device.close()
        finally:
            os.close(self._wake_write)
            if close_wake_read:
                os.close(self._wake_read)

    def _is_closed(self) -> bool:
        """Whether close() has been called."""
        with self._lock:
            return self._closed

    def _finish_reading(self) -> None:
        """Release the read end of the pipe once the reader has stopped."""
        with self._lock:
            self._reading = False
            should_close = self._closed and self._wake_read_open
            if should_close:
                self._wake_read_open = False
        if should_close:
            os.close(self._wake_read)
