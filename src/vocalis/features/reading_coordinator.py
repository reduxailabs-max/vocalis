"""Ensures only one of Quick Reader and Document Reader is reading aloud
at a time.

Each feature identifies itself with a stable `owner` object — pass the
feature instance itself (`self`) — never with its stop callback. A bound
method obtained via `self.method` is a fresh object on every attribute
access in Python, so comparing stop callbacks with `is` would never
match, even across two calls from the very same instance. `owner` is
compared with `is`; `stopper` is only ever called, never compared.

Follows the same lock discipline already hard-won in quick_reader.py:
this coordinator only ever guards a plain reference swap under its lock.
The actual stop() call — which can block briefly on AudioPlayer.stop()
or a notify-send subprocess — happens after the lock has been released.
"""
import threading
from typing import Callable

StopCallback = Callable[[], None]


class ReadingCoordinator:
    """Tracks which feature (if any) currently owns audio playback."""

    def __init__(self) -> None:
        """Create a ReadingCoordinator with no current owner."""
        self._lock = threading.Lock()
        self._owner: object | None = None
        self._stopper: StopCallback | None = None

    def start_reading(self, owner: object, stopper: StopCallback) -> None:
        """Claim ownership of playback for `owner`.

        If a different owner currently holds it, that owner's registered
        stopper is called now, after this method's own lock has been
        released, so it is free to block briefly (e.g. on AudioPlayer.stop()
        or a notification) without holding up this call's caller.

        Args:
            owner: A stable identifier for the calling feature — pass the
                feature instance itself. Compared with `is`, never `==`.
            stopper: Called (with no arguments) to stop `owner`'s own
                in-progress reading, if a later start_reading() call from
                a different owner preempts it.
        """
        with self._lock:
            previous_owner = self._owner
            previous_stopper = self._stopper
            self._owner = owner
            self._stopper = stopper
        if previous_owner is not None and previous_owner is not owner:
            assert previous_stopper is not None
            previous_stopper()

    def finish_reading(self, owner: object) -> None:
        """Release ownership for `owner`, if it is still the current owner.

        A stale call from a reading that already lost ownership to a
        later start_reading() must not clear that newer owner's claim.

        Args:
            owner: The same object passed to the matching start_reading().
        """
        with self._lock:
            if self._owner is owner:
                self._owner = None
                self._stopper = None
