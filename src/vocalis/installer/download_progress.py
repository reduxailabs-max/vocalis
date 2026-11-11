"""Speed, time-left and text formatting for the installer's model downloads."""

from __future__ import annotations

import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import TextIO

_SPEED_WINDOW_SECONDS = 5.0
_REPORT_INTERVAL_SECONDS = 0.25
_UNITS = ("B", "KB", "MB", "GB", "TB")


def format_bytes(count: float) -> str:
    """Human size using decimal units: 999 B, 1.0 KB, 12.5 MB, 3.4 GB."""
    value = max(float(count), 0.0)
    unit = 0
    while value >= 1000.0 and unit < len(_UNITS) - 1:
        value /= 1000.0
        unit += 1
    if unit == 0:
        return f"{int(value)} B"
    return f"{value:.1f} {_UNITS[unit]}"


def format_speed(bytes_per_second: float) -> str:
    """Human speed such as '850.0 KB/s' or '12.5 MB/s'."""
    return f"{format_bytes(bytes_per_second)}/s"


def format_eta(seconds: float | None) -> str:
    """Plain-English time left, or an empty string when unknown."""
    if seconds is None or seconds < 0:
        return ""
    whole = int(seconds + 0.5)
    if whole < 60:
        return f"about {max(whole, 1)} sec left"
    if whole < 3600:
        return f"about {(whole + 30) // 60} min left"
    return f"about {whole // 3600} h {(whole % 3600) // 60} min left"


class SpeedMeter:
    """Moving-average speed over the last few seconds.

    Bytes that were already on disk from an earlier run count toward the
    total done but never toward the speed.
    """

    def __init__(self, total: int, already_done: int = 0, clock: Callable[[], float] = time.monotonic) -> None:
        self._total = total
        self._done = already_done
        self._clock = clock
        self._lock = threading.Lock()
        self._samples: deque[tuple[float, int]] = deque()
        self._samples.append((clock(), already_done))

    def add(self, count: int) -> None:
        """Record that count more bytes arrived."""
        with self._lock:
            self._done = min(self._done + count, self._total)
            now = self._clock()
            self._samples.append((now, self._done))
            while len(self._samples) > 2 and now - self._samples[0][0] > _SPEED_WINDOW_SECONDS:
                self._samples.popleft()

    @property
    def done(self) -> int:
        """Bytes finished so far."""
        with self._lock:
            return self._done

    @property
    def total(self) -> int:
        """Bytes in the whole download."""
        return self._total

    def speed(self) -> float:
        """Bytes per second over the recent window (0.0 if unknown)."""
        with self._lock:
            start_time, start_done = self._samples[0]
            elapsed = self._clock() - start_time
            if elapsed <= 0:
                return 0.0
            return max(self._done - start_done, 0) / elapsed

    def eta_seconds(self) -> float | None:
        """Seconds left at the current speed, or None if the speed is zero."""
        speed = self.speed()
        if speed <= 0:
            return None
        return max(self._total - self.done, 0) / speed


def describe(meter: SpeedMeter) -> str:
    """One status sentence: '1.2 GB of 3.4 GB · 12.5 MB/s · about 3 min left'."""
    parts = [f"{format_bytes(meter.done)} of {format_bytes(meter.total)}"]
    if meter.speed() > 0:
        parts.append(format_speed(meter.speed()))
        eta = format_eta(meter.eta_seconds())
        if eta:
            parts.append(eta)
    return " · ".join(parts)


class Reporter:
    """Prints throttled progress: a PROGRESS line for pipes, a live bar for terminals."""

    def __init__(
        self,
        meter: SpeedMeter,
        stream: TextIO | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._meter = meter
        self._stream = stream if stream is not None else sys.stdout
        self._clock = clock
        self._last = float("-inf")
        self._lock = threading.Lock()
        self._interactive = bool(getattr(self._stream, "isatty", lambda: False)())

    def update(self, force: bool = False) -> None:
        """Print a progress update unless one was printed a moment ago."""
        with self._lock:
            now = self._clock()
            if not force and now - self._last < _REPORT_INTERVAL_SECONDS:
                return
            self._last = now
            if self._interactive:
                percent = 100 * self._meter.done // max(self._meter.total, 1)
                self._stream.write(f"\r  {percent:3d}%  {describe(self._meter)}   ")
            else:
                eta = self._meter.eta_seconds()
                self._stream.write(
                    f"PROGRESS {self._meter.done} {self._meter.total} "
                    f"{int(self._meter.speed())} {-1 if eta is None else int(eta)}\n"
                )
            self._stream.flush()

    def finish(self) -> None:
        """Print the final update and end the terminal line."""
        self.update(force=True)
        if self._interactive:
            self._stream.write("\n")
            self._stream.flush()
