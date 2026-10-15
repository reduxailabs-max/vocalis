"""Chooses a sound output device that really plays sound.

A "probe" opens a short-lived output stream on a device, starts it and waits
for the audio callback to fire. On some machines PortAudio's ALSA "default"
device opens and starts a stream whose callback never fires (the sound is
silently lost), while the "pulse" and "pipewire" devices work. The chooser
probes candidates in a sensible order and remembers the first one that works.

Lock discipline: ``_probe_lock`` serialises probe sequences; ``_cache_lock``
only guards the small cache/bad-device state. Neither is ever held while
waiting for the audio callback, except that ``current()`` holds
``_probe_lock`` across its own probe sequence so concurrent callers produce
one sequence (the cache is re-checked after taking it).

Caching: a working device is remembered for the rest of the session. A "nothing
worked" answer is only remembered for ``negative_ttl_seconds`` (default 15 s):
Vocalis is started by a systemd user service at login, possibly before the sound
server (PipeWire/PulseAudio) is ready, and a headset may be plugged in later, so
after the wait the next call probes again. Within the wait, ``current()`` answers
None immediately instead of re-probing for seconds on every call.
"""
import logging
import threading
import time
from typing import Any, Callable

import sounddevice as sd

logger = logging.getLogger("vocalis")

PROBE_SAMPLE_RATE = 24000
PREFERRED_NAMES = ("pulse", "pipewire", "default")
AUTO = "auto"


class OutputDeviceChooser:
    """Finds, caches and (when told) replaces a working sound output device."""

    def __init__(
        self,
        preference: Callable[[], str] = lambda: AUTO,
        query_devices: Callable[[], Any] = sd.query_devices,
        stream_factory: Callable[..., Any] = sd.OutputStream,
        probe_timeout_seconds: float = 1.0,
        total_timeout_seconds: float = 6.0,
        negative_ttl_seconds: float = 15.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Store the injected collaborators; nothing is probed until needed."""
        self._preference = preference
        self._query_devices = query_devices
        self._stream_factory = stream_factory
        self._probe_timeout = probe_timeout_seconds
        self._total_timeout = total_timeout_seconds
        self._negative_ttl = negative_ttl_seconds
        self._clock = clock
        self._probe_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._cache: "dict[str, tuple[int | None, float]]" = {}
        self._bad: "set[int]" = set()

    def _output_devices(self) -> "list[tuple[int, str]]":
        """Return (index, name) for every device that has output channels."""
        devices = self._query_devices()
        return [
            (index, str(entry["name"]))
            for index, entry in enumerate(devices)
            if entry["max_output_channels"] > 0
        ]

    def output_names(self) -> "list[str]":
        """Return the unique names of output devices, in device order."""
        names: "list[str]" = []
        for _, name in self._output_devices():
            if name not in names:
                names.append(name)
        return names

    def _candidates(self, preference: str) -> "list[tuple[int, str]]":
        """Return devices to try in order, skipping those marked failed."""
        devices = self._output_devices()
        wanted: "list[str]" = []
        if preference.lower() != AUTO:
            wanted.append(preference.lower())
        wanted.extend(PREFERRED_NAMES)
        ordered: "list[tuple[int, str]]" = []
        for name in wanted:
            ordered.extend(d for d in devices if d[1].lower() == name and d not in ordered)
        ordered.extend(d for d in devices if d not in ordered)
        with self._cache_lock:
            bad = set(self._bad)
        return [d for d in ordered if d[0] not in bad]

    def probe(self, index: int, timeout: "float | None" = None) -> bool:
        """Return True if a stream on this device actually fires its callback.

        ``timeout`` caps the wait (default: the configured probe timeout).
        """
        fired = threading.Event()

        def callback(outdata: Any, frames: int, time_info: Any, status: Any) -> None:
            """Fill the buffer with silence and note that the device is alive."""
            outdata.fill(0)
            fired.set()

        stream: Any = None
        try:
            stream = self._stream_factory(
                device=index,
                samplerate=PROBE_SAMPLE_RATE,
                channels=1,
                dtype="float32",
                callback=callback,
            )
            stream.start()
            return fired.wait(self._probe_timeout if timeout is None else timeout)
        except Exception as exc:  # any open/start failure just means "not usable"
            logger.warning("Output device %d could not be opened: %s", index, exc)
            return False
        finally:
            self._close_quietly(stream)

    @staticmethod
    def _close_quietly(stream: Any) -> None:
        """Stop and close a probe stream; log, never raise, on failure."""
        if stream is None:
            return
        for action in (stream.stop, stream.close):
            try:
                action()
            except Exception as exc:
                logger.warning("Cleaning up a probe stream failed: %s", exc)

    def current(self) -> "int | None":
        """Return a working output index (None = let PortAudio choose)."""
        preference = self._preference()
        key = preference.lower()
        cached = self._cached(key)
        if cached is not None:
            return cached[0]
        with self._probe_lock:
            cached = self._cached(key)
            if cached is not None:
                return cached[0]
            chosen = self._choose(preference)
            with self._cache_lock:
                self._cache[key] = (chosen, self._clock())
            return chosen

    def _cached(self, key: str) -> "tuple[int | None] | None":
        """Return (index,) if a still-valid answer is cached, else None."""
        with self._cache_lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            index, when = entry
            if index is None and self._clock() - when >= self._negative_ttl:
                del self._cache[key]
                return None
            return (index,)

    def _choose(self, preference: str) -> "int | None":
        """Probe candidates in order until one works or the budget is spent."""
        deadline = time.monotonic() + self._total_timeout
        for index, name in self._candidates(preference):
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            if self.probe(index, min(self._probe_timeout, remaining)):
                logger.info("Using sound output device %d (%s)", index, name)
                return index
            logger.warning("Sound output device %d (%s) did not respond", index, name)
        logger.error("No working sound output device found; using the system default")
        return None

    def mark_failed(self, device: "int | None") -> None:
        """Remember a device as bad and forget the cached choice.

        None means the system default stalled: only a cached "nothing worked"
        answer is dropped (so the next call re-probes at once); no device is
        marked bad and a cached working device is kept.
        """
        with self._cache_lock:
            if device is None:
                self._cache = {k: v for k, v in self._cache.items() if v[0] is not None}
                return
            self._bad.add(device)
            self._cache.clear()

    def reset(self) -> None:
        """Forget bad devices and the cached choice."""
        with self._cache_lock:
            self._bad.clear()
            self._cache.clear()
