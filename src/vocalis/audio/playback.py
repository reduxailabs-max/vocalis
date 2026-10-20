"""Audio playback for Quick Reader.

Plays a float32 waveform through the chosen output device (or the system
default when none is chosen) via sounddevice (talks to PipeWire, the same way MicRecorder already
captures audio) instead of shelling out to a separate player process —
this makes playback easy to interrupt cleanly and easy to unit test with
an injectable fake, mirroring MicRecorder's stream_factory pattern.

Output choice: an optional `device_provider` callable is asked for a device
index at the start of every play(); None (or no provider) means the system
default. An optional `on_device_failed(device)` hook is told which device was
in use when playback stalls (PlaybackTimeoutError) or the stream fails to
start, so a chooser can avoid that device next time.

Lock discipline: exactly like the fix for Phase 2's MicRecorder deadlock
(Task 9), the lock here only ever guards the swap of self._stream to/from
None — it is never held across a call into stream.stop()/close(), which
themselves block waiting on the audio callback thread.

Timeout safety: wait_until_done() never blocks forever. It bounds its
wait to the expected playback duration plus a safety margin, so a
misbehaving audio device (e.g. a default output routed through a broken
PipeWire/ALSA configuration where the stream opens but its callback is
never invoked) fails fast with PlaybackTimeoutError instead of hanging
the calling thread indefinitely.
"""
import logging
import threading
from typing import Any, Callable

import numpy as np
import numpy.typing as npt
import sounddevice as sd

logger = logging.getLogger("vocalis")

DEFAULT_TIMEOUT_MARGIN_SECONDS = 10.0


class PlaybackAlreadyActiveError(Exception):
    """Raised when play() is called while playback is already active."""


class PlaybackTimeoutError(Exception):
    """Raised when wait_until_done() times out waiting for playback to finish.

    Indicates the audio callback never fired (e.g. a misconfigured default
    sound device) rather than actual normal playback progress."""


class AudioPlayer:
    """Plays a float32 waveform through the chosen (or default) output device.

    `device_provider` is called at the start of every play(), outside the
    internal lock; an int selects that output device (passed to the stream
    factory as `device=`), None or no provider keeps the system default
    (no `device` kwarg). `on_device_failed(device)` is called when
    wait_until_done() times out or stream.start() raises, with the device
    used (None for the default); never on natural finish or stop(). Errors
    raised by either callable never change the original error: a failing
    provider means the default device is used.

    wait_until_done() bounds its wait to the audio's expected duration
    plus `timeout_margin_seconds`, so a stream whose callback never fires
    (e.g. a broken default audio device) raises PlaybackTimeoutError
    instead of blocking forever.
    """

    def __init__(
        self,
        stream_factory: Callable[..., Any] | None = None,
        timeout_margin_seconds: float = DEFAULT_TIMEOUT_MARGIN_SECONDS,
        device_provider: Callable[[], int | None] | None = None,
        on_device_failed: Callable[[int | None], None] | None = None,
    ) -> None:
        """Create an AudioPlayer.

        Args:
            stream_factory: Callable(**kwargs) -> stream object with
                start()/stop()/close() methods, used to construct the
                output stream. Defaults to sounddevice.OutputStream;
                injectable for tests.
            timeout_margin_seconds: Extra time, beyond the expected
                playback duration, that wait_until_done() allows before
                treating the wait as hung and raising PlaybackTimeoutError.
                Defaults to 10 seconds; injectable so tests can use a
                short margin instead of waiting on production timing.
            device_provider: Callable() -> int | None giving the output
                device index for each play(); None means system default.
            on_device_failed: Callable(device) told which device stalled
                or failed to start (None = system default).
        """
        self._stream_factory = stream_factory or sd.OutputStream
        self._timeout_margin_seconds = timeout_margin_seconds
        self._device_provider = device_provider
        self._on_device_failed = on_device_failed
        self._current_device: int | None = None
        self._stream: Any | None = None
        self._lock = threading.Lock()
        self._done_event = threading.Event()
        self._stopped_early = False
        self._expected_duration_seconds = 0.0

    @property
    def is_playing(self) -> bool:
        """Whether a playback stream is currently open."""
        return self._stream is not None

    def play(self, audio: npt.NDArray[np.float32], sample_rate: int) -> None:
        """Begin playing audio. Returns immediately; does not block.

        Args:
            audio: 1-D float32 numpy array of audio samples.
            sample_rate: Sample rate of `audio` in Hz.

        Raises:
            PlaybackAlreadyActiveError: If playback is already active.
        """
        if self.is_playing:
            raise PlaybackAlreadyActiveError("AudioPlayer.play() called while already playing")

        self._done_event.clear()
        self._stopped_early = False
        self._expected_duration_seconds = len(audio) / sample_rate
        position = {"i": 0}

        def callback(
            outdata: npt.NDArray[np.float32], frames: int, time_info: object, status: object
        ) -> None:
            if status:
                logger.warning("Audio output status: %s", status)
            start = position["i"]
            end = start + frames
            chunk = audio[start:end]
            outdata[: len(chunk), 0] = chunk
            if len(chunk) < frames:
                outdata[len(chunk) :, 0] = 0
                position["i"] = len(audio)
                raise sd.CallbackStop()
            position["i"] = end

        def finished_callback() -> None:
            self._done_event.set()

        # The provider may probe for a few seconds; it runs between the
        # is_playing check above and publishing the stream below. Readers play
        # one thing at a time, so they never exercise that window.
        self._open_stream(sample_rate, callback, finished_callback)
        logger.info(
            "Playback started (%.2fs, sample_rate=%d)", self._expected_duration_seconds, sample_rate
        )

    def open_stream(
        self,
        sample_rate: int,
        callback: Callable[..., None],
        finished_callback: Callable[[], None],
    ) -> None:
        """Open and start an output stream that the caller feeds itself.

        Used by SpeechStream for gapless readings: instead of handing over
        one finished clip (play()), the caller supplies the audio callback
        and keeps the stream open across many clips. Output-device choice,
        the one retry, and the device-failure hook behave exactly as for
        play().

        Args:
            sample_rate: Sample rate of the audio the callback will supply.
            callback: sounddevice-style callback(outdata, frames, time_info, status).
            finished_callback: Called by the stream when it finishes.

        Raises:
            PlaybackAlreadyActiveError: If a stream is already open.
        """
        if self.is_playing:
            raise PlaybackAlreadyActiveError("AudioPlayer.open_stream() called while already playing")
        self._done_event.clear()
        self._stopped_early = False
        self._open_stream(sample_rate, callback, finished_callback)

    def close_stream(self) -> None:
        """Stop and close the stream opened by open_stream(); no-op if none.

        Never raises (a failure to close is logged). Unlike stop(), this is
        an ordinary end of a reading, so it does not mark the playback as
        stopped early.
        """
        with self._lock:
            stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.stop()
        except Exception:
            logger.warning("Failed to stop the output stream", exc_info=True)
        try:
            stream.close()
        except Exception:
            logger.warning("Failed to close the output stream", exc_info=True)

    def report_device_failed(self) -> None:
        """Tell the on_device_failed hook which device was in use; never raises."""
        self._report_device_failed()

    def _open_stream(
        self,
        sample_rate: int,
        callback: Callable[..., None],
        finished_callback: Callable[[], None],
    ) -> None:
        """Open and start the stream; on failure report the device, retry once.

        The retry asks the provider again and only happens if it now names a
        different device (None = system default). Otherwise, or if the retry
        fails too (that device is reported as well), the error is re-raised.
        """
        device = self._ask_device_provider()
        try:
            self._try_open(device, sample_rate, callback, finished_callback)
            return
        except Exception:
            self._report_device_failed()
            if self._device_provider is None:
                raise
            retry_device = self._ask_device_provider()
            if retry_device == device:
                raise
        try:
            self._try_open(retry_device, sample_rate, callback, finished_callback)
        except Exception:
            self._report_device_failed()
            raise

    def _try_open(
        self,
        device: int | None,
        sample_rate: int,
        callback: Callable[..., None],
        finished_callback: Callable[[], None],
    ) -> None:
        """One attempt: build the stream for `device`, publish it, start it.

        On any failure the stream (if built) is closed and unpublished, the
        device is recorded as in use, and the error propagates; the caller
        reports the failure.
        """
        self._current_device = device
        stream_kwargs: dict[str, Any] = {
            "samplerate": sample_rate,
            "channels": 1,
            "dtype": "float32",
            "callback": callback,
            "finished_callback": finished_callback,
        }
        if device is not None:
            stream_kwargs["device"] = device
        stream = self._stream_factory(**stream_kwargs)
        with self._lock:
            self._stream = stream
        try:
            stream.start()
        except Exception:
            with self._lock:
                self._stream = None
            try:
                stream.close()
            except Exception:
                logger.warning("Failed to close stream after start() failure", exc_info=True)
            raise

    def _ask_device_provider(self) -> int | None:
        """Return the provider's device, or None (default) if none/it raises."""
        if self._device_provider is None:
            return None
        try:
            return self._device_provider()
        except Exception:
            logger.warning(
                "Output device provider failed; using the default device", exc_info=True
            )
            return None

    def _report_device_failed(self) -> None:
        """Tell the on_device_failed hook about the device in use; never raises."""
        if self._on_device_failed is None:
            return
        try:
            self._on_device_failed(self._current_device)
        except Exception:
            logger.warning("on_device_failed hook raised", exc_info=True)

    def wait_until_done(self) -> bool:
        """Block until playback finishes naturally or stop() is called.

        The wait is bounded by the expected playback duration plus
        `timeout_margin_seconds`. If that bound is exceeded, the audio
        callback never fired -- most likely a misconfigured default
        sound device -- and this raises PlaybackTimeoutError after
        performing the same safe cleanup stop() does.

        Returns:
            True if playback finished naturally; False if it was stopped
            early via stop().

        Raises:
            PlaybackTimeoutError: If playback did not finish within the
                expected duration plus the safety margin.
        """
        timeout = self._expected_duration_seconds + self._timeout_margin_seconds
        finished_in_time = self._done_event.wait(timeout=timeout)
        if not finished_in_time:
            with self._lock:
                stream, self._stream = self._stream, None
            if stream is None:
                # A concurrent stop() already claimed and cleaned up the
                # stream in the narrow window between our wait() timing
                # out and us acquiring the lock. That was a real,
                # user-initiated stop -- not a hang -- so this is not a
                # timeout at all.
                return not self._stopped_early
            self._report_device_failed()
            stream.stop()
            stream.close()
            logger.error(
                "Playback timed out after %.2fs (expected duration %.2fs, "
                "timeout margin %.2fs) -- the audio callback never fired, "
                "which usually means the default sound device is "
                "misconfigured.",
                timeout,
                self._expected_duration_seconds,
                self._timeout_margin_seconds,
            )
            raise PlaybackTimeoutError(
                f"Playback did not finish within {timeout:.2f}s "
                f"(expected duration {self._expected_duration_seconds:.2f}s); "
                "the audio callback never fired."
            )
        with self._lock:
            stream, self._stream = self._stream, None
        if stream is not None:
            stream.close()
        return not self._stopped_early

    def stop(self) -> None:
        """Stop playback immediately if active; no-op otherwise."""
        with self._lock:
            stream, self._stream = self._stream, None
            if stream is not None:
                self._stopped_early = True
        if stream is not None:
            stream.stop()
            stream.close()
            self._done_event.set()
            logger.info("Playback stopped early")
