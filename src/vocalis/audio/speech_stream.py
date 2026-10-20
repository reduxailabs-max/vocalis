"""One persistent, gapless speech output for a whole reading.

Opening a new sound stream for every piece (the old way) leaves a click or
gap at every join and makes long readings stutter. SpeechStream opens ONE
stream through AudioPlayer and keeps it open: pieces of speech (clips) are
pushed into a small bounded queue and the audio callback plays them
back-to-back. Three details make the joins inaudible:

- every clip gets a tiny fade-in and fade-out (no click at a join);
- if the next clip is not ready in time the callback plays silence instead
  of stopping (an "underrun" is a short pause, never a restart);
- the queue is bounded, so the producer gets ahead of the listener by a
  few pieces only and then waits (back-pressure), instead of racing ahead
  and heating the machine.

Pause outputs silence immediately and rewinds the clip being heard to its
start, so resume replays that piece from the beginning (the long-standing
behaviour). Clips at a different sample rate than the stream (the voice
changed mid-reading) are resampled, never played at the wrong speed.

Stall safety: if clips are waiting but the sound device has not called back
for `stall_seconds`, the device is broken (the stream opened but never
runs). push()/wait_until_done() then fail with PlaybackTimeoutError and the
device is reported to AudioPlayer's failure hook, instead of hanging.

Lock discipline: the one condition lock only guards the clip queue and a few
flags. Anything slow (opening/closing the device, user callbacks) happens
outside it; the audio callback only copies samples and does a non-blocking
put on a SimpleQueue that a helper thread drains.
"""
import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, NoReturn

import numpy as np
import numpy.typing as npt

from vocalis.audio.playback import AudioPlayer, PlaybackTimeoutError

logger = logging.getLogger("vocalis")

DEFAULT_FADE_SECONDS = 0.008
DEFAULT_STALL_SECONDS = 10.0
DEFAULT_MAX_QUEUED_CLIPS = 3
_WAIT_SLICE_SECONDS = 0.1


def _ignore_finished() -> None:
    """The stream's finished callback: nothing to do, the stream is closed explicitly."""


@dataclass
class _Clip:
    """One queued piece of audio and how far it has been played."""

    tag: int
    samples: npt.NDArray[np.float32]
    position: int = 0
    announced: bool = False


class SpeechStream:
    """Plays queued clips back-to-back on one open output stream."""

    def __init__(
        self,
        player: AudioPlayer,
        on_clip_started: Callable[[int], None] | None = None,
        fade_seconds: float = DEFAULT_FADE_SECONDS,
        stall_seconds: float = DEFAULT_STALL_SECONDS,
        max_queued_clips: int = DEFAULT_MAX_QUEUED_CLIPS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create a SpeechStream (does not open the device yet).

        Args:
            player: Opens/closes the real output stream and knows the device logic.
            on_clip_started: Called (on a helper thread, never the audio
                thread) with a clip's tag when that clip starts to be heard.
            fade_seconds: Length of the fade at each end of every clip.
            stall_seconds: How long the device may stay silent (no callback)
                while clips are waiting before the stream is declared stalled.
            max_queued_clips: Most clips held at once (playing + waiting);
                push() waits when the queue is full.
            clock: Monotonic clock; injectable for tests.
        """
        self._player = player
        self._on_clip_started = on_clip_started
        self._fade_seconds = fade_seconds
        self._stall_seconds = stall_seconds
        self._max_queued_clips = max_queued_clips
        self._clock = clock
        self._cond = threading.Condition()
        self._clips: "deque[_Clip]" = deque()
        self._sample_rate = 0
        self._open = False
        self._finishing = False
        self._paused = False
        self._stopped = False
        self._shut = False
        self._opening = False
        self._last_callback = 0.0
        self._underruns = 0
        self._played_any = False
        self._started_tags: "queue.SimpleQueue[int | None]" = queue.SimpleQueue()

    @property
    def underruns(self) -> int:
        """How many audio callbacks ran dry while more audio was still expected."""
        with self._cond:
            return self._underruns

    @property
    def queued_clips(self) -> int:
        """Clips currently held (the one being heard plus those waiting)."""
        with self._cond:
            return len(self._clips)

    def start(self, sample_rate: int) -> None:
        """Open the output stream at `sample_rate`. No-op once stopped.

        Raises:
            Exception: Whatever AudioPlayer.open_stream raises (a device failure).
        """
        with self._cond:
            if self._stopped or self._open or self._shut:
                return
            self._sample_rate = sample_rate
            self._open = True
            self._last_callback = self._clock()
            self._opening = True
        try:
            self._player.open_stream(sample_rate, self._callback, _ignore_finished)
        except Exception:
            with self._cond:
                self._opening = False
                self._open = False
                stopped = self._stopped
            if stopped:
                # The user pressed Stop during the open: nothing to report.
                self._finish_stopped_open()
                return
            raise
        with self._cond:
            self._opening = False
            stopped = self._stopped
            if not stopped:
                # Opening may have been slow; the stall clock starts now.
                self._last_callback = self._clock()
        if stopped:
            # stop() ran while the device was opening; nobody else will close it.
            self._finish_stopped_open()
            return
        if self._on_clip_started is not None:
            threading.Thread(target=self._dispatch_started, daemon=True).start()

    def _finish_stopped_open(self) -> None:
        """Close the device and post the end marker after a stop() during open."""
        self._player.close_stream()
        self._started_tags.put(None)

    def push(self, audio: "npt.NDArray[np.float32]", sample_rate: int, tag: int) -> bool:
        """Queue one clip, waiting while the queue is full.

        Args:
            audio: Mono float32 samples.
            sample_rate: Sample rate of `audio`; resampled if it differs from the stream's.
            tag: Handed to on_clip_started when this clip starts to be heard.

        Returns:
            True if accepted (an empty clip is accepted but not queued);
            False if the stream was stopped or never opened.

        Raises:
            PlaybackTimeoutError: The sound device stopped calling back.
        """
        if not self._open and not self._stopped:
            return False
        samples = self._prepared(audio, sample_rate)
        stalled = False
        with self._cond:
            while True:
                if self._stopped or not self._open:
                    return False
                if len(self._clips) < self._max_queued_clips:
                    break
                if self._stalled_locked():
                    stalled = True
                    break
                self._cond.wait(_WAIT_SLICE_SECONDS)
            if not stalled:
                if samples.size:
                    self._clips.append(_Clip(tag, samples))
                self._cond.notify_all()
                return True
        self._raise_stalled()

    def finish(self) -> None:
        """Say that no more clips will be pushed."""
        with self._cond:
            self._finishing = True
            self._cond.notify_all()

    def wait_until_done(self) -> bool:
        """Wait until every queued clip has been played, then close the stream.

        Call finish() first, otherwise this waits until stop().

        Returns:
            True if everything played; False if the stream was stopped first.

        Raises:
            PlaybackTimeoutError: The sound device stopped calling back.
        """
        stalled = False
        with self._cond:
            while True:
                if self._stopped:
                    finished = False
                    break
                if self._finishing and not self._clips:
                    finished = True
                    break
                if self._stalled_locked():
                    stalled = True
                    finished = False
                    break
                self._cond.wait(_WAIT_SLICE_SECONDS)
        if stalled:
            self._raise_stalled()
        self._shutdown()
        return finished

    def pause(self) -> None:
        """Go silent at once and rewind the clip being heard to its start."""
        with self._cond:
            self._paused = True
            if self._clips:
                self._clips[0].position = 0
            self._cond.notify_all()

    def resume(self) -> None:
        """Carry on (the rewound clip plays again from its start)."""
        with self._cond:
            self._paused = False
            self._cond.notify_all()

    def stop(self) -> None:
        """End the reading now: drop queued audio, wake blocked callers, close the device."""
        with self._cond:
            self._stopped = True
            self._clips.clear()
            self._cond.notify_all()
        self._shutdown()

    def _prepared(self, audio: "npt.NDArray[np.float32]", sample_rate: int) -> "npt.NDArray[np.float32]":
        """Resample to the stream's rate if needed, then fade both ends."""
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if samples.size and self._sample_rate and sample_rate > 0 and sample_rate != self._sample_rate:
            target = max(1, int(round(samples.size * self._sample_rate / sample_rate)))
            samples = np.interp(
                np.linspace(0, samples.size - 1, target), np.arange(samples.size), samples
            ).astype(np.float32)
        fade = min(int(self._sample_rate * self._fade_seconds), samples.size // 2)
        if fade > 1:
            samples = samples.copy()
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            samples[:fade] *= ramp
            samples[-fade:] *= ramp[::-1]
        return samples

    def _stalled_locked(self) -> bool:
        """Whether clips are waiting but the device has gone quiet. Caller holds the lock."""
        return not self._paused and bool(self._clips) and self._clock() - self._last_callback > self._stall_seconds

    def _raise_stalled(self) -> NoReturn:
        """Stop everything, blame the device, and raise PlaybackTimeoutError."""
        with self._cond:
            self._stopped = True
            self._clips.clear()
            self._cond.notify_all()
        self._player.report_device_failed()
        self._shutdown()
        logger.error(
            "Speech output stalled: the sound device made no callback for over %.1fs "
            "while audio was waiting (usually a misconfigured default sound device).",
            self._stall_seconds,
        )
        raise PlaybackTimeoutError(
            f"The sound device did not play anything for over {self._stall_seconds:.0f}s."
        )

    def _shutdown(self) -> None:
        """Close the device and the helper thread once (safe to call many times)."""
        with self._cond:
            if self._shut:
                return
            self._shut = True
            self._open = False
            opening = self._opening
            self._cond.notify_all()
        if opening:
            # start() is still opening the device; it closes it when done.
            return
        self._player.close_stream()
        self._started_tags.put(None)

    def _dispatch_started(self) -> None:
        """Helper thread: deliver clip-started tags outside the audio thread."""
        while True:
            tag = self._started_tags.get()
            if tag is None:
                return
            hook = self._on_clip_started
            if hook is None or self._stopped:
                continue
            try:
                hook(tag)
            except Exception:
                logger.exception("on_clip_started hook failed")

    def _callback(
        self, outdata: "npt.NDArray[np.float32]", frames: int, time_info: object, status: object
    ) -> None:
        """Audio thread: copy queued samples into `outdata`, silence when dry."""
        if status:
            logger.warning("Audio output status: %s", status)
        out = outdata[:, 0]
        with self._cond:
            self._last_callback = self._clock()
            if self._paused or self._stopped:
                out[:] = 0
                return
            filled = 0
            while filled < frames and self._clips:
                clip = self._clips[0]
                take = min(frames - filled, len(clip.samples) - clip.position)
                if take > 0:
                    out[filled : filled + take] = clip.samples[clip.position : clip.position + take]
                    if not clip.announced:
                        clip.announced = True
                        self._played_any = True
                        self._started_tags.put(clip.tag)
                    clip.position += take
                    filled += take
                if clip.position >= len(clip.samples):
                    self._clips.popleft()
                    self._cond.notify_all()
            if filled < frames:
                out[filled:] = 0
                if self._played_any and not self._finishing:
                    self._underruns += 1
