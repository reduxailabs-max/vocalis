"""Speaks a list of text chunks as one continuous, gap-free reading.

Shared by Quick Reader and Document Reader. run() is a producer loop on the
caller's worker thread: it makes each chunk with the voice source (one
EngineChain holds the voice loaded for the whole reading), applies the
speaking pace, and pushes the audio into a SpeechStream; the sound card
plays from that stream on its own thread. Because the stream's queue is
bounded, the producer stays a few chunks ahead of the listener and then
waits.

The voice is released (and so ModelManager's idle-unload timer starts)
only when run() leaves — i.e. when playback has ended, not when the last
chunk was merely synthesized.

One ChunkedSpeaker is made per reading. stop() is a permanent latch: even
if it lands before run() has opened the stream (while the model is still
loading), run() returns STOPPED without ever making sound.
"""
import logging
import threading
from enum import Enum
from typing import Callable, Protocol, Sequence

import numpy as np
import numpy.typing as npt

from vocalis.audio.pace import paced_audio
from vocalis.audio.playback import AudioPlayer
from vocalis.audio.speech_stream import (
    DEFAULT_FADE_SECONDS,
    DEFAULT_MAX_QUEUED_CLIPS,
    DEFAULT_STALL_SECONDS,
    SpeechStream,
)

logger = logging.getLogger("vocalis")


class SpeechSource(Protocol):
    """Makes speech for one reading and releases its voice afterwards."""

    def synthesize(self, text: str) -> "tuple[npt.NDArray[np.float32], int]":
        """Speak `text`; returns mono float32 audio and its sample rate."""
        ...

    def close(self) -> None:
        """Release the voice (idempotent)."""
        ...


class SpeakOutcome(Enum):
    """How a reading ended."""

    DONE = "done"
    STOPPED = "stopped"
    SPEECH_FAILED = "speech_failed"
    PLAYBACK_FAILED = "playback_failed"


class ChunkedSpeaker:
    """One reading: make each chunk, queue it, let the stream play it."""

    def __init__(
        self,
        player: AudioPlayer,
        source_factory: Callable[[], SpeechSource],
        speed_provider: Callable[[], float] | None = None,
        on_chunk_started: Callable[[int, int], None] | None = None,
        fade_seconds: float = DEFAULT_FADE_SECONDS,
        stall_seconds: float = DEFAULT_STALL_SECONDS,
        max_queued_clips: int = DEFAULT_MAX_QUEUED_CLIPS,
    ) -> None:
        """Create the speaker for ONE reading.

        Args:
            player: Opens the real output stream.
            source_factory: Builds the voice source for this reading (called once in run()).
            speed_provider: Current speaking pace, read once per chunk; None = no pacing.
            on_chunk_started: Called with (current, total), 1-based, when a
                chunk starts to be heard. Runs on a helper thread.
            fade_seconds: Fade at each end of every chunk's audio.
            stall_seconds: Silence tolerated from a broken sound device.
            max_queued_clips: How far ahead of the listener the producer may get.
        """
        self._source_factory = source_factory
        self._speed_provider = speed_provider
        self._on_chunk_started = on_chunk_started
        self._stop = threading.Event()
        self._total = 0
        self._stream = SpeechStream(
            player,
            on_clip_started=self._clip_started,
            fade_seconds=fade_seconds,
            stall_seconds=stall_seconds,
            max_queued_clips=max_queued_clips,
        )

    @property
    def queued_clips(self) -> int:
        """Chunks currently queued in the stream (playing + waiting)."""
        return self._stream.queued_clips

    def pause(self) -> None:
        """Go silent at once (the heard chunk replays from its start on resume)."""
        self._stream.pause()

    def resume(self) -> None:
        """Carry on after pause()."""
        self._stream.resume()

    def stop(self) -> None:
        """End the reading for good; safe before or during run()."""
        self._stop.set()
        self._stream.stop()

    def run(self, chunks: Sequence[str]) -> SpeakOutcome:
        """Speak every chunk in order; block until the reading is over.

        Returns:
            DONE if everything played; STOPPED if stop() ended it;
            SPEECH_FAILED if no voice could make a chunk; PLAYBACK_FAILED if
            the sound device failed or stalled.
        """
        self._total = len(chunks)
        if self._stop.is_set():
            return SpeakOutcome.STOPPED
        try:
            source = self._source_factory()
        except Exception:
            logger.exception("Could not prepare the voice engines")
            return SpeakOutcome.STOPPED if self._stop.is_set() else SpeakOutcome.SPEECH_FAILED
        outcome = SpeakOutcome.STOPPED
        try:
            outcome = self._speak(source, chunks)
            return outcome
        finally:
            # After a clean finish the stream has already shut itself down. Calling
            # stop() then would only set its stopped flag and make the helper thread
            # drop the last chunk-started notifications that are still queued.
            if outcome is not SpeakOutcome.DONE:
                self._stream.stop()
            try:
                source.close()
            except Exception:
                logger.exception("Could not release the voice engine")

    def _speak(self, source: SpeechSource, chunks: Sequence[str]) -> SpeakOutcome:
        """The producer loop (see run())."""
        opened = False
        for index, text in enumerate(chunks):
            if self._stop.is_set():
                return SpeakOutcome.STOPPED
            try:
                audio, sample_rate = source.synthesize(text)
            except Exception:
                if self._stop.is_set():
                    return SpeakOutcome.STOPPED
                logger.exception("Both TTS engines failed")
                return SpeakOutcome.SPEECH_FAILED
            if self._stop.is_set():
                return SpeakOutcome.STOPPED
            audio = paced_audio(audio, sample_rate, self._speed_provider)
            try:
                if not opened:
                    self._stream.start(sample_rate)
                    opened = True
                if not self._stream.push(audio, sample_rate, index):
                    return SpeakOutcome.STOPPED
            except Exception:
                if self._stop.is_set():
                    return SpeakOutcome.STOPPED
                logger.exception("Playback failed")
                return SpeakOutcome.PLAYBACK_FAILED
        if not opened:
            # Nothing to say (or stop() landed while the last piece was being made).
            return SpeakOutcome.STOPPED if self._stop.is_set() else SpeakOutcome.DONE
        self._stream.finish()
        try:
            finished = self._stream.wait_until_done()
        except Exception:
            if self._stop.is_set():
                return SpeakOutcome.STOPPED
            logger.exception("Playback failed")
            return SpeakOutcome.PLAYBACK_FAILED
        return SpeakOutcome.DONE if finished else SpeakOutcome.STOPPED

    def _clip_started(self, tag: int) -> None:
        """Stream hook: chunk `tag` (0-based) has started to be heard."""
        if self._on_chunk_started is not None:
            self._on_chunk_started(tag + 1, self._total)
