"""Fake audio output for tests — stands in for sounddevice.OutputStream.

Both flavours pull audio through the stream's callback exactly the way a
real sound card does, so code that feeds a persistent stream is tested for
real, with no hardware:

- ManualStream: the TEST drives the audio clock by calling pull(frames).
  Fully deterministic; use it for ordering, fades, pause and stall tests.
- PumpStream: a background thread pulls audio continuously, `speedup`
  times faster than real time. Use it where a whole reading must simply
  play out (feature-level tests).
"""
import threading
import time
from typing import Any, Callable

import numpy as np
import numpy.typing as npt
import sounddevice as sd

from vocalis.audio.playback import AudioPlayer

_PUMP_FRAMES = 256


class OutputRecorder:
    """Everything the fake streams were asked to do, and all audio they pulled."""

    def __init__(self) -> None:
        """Start empty."""
        self._lock = threading.Lock()
        self.opened = 0
        self.closed = 0
        self.closed_event = threading.Event()
        self.kwargs: "list[dict[str, Any]]" = []
        self.streams: "list[Any]" = []
        self._pulled: "list[npt.NDArray[np.float32]]" = []

    def note_created(self, stream: Any, kwargs: "dict[str, Any]") -> None:
        """Remember a newly built stream and the arguments it was built with."""
        with self._lock:
            self.streams.append(stream)
            self.kwargs.append(kwargs)

    def note_started(self) -> None:
        """Count a stream whose start() ran."""
        with self._lock:
            self.opened += 1

    def note_closed(self) -> None:
        """Count a stream whose close() ran (first close of each stream only)."""
        with self._lock:
            self.closed += 1
        self.closed_event.set()

    def add_pulled(self, samples: "npt.NDArray[np.float32]") -> None:
        """Record audio handed to the 'sound card'."""
        with self._lock:
            self._pulled.append(samples)

    def audio(self) -> "npt.NDArray[np.float32]":
        """Everything pulled so far, silence included."""
        with self._lock:
            parts = list(self._pulled)
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

    def played(self) -> "npt.NDArray[np.float32]":
        """Only the non-silent samples pulled so far."""
        audio = self.audio()
        nonsilent: npt.NDArray[np.float32] = audio[audio != 0]
        return nonsilent

    @property
    def frames_pulled(self) -> int:
        """How many frames (silence included) the 'sound card' has pulled."""
        with self._lock:
            return sum(len(part) for part in self._pulled)


class ManualStream:
    """A fake output stream whose audio clock is driven by the test."""

    def __init__(self, recorder: OutputRecorder, **kwargs: Any) -> None:
        """Remember the callback; register with the recorder."""
        self._recorder = recorder
        self._callback = kwargs["callback"]
        self._finished_callback = kwargs.get("finished_callback")
        self.samplerate: int = kwargs["samplerate"]
        self.running = False
        self.closed = False
        self._finished = False
        recorder.note_created(self, dict(kwargs))

    def start(self) -> None:
        """Mark the stream running."""
        self.running = True
        self._recorder.note_started()

    def pull(self, frames: int) -> "npt.NDArray[np.float32]":
        """Ask the audio callback for `frames` frames, as the sound card would."""
        buffer = np.zeros((frames, 1), dtype=np.float32)
        try:
            self._callback(buffer, frames, None, None)
        except sd.CallbackStop:
            # A real stream swallows CallbackStop, plays out this buffer, then stops.
            self.running = False
            self._notify_finished()
        samples = buffer[:, 0].copy()
        self._recorder.add_pulled(samples)
        return samples

    def _notify_finished(self) -> None:
        """Tell the finished callback once, as a real stream does when it ends."""
        if self._finished_callback is not None and not self._finished:
            self._finished = True
            self._finished_callback()

    def stop(self) -> None:
        """Mark the stream stopped."""
        self.running = False

    def close(self) -> None:
        """Mark the stream closed (counted once)."""
        if not self.closed:
            self.closed = True
            self._recorder.note_closed()


class PumpStream:
    """A fake output stream that pulls audio on its own thread, faster than real time."""

    def __init__(self, recorder: OutputRecorder, speedup: float = 50.0, **kwargs: Any) -> None:
        """Remember the callbacks; register with the recorder."""
        self._recorder = recorder
        self._callback = kwargs["callback"]
        self._finished_callback = kwargs["finished_callback"]
        self.samplerate: int = kwargs["samplerate"]
        self._interval = _PUMP_FRAMES / self.samplerate / speedup
        self._halt = threading.Event()
        self._thread: "threading.Thread | None" = None
        self._finish_lock = threading.Lock()
        self._finished = False
        self.closed = False
        recorder.note_created(self, dict(kwargs))

    def start(self) -> None:
        """Start pulling audio on a background thread."""
        self._recorder.note_started()
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        """Pull audio until stopped, or until the callback ends the stream."""
        while not self._halt.is_set():
            buffer = np.zeros((_PUMP_FRAMES, 1), dtype=np.float32)
            try:
                self._callback(buffer, _PUMP_FRAMES, None, None)
            except sd.CallbackStop:
                self._recorder.add_pulled(buffer[:, 0].copy())
                self._notify_finished()
                return
            self._recorder.add_pulled(buffer[:, 0].copy())
            time.sleep(self._interval)

    def _notify_finished(self) -> None:
        """Tell the finished callback exactly once, however the stream ends."""
        with self._finish_lock:
            if self._finished:
                return
            self._finished = True
        self._finished_callback()

    def stop(self) -> None:
        """Stop pulling audio and wait for the thread (unless called from it)."""
        self._halt.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)

    def close(self) -> None:
        """Stop, then count the close once and tell the finished callback."""
        self.stop()
        if not self.closed:
            self.closed = True
            self._recorder.note_closed()
            self._notify_finished()


def manual_stream_factory(recorder: OutputRecorder) -> "Callable[..., ManualStream]":
    """A stream_factory for AudioPlayer that builds ManualStreams."""

    def factory(**kwargs: Any) -> ManualStream:
        """Build one ManualStream."""
        return ManualStream(recorder, **kwargs)

    return factory


def pump_stream_factory(recorder: OutputRecorder, speedup: float = 50.0) -> "Callable[..., PumpStream]":
    """A stream_factory for AudioPlayer that builds PumpStreams."""

    def factory(**kwargs: Any) -> PumpStream:
        """Build one PumpStream."""
        return PumpStream(recorder, speedup=speedup, **kwargs)

    return factory


class FakeAudioPlayer(AudioPlayer):
    """A real AudioPlayer wired to a PumpStream; `output` records what it played.

    Drop-in for tests that patch `vocalis.cli.AudioPlayer`. `done_event` is
    set the first time a stream is closed (i.e. a reading finished or stopped).
    """

    def __init__(self, speedup: float = 50.0) -> None:
        """Create the recorder and wire the player to it."""
        self.output = OutputRecorder()
        super().__init__(stream_factory=pump_stream_factory(self.output, speedup))

    @property
    def done_event(self) -> threading.Event:
        """Set once the first stream has been closed."""
        return self.output.closed_event


def wait_until(predicate: "Callable[[], bool]", timeout: float = 5.0) -> None:
    """Block until `predicate()` is true; raise AssertionError after `timeout` seconds."""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError(f"condition not met within {timeout} s")
        time.sleep(0.005)
