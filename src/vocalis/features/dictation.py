"""Orchestrates the dictation feature: on NUMLOCK press, start recording;
on release, stop recording, transcribe via the Parakeet engine (borrowed
from ModelManager), and deliver the text (typed, or clipboard fallback) —
with desktop notifications at each stage.
"""
import logging
import threading
from typing import Callable, Protocol, cast

import numpy as np
import numpy.typing as npt

from vocalis.model_manager import ModelEngine, ModelManager
from vocalis.notify import Notifier
from vocalis.output.typer import TextOutput

logger = logging.getLogger("vocalis")

_ENGINE_ID = "parakeet"
_TITLE = "Vocalis — Dictation"
NO_AUDIO_MESSAGE = (
    "No sound was recorded. Check that your microphone is plugged in, not muted, "
    "and chosen as the input device, then try again."
)


class Transcriber(Protocol):
    """The subset of ParakeetSTT this module depends on."""

    def transcribe(self, audio: npt.NDArray[np.float32], sample_rate: int) -> str: ...


class Recorder(Protocol):
    """The subset of MicRecorder this module depends on."""

    def start(self) -> None: ...
    def stop(self) -> npt.NDArray[np.float32]: ...


class DictationFeature:
    """Wires NUMLOCK press/release to record -> transcribe -> deliver."""

    def __init__(
        self,
        model_manager: ModelManager,
        engine_factory: Callable[[], ModelEngine],
        recorder: Recorder,
        text_output: TextOutput,
        notifier: Notifier,
        sample_rate: int = 16000,
        on_listening_changed: Callable[[bool], None] | None = None,
    ) -> None:
        """Create a DictationFeature.

        Args:
            model_manager: Shared ModelManager enforcing single-residency.
            engine_factory: Constructs a fresh engine (e.g. ParakeetSTT)
                when the model needs to be (re)loaded. The constructed
                engine must also implement Transcriber (have a
                transcribe() method) — ModelManager itself only requires
                the ModelEngine protocol, so this is a documented
                requirement on the factory, not something the type system
                enforces here.
            recorder: Microphone recorder.
            text_output: Delivers transcribed text (typed or clipboard).
            notifier: Sends start/transcribing/done notifications.
            sample_rate: Sample rate to record and transcribe at.
            on_listening_changed: Called with True once recording has
                started and False when the key is released. A callback
                that raises is logged and ignored. None means no hook.
        """
        self._model_manager = model_manager
        self._engine_factory = engine_factory
        self._recorder = recorder
        self._text_output = text_output
        self._notifier = notifier
        self._sample_rate = sample_rate
        self._on_listening_changed = on_listening_changed
        self._active_lock = threading.Lock()
        self._active = False

    def on_press(self) -> None:
        """Call when NUMLOCK is pressed: begin recording."""
        with self._active_lock:
            if self._active:
                logger.info("Ignoring duplicate NUMLOCK press while already recording")
                return
            self._active = True
        self._notifier.notify(_TITLE, "Listening...")
        try:
            self._recorder.start()
            started = True
        except Exception:
            logger.exception("Failed to start recording")
            self._notifier.notify(_TITLE, "Could not start recording")
            with self._active_lock:
                self._active = False
            started = False
        if started:
            self._report_listening(True)

    def _report_listening(self, active: bool) -> None:
        """Tell the optional hook whether we are listening; a failing
        hook is logged and never breaks dictation."""
        if self._on_listening_changed is None:
            return
        try:
            self._on_listening_changed(active)
        except Exception:
            logger.exception("on_listening_changed hook failed")

    def on_release(self) -> None:
        """Call when NUMLOCK is released: transcribe and deliver the text."""
        with self._active_lock:
            if not self._active:
                logger.info("Ignoring duplicate NUMLOCK release while not recording")
                return
            self._active = False
        self._report_listening(False)
        try:
            audio = self._recorder.stop()
        except Exception:
            logger.exception("Failed to stop recording")
            self._notifier.notify(_TITLE, "Recording failed")
            return

        if audio.size == 0:
            self._notifier.notify(_TITLE, NO_AUDIO_MESSAGE)
            return

        self._notifier.notify(_TITLE, "Transcribing...")
        try:
            engine = self._model_manager.acquire(_ENGINE_ID, self._engine_factory)
            try:
                text = cast(Transcriber, engine).transcribe(audio, sample_rate=self._sample_rate)
            finally:
                self._model_manager.release(_ENGINE_ID)
        except Exception:
            logger.exception("Transcription failed")
            self._notifier.notify(_TITLE, "Transcription failed")
            return

        if not text.strip():
            self._notifier.notify(_TITLE, "(no speech detected)")
            return

        self._notifier.notify(_TITLE, text)
        result = self._text_output.deliver(text)
        if result.method == "clipboard":
            self._notifier.notify(_TITLE, "Typing failed — copied to clipboard instead")
        else:
            self._notifier.notify(_TITLE, "Done")
