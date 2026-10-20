"""Orchestrates Document Reader: extract text from a file (or take
pasted text directly), normalize + chunk it, speak each chunk via
Qwen3-TTS (falling back to Piper on any failure), and play it back chunk
by chunk -- with pause/resume/stop, progress reporting, and "one reading
at a time" shared with Quick Reader via ReadingCoordinator.

Has no PySide6/Qt import at all: status, progress, and completion are
reported through plain callables supplied by the caller (the popup, see
ui/doc_popup.py), keeping this module independently testable exactly
like every other feature module. Reuses the never-block-the-caller and
generation-counter patterns already hard-won in quick_reader.py.

Reading design: the document is split into sentence-bounded chunks and
handed to a ChunkedSpeaker (features/chunked_speaker.py), which makes each
chunk with the voice chain while the previous ones are being heard, so the
reading plays on one continuous stream with no gaps. pause() silences the
stream at once and resume() replays the chunk being heard from its start;
stop() (or a ReadingCoordinator preemption) ends the reading at any point,
including before the first sound exists. The voice stays loaded for the
whole reading and is released when playback has ended.
"""
import logging
import queue
import threading
from pathlib import Path
from typing import Callable, Mapping

from vocalis.audio.playback import AudioPlayer
from vocalis.engines.base import Synthesizer
from vocalis.engines.chain import EngineChain, make_downshift_handler, new_reading_chain
from vocalis.engines.registry import EngineSelector
from vocalis.features.chunked_speaker import ChunkedSpeaker, SpeakOutcome
from vocalis.features.reading_coordinator import ReadingCoordinator
from vocalis.model_manager import ModelEngine, ModelManager
from vocalis.notify import Notifier
from vocalis.text.chunker import chunk_text
from vocalis.text.extractors import (
    DocumentOpenError,
    DocumentTooLargeError,
    NoReadableTextError,
    UnsupportedFileTypeError,
    extract_text,
)
from vocalis.text.normalizer import normalize_markdown

logger = logging.getLogger("vocalis")

_TITLE = "Vocalis — Document Reader"

_EXTRACTION_ERRORS = (
    UnsupportedFileTypeError,
    DocumentOpenError,
    NoReadableTextError,
    DocumentTooLargeError,
)


class DocumentReaderFeature:
    """Wires Ctrl+Alt+D / the popup's controls to extract -> chunk ->
    speak (with fallback) -> play, chunk by chunk, with pause/resume/stop."""

    def __init__(
        self,
        model_manager: ModelManager,
        primary_engine_factory: Callable[[], ModelEngine],
        fallback_engine: Synthesizer,
        player: AudioPlayer,
        notifier: Notifier,
        reading_coordinator: ReadingCoordinator,
        on_status: Callable[[str], None],
        on_progress: Callable[[int, int], None],
        on_finished: Callable[[], None],
        language: str = "English",
        on_started: Callable[[], None] | None = None,
        speed_provider: Callable[[], float] | None = None,
        engine_selector: EngineSelector | None = None,
        engine_factories: Mapping[str, Callable[[], ModelEngine]] | None = None,
    ) -> None:
        """Create a DocumentReaderFeature.

        Args:
            model_manager: Shared ModelManager (same instance every
                other feature uses) enforcing single-GPU-residency; used
                only for the primary engine.
            primary_engine_factory: Constructs a fresh primary engine
                (Qwen3TTS) when it needs to be (re)loaded.
            fallback_engine: An already-loaded, CPU-only engine (Piper)
                used when the primary engine fails.
            player: Plays generated speech and can be interrupted.
            notifier: Sends desktop notifications.
            reading_coordinator: Shared with Quick Reader so only one
                feature plays audio at a time.
            on_status: Called with a human-readable status string at
                each stage (e.g. "Reading...", "Paused", "Done"). The
                popup wires this to a Qt signal to update its status line.
            on_progress: Called with (current_chunk, total_chunks), both
                1-based, at the start of each chunk. The popup wires this
                to a Qt signal to update its progress label.
            on_finished: Called once when a reading ends for any reason
                (finished, stopped, or failed). The popup wires this to
                a Qt signal that re-enables its controls.
            language: Spoken language. Currently only stored for future
                use; it is not yet passed to the engine. The two readers
                can share one engine id only while that holds.
            on_started: Called with no arguments when a reading starts;
                the app uses it to show the widget. A callback that
                raises is logged and ignored. None means no hook.
            speed_provider: Called (with no arguments) once per chunk,
                right after that chunk is synthesized, to get the
                current speaking speed to apply before playback. None
                (the default) means no pacing is applied. A failure
                reading the speed or applying it never breaks the
                reading -- see vocalis.audio.pace.paced_audio.
        """
        self._model_manager = model_manager
        self._primary_engine_factory = primary_engine_factory
        self._fallback_engine = fallback_engine
        self._player = player
        self._notifier = notifier
        self._reading_coordinator = reading_coordinator
        self._on_status = on_status
        self._on_progress = on_progress
        self._on_finished = on_finished
        self._language = language
        self._on_started = on_started
        self._speed_provider = speed_provider
        self._engine_selector = engine_selector
        self._engine_factories = engine_factories

        self._lock = threading.Lock()
        self._active = False
        self._paused = False
        self._speaker: ChunkedSpeaker | None = None
        self._generation = 0

        self._notify_queue: "queue.Queue[str]" = queue.Queue()
        self._notify_thread: "threading.Thread | None" = None

    @property
    def is_reading(self) -> bool:
        """Whether a reading is currently active (including while paused)."""
        with self._lock:
            return self._active

    @property
    def generation(self) -> int:
        """Number of the most recent reading started (0 before any).

        Lets a caller that started a reading tell later whether the
        reading now in progress is still its own.
        """
        with self._lock:
            return self._generation

    def read_file(self, path: Path) -> None:
        """Start reading a document file aloud from the beginning.

        Extraction, normalization, and chunking all happen on the
        background worker thread spawned here, so this call returns
        immediately -- safe to call from the Qt UI thread.

        Args:
            path: Path to the file to read.
        """
        generation = self._start_new_reading()
        self._reading_coordinator.start_reading(self, self.stop)
        # Announce only after the coordinator has stopped any other
        # feature's reading: that reading's "ended" report must reach the
        # status hub BEFORE this "started", or the hub would ignore ours
        # (already busy) and then go idle while this document plays.
        self._report_started()
        threading.Thread(
            target=self._prepare_and_read, args=(generation, path, None), daemon=True
        ).start()

    def read_text(self, text: str) -> None:
        """Start reading pasted text aloud from the beginning.

        Args:
            text: The raw text to read (markdown-normalized before speaking).
        """
        generation = self._start_new_reading()
        self._reading_coordinator.start_reading(self, self.stop)
        # Announce only after the coordinator has stopped any other
        # feature's reading: that reading's "ended" report must reach the
        # status hub BEFORE this "started", or the hub would ignore ours
        # (already busy) and then go idle while this document plays.
        self._report_started()
        threading.Thread(
            target=self._prepare_and_read, args=(generation, None, text), daemon=True
        ).start()

    def _report_started(self) -> None:
        """Tell the optional on_started hook a reading began; a failing
        hook is logged and never breaks the reading."""
        if self._on_started is None:
            return
        try:
            self._on_started()
        except Exception:
            logger.exception("on_started hook failed")

    def pause(self) -> None:
        """Pause the current reading immediately. No-op if not reading
        or already paused."""
        with self._lock:
            if not self._active or self._paused:
                return
            self._paused = True
            speaker = self._speaker
        if speaker is not None:
            speaker.pause()
        self._notify_async("Paused")
        self._on_status("Paused")

    def resume(self) -> None:
        """Resume a paused reading, replaying the chunk being heard from its
        start. No-op if not currently paused."""
        with self._lock:
            if not self._active or not self._paused:
                return
            self._paused = False
            speaker = self._speaker
        if speaker is not None:
            speaker.resume()
        self._notify_async("Resuming...")
        self._on_status("Resuming...")

    def stop(self) -> None:
        """Stop the current reading immediately. No-op if not reading.

        Safe to call from any thread, including as a ReadingCoordinator
        stopper callback when Quick Reader preempts this reading. Works at
        any point of a reading -- even before the first sound exists: the
        speaker's stop is a permanent latch, so a late-finishing first
        piece never plays.
        """
        with self._lock:
            if not self._active:
                return
            self._active = False
            self._paused = False
            speaker, self._speaker = self._speaker, None
        self._reading_coordinator.finish_reading(self)
        if speaker is not None:
            speaker.stop()
        self._notify_async("Stopped")
        self._on_status("Stopped")
        self._on_finished()

    def _notify_async(self, message: str) -> None:
        """Queue a desktop notification for in-order delivery by a single
        background thread, so callers on the hotkey or Qt UI thread never
        wait for notify-send and messages never arrive out of order."""
        with self._lock:
            if self._notify_thread is None:
                self._notify_thread = threading.Thread(
                    target=self._deliver_notifications, daemon=True
                )
                self._notify_thread.start()
        self._notify_queue.put(message)

    def _deliver_notifications(self) -> None:
        """Consumer loop: send queued notifications one at a time, in
        order. A failing notify() is logged and never ends the loop."""
        while True:
            message = self._notify_queue.get()
            try:
                self._notifier.notify(_TITLE, message)
            except Exception:
                logger.exception("Desktop notification failed")

    def _start_new_reading(self) -> int:
        """Mark a fresh reading as active and return its generation number."""
        with self._lock:
            self._active = True
            self._paused = False
            self._speaker = None
            self._generation += 1
            return self._generation

    def _is_current(self, generation: int) -> bool:
        """Whether `generation` is still the active reading."""
        with self._lock:
            return self._active and self._generation == generation

    def _deactivate(self, generation: int) -> bool:
        """Mark `generation` as finished and release ReadingCoordinator
        ownership, but only if `generation` is still current and still
        active -- a stale worker thread must not clobber a newer
        reading's state, and a concurrent stop() (direct, or via a
        ReadingCoordinator preemption) may have already deactivated this
        same generation an instant earlier.

        Returns:
            True if this call is the one that actually transitioned the
            reading from active to inactive -- the caller should go on to
            fire its notify/on_status/on_finished calls. False if this
            generation is stale, or a concurrent stop() already made that
            transition (and therefore already fired its own
            notify/on_status("Stopped")/on_finished()) -- the caller must
            not fire its own on_finished() a second time in that case.
        """
        with self._lock:
            if self._generation != generation or not self._active:
                return False
            self._active = False
            self._paused = False
            # Under the feature lock so a newer reading cannot claim the
            # coordinator in between and then lose its claim to this call.
            # Safe: the coordinator never calls a stopper under its own lock.
            self._reading_coordinator.finish_reading(self)
        return True

    def _prepare_and_read(
        self, generation: int, path: "Path | None", text: "str | None"
    ) -> None:
        """Extract (if a file), normalize, and chunk the text, then read
        it chunk by chunk. Runs entirely on a background thread."""
        try:
            raw_text = extract_text(path) if path is not None else text or ""
        except _EXTRACTION_ERRORS as exc:
            logger.warning("Document extraction failed: %s", exc)
            if self._deactivate(generation):
                self._on_status(str(exc))
                self._on_finished()
                # Notify last: it can block, and the end must reach the hub first.
                self._notifier.notify(_TITLE, str(exc))
            return

        normalized = normalize_markdown(raw_text)
        chunks = chunk_text(normalized)
        if not chunks:
            if self._deactivate(generation):
                self._on_status("Nothing to read")
                self._on_finished()
                # Notify last: it can block, and the end must reach the hub first.
                self._notifier.notify(_TITLE, "Nothing to read")
            return

        if not self._is_current(generation):
            return  # stopped while extracting/chunking

        self._notifier.notify(_TITLE, "Reading...")
        self._on_status("Reading...")
        self._read_worker(generation, chunks)

    def _read_worker(self, generation: int, chunks: "list[str]") -> None:
        """Hand the chunks to a ChunkedSpeaker and turn its outcome into the
        usual status / notification / finished reports.

        A pause() that landed before the speaker existed is applied here,
        under the same lock that publishes the speaker, so a pause or stop
        can never fall between "no speaker yet" and "speaker running".
        """
        speaker = ChunkedSpeaker(
            player=self._player,
            source_factory=self._new_source,
            speed_provider=self._speed_provider,
            on_chunk_started=lambda current, total: self._report_progress(generation, current, total),
        )
        with self._lock:
            if not (self._active and self._generation == generation):
                return  # stopped while extracting/chunking
            self._speaker = speaker
            start_paused = self._paused
        if start_paused:
            speaker.pause()
        outcome = speaker.run(chunks)
        with self._lock:
            if self._speaker is speaker:
                self._speaker = None
        self._finish_reading(generation, outcome)

    def _report_progress(self, generation: int, current: int, total: int) -> None:
        """Forward 'piece N of M' while this reading is still the current one."""
        if self._is_current(generation):
            self._on_progress(current, total)

    def _finish_reading(self, generation: int, outcome: SpeakOutcome) -> None:
        """Report how the reading ended (a stop() has already reported its own end)."""
        if outcome is SpeakOutcome.SPEECH_FAILED:
            message = "Could not generate speech"
        elif outcome is SpeakOutcome.PLAYBACK_FAILED:
            message = "Playback failed"
        elif outcome is SpeakOutcome.DONE:
            message = "Done"
        else:
            # STOPPED: stop() already ended and reported this reading. If it
            # is somehow still current, end it quietly so nothing stays "active".
            if self._deactivate(generation):
                self._on_finished()
            return
        if self._deactivate(generation):
            self._on_status(message)
            self._on_finished()
            # Notify last: it can block, and the end must reach the hub first.
            self._notifier.notify(_TITLE, message)

    def _new_source(self) -> EngineChain:
        """Build the voice chain for one reading; it holds the voice loaded until closed."""
        return new_reading_chain(
            self._engine_selector,
            self._engine_factories,
            self._primary_engine_factory,
            self._model_manager,
            self._fallback_engine,
            make_downshift_handler(
                lambda message: self._notifier.notify(_TITLE, message), self._on_status
            ),
            hold_engines=True,
        )
