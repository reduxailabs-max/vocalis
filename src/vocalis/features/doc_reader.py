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

Pause/resume design: pause() stops audio immediately; resume() replays
the current chunk from its start (simpler than tracking a mid-chunk
playback offset, and each chunk is short by construction -- see
text/chunker.py). A worker thread that is stopped early by pause()
blocks on self._resume_event until resume() sets it, then re-checks
whether this reading is still current before replaying -- this is what
lets stop() (or a ReadingCoordinator preemption) while paused actually
end the reading, rather than leaving the worker thread stuck waiting for
a resume() that will never come.
"""
import logging
import queue
import threading
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import numpy.typing as npt

from vocalis.audio.pace import paced_audio
from vocalis.audio.playback import AudioPlayer
from vocalis.engines.base import Synthesizer
from vocalis.engines.chain import EngineChain
from vocalis.engines.registry import EnginePlan, EngineSelector
from vocalis.features.reading_coordinator import ReadingCoordinator
from vocalis.model_manager import ModelEngine, ModelManager, TTS_ENGINE_ID
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
        self._chain: EngineChain | None = None
        self._generation = 0
        self._resume_event = threading.Event()
        self._resume_event.set()

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
            self._resume_event.clear()
        self._player.stop()
        self._notify_async("Paused")
        self._on_status("Paused")

    def resume(self) -> None:
        """Resume a paused reading, replaying the current chunk from its
        start. No-op if not currently paused."""
        with self._lock:
            if not self._active or not self._paused:
                return
            self._paused = False
        self._resume_event.set()
        self._notify_async("Resuming...")
        self._on_status("Resuming...")

    def stop(self) -> None:
        """Stop the current reading immediately. No-op if not reading.

        Safe to call from any thread, including as a ReadingCoordinator
        stopper callback when Quick Reader preempts this reading.
        Unblocks a worker thread that is currently waiting on a pause,
        so a stop while paused actually ends the reading instead of
        leaving that thread stuck forever.
        """
        with self._lock:
            if not self._active:
                return
            self._active = False
            self._paused = False
        self._resume_event.set()
        self._reading_coordinator.finish_reading(self)
        self._player.stop()
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
        """Mark a fresh reading as active and return its generation
        number. Also unconditionally releases any stale pause-wait from
        a previous reading (belt-and-braces: the popup disables its Read
        button while is_reading is True, so this should not be reachable
        mid-pause in normal use, but a blocked-forever worker thread
        would be a real bug if it ever were)."""
        with self._lock:
            self._active = True
            self._paused = False
            self._chain = None
            self._generation += 1
            generation = self._generation
        self._resume_event.set()
        return generation

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
        """Synthesize and play each chunk in order. Bails out cleanly at
        each check if this generation has been superseded by a stop, a
        coordinator preemption, or a fresh reading.

        Each chunk is synthesized and paced (see paced_audio()) exactly
        once; a pause/resume replay reuses that same already-paced audio
        rather than re-synthesizing and re-pacing it.

        Waits on _resume_event at two points so a pause() is honored
        whether it lands during synthesis or during playback: once right
        before play() is called (covers a pause that happened while this
        chunk was still synthesizing, when there was no stream yet for
        pause() to stop()), and once after wait_until_done() returns early
        (covers a pause that stopped an already-playing chunk). Both waits
        re-check _is_current() immediately after waking, so a stop() (or a
        ReadingCoordinator preemption, which stops via the same path) that
        arrives while waiting ends the reading cleanly instead of playing
        the chunk or replaying it.
        """
        total = len(chunks)
        index = 0
        while index < total:
            if not self._is_current(generation):
                return
            chunk = chunks[index]
            self._on_progress(index + 1, total)
            try:
                audio, sample_rate = self._synthesize_with_fallback(chunk, generation)
            except Exception:
                logger.exception("Both TTS engines failed")
                if self._deactivate(generation):
                    self._on_status("Could not generate speech")
                    self._on_finished()
                    # Notify last: it can block, and the end must reach the hub first.
                    self._notifier.notify(_TITLE, "Could not generate speech")
                return
            if not self._is_current(generation):
                return
            audio = paced_audio(audio, sample_rate, self._speed_provider)

            # Inner retry loop: a pause/resume cycle replays this exact
            # already-paced `audio` without ever calling
            # _synthesize_with_fallback() or paced_audio() again for it.
            while True:
                finished_naturally = self._play_chunk_until_done(generation, audio, sample_rate)
                if finished_naturally is None:
                    return  # reading ended (stopped, preempted, or playback failed)
                if finished_naturally:
                    break
            index += 1

        if self._deactivate(generation):
            self._on_status("Done")
            self._on_finished()
            # Notify last: it can block, and the end must reach the hub first.
            self._notifier.notify(_TITLE, "Done")

    def _play_chunk_until_done(
        self, generation: int, audio: "npt.NDArray[np.float32]", sample_rate: int
    ) -> "bool | None":
        """Play one already-synthesized, already-paced chunk, honoring a
        pause that lands either before playback starts or while it is
        under way.

        Returns:
            True if the chunk finished playing naturally (the caller
            should move on to the next chunk); False if it was paused
            (the caller should replay the exact same `audio` once
            resumed); None if the reading ended for any other reason
            (stopped, preempted, or playback failed) -- the caller must
            return immediately without doing anything further.
        """
        # A pause() landing while this chunk was still synthesizing
        # cleared _resume_event but had no stream to stop() yet (play()
        # hasn't been called). Without this wait, playback would start
        # anyway and the pause would be silently ignored until the next
        # chunk boundary. Block here until resume() (or a stop()/
        # preemption, which also sets the event to avoid hanging) wakes
        # this thread, then re-check before actually starting playback.
        self._resume_event.wait()
        if not self._is_current(generation):
            return None  # stopped (or preempted) while paused before playback started

        try:
            self._player.play(audio, sample_rate)
            # A stop() landing between the last _is_current() check and
            # play() found no stream to stop; stop it now rather than
            # let this chunk play out. The same goes for a pause()
            # landing in that window: stop the chunk, then fall
            # through to the paused-wait below so resume() replays it.
            if not self._is_current(generation) or not self._resume_event.is_set():
                self._player.stop()
            finished_naturally = self._player.wait_until_done()
        except Exception:
            logger.exception("Playback failed")
            if self._deactivate(generation):
                self._on_status("Playback failed")
                self._on_finished()
                # Notify last: it can block, and the end must reach the hub first.
                self._notifier.notify(_TITLE, "Playback failed")
            return None

        if finished_naturally:
            return True

        if not self._is_current(generation):
            # A real stop() -- direct, or via a ReadingCoordinator
            # preemption -- already deactivated this generation and
            # called on_finished() itself before unblocking
            # wait_until_done(), so there is nothing left to do here.
            # Checking self._paused instead would race with a fast
            # resume() that flips it back to False before this check
            # runs, which would wrongly take this branch for a
            # genuine pause and end the reading early.
            return None
        self._resume_event.wait()
        if not self._is_current(generation):
            return None  # stopped while waiting for resume()
        return False  # replay the current chunk once resumed

    def _new_chain(self) -> EngineChain:
        """Build the engine chain for one reading: the selector's plan, or the old Qwen3-TTS-then-Piper."""
        if self._engine_selector is not None:
            plan = self._engine_selector.plan()
            factories = self._engine_factories or {}
        else:
            plan = EnginePlan((TTS_ENGINE_ID,), "Fixed to the built-in voice.")
            factories = {TTS_ENGINE_ID: self._primary_engine_factory}
        return EngineChain(plan, factories, self._model_manager, self._fallback_engine, self._notify_downshift)

    def _notify_downshift(self, failed_engine_id: str, next_engine_id: "str | None") -> None:
        """Tell the user the first-choice voice failed (silent if the backup voice itself failed)."""
        if failed_engine_id == "piper":
            return
        self._notifier.notify(_TITLE, "Primary voice unavailable — using backup voice")
        self._on_status("Using backup voice")

    def _synthesize_with_fallback(self, text: str, generation: "int | None" = None) -> "tuple[npt.NDArray[np.float32], int]":
        """Speak one piece; the same chain is reused for the whole document so a dead engine is not retried.

        A chain created for a reading that has since been stopped is used for
        this call only and is not stored, so it cannot leak into the next reading.
        """
        chain = self._chain
        if chain is None:
            chain = self._new_chain()
            with self._lock:
                if generation is None or (self._active and self._generation == generation):
                    self._chain = chain
        return chain.synthesize(text)
