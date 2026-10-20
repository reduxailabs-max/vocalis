"""Orchestrates Quick Reader: on Ctrl+Alt+R, read the clipboard, speak it
via Qwen3-TTS (falling back to Piper on any failure), and play it back —
with desktop notifications at each stage, and press-again-to-stop.

The clipboard text is split into sentence chunks and handed to a
ChunkedSpeaker (features/chunked_speaker.py), which makes each chunk while
the previous ones are being heard, so even a long text plays on one
continuous stream with no gaps.

Never blocks the caller (the hotkey listener's watcher thread): on_trigger()
only ever does a quick lock-guarded flag flip itself, then hands all real
work — including stopping playback and sending notifications — to a
freshly spawned background thread. This is essential for "press again to
stop" to actually work — evdev's read_loop() processes one event at a
time per device, so if on_trigger() ever blocked (even briefly, on a
notify-send subprocess or a stream teardown), a second press could be
delayed or missed until that call returned.
"""
import logging
import threading
from typing import Callable, Mapping

from vocalis.audio.playback import AudioPlayer
from vocalis.engines.base import Synthesizer
from vocalis.engines.chain import EngineChain, make_downshift_handler, new_reading_chain
from vocalis.engines.registry import EngineSelector
from vocalis.features.chunked_speaker import ChunkedSpeaker, SpeakOutcome
from vocalis.features.reading_coordinator import ReadingCoordinator
from vocalis.model_manager import ModelEngine, ModelManager
from vocalis.notify import Notifier
from vocalis.output.clipboard import ClipboardReader
from vocalis.text.chunker import chunk_text

logger = logging.getLogger("vocalis")

_TITLE = "Vocalis — Quick Reader"

# How long "Done" / a failure message waits for the copied-text notification
# (sent from a side thread) so the notifications still arrive in order.
_TEXT_NOTIFY_JOIN_SECONDS = 6.0


class QuickReaderFeature:
    """Wires Ctrl+Alt+R to read -> speak (with fallback) -> play."""

    def __init__(
        self,
        model_manager: ModelManager,
        primary_engine_factory: Callable[[], ModelEngine],
        fallback_engine: Synthesizer,
        clipboard_reader: ClipboardReader,
        player: AudioPlayer,
        notifier: Notifier,
        language: str = "English",
        reading_coordinator: ReadingCoordinator | None = None,
        on_reading_changed: Callable[[bool], None] | None = None,
        speed_provider: Callable[[], float] | None = None,
        engine_selector: EngineSelector | None = None,
        engine_factories: Mapping[str, Callable[[], ModelEngine]] | None = None,
    ) -> None:
        """Create a QuickReaderFeature.

        Args:
            model_manager: Shared ModelManager enforcing single-residency
                (used only for the primary, GPU-backed engine).
            primary_engine_factory: Constructs a fresh primary engine
                (e.g. Qwen3TTS) when it needs to be (re)loaded. The
                constructed engine must also implement Synthesizer.
            fallback_engine: An already-loaded, CPU-only engine (e.g.
                PiperTTS) used when the primary engine fails. Never
                routed through ModelManager since it never touches the GPU.
            clipboard_reader: Reads the text to speak.
            player: Plays the generated speech and can be interrupted.
            notifier: Sends start/live/done/stopped notifications.
            language: Spoken language. Currently only stored for future
                use; it is not yet passed to the engine. The two readers
                can share one engine id only while that holds.
            reading_coordinator: Shared with Document Reader so only one
                feature plays audio at a time. None (the default) means
                this feature never coordinates with another — used by
                existing tests that predate Document Reader.
            on_reading_changed: Called with True when a reading starts and
                False when it ends (stopped or finished). A callback that
                raises is logged and ignored. None means no hook.
            speed_provider: Called (with no arguments) once per chunk,
                right after that chunk is synthesized, to get the current
                speaking speed to apply before playback. None (the
                default) means no pacing is applied. A failure reading
                the speed or applying it never breaks the reading -- see
                vocalis.audio.pace.paced_audio.
            engine_selector: Chooses which voice engines to try, in order,
                for each reading. None (the default) means the fixed
                built-in Qwen3-TTS voice, then the backup voice.
            engine_factories: Maps an engine id to a function that builds
                that engine; used together with engine_selector.
        """
        self._model_manager = model_manager
        self._primary_engine_factory = primary_engine_factory
        self._fallback_engine = fallback_engine
        self._clipboard_reader = clipboard_reader
        self._player = player
        self._notifier = notifier
        self._language = language
        self._reading_coordinator = reading_coordinator
        self._on_reading_changed = on_reading_changed
        self._speed_provider = speed_provider
        self._engine_selector = engine_selector
        self._engine_factories = engine_factories
        self._report_lock = threading.Lock()
        self._last_reported = False
        self._lock = threading.Lock()
        self._active = False
        self._speaker: ChunkedSpeaker | None = None
        # Generation counter guards against a stale background thread from
        # an already-stopped reading clobbering a brand-new reading that
        # started (via a fresh on_trigger()) before that stale thread's
        # deactivate() call gets scheduled. Same pattern as ModelManager's
        # idle-timer generation guard, for the same class of race.
        self._generation = 0

    def on_trigger(self) -> None:
        """Call when Ctrl+Alt+R is pressed: start a new reading, or stop
        an in-progress one. Never blocks the caller (the hotkey
        listener's watcher thread): the state flip happens under a lock
        scoped to just that flip (see _flip_active_state()), and every
        potentially slow call — stopping the speaker, which can wait on the
        audio device, and Notifier.notify(), a notify-send subprocess that
        can take up to its own timeout — is dispatched onto a short-lived
        background thread rather than run inline.
        """
        interrupting, generation, speaker = self._flip_active_state()
        if interrupting:
            self._report_reading_state()
            threading.Thread(target=self._stop_and_notify, args=(speaker,), daemon=True).start()
            return
        if self._reading_coordinator is not None:
            self._reading_coordinator.start_reading(self, self._stop_for_coordinator)
        self._report_reading_state()
        threading.Thread(target=self._read_aloud, args=(generation,), daemon=True).start()

    def _stop_for_coordinator(self) -> None:
        """Stop this feature's in-progress reading because a different
        feature (via ReadingCoordinator) is claiming playback ownership.

        Clear-only: if this feature is idle (e.g. it was already stopped
        by a second press, leaving a stale coordinator claim) this is a
        harmless no-op — it must never flip an idle feature to active.
        Never calls back into the coordinator and never blocks: it only
        clears the flag under the lock, then spawns the same background
        stop-and-notify thread on_trigger() uses, off the lock.
        """
        with self._lock:
            if not self._active:
                return
            self._active = False
            speaker, self._speaker = self._speaker, None
        self._report_reading_state()
        threading.Thread(target=self._stop_and_notify, args=(speaker,), daemon=True).start()

    def stop(self) -> None:
        """Stop the reading in progress, if any (used by the floating widget and the tray)."""
        self._stop_for_coordinator()

    def _report_reading_state(self) -> None:
        """Tell the optional hook the CURRENT active state, but only if it
        differs from what was last reported. Reporting from the state
        (not from the event that triggered the call) means a late or
        stale report is simply a no-op, so the hook's view always
        converges on the real state. A failing hook is logged and never
        breaks reading. The report lock is separate from the state lock
        and nothing slow runs under it (hooks only emit signals)."""
        if self._on_reading_changed is None:
            return
        with self._report_lock:
            with self._lock:
                active = self._active
            if active == self._last_reported:
                return
            self._last_reported = active
            try:
                self._on_reading_changed(active)
            except Exception:
                logger.exception("on_reading_changed hook failed")

    def _flip_active_state(self) -> "tuple[bool, int, ChunkedSpeaker | None]":
        """Atomically flip self._active under the lock and report what
        the caller should do next, without making any blocking calls
        while the lock is held.

        Returns:
            (True, 0, speaker) if this call is interrupting an already-active
            reading (the caller should stop `speaker` — None if the sound had
            not started yet — and notify, off the lock); (False, generation,
            None) if this call is starting a fresh reading (the caller should
            spawn a worker thread scoped to `generation`).
        """
        with self._lock:
            if self._active:
                self._active = False
                speaker, self._speaker = self._speaker, None
                return True, 0, speaker
            self._active = True
            self._generation += 1
            return False, self._generation, None

    def _stop_and_notify(self, speaker: "ChunkedSpeaker | None") -> None:
        """Stop the sound and tell the user, off the caller's thread so
        on_trigger() itself never blocks on either call. A speaker that does
        not exist yet needs no stopping: the worker sees the cleared flag."""
        if speaker is not None:
            speaker.stop()
        self._notifier.notify(_TITLE, "Stopped")

    def _is_current(self, generation: int) -> bool:
        """Whether `generation` is still the active reading — i.e. it
        hasn't been superseded by a stop or by a newer reading starting.
        Used by the worker thread to decide whether it may still proceed
        (or must abandon its work) at each stage."""
        with self._lock:
            return self._active and self._generation == generation

    def _deactivate(self, generation: int) -> bool:
        """Mark `generation` as finished, but only if it is still the
        current generation — a stale worker thread whose reading was
        already superseded by a newer one must not clobber that newer
        reading's active state. Also releases ReadingCoordinator
        ownership, if one is configured and no newer reading is active;
        ReadingCoordinator.finish_reading()
        already no-ops safely if ownership moved on to a different
        feature in the meantime.

        Returns:
            True only if this call itself ended the active reading (it was the
            current generation and still active); False if a stop or a newer
            reading got there first."""
        with self._lock:
            is_current = self._generation == generation
            ended_here = is_current and self._active
            if is_current:
                self._active = False
            # A stale generation may release the claim only if no newer
            # reading is active; otherwise it would clear that reading's claim.
            may_release = is_current or not self._active
            # Released under the feature lock so a newer reading cannot
            # claim the coordinator in between. Safe: the coordinator
            # never calls a stopper under its own lock.
            if may_release and self._reading_coordinator is not None:
                self._reading_coordinator.finish_reading(self)
        if is_current:
            # State-based: a no-op if a stop already reported False, or if
            # a newer reading is active by now.
            self._report_reading_state()
        return ended_here

    def _read_aloud(self, generation: int) -> None:
        """Do the real work for one reading, entirely on a background
        thread: read the clipboard, split it into sentence chunks, and speak
        them gaplessly through a ChunkedSpeaker — bailing out cleanly at each
        stage if `on_trigger()` has cancelled this generation in the meantime.

        The copied text is announced from a one-off side thread so a slow
        notify-send can never delay the sound; "Done" (or a failure message)
        waits briefly for that thread so the notifications still arrive in
        the same order.
        """
        text = self._clipboard_reader.read()
        if not self._is_current(generation):
            return  # cancelled while reading the clipboard
        if not text.strip():
            if self._deactivate(generation):
                self._notifier.notify(_TITLE, "Nothing copied")
            return

        self._notifier.notify(_TITLE, "Reading...")
        speaker = ChunkedSpeaker(
            player=self._player,
            source_factory=self._new_source,
            speed_provider=self._speed_provider,
        )
        with self._lock:
            # Published only if still current, under the same lock stop uses to
            # take the speaker: a stop can never fall between "no speaker yet"
            # and "speaker running", and a stale worker never leaves one behind.
            if not (self._active and self._generation == generation):
                return  # cancelled before the sound could start
            self._speaker = speaker
        text_notice = threading.Thread(target=self._notifier.notify, args=(_TITLE, text), daemon=True)
        text_notice.start()
        outcome = speaker.run(chunk_text(text))
        with self._lock:
            if self._speaker is speaker:
                self._speaker = None
        self._finish_reading(generation, outcome, text_notice)

    def _finish_reading(self, generation: int, outcome: SpeakOutcome, text_notice: threading.Thread) -> None:
        """End the reading and send the closing notification for `outcome`.

        The end is reported first (so a slow notification can never delay the
        hook); then the copied-text notification is given a moment to land so
        the closing message follows it. A stopped reading was already ended
        and announced by the stop, so it waits for nothing.
        """
        if not self._deactivate(generation) or outcome is SpeakOutcome.STOPPED:
            return
        text_notice.join(timeout=_TEXT_NOTIFY_JOIN_SECONDS)
        if outcome is SpeakOutcome.SPEECH_FAILED:
            self._notifier.notify(_TITLE, "Could not generate speech")
        elif outcome is SpeakOutcome.PLAYBACK_FAILED:
            self._notifier.notify(_TITLE, "Playback failed")
        else:
            self._notifier.notify(_TITLE, "Done")

    def _new_source(self) -> EngineChain:
        """Build the voice chain for one reading; it holds the voice loaded until closed."""
        return new_reading_chain(
            self._engine_selector,
            self._engine_factories,
            self._primary_engine_factory,
            self._model_manager,
            self._fallback_engine,
            make_downshift_handler(lambda message: self._notifier.notify(_TITLE, message)),
            hold_engines=True,
        )
