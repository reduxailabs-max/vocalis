"""Orchestrates Quick Reader: on Ctrl+Alt+R, read the clipboard, speak it
via Qwen3-TTS (falling back to Piper on any failure), and play it back —
with desktop notifications at each stage, and press-again-to-stop.

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
from vocalis.output.clipboard import ClipboardReader

logger = logging.getLogger("vocalis")

_TITLE = "Vocalis — Quick Reader"


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
            speed_provider: Called (with no arguments) once per reading,
                right after the text is synthesized, to get the current
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
        potentially slow call — AudioPlayer.stop(), which can wait on the
        audio callback thread, and Notifier.notify(), a notify-send
        subprocess that can take up to its own timeout — is dispatched
        onto a short-lived background thread rather than run inline.
        """
        interrupting, generation = self._flip_active_state()
        if interrupting:
            self._report_reading_state()
            threading.Thread(target=self._stop_and_notify, daemon=True).start()
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
        self._report_reading_state()
        threading.Thread(target=self._stop_and_notify, daemon=True).start()

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

    def _flip_active_state(self) -> "tuple[bool, int]":
        """Atomically flip self._active under the lock and report what
        the caller should do next, without making any blocking calls
        while the lock is held.

        Returns:
            (True, 0) if this call is interrupting an already-active
            reading (the caller should stop playback and notify, off the
            lock); (False, generation) if this call is starting a fresh
            reading (the caller should spawn a worker thread scoped to
            `generation`).
        """
        with self._lock:
            if self._active:
                self._active = False
                return True, 0
            self._active = True
            self._generation += 1
            return False, self._generation

    def _stop_and_notify(self) -> None:
        """Stop playback and tell the user, off the caller's thread so
        on_trigger() itself never blocks on either call."""
        self._player.stop()
        self._notifier.notify(_TITLE, "Stopped")

    def _is_current(self, generation: int) -> bool:
        """Whether `generation` is still the active reading — i.e. it
        hasn't been superseded by a stop or by a newer reading starting.
        Used by the worker thread to decide whether it may still proceed
        (or must abandon its work) at each stage."""
        with self._lock:
            return self._active and self._generation == generation

    def _deactivate(self, generation: int) -> None:
        """Mark `generation` as finished, but only if it is still the
        current generation — a stale worker thread whose reading was
        already superseded by a newer one must not clobber that newer
        reading's active state. Also releases ReadingCoordinator
        ownership, if one is configured and no newer reading is active;
        ReadingCoordinator.finish_reading()
        already no-ops safely if ownership moved on to a different
        feature in the meantime."""
        with self._lock:
            is_current = self._generation == generation
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

    def _read_aloud(self, generation: int) -> None:
        """Do the real work for one reading, entirely on a background
        thread: read the clipboard, synthesize speech (with fallback),
        and play it back — bailing out cleanly at each stage if `on_trigger()`
        has cancelled this generation in the meantime.
        """
        text = self._clipboard_reader.read()
        if not self._is_current(generation):
            return  # cancelled while reading the clipboard
        if not text.strip():
            self._notifier.notify(_TITLE, "Nothing copied")
            self._deactivate(generation)
            return

        self._notifier.notify(_TITLE, "Reading...")
        try:
            audio, sample_rate = self._synthesize_with_fallback(text)
        except Exception:
            logger.exception("Both TTS engines failed")
            self._notifier.notify(_TITLE, "Could not generate speech")
            self._deactivate(generation)
            return
        if not self._is_current(generation):
            return  # cancelled while synthesizing; must not start playback
        audio = paced_audio(audio, sample_rate, self._speed_provider)

        # notify() below is a real subprocess call that can take a
        # noticeable amount of time (up to its own timeout) — a press
        # can land and cancel this generation while it runs. play() must
        # still be given a chance to start (there is no way to abort a
        # synthesize() already in flight and text is already spoken-for),
        # but immediately after starting it we re-check and stop it
        # right away rather than let it run to completion.
        self._notifier.notify(_TITLE, text)
        try:
            self._player.play(audio, sample_rate)
            if not self._is_current(generation):
                self._player.stop()
            finished_naturally = self._player.wait_until_done()
        except Exception:
            logger.exception("Playback failed")
            self._notifier.notify(_TITLE, "Playback failed")
            self._deactivate(generation)
            return

        self._deactivate(generation)
        if finished_naturally:
            self._notifier.notify(_TITLE, "Done")

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

    def _synthesize_with_fallback(self, text: str) -> "tuple[npt.NDArray[np.float32], int]":
        """Speak `text` with the planned engines, falling down the chain on failure."""
        return self._new_chain().synthesize(text)
