"""Enforces vocalis's GPU policy: only one AI model resident in VRAM at a
time, automatically released a few seconds after it goes idle.

Feature code calls acquire() to get a loaded engine and release() when done
with it for now. The manager decides the actual load/unload timing: it
unloads the previous engine before loading a different one, and starts an
idle timer on release() that unloads the current engine if nothing
reacquires it before the timeout elapses.

A monotonic "generation" counter guards against a race between a firing
idle timer and a concurrent acquire(): threading.Timer.cancel() cannot stop
a callback that has already started running, so every acquire() bumps the
generation, and an idle-unload callback only acts if the generation it was
scheduled under is still current — otherwise something reacquired the
engine after the timer fired but before it got the lock, and unloading
would pull the rug out from under that caller.
"""
import logging
import threading
from typing import Callable, Protocol

logger = logging.getLogger("vocalis")

# One id for the Qwen3-TTS model, shared by Quick Reader and Document Reader
# so switching between them reuses the loaded model instead of reloading it.
TTS_ENGINE_ID = "qwen3_tts"


class ModelEngine(Protocol):
    """Interface every AI model wrapper (STT/TTS/OCR) must implement."""

    def load(self) -> None:
        """Load the model into memory/VRAM. Must be safe to call once per instance."""
        ...

    def unload(self) -> None:
        """Release the model's memory/VRAM. Must be safe to call after load()."""
        ...

    @property
    def is_loaded(self) -> bool:
        """Whether the model is currently loaded."""
        ...


class TimerHandle(Protocol):
    """A cancellable handle returned by a TimerFactory."""

    def cancel(self) -> None:
        """Cancel the pending callback if it has not fired yet."""
        ...


TimerFactory = Callable[[float, Callable[[], None]], TimerHandle]


class ModelManager:
    """Owns at most one loaded ModelEngine and enforces idle-unload."""

    def __init__(
        self,
        idle_timeout_seconds: float = 30.0,
        timer_factory: TimerFactory | None = None,
    ) -> None:
        """Create a ModelManager.

        Args:
            idle_timeout_seconds: Seconds of inactivity after release()
                before the current engine is unloaded.
            timer_factory: Callable(delay_seconds, callback) -> TimerHandle,
                used to schedule the idle-unload. Defaults to a real
                threading.Timer-backed factory (see _real_timer_factory).
        """
        self._idle_timeout_seconds = idle_timeout_seconds
        self._timer_factory = timer_factory or _real_timer_factory
        self._lock = threading.Lock()
        self._current_engine_id: str | None = None
        self._current_engine: ModelEngine | None = None
        self._idle_timer: TimerHandle | None = None
        self._generation = 0
        # How many callers are between acquire() and release() for the
        # currently loaded engine. Reset whenever the engine changes.
        self._use_count = 0

    @property
    def current_engine_id(self) -> str | None:
        """The engine_id currently loaded, or None if nothing is loaded."""
        with self._lock:
            return self._current_engine_id

    @property
    def current_engine(self) -> ModelEngine | None:
        """The loaded engine object itself, or None if nothing is loaded."""
        with self._lock:
            return self._current_engine

    def acquire(self, engine_id: str, factory: Callable[[], ModelEngine]) -> ModelEngine:
        """Get a loaded engine for engine_id, loading or reusing as needed.

        Cancels any pending idle-unload and advances the generation counter
        (invalidating any idle-timer callback already in flight from a
        previous release()). If a different engine is currently loaded, it
        is unloaded first so only one model ever sits in VRAM — even if that
        other engine's use count is still above zero (deliberate, conservative
        choice that keeps the original single-residency rule; callers of the
        evicted engine may later call release(); that is a known, low-risk
        limitation: release is counted per engine id, not per loaded copy,
        so if another reader has reloaded the same id in between, the stale
        release lowers the new load's count and can arm the idle-unload
        timer. The generation guard means a re-acquire invalidates that
        timer, so the worst case is one unnecessary unload and reload).

        Each successful acquire() adds one to the engine's use count; the
        matching release() removes it.

        Args:
            engine_id: Stable identifier for the requested engine, e.g.
                "parakeet" or "qwen3_tts".
            factory: Called to construct a new engine instance if one isn't
                already loaded for engine_id.

        Returns:
            A loaded ModelEngine for engine_id.

        Raises:
            Exception: Whatever factory() or engine.load() raises, after
                first ensuring the manager's own state reflects that
                nothing is loaded (the failed engine is never left as the
                recorded "current" engine).
        """
        with self._lock:
            self._cancel_idle_timer_locked()
            self._generation += 1

            if self._current_engine_id == engine_id and self._current_engine is not None:
                logger.info("Reusing already-loaded engine %r", engine_id)
                self._use_count += 1
                return self._current_engine

            if self._current_engine is not None:
                logger.info(
                    "Unloading engine %r to load %r", self._current_engine_id, engine_id
                )
                self._unload_current_locked()

            logger.info("Loading engine %r", engine_id)
            try:
                engine = factory()
                engine.load()
            except Exception:
                logger.exception("Failed to load engine %r", engine_id)
                raise

            self._current_engine_id = engine_id
            self._current_engine = engine
            self._use_count = 1
            return engine

    def release(self, engine_id: str) -> None:
        """Mark engine_id as no longer actively in use.

        Decrements the engine's use count. The idle-unload timer is only
        started when the count reaches zero, so one reader finishing does not
        arm an unload while another reader (sharing the same engine id) is
        still inside the engine. An extra release() at count zero logs a
        warning, keeps the count at zero and still (re)arms the timer.
        Any pending timer is cancelled first.
        If engine_id does not match the currently loaded engine (or nothing
        is loaded), logs a warning and does nothing else — this indicates a
        bug in the caller, not a state we should act on destructively.

        Args:
            engine_id: The engine_id previously returned by acquire().
        """
        with self._lock:
            if self._current_engine_id != engine_id:
                logger.warning(
                    "release() called for %r but currently loaded engine is %r",
                    engine_id,
                    self._current_engine_id,
                )
                return

            if self._use_count > 0:
                self._use_count -= 1
            else:
                logger.warning(
                    "Engine %r released more times than acquired", engine_id
                )
            if self._use_count > 0:
                logger.info(
                    "Engine %r released; %d other user(s) still active",
                    engine_id,
                    self._use_count,
                )
                return

            self._cancel_idle_timer_locked()

            logger.info(
                "Engine %r released; will unload after %.1fs if not reacquired",
                engine_id,
                self._idle_timeout_seconds,
            )
            generation_at_release = self._generation
            self._idle_timer = self._timer_factory(
                self._idle_timeout_seconds,
                lambda: self._unload_if_current_generation(generation_at_release),
            )

    def shutdown(self) -> None:
        """Cancel any pending idle timer and unload the current engine now.

        Called when the daemon is stopping. Unconditional: unlike the
        idle-timer callback, shutdown always unloads whatever is currently
        loaded, regardless of generation.
        """
        with self._lock:
            self._cancel_idle_timer_locked()
            self._unload_current_locked()

    def _unload_if_current_generation(self, generation: int) -> None:
        """Idle-timer callback: unload only if nothing reacquired since release().

        Guards against the race where this callback's thread was already
        past threading.Timer's cancellation window when a concurrent
        acquire() reused the engine and bumped the generation counter.

        Args:
            generation: The generation captured when this timer was scheduled.
        """
        with self._lock:
            if self._generation != generation:
                logger.info(
                    "Idle-unload skipped: engine was reacquired since release()"
                )
                return
            self._unload_current_locked()

    def _unload_current_locked(self) -> None:
        """Unload the current engine. Caller must hold self._lock.

        Guarantees bookkeeping is reset to "nothing loaded" even if the
        engine's own unload() raises (e.g. a CUDA teardown error) — losing
        track of a resident model would be worse than losing the error.
        """
        try:
            if self._current_engine is not None:
                logger.info("Unloading idle engine %r", self._current_engine_id)
                self._current_engine.unload()
        except Exception:
            logger.exception("Error unloading engine %r", self._current_engine_id)
        finally:
            self._current_engine = None
            self._current_engine_id = None
            self._use_count = 0

    def _cancel_idle_timer_locked(self) -> None:
        """Cancel the pending idle timer, if any. Caller must hold self._lock."""
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None


def _real_timer_factory(delay_seconds: float, callback: Callable[[], None]) -> TimerHandle:
    """Default TimerFactory backed by threading.Timer (daemon thread)."""
    timer = threading.Timer(delay_seconds, callback)
    timer.daemon = True
    timer.start()
    return timer
