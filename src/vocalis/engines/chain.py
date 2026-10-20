"""Walks one reading's list of voice engines, falling down it on failure."""
import logging
from typing import Callable, Mapping, cast

import numpy as np
import numpy.typing as npt

from vocalis.engines.auto_select import PIPER
from vocalis.engines.base import Synthesizer, TTSEngine
from vocalis.engines.registry import EnginePlan, EngineSelector
from vocalis.model_manager import TTS_ENGINE_ID, ModelEngine, ModelManager

logger = logging.getLogger("vocalis")


class EngineChain:
    """Speaks text with the first working engine of a plan.

    Remembers which engines already failed, so later pieces of the same
    reading never retry a dead engine and the user is told only once.

    With `hold_engines=True` (used for a whole reading) the engine that spoke
    is NOT released between pieces: it stays loaded, the idle-unload timer
    stays off, and `close()` releases it when the reading has finished
    playing. The engine is still re-acquired before every piece, so if
    another feature took the graphics card in the meantime it is simply
    reloaded. Engine factories must build a NEW engine object every time they
    are called (the chain tells a reloaded engine from the still-loaded one by
    object identity).
    """

    def __init__(
        self,
        plan: EnginePlan,
        factories: Mapping[str, Callable[[], ModelEngine]],
        model_manager: ModelManager,
        fallback_engine: Synthesizer,
        on_downshift: Callable[[str, "str | None"], None],
        hold_engines: bool = False,
    ) -> None:
        """Create a chain for one reading.

        Args:
            plan: The engines to try, in order.
            factories: Builds each managed engine (qwen3_tts, kokoro) by id.
            model_manager: Loads/unloads managed engines (one at a time).
            fallback_engine: The always-ready Piper engine, used for the
                "piper" step and as the very last resort.
            on_downshift: Called with (failed engine id, next id or None)
                each time an engine fails and the chain moves on.
            hold_engines: Keep the speaking engine loaded until close().
        """
        self._plan = plan
        self._factories = factories
        self._model_manager = model_manager
        self._fallback_engine = fallback_engine
        self._on_downshift = on_downshift
        self._position = 0
        self._last_error: Exception | None = None
        self._hold_engines = hold_engines
        self._held: str | None = None
        self._held_engine: ModelEngine | None = None

    def synthesize(self, text: str) -> "tuple[npt.NDArray[np.float32], int]":
        """Speak `text` with the first engine that works.

        Raises:
            Exception: The last engine's error if nothing could speak.
        """
        chain = self._plan.chain
        for index in range(self._position, len(chain)):
            engine_id = chain[index]
            try:
                return self._synthesize_with(engine_id, text)
            except Exception as exc:
                logger.exception("Voice engine %r failed", engine_id)
                self._last_error = exc
                self._release_failed_hold(engine_id)
                self._position = index + 1
                self._on_downshift(engine_id, chain[index + 1] if index + 1 < len(chain) else None)
        if PIPER not in chain:
            return self._fallback_engine.synthesize(text)
        raise self._last_error if self._last_error is not None else RuntimeError("No voice engine is available")

    def _release_failed_hold(self, engine_id: str) -> None:
        """Give back the hold on an engine that just failed, so it can idle-unload."""
        if not (self._hold_engines and self._held == engine_id):
            return
        held_engine = self._held_engine
        self._held = None
        self._held_engine = None
        if self._model_manager.current_engine is held_engine:
            self._model_manager.release(engine_id)

    def _synthesize_with(self, engine_id: str, text: str) -> "tuple[npt.NDArray[np.float32], int]":
        """Speak with exactly one engine (Piper directly, others through ModelManager)."""
        if engine_id == PIPER:
            return self._fallback_engine.synthesize(text)
        factory = self._factories.get(engine_id)
        if factory is None:
            raise LookupError(f"No voice engine is set up for {engine_id!r}")
        engine = self._model_manager.acquire(engine_id, factory)
        try:
            return cast(TTSEngine, engine).synthesize(text)
        finally:
            self._after_use(engine_id, engine)

    def _after_use(self, engine_id: str, engine: ModelEngine) -> None:
        """Release the engine now, or remember it as held until close()."""
        if not self._hold_engines:
            self._model_manager.release(engine_id)
            return
        if self._held == engine_id and self._held_engine is engine:
            # Same loaded copy as last piece: ModelManager counted one more
            # user for this acquire, so give that extra one back. The single
            # count for the hold itself stays until close().
            self._model_manager.release(engine_id)
        # A newly acquired engine has already unloaded the one held before it.
        self._held = engine_id
        self._held_engine = engine

    def close(self) -> None:
        """Release the held engine, starting the idle-unload timer. Idempotent."""
        held, held_engine = self._held, self._held_engine
        self._held, self._held_engine = None, None
        if held is None:
            return
        if self._model_manager.current_engine is not held_engine:
            return  # our copy is gone (evicted or replaced); nothing of ours to release
        self._model_manager.release(held)


def new_reading_chain(
    engine_selector: EngineSelector | None,
    engine_factories: Mapping[str, Callable[[], ModelEngine]] | None,
    primary_engine_factory: Callable[[], ModelEngine],
    model_manager: ModelManager,
    fallback_engine: Synthesizer,
    on_downshift: Callable[[str, "str | None"], None],
    hold_engines: bool = False,
) -> EngineChain:
    """Build the engine chain for one reading.

    With a selector: its plan and factories. Without one: the old fixed
    Qwen3-TTS voice (then the backup voice). Shared by Quick Reader and
    Document Reader so the two never drift apart.
    """
    if engine_selector is not None:
        plan = engine_selector.plan()
        factories = engine_factories or {}
    else:
        plan = EnginePlan((TTS_ENGINE_ID,), "Fixed to the built-in voice.")
        factories = {TTS_ENGINE_ID: primary_engine_factory}
    return EngineChain(plan, factories, model_manager, fallback_engine, on_downshift, hold_engines=hold_engines)


def make_downshift_handler(
    notify: Callable[[str], None],
    on_status: Callable[[str], None] | None = None,
) -> Callable[[str, "str | None"], None]:
    """Make the on_downshift callback both readers use.

    Stays silent when the backup voice (Piper) itself failed — there is
    nothing further to fall back to. Otherwise tells the user the first-choice
    voice is unavailable via `notify` and, if given, `on_status`.
    """

    def handler(failed_engine_id: str, next_engine_id: "str | None") -> None:
        """Announce one engine failure."""
        if failed_engine_id == PIPER:
            return
        notify("Primary voice unavailable — using backup voice")
        if on_status is not None:
            on_status("Using backup voice")

    return handler
