"""Walks one reading's list of voice engines, falling down it on failure."""
import logging
from typing import Callable, Mapping, cast

import numpy as np
import numpy.typing as npt

from vocalis.engines.auto_select import PIPER
from vocalis.engines.base import Synthesizer, TTSEngine
from vocalis.engines.registry import EnginePlan
from vocalis.model_manager import ModelEngine, ModelManager

logger = logging.getLogger("vocalis")


class EngineChain:
    """Speaks text with the first working engine of a plan.

    Remembers which engines already failed, so later pieces of the same
    reading never retry a dead engine and the user is told only once.
    """

    def __init__(
        self,
        plan: EnginePlan,
        factories: Mapping[str, Callable[[], ModelEngine]],
        model_manager: ModelManager,
        fallback_engine: Synthesizer,
        on_downshift: Callable[[str, "str | None"], None],
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
        """
        self._plan = plan
        self._factories = factories
        self._model_manager = model_manager
        self._fallback_engine = fallback_engine
        self._on_downshift = on_downshift
        self._position = 0
        self._last_error: Exception | None = None

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
                self._position = index + 1
                self._on_downshift(engine_id, chain[index + 1] if index + 1 < len(chain) else None)
        if PIPER not in chain:
            return self._fallback_engine.synthesize(text)
        raise self._last_error if self._last_error is not None else RuntimeError("No voice engine is available")

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
            self._model_manager.release(engine_id)
