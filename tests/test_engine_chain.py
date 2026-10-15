"""Tests for vocalis.engines.chain.EngineChain — walking the engine list for one reading."""
import numpy as np
import numpy.typing as npt
import pytest

from vocalis.engines.chain import EngineChain
from vocalis.engines.registry import EnginePlan
from vocalis.model_manager import ModelManager


class _Tts:
    """A fake managed engine; its audio is filled with `value` so tests can tell engines apart."""

    def __init__(self, value: float, fail_load: bool = False, fail_synth: bool = False) -> None:
        """Remember the audio value and which steps should fail."""
        self.value, self.fail_load, self.fail_synth = value, fail_load, fail_synth
        self.loaded = False

    def load(self) -> None:
        """Pretend to load, or fail when asked to."""
        if self.fail_load:
            raise RuntimeError("load failed")
        self.loaded = True

    def unload(self) -> None:
        """Pretend to unload."""
        self.loaded = False

    @property
    def is_loaded(self) -> bool:
        """Whether the fake is loaded."""
        return self.loaded

    def synthesize(self, text: str) -> "tuple[npt.NDArray[np.float32], int]":
        """Return filler audio, or fail when asked to."""
        if self.fail_synth:
            raise RuntimeError("synth failed")
        return np.full(4, self.value, dtype=np.float32), 24000


class _Piper:
    """A fake always-ready fallback engine."""

    def __init__(self, fail: bool = False) -> None:
        """Remember whether to fail and count calls."""
        self.fail, self.calls = fail, 0

    def synthesize(self, text: str) -> "tuple[npt.NDArray[np.float32], int]":
        """Return filler audio (value 3.0), or fail when asked to."""
        self.calls += 1
        if self.fail:
            raise RuntimeError("piper failed")
        return np.full(4, 3.0, dtype=np.float32), 22050


def _chain(chain: tuple[str, ...], qwen: _Tts, kokoro: _Tts, piper: _Piper, events: "list[tuple[str, str | None]]") -> EngineChain:
    """Build an EngineChain over the fakes, recording downshift events."""
    factories = {"qwen3_tts": lambda: qwen, "kokoro": lambda: kokoro}
    return EngineChain(EnginePlan(chain, "test"), factories, ModelManager(idle_timeout_seconds=9999.0), piper,
                       lambda failed, nxt: events.append((failed, nxt)))


def test_the_first_engine_is_used_when_it_works() -> None:
    """No downshift happens when the first engine succeeds."""
    events: list[tuple[str, str | None]] = []
    chain = _chain(("qwen3_tts", "kokoro", "piper"), _Tts(1.0), _Tts(2.0), _Piper(), events)
    audio, rate = chain.synthesize("hi")
    assert audio[0] == 1.0 and rate == 24000 and events == []


def test_a_failing_load_falls_to_the_next_engine_and_is_remembered() -> None:
    """After a failure the dead engine is skipped for the rest of the reading."""
    events: list[tuple[str, str | None]] = []
    qwen, kokoro = _Tts(1.0, fail_load=True), _Tts(2.0)
    chain = _chain(("qwen3_tts", "kokoro", "piper"), qwen, kokoro, _Piper(), events)
    assert chain.synthesize("a")[0][0] == 2.0
    assert chain.synthesize("b")[0][0] == 2.0
    assert events == [("qwen3_tts", "kokoro")]


def test_a_failing_synthesis_also_falls_down_the_chain() -> None:
    """A synthesis error is handled like a load error."""
    events: list[tuple[str, str | None]] = []
    chain = _chain(("qwen3_tts", "kokoro", "piper"), _Tts(1.0, fail_synth=True), _Tts(2.0), _Piper(), events)
    assert chain.synthesize("a")[0][0] == 2.0
    assert events == [("qwen3_tts", "kokoro")]


def test_when_every_managed_engine_fails_piper_speaks() -> None:
    """Piper is the last resort."""
    events: list[tuple[str, str | None]] = []
    piper = _Piper()
    chain = _chain(("qwen3_tts", "kokoro", "piper"), _Tts(1.0, fail_load=True), _Tts(2.0, fail_load=True), piper, events)
    assert chain.synthesize("a")[0][0] == 3.0
    assert events == [("qwen3_tts", "kokoro"), ("kokoro", "piper")]
    assert piper.calls == 1


def test_a_piper_first_plan_never_touches_the_model_manager() -> None:
    """Choosing Piper (no graphics card) loads no managed model at all."""
    manager = ModelManager(idle_timeout_seconds=9999.0)
    piper = _Piper()
    chain = EngineChain(EnginePlan(("piper", "kokoro"), "t"), {"kokoro": lambda: _Tts(2.0)}, manager, piper, lambda *_: None)
    assert chain.synthesize("a")[0][0] == 3.0
    assert manager.current_engine_id is None


def test_a_forced_engine_with_no_factory_is_skipped_not_fatal() -> None:
    """A missing engine (e.g. no model installed) falls down the chain."""
    events: list[tuple[str, str | None]] = []
    chain = EngineChain(EnginePlan(("kokoro", "piper"), "t"), {}, ModelManager(idle_timeout_seconds=9999.0),
                        _Piper(), lambda failed, nxt: events.append((failed, nxt)))
    assert chain.synthesize("a")[0][0] == 3.0
    assert events == [("kokoro", "piper")]


def test_the_old_single_engine_plan_still_ends_on_piper() -> None:
    """A chain with only Qwen3-TTS falls back to Piper and reports the downshift (legacy behaviour)."""
    events: list[tuple[str, str | None]] = []
    chain = _chain(("qwen3_tts",), _Tts(1.0, fail_load=True), _Tts(2.0), _Piper(), events)
    assert chain.synthesize("a")[0][0] == 3.0
    assert events == [("qwen3_tts", None)]
    assert chain.synthesize("b")[0][0] == 3.0
    assert events == [("qwen3_tts", None)]


def test_when_everything_including_piper_fails_the_error_surfaces() -> None:
    """If nothing can speak, the caller gets an exception to turn into a notification."""
    chain = _chain(("qwen3_tts", "piper"), _Tts(1.0, fail_load=True), _Tts(2.0), _Piper(fail=True), [])
    with pytest.raises(RuntimeError, match="piper failed"):
        chain.synthesize("a")
