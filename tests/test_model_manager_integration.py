"""Integration test proving ModelManager's real (non-fake) timer actually
unloads an idle engine after real wall-clock time passes."""
import time

from vocalis.model_manager import ModelManager


class _RecordingEngine:
    def __init__(self) -> None:
        self.unloaded = False
        self._loaded = False

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self.unloaded = True
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded


def test_real_timer_unloads_engine_after_idle_timeout() -> None:
    manager = ModelManager(idle_timeout_seconds=0.05)
    engine = _RecordingEngine()
    manager.acquire("parakeet", lambda: engine)
    manager.release("parakeet")

    assert engine.unloaded is False

    time.sleep(0.2)

    assert engine.unloaded is True
    assert manager.current_engine_id is None
