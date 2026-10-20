"""Tests for the ModelManager single-residency + idle-unload policy.

Uses FakeModelEngine and FakeTimerFactory so the idle-timeout logic is
proven deterministically, with no real GPU model and no real sleeping.
"""
import logging

import pytest

from vocalis.model_manager import ModelManager


class FakeModelEngine:
    def __init__(self, name: str) -> None:
        self.name = name
        self.load_calls = 0
        self.unload_calls = 0
        self._loaded = False

    def load(self) -> None:
        self.load_calls += 1
        self._loaded = True

    def unload(self) -> None:
        self.unload_calls += 1
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded


class FakeTimerHandle:
    def __init__(self, callback):  # type: ignore[no-untyped-def]
        self.callback = callback
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True

    def fire(self) -> None:
        if not self.cancelled:
            self.callback()


class FakeTimerFactory:
    def __init__(self) -> None:
        self.handles: list[FakeTimerHandle] = []
        self.last_delay: float | None = None

    def __call__(self, delay, callback):  # type: ignore[no-untyped-def]
        self.last_delay = delay
        handle = FakeTimerHandle(callback)  # type: ignore[no-untyped-call]
        self.handles.append(handle)
        return handle


def make_manager(idle_timeout: float = 30.0) -> tuple[ModelManager, FakeTimerFactory]:
    fake_timers = FakeTimerFactory()
    manager = ModelManager(idle_timeout_seconds=idle_timeout, timer_factory=fake_timers)
    return manager, fake_timers


def test_acquire_loads_model() -> None:
    manager, _ = make_manager()
    engine = FakeModelEngine("parakeet")
    result = manager.acquire("parakeet", lambda: engine)
    assert result is engine
    assert engine.load_calls == 1
    assert manager.current_engine_id == "parakeet"


def test_acquire_same_engine_reuses_without_reloading() -> None:
    manager, _ = make_manager()
    engine = FakeModelEngine("parakeet")
    manager.acquire("parakeet", lambda: engine)
    manager.acquire("parakeet", lambda: engine)
    assert engine.load_calls == 1


def test_acquire_different_engine_unloads_previous_first() -> None:
    manager, _ = make_manager()
    parakeet = FakeModelEngine("parakeet")
    qwen = FakeModelEngine("qwen3_tts")
    manager.acquire("parakeet", lambda: parakeet)
    manager.acquire("qwen3_tts", lambda: qwen)
    assert parakeet.unload_calls == 1
    assert qwen.load_calls == 1
    assert manager.current_engine_id == "qwen3_tts"


def test_release_starts_idle_timer_with_configured_timeout() -> None:
    manager, fake_timers = make_manager(idle_timeout=45.0)
    engine = FakeModelEngine("parakeet")
    manager.acquire("parakeet", lambda: engine)
    manager.release("parakeet")
    assert fake_timers.last_delay == 45.0
    assert len(fake_timers.handles) == 1


def test_idle_timer_firing_unloads_the_model() -> None:
    manager, fake_timers = make_manager()
    engine = FakeModelEngine("parakeet")
    manager.acquire("parakeet", lambda: engine)
    manager.release("parakeet")
    fake_timers.handles[0].fire()
    assert engine.unload_calls == 1
    assert manager.current_engine_id is None


def test_reacquire_before_idle_fires_cancels_timer_and_reuses_model() -> None:
    manager, fake_timers = make_manager()
    engine = FakeModelEngine("parakeet")
    manager.acquire("parakeet", lambda: engine)
    manager.release("parakeet")
    manager.acquire("parakeet", lambda: engine)
    assert fake_timers.handles[0].cancelled is True
    assert engine.unload_calls == 0
    assert engine.load_calls == 1


def test_release_with_no_model_loaded_logs_warning_and_starts_no_timer(
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager, fake_timers = make_manager()
    with caplog.at_level(logging.WARNING, logger="vocalis"):
        manager.release("parakeet")
    assert len(fake_timers.handles) == 0
    assert any("parakeet" in record.message for record in caplog.records)


def test_release_wrong_engine_id_logs_warning_and_starts_no_timer(
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager, fake_timers = make_manager()
    engine = FakeModelEngine("parakeet")
    manager.acquire("parakeet", lambda: engine)
    with caplog.at_level(logging.WARNING, logger="vocalis"):
        manager.release("qwen3_tts")
    assert len(fake_timers.handles) == 0
    assert engine.unload_calls == 0


def test_shutdown_unloads_immediately_and_cancels_pending_timer() -> None:
    manager, fake_timers = make_manager()
    engine = FakeModelEngine("parakeet")
    manager.acquire("parakeet", lambda: engine)
    manager.release("parakeet")
    manager.shutdown()
    assert fake_timers.handles[0].cancelled is True
    assert engine.unload_calls == 1
    assert manager.current_engine_id is None


def test_idle_timer_from_before_reacquire_does_not_unload_reused_engine() -> None:
    """Simulates the TOCTOU race: a stale timer fires after a reacquire."""
    manager, fake_timers = make_manager()
    engine = FakeModelEngine("parakeet")
    manager.acquire("parakeet", lambda: engine)
    manager.release("parakeet")
    stale_timer = fake_timers.handles[0]

    # Reacquire the same engine before the stale timer is told to fire.
    # In the real race, threading.Timer.cancel() would fail to stop a
    # callback thread already past its cancellation check; here we
    # simulate that by firing the stale handle even though acquire()
    # "cancelled" it, proving the generation guard (not cancellation
    # alone) is what protects the engine.
    manager.acquire("parakeet", lambda: engine)
    stale_timer.callback()

    assert engine.unload_calls == 0
    assert manager.current_engine_id == "parakeet"


def test_double_release_without_reacquire_cancels_first_timer() -> None:
    manager, fake_timers = make_manager()
    engine = FakeModelEngine("parakeet")
    manager.acquire("parakeet", lambda: engine)
    manager.release("parakeet")
    manager.release("parakeet")

    assert len(fake_timers.handles) == 2
    assert fake_timers.handles[0].cancelled is True
    assert fake_timers.handles[1].cancelled is False


def test_acquire_failure_after_switching_engines_resets_state_and_reraises() -> None:
    manager, _ = make_manager()
    good_engine = FakeModelEngine("parakeet")
    manager.acquire("parakeet", lambda: good_engine)

    class FailingEngine:
        def load(self) -> None:
            raise RuntimeError("boom")

        def unload(self) -> None:
            pass

        @property
        def is_loaded(self) -> bool:
            return False

    import pytest

    with pytest.raises(RuntimeError, match="boom"):
        manager.acquire("qwen3_tts", lambda: FailingEngine())

    assert good_engine.unload_calls == 1
    assert manager.current_engine_id is None


class RaisingUnloadEngine:
    def __init__(self) -> None:
        self.load_calls = 0
        self._loaded = False

    def load(self) -> None:
        self.load_calls += 1
        self._loaded = True

    def unload(self) -> None:
        raise RuntimeError("teardown failed")

    @property
    def is_loaded(self) -> bool:
        return self._loaded


def test_shutdown_survives_unload_raising_and_resets_state() -> None:
    manager, _ = make_manager()
    engine = RaisingUnloadEngine()
    manager.acquire("parakeet", lambda: engine)

    manager.shutdown()  # must not raise

    assert manager.current_engine_id is None


def test_acquire_switching_engines_survives_old_engine_unload_raising() -> None:
    manager, _ = make_manager()
    old_engine = RaisingUnloadEngine()
    manager.acquire("parakeet", lambda: old_engine)

    new_engine = FakeModelEngine("qwen3_tts")
    result = manager.acquire("qwen3_tts", lambda: new_engine)  # must not raise

    assert result is new_engine
    assert manager.current_engine_id == "qwen3_tts"
    assert new_engine.load_calls == 1


def test_idle_timer_waits_for_every_user_of_a_shared_engine_to_release() -> None:
    manager, fake_timers = make_manager()
    engine = FakeModelEngine("qwen3_tts")
    manager.acquire("qwen3_tts", lambda: engine)
    manager.acquire("qwen3_tts", lambda: engine)

    manager.release("qwen3_tts")
    assert len(fake_timers.handles) == 0  # second user is still inside

    manager.release("qwen3_tts")
    assert len(fake_timers.handles) == 1
    fake_timers.handles[0].fire()
    assert engine.unload_calls == 1


def test_reacquire_after_all_users_released_cancels_timer_and_rearms() -> None:
    manager, fake_timers = make_manager()
    engine = FakeModelEngine("qwen3_tts")
    manager.acquire("qwen3_tts", lambda: engine)
    manager.acquire("qwen3_tts", lambda: engine)
    manager.release("qwen3_tts")
    manager.release("qwen3_tts")
    manager.acquire("qwen3_tts", lambda: engine)  # count back to 1
    manager.release("qwen3_tts")  # count 0 -> new timer
    assert len(fake_timers.handles) == 2
    assert fake_timers.handles[0].cancelled is True
    assert engine.load_calls == 1


def test_release_below_zero_warns_and_never_goes_negative(
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager, fake_timers = make_manager()
    engine = FakeModelEngine("qwen3_tts")
    manager.acquire("qwen3_tts", lambda: engine)
    manager.release("qwen3_tts")
    with caplog.at_level(logging.WARNING, logger="vocalis"):
        manager.release("qwen3_tts")  # unbalanced extra release
    assert any("more times than acquired" in r.message for r in caplog.records)

    # Count is clamped at zero: one acquire + one release still arms the timer.
    handles_before = len(fake_timers.handles)
    manager.acquire("qwen3_tts", lambda: engine)
    manager.release("qwen3_tts")
    assert len(fake_timers.handles) == handles_before + 1


def test_acquiring_different_engine_still_unloads_one_that_is_in_use() -> None:
    """Documented conservative choice: single residency wins over use counts."""
    manager, _ = make_manager()
    old = FakeModelEngine("parakeet")
    new = FakeModelEngine("qwen3_tts")
    manager.acquire("parakeet", lambda: old)
    manager.acquire("qwen3_tts", lambda: new)
    assert old.unload_calls == 1
    assert manager.current_engine_id == "qwen3_tts"


def test_current_engine_is_the_loaded_object_or_none() -> None:
    """current_engine returns the loaded engine object, or None when nothing is loaded."""
    manager, _timers = make_manager()
    assert manager.current_engine is None

    engine = FakeModelEngine("a")
    manager.acquire("a", lambda: engine)

    assert manager.current_engine is engine
    manager.shutdown()
    assert manager.current_engine is None
