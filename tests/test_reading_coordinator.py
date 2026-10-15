"""Tests for vocalis.features.reading_coordinator.ReadingCoordinator.

Ownership is identified by a stable `owner` object (pass the feature
instance itself), never by comparing the `stopper` callable with `is` —
a bound method obtained via `self.method` is a fresh object on every
attribute access in Python, so two calls from the very same instance
would never compare equal that way. These tests pin that down directly.
"""
from vocalis.features.reading_coordinator import ReadingCoordinator


class _Owner:
    """A stand-in for a feature instance, with a bound-method stopper."""

    def __init__(self) -> None:
        self.stop_calls = 0

    def stop(self) -> None:
        self.stop_calls += 1


def test_first_start_reading_does_not_call_its_own_stopper() -> None:
    coordinator = ReadingCoordinator()
    owner = _Owner()

    coordinator.start_reading(owner, owner.stop)

    assert owner.stop_calls == 0


def test_same_owner_starting_again_does_not_stop_itself() -> None:
    """Regression test: two bound-method accesses on the same instance
    are different objects, so ownership must be tracked by `owner`
    identity, never by comparing the stopper callable itself."""
    coordinator = ReadingCoordinator()
    owner = _Owner()

    coordinator.start_reading(owner, owner.stop)
    coordinator.start_reading(owner, owner.stop)  # fresh bound method each time

    assert owner.stop_calls == 0


def test_a_different_owner_starting_stops_the_previous_owner() -> None:
    coordinator = ReadingCoordinator()
    first = _Owner()
    second = _Owner()

    coordinator.start_reading(first, first.stop)
    coordinator.start_reading(second, second.stop)

    assert first.stop_calls == 1
    assert second.stop_calls == 0


def test_finish_reading_clears_ownership() -> None:
    coordinator = ReadingCoordinator()
    first = _Owner()
    second = _Owner()

    coordinator.start_reading(first, first.stop)
    coordinator.finish_reading(first)
    coordinator.start_reading(second, second.stop)

    assert first.stop_calls == 0  # first already released; no stop needed


def test_stale_finish_reading_does_not_clobber_a_newer_owner() -> None:
    coordinator = ReadingCoordinator()
    first = _Owner()
    second = _Owner()

    coordinator.start_reading(first, first.stop)
    coordinator.start_reading(second, second.stop)  # first is stopped, second owns it
    coordinator.finish_reading(first)  # stale: first no longer owns anything
    assert coordinator._owner is second  # noqa: SLF001

    coordinator.start_reading(_Owner(), lambda: None)
    assert second.stop_calls == 1  # second was already the current owner, so the next start_reading() legitimately preempts it
