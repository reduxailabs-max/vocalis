"""Tests for vocalis.audio.output_device.OutputDeviceChooser (fakes only; no sound is played)."""
import threading
from typing import Any

from vocalis.audio.output_device import OutputDeviceChooser

DEVICES = [
    {"name": "HDMI 0", "max_output_channels": 8},
    {"name": "mic only", "max_output_channels": 0},
    {"name": "pipewire", "max_output_channels": 64},
    {"name": "pulse", "max_output_channels": 32},
    {"name": "default", "max_output_channels": 32},
]


class FakeStream:
    """A fake output stream whose behaviour is works / silent / raise."""

    def __init__(self, behaviour: str, callback: Any) -> None:
        """Remember how to behave and the callback to fire."""
        self.behaviour = behaviour
        self.callback = callback
        self.closed = False
        self.stopped = False

    def start(self) -> None:
        """Fire the callback (works), never fire it (silent) or fail (raise)."""
        if self.behaviour == "raise":
            raise OSError("cannot open")
        if self.behaviour == "works":
            import numpy as np

            self.callback(np.zeros((256, 1), dtype="float32"), 256, None, None)

    def stop(self) -> None:
        """Record that the stream was stopped."""
        self.stopped = True

    def close(self) -> None:
        """Record that the stream was closed."""
        self.closed = True


class Factory:
    """Behaviour per device index: works / silent (callback never fires) / raise."""

    def __init__(self, behaviours: "dict[int, str]") -> None:
        """Remember per-device behaviours and record what gets opened."""
        self.behaviours = behaviours
        self.opened: "list[int]" = []
        self.streams: "list[FakeStream]" = []

    def __call__(self, **kwargs: Any) -> FakeStream:
        """Open a fake stream for kwargs['device']."""
        index = kwargs["device"]
        self.opened.append(index)
        stream = FakeStream(self.behaviours.get(index, "silent"), kwargs["callback"])
        self.streams.append(stream)
        return stream


def make(
    behaviours: "dict[int, str]", preference: str = "auto", probe_timeout: float = 0.05
) -> "tuple[OutputDeviceChooser, Factory]":
    """Build a chooser wired to a fake device list and fake stream factory."""
    factory = Factory(behaviours)
    chooser = OutputDeviceChooser(
        preference=lambda: preference,
        query_devices=lambda: DEVICES,
        stream_factory=factory,
        probe_timeout_seconds=probe_timeout,
    )
    return chooser, factory


def test_output_names_lists_unique_devices_with_output_channels() -> None:
    chooser, _ = make({})
    assert chooser.output_names() == ["HDMI 0", "pipewire", "pulse", "default"]


def test_auto_prefers_pulse_and_skips_a_silent_default() -> None:
    chooser, factory = make({3: "works", 4: "silent", 2: "works"})
    assert chooser.current() == 3          # pulse, not pipewire/default/HDMI
    assert factory.opened == [3]


def test_auto_falls_through_a_dead_pulse_to_pipewire() -> None:
    chooser, factory = make({3: "silent", 2: "works"})
    assert chooser.current() == 2
    assert factory.opened == [3, 2]


def test_a_device_that_raises_is_skipped() -> None:
    chooser, _ = make({3: "raise", 2: "raise", 4: "works"})
    assert chooser.current() == 4


def test_explicit_choice_is_tried_first() -> None:
    chooser, factory = make({0: "works", 3: "works"}, preference="HDMI 0")
    assert chooser.current() == 0
    assert factory.opened == [0]


def test_an_unknown_choice_behaves_like_auto() -> None:
    chooser, _ = make({3: "works"}, preference="Unplugged headset")
    assert chooser.current() == 3


def test_nothing_works_returns_none() -> None:
    chooser, _ = make({})
    assert chooser.current() is None


def test_result_is_cached_until_marked_failed() -> None:
    chooser, factory = make({3: "works", 2: "works"})
    assert chooser.current() == 3
    assert chooser.current() == 3
    assert factory.opened == [3]            # second call did not probe again
    chooser.mark_failed(3)
    assert chooser.current() == 2           # next working device, 3 is remembered as bad


def test_reset_forgets_bad_devices() -> None:
    chooser, _ = make({3: "works"})
    chooser.mark_failed(3)
    assert chooser.current() is None
    chooser.reset()
    assert chooser.current() == 3


def test_every_probe_stream_is_stopped_and_closed() -> None:
    chooser, factory = make({3: "silent", 2: "works"})
    chooser.current()
    assert factory.streams and all(s.stopped and s.closed for s in factory.streams)


def test_probe_time_is_bounded_by_the_total_budget() -> None:
    """The whole current() call stays near total_timeout, not a probe over."""
    import time

    chooser, factory = make({}, probe_timeout=0.2)
    chooser._total_timeout = 0.3  # noqa: SLF001
    started = time.monotonic()
    assert chooser.current() is None
    assert time.monotonic() - started < 0.3 + 0.15
    assert len(factory.opened) <= 2


def test_changing_the_preference_re_chooses() -> None:
    state = {"pref": "auto"}
    factory = Factory({3: "works", 0: "works"})
    chooser = OutputDeviceChooser(preference=lambda: state["pref"], query_devices=lambda: DEVICES,
                                  stream_factory=factory, probe_timeout_seconds=0.05)
    assert chooser.current() == 3
    state["pref"] = "HDMI 0"
    assert chooser.current() == 0


def test_concurrent_callers_get_one_consistent_answer() -> None:
    """Six threads calling current() all see the same device."""
    chooser, factory = make({3: "works"})
    results: "list[int | None]" = []
    threads = [threading.Thread(target=lambda: results.append(chooser.current())) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)
    assert results == [3] * 6
    assert factory.opened == [3]


class FakeClock:
    """A clock the test moves by hand."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        """Return the current fake time."""
        return self.now


def make_timed(
    behaviours: "dict[int, str]", state: "dict[str, str] | None" = None
) -> "tuple[OutputDeviceChooser, Factory, FakeClock]":
    """Build a chooser with a fake clock and a 15 second negative TTL."""
    clock = FakeClock()
    factory = Factory(behaviours)
    prefs = state if state is not None else {"pref": "auto"}
    chooser = OutputDeviceChooser(
        preference=lambda: prefs["pref"],
        query_devices=lambda: DEVICES,
        stream_factory=factory,
        probe_timeout_seconds=0.01,
        negative_ttl_seconds=15.0,
        clock=clock,
    )
    return chooser, factory, clock


def test_none_is_cached_only_within_the_ttl() -> None:
    chooser, factory, clock = make_timed({})
    assert chooser.current() is None
    opened = list(factory.opened)
    factory.behaviours[3] = "works"
    clock.now += 5.0
    assert chooser.current() is None
    assert factory.opened == opened


def test_none_is_probed_again_after_the_ttl_and_then_cached_positively() -> None:
    chooser, factory, clock = make_timed({})
    assert chooser.current() is None
    factory.behaviours[3] = "works"
    clock.now += 15.0
    assert chooser.current() == 3
    opened = list(factory.opened)
    clock.now += 10000.0
    assert chooser.current() == 3
    assert factory.opened == opened


def test_a_positive_result_never_expires() -> None:
    chooser, factory, clock = make_timed({3: "works"})
    assert chooser.current() == 3
    clock.now += 1_000_000.0
    assert chooser.current() == 3
    assert factory.opened == [3]


def test_preference_change_clears_a_cached_none_at_once() -> None:
    state = {"pref": "auto"}
    chooser, factory, _ = make_timed({}, state)
    assert chooser.current() is None
    factory.behaviours[0] = "works"
    state["pref"] = "HDMI 0"
    assert chooser.current() == 0


def test_reset_clears_a_cached_none_at_once() -> None:
    chooser, factory, _ = make_timed({})
    assert chooser.current() is None
    factory.behaviours[3] = "works"
    chooser.reset()
    assert chooser.current() == 3


def test_a_budget_ended_probe_sequence_is_a_none_that_expires() -> None:
    chooser, factory, clock = make_timed({})
    chooser._probe_timeout = 0.2  # noqa: SLF001
    chooser._total_timeout = 0.3  # noqa: SLF001
    assert chooser.current() is None
    assert len(factory.opened) <= 2
    factory.behaviours[3] = "works"
    factory.behaviours[2] = "works"
    factory.behaviours[4] = "works"
    factory.behaviours[0] = "works"
    clock.now += 15.0
    assert chooser.current() in (0, 2, 3, 4)


def test_mark_failed_none_clears_a_cached_none_so_the_next_call_probes() -> None:
    """A stalled system default (None) triggers an immediate re-probe."""
    chooser, factory, _ = make_timed({})
    assert chooser.current() is None
    opened = len(factory.opened)
    factory.behaviours[3] = "works"
    chooser.mark_failed(None)
    assert chooser.current() == 3
    assert len(factory.opened) > opened
    assert chooser._bad == set()  # noqa: SLF001


def test_mark_failed_none_leaves_a_cached_device_alone() -> None:
    """mark_failed(None) does not drop a positive cache entry."""
    chooser, factory, _ = make_timed({3: "works"})
    assert chooser.current() == 3
    chooser.mark_failed(None)
    assert chooser.current() == 3
    assert factory.opened == [3]


def test_probe_callback_stays_silent() -> None:
    """The probe callback writes zeros into the output buffer."""
    import numpy as np

    chooser, factory = make({3: "works"})
    chooser.probe(3)
    buffer = np.ones((8, 1), dtype="float32")
    factory.streams[0].callback(buffer, 8, None, None)
    assert not buffer.any()


class GatedFactory(Factory):
    """Streams whose start() waits until every caller thread is inside current()."""

    def __init__(self, behaviours: "dict[int, str]", expected: int) -> None:
        """Set up the counter and the gate."""
        super().__init__(behaviours)
        self.expected = expected
        self.entered = 0
        self.count_lock = threading.Lock()
        self.all_in = threading.Event()

    def arrive(self) -> str:
        """Count one caller as inside current(); open the gate on the last."""
        with self.count_lock:
            self.entered += 1
            if self.entered >= self.expected:
                self.all_in.set()
        return "auto"

    def __call__(self, **kwargs: Any) -> FakeStream:
        """Open a stream, but hold start() until the gate opens."""
        stream = super().__call__(**kwargs)
        real_start = stream.start

        def gated_start() -> None:
            self.all_in.wait(5.0)
            real_start()

        stream.start = gated_start  # type: ignore[method-assign]
        return stream


def test_concurrent_callers_share_one_probe_sequence_while_probing() -> None:
    """All six threads are inside current() while the probe runs; only one sequence happens."""
    factory = GatedFactory({3: "works"}, expected=6)
    chooser = OutputDeviceChooser(
        preference=factory.arrive,
        query_devices=lambda: DEVICES,
        stream_factory=factory,
        probe_timeout_seconds=0.5,
    )
    results: "list[int | None]" = []
    results_lock = threading.Lock()

    def call() -> None:
        value = chooser.current()
        with results_lock:
            results.append(value)

    threads = [threading.Thread(target=call) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)
    assert factory.all_in.is_set()
    assert results == [3] * 6
    assert factory.opened == [3]


def test_a_long_probe_wait_is_capped_by_the_remaining_budget() -> None:
    """With a 1 s probe wait but a 0.2 s budget, current() returns in about 0.2 s."""
    import time

    chooser, _ = make({}, probe_timeout=1.0)
    chooser._total_timeout = 0.2  # noqa: SLF001
    started = time.monotonic()
    assert chooser.current() is None
    assert time.monotonic() - started < 0.2 + 0.15
