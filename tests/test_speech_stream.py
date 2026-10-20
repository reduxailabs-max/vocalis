"""Tests for vocalis.audio.speech_stream.SpeechStream.

Most tests use ManualStream (the test pulls audio itself), so ordering,
fades, pause, underrun and stall behaviour are fully deterministic.
"""
import threading
import time
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

from tests.fake_output import ManualStream, OutputRecorder, manual_stream_factory, wait_until
from vocalis.audio.playback import AudioPlayer, PlaybackTimeoutError
from vocalis.audio.speech_stream import SpeechStream

RATE = 1000


def _clip(length: int, value: float = 0.5) -> "npt.NDArray[np.float32]":
    return np.full(length, value, dtype=np.float32)


def _make(
    fade_seconds: float = 0.0, **stream_kwargs: Any
) -> "tuple[SpeechStream, OutputRecorder, list[int | None]]":
    recorder = OutputRecorder()
    failures: "list[int | None]" = []
    player = AudioPlayer(stream_factory=manual_stream_factory(recorder), on_device_failed=failures.append)
    return SpeechStream(player, fade_seconds=fade_seconds, **stream_kwargs), recorder, failures


def _manual(recorder: OutputRecorder) -> ManualStream:
    return cast(ManualStream, recorder.streams[-1])


def test_clips_play_back_to_back_inside_one_callback() -> None:
    stream, recorder, _ = _make()
    stream.start(RATE)
    assert stream.push(_clip(30, 0.1), RATE, 0)
    assert stream.push(_clip(30, 0.2), RATE, 1)

    out = _manual(recorder).pull(50)

    np.testing.assert_allclose(out[:30], 0.1, rtol=1e-6)
    np.testing.assert_allclose(out[30:50], 0.2, rtol=1e-6)
    assert stream.underruns == 0


def test_many_clips_use_a_single_open_stream() -> None:
    stream, recorder, _ = _make()
    stream.start(RATE)
    for tag in range(3):
        assert stream.push(_clip(20), RATE, tag)
    _manual(recorder).pull(60)

    assert recorder.opened == 1
    assert len(recorder.streams) == 1


def test_every_clip_is_faded_in_and_out() -> None:
    stream, recorder, _ = _make(fade_seconds=0.01)  # 10 samples at 1000 Hz
    stream.start(RATE)
    stream.push(_clip(100, 1.0), RATE, 0)

    out = _manual(recorder).pull(100)

    assert out[0] == 0.0
    assert 0.0 < out[1] < 1.0
    assert out[9] == pytest.approx(1.0)
    assert out[50] == pytest.approx(1.0)
    assert out[99] == 0.0


def test_pause_goes_silent_and_resume_replays_the_clip_from_its_start() -> None:
    stream, recorder, _ = _make()
    clip = np.linspace(0.1, 1.0, 100, dtype=np.float32)
    stream.start(RATE)
    stream.push(clip, RATE, 0)
    manual = _manual(recorder)
    np.testing.assert_allclose(manual.pull(40), clip[:40], rtol=1e-6)

    stream.pause()
    assert not manual.pull(30).any()

    stream.resume()
    np.testing.assert_allclose(manual.pull(100), clip, rtol=1e-6)


def test_a_dry_queue_plays_silence_and_counts_an_underrun_then_recovers() -> None:
    stream, recorder, _ = _make()
    stream.start(RATE)
    stream.push(_clip(50, 0.3), RATE, 0)
    manual = _manual(recorder)
    manual.pull(50)

    assert not manual.pull(20).any()
    assert stream.underruns == 1

    stream.push(_clip(50, 0.4), RATE, 1)
    np.testing.assert_allclose(manual.pull(50), 0.4, rtol=1e-6)


def test_clip_started_hook_gets_each_tag_in_order_on_a_helper_thread() -> None:
    started: "list[int]" = []
    threads: "list[str]" = []

    def hook(tag: int) -> None:
        started.append(tag)
        threads.append(threading.current_thread().name)

    stream, recorder, _ = _make(on_clip_started=hook)
    stream.start(RATE)
    stream.push(_clip(20), RATE, 5)
    stream.push(_clip(20), RATE, 7)
    _manual(recorder).pull(40)

    wait_until(lambda: started == [5, 7])
    assert all(name != threading.main_thread().name for name in threads)


def test_wait_until_done_returns_true_when_everything_played_and_closes_the_stream() -> None:
    stream, recorder, _ = _make()
    stream.start(RATE)
    stream.push(_clip(30), RATE, 0)
    stream.finish()
    _manual(recorder).pull(30)

    assert stream.wait_until_done() is True
    assert recorder.closed == 1


def test_wait_until_done_blocks_until_the_audio_has_been_pulled() -> None:
    stream, recorder, _ = _make()
    stream.start(RATE)
    stream.push(_clip(30), RATE, 0)
    stream.finish()
    results: "list[bool]" = []
    worker = threading.Thread(target=lambda: results.append(stream.wait_until_done()), daemon=True)
    worker.start()
    worker.join(timeout=0.2)
    assert results == []  # still waiting: nothing has been played yet

    _manual(recorder).pull(30)
    wait_until(lambda: results == [True])


def test_stop_ends_everything_and_wakes_waiters() -> None:
    stream, recorder, _ = _make()
    stream.start(RATE)
    stream.push(_clip(30), RATE, 0)
    stream.finish()
    results: "list[bool]" = []
    worker = threading.Thread(target=lambda: results.append(stream.wait_until_done()), daemon=True)
    worker.start()

    stream.stop()

    wait_until(lambda: results == [False])
    assert stream.queued_clips == 0
    assert recorder.closed == 1
    assert stream.push(_clip(10), RATE, 1) is False


def test_stop_before_start_means_the_device_is_never_opened() -> None:
    stream, recorder, _ = _make()
    stream.stop()

    stream.start(RATE)

    assert recorder.opened == 0
    assert stream.push(_clip(10), RATE, 0) is False


def test_stop_while_the_device_is_opening_closes_it() -> None:
    recorder = OutputRecorder()
    holder: "list[SpeechStream]" = []

    def factory(**kwargs: Any) -> ManualStream:
        holder[0].stop()  # the user presses stop during the open
        return ManualStream(recorder, **kwargs)

    stream = SpeechStream(AudioPlayer(stream_factory=factory), fade_seconds=0.0)
    holder.append(stream)

    stream.start(RATE)

    assert recorder.closed == 1


def test_a_full_queue_makes_push_wait_until_a_clip_has_played() -> None:
    stream, recorder, _ = _make(max_queued_clips=2)
    stream.start(RATE)
    stream.push(_clip(30), RATE, 0)
    stream.push(_clip(30), RATE, 1)
    results: "list[bool]" = []
    worker = threading.Thread(target=lambda: results.append(stream.push(_clip(30), RATE, 2)), daemon=True)
    worker.start()
    worker.join(timeout=0.2)
    assert results == []  # blocked by back-pressure

    _manual(recorder).pull(30)  # the first clip finishes playing

    wait_until(lambda: results == [True])


def test_stop_wakes_a_push_that_is_blocked_on_a_full_queue() -> None:
    stream, _recorder, _ = _make(max_queued_clips=1)
    stream.start(RATE)
    stream.push(_clip(30), RATE, 0)
    results: "list[bool]" = []
    worker = threading.Thread(target=lambda: results.append(stream.push(_clip(30), RATE, 1)), daemon=True)
    worker.start()
    worker.join(timeout=0.2)
    assert results == []

    stream.stop()

    wait_until(lambda: results == [False])


def test_a_stalled_device_raises_and_reports_the_device() -> None:
    stream, recorder, failures = _make(stall_seconds=0.2)
    stream.start(RATE)
    stream.push(_clip(30), RATE, 0)
    stream.finish()

    with pytest.raises(PlaybackTimeoutError):
        stream.wait_until_done()  # the fake device is never pulled, so it never "calls back"

    assert failures == [None]
    assert recorder.closed == 1


def test_a_stalled_device_also_fails_a_push_blocked_on_a_full_queue() -> None:
    stream, _recorder, failures = _make(stall_seconds=0.2, max_queued_clips=1)
    stream.start(RATE)
    stream.push(_clip(30), RATE, 0)

    with pytest.raises(PlaybackTimeoutError):
        stream.push(_clip(30), RATE, 1)

    assert failures == [None]


def test_a_quiet_queue_is_not_a_stall() -> None:
    stream, _recorder, _ = _make(stall_seconds=0.1)
    stream.start(RATE)
    threading.Event().wait(0.3)  # nothing queued, nothing to play: perfectly fine

    assert stream.push(_clip(10), RATE, 0) is True


def test_a_clip_at_another_sample_rate_is_resampled() -> None:
    stream, recorder, _ = _make()
    stream.start(RATE)
    stream.push(_clip(100, 0.5), 500, 0)  # 100 samples at 500 Hz = 0.2 s -> 200 samples at 1000 Hz

    out = _manual(recorder).pull(250)

    assert np.count_nonzero(out) == 200
    np.testing.assert_allclose(out[:200], 0.5, rtol=1e-6)


def test_an_empty_clip_is_accepted_but_not_queued() -> None:
    stream, _recorder, _ = _make()
    stream.start(RATE)

    assert stream.push(np.zeros(0, dtype=np.float32), RATE, 0) is True
    assert stream.queued_clips == 0


def test_a_device_that_fails_to_open_raises_and_leaves_the_stream_unusable() -> None:
    class RefusingStream:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def start(self) -> None:
            raise RuntimeError("device busy")

        def stop(self) -> None:
            pass

        def close(self) -> None:
            pass

    stream = SpeechStream(AudioPlayer(stream_factory=lambda **kw: RefusingStream(**kw)), fade_seconds=0.0)

    with pytest.raises(RuntimeError):
        stream.start(RATE)

    assert stream.push(_clip(10), RATE, 0) is False


class _SlowOpenPlayer(AudioPlayer):
    """A player whose open_stream takes (fake) clock time."""

    def __init__(self, recorder: OutputRecorder, clock: "list[float]", failures: "list[int | None]") -> None:
        super().__init__(stream_factory=manual_stream_factory(recorder), on_device_failed=failures.append)
        self._fake_clock = clock

    def open_stream(self, sample_rate: int, callback: Any, finished_callback: Any) -> None:
        self._fake_clock[0] += 30.0  # far longer than stall_seconds
        super().open_stream(sample_rate, callback, finished_callback)


def test_a_slow_device_open_is_not_mistaken_for_a_stall() -> None:
    recorder = OutputRecorder()
    now = [0.0]
    failures: "list[int | None]" = []
    stream = SpeechStream(
        _SlowOpenPlayer(recorder, now, failures), fade_seconds=0.0, stall_seconds=10.0, clock=lambda: now[0]
    )
    stream.start(RATE)
    stream.push(_clip(30), RATE, 0)
    stream.finish()
    _manual(recorder).pull(30)

    assert stream.wait_until_done() is True
    assert failures == []


def test_stop_during_a_slow_real_style_open_is_quiet_and_closes_once() -> None:
    recorder = OutputRecorder()
    entered = threading.Event()

    class PortAudioLike(ManualStream):
        def start(self) -> None:
            entered.set()
            time.sleep(0.05)
            if self.closed:
                raise RuntimeError("PortAudioError: stream is closed")
            super().start()

    failures: "list[int | None]" = []
    player = AudioPlayer(stream_factory=lambda **kw: PortAudioLike(recorder, **kw), on_device_failed=failures.append)
    stream = SpeechStream(player, fade_seconds=0.0)
    errors: "list[BaseException]" = []

    def run() -> None:
        try:
            stream.start(RATE)
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=run)
    worker.start()
    assert entered.wait(2.0)
    stream.stop()
    worker.join(5.0)

    assert errors == []
    assert failures == []
    assert recorder.closed == 1


def test_a_deliberately_paused_stream_is_never_a_stall() -> None:
    now = [0.0]
    stream, _recorder, failures = _make(max_queued_clips=1, stall_seconds=10.0, clock=lambda: now[0])
    stream.start(RATE)
    stream.push(_clip(30), RATE, 0)
    stream.pause()
    results: "list[bool]" = []
    errors: "list[BaseException]" = []

    def run() -> None:
        try:
            results.append(stream.push(_clip(30), RATE, 1))
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout=0.2)
    now[0] += 60.0
    worker.join(timeout=0.4)

    assert errors == [] and results == []
    assert failures == []
    stream.stop()
    wait_until(lambda: results == [False])


def test_a_stream_that_never_opened_cannot_close_another_streams_device() -> None:
    """Two SpeechStreams share one player: stopping the never-opened one leaves the other's device open."""
    recorder = OutputRecorder()
    player = AudioPlayer(stream_factory=manual_stream_factory(recorder))
    opened = SpeechStream(player, fade_seconds=0.0)
    never_opened = SpeechStream(player, fade_seconds=0.0)
    opened.start(RATE)

    never_opened.stop()

    assert recorder.closed == 0
    assert opened.push(_clip(10), RATE, 0)
    opened.stop()
    assert recorder.closed == 1
