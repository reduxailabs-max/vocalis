"""Tests for vocalis.audio.playback.AudioPlayer.

Uses a FakeOutputStream in place of sounddevice.OutputStream so tests
never touch real speakers. Mirrors test_capture.py's FakeStream pattern.
"""
import threading
import time
from typing import Any

import numpy as np
import pytest

from vocalis.audio.playback import AudioPlayer, PlaybackAlreadyActiveError, PlaybackTimeoutError


class FakeOutputStream:
    def __init__(self, samplerate, channels, dtype, callback, finished_callback=None) -> None:  # type: ignore[no-untyped-def]
        self.samplerate = samplerate
        self.callback = callback
        self.finished_callback = finished_callback
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True


def make_player() -> tuple[AudioPlayer, list[FakeOutputStream]]:
    streams: list[FakeOutputStream] = []

    def factory(**kwargs):  # type: ignore[no-untyped-def]
        stream = FakeOutputStream(**kwargs)
        streams.append(stream)
        return stream

    return AudioPlayer(stream_factory=factory), streams


def _run_callback_to_completion(stream: FakeOutputStream, audio: np.ndarray) -> None:
    """Drives a fake stream's callback exactly like real sounddevice would:
    repeatedly asks for chunks until the audio is exhausted, then calls
    finished_callback() (as real PortAudio does once the stream drains
    after the callback raises CallbackStop)."""
    frame_size = 4
    outdata = np.zeros((frame_size, 1), dtype=np.float32)
    while True:
        try:
            stream.callback(outdata, frame_size, None, None)
        except Exception:
            break
    if stream.finished_callback is not None:
        stream.finished_callback()


def test_play_starts_a_stream() -> None:
    player, streams = make_player()
    player.play(np.array([0.1, 0.2, 0.3], dtype=np.float32), 22050)
    assert len(streams) == 1
    assert streams[0].started is True
    assert player.is_playing is True


def test_play_while_already_playing_raises() -> None:
    player, _ = make_player()
    player.play(np.array([0.1], dtype=np.float32), 22050)
    with pytest.raises(PlaybackAlreadyActiveError):
        player.play(np.array([0.2], dtype=np.float32), 22050)


def test_wait_until_done_returns_true_after_natural_completion() -> None:
    player, streams = make_player()
    audio = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    player.play(audio, 22050)

    driver = threading.Thread(target=_run_callback_to_completion, args=(streams[0], audio))
    driver.start()
    finished_naturally = player.wait_until_done()
    driver.join()

    assert finished_naturally is True
    assert player.is_playing is False
    assert streams[0].closed is True


def test_stop_stops_playback_and_wait_until_done_returns_false() -> None:
    player, streams = make_player()
    player.play(np.array([0.1, 0.2, 0.3], dtype=np.float32), 22050)

    def stop_soon() -> None:
        time.sleep(0.05)
        player.stop()

    stopper = threading.Thread(target=stop_soon)
    stopper.start()
    finished_naturally = player.wait_until_done()
    stopper.join()

    assert finished_naturally is False
    assert player.is_playing is False
    assert streams[0].stopped is True


def test_stop_when_not_playing_is_a_no_op() -> None:
    player, _ = make_player()
    player.stop()  # must not raise
    assert player.is_playing is False


def test_stop_racing_with_play_start_does_not_drop_interrupt() -> None:
    """Regression test for a narrow race in play(): the old code called
    stream.start() and only assigned self._stream afterward, with no lock
    around either step. A stop() landing in that window would see
    self._stream still as None and silently do nothing -- dropping the
    interrupt while playback kept going uninterrupted.

    The fix assigns self._stream (under the lock) BEFORE stream.start()
    is called, so a concurrent stop() can never observe self._stream as
    None once the stream has been created. We simulate the race by having
    the fake stream's start() trigger a concurrent stop() call and wait
    for it to finish before returning -- landing stop() exactly in the
    old code's vulnerable gap. Against the old ordering, this test would
    fail: self._stream would still be None when stop() ran, so stop()
    would no-op, stream.stop()/close() would never be called, and the
    stream would be left "playing" after play() returns.
    """
    streams: list[FakeOutputStream] = []
    holder: dict[str, AudioPlayer] = {}

    class RacingOutputStream(FakeOutputStream):
        def start(self) -> None:
            # Simulate stop() being called by another thread (e.g. a
            # second hotkey press) concurrently with play() still setting
            # up the stream.
            stopper = threading.Thread(target=holder["player"].stop)
            stopper.start()
            stopper.join()
            super().start()

    def factory(**kwargs):  # type: ignore[no-untyped-def]
        stream = RacingOutputStream(**kwargs)
        streams.append(stream)
        return stream

    player = AudioPlayer(stream_factory=factory)
    holder["player"] = player

    player.play(np.array([0.1, 0.2, 0.3], dtype=np.float32), 22050)

    assert player.is_playing is False
    assert streams[0].stopped is True
    assert streams[0].closed is True


def test_wait_until_done_raises_timeout_error_when_callback_never_fires() -> None:
    """Regression test for the real production bug: wait_until_done() used
    to call self._done_event.wait() with no timeout at all, so if the
    audio callback was never invoked (e.g. a misconfigured default sound
    device where the stream opens but PortAudio never drives it), the
    calling thread hung forever.

    Here the fake stream's finished_callback is deliberately never
    invoked -- simulating exactly that scenario -- with a short audio
    clip and a tiny timeout_margin_seconds so the test itself stays fast.
    wait_until_done() must still raise PlaybackTimeoutError instead of
    blocking, and must still perform safe cleanup (stop()/close() called)
    despite the timeout.
    """
    streams: list[FakeOutputStream] = []

    def factory(**kwargs):  # type: ignore[no-untyped-def]
        stream = FakeOutputStream(**kwargs)
        streams.append(stream)
        return stream

    player = AudioPlayer(stream_factory=factory, timeout_margin_seconds=0.1)
    # Very short audio at a high sample rate keeps expected_duration_seconds
    # tiny, so total timeout (duration + 0.1s margin) is well under 1s.
    audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    player.play(audio, sample_rate=48000)

    # Deliberately do NOT drive the fake stream's callback or call
    # finished_callback() -- this is the bug scenario: the callback never
    # fires, so self._done_event is never set.
    start = time.monotonic()
    with pytest.raises(PlaybackTimeoutError):
        player.wait_until_done()
    elapsed = time.monotonic() - start

    assert elapsed < 5.0, f"wait_until_done() took {elapsed:.2f}s -- timeout was not bounded"
    assert player.is_playing is False
    assert streams[0].stopped is True
    assert streams[0].closed is True


class ClaimingEvent:
    """Fake stand-in for AudioPlayer._done_event that simulates a concurrent
    stop() call landing in the exact race window: after
    self._done_event.wait() has decided to time out (i.e. is about to
    return False) but before wait_until_done() acquires the lock to claim
    self._stream itself.

    Calling wait() here plays the part of that race by invoking stop()
    directly -- which claims and cleans up the stream and sets
    _stopped_early -- before returning False, exactly mirroring what a
    real concurrent stop() would have already done by the time
    wait_until_done() gets the lock.
    """

    def __init__(self, player: AudioPlayer) -> None:
        self._player = player

    def wait(self, timeout: float | None = None) -> bool:
        self._player.stop()
        return False

    def set(self) -> None:
        pass


def test_wait_until_done_does_not_raise_when_stop_wins_race_at_timeout() -> None:
    """Regression test for a race in the timeout branch of wait_until_done().

    self._done_event.wait(timeout=...) can return False (meaning "timed
    out") in the narrow window just before a concurrent stop() call
    (running on another thread, e.g. QuickReaderFeature's interrupt path)
    finishes claiming and cleaning up the stream and setting the event.
    By the time wait_until_done() then acquires its own lock to do its own
    swap, self._stream is already None -- stop() got there first, and did
    a completely legitimate, clean stop.

    Before the fix, wait_until_done() didn't check for this and raised
    PlaybackTimeoutError unconditionally in the timeout branch, even
    though nothing actually hung. This test simulates that exact
    interleaving deterministically (via ClaimingEvent, which calls
    stop() itself right before wait() reports a timeout) and asserts
    wait_until_done() returns False -- the correct "stopped early"
    outcome -- rather than raising. Against the pre-fix code, this test
    fails with PlaybackTimeoutError.
    """
    player, streams = make_player()
    audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    player.play(audio, sample_rate=48000)

    player._done_event = ClaimingEvent(player)  # type: ignore[assignment]

    finished_naturally = player.wait_until_done()

    assert finished_naturally is False
    assert player.is_playing is False
    assert streams[0].stopped is True
    assert streams[0].closed is True


def test_play_recovers_when_stream_start_fails() -> None:
    """Regression test for the real production bug: play() assigned
    self._stream = stream BEFORE calling stream.start() (a deliberate fix
    for an earlier race -- see test_stop_racing_with_play_start_does_not_drop_interrupt),
    but if stream.start() raised (e.g. device busy, PipeWire restarting, a
    Bluetooth headset disconnecting), self._stream was never cleared back
    to None. is_playing then stayed True forever, so every subsequent
    play() call raised PlaybackAlreadyActiveError -- Quick Reader would get
    permanently stuck saying "Playback failed" until the whole daemon
    process was restarted.

    Against the pre-fix code (no try/except around stream.start()), this
    test fails at the second player.play(...) call: is_playing is still
    True after the first (failed) play(), so the second call raises
    PlaybackAlreadyActiveError instead of succeeding.
    """

    class FailingStartStream(FakeOutputStream):
        def start(self) -> None:
            raise RuntimeError("device busy")

    streams: list[FakeOutputStream] = []

    def factory(**kwargs):  # type: ignore[no-untyped-def]
        stream = FailingStartStream(**kwargs)
        streams.append(stream)
        return stream

    player = AudioPlayer(stream_factory=factory)

    with pytest.raises(RuntimeError, match="device busy"):
        player.play(np.array([0.1, 0.2, 0.3], dtype=np.float32), 22050)

    assert player.is_playing is False
    assert streams[0].closed is True

    # A second play() call, with a working fake stream, must succeed
    # rather than raising PlaybackAlreadyActiveError.
    player._stream_factory = lambda **kwargs: FakeOutputStream(**kwargs)
    player.play(np.array([0.4, 0.5], dtype=np.float32), 22050)
    assert player.is_playing is True


def test_callback_writes_audio_chunks_correctly() -> None:
    player, streams = make_player()
    audio = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    player.play(audio, 22050)
    stream = streams[0]

    outdata = np.zeros((3, 1), dtype=np.float32)
    stream.callback(outdata, 3, None, None)
    np.testing.assert_array_almost_equal(outdata[:, 0], audio[0:3])

    outdata2 = np.zeros((3, 1), dtype=np.float32)
    with pytest.raises(Exception):  # sd.CallbackStop signals natural end
        stream.callback(outdata2, 3, None, None)
    np.testing.assert_array_almost_equal(outdata2[:2, 0], audio[3:5])


# --- Chosen output device (device_provider / on_device_failed) ---------------

_AUDIO = np.array([0.1, 0.2, 0.3], dtype=np.float32)


def make_recording_player(
    stream_cls: type = FakeOutputStream, **player_kwargs: Any
) -> tuple[AudioPlayer, list[dict[str, object]], list[FakeOutputStream]]:
    """Build a player whose fake factory records every kwargs dict it receives."""
    received: list[dict[str, object]] = []
    streams: list[FakeOutputStream] = []

    def factory(**kwargs):  # type: ignore[no-untyped-def]
        received.append(dict(kwargs))
        stream = stream_cls(**{k: v for k, v in kwargs.items() if k != "device"})
        streams.append(stream)
        return stream

    player = AudioPlayer(stream_factory=factory, **player_kwargs)
    return player, received, streams


def test_play_passes_the_provider_device_to_the_stream_factory() -> None:
    player, received, _ = make_recording_player(device_provider=lambda: 7)
    player.play(_AUDIO, 48000)
    assert len(received) == 1
    assert received[0]["device"] == 7


def test_play_without_a_provider_passes_no_device_kwarg() -> None:
    player, received, _ = make_recording_player()
    player.play(_AUDIO, 48000)
    assert "device" not in received[0]


def test_play_with_a_provider_returning_none_passes_no_device_kwarg() -> None:
    player, received, _ = make_recording_player(device_provider=lambda: None)
    player.play(_AUDIO, 48000)
    assert "device" not in received[0]


def test_provider_is_asked_on_every_play() -> None:
    answers = iter([3, 5])
    player, received, _ = make_recording_player(device_provider=lambda: next(answers))
    player.play(_AUDIO, 48000)
    player.stop()
    player.play(_AUDIO, 48000)
    assert [r["device"] for r in received] == [3, 5]


def test_provider_is_called_outside_the_player_lock() -> None:
    holder: dict[str, AudioPlayer] = {}
    lock_was_free: list[bool] = []

    def provider() -> int | None:
        acquired = holder["player"]._lock.acquire(blocking=False)
        lock_was_free.append(acquired)
        if acquired:
            holder["player"]._lock.release()
        return 2

    player, _, _ = make_recording_player(device_provider=provider)
    holder["player"] = player
    player.play(_AUDIO, 48000)
    assert lock_was_free == [True]


def test_raising_provider_falls_back_to_the_default_device() -> None:
    def provider() -> int | None:
        raise RuntimeError("no devices")

    player, received, streams = make_recording_player(device_provider=provider)
    player.play(_AUDIO, 48000)
    assert "device" not in received[0]
    assert streams[0].started is True


def test_timeout_reports_the_device_once_and_still_raises() -> None:
    failed: list[int | None] = []
    player, _, _ = make_recording_player(
        device_provider=lambda: 4,
        on_device_failed=failed.append,
        timeout_margin_seconds=0.1,
    )
    player.play(_AUDIO, 48000)
    with pytest.raises(PlaybackTimeoutError):
        player.wait_until_done()
    assert failed == [4]


def test_timeout_on_the_default_device_reports_none() -> None:
    failed: list[int | None] = []
    player, _, _ = make_recording_player(on_device_failed=failed.append, timeout_margin_seconds=0.1)
    player.play(_AUDIO, 48000)
    with pytest.raises(PlaybackTimeoutError):
        player.wait_until_done()
    assert failed == [None]


def test_failing_start_reports_the_device_once_and_reraises_original() -> None:
    class FailingStartStream(FakeOutputStream):
        def start(self) -> None:
            raise RuntimeError("device busy")

    failed: list[int | None] = []
    player, _, _ = make_recording_player(
        stream_cls=FailingStartStream, device_provider=lambda: 9, on_device_failed=failed.append
    )
    with pytest.raises(RuntimeError, match="device busy"):
        player.play(_AUDIO, 48000)
    assert failed == [9]
    assert player.is_playing is False


def test_natural_finish_does_not_report_a_failure() -> None:
    failed: list[int | None] = []
    player, _, streams = make_recording_player(
        device_provider=lambda: 1, on_device_failed=failed.append
    )
    player.play(_AUDIO, 48000)
    driver = threading.Thread(target=_run_callback_to_completion, args=(streams[0], _AUDIO))
    driver.start()
    assert player.wait_until_done() is True
    driver.join()
    assert failed == []


def test_stop_does_not_report_a_failure() -> None:
    failed: list[int | None] = []
    player, _, _ = make_recording_player(
        device_provider=lambda: 1, on_device_failed=failed.append
    )
    player.play(_AUDIO, 48000)
    player.stop()
    assert player.wait_until_done() is False
    assert failed == []


def test_raising_hook_does_not_replace_the_timeout_error() -> None:
    def hook(device: int | None) -> None:
        raise ValueError("hook broke")

    player, _, _ = make_recording_player(on_device_failed=hook, timeout_margin_seconds=0.1)
    player.play(_AUDIO, 48000)
    with pytest.raises(PlaybackTimeoutError):
        player.wait_until_done()


def test_raising_hook_does_not_replace_the_start_error() -> None:
    class FailingStartStream(FakeOutputStream):
        def start(self) -> None:
            raise RuntimeError("device busy")

    def hook(device: int | None) -> None:
        raise ValueError("hook broke")

    player, _, _ = make_recording_player(stream_cls=FailingStartStream, on_device_failed=hook)
    with pytest.raises(RuntimeError, match="device busy"):
        player.play(_AUDIO, 48000)


# --- Open-time failures: report the device and retry once ---------------------


def make_scripted_player(
    answers: list[int | None] | None,
    bad_devices: dict[int | None, str],
    fail_at: str,
    **player_kwargs: Any,
) -> tuple[AudioPlayer, list[dict[str, Any]], list[int | None]]:
    """Player whose provider gives `answers` in order (None list = no provider).

    A device in `bad_devices` fails with RuntimeError(<message>) either in the
    factory (fail_at="factory") or in stream.start() (fail_at="start").
    Returns the player, the kwargs seen by the factory and the hook calls.
    """
    received: list[dict[str, Any]] = []
    failed: list[int | None] = []

    def factory(**kwargs):  # type: ignore[no-untyped-def]
        received.append(dict(kwargs))
        device = kwargs.get("device")
        message = bad_devices.get(device)
        if message is not None and fail_at == "factory":
            raise RuntimeError(message)

        class Stream(FakeOutputStream):
            def start(self) -> None:
                if message is not None and fail_at == "start":
                    raise RuntimeError(message)
                super().start()

        return Stream(**{k: v for k, v in kwargs.items() if k != "device"})

    provider = None
    if answers is not None:
        queue = iter(answers)
        provider = lambda: next(queue)  # noqa: E731
    player = AudioPlayer(
        stream_factory=factory,
        device_provider=provider,
        on_device_failed=failed.append,
        **player_kwargs,
    )
    return player, received, failed


@pytest.mark.parametrize("fail_at", ["factory", "start"])
def test_open_failure_retries_with_a_different_device(fail_at: str) -> None:
    player, received, failed = make_scripted_player([9, 4], {9: "bad device"}, fail_at)
    player.play(_AUDIO, 48000)
    assert [r.get("device") for r in received] == [9, 4]
    assert failed == [9]
    assert player.is_playing is True


@pytest.mark.parametrize("fail_at", ["factory", "start"])
def test_open_failure_with_the_same_device_again_raises_the_original(fail_at: str) -> None:
    player, received, failed = make_scripted_player([9, 9], {9: "bad device"}, fail_at)
    with pytest.raises(RuntimeError, match="bad device"):
        player.play(_AUDIO, 48000)
    assert len(received) == 1
    assert failed == [9]
    assert player.is_playing is False


@pytest.mark.parametrize("fail_at", ["factory", "start"])
def test_open_failure_when_the_retry_device_also_fails_raises_the_retry_error(
    fail_at: str,
) -> None:
    player, received, failed = make_scripted_player(
        [9, 4], {9: "bad device", 4: "retry bad"}, fail_at
    )
    with pytest.raises(RuntimeError, match="retry bad"):
        player.play(_AUDIO, 48000)
    assert [r.get("device") for r in received] == [9, 4]
    assert failed == [9, 4]
    assert player.is_playing is False


@pytest.mark.parametrize("fail_at", ["factory", "start"])
def test_open_failure_retry_with_none_uses_the_default_device(fail_at: str) -> None:
    player, received, failed = make_scripted_player([9, None], {9: "bad device"}, fail_at)
    player.play(_AUDIO, 48000)
    assert len(received) == 2
    assert received[0]["device"] == 9
    assert "device" not in received[1]
    assert failed == [9]
    assert player.is_playing is True


@pytest.mark.parametrize("fail_at", ["factory", "start"])
def test_open_failure_without_a_provider_is_a_single_attempt(fail_at: str) -> None:
    player, received, failed = make_scripted_player(None, {None: "bad device"}, fail_at)
    with pytest.raises(RuntimeError, match="bad device"):
        player.play(_AUDIO, 48000)
    assert len(received) == 1
    assert failed == [None]
    assert player.is_playing is False


def test_timeout_reports_the_device_even_when_stop_raises() -> None:
    class RaisingStopStream(FakeOutputStream):
        def stop(self) -> None:
            raise RuntimeError("stop broke")

    failed: list[int | None] = []
    player, _, _ = make_recording_player(
        stream_cls=RaisingStopStream,
        device_provider=lambda: 4,
        on_device_failed=failed.append,
        timeout_margin_seconds=0.1,
    )
    player.play(_AUDIO, 48000)
    with pytest.raises(RuntimeError):
        player.wait_until_done()
    assert failed == [4]


def test_stop_winning_the_race_at_timeout_does_not_report_a_failure() -> None:
    failed: list[int | None] = []
    player, _, _ = make_recording_player(
        device_provider=lambda: 4, on_device_failed=failed.append
    )
    player.play(_AUDIO, 48000)
    player._done_event = ClaimingEvent(player)  # type: ignore[assignment]
    assert player.wait_until_done() is False
    assert failed == []
