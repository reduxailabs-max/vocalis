"""Tests for vocalis.audio.capture.MicRecorder.

Uses a FakeStream in place of sounddevice.InputStream so tests never touch
real microphone hardware.
"""
import threading
import time

import numpy as np
import pytest

from vocalis.audio.capture import MicRecorder, RecordingAlreadyActiveError, pick_input_device


class FakeStream:
    def __init__(self, samplerate, channels, dtype, callback) -> None:  # type: ignore[no-untyped-def]
        self.samplerate = samplerate
        self.callback = callback
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True


def make_recorder(max_duration_seconds: float = 60.0) -> tuple[MicRecorder, list[FakeStream]]:
    streams: list[FakeStream] = []

    def factory(**kwargs):  # type: ignore[no-untyped-def]
        stream = FakeStream(**kwargs)
        streams.append(stream)
        return stream

    recorder = MicRecorder(max_duration_seconds=max_duration_seconds, stream_factory=factory)
    return recorder, streams


def test_start_creates_and_starts_a_stream() -> None:
    recorder, streams = make_recorder()
    recorder.start()
    assert len(streams) == 1
    assert streams[0].started is True
    assert recorder.is_recording is True


def test_stop_without_any_audio_returns_empty_array() -> None:
    recorder, _ = make_recorder()
    recorder.start()
    audio = recorder.stop()
    assert audio.shape == (0,)
    assert audio.dtype == np.float32
    assert recorder.is_recording is False


def test_stop_returns_captured_audio() -> None:
    recorder, streams = make_recorder()
    recorder.start()
    chunk1 = np.array([[0.1], [0.2]], dtype=np.float32)
    chunk2 = np.array([[0.3]], dtype=np.float32)
    streams[0].callback(chunk1, 2, None, None)
    streams[0].callback(chunk2, 1, None, None)
    audio = recorder.stop()
    np.testing.assert_array_almost_equal(audio, np.array([0.1, 0.2, 0.3], dtype=np.float32))


def test_double_start_raises() -> None:
    recorder, _ = make_recorder()
    recorder.start()
    with pytest.raises(RecordingAlreadyActiveError):
        recorder.start()
    recorder.stop()


def test_stream_stopped_and_closed_on_stop() -> None:
    recorder, streams = make_recorder()
    recorder.start()
    recorder.stop()
    assert streams[0].stopped is True
    assert streams[0].closed is True


def test_safety_timeout_stops_stream_but_stop_still_returns_captured_audio() -> None:
    recorder, streams = make_recorder(max_duration_seconds=0.05)
    recorder.start()
    streams[0].callback(np.array([[0.5]], dtype=np.float32), 1, None, None)
    time.sleep(0.2)
    assert recorder.is_recording is False  # safety cutoff already stopped the stream
    audio = recorder.stop()
    np.testing.assert_array_almost_equal(audio, np.array([0.5], dtype=np.float32))


class BlockingStream(FakeStream):
    """Mimics PortAudio's Pa_StopStream: stop() blocks until the audio
    callback thread finishes. If MicRecorder holds its lock across this
    call while the callback thread needs that same lock, both deadlock —
    which is exactly the bug this test targets.
    """

    def stop(self) -> None:
        callback_done = threading.Event()

        def run_callback() -> None:
            self.callback(np.array([[0.1]], dtype=np.float32), 1, None, None)
            callback_done.set()

        threading.Thread(target=run_callback, daemon=True).start()
        if not callback_done.wait(timeout=2.0):
            raise AssertionError(
                "Deadlock reproduced: stream.stop() never returned because "
                "the audio callback could not acquire the lock stop() holds"
            )
        self.stopped = True


def test_stop_does_not_deadlock_when_stream_stop_blocks_on_audio_callback() -> None:
    streams: list[BlockingStream] = []

    def factory(**kwargs):  # type: ignore[no-untyped-def]
        stream = BlockingStream(**kwargs)
        streams.append(stream)
        return stream

    recorder = MicRecorder(stream_factory=factory)
    recorder.start()

    audio = recorder.stop()  # must not hang and must not raise

    assert audio.shape == (1,)


def _device(name: str, inputs: int) -> dict[str, object]:
    return {"name": name, "max_input_channels": inputs}


def test_pick_input_device_prefers_pipewire_over_pulse() -> None:
    devices = [_device("hw:1,0", 2), _device("pulse", 32), _device("pipewire", 128), _device("default", 128)]
    assert pick_input_device(lambda: devices) == 2


def test_pick_input_device_falls_back_to_pulse() -> None:
    devices = [_device("hw:1,0", 2), _device("pulse", 32), _device("default", 128)]
    assert pick_input_device(lambda: devices) == 1


def test_pick_input_device_ignores_output_only_devices() -> None:
    devices = [_device("pipewire", 0), _device("default", 128)]
    assert pick_input_device(lambda: devices) is None


def test_pick_input_device_returns_none_when_no_sound_server_device() -> None:
    assert pick_input_device(lambda: [_device("default", 128)]) is None


def test_pick_input_device_returns_none_if_listing_devices_fails() -> None:
    def boom() -> list[dict[str, object]]:
        raise RuntimeError("no audio system")

    assert pick_input_device(boom) is None


def test_recorder_passes_chosen_device_to_the_stream() -> None:
    kwargs_seen: list[dict[str, object]] = []

    def factory(**kwargs):  # type: ignore[no-untyped-def]
        kwargs_seen.append(kwargs)
        return FakeStream(kwargs["samplerate"], kwargs["channels"], kwargs["dtype"], kwargs["callback"])

    MicRecorder(device=7, stream_factory=factory).start()
    assert kwargs_seen[0]["device"] == 7


def test_recorder_without_device_leaves_system_default() -> None:
    kwargs_seen: list[dict[str, object]] = []

    def factory(**kwargs):  # type: ignore[no-untyped-def]
        kwargs_seen.append(kwargs)
        return FakeStream(kwargs["samplerate"], kwargs["channels"], kwargs["dtype"], kwargs["callback"])

    MicRecorder(stream_factory=factory).start()
    assert "device" not in kwargs_seen[0]
