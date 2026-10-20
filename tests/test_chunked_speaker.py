"""Tests for vocalis.features.chunked_speaker.ChunkedSpeaker."""
import threading
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from tests.fake_output import FakeAudioPlayer, ManualStream, OutputRecorder, manual_stream_factory, wait_until
from vocalis.audio.playback import AudioPlayer
from vocalis.features.chunked_speaker import ChunkedSpeaker, SpeakOutcome

RATE = 1000


class FakeSource:
    """Makes `length` samples per piece, valued 0.1 * (call number)."""

    def __init__(self, length: int = 100, rate: int = RATE, fail_on_call: "int | None" = None) -> None:
        """Set the piece length, rate and which call (1-based) should fail."""
        self.length = length
        self.rate = rate
        self.fail_on_call = fail_on_call
        self.calls: "list[str]" = []
        self.closed = 0

    def synthesize(self, text: str) -> "tuple[npt.NDArray[np.float32], int]":
        """Record the call; fail on the chosen call number."""
        self.calls.append(text)
        if self.fail_on_call == len(self.calls):
            raise RuntimeError("no voice could speak")
        return np.full(self.length, 0.1 * len(self.calls), dtype=np.float32), self.rate

    def close(self) -> None:
        """Count a release of the voice."""
        self.closed += 1


def _manual_player() -> "tuple[AudioPlayer, OutputRecorder]":
    """A player whose fake device is driven by the test (pull)."""
    recorder = OutputRecorder()
    return AudioPlayer(stream_factory=manual_stream_factory(recorder)), recorder


def _start(speaker: ChunkedSpeaker, chunks: "list[str]") -> "tuple[threading.Thread, list[SpeakOutcome]]":
    """Run the reading on a worker thread; outcomes collects its result."""
    outcomes: "list[SpeakOutcome]" = []
    thread = threading.Thread(target=lambda: outcomes.append(speaker.run(chunks)), daemon=True)
    thread.start()
    return thread, outcomes


def test_a_whole_reading_plays_every_chunk_once_in_order_on_one_stream() -> None:
    """A whole reading plays every chunk once in order on one stream."""
    player = FakeAudioPlayer()
    source = FakeSource(length=300)
    progress: "list[tuple[int, int]]" = []
    speaker = ChunkedSpeaker(
        player, lambda: source, on_chunk_started=lambda current, total: progress.append((current, total)),
        fade_seconds=0.0,
    )

    outcome = speaker.run(["one", "two", "three"])

    assert outcome is SpeakOutcome.DONE
    assert source.calls == ["one", "two", "three"]
    assert player.output.opened == 1
    played = player.output.played()
    assert len(played) == 900
    assert played[0] == np.float32(0.1) and played[-1] == np.float32(0.1 * 3)
    wait_until(lambda: progress == [(1, 3), (2, 3), (3, 3)])
    assert source.closed == 1


def test_no_chunks_is_done_without_opening_a_stream() -> None:
    """No chunks is done without opening a stream."""
    player = FakeAudioPlayer()
    source = FakeSource()

    assert ChunkedSpeaker(player, lambda: source).run([]) is SpeakOutcome.DONE
    assert player.output.opened == 0
    assert source.closed == 1


def test_the_producer_waits_when_the_queue_is_full_and_stop_ends_it() -> None:
    """The producer waits when the queue is full and stop ends it."""
    player, recorder = _manual_player()
    source = FakeSource()
    speaker = ChunkedSpeaker(player, lambda: source, max_queued_clips=2, fade_seconds=0.0)
    thread, outcomes = _start(speaker, [f"piece {n}" for n in range(6)])

    wait_until(lambda: len(source.calls) == 3)  # 2 queued, the 3rd is made and waits to be pushed
    threading.Event().wait(0.2)
    assert len(source.calls) == 3  # back-pressure: no more are made while the queue is full

    speaker.stop()
    thread.join(timeout=5.0)
    assert outcomes == [SpeakOutcome.STOPPED]
    assert source.closed == 1
    assert recorder.closed == 1


def test_stop_before_run_returns_stopped_without_opening_a_stream() -> None:
    """Stop before run returns stopped without opening a stream."""
    player, recorder = _manual_player()
    source = FakeSource()
    speaker = ChunkedSpeaker(player, lambda: source)

    speaker.stop()

    assert speaker.run(["anything"]) is SpeakOutcome.STOPPED
    assert recorder.opened == 0
    assert source.calls == []


def test_a_stop_during_synthesis_is_honoured_when_the_piece_arrives() -> None:
    """A stop during synthesis is honoured when the piece arrives."""
    player, recorder = _manual_player()
    entered = threading.Event()
    release = threading.Event()

    class SlowSource(FakeSource):
        """A source whose synthesis blocks until the test lets it go."""

        def synthesize(self, text: str) -> "tuple[npt.NDArray[np.float32], int]":
            """Wait to be released, then make the piece."""
            entered.set()
            release.wait(5.0)
            return super().synthesize(text)

    source = SlowSource()
    speaker = ChunkedSpeaker(player, lambda: source)
    thread, outcomes = _start(speaker, ["only piece"])
    assert entered.wait(5.0)

    speaker.stop()
    release.set()
    thread.join(timeout=5.0)

    assert outcomes == [SpeakOutcome.STOPPED]
    assert recorder.opened == 0  # the late piece never made a sound
    assert source.closed == 1


def test_a_failing_voice_gives_speech_failed_and_releases_the_voice() -> None:
    """A failing voice gives speech failed and releases the voice."""
    player, _recorder = _manual_player()
    source = FakeSource(fail_on_call=2)
    speaker = ChunkedSpeaker(player, lambda: source, fade_seconds=0.0)
    thread, outcomes = _start(speaker, ["one", "two", "three"])

    thread.join(timeout=5.0)

    assert outcomes == [SpeakOutcome.SPEECH_FAILED]
    assert source.closed == 1


def test_a_source_that_cannot_be_built_gives_speech_failed() -> None:
    """A source that cannot be built gives speech failed."""
    player, _recorder = _manual_player()

    def broken() -> FakeSource:
        """A source factory that fails."""
        raise RuntimeError("no engines")

    assert ChunkedSpeaker(player, broken).run(["x"]) is SpeakOutcome.SPEECH_FAILED


def test_a_stalled_sound_device_gives_playback_failed() -> None:
    """A stalled sound device gives playback failed."""
    player, recorder = _manual_player()
    source = FakeSource()
    speaker = ChunkedSpeaker(player, lambda: source, stall_seconds=0.2)
    thread, outcomes = _start(speaker, ["one"])  # the fake device is never pulled

    thread.join(timeout=5.0)

    assert outcomes == [SpeakOutcome.PLAYBACK_FAILED]
    assert recorder.closed == 1
    assert source.closed == 1


def test_a_device_that_refuses_to_open_gives_playback_failed() -> None:
    """A device that refuses to open gives playback failed."""
    class RefusingStream:
        """A sound stream that cannot be started."""

        def __init__(self, **kwargs: Any) -> None:
            """Accept and ignore the stream arguments."""

        def start(self) -> None:
            """Refuse to start."""
            raise RuntimeError("device busy")

        def stop(self) -> None:
            """Nothing to stop."""

        def close(self) -> None:
            """Nothing to close."""

    player = AudioPlayer(stream_factory=lambda **kw: RefusingStream(**kw))

    assert ChunkedSpeaker(player, lambda: FakeSource()).run(["x"]) is SpeakOutcome.PLAYBACK_FAILED


def test_pause_goes_silent_and_resume_replays_the_heard_chunk_from_its_start() -> None:
    """Pause goes silent and resume replays the heard chunk from its start."""
    player, recorder = _manual_player()
    source = FakeSource()
    speaker = ChunkedSpeaker(player, lambda: source, fade_seconds=0.0)
    thread, outcomes = _start(speaker, ["one", "two"])
    wait_until(lambda: speaker.queued_clips == 2)
    manual = cast(ManualStream, recorder.streams[-1])
    np.testing.assert_allclose(manual.pull(50), 0.1, rtol=1e-6)

    speaker.pause()
    assert not manual.pull(40).any()
    speaker.resume()
    np.testing.assert_allclose(manual.pull(100), 0.1, rtol=1e-6)  # replayed from its start
    np.testing.assert_allclose(manual.pull(100), 0.2, rtol=1e-6)

    thread.join(timeout=5.0)
    assert outcomes == [SpeakOutcome.DONE]
    assert source.calls == ["one", "two"]  # the pause did not re-make anything


def test_pause_before_the_stream_opens_starts_the_reading_paused() -> None:
    """Pause before the stream opens starts the reading paused."""
    player, recorder = _manual_player()
    source = FakeSource()
    speaker = ChunkedSpeaker(player, lambda: source, fade_seconds=0.0)
    speaker.pause()
    thread, outcomes = _start(speaker, ["one"])
    wait_until(lambda: speaker.queued_clips == 1)
    manual = cast(ManualStream, recorder.streams[-1])

    assert not manual.pull(40).any()  # paused from the very first sound
    speaker.resume()
    np.testing.assert_allclose(manual.pull(100), 0.1, rtol=1e-6)
    thread.join(timeout=5.0)
    assert outcomes == [SpeakOutcome.DONE]


def test_the_voice_is_released_only_after_playback_has_ended() -> None:
    """The voice is released only after playback has ended."""
    player, recorder = _manual_player()
    source = FakeSource(length=100)
    speaker = ChunkedSpeaker(player, lambda: source, fade_seconds=0.0)
    thread, outcomes = _start(speaker, ["one", "two"])
    wait_until(lambda: speaker.queued_clips == 2)  # everything is MADE, nothing has been HEARD

    assert source.closed == 0  # synthesis is over, but the voice must stay loaded

    cast(ManualStream, recorder.streams[-1]).pull(200)
    thread.join(timeout=5.0)
    assert outcomes == [SpeakOutcome.DONE]
    assert source.closed == 1


def test_the_speaking_pace_is_applied_to_each_chunk() -> None:
    """The speaking pace is applied to each chunk."""
    player, recorder = _manual_player()
    source = FakeSource(length=8000)
    speaker = ChunkedSpeaker(player, lambda: source, speed_provider=lambda: 1.6, fade_seconds=0.0)
    thread, outcomes = _start(speaker, ["one"])
    wait_until(lambda: speaker.queued_clips == 1)
    manual = cast(ManualStream, recorder.streams[-1])

    out = manual.pull(9000)
    thread.join(timeout=5.0)

    assert outcomes == [SpeakOutcome.DONE]
    # 8000 samples sped up by the maximum pace (1.6x) is about 5000 samples, clearly shorter.
    assert 3500 < np.count_nonzero(out) < 6500
