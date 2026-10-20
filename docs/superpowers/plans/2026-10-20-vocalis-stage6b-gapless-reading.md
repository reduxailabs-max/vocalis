# Stage 6b — Gapless Long Reading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make long readings (Document Reader, and now Quick Reader too) play as one continuous, gap-free stream: the next piece is always being made while the current one plays, the sound never restarts between pieces, and the voice stays loaded until the reading has really finished playing.

**Architecture:** A new `SpeechStream` keeps ONE sound-card stream open for a whole reading and feeds it from a small bounded queue of clips, with a tiny fade at every join and silence (not a click) if the queue ever runs dry. A new `ChunkedSpeaker` runs a producer loop that makes each piece with the voice chain (holding the voice loaded for the entire reading) and pushes it into the stream; both readers drive it and keep only their own bookkeeping (clipboard / file extraction, notifications, stop/pause state, status board). The chunker becomes sentence-aware and reports exact character positions.

**Tech Stack:** Python 3.13, numpy, sounddevice (faked in tests), pytest, strict mypy. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-11-vocalis-design.md` — section "Stage 6 … 6b — Gapless long reading" (design committed `acd4818`). 6a is merged on `main` at `b6bd8eb`.

## Global Constraints

- Tests ≥ 85% coverage (`--cov-fail-under=85` already in `pyproject.toml`); `mypy` strict clean on `src` and `tests`; zero warnings in the suite (the two known `tests/test_offline_guard.py` failures are pre-existing and out of scope).
- All hardware stays faked in tests — "nothing here changes the project's standing rule that automated tests never touch real hardware". Real audio is only checked by the user at the end (memory: the user installs and tests only after all stages are done).
- Every function and class has a docstring; structured error handling with logging; no hardcoded secrets; `old/` is never modified; docs in `docs/` stay in sync with the code; conventional-commit messages; changelog at `.claude/<timestamp>_CHANGELOG.md`.
- Pause / resume / stop keep their current user-visible behaviour: pause stops the sound at once, resume replays the current piece from its start, stop ends the reading, one reading at a time via `ReadingCoordinator`.
- "The idle-unload timer must not fire while a reading is actively in progress, and should start counting from when playback ends, not when synthesis ends" (spec 6b, verbatim).
- Never hold a lock across a call into blocking or unknown code (the hard-won rule from Phase 2 Task 9 and Phase 3 Task 6): locks only guard a flag or reference swap.
- Process rule from the user (memory `vocalis-implement-fully-then-test`): for every task, **write all the code first (implementation AND tests), then run the tests once, then fix, then commit** — no run-a-test-after-every-edit cycling.
- Process rule (memory `vocalis-no-concurrent-worktree-tests`): never run pytest/mypy in a worktree while another agent may be running tests in it.
- New dependencies: none. After any `pip install` (there should be none) re-check `onnxruntime.get_available_providers()` still lists `CUDAExecutionProvider`.

## Rulings made while writing this plan (the spec is silent or ambiguous)

1. **Where the persistent stream lives.** `AudioPlayer` keeps all the output-device logic (device chooser, retry, failure hook). It gains a tiny public API (`open_stream` / `close_stream` / `report_device_failed`) and the new `SpeechStream` is built on it, so there is exactly one place that talks to `sounddevice`. The old `play()` / `wait_until_done()` / `stop()` stay (OCR/other callers and ~600 lines of existing playback tests keep working).
2. **Sample-rate changes mid-reading** (e.g. Qwen3 24 kHz → Piper 22.05 kHz after a failure): the stream opens at the first clip's rate; later clips at another rate are linearly resampled so the voice never plays at the wrong speed/pitch.
3. **Fade length** 8 ms at both ends of every clip — long enough to remove clicks, short enough to be inaudible inside natural sentence pauses.
4. **Back-pressure** bound is 3 clips (one playing + two waiting). More would just burn CPU/GPU heat ahead of the listener.
5. **Stall detection.** A reading whose sound device never calls back (the real `~/.asoundrc` failure found in Phase 3) fails with "Playback failed" after 10 s without a single audio callback while clips are waiting. This replaces the old "expected duration + 10 s" timeout, which cannot work for a stream that stays open.
6. **Progress** ("piece N of M") is now reported when a piece *starts being heard*, not when it starts being made — it matches what the user hears. The first piece's report therefore arrives when the first sound starts (after the model has loaded).
7. **Voice held while paused.** The voice engine stays loaded for the whole reading including a long pause (spec: never unload mid-reading). It is released — and the idle timer starts — when the reading ends.
8. **Another feature taking the GPU mid-reading** (e.g. Dictation loads Parakeet): the voice chain re-acquires its engine before every piece (cheap when still loaded), so a stolen model is simply reloaded, exactly like today, rather than crashing the reading.
9. **Quick Reader notification order.** "Reading..." is sent first; the copied text is sent from a one-off side thread so it can never delay the sound; "Done" waits (≤ 6 s) for that thread so the order the user sees never changes.
10. **Chunker newline rule.** A blank line is always a break. A single newline is a break unless the line before it looks hard-wrapped (≥ 70 characters, not ending in a full stop/question/exclamation mark, and not ending in `,` `;` `-`). Short lines without punctuation (headings, list items) break. A single capital letter followed by a full stop ("J. Smith", "U.S.") never breaks — a missed break only means two sentences share a chunk, while a false break after "Dr." is audible.
11. **Spans are into the text handed to the chunker** (the normalized text for Document Reader). Mapping back to the original file for Stage 7 highlighting is Stage 7's job.
12. **Obsolete tests are deleted, not ported,** where they only exercised the old per-chunk `play()` windows (list in Task 6); their *intent* is re-covered by the new `SpeechStream` / `ChunkedSpeaker` tests.

## Review Focus

Failure modes the spec implies but no obvious task test would exercise on its own; each is pinned by a named test in the owning task.

1. **Stop or pause pressed before any sound exists** (while the model is still loading or the first piece is still being made) must still stop/pause the reading, and the late-finishing producer must not start playing afterwards. → Task 5 (`test_stop_before_run_returns_stopped_without_opening_a_stream`, `test_pause_before_the_stream_opens_starts_the_reading_paused`), Task 6, Task 7.
2. **Sound device that never calls back** must end the reading with "Playback failed" in about 10 s, never hang, and must tell the output-device chooser which device failed. → Task 3 (`test_a_stalled_device_raises_and_reports_the_device`), Task 5, Task 6.
3. **Voice switch mid-document changes the sample rate** and must not play at the wrong pitch. → Task 3 (`test_a_clip_at_another_sample_rate_is_resampled`).
4. **Messy text:** abbreviations ("Dr.", "e.g."), initials, decimals, hard-wrapped lines, headings without punctuation, blank-line paragraphs, one enormous unpunctuated line. → Task 1.
5. **Another feature loads its own model mid-reading** (Dictation → Parakeet unloads the held voice): the next piece must reload the voice or fall down the chain, never crash or stay silent. → Task 4 (`test_a_held_engine_taken_by_another_feature_is_reloaded_for_the_next_piece`).
6. **Empty audio from a piece** (e.g. a punctuation-only piece) must not stall progress or the queue. → Task 3 (`test_an_empty_clip_is_accepted_but_not_queued`).
7. **Stop while the producer is blocked** because the queue is full must wake it and end cleanly. → Task 3, Task 5.

---

## Task 1: Sentence-aware chunker with exact character spans

**Files:**
- Modify: `src/vocalis/text/chunker.py` (full rewrite, ~120 lines)
- Modify: `tests/test_chunker.py` (keep every existing test, append new ones)

**Interfaces:**
- Produces: `Chunk` (frozen dataclass: `text: str`, `start: int`, `end: int`); `chunk_spans(text: str, max_chars: int = DEFAULT_MAX_CHUNK_CHARS) -> list[Chunk]`; `chunk_text(text, max_chars=...) -> list[str]` (unchanged signature and contract, now `[c.text for c in chunk_spans(...)]`); `DEFAULT_MAX_CHUNK_CHARS = 300`.
- `Chunk.text` is the chunk's words joined by single spaces; `text_in[start:end]` is the exact slice of the input it came from (no leading/trailing whitespace).

- [ ] **Step 1: Write the implementation** — replace the whole of `src/vocalis/text/chunker.py` with:

```python
"""Splits normalized text into sentence-bounded chunks for streaming
synthesis and playback, so a reading can start speaking before the whole
document has been processed and can show "piece N of M" progress.

Sentence breaks are found by punctuation (. ! ? followed by whitespace) and
by line breaks, but never after a known abbreviation ("Dr.", "e.g."), after
an initial ("J. Smith", "U.S."), or before a lowercase word ("etc. and").
A blank line is always a break. A single line break is a break unless the
line before it looks hard-wrapped (long, with no punctuation at its end).

A sentence is never split mid-word: a single sentence longer than
max_chars becomes its own (over-limit) chunk rather than being cut off,
since a TTS engine reading a word fragment sounds far worse than reading
one long chunk (the voice engines split very long text themselves).

Every chunk also carries the exact (start, end) character positions of the
text it came from, which a later stage uses to highlight the line being read.
"""
import re
from dataclasses import dataclass

DEFAULT_MAX_CHUNK_CHARS = 300
_HARD_WRAP_MIN_LINE_CHARS = 70

_BOUNDARY_RE = re.compile(r"[.!?][\"'”’)\]]*(?=\s)|\n")
_WORD_BEFORE_RE = re.compile(r"([A-Za-z][A-Za-z.]*)$")
_INITIALS_RE = re.compile(r"(?:[A-Za-z]\.)*[A-Za-z]")
_ABBREVIATIONS = frozenset(
    {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e",
        "inc", "ltd", "co", "no", "fig", "approx", "dept", "est", "mt",
    }
)


@dataclass(frozen=True)
class Chunk:
    """One piece of text to speak, plus where it came from in the input."""

    text: str
    start: int
    end: int


def _period_breaks(text: str, mark_index: int, match_end: int) -> bool:
    """Whether the sentence-ending mark at `mark_index` really ends a sentence."""
    if text[mark_index] != ".":
        return True
    word = _WORD_BEFORE_RE.search(text[:mark_index])
    if word is not None:
        token = word.group(1)
        if token.lower() in _ABBREVIATIONS or _INITIALS_RE.fullmatch(token):
            return False
    following = text[match_end:].lstrip()
    return not following[:1].islower()


def _newline_breaks(text: str, index: int) -> bool:
    """Whether the line break at `index` ends a sentence (see module docstring)."""
    after = text[index + 1 :]
    if not after.strip() or after.lstrip(" \t").startswith("\n"):
        return True
    line_start = text.rfind("\n", 0, index) + 1
    line = text[line_start:index].strip()
    if not line or line[-1] in ".!?":
        return True
    if line[-1] in ",;-":
        return False
    return len(line) < _HARD_WRAP_MIN_LINE_CHARS


def _sentence_ends(text: str) -> list[int]:
    """Sorted, de-duplicated offsets where a sentence ends."""
    ends: set[int] = set()
    for match in _BOUNDARY_RE.finditer(text):
        if match.group() == "\n":
            if _newline_breaks(text, match.start()):
                ends.add(match.start())
        elif _period_breaks(text, match.start(), match.end()):
            ends.add(match.end())
    return sorted(ends)


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) of every non-empty sentence, whitespace trimmed."""
    spans: list[tuple[int, int]] = []
    begin = 0
    for end in [*_sentence_ends(text), len(text)]:
        piece = text[begin:end]
        stripped = piece.strip()
        if stripped:
            lead = len(piece) - len(piece.lstrip())
            spans.append((begin + lead, begin + lead + len(stripped)))
        begin = end
    return spans


def _make_chunk(text: str, start: int, end: int) -> Chunk:
    """Build a Chunk whose text is the slice's words joined by single spaces."""
    return Chunk(" ".join(text[start:end].split()), start, end)


def chunk_spans(text: str, max_chars: int = DEFAULT_MAX_CHUNK_CHARS) -> list[Chunk]:
    """Split text into sentence-bounded chunks no longer than max_chars.

    Args:
        text: Normalized, speakable text (see normalizer.normalize_markdown).
        max_chars: Soft maximum characters per chunk. A single sentence
            longer than this is still returned whole, as its own chunk.

    Returns:
        Chunks in reading order, each with the exact (start, end) positions
        of the input it covers. Empty or whitespace-only input gives [].
    """
    chunks: list[Chunk] = []
    current_start = current_end = -1
    current_len = 0
    for start, end in _sentence_spans(text):
        sentence_len = len(" ".join(text[start:end].split()))
        if current_start < 0:
            current_start, current_end, current_len = start, end, sentence_len
        elif current_len + 1 + sentence_len > max_chars:
            chunks.append(_make_chunk(text, current_start, current_end))
            current_start, current_end, current_len = start, end, sentence_len
        else:
            current_end = end
            current_len += 1 + sentence_len
    if current_start >= 0:
        chunks.append(_make_chunk(text, current_start, current_end))
    return chunks


def chunk_text(text: str, max_chars: int = DEFAULT_MAX_CHUNK_CHARS) -> list[str]:
    """Split text into sentence-bounded chunk strings (see chunk_spans)."""
    return [chunk.text for chunk in chunk_spans(text, max_chars)]
```

- [ ] **Step 2: Write the tests** — keep every existing test in `tests/test_chunker.py`; add `from vocalis.text.chunker import Chunk, chunk_spans, chunk_text` to its imports (adjust the existing import line) and append:

```python
def test_a_known_abbreviation_does_not_end_a_sentence() -> None:
    chunks = chunk_text("Dr. Smith arrived. Mr. Jones left.", max_chars=25)
    assert chunks == ["Dr. Smith arrived.", "Mr. Jones left."]


def test_e_g_does_not_end_a_sentence() -> None:
    chunks = chunk_text("Use tools, e.g. a hammer. Then rest.", max_chars=20)
    assert chunks == ["Use tools, e.g. a hammer.", "Then rest."]


def test_an_initial_does_not_end_a_sentence() -> None:
    chunks = chunk_text("Ask J. Smith about it. Then go.", max_chars=24)
    assert chunks == ["Ask J. Smith about it.", "Then go."]


def test_a_lowercase_word_after_a_full_stop_keeps_the_sentence_going() -> None:
    chunks = chunk_text("We sell pens, etc. and paper. Bye.", max_chars=30)
    assert chunks == ["We sell pens, etc. and paper.", "Bye."]


def test_a_decimal_number_is_not_a_sentence_end() -> None:
    assert chunk_text("Pi is 3.14 today. Done.", max_chars=300) == ["Pi is 3.14 today. Done."]


def test_a_short_line_without_punctuation_is_its_own_sentence() -> None:
    text = "Chapter One\nIt was dark. It rained.\nThe end"
    assert chunk_text(text, max_chars=20) == ["Chapter One", "It was dark.", "It rained. The end"]


def test_a_hard_wrapped_long_line_is_joined_to_the_next_line() -> None:
    long_line = " ".join(["word"] * 15)  # 74 characters, no punctuation
    chunks = chunk_text(f"{long_line}\nmore words here.", max_chars=80)
    assert len(chunks) == 1
    assert "\n" not in chunks[0]


def test_a_blank_line_always_breaks_even_after_a_long_line() -> None:
    long_line = " ".join(["word"] * 16)  # 79 characters
    chunks = chunk_text(f"{long_line}\n\nNext.", max_chars=82)
    assert len(chunks) == 2
    assert chunks[1] == "Next."


def test_a_line_ending_in_a_comma_is_not_a_break() -> None:
    chunks = chunk_text("First part,\nsecond part.", max_chars=20)
    assert chunks == ["First part, second part."]


def test_one_enormous_unpunctuated_line_stays_one_chunk() -> None:
    text = " ".join(["word"] * 2000)
    chunks = chunk_text(text, max_chars=300)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_spans_point_at_the_exact_source_text() -> None:
    text = "  Hello there.  General Kenobi.\n"
    chunks = chunk_spans(text, max_chars=15)
    assert [text[c.start : c.end] for c in chunks] == ["Hello there.", "General Kenobi."]
    assert all(isinstance(c, Chunk) for c in chunks)


def test_a_chunk_that_joins_two_sentences_spans_both() -> None:
    text = "One. Two.\nThree."
    chunks = chunk_spans(text, max_chars=300)
    assert len(chunks) == 1
    assert (chunks[0].start, chunks[0].end) == (0, len(text))
    assert chunks[0].text == "One. Two. Three."


def test_chunk_text_matches_chunk_spans_text() -> None:
    text = "A first sentence. A second one! And a third?"
    assert chunk_text(text, max_chars=20) == [c.text for c in chunk_spans(text, max_chars=20)]
```

- [ ] **Step 3: Run once and fix.** Run `.venv/bin/python -m pytest tests/test_chunker.py -q --no-cov`, then `.venv/bin/mypy src tests`. Expected: all pass, mypy clean. (If a new test's expected list differs from the code, first re-derive by hand which one is right; the expectations above were hand-traced against the code.)

- [ ] **Step 4: Commit**

```bash
git add src/vocalis/text/chunker.py tests/test_chunker.py
git commit -m "feat: make the chunker sentence-aware and report exact character spans"
```

---

## Task 2: AudioPlayer stream API and the shared fake-output test helper

**Files:**
- Modify: `src/vocalis/audio/playback.py` (add three public methods after `play()`; no existing code changes)
- Create: `tests/fake_output.py`
- Modify: `tests/test_playback.py` (append tests; keep all existing)

**Interfaces:**
- Produces on `AudioPlayer`:
  - `open_stream(self, sample_rate: int, callback: Callable[..., None], finished_callback: Callable[[], None]) -> None` — opens and starts an output stream driven by the caller's `callback(outdata, frames, time_info, status)` using the same device-provider/retry/failure-hook logic as `play()`; raises `PlaybackAlreadyActiveError` if a stream is already open.
  - `close_stream(self) -> None` — stops and closes the stream opened by `open_stream`; no-op if none; never raises; does NOT set the "stopped early" flag.
  - `report_device_failed(self) -> None` — tells the `on_device_failed` hook which device was in use (never raises).
- Produces in `tests/fake_output.py`: `OutputRecorder`, `ManualStream`, `PumpStream`, `manual_stream_factory(recorder)`, `pump_stream_factory(recorder, speedup=50.0)`, `FakeAudioPlayer`, `wait_until(predicate, timeout=5.0)`. Importable as `from tests.fake_output import ...` (the tests folder is already a package).

- [ ] **Step 1: Write the implementation** — in `src/vocalis/audio/playback.py` add these methods to `AudioPlayer`, directly after `play()` and before `_open_stream()`:

```python
    def open_stream(
        self,
        sample_rate: int,
        callback: Callable[..., None],
        finished_callback: Callable[[], None],
    ) -> None:
        """Open and start an output stream that the caller feeds itself.

        Used by SpeechStream for gapless readings: instead of handing over
        one finished clip (play()), the caller supplies the audio callback
        and keeps the stream open across many clips. Output-device choice,
        the one retry, and the device-failure hook behave exactly as for
        play().

        Args:
            sample_rate: Sample rate of the audio the callback will supply.
            callback: sounddevice-style callback(outdata, frames, time_info, status).
            finished_callback: Called by the stream when it finishes.

        Raises:
            PlaybackAlreadyActiveError: If a stream is already open.
        """
        if self.is_playing:
            raise PlaybackAlreadyActiveError("AudioPlayer.open_stream() called while already playing")
        self._done_event.clear()
        self._stopped_early = False
        self._open_stream(sample_rate, callback, finished_callback)

    def close_stream(self) -> None:
        """Stop and close the stream opened by open_stream(); no-op if none.

        Never raises (a failure to close is logged). Unlike stop(), this is
        an ordinary end of a reading, so it does not mark the playback as
        stopped early.
        """
        with self._lock:
            stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.stop()
        except Exception:
            logger.warning("Failed to stop the output stream", exc_info=True)
        try:
            stream.close()
        except Exception:
            logger.warning("Failed to close the output stream", exc_info=True)

    def report_device_failed(self) -> None:
        """Tell the on_device_failed hook which device was in use; never raises."""
        self._report_device_failed()
```

Create `tests/fake_output.py`:

```python
"""Fake audio output for tests — stands in for sounddevice.OutputStream.

Both flavours pull audio through the stream's callback exactly the way a
real sound card does, so code that feeds a persistent stream is tested for
real, with no hardware:

- ManualStream: the TEST drives the audio clock by calling pull(frames).
  Fully deterministic; use it for ordering, fades, pause and stall tests.
- PumpStream: a background thread pulls audio continuously, `speedup`
  times faster than real time. Use it where a whole reading must simply
  play out (feature-level tests).
"""
import threading
import time
from typing import Any, Callable

import numpy as np
import numpy.typing as npt
import sounddevice as sd

from vocalis.audio.playback import AudioPlayer

_PUMP_FRAMES = 256


class OutputRecorder:
    """Everything the fake streams were asked to do, and all audio they pulled."""

    def __init__(self) -> None:
        """Start empty."""
        self._lock = threading.Lock()
        self.opened = 0
        self.closed = 0
        self.closed_event = threading.Event()
        self.kwargs: "list[dict[str, Any]]" = []
        self.streams: "list[Any]" = []
        self._pulled: "list[npt.NDArray[np.float32]]" = []

    def note_created(self, stream: Any, kwargs: "dict[str, Any]") -> None:
        """Remember a newly built stream and the arguments it was built with."""
        with self._lock:
            self.streams.append(stream)
            self.kwargs.append(kwargs)

    def note_started(self) -> None:
        """Count a stream whose start() ran."""
        with self._lock:
            self.opened += 1

    def note_closed(self) -> None:
        """Count a stream whose close() ran (first close of each stream only)."""
        with self._lock:
            self.closed += 1
        self.closed_event.set()

    def add_pulled(self, samples: "npt.NDArray[np.float32]") -> None:
        """Record audio handed to the 'sound card'."""
        with self._lock:
            self._pulled.append(samples)

    def audio(self) -> "npt.NDArray[np.float32]":
        """Everything pulled so far, silence included."""
        with self._lock:
            parts = list(self._pulled)
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

    def played(self) -> "npt.NDArray[np.float32]":
        """Only the non-silent samples pulled so far."""
        audio = self.audio()
        return audio[audio != 0]

    @property
    def frames_pulled(self) -> int:
        """How many frames (silence included) the 'sound card' has pulled."""
        with self._lock:
            return sum(len(part) for part in self._pulled)


class ManualStream:
    """A fake output stream whose audio clock is driven by the test."""

    def __init__(self, recorder: OutputRecorder, **kwargs: Any) -> None:
        """Remember the callback; register with the recorder."""
        self._recorder = recorder
        self._callback = kwargs["callback"]
        self.samplerate: int = kwargs["samplerate"]
        self.running = False
        self.closed = False
        recorder.note_created(self, dict(kwargs))

    def start(self) -> None:
        """Mark the stream running."""
        self.running = True
        self._recorder.note_started()

    def pull(self, frames: int) -> "npt.NDArray[np.float32]":
        """Ask the audio callback for `frames` frames, as the sound card would."""
        buffer = np.zeros((frames, 1), dtype=np.float32)
        self._callback(buffer, frames, None, None)
        samples = buffer[:, 0].copy()
        self._recorder.add_pulled(samples)
        return samples

    def stop(self) -> None:
        """Mark the stream stopped."""
        self.running = False

    def close(self) -> None:
        """Mark the stream closed (counted once)."""
        if not self.closed:
            self.closed = True
            self._recorder.note_closed()


class PumpStream:
    """A fake output stream that pulls audio on its own thread, faster than real time."""

    def __init__(self, recorder: OutputRecorder, speedup: float = 50.0, **kwargs: Any) -> None:
        """Remember the callbacks; register with the recorder."""
        self._recorder = recorder
        self._callback = kwargs["callback"]
        self._finished_callback = kwargs["finished_callback"]
        self.samplerate: int = kwargs["samplerate"]
        self._interval = _PUMP_FRAMES / self.samplerate / speedup
        self._halt = threading.Event()
        self._thread: "threading.Thread | None" = None
        self.closed = False
        recorder.note_created(self, dict(kwargs))

    def start(self) -> None:
        """Start pulling audio on a background thread."""
        self._recorder.note_started()
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        """Pull audio until stopped, or until the callback ends the stream."""
        while not self._halt.is_set():
            buffer = np.zeros((_PUMP_FRAMES, 1), dtype=np.float32)
            try:
                self._callback(buffer, _PUMP_FRAMES, None, None)
            except sd.CallbackStop:
                self._recorder.add_pulled(buffer[:, 0].copy())
                self._finished_callback()
                return
            self._recorder.add_pulled(buffer[:, 0].copy())
            time.sleep(self._interval)

    def stop(self) -> None:
        """Stop pulling audio and wait for the thread (unless called from it)."""
        self._halt.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)

    def close(self) -> None:
        """Stop, then count the close once and tell the finished callback."""
        self.stop()
        if not self.closed:
            self.closed = True
            self._recorder.note_closed()
            self._finished_callback()


def manual_stream_factory(recorder: OutputRecorder) -> "Callable[..., ManualStream]":
    """A stream_factory for AudioPlayer that builds ManualStreams."""

    def factory(**kwargs: Any) -> ManualStream:
        """Build one ManualStream."""
        return ManualStream(recorder, **kwargs)

    return factory


def pump_stream_factory(recorder: OutputRecorder, speedup: float = 50.0) -> "Callable[..., PumpStream]":
    """A stream_factory for AudioPlayer that builds PumpStreams."""

    def factory(**kwargs: Any) -> PumpStream:
        """Build one PumpStream."""
        return PumpStream(recorder, speedup=speedup, **kwargs)

    return factory


class FakeAudioPlayer(AudioPlayer):
    """A real AudioPlayer wired to a PumpStream; `output` records what it played.

    Drop-in for tests that patch `vocalis.cli.AudioPlayer`. `done_event` is
    set the first time a stream is closed (i.e. a reading finished or stopped).
    """

    def __init__(self, speedup: float = 50.0) -> None:
        """Create the recorder and wire the player to it."""
        self.output = OutputRecorder()
        super().__init__(stream_factory=pump_stream_factory(self.output, speedup))

    @property
    def done_event(self) -> threading.Event:
        """Set once the first stream has been closed."""
        return self.output.closed_event


def wait_until(predicate: "Callable[[], bool]", timeout: float = 5.0) -> None:
    """Block until `predicate()` is true; raise AssertionError after `timeout` seconds."""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError(f"condition not met within {timeout} s")
        time.sleep(0.005)
```

- [ ] **Step 2: Write the tests** — append to `tests/test_playback.py` (add the imports at the top of the file; reuse the existing `AudioPlayer`, `PlaybackAlreadyActiveError` imports if present):

```python
from typing import Any

import pytest

from tests.fake_output import OutputRecorder, manual_stream_factory


def test_open_stream_builds_one_stream_with_the_callers_callback() -> None:
    recorder = OutputRecorder()
    player = AudioPlayer(stream_factory=manual_stream_factory(recorder))

    player.open_stream(24000, lambda *args: None, lambda: None)

    assert recorder.opened == 1
    assert recorder.kwargs[0]["samplerate"] == 24000
    assert recorder.kwargs[0]["channels"] == 1
    assert player.is_playing is True


def test_open_stream_while_a_stream_is_open_raises() -> None:
    recorder = OutputRecorder()
    player = AudioPlayer(stream_factory=manual_stream_factory(recorder))
    player.open_stream(24000, lambda *args: None, lambda: None)

    with pytest.raises(PlaybackAlreadyActiveError):
        player.open_stream(24000, lambda *args: None, lambda: None)


def test_close_stream_stops_and_closes_and_is_a_noop_when_nothing_is_open() -> None:
    recorder = OutputRecorder()
    player = AudioPlayer(stream_factory=manual_stream_factory(recorder))
    player.close_stream()  # nothing open: must not raise
    player.open_stream(24000, lambda *args: None, lambda: None)

    player.close_stream()
    player.close_stream()

    assert recorder.closed == 1
    assert player.is_playing is False


def test_close_stream_swallows_errors_from_a_misbehaving_stream() -> None:
    class BadStream:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def start(self) -> None:
            pass

        def stop(self) -> None:
            raise RuntimeError("stop failed")

        def close(self) -> None:
            raise RuntimeError("close failed")

    player = AudioPlayer(stream_factory=lambda **kwargs: BadStream(**kwargs))
    player.open_stream(24000, lambda *args: None, lambda: None)

    player.close_stream()  # must not raise

    assert player.is_playing is False


def test_report_device_failed_tells_the_hook_which_device_was_in_use() -> None:
    recorder = OutputRecorder()
    failures: "list[int | None]" = []
    player = AudioPlayer(
        stream_factory=manual_stream_factory(recorder),
        device_provider=lambda: 3,
        on_device_failed=failures.append,
    )
    player.open_stream(24000, lambda *args: None, lambda: None)

    player.report_device_failed()

    assert failures == [3]
    assert recorder.kwargs[0]["device"] == 3
```

- [ ] **Step 3: Run once and fix.** `.venv/bin/python -m pytest tests/test_playback.py -q --no-cov` then `.venv/bin/mypy src tests`. Expected: pass / clean. (`from tests.fake_output import ...` must import; if `tests` is not importable as a package from the repo root, the other test files already do `from tests.test_doc_reader import ...` so it is.)

- [ ] **Step 4: Commit**

```bash
git add src/vocalis/audio/playback.py tests/fake_output.py tests/test_playback.py
git commit -m "feat: let AudioPlayer open a caller-fed stream, and add a shared fake sound output for tests"
```

---

## Task 3: SpeechStream — one persistent, gapless output stream

**Files:**
- Create: `src/vocalis/audio/speech_stream.py`
- Create: `tests/test_speech_stream.py`

**Interfaces:**
- Consumes: `AudioPlayer.open_stream / close_stream / report_device_failed` (Task 2); `PlaybackTimeoutError` from `vocalis.audio.playback`; test helpers from `tests/fake_output.py`.
- Produces (`vocalis.audio.speech_stream`):
  - constants `DEFAULT_FADE_SECONDS = 0.008`, `DEFAULT_STALL_SECONDS = 10.0`, `DEFAULT_MAX_QUEUED_CLIPS = 3`
  - `SpeechStream(player: AudioPlayer, on_clip_started: Callable[[int], None] | None = None, fade_seconds: float = DEFAULT_FADE_SECONDS, stall_seconds: float = DEFAULT_STALL_SECONDS, max_queued_clips: int = DEFAULT_MAX_QUEUED_CLIPS, clock: Callable[[], float] = time.monotonic)`
  - `start(sample_rate: int) -> None` (no-op after `stop()`; raises whatever the player raises on a device failure)
  - `push(audio: npt.NDArray[np.float32], sample_rate: int, tag: int) -> bool` — blocks while the queue is full; `True` = accepted; `False` = the stream was stopped/closed; raises `PlaybackTimeoutError` on a stall. `tag` is handed back to `on_clip_started` when that clip begins to be heard.
  - `finish() -> None` (no more clips will come)
  - `wait_until_done() -> bool` — `True` = every clip played; `False` = stopped; raises `PlaybackTimeoutError` on a stall. Closes the stream before returning.
  - `pause() -> None` (silence at once; the current clip is rewound to its start), `resume() -> None`, `stop() -> None` (idempotent, wakes any blocked caller)
  - properties `underruns: int`, `queued_clips: int`

- [ ] **Step 1: Write the implementation** — create `src/vocalis/audio/speech_stream.py`:

```python
"""One persistent, gapless speech output for a whole reading.

Opening a new sound stream for every piece (the old way) leaves a click or
gap at every join and makes long readings stutter. SpeechStream opens ONE
stream through AudioPlayer and keeps it open: pieces of speech (clips) are
pushed into a small bounded queue and the audio callback plays them
back-to-back. Three details make the joins inaudible:

- every clip gets a tiny fade-in and fade-out (no click at a join);
- if the next clip is not ready in time the callback plays silence instead
  of stopping (an "underrun" is a short pause, never a restart);
- the queue is bounded, so the producer gets ahead of the listener by a
  few pieces only and then waits (back-pressure), instead of racing ahead
  and heating the machine.

Pause outputs silence immediately and rewinds the clip being heard to its
start, so resume replays that piece from the beginning (the long-standing
behaviour). Clips at a different sample rate than the stream (the voice
changed mid-reading) are resampled, never played at the wrong speed.

Stall safety: if clips are waiting but the sound device has not called back
for `stall_seconds`, the device is broken (the stream opened but never
runs). push()/wait_until_done() then fail with PlaybackTimeoutError and the
device is reported to AudioPlayer's failure hook, instead of hanging.

Lock discipline: the one condition lock only guards the clip queue and a few
flags. Anything slow (opening/closing the device, user callbacks) happens
outside it; the audio callback only copies samples and does a non-blocking
put on a SimpleQueue that a helper thread drains.
"""
import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, NoReturn

import numpy as np
import numpy.typing as npt

from vocalis.audio.playback import AudioPlayer, PlaybackTimeoutError

logger = logging.getLogger("vocalis")

DEFAULT_FADE_SECONDS = 0.008
DEFAULT_STALL_SECONDS = 10.0
DEFAULT_MAX_QUEUED_CLIPS = 3
_WAIT_SLICE_SECONDS = 0.1


def _ignore_finished() -> None:
    """The stream's finished callback: nothing to do, the stream is closed explicitly."""


@dataclass
class _Clip:
    """One queued piece of audio and how far it has been played."""

    tag: int
    samples: npt.NDArray[np.float32]
    position: int = 0
    announced: bool = False


class SpeechStream:
    """Plays queued clips back-to-back on one open output stream."""

    def __init__(
        self,
        player: AudioPlayer,
        on_clip_started: Callable[[int], None] | None = None,
        fade_seconds: float = DEFAULT_FADE_SECONDS,
        stall_seconds: float = DEFAULT_STALL_SECONDS,
        max_queued_clips: int = DEFAULT_MAX_QUEUED_CLIPS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create a SpeechStream (does not open the device yet).

        Args:
            player: Opens/closes the real output stream and knows the device logic.
            on_clip_started: Called (on a helper thread, never the audio
                thread) with a clip's tag when that clip starts to be heard.
            fade_seconds: Length of the fade at each end of every clip.
            stall_seconds: How long the device may stay silent (no callback)
                while clips are waiting before the stream is declared stalled.
            max_queued_clips: Most clips held at once (playing + waiting);
                push() waits when the queue is full.
            clock: Monotonic clock; injectable for tests.
        """
        self._player = player
        self._on_clip_started = on_clip_started
        self._fade_seconds = fade_seconds
        self._stall_seconds = stall_seconds
        self._max_queued_clips = max_queued_clips
        self._clock = clock
        self._cond = threading.Condition()
        self._clips: "deque[_Clip]" = deque()
        self._sample_rate = 0
        self._open = False
        self._finishing = False
        self._paused = False
        self._stopped = False
        self._shut = False
        self._last_callback = 0.0
        self._underruns = 0
        self._played_any = False
        self._started_tags: "queue.SimpleQueue[int | None]" = queue.SimpleQueue()

    @property
    def underruns(self) -> int:
        """How many audio callbacks ran dry while more audio was still expected."""
        with self._cond:
            return self._underruns

    @property
    def queued_clips(self) -> int:
        """Clips currently held (the one being heard plus those waiting)."""
        with self._cond:
            return len(self._clips)

    def start(self, sample_rate: int) -> None:
        """Open the output stream at `sample_rate`. No-op once stopped.

        Raises:
            Exception: Whatever AudioPlayer.open_stream raises (a device failure).
        """
        with self._cond:
            if self._stopped or self._open:
                return
            self._sample_rate = sample_rate
            self._open = True
            self._last_callback = self._clock()
        try:
            self._player.open_stream(sample_rate, self._callback, _ignore_finished)
        except Exception:
            with self._cond:
                self._open = False
            raise
        with self._cond:
            stopped = self._stopped
        if stopped:
            # stop() ran while the device was opening; nobody else will close it.
            self._player.close_stream()
            return
        if self._on_clip_started is not None:
            threading.Thread(target=self._dispatch_started, daemon=True).start()

    def push(self, audio: "npt.NDArray[np.float32]", sample_rate: int, tag: int) -> bool:
        """Queue one clip, waiting while the queue is full.

        Args:
            audio: Mono float32 samples.
            sample_rate: Sample rate of `audio`; resampled if it differs from the stream's.
            tag: Handed to on_clip_started when this clip starts to be heard.

        Returns:
            True if accepted (an empty clip is accepted but not queued);
            False if the stream was stopped or never opened.

        Raises:
            PlaybackTimeoutError: The sound device stopped calling back.
        """
        if not self._open and not self._stopped:
            return False
        samples = self._prepared(audio, sample_rate)
        stalled = False
        with self._cond:
            while True:
                if self._stopped or not self._open:
                    return False
                if len(self._clips) < self._max_queued_clips:
                    break
                if self._stalled_locked():
                    stalled = True
                    break
                self._cond.wait(_WAIT_SLICE_SECONDS)
            if not stalled:
                if samples.size:
                    self._clips.append(_Clip(tag, samples))
                self._cond.notify_all()
                return True
        self._raise_stalled()

    def finish(self) -> None:
        """Say that no more clips will be pushed."""
        with self._cond:
            self._finishing = True
            self._cond.notify_all()

    def wait_until_done(self) -> bool:
        """Wait until every queued clip has been played, then close the stream.

        Returns:
            True if everything played; False if the stream was stopped first.

        Raises:
            PlaybackTimeoutError: The sound device stopped calling back.
        """
        stalled = False
        with self._cond:
            while True:
                if self._stopped:
                    finished = False
                    break
                if self._finishing and not self._clips:
                    finished = True
                    break
                if self._stalled_locked():
                    stalled = True
                    finished = False
                    break
                self._cond.wait(_WAIT_SLICE_SECONDS)
        if stalled:
            self._raise_stalled()
        self._shutdown()
        return finished

    def pause(self) -> None:
        """Go silent at once and rewind the clip being heard to its start."""
        with self._cond:
            self._paused = True
            if self._clips:
                self._clips[0].position = 0
            self._cond.notify_all()

    def resume(self) -> None:
        """Carry on (the rewound clip plays again from its start)."""
        with self._cond:
            self._paused = False
            self._cond.notify_all()

    def stop(self) -> None:
        """End the reading now: drop queued audio, wake blocked callers, close the device."""
        with self._cond:
            self._stopped = True
            self._clips.clear()
            self._cond.notify_all()
        self._shutdown()

    def _prepared(self, audio: "npt.NDArray[np.float32]", sample_rate: int) -> "npt.NDArray[np.float32]":
        """Resample to the stream's rate if needed, then fade both ends."""
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if samples.size and self._sample_rate and sample_rate > 0 and sample_rate != self._sample_rate:
            target = max(1, int(round(samples.size * self._sample_rate / sample_rate)))
            samples = np.interp(
                np.linspace(0, samples.size - 1, target), np.arange(samples.size), samples
            ).astype(np.float32)
        fade = min(int(self._sample_rate * self._fade_seconds), samples.size // 2)
        if fade > 1:
            samples = samples.copy()
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            samples[:fade] *= ramp
            samples[-fade:] *= ramp[::-1]
        return samples

    def _stalled_locked(self) -> bool:
        """Whether clips are waiting but the device has gone quiet. Caller holds the lock."""
        return bool(self._clips) and self._clock() - self._last_callback > self._stall_seconds

    def _raise_stalled(self) -> NoReturn:
        """Stop everything, blame the device, and raise PlaybackTimeoutError."""
        with self._cond:
            self._stopped = True
            self._clips.clear()
            self._cond.notify_all()
        self._player.report_device_failed()
        self._shutdown()
        logger.error(
            "Speech output stalled: the sound device made no callback for over %.1fs "
            "while audio was waiting (usually a misconfigured default sound device).",
            self._stall_seconds,
        )
        raise PlaybackTimeoutError(
            f"The sound device did not play anything for over {self._stall_seconds:.0f}s."
        )

    def _shutdown(self) -> None:
        """Close the device and the helper thread once (safe to call many times)."""
        with self._cond:
            if self._shut:
                return
            self._shut = True
            self._open = False
            self._cond.notify_all()
        self._player.close_stream()
        self._started_tags.put(None)

    def _dispatch_started(self) -> None:
        """Helper thread: deliver clip-started tags outside the audio thread."""
        while True:
            tag = self._started_tags.get()
            if tag is None:
                return
            hook = self._on_clip_started
            if hook is None:
                continue
            try:
                hook(tag)
            except Exception:
                logger.exception("on_clip_started hook failed")

    def _callback(
        self, outdata: "npt.NDArray[np.float32]", frames: int, time_info: object, status: object
    ) -> None:
        """Audio thread: copy queued samples into `outdata`, silence when dry."""
        if status:
            logger.warning("Audio output status: %s", status)
        out = outdata[:, 0]
        with self._cond:
            self._last_callback = self._clock()
            if self._paused or self._stopped:
                out[:] = 0
                return
            filled = 0
            while filled < frames and self._clips:
                clip = self._clips[0]
                take = min(frames - filled, len(clip.samples) - clip.position)
                if take > 0:
                    out[filled : filled + take] = clip.samples[clip.position : clip.position + take]
                    if not clip.announced:
                        clip.announced = True
                        self._played_any = True
                        self._started_tags.put(clip.tag)
                    clip.position += take
                    filled += take
                if clip.position >= len(clip.samples):
                    self._clips.popleft()
                    self._cond.notify_all()
            if filled < frames:
                out[filled:] = 0
                if self._played_any and not self._finishing:
                    self._underruns += 1
```

- [ ] **Step 2: Write the tests** — create `tests/test_speech_stream.py`:

```python
"""Tests for vocalis.audio.speech_stream.SpeechStream.

Most tests use ManualStream (the test pulls audio itself), so ordering,
fades, pause, underrun and stall behaviour are fully deterministic.
"""
import threading
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
```

- [ ] **Step 3: Run once and fix.** `.venv/bin/python -m pytest tests/test_speech_stream.py -q --no-cov` then `.venv/bin/mypy src tests`. Expected: pass / clean.

- [ ] **Step 4: Commit**

```bash
git add src/vocalis/audio/speech_stream.py tests/test_speech_stream.py
git commit -m "feat: add SpeechStream, one persistent gapless output stream with fades and back-pressure"
```

---

## Task 4: EngineChain keeps the voice loaded for a whole reading; shared chain builders

**Files:**
- Modify: `src/vocalis/engines/chain.py`
- Modify: `tests/test_engine_chain.py` (append; keep all existing tests)

**Interfaces:**
- Consumes: `ModelManager.acquire / release`, `EnginePlan`, `EngineSelector`, `TTS_ENGINE_ID`, `PIPER`.
- Produces (`vocalis.engines.chain`):
  - `EngineChain.__init__(..., on_downshift, hold_engines: bool = False)` — existing positional parameters unchanged; the new one is last and defaults to `False` (old behaviour: acquire + release around every piece).
  - `EngineChain.close() -> None` — releases the held engine (starting ModelManager's idle timer); idempotent; a no-op when `hold_engines` is False or nothing is held.
  - `new_reading_chain(engine_selector: EngineSelector | None, engine_factories: Mapping[str, Callable[[], ModelEngine]] | None, primary_engine_factory: Callable[[], ModelEngine], model_manager: ModelManager, fallback_engine: Synthesizer, on_downshift: Callable[[str, str | None], None], hold_engines: bool = False) -> EngineChain` — the plan from the selector, or (no selector) the fixed built-in Qwen3 voice then Piper.
  - `make_downshift_handler(notify: Callable[[str], None], on_status: Callable[[str], None] | None = None) -> Callable[[str, str | None], None]` — silent when Piper itself failed; otherwise `notify("Primary voice unavailable — using backup voice")` and, if given, `on_status("Using backup voice")`.

- [ ] **Step 1: Write the implementation** — in `src/vocalis/engines/chain.py`:

Change the imports to:

```python
"""Walks one reading's list of voice engines, falling down it on failure."""
import logging
from typing import Callable, Mapping, cast

import numpy as np
import numpy.typing as npt

from vocalis.engines.auto_select import PIPER
from vocalis.engines.base import Synthesizer, TTSEngine
from vocalis.engines.registry import EnginePlan, EngineSelector
from vocalis.model_manager import TTS_ENGINE_ID, ModelEngine, ModelManager
```

Extend the class docstring with: "With `hold_engines=True` (used for a whole reading) the engine that spoke is NOT released between pieces: it stays loaded, the idle-unload timer stays off, and `close()` releases it when the reading has finished playing. The engine is still re-acquired before every piece, so if another feature took the graphics card in the meantime it is simply reloaded."

Add the parameter and state to `__init__` (signature gains `hold_engines: bool = False` after `on_downshift`; docstring line: `hold_engines: Keep the speaking engine loaded until close().`):

```python
        self._hold_engines = hold_engines
        self._held: str | None = None
```

Replace the `finally:` part of `_synthesize_with` and add `_after_use` and `close`:

```python
        engine = self._model_manager.acquire(engine_id, factory)
        try:
            return cast(TTSEngine, engine).synthesize(text)
        finally:
            self._after_use(engine_id)

    def _after_use(self, engine_id: str) -> None:
        """Release the engine now, or remember it as held until close()."""
        if not self._hold_engines:
            self._model_manager.release(engine_id)
            return
        # A newly acquired engine has already unloaded the one held before it.
        self._held = engine_id

    def close(self) -> None:
        """Release the held engine, starting the idle-unload timer. Idempotent."""
        held, self._held = self._held, None
        if held is not None:
            self._model_manager.release(held)
```

Append the two module-level functions at the end of the file:

```python
def new_reading_chain(
    engine_selector: EngineSelector | None,
    engine_factories: Mapping[str, Callable[[], ModelEngine]] | None,
    primary_engine_factory: Callable[[], ModelEngine],
    model_manager: ModelManager,
    fallback_engine: Synthesizer,
    on_downshift: Callable[[str, "str | None"], None],
    hold_engines: bool = False,
) -> EngineChain:
    """Build the engine chain for one reading.

    With a selector: its plan and factories. Without one: the old fixed
    Qwen3-TTS voice (then the backup voice). Shared by Quick Reader and
    Document Reader so the two never drift apart.
    """
    if engine_selector is not None:
        plan = engine_selector.plan()
        factories = engine_factories or {}
    else:
        plan = EnginePlan((TTS_ENGINE_ID,), "Fixed to the built-in voice.")
        factories = {TTS_ENGINE_ID: primary_engine_factory}
    return EngineChain(plan, factories, model_manager, fallback_engine, on_downshift, hold_engines=hold_engines)


def make_downshift_handler(
    notify: Callable[[str], None],
    on_status: Callable[[str], None] | None = None,
) -> Callable[[str, "str | None"], None]:
    """Make the on_downshift callback both readers use.

    Stays silent when the backup voice (Piper) itself failed — there is
    nothing further to fall back to. Otherwise tells the user the first-choice
    voice is unavailable via `notify` and, if given, `on_status`.
    """

    def handler(failed_engine_id: str, next_engine_id: "str | None") -> None:
        """Announce one engine failure."""
        if failed_engine_id == PIPER:
            return
        notify("Primary voice unavailable — using backup voice")
        if on_status is not None:
            on_status("Using backup voice")

    return handler
```

- [ ] **Step 2: Write the tests** — append to `tests/test_engine_chain.py` (add `from vocalis.engines.chain import EngineChain, make_downshift_handler, new_reading_chain` and `from vocalis.model_manager import TTS_ENGINE_ID` to the imports; reuse the file's existing `_Tts` and `_Piper` fakes):

```python
class _Timers:
    """Counts how many idle-unload timers ModelManager schedules."""

    def __init__(self) -> None:
        self.scheduled = 0

    def __call__(self, delay: float, callback: "Callable[[], None]") -> "_TimerHandle":
        self.scheduled += 1
        return _TimerHandle()


class _TimerHandle:
    def cancel(self) -> None:
        pass


def _held_chain(
    qwen: _Tts, kokoro: _Tts, plan: "tuple[str, ...]", timers: _Timers, hold: bool
) -> "tuple[EngineChain, ModelManager]":
    manager = ModelManager(idle_timeout_seconds=30.0, timer_factory=timers)
    factories = {"qwen3_tts": lambda: qwen, "kokoro": lambda: kokoro}
    chain = EngineChain(EnginePlan(plan, "test"), factories, manager, _Piper(), lambda failed, nxt: None, hold_engines=hold)
    return chain, manager


def test_a_holding_chain_starts_no_idle_timer_until_closed() -> None:
    timers = _Timers()
    chain, manager = _held_chain(_Tts(1.0), _Tts(2.0), ("qwen3_tts", "piper"), timers, hold=True)

    chain.synthesize("one")
    chain.synthesize("two")
    assert timers.scheduled == 0
    assert manager.current_engine_id == "qwen3_tts"

    chain.close()
    assert timers.scheduled == 1


def test_a_normal_chain_releases_after_every_piece() -> None:
    timers = _Timers()
    chain, _manager = _held_chain(_Tts(1.0), _Tts(2.0), ("qwen3_tts", "piper"), timers, hold=False)

    chain.synthesize("one")
    chain.synthesize("two")

    assert timers.scheduled == 2


def test_close_is_safe_to_call_twice_and_when_nothing_is_held() -> None:
    timers = _Timers()
    chain, _manager = _held_chain(_Tts(1.0), _Tts(2.0), ("qwen3_tts", "piper"), timers, hold=True)
    chain.close()  # nothing held yet
    chain.synthesize("one")

    chain.close()
    chain.close()

    assert timers.scheduled == 1


def test_a_holding_chain_that_falls_to_the_next_engine_holds_the_new_one() -> None:
    timers = _Timers()
    qwen, kokoro = _Tts(1.0, fail_synth=True), _Tts(2.0)
    chain, manager = _held_chain(qwen, kokoro, ("qwen3_tts", "kokoro", "piper"), timers, hold=True)

    audio, _rate = chain.synthesize("one")

    assert audio[0] == 2.0
    assert manager.current_engine_id == "kokoro"
    assert timers.scheduled == 0
    chain.close()
    assert timers.scheduled == 1


def test_a_held_engine_taken_by_another_feature_is_reloaded_for_the_next_piece() -> None:
    timers = _Timers()
    built: "list[_Tts]" = []

    def factory() -> _Tts:
        engine = _Tts(1.0)
        built.append(engine)
        return engine

    manager = ModelManager(idle_timeout_seconds=30.0, timer_factory=timers)
    chain = EngineChain(EnginePlan(("qwen3_tts", "piper"), "test"), {"qwen3_tts": factory}, manager, _Piper(), lambda f, n: None, hold_engines=True)
    chain.synthesize("one")

    manager.acquire("parakeet", lambda: _Tts(9.0))  # Dictation loads its own model, unloading the voice
    audio, _rate = chain.synthesize("two")

    assert audio[0] == 1.0
    assert len(built) == 2  # the voice was reloaded, not crashed or skipped


def test_new_reading_chain_without_a_selector_uses_the_fixed_builtin_voice() -> None:
    qwen = _Tts(1.0)
    chain = new_reading_chain(None, None, lambda: qwen, ModelManager(idle_timeout_seconds=9999.0), _Piper(), lambda f, n: None)

    audio, _rate = chain.synthesize("hello")

    assert audio[0] == 1.0
    assert TTS_ENGINE_ID == "qwen3_tts"


def test_new_reading_chain_with_a_selector_follows_its_plan() -> None:
    class _Selector:
        def plan(self) -> EnginePlan:
            return EnginePlan(("kokoro", "piper"), "test")

    kokoro = _Tts(2.0)
    chain = new_reading_chain(
        _Selector(),  # type: ignore[arg-type]
        {"kokoro": lambda: kokoro},
        lambda: _Tts(1.0),
        ModelManager(idle_timeout_seconds=9999.0),
        _Piper(),
        lambda f, n: None,
    )

    audio, _rate = chain.synthesize("hello")

    assert audio[0] == 2.0


def test_the_downshift_handler_notifies_and_sets_status_but_is_silent_for_piper() -> None:
    messages: "list[str]" = []
    statuses: "list[str]" = []
    handler = make_downshift_handler(messages.append, statuses.append)

    handler("qwen3_tts", "piper")
    handler("piper", None)

    assert messages == ["Primary voice unavailable — using backup voice"]
    assert statuses == ["Using backup voice"]


def test_the_downshift_handler_works_without_a_status_callback() -> None:
    messages: "list[str]" = []
    handler = make_downshift_handler(messages.append)

    handler("kokoro", "piper")

    assert messages == ["Primary voice unavailable — using backup voice"]
```

Add `from typing import Callable` to the test file's imports if it is not there already.

- [ ] **Step 3: Run once and fix.** `.venv/bin/python -m pytest tests/test_engine_chain.py -q --no-cov` then `.venv/bin/mypy src tests`. Expected: pass / clean.

- [ ] **Step 4: Commit**

```bash
git add src/vocalis/engines/chain.py tests/test_engine_chain.py
git commit -m "feat: let the engine chain hold the voice for a whole reading, and share the chain builders"
```

---

## Task 5: ChunkedSpeaker — the shared producer/consumer for a reading

**Files:**
- Create: `src/vocalis/features/chunked_speaker.py`
- Create: `tests/test_chunked_speaker.py`

**Interfaces:**
- Consumes: `SpeechStream` (Task 3), `paced_audio` (existing, `vocalis.audio.pace`), `AudioPlayer`; any object with `synthesize(text)` and `close()` (an `EngineChain` fits).
- Produces (`vocalis.features.chunked_speaker`):
  - `SpeechSource` (Protocol: `synthesize(text: str) -> tuple[NDArray[float32], int]`, `close() -> None`)
  - `SpeakOutcome` (Enum: `DONE`, `STOPPED`, `SPEECH_FAILED`, `PLAYBACK_FAILED`)
  - `ChunkedSpeaker(player: AudioPlayer, source_factory: Callable[[], SpeechSource], speed_provider: Callable[[], float] | None = None, on_chunk_started: Callable[[int, int], None] | None = None, fade_seconds=DEFAULT_FADE_SECONDS, stall_seconds=DEFAULT_STALL_SECONDS, max_queued_clips=DEFAULT_MAX_QUEUED_CLIPS)` — ONE object per reading.
  - `run(chunks: Sequence[str]) -> SpeakOutcome` — blocks the calling worker thread until the reading is over; ALWAYS closes the source (releasing the voice) and the stream on the way out.
  - `pause()`, `resume()`, `stop()` — safe from any thread, before or during `run()`; `stop()` is permanent.
  - property `queued_clips: int`.
  - `on_chunk_started(current, total)` is 1-based and fires (from a helper thread) when that piece starts to be HEARD.

- [ ] **Step 1: Write the implementation** — create `src/vocalis/features/chunked_speaker.py`:

```python
"""Speaks a list of text chunks as one continuous, gap-free reading.

Shared by Quick Reader and Document Reader. run() is a producer loop on the
caller's worker thread: it makes each chunk with the voice source (one
EngineChain holds the voice loaded for the whole reading), applies the
speaking pace, and pushes the audio into a SpeechStream; the sound card
plays from that stream on its own thread. Because the stream's queue is
bounded, the producer stays a few chunks ahead of the listener and then
waits.

The voice is released (and so ModelManager's idle-unload timer starts)
only when run() leaves — i.e. when playback has ended, not when the last
chunk was merely synthesized.

One ChunkedSpeaker is made per reading. stop() is a permanent latch: even
if it lands before run() has opened the stream (while the model is still
loading), run() returns STOPPED without ever making sound.
"""
import logging
import threading
from enum import Enum
from typing import Callable, Protocol, Sequence

import numpy as np
import numpy.typing as npt

from vocalis.audio.pace import paced_audio
from vocalis.audio.playback import AudioPlayer
from vocalis.audio.speech_stream import (
    DEFAULT_FADE_SECONDS,
    DEFAULT_MAX_QUEUED_CLIPS,
    DEFAULT_STALL_SECONDS,
    SpeechStream,
)

logger = logging.getLogger("vocalis")


class SpeechSource(Protocol):
    """Makes speech for one reading and releases its voice afterwards."""

    def synthesize(self, text: str) -> "tuple[npt.NDArray[np.float32], int]":
        """Speak `text`; returns mono float32 audio and its sample rate."""
        ...

    def close(self) -> None:
        """Release the voice (idempotent)."""
        ...


class SpeakOutcome(Enum):
    """How a reading ended."""

    DONE = "done"
    STOPPED = "stopped"
    SPEECH_FAILED = "speech_failed"
    PLAYBACK_FAILED = "playback_failed"


class ChunkedSpeaker:
    """One reading: make each chunk, queue it, let the stream play it."""

    def __init__(
        self,
        player: AudioPlayer,
        source_factory: Callable[[], SpeechSource],
        speed_provider: Callable[[], float] | None = None,
        on_chunk_started: Callable[[int, int], None] | None = None,
        fade_seconds: float = DEFAULT_FADE_SECONDS,
        stall_seconds: float = DEFAULT_STALL_SECONDS,
        max_queued_clips: int = DEFAULT_MAX_QUEUED_CLIPS,
    ) -> None:
        """Create the speaker for ONE reading.

        Args:
            player: Opens the real output stream.
            source_factory: Builds the voice source for this reading (called once in run()).
            speed_provider: Current speaking pace, read once per chunk; None = no pacing.
            on_chunk_started: Called with (current, total), 1-based, when a
                chunk starts to be heard. Runs on a helper thread.
            fade_seconds: Fade at each end of every chunk's audio.
            stall_seconds: Silence tolerated from a broken sound device.
            max_queued_clips: How far ahead of the listener the producer may get.
        """
        self._source_factory = source_factory
        self._speed_provider = speed_provider
        self._on_chunk_started = on_chunk_started
        self._stop = threading.Event()
        self._total = 0
        self._stream = SpeechStream(
            player,
            on_clip_started=self._clip_started,
            fade_seconds=fade_seconds,
            stall_seconds=stall_seconds,
            max_queued_clips=max_queued_clips,
        )

    @property
    def queued_clips(self) -> int:
        """Chunks currently queued in the stream (playing + waiting)."""
        return self._stream.queued_clips

    def pause(self) -> None:
        """Go silent at once (the heard chunk replays from its start on resume)."""
        self._stream.pause()

    def resume(self) -> None:
        """Carry on after pause()."""
        self._stream.resume()

    def stop(self) -> None:
        """End the reading for good; safe before or during run()."""
        self._stop.set()
        self._stream.stop()

    def run(self, chunks: Sequence[str]) -> SpeakOutcome:
        """Speak every chunk in order; block until the reading is over.

        Returns:
            DONE if everything played; STOPPED if stop() ended it;
            SPEECH_FAILED if no voice could make a chunk; PLAYBACK_FAILED if
            the sound device failed or stalled.
        """
        self._total = len(chunks)
        if self._stop.is_set():
            return SpeakOutcome.STOPPED
        try:
            source = self._source_factory()
        except Exception:
            logger.exception("Could not prepare the voice engines")
            return SpeakOutcome.STOPPED if self._stop.is_set() else SpeakOutcome.SPEECH_FAILED
        try:
            return self._speak(source, chunks)
        finally:
            self._stream.stop()
            try:
                source.close()
            except Exception:
                logger.exception("Could not release the voice engine")

    def _speak(self, source: SpeechSource, chunks: Sequence[str]) -> SpeakOutcome:
        """The producer loop (see run())."""
        opened = False
        for index, text in enumerate(chunks):
            if self._stop.is_set():
                return SpeakOutcome.STOPPED
            try:
                audio, sample_rate = source.synthesize(text)
            except Exception:
                if self._stop.is_set():
                    return SpeakOutcome.STOPPED
                logger.exception("Both TTS engines failed")
                return SpeakOutcome.SPEECH_FAILED
            if self._stop.is_set():
                return SpeakOutcome.STOPPED
            audio = paced_audio(audio, sample_rate, self._speed_provider)
            try:
                if not opened:
                    self._stream.start(sample_rate)
                    opened = True
                if not self._stream.push(audio, sample_rate, index):
                    return SpeakOutcome.STOPPED
            except Exception:
                if self._stop.is_set():
                    return SpeakOutcome.STOPPED
                logger.exception("Playback failed")
                return SpeakOutcome.PLAYBACK_FAILED
        if not opened:
            return SpeakOutcome.DONE  # nothing to say
        self._stream.finish()
        try:
            finished = self._stream.wait_until_done()
        except Exception:
            if self._stop.is_set():
                return SpeakOutcome.STOPPED
            logger.exception("Playback failed")
            return SpeakOutcome.PLAYBACK_FAILED
        return SpeakOutcome.DONE if finished else SpeakOutcome.STOPPED

    def _clip_started(self, tag: int) -> None:
        """Stream hook: chunk `tag` (0-based) has started to be heard."""
        if self._on_chunk_started is not None:
            self._on_chunk_started(tag + 1, self._total)
```

- [ ] **Step 2: Write the tests** — create `tests/test_chunked_speaker.py`:

```python
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
        self.length = length
        self.rate = rate
        self.fail_on_call = fail_on_call
        self.calls: "list[str]" = []
        self.closed = 0

    def synthesize(self, text: str) -> "tuple[npt.NDArray[np.float32], int]":
        self.calls.append(text)
        if self.fail_on_call == len(self.calls):
            raise RuntimeError("no voice could speak")
        return np.full(self.length, 0.1 * len(self.calls), dtype=np.float32), self.rate

    def close(self) -> None:
        self.closed += 1


def _manual_player() -> "tuple[AudioPlayer, OutputRecorder]":
    recorder = OutputRecorder()
    return AudioPlayer(stream_factory=manual_stream_factory(recorder)), recorder


def _start(speaker: ChunkedSpeaker, chunks: "list[str]") -> "tuple[threading.Thread, list[SpeakOutcome]]":
    outcomes: "list[SpeakOutcome]" = []
    thread = threading.Thread(target=lambda: outcomes.append(speaker.run(chunks)), daemon=True)
    thread.start()
    return thread, outcomes


def test_a_whole_reading_plays_every_chunk_once_in_order_on_one_stream() -> None:
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
    player = FakeAudioPlayer()
    source = FakeSource()

    assert ChunkedSpeaker(player, lambda: source).run([]) is SpeakOutcome.DONE
    assert player.output.opened == 0
    assert source.closed == 1


def test_the_producer_waits_when_the_queue_is_full_and_stop_ends_it() -> None:
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
    player, recorder = _manual_player()
    source = FakeSource()
    speaker = ChunkedSpeaker(player, lambda: source)

    speaker.stop()

    assert speaker.run(["anything"]) is SpeakOutcome.STOPPED
    assert recorder.opened == 0
    assert source.calls == []


def test_a_stop_during_synthesis_is_honoured_when_the_piece_arrives() -> None:
    player, recorder = _manual_player()
    entered = threading.Event()
    release = threading.Event()

    class SlowSource(FakeSource):
        def synthesize(self, text: str) -> "tuple[npt.NDArray[np.float32], int]":
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
    player, _recorder = _manual_player()
    source = FakeSource(fail_on_call=2)
    speaker = ChunkedSpeaker(player, lambda: source, fade_seconds=0.0)
    thread, outcomes = _start(speaker, ["one", "two", "three"])

    thread.join(timeout=5.0)

    assert outcomes == [SpeakOutcome.SPEECH_FAILED]
    assert source.closed == 1


def test_a_source_that_cannot_be_built_gives_speech_failed() -> None:
    player, _recorder = _manual_player()

    def broken() -> FakeSource:
        raise RuntimeError("no engines")

    assert ChunkedSpeaker(player, broken).run(["x"]) is SpeakOutcome.SPEECH_FAILED


def test_a_stalled_sound_device_gives_playback_failed() -> None:
    player, recorder = _manual_player()
    source = FakeSource()
    speaker = ChunkedSpeaker(player, lambda: source, stall_seconds=0.2)
    thread, outcomes = _start(speaker, ["one"])  # the fake device is never pulled

    thread.join(timeout=5.0)

    assert outcomes == [SpeakOutcome.PLAYBACK_FAILED]
    assert recorder.closed == 1
    assert source.closed == 1


def test_a_device_that_refuses_to_open_gives_playback_failed() -> None:
    class RefusingStream:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def start(self) -> None:
            raise RuntimeError("device busy")

        def stop(self) -> None:
            pass

        def close(self) -> None:
            pass

    player = AudioPlayer(stream_factory=lambda **kw: RefusingStream(**kw))

    assert ChunkedSpeaker(player, lambda: FakeSource()).run(["x"]) is SpeakOutcome.PLAYBACK_FAILED


def test_pause_goes_silent_and_resume_replays_the_heard_chunk_from_its_start() -> None:
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
```

- [ ] **Step 3: Run once and fix.** `.venv/bin/python -m pytest tests/test_chunked_speaker.py -q --no-cov` then `.venv/bin/mypy src tests`. Expected: pass / clean.

- [ ] **Step 4: Commit**

```bash
git add src/vocalis/features/chunked_speaker.py tests/test_chunked_speaker.py
git commit -m "feat: add ChunkedSpeaker, the shared producer/consumer for gapless readings"
```

---

## Task 6: Document Reader speaks through ChunkedSpeaker

**Files:**
- Modify: `src/vocalis/features/doc_reader.py`
- Modify: `tests/test_doc_reader.py` (large port — see Step 2)
- Modify: `tests/test_shared_tts_engine.py` (Document Reader side only)
- Modify: `tests/test_reading_hooks_integration.py` (Document Reader side only)

**Interfaces:**
- Consumes: `ChunkedSpeaker`, `SpeakOutcome` (Task 5); `new_reading_chain`, `make_downshift_handler`, `EngineChain.close` via the speaker (Task 4); `chunk_text` (Task 1).
- Produces: `DocumentReaderFeature` with the **same public interface and constructor** as today (`is_reading`, `generation`, `read_file`, `read_text`, `pause`, `resume`, `stop`). Behaviour changes: pieces play gaplessly on one stream; progress `(current, total)` is reported when a piece starts being heard; the voice stays loaded for the whole reading and is released when playback ends.

- [ ] **Step 1: Write the implementation** — edit `src/vocalis/features/doc_reader.py`:

*Module docstring*: replace the "Pause/resume design" paragraph with:

```
Reading design: the document is split into sentence-bounded chunks and
handed to a ChunkedSpeaker (features/chunked_speaker.py), which makes each
chunk with the voice chain while the previous ones are being heard, so the
reading plays on one continuous stream with no gaps. pause() silences the
stream at once and resume() replays the chunk being heard from its start;
stop() (or a ReadingCoordinator preemption) ends the reading at any point,
including before the first sound exists. The voice stays loaded for the
whole reading and is released when playback has ended.
```

*Imports*: remove `from vocalis.audio.pace import paced_audio`, `from vocalis.engines.chain import EngineChain`, `from vocalis.engines.registry import EnginePlan, EngineSelector` → keep `EngineSelector` only, and `from vocalis.model_manager import ModelEngine, ModelManager, TTS_ENGINE_ID` → `ModelEngine, ModelManager`; remove `numpy`/`numpy.typing` imports (no longer used). Add:

```python
from vocalis.engines.chain import EngineChain, make_downshift_handler, new_reading_chain
from vocalis.engines.registry import EngineSelector
from vocalis.features.chunked_speaker import ChunkedSpeaker, SpeakOutcome
```

*`__init__`*: delete `self._chain`, `self._resume_event` (both lines) and add `self._speaker: ChunkedSpeaker | None = None`.

*Replace* `pause`, `resume`, `stop`, `_start_new_reading` with:

```python
    def pause(self) -> None:
        """Pause the current reading immediately. No-op if not reading
        or already paused."""
        with self._lock:
            if not self._active or self._paused:
                return
            self._paused = True
            speaker = self._speaker
        if speaker is not None:
            speaker.pause()
        self._notify_async("Paused")
        self._on_status("Paused")

    def resume(self) -> None:
        """Resume a paused reading, replaying the chunk being heard from its
        start. No-op if not currently paused."""
        with self._lock:
            if not self._active or not self._paused:
                return
            self._paused = False
            speaker = self._speaker
        if speaker is not None:
            speaker.resume()
        self._notify_async("Resuming...")
        self._on_status("Resuming...")

    def stop(self) -> None:
        """Stop the current reading immediately. No-op if not reading.

        Safe to call from any thread, including as a ReadingCoordinator
        stopper callback when Quick Reader preempts this reading. Works at
        any point of a reading — even before the first sound exists: the
        speaker's stop is a permanent latch, so a late-finishing first
        piece never plays.
        """
        with self._lock:
            if not self._active:
                return
            self._active = False
            self._paused = False
            speaker, self._speaker = self._speaker, None
        self._reading_coordinator.finish_reading(self)
        if speaker is not None:
            speaker.stop()
        self._notify_async("Stopped")
        self._on_status("Stopped")
        self._on_finished()
```

and

```python
    def _start_new_reading(self) -> int:
        """Mark a fresh reading as active and return its generation number."""
        with self._lock:
            self._active = True
            self._paused = False
            self._speaker = None
            self._generation += 1
            return self._generation
```

*Replace* `_read_worker`, `_play_chunk_until_done`, `_new_chain`, `_notify_downshift`, `_synthesize_with_fallback` (everything from `def _read_worker` to the end of the file) with:

```python
    def _read_worker(self, generation: int, chunks: "list[str]") -> None:
        """Hand the chunks to a ChunkedSpeaker and turn its outcome into the
        usual status / notification / finished reports.

        A pause() that landed before the speaker existed is applied here,
        under the same lock that publishes the speaker, so a pause or stop
        can never fall between "no speaker yet" and "speaker running".
        """
        speaker = ChunkedSpeaker(
            player=self._player,
            source_factory=self._new_source,
            speed_provider=self._speed_provider,
            on_chunk_started=lambda current, total: self._report_progress(generation, current, total),
        )
        with self._lock:
            if not (self._active and self._generation == generation):
                return  # stopped while extracting/chunking
            self._speaker = speaker
            start_paused = self._paused
        if start_paused:
            speaker.pause()
        outcome = speaker.run(chunks)
        with self._lock:
            if self._speaker is speaker:
                self._speaker = None
        self._finish_reading(generation, outcome)

    def _report_progress(self, generation: int, current: int, total: int) -> None:
        """Forward 'piece N of M' while this reading is still the current one."""
        if self._is_current(generation):
            self._on_progress(current, total)

    def _finish_reading(self, generation: int, outcome: SpeakOutcome) -> None:
        """Report how the reading ended (a stop() has already reported its own end)."""
        if outcome is SpeakOutcome.SPEECH_FAILED:
            message = "Could not generate speech"
        elif outcome is SpeakOutcome.PLAYBACK_FAILED:
            message = "Playback failed"
        elif outcome is SpeakOutcome.DONE:
            message = "Done"
        else:
            # STOPPED: stop() already ended and reported this reading. If it
            # is somehow still current, end it quietly so nothing stays "active".
            if self._deactivate(generation):
                self._on_finished()
            return
        if self._deactivate(generation):
            self._on_status(message)
            self._on_finished()
            # Notify last: it can block, and the end must reach the hub first.
            self._notifier.notify(_TITLE, message)

    def _new_source(self) -> EngineChain:
        """Build the voice chain for one reading; it holds the voice loaded until closed."""
        return new_reading_chain(
            self._engine_selector,
            self._engine_factories,
            self._primary_engine_factory,
            self._model_manager,
            self._fallback_engine,
            make_downshift_handler(
                lambda message: self._notifier.notify(_TITLE, message), self._on_status
            ),
            hold_engines=True,
        )
```

Leave `_deactivate`, `_is_current`, `_prepare_and_read` (still ends with `self._read_worker(generation, chunks)`), `_notify_async`, `_deliver_notifications`, `read_file`, `read_text`, `_report_started` exactly as they are. Run `.venv/bin/mypy src` mentally for the unused imports (`Callable`, `Mapping`, `queue`, `threading`, `Path` are still used).

- [ ] **Step 2: Port the tests.** Keep `tests/test_doc_reader.py`'s file name and the helper names other files import. Rules:

  1. **New shared fixtures** at the top of the file: `from tests.fake_output import FakeAudioPlayer, OutputRecorder, manual_stream_factory, pump_stream_factory, wait_until as _wait_until`. Keep the file's own `wait_until` (other files import it). Delete `FakeStream`, `_fake_stream_factory`, `_recording_stream_factory`, `_endless_stream_factory`, `LengthRecordingPlayer`, `RacingPlayer`, `WindowPlayer`, `PacedWindowPlayer`, `_CountingEvent`, `_SignalingEvent`, `_join_workers`, `_window_feature` — and **add compatibility aliases** so `tests/test_reading_hooks_integration.py` keeps importing: `StreamStartRecorder = OutputRecorder` and `def _endless_stream_factory(recorder: OutputRecorder) -> Callable[..., Any]: return manual_stream_factory(recorder)` (a manual stream that nobody pulls = a reading that never finishes).
  2. **`_make_feature`**: its `stream_factory` parameter becomes `stream_factory: "Callable[..., Any] | None" = None` and, when None, the helper builds a fresh `pump_stream_factory(OutputRecorder())`; `player=AudioPlayer(stream_factory=...)` stays. Tests that need to inspect or drive the output create their own `recorder = OutputRecorder()` and pass `pump_stream_factory(recorder)` (plays out by itself) or `manual_stream_factory(recorder)` (the test pulls the audio).
  2b. **Audible fake voices.** `FakeSynthesizer` and `GatedSynthesizer` currently return 100 samples of silence, which cannot be told apart from "nothing played". Give both a `level: float = 0.0` constructor argument and make `synthesize()` return `np.full(100, level, dtype=np.float32)`; every test that looks for sound (`recorder.played()`, `pull(...)` non-empty / all-zero checks) constructs them with `level=0.5`. Tests that never look at sound keep the default.
  3. **Keep as-is (they only need the new default factory):** `test_read_text_speaks_every_chunk_and_reports_done` (progress now arrives asynchronously — replace `assert progresses[0] == (1, 1)` with `wait_until(lambda: progresses[:1] == [(1, 1)])`), `test_read_file_extracts_and_speaks_a_real_file`, `test_unsupported_file_reports_status_and_finishes`, `test_primary_failure_falls_back_to_piper_once_per_document`, `test_empty_pasted_text_reports_nothing_to_read`, `test_extraction_errors_reach_status_and_notification_and_finish_once`, `test_both_engines_failing_reports_could_not_generate_speech`, `test_after_one_failure_later_chunks_go_straight_to_piper`, `test_an_already_loaded_primary_engine_is_reused_without_fallback`, `test_generation_increases_by_one_for_each_reading`, `test_on_started_is_called_once_per_reading`, `test_raising_on_started_hook_does_not_break_reading`, `test_reading_end_is_reported_before_the_done_notification_is_sent`, `test_the_reading_claim_is_released_while_holding_the_feature_lock`, the three `test_a_selector_*`/`test_a_failing_backup_voice_*` tests, and the notifier tests (`test_pause_resume_stop_do_not_block_on_a_slow_notifier`, `test_control_notifications_are_delivered_in_call_order`, `test_a_notifier_that_raises_is_survived_and_later_messages_still_arrive`) — for the notifier tests, where they start a reading and wait for playback, use `manual_stream_factory(recorder)` (an unpulled stream = a reading that stays in progress) so pause/resume/stop have something to act on.
  4. **`test_playback_raising_reports_playback_failed`**: replace its raising `play()` with a stream factory whose `start()` raises `RuntimeError("device busy")` (use the `RefusingStream` pattern from `tests/test_chunked_speaker.py`); assert status `"Playback failed"` and one finish.
  5. **Speed tests** (`test_no_speed_provider_leaves_chunk_audio_unchanged`, `test_speed_provider_is_read_per_chunk_and_produces_two_lengths`, `test_a_raising_speed_provider_still_plays_the_original_chunk`): replace `LengthRecordingPlayer.play_lengths` with the number of non-silent samples in `recorder.played()` per chunk — use `manual_stream_factory(recorder)`, wait for the pushes with `wait_until(lambda: feature._speaker is not None and feature._speaker.queued_clips == N)`, then `pull` the stream (`recorder.streams[-1].pull(...)`), and compare lengths exactly as the old assertions did (use `SineSynthesizer`, which is unchanged).
  6. **Re-written for the new shape** (replace the old test body, keep the name unless noted):
     - `test_pause_stops_audio_immediately_and_resume_replays_current_chunk`: manual stream; start `read_text` with two short sentences and `max` chunking default (one chunk) → use a text of two sentences each > 300 chars apart? Simpler: patch nothing — use `read_text("A. B.")`, wait until `feature._speaker.queued_clips == 1`, `pull(10)` non-silent, `feature.pause()`, `pull(10)` all zeros and `"Paused" in statuses`, `feature.resume()`, `pull` the whole clip again and assert it equals the original from index 0, finish reading; assert the synthesizer was called exactly once.
     - `test_pause_during_synthesis_blocks_playback_until_resume`: with `GatedSynthesizer`, `pause()` while synthesis is gated, release the gate, wait for the clip to be queued, `pull` → all zeros; `resume()` → non-silent audio appears.
     - `test_stop_while_paused_actually_stops`: pause then stop; `finishes == [None]`, `feature.is_reading is False`, `recorder.closed == 1`.
     - `test_a_second_reader_preempts_this_one_via_the_coordinator` and `test_ocr_reading_replaces_a_document_reading_in_progress`: keep, using the manual (never-finishing) stream.
  7. **Delete (obsolete — they exercised per-chunk `play()` timing windows that no longer exist; ruling 12):** `test_stop_landing_just_before_play_starts_still_stops_that_chunk`, `test_pause_landing_just_before_play_starts_still_pauses_that_chunk`, `test_pause_then_stop_around_play_start_ends_the_reading_once`, `test_pause_then_resume_replays_the_same_already_paced_chunk` (re-covered by `test_pause_goes_silent_and_resume_replays_the_heard_chunk_from_its_start` and by item 6's first test), `test_a_stale_generation_does_not_store_its_chain_for_the_next_reading` (no stored chain exists any more), and the `RacingPlayer`/`WindowPlayer` classes they used.
  8. **New tests to add** (write them in full in this file; they use the `level=0.5` fakes from 2b):

```python
def test_progress_is_reported_when_each_piece_starts_to_be_heard() -> None:
    recorder = OutputRecorder()
    primary = FakeSynthesizer(level=0.5)
    feature, _statuses, progresses, _finishes = _make_feature(
        fallback=FakeSynthesizer(), primary=primary, notifier=RecordingNotifier(),
        stream_factory=manual_stream_factory(recorder),
    )
    sentence = " ".join(["word"] * 70) + "."  # 349 characters: each one becomes its own chunk
    feature.read_text(" ".join([sentence] * 3))

    wait_until(lambda: feature._speaker is not None and feature._speaker.queued_clips == 3)
    assert progresses == []  # all three are made, none has been heard yet

    stream = recorder.streams[-1]
    stream.pull(10)
    wait_until(lambda: progresses == [(1, 3)])
    stream.pull(10_000)
    wait_until(lambda: progresses == [(1, 3), (2, 3), (3, 3)])


class _CountingTimers:
    """A ModelManager timer factory that only counts how many idle timers were started."""

    def __init__(self) -> None:
        self.scheduled = 0

    def __call__(self, delay: float, callback: Callable[[], None]) -> "_NoopTimer":
        self.scheduled += 1
        return _NoopTimer()


class _NoopTimer:
    def cancel(self) -> None:
        pass


def test_the_voice_stays_loaded_for_the_whole_reading_and_the_idle_timer_starts_after_playback() -> None:
    timers = _CountingTimers()
    recorder = OutputRecorder()
    primary = FakeSynthesizer(level=0.5)
    finishes: "list[None]" = []
    feature = DocumentReaderFeature(
        model_manager=ModelManager(idle_timeout_seconds=30.0, timer_factory=timers),
        primary_engine_factory=lambda: primary,
        fallback_engine=FakeSynthesizer(),
        player=AudioPlayer(stream_factory=manual_stream_factory(recorder)),
        notifier=RecordingNotifier(),  # type: ignore[arg-type]
        reading_coordinator=ReadingCoordinator(),
        on_status=lambda message: None,
        on_progress=lambda current, total: None,
        on_finished=lambda: finishes.append(None),
    )

    feature.read_text("One short sentence. Another short sentence.")
    wait_until(lambda: feature._speaker is not None and feature._speaker.queued_clips >= 1)
    assert timers.scheduled == 0  # every piece is made, none has been heard: the voice must stay loaded

    recorder.streams[-1].pull(100_000)
    wait_until(lambda: bool(finishes))
    assert timers.scheduled == 1  # the idle timer starts only now that playback has ended


def test_stop_before_the_first_sound_exists_never_makes_sound() -> None:
    recorder = OutputRecorder()
    primary = GatedSynthesizer(level=0.5)
    feature, _statuses, _progresses, finishes = _make_feature(
        fallback=FakeSynthesizer(), primary=primary, notifier=RecordingNotifier(),
        stream_factory=manual_stream_factory(recorder),
    )

    feature.read_text("Only one short sentence here.")
    primary.wait_until_entered()
    feature.stop()
    primary.release()
    wait_until(lambda: bool(finishes))
    threading.Event().wait(0.2)

    assert recorder.opened == 0
    assert feature.is_reading is False
```

  9. **`tests/test_shared_tts_engine.py`**: replace `_InstantStream` with a pump stream for the Document Reader player only: `player=AudioPlayer(stream_factory=pump_stream_factory(OutputRecorder()))` (keep the Quick Reader player as is — it changes in Task 7). The `len(built) == 1` / `load_calls == 1` assertions must still hold.
  10. **`tests/test_reading_hooks_integration.py`**: the Document Reader side keeps working through the compatibility aliases from rule 1; run it and fix only genuine breakage.

- [ ] **Step 3: Run once and fix.** Run `.venv/bin/python -m pytest tests/test_doc_reader.py tests/test_shared_tts_engine.py tests/test_reading_hooks_integration.py tests/test_doc_popup.py -q --no-cov`, fix, then the **full suite** once: `.venv/bin/python -m pytest -q -p no:cacheprovider` (only the 2 known offline-guard failures may remain) and `.venv/bin/mypy src tests`. Quick Reader tests must still pass untouched at this point (it is unchanged until Task 7).

- [ ] **Step 4: Commit**

```bash
git add src/vocalis/features/doc_reader.py tests/test_doc_reader.py tests/test_shared_tts_engine.py tests/test_reading_hooks_integration.py
git commit -m "feat: Document Reader speaks through ChunkedSpeaker for gapless long readings"
```

---

## Task 7: Quick Reader speaks through ChunkedSpeaker (and gains chunking)

**Files:**
- Modify: `src/vocalis/features/quick_reader.py`
- Modify: `tests/test_quick_reader.py` (port)
- Modify: `tests/test_cross_feature_integration.py` (replace its fake player)
- Modify: `tests/test_shared_tts_engine.py` (Quick Reader side)
- Modify: `tests/test_reading_hooks_integration.py` (Quick Reader side)

**Interfaces:**
- Consumes: `ChunkedSpeaker`, `SpeakOutcome`, `chunk_text`, `new_reading_chain`, `make_downshift_handler`, `FakeAudioPlayer` (test helper).
- Produces: `QuickReaderFeature` with the **same constructor and public interface** (`on_trigger`, `stop`). Internals: `_flip_active_state()` now returns `tuple[bool, int, ChunkedSpeaker | None]`; a long clipboard is read in sentence chunks, gaplessly.

- [ ] **Step 1: Write the implementation** — edit `src/vocalis/features/quick_reader.py`:

*Imports*: remove `from vocalis.audio.pace import paced_audio`, the numpy imports, `from vocalis.engines.base import Synthesizer`? **No — keep `Synthesizer`** (it types `fallback_engine`); remove `EngineChain`/`EnginePlan` usage and `TTS_ENGINE_ID` import. Add:

```python
from vocalis.engines.chain import EngineChain, make_downshift_handler, new_reading_chain
from vocalis.engines.registry import EngineSelector
from vocalis.features.chunked_speaker import ChunkedSpeaker, SpeakOutcome
from vocalis.text.chunker import chunk_text
```
and add the module constant `_TEXT_NOTIFY_JOIN_SECONDS = 6.0` below `_TITLE`.

*`__init__`*: add `self._speaker: ChunkedSpeaker | None = None` after `self._active = False`.

*Replace* `on_trigger`, `_stop_for_coordinator`, `_flip_active_state`, `_stop_and_notify` with:

```python
    def on_trigger(self) -> None:
        """Call when Ctrl+Alt+R is pressed: start a new reading, or stop
        an in-progress one. Never blocks the caller (the hotkey
        listener's watcher thread): the state flip happens under a lock
        scoped to just that flip (see _flip_active_state()), and every
        potentially slow call — stopping the speaker, which can wait on the
        audio device, and Notifier.notify(), a notify-send subprocess that
        can take up to its own timeout — is dispatched onto a short-lived
        background thread rather than run inline.
        """
        interrupting, generation, speaker = self._flip_active_state()
        if interrupting:
            self._report_reading_state()
            threading.Thread(target=self._stop_and_notify, args=(speaker,), daemon=True).start()
            return
        if self._reading_coordinator is not None:
            self._reading_coordinator.start_reading(self, self._stop_for_coordinator)
        self._report_reading_state()
        threading.Thread(target=self._read_aloud, args=(generation,), daemon=True).start()

    def _stop_for_coordinator(self) -> None:
        """Stop this feature's in-progress reading because a different
        feature (via ReadingCoordinator) is claiming playback ownership.

        Clear-only: if this feature is idle (e.g. it was already stopped
        by a second press, leaving a stale coordinator claim) this is a
        harmless no-op — it must never flip an idle feature to active.
        Never calls back into the coordinator and never blocks: it only
        clears the flag under the lock, then spawns the same background
        stop-and-notify thread on_trigger() uses, off the lock.
        """
        with self._lock:
            if not self._active:
                return
            self._active = False
            speaker, self._speaker = self._speaker, None
        self._report_reading_state()
        threading.Thread(target=self._stop_and_notify, args=(speaker,), daemon=True).start()
```

```python
    def _flip_active_state(self) -> "tuple[bool, int, ChunkedSpeaker | None]":
        """Atomically flip self._active under the lock and report what
        the caller should do next, without making any blocking calls
        while the lock is held.

        Returns:
            (True, 0, speaker) if this call is interrupting an already-active
            reading (the caller should stop `speaker` — None if the sound had
            not started yet — and notify, off the lock); (False, generation,
            None) if this call is starting a fresh reading (the caller should
            spawn a worker thread scoped to `generation`).
        """
        with self._lock:
            if self._active:
                self._active = False
                speaker, self._speaker = self._speaker, None
                return True, 0, speaker
            self._active = True
            self._generation += 1
            return False, self._generation, None

    def _stop_and_notify(self, speaker: "ChunkedSpeaker | None") -> None:
        """Stop the sound and tell the user, off the caller's thread so
        on_trigger() itself never blocks on either call. A speaker that does
        not exist yet needs no stopping: the worker sees the cleared flag."""
        if speaker is not None:
            speaker.stop()
        self._notifier.notify(_TITLE, "Stopped")
```

*Replace* `_read_aloud` and everything after it (`_new_chain`, `_notify_downshift`, `_synthesize_with_fallback`) with:

```python
    def _read_aloud(self, generation: int) -> None:
        """Do the real work for one reading, entirely on a background
        thread: read the clipboard, split it into sentence chunks, and speak
        them gaplessly through a ChunkedSpeaker — bailing out cleanly at each
        stage if `on_trigger()` has cancelled this generation in the meantime.

        The copied text is announced from a one-off side thread so a slow
        notify-send can never delay the sound; "Done" waits briefly for that
        thread so the notifications still arrive in the same order.
        """
        text = self._clipboard_reader.read()
        if not self._is_current(generation):
            return  # cancelled while reading the clipboard
        if not text.strip():
            self._notifier.notify(_TITLE, "Nothing copied")
            self._deactivate(generation)
            return

        self._notifier.notify(_TITLE, "Reading...")
        speaker = ChunkedSpeaker(
            player=self._player,
            source_factory=self._new_source,
            speed_provider=self._speed_provider,
        )
        with self._lock:
            if not (self._active and self._generation == generation):
                return  # cancelled before the sound could start
            self._speaker = speaker
        text_notice = threading.Thread(target=self._notifier.notify, args=(_TITLE, text), daemon=True)
        text_notice.start()
        outcome = speaker.run(chunk_text(text))
        with self._lock:
            if self._speaker is speaker:
                self._speaker = None
        text_notice.join(timeout=_TEXT_NOTIFY_JOIN_SECONDS)

        if outcome is SpeakOutcome.SPEECH_FAILED:
            self._notifier.notify(_TITLE, "Could not generate speech")
            self._deactivate(generation)
            return
        if outcome is SpeakOutcome.PLAYBACK_FAILED:
            self._notifier.notify(_TITLE, "Playback failed")
            self._deactivate(generation)
            return
        self._deactivate(generation)
        if outcome is SpeakOutcome.DONE:
            self._notifier.notify(_TITLE, "Done")

    def _new_source(self) -> EngineChain:
        """Build the voice chain for one reading; it holds the voice loaded until closed."""
        return new_reading_chain(
            self._engine_selector,
            self._engine_factories,
            self._primary_engine_factory,
            self._model_manager,
            self._fallback_engine,
            make_downshift_handler(lambda message: self._notifier.notify(_TITLE, message)),
            hold_engines=True,
        )
```

- [ ] **Step 2: Port the tests.** In `tests/test_quick_reader.py`:

  1. **Replace `FakePlayer`/`ResettablePlayer`/`FakeStream`** with the real `AudioPlayer` on a fake output. `make_feature(...)` takes `player: AudioPlayer | None = None` and, when None, builds `FakeAudioPlayer()` (from `tests.fake_output`); it returns the same 6-tuple but the 4th item is now that player (tests read `player.output`). Keep `ResettablePlayer` as a name (other files import it) — redefine it as `class ResettablePlayer(FakeAudioPlayer)` with `__init__(self, delay_seconds: float = 0.0)` where `delay_seconds > 0` means "slow, real-time-ish output" (`speedup = 1.0`) and `0` means fast (`50.0`), and a `reset()` method that creates a fresh `OutputRecorder` (assign `self.output = OutputRecorder()` and rebuild `self._stream_factory = pump_stream_factory(self.output, speedup)`).
  2. **Substitution rules for assertions:** `player.play_calls == []` → `player.output.opened == 0`; `len(player.play_calls) == 1` / `wait_until(lambda: len(player.play_calls) == 1)` → `player.output.opened == 1` / `wait_until(lambda: player.output.opened == 1)`; `player.play_calls == [(audio, rate)]` → `np.testing.assert_allclose(player.output.played()[: len(audio)], audio)` (use `fade_seconds=0` is NOT available through the feature, so compare lengths and `played().max()` instead of exact samples — fades only touch the first/last ~8 ms); `player.stop_calls == 1` → `player.output.closed >= 1`; "stays playing for 2 s" (`FakePlayer(delay_seconds=2.0)`) → a **slow** output (`FakeAudioPlayer(speedup=1.0)`) plus a synthesizer returning ≥ 3 s of audio (`np.zeros(72000, dtype=np.float32)` at 24000 Hz would be silent — use `np.full(72000, 0.1, dtype=np.float32)`), so playback is genuinely still in progress when the second press lands.
  3. **Keep (with the substitutions above):** every test whose name begins `test_trigger_`, `test_primary_failure_`, `test_both_engines_failing_`, `test_second_trigger_`, `test_third_trigger_`, `test_starting_a_reading_`, `test_a_different_owner_`, `test_preempting_an_idle_`, `test_reading_changes_`, `test_press_stop_press_`, `test_public_stop_`, `test_natural_end_`, `test_raising_reading_hook_`, `test_late_end_report_`, `test_widget_stop_`, `test_reporting_twice_`, `test_a_stale_reading_`, `test_the_reading_claim_`, `test_no_speed_provider_`, `test_speed_provider_`, `test_a_raising_speed_provider_`, `test_a_selector_`, `test_a_failing_planned_engine_`, `test_a_failing_backup_voice_`.
  4. **`test_stop_landing_during_live_text_notification_stops_actual_playback`**: keep its intent — `TriggeringNotifier` presses the hotkey again while the copied-text notification is being sent. The notification now runs on a side thread while the sound is already starting, so assert: `Stopped` is notified, `"Done"` is never notified, and the output stream was closed before the clip could have finished playing (`player.output.frames_pulled` is far less than the clip's length with the slow output).
  5. **Add these tests in full:**

```python
def test_a_long_clipboard_is_read_in_several_chunks_on_one_stream() -> None:
    sentences = " ".join(f"This is sentence number {n} of a rather long clipboard text." for n in range(30))
    primary = FakeSynthesizer(audio=np.full(2400, 0.1, dtype=np.float32), sample_rate=24000)
    feature, _, _, player, notifier, _ = make_feature(primary, FakeSynthesizer(audio=np.zeros(1, dtype=np.float32)), clipboard_text=sentences)

    feature.on_trigger()
    wait_until(lambda: any(msg == "Done" for _, msg in notifier.messages), timeout=10.0)

    assert len(primary.calls) > 1  # chunked, not one giant call
    assert player.output.opened == 1  # but one continuous stream
    assert " ".join(primary.calls).split() == sentences.split()  # nothing lost, nothing repeated


def test_stop_pressed_while_the_first_piece_is_still_being_made_never_makes_sound() -> None:
    primary = FakeSynthesizer(audio=np.full(2400, 0.1, dtype=np.float32), delay_seconds=0.4)
    feature, _, _, player, notifier, _ = make_feature(primary, FakeSynthesizer(audio=np.zeros(1, dtype=np.float32)), clipboard_text="One sentence here.")

    feature.on_trigger()
    wait_until(lambda: any(msg == "Reading..." for _, msg in notifier.messages))
    feature.on_trigger()  # stop while the very first piece is still being made

    wait_until(lambda: any(msg == "Stopped" for _, msg in notifier.messages))
    threading.Event().wait(0.7)
    assert player.output.opened == 0
    assert not any(msg == "Done" for _, msg in notifier.messages)


def test_a_broken_sound_device_reports_playback_failed() -> None:
    class RefusingStream:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def start(self) -> None:
            raise RuntimeError("device busy")

        def stop(self) -> None:
            pass

        def close(self) -> None:
            pass

    primary = FakeSynthesizer(audio=np.full(2400, 0.1, dtype=np.float32), sample_rate=24000)
    feature, _, _, _, notifier, _ = make_feature(
        primary,
        FakeSynthesizer(audio=np.zeros(1, dtype=np.float32)),
        clipboard_text="Hello there.",
        player=AudioPlayer(stream_factory=lambda **kw: RefusingStream(**kw)),
    )

    feature.on_trigger()

    wait_until(lambda: any(msg == "Playback failed" for _, msg in notifier.messages))
    assert not any(msg == "Done" for _, msg in notifier.messages)
```
  (add `from typing import Any` and `from vocalis.audio.playback import AudioPlayer` where missing.)
  6. **`tests/test_cross_feature_integration.py`**: it patches `vocalis.cli.AudioPlayer` with a `_FakePlayer` that has `play()`/`done_event`. Replace `_FakePlayer` with `tests.fake_output.FakeAudioPlayer` (same `done_event` attribute, set when a reading's stream closes). Update any assertion on `play_calls` to `player.output.opened >= 1`.
  7. **`tests/test_shared_tts_engine.py`**: switch the Quick Reader player to `AudioPlayer(stream_factory=pump_stream_factory(OutputRecorder()))` too; delete `_InstantStream`.
  8. **`tests/test_reading_hooks_integration.py`**: its Quick Reader uses `ResettablePlayer(delay_seconds=3.0)` and reads `quick_player.play_calls`; change the reads to `quick_player.output.opened`.

- [ ] **Step 3: Run once and fix.** Run the touched test files, then the **full suite** once (`.venv/bin/python -m pytest -q -p no:cacheprovider`; only the 2 known offline-guard failures may remain) and `.venv/bin/mypy src tests`.

- [ ] **Step 4: Commit**

```bash
git add src/vocalis/features/quick_reader.py tests/test_quick_reader.py tests/test_cross_feature_integration.py tests/test_shared_tts_engine.py tests/test_reading_hooks_integration.py
git commit -m "feat: Quick Reader speaks through ChunkedSpeaker and reads long text in chunks"
```

---

## Task 8: Documentation, changelog and final verification

**Files:**
- Modify: `docs/architecture.md`, `docs/workflow.md`, `docs/file-purpose.md`, `docs/folder-structure.md`, `docs/developer-docs.md`, `docs/user-docs.md`
- Create: `.claude/2026-10-20_<HHMM>_CHANGELOG.md`

**Interfaces:** none (documentation only; no code changes).

- [ ] **Step 1: Write the docs** — read each file's existing Stage 6a section first and match its style. Required content:
  - `architecture.md`: a "Gapless long reading (Stage 6b)" section: the producer/consumer picture (`ChunkedSpeaker` → bounded queue → `SpeechStream` → one open sound stream), the voice held for the whole reading and released when playback ends (idle timer then starts), fades/underrun-silence/back-pressure, stall detection → "Playback failed", rate resampling after a voice switch, pause = silence + rewind of the heard piece, and that `AudioPlayer.play()` still exists for other callers.
  - `workflow.md`: update the Quick Reader and Document Reader flow descriptions (user view and bot logic): long clipboard text is now read in sentence chunks; progress "piece N of M" appears when a piece starts being heard; notification order ("Reading..." → copied text → "Done").
  - `file-purpose.md` and `folder-structure.md`: add `audio/speech_stream.py`, `features/chunked_speaker.py`, `tests/fake_output.py`; update `text/chunker.py` (sentence-aware, spans), `engines/chain.py` (hold/close, `new_reading_chain`, `make_downshift_handler`), `audio/playback.py` (`open_stream`/`close_stream`/`report_device_failed`), `features/quick_reader.py` and `features/doc_reader.py`.
  - `developer-docs.md`: how to test code that feeds a persistent stream (`tests/fake_output.py`: `ManualStream` vs `PumpStream`), and the "write all the code, then run the tests once" process the user asked for.
  - `user-docs.md`: one short paragraph — long readings now flow without gaps; a "Playback failed" after about 10 seconds of silence still means the sound device is the problem (same `~/.asoundrc` advice as before). Do not describe internals here.
  - Rulings 1–12 above are copied into `architecture.md` as "Decisions made for Stage 6b" so the reasoning survives.
- [ ] **Step 2: Changelog** — create `.claude/2026-10-20_<HHMM>_CHANGELOG.md` (use the real time) summarising Stage 6b by task, the new files, the deleted obsolete tests (list from Task 6 item 7), and the final numbers from Step 3.
- [ ] **Step 3: Final verification (once, nothing else running in this worktree):** `.venv/bin/python -m pytest -q -p no:cacheprovider` (only the 2 known offline-guard failures may remain; coverage ≥ 85%), `.venv/bin/mypy src tests` (clean), `bats tests/installer` (all ok — nothing here touches the installer, this is a regression check). Record the three results in the changelog.
- [ ] **Step 4: Commit**

```bash
git add docs .claude/*_CHANGELOG.md
git commit -m "docs: document gapless long reading (Stage 6b)"
```

---

## Self-review notes (done while writing)

- **Spec coverage (6b bullets):** shared `ChunkedSpeaker` for both features → Tasks 5–7; producer holds ONE acquire for the whole reading → Task 4 (`hold_engines`) + Task 5 (`close()` in `finally`); bounded 2–3 queue with back-pressure → Task 3; persistent output stream replacing the per-chunk stream → Tasks 2–3; fade at joins and zero-fill on underrun → Task 3; Quick Reader gains chunking → Task 7; chunker sentence-aware (abbreviations, bare newlines) with `(start, end)` spans → Task 1; idle timer must not fire mid-reading and counts from playback end → Task 4 + Task 5 + Task 6 test; pause/resume/stop unchanged for the user → Tasks 5–7; existing fakes rewritten, not discarded → Tasks 2, 6, 7; 6a-parked duplicated `_new_chain`/`_notify_downshift` → Task 4 (`new_reading_chain`, `make_downshift_handler`) used by both readers.
- **Placeholder scan:** no "TBD/TODO"; the only deliberately instruction-form (not code) test content is the mechanical port of ~40 existing tests in Tasks 6–7, specified as substitution rules over named, existing tests, with every genuinely new test written out in full.
- **Type consistency:** `SpeechStream.push(audio, sample_rate, tag)` / `on_clip_started(tag)` (Task 3) ↔ `ChunkedSpeaker._speak` pushes `index` and `_clip_started(tag)` reports `tag + 1` (Task 5) ↔ `on_chunk_started(current, total)` consumed by Document Reader's `_report_progress` (Task 6). `EngineChain.close()` satisfies `SpeechSource.close()`. `AudioPlayer.open_stream/close_stream/report_device_failed` (Task 2) are exactly what `SpeechStream` calls (Task 3). `_flip_active_state()` returns a 3-tuple everywhere in Task 7.
- **Known ordering risk:** between Task 6 and Task 7 Quick Reader still uses the old path and its own tests; the shared files (`test_shared_tts_engine.py`, `test_reading_hooks_integration.py`) are edited one side at a time so the suite stays green after each task.
