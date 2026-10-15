# Vocalis Stage 1 — Sound and Voice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Vocalis finds a working sound output by itself (fixing the silent default route on the author's machine), lets the user pick one in Settings with a "Test sound" button, adds a speaking-pace setting (default 1.15x) that applies to every reader, and adds voice-naturalness controls plus a "Preview voice" button.

**Architecture:** A new `OutputDeviceChooser` probes sound outputs (a silent stream whose audio callback must fire) and caches the first working one; `AudioPlayer` asks it which device to use and reports devices that stall. A new `apply_speed()` time-stretches finished audio (any engine) and is applied in Quick Reader and Document Reader (so OCR too) from a live setting. Settings gains Output / Test sound / Speed / Preview controls; the style control is added only if a measured spike shows the engine honours it.

**Tech Stack:** Python 3.13, sounddevice (PortAudio), librosa (already a dependency, imported lazily), numpy, PySide6, pytest + pytest-qt, mypy strict.

**Spec:** `docs/superpowers/specs/2026-09-11-vocalis-design.md` — section "Completion roadmap and Stage 1 amendment — Sound and voice".

## Global Constraints

- Work in a fresh git worktree on branch `worktree-vocalis-stage1-sound-voice`; run everything with `export PYTHONPATH=$PWD/src` and `.venv/bin/pytest` / `.venv/bin/mypy` (link `.venv` to the main checkout's `.venv`; never `git add` it).
- Conventional commits, **no Co-Authored-By or attribution lines**, explicit `git add` paths, never touch `old/`.
- Every function, method and class has a docstring. `mypy` strict: **zero errors project-wide**. Full suite **zero warnings**, coverage **>= 85%** (baseline at plan time: 448 tests, 96.12%). Run full-suite runs strictly one after another (never overlapping: it corrupts `.coverage`).
- New constructor parameters on existing classes are **optional keyword parameters with defaults**; existing tests stay green unchanged; never weaken an existing assertion. Tests deterministic (Events/bounded waits, no guessed sleeps).
- Qt lifetime rule (a flaky segfault was traced to a Qt object destroyed at a GC-dependent moment): no `QTimer`s; connect widget signals only to bound methods; every Qt object a task creates is owned by a parent or torn down explicitly.
- Never run the real `vocalis` program and never play real sound from tests or from implementers (use fake streams/devices). Do not touch the user's real `~/.config/vocalis`. The user tests everything only at the very end with the installer.
- Config additions: `audio_output_device: str = "auto"`, `speaking_speed: float` with `ge=0.7, le=1.6, default=1.15`. Pace is clamped to 0.7–1.6 everywhere.
- The Qwen model probe in Task 6 (the only task that loads a model) may use the GPU but must write only to the scratchpad directory and must free the model afterwards.
- Record the change log in `.claude/<YYYY-MM-DD_HHMM>_CHANGELOG.md`; keep `docs/` in sync.

## Review Focus

1. The output probe never plays an audible sound, is bounded (about 1 s per candidate, about 6 s total) and never blocks a reader thread longer than that (Task 2).
2. A device that stalls mid-run (stream opens, callback never fires) is marked failed; the next reading uses the next working device with ONE notification; no infinite retry loop; if nothing works a clear error is shown and the system default is tried last (Tasks 2, 3, 9).
3. A crazy speed value in the config file (e.g. 9.0 or -1) is rejected with a clear error at load; the slider and `apply_speed` also clamp (Tasks 1, 4, 8).
4. A failure inside the pace changer never breaks a reading: the original audio is used and a warning is logged (Task 5).
5. Pause/resume still replays the same (already sped-up) chunk correctly (Task 5).
6. "Test sound" and "Preview voice" pressed while a reading is playing must not crash (player busy) — preview preempts through the coordinator; test sound reports "busy" cleanly (Tasks 8, 9).
7. A stored output name that no longer exists (unplugged headset) is kept in the config but runtime behaves as "auto"; "auto" stays "auto" when the device list changes (Tasks 2, 8).
8. Offline: `librosa` is imported lazily inside `apply_speed` and nothing downloads data (Task 1).

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/vocalis/audio/pace.py` | create | `apply_speed`, `clamp_speed`, speed limits |
| `src/vocalis/audio/output_device.py` | create | `OutputDeviceChooser` (probe, choose, cache, mark failed, list names) |
| `src/vocalis/audio/playback.py` | modify | `AudioPlayer` takes a device provider + failure hook |
| `src/vocalis/config.py`, `config/default_config.toml` | modify | two new fields + defaults |
| `src/vocalis/settings.py` | modify | `speaking_speed()`, `audio_output_device()` accessors |
| `src/vocalis/features/quick_reader.py`, `doc_reader.py` | modify | optional `speed_provider`, pace after synthesis |
| `src/vocalis/engines/qwen3_tts.py` | modify | (Task 7, only if the spike says so) optional style instruction |
| `src/vocalis/ui/settings_window.py` | modify | Output, Test sound, Speed, Preview (+ Style) controls |
| `src/vocalis/cli.py` | modify | build the chooser, wire player/pace/test sound/preview |
| tests + docs + change log | create/modify | matching tests and documentation |

---

### Task 1: Speaking pace changer

**Files:** Create `src/vocalis/audio/pace.py`; Test `tests/test_pace.py`.

**Interfaces — Produces:** `MIN_SPEED = 0.7`, `MAX_SPEED = 1.6`, `clamp_speed(speed: float) -> float`, `apply_speed(audio: npt.NDArray[np.float32], sample_rate: int, speed: float) -> npt.NDArray[np.float32]`.

- [ ] **Step 1: Failing tests** — create `tests/test_pace.py`:

```python
"""Tests for vocalis.audio.pace (real librosa, tiny synthetic audio, no models)."""
import numpy as np

from vocalis.audio.pace import MAX_SPEED, MIN_SPEED, apply_speed, clamp_speed

RATE = 24000


def sine(freq: float = 440.0, seconds: float = 1.0) -> "np.ndarray":
    t = np.arange(int(RATE * seconds)) / RATE
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def dominant_frequency(audio: "np.ndarray") -> float:
    spectrum = np.abs(np.fft.rfft(audio))
    return float(np.argmax(spectrum) * RATE / len(audio))


def test_clamp_keeps_values_inside_the_limits() -> None:
    assert clamp_speed(0.1) == MIN_SPEED
    assert clamp_speed(9.0) == MAX_SPEED
    assert clamp_speed(1.2) == 1.2
    assert clamp_speed(float("nan")) == 1.0


def test_speed_one_returns_the_same_audio_object() -> None:
    audio = sine()
    assert apply_speed(audio, RATE, 1.0) is audio


def test_faster_speed_shortens_and_slower_lengthens_proportionally() -> None:
    audio = sine(seconds=2.0)
    fast = apply_speed(audio, RATE, 1.5)
    slow = apply_speed(audio, RATE, 0.8)
    assert abs(len(fast) - len(audio) / 1.5) < 0.03 * len(audio)
    assert abs(len(slow) - len(audio) / 0.8) < 0.03 * len(audio)
    assert fast.dtype == np.float32 and slow.dtype == np.float32


def test_pitch_is_preserved() -> None:
    fast = apply_speed(sine(440.0, 2.0), RATE, 1.4)
    assert abs(dominant_frequency(fast) - 440.0) < 12.0


def test_silence_stays_silent() -> None:
    out = apply_speed(np.zeros(RATE, dtype=np.float32), RATE, 1.3)
    assert float(np.max(np.abs(out))) < 1e-6


def test_empty_and_tiny_inputs_are_returned_unchanged() -> None:
    empty = np.zeros(0, dtype=np.float32)
    tiny = np.zeros(100, dtype=np.float32)
    assert apply_speed(empty, RATE, 1.3) is empty
    assert apply_speed(tiny, RATE, 1.3) is tiny


def test_out_of_range_speed_is_clamped_not_rejected() -> None:
    audio = sine(seconds=2.0)
    assert abs(len(apply_speed(audio, RATE, 9.0)) - len(audio) / MAX_SPEED) < 0.03 * len(audio)
```

- [ ] **Step 2: Run, verify FAIL** — `.venv/bin/pytest tests/test_pace.py -q --no-cov` → `ModuleNotFoundError`.
- [ ] **Step 3: Implement** — create `src/vocalis/audio/pace.py`:

```python
"""Changes the speaking pace of finished speech audio.

The voice engines have no speed control, so the pace is applied to the
finished audio with a pitch-preserving time stretch (librosa's phase
vocoder). librosa is imported inside the function so merely importing
Vocalis does not pay its start-up cost, and nothing here touches the
network.
"""
import logging

import numpy as np
import numpy.typing as npt

logger = logging.getLogger("vocalis")

MIN_SPEED = 0.7
MAX_SPEED = 1.6
_NEUTRAL_TOLERANCE = 0.01
_MIN_SAMPLES = 4096


def clamp_speed(speed: float) -> float:
    """Limit a speed factor to the supported range (NaN counts as 1.0)."""
    if speed != speed:  # NaN
        return 1.0
    return max(MIN_SPEED, min(MAX_SPEED, speed))


def apply_speed(
    audio: npt.NDArray[np.float32], sample_rate: int, speed: float
) -> npt.NDArray[np.float32]:
    """Return `audio` played `speed` times faster (>1) or slower (<1).

    Pitch is preserved. The speed is clamped to MIN_SPEED..MAX_SPEED.
    Audio that is empty, too short to stretch, or whose speed is within 1%
    of 1.0 is returned unchanged (the same object).

    Args:
        audio: 1-D float32 samples.
        sample_rate: Sample rate in Hz (kept for the signature's clarity;
            the stretch itself is rate-independent).
        speed: Desired speed factor.
    """
    speed = clamp_speed(speed)
    if abs(speed - 1.0) < _NEUTRAL_TOLERANCE or audio.size < _MIN_SAMPLES:
        return audio
    import librosa

    stretched = librosa.effects.time_stretch(audio.astype(np.float32, copy=False), rate=speed)
    return np.asarray(stretched, dtype=np.float32)
```

- [ ] **Step 4: Run, verify PASS** — `.venv/bin/pytest tests/test_pace.py -q --no-cov && .venv/bin/mypy` (`Success`; add a narrow per-module `ignore_missing_imports` for `librosa` in pyproject only if mypy asks for it).
- [ ] **Step 5: Commit** — `git add src/vocalis/audio/pace.py tests/test_pace.py (+pyproject.toml if changed) && git commit -m "feat: add a pitch-preserving speaking pace changer"`

---

### Task 2: OutputDeviceChooser

**Files:** Create `src/vocalis/audio/output_device.py`; Test `tests/test_output_device.py`.

**Interfaces — Produces:** `OutputDeviceChooser(preference: Callable[[], str] = lambda: "auto", query_devices=sd.query_devices, stream_factory=sd.OutputStream, probe_timeout_seconds: float = 1.0, total_timeout_seconds: float = 6.0)` with `output_names() -> list[str]` (unique names of devices that have output channels, in a stable order), `current() -> int | None` (index of a working output, probing lazily and caching per preference value; `None` when nothing works, meaning "let PortAudio use its default"), `mark_failed(device: int | None) -> None` (remember the index as bad, drop the cache), `reset() -> None` (forget bad devices and the cache), `probe(index: int) -> bool`.

Candidate order: if the preference is not `"auto"` (case-insensitive) the device(s) whose name equals it, then devices named `pulse`, `pipewire`, `default` (in that order), then every other output device; devices already marked failed are skipped. A probe = open `stream_factory(device=index, samplerate=24000, channels=1, dtype="float32", callback=cb)` where `cb` zero-fills `outdata` and sets a `threading.Event`; start it; wait up to `probe_timeout_seconds` for the event; always `stop()` and `close()` in a `finally` (exceptions during cleanup are logged, never raised); any exception while opening/starting → probe False. `current()` stops probing when `total_timeout_seconds` is exhausted. A lock serialises probes; the cache is guarded by a separate lock; no lock is held while waiting on the audio callback.

- [ ] **Step 1: Failing tests** — create `tests/test_output_device.py` with a fake device list and fake stream factory:

```python
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
    def __init__(self, behaviour: str, callback: Any) -> None:
        self.behaviour = behaviour
        self.callback = callback
        self.closed = False
        self.stopped = False

    def start(self) -> None:
        if self.behaviour == "raise":
            raise OSError("cannot open")
        if self.behaviour == "works":
            import numpy as np

            self.callback(np.zeros((256, 1), dtype="float32"), 256, None, None)

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True


class Factory:
    """Behaviour per device index: works / silent (callback never fires) / raise."""

    def __init__(self, behaviours: "dict[int, str]") -> None:
        self.behaviours = behaviours
        self.opened: "list[int]" = []
        self.streams: "list[FakeStream]" = []

    def __call__(self, **kwargs: Any) -> FakeStream:
        index = kwargs["device"]
        self.opened.append(index)
        stream = FakeStream(self.behaviours.get(index, "silent"), kwargs["callback"])
        self.streams.append(stream)
        return stream


def make(behaviours: "dict[int, str]", preference: str = "auto", probe_timeout: float = 0.05):  # type: ignore[no-untyped-def]
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
    import time

    chooser, factory = make({}, probe_timeout=0.2)
    chooser._total_timeout = 0.3  # noqa: SLF001
    started = time.monotonic()
    assert chooser.current() is None
    assert time.monotonic() - started < 1.0
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
    chooser, factory = make({3: "works"})
    results: "list[int | None]" = []
    threads = [threading.Thread(target=lambda: results.append(chooser.current())) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)
    assert results == [3] * 6
    assert factory.opened == [3]
```
(The attribute `_total_timeout` in the bounded-budget test is the stored `total_timeout_seconds`; keep that name.)

- [ ] **Step 2: Run, verify FAIL** (`ModuleNotFoundError`).
- [ ] **Step 3: Implement** `src/vocalis/audio/output_device.py` as specified above (module docstring: what a probe is and why — "the ALSA default device on some machines starts a stream whose callback never fires"; class and method docstrings; `logger.info` once when a device is chosen with its name, `logger.warning` for each failed candidate, `logger.error` when none works). Implementation notes: `query_devices()` returns a list of dict-like entries (use `d["name"]`, `d["max_output_channels"]`); the preference compare is case-insensitive on the full name; `current()` takes `_probe_lock`, re-checks the cache inside it (so concurrent callers probe once), probes candidates in order until one works or the total budget is spent, caches `{preference_value: index}`; `mark_failed(device)` adds a non-None index to `_bad` and clears the cache; `None` is a no-op.
- [ ] **Step 4: Run, verify PASS** — `.venv/bin/pytest tests/test_output_device.py -q --no-cov && .venv/bin/mypy`; run the test file 10x.
- [ ] **Step 5: Commit** — `git add src/vocalis/audio/output_device.py tests/test_output_device.py && git commit -m "feat: add a sound output chooser that probes for a working device"`

---

### Task 3: AudioPlayer uses the chooser

**Files:** Modify `src/vocalis/audio/playback.py`; Test `tests/test_playback.py` (append only).

**Interfaces — Consumes:** nothing new (takes plain callables). **Produces:** `AudioPlayer(stream_factory=None, timeout_margin_seconds=10.0, device_provider: Callable[[], int | None] | None = None, on_device_failed: Callable[[int | None], None] | None = None)`.

Behaviour: in `play()`, `device = self._device_provider() if self._device_provider else None`; remember it as `self._current_device`; when `device is not None` add `device=device` to the stream factory kwargs (when None, pass no `device` key, so existing tests/behaviour are unchanged). When `wait_until_done()` raises `PlaybackTimeoutError`, and when `stream.start()` raises, call `self._on_device_failed(self._current_device)` (inside try/except Exception that logs; never let a broken hook change the original exception) before the error propagates. A normal finish or an early `stop()` never calls the hook. The provider must be called OUTSIDE `self._lock`.

- [ ] **Step 1: Failing tests** (append; reuse that file's fake stream factory and short timeout margins — read the file first): the provider's device is passed as the `device` kwarg; no provider → no `device` kwarg (assert the kwargs of the fake factory); a provider returning None → no `device` kwarg; a timeout (callback never fires with a tiny `timeout_margin_seconds`) calls `on_device_failed` once with that device and still raises `PlaybackTimeoutError`; a failing `start()` calls it once and re-raises the original exception; natural finish and `stop()` do not call it; a raising hook does not replace the original exception.
- [ ] **Step 2: Run, verify FAIL; Step 3: Implement; Step 4: PASS** — `.venv/bin/pytest tests/test_playback.py -q --no-cov && .venv/bin/mypy`.
- [ ] **Step 5: Commit** — `git add src/vocalis/audio/playback.py tests/test_playback.py && git commit -m "feat: let the audio player use a chosen output and report a stalled one"`

---

### Task 4: Config fields and live accessors

**Files:** Modify `src/vocalis/config.py`, `config/default_config.toml`, `src/vocalis/settings.py`; Test `tests/test_config.py`, `tests/test_settings.py` (append).

**Interfaces — Produces:** `VocalisConfig.audio_output_device: str = "auto"`, `VocalisConfig.speaking_speed: float = Field(default=1.15, ge=0.7, le=1.6)`; `LiveSettings.audio_output_device() -> str`, `LiveSettings.speaking_speed() -> float`.

- [ ] **Step 1: Failing tests:** defaults (`"auto"`, `1.15`); `speaking_speed` 0.5 / 9.0 / -1 are rejected by `load_config` of a TOML containing them with a `ConfigError` naming the field; 0.7 and 1.6 accepted; round trip through `save_config`/`load_config` keeps both; the existing `test_default_toml_matches_model_defaults` stays green only if `config/default_config.toml` gains the two lines (that is the proof they are in sync); `LiveSettings` accessors return the live values after `update(...)`.
- [ ] **Step 2: Run, verify FAIL; Step 3: Implement** (add the two fields after `floating_widget_enabled`; add both keys to `config/default_config.toml` with a short comment; add the accessors with docstrings). **Step 4: PASS** — `.venv/bin/pytest tests/test_config.py tests/test_settings.py -q --no-cov && .venv/bin/mypy`.
- [ ] **Step 5: Commit** — `git add src/vocalis/config.py config/default_config.toml src/vocalis/settings.py tests/test_config.py tests/test_settings.py && git commit -m "feat: add sound output and speaking pace settings"`

---

### Task 5: Readers apply the pace

**Files:** Modify `src/vocalis/features/quick_reader.py`, `src/vocalis/features/doc_reader.py`; Test `tests/test_quick_reader.py`, `tests/test_doc_reader.py` (append only).

**Interfaces — Consumes:** Task 1 `apply_speed`. **Produces:** `QuickReaderFeature(..., speed_provider: Callable[[], float] | None = None)` and `DocumentReaderFeature(..., speed_provider: ... = None)` as the LAST optional keyword parameters, stored as `self._speed_provider`.

Behaviour: right after `_synthesize_with_fallback(...)` returns `(audio, sample_rate)` (Quick Reader around line 238, Document Reader per chunk around line 372) call a shared private helper `self._paced(audio, sample_rate)` that returns `apply_speed(audio, sample_rate, self._speed_provider())` when a provider is set, and the original audio when not; any exception inside it is caught, `logger.warning(..., exc_info=True)`, and the original audio is used (a pace failure must never break a reading). Implement the helper once in a tiny shared place (for example a function `paced_audio(audio, sample_rate, provider)` in `vocalis/audio/pace.py` used by both features) rather than duplicating the try/except. Pause/resume is unchanged: the chunk that is replayed on resume is the already-stretched audio.

- [ ] **Step 1: Failing tests** (reuse each file's fake synthesizer/player; make the fake synthesizer return a sine long enough, e.g. 2 s at 24000 Hz): with `speed_provider=lambda: 1.5` the audio handed to the fake player is ~1/1.5 as long (within 3%); no provider → unchanged length; provider returning 1.0 → same object/length; the provider is read per reading/per chunk (change the value between two chunks of a document and see two different lengths); a provider that raises → the reading still plays the original audio and a warning was logged (use caplog); Document Reader pause then resume replays the same stretched chunk (length equal on both plays).
- [ ] **Step 2: Run, verify FAIL; Step 3: Implement; Step 4: PASS** — the two files 10x, `.venv/bin/mypy`.
- [ ] **Step 5: Commit** — `git add src/vocalis/audio/pace.py src/vocalis/features/quick_reader.py src/vocalis/features/doc_reader.py tests/test_pace.py tests/test_quick_reader.py tests/test_doc_reader.py && git commit -m "feat: apply the speaking pace to every reading"`

---

### Task 6: Spike — does the 0.6B voice honour a style hint?

**Files:** none in the repo (the script and results live in the scratchpad directory; the verdict goes into the task report and the ledger). **No source changes, no commit.**

- [ ] **Step 1:** In the scratchpad directory write `style_probe.py` that, with `PYTHONPATH=$PWD/src` and the model at `~/.local/share/vocalis/models/qwen3-tts-customvoice-0.6b`, loads `Qwen3TTSModel` the way `Qwen3TTS.load()` does (bfloat16 on cuda:0, sdpa attention, `TORCH_DISABLE_NATIVE_JIT=1`), then for speaker `Ryan` and this fixed sentence — "The quick brown fox jumps over the lazy dog, and then it quietly walks home." — generates audio with: no `instruct`; `instruct="Speak naturally and conversationally."`; `instruct="Speak slowly and calmly."`; `instruct="Speak quickly and with energy."`; each repeated 3 times with `do_sample=False` (deterministic) and also once with the defaults. Free the model (`del`, `torch.cuda.empty_cache()`) at the end.
- [ ] **Step 2:** For every output compute and print a table: duration (s), RMS, mean absolute difference from the no-instruct audio after trimming to equal length (is it identical?), and a rough pitch statistic (median fundamental from `librosa.yin`, fmin 60, fmax 400). Run each configuration at least twice to see run-to-run variation.
- [ ] **Step 3: Verdict** (write it in the report): **EFFECTIVE** if the instructions change duration by more than 8% or median pitch by more than 5% in the direction the text asks for, consistently over repeats and beyond the run-to-run variation; otherwise **NOT EFFECTIVE** (the 0.6B model ignores `instruct`). Also record the measured effect of `temperature`/`top_p` changes (0.6 vs 0.9) on duration variance. Do not claim anything about "naturalness": only the user can judge by ear.

---

### Task 7: Voice style setting (only if Task 6 says EFFECTIVE)

**Files:** Modify `src/vocalis/engines/qwen3_tts.py`, `src/vocalis/config.py`, `config/default_config.toml`, `src/vocalis/settings.py`; Test `tests/test_qwen3_tts.py`, `tests/test_config.py`, `tests/test_settings.py`.

If the Task 6 verdict is NOT EFFECTIVE the controller skips this task, records `Ruling: style hint skipped — the 0.6B model ignored instruct (measured)` in the ledger, and the Settings window simply has no style control; sampling defaults are left as they are.

**If EFFECTIVE:** `STYLES: dict[str, str]` in `settings.py` mapping the Settings label to the instruction text, exactly: `"Natural"` -> `"Speak naturally and conversationally."`, `"Calm"` -> `"Speak slowly and calmly."`, `"Energetic"` -> `"Speak quickly and with energy."`, `"Default"` -> `""` (no instruction; the existing behaviour). Config field `voice_style: str = "Default"` (validated against the keys of `STYLES`, unknown values rejected with a clear message); `LiveSettings.voice_style() -> str` returns the instruction TEXT for the current label; `Qwen3TTS(style: str | Callable[[], str] = "")` stores it like `speaker` and passes `instruct=<text>` to `generate_custom_voice` only when non-empty (re-read on every synthesis, like the voice). Tests: `instruct` is passed only when non-empty and re-read per call (fake model that records kwargs); config rejects an unknown style; round trip; accessor returns the text. Commit: `feat: add a speaking style setting for the voice`.

---

### Task 8: Settings window controls

**Files:** Modify `src/vocalis/ui/settings_window.py`; Test `tests/test_settings_window.py` (append; existing tests unchanged).

**Interfaces:** `SettingsWindow(settings, on_applied=..., output_names: Callable[[], list[str]] = lambda: [], on_test_sound: Callable[[str], str] = lambda _o: "", on_preview_voice: Callable[[], None] = lambda: None)` (new parameters appended, all optional). New widgets: `_output_combo` (first item `"Automatic (recommended)"` which stores `"auto"`, then `output_names()`; a stored name that is not in the list is added so it stays selectable and is kept), `_test_sound_button` ("Test sound"; calls `on_test_sound(selected_value)` and shows the returned text in `_message_label`; an exception from the callback is caught and shown as "Sound test failed: <reason>"), `_speed_slider` (`QSlider`, horizontal, minimum 70, maximum 160, value = round(speed*100)) with `_speed_label` showing `"1.15x"` style text that updates as the slider moves, `_preview_button` ("Preview voice (uses saved settings)"; calls `on_preview_voice()` inside try/except showing "Preview failed: <reason>"), and, if Task 7 was implemented, `_style_combo` (the STYLES keys). `_config_from_widgets` writes `audio_output_device` (`"auto"` or the chosen name) and `speaking_speed` (slider/100) (and `voice_style`); `load_from_settings` fills them (an out-of-range stored speed is clamped for display only; it can only come from a hand-edited file and is rejected at load anyway).

- [ ] **Step 1: Failing tests** (qtbot; tmp_path settings): defaults shown (Automatic selected, slider 115, label "1.15x"); moving the slider updates the label; saving writes `audio_output_device` and `speaking_speed` to the file and `settings.speaking_speed()` changes live; choosing a named output saves its name; a stored unknown output name stays selectable and is kept on save; the test-sound button passes the selected value ("auto" for Automatic) to the callback and shows its text; a raising test-sound callback shows "Sound test failed"; the preview button calls its callback once and a raising one shows "Preview failed"; the speed slider can never produce a value outside 0.7–1.6; (if Task 7) the style combo saves and reloads.
- [ ] **Step 2: Run, verify FAIL; Step 3: Implement** (add the rows to the form: "Sound output" with the combo and the Test button on one row, "Speaking speed" with slider and label, "Voice preview" button; every new function with a docstring; keep the existing duplicate `self._voice_combo = _combo(VOICES)` line out — remove that accidental duplicate while you are here). **Step 4: PASS** — `.venv/bin/pytest tests/test_settings_window.py -q --no-cov && .venv/bin/mypy`, 10x loop, then the full suite twice sequentially.
- [ ] **Step 5: Commit** — `git add src/vocalis/ui/settings_window.py tests/test_settings_window.py && git commit -m "feat: add sound output, speed and preview controls to the settings window"`

---

### Task 9: Wire it together in cli.py

**Files:** Modify `src/vocalis/cli.py`; Test `tests/test_cli.py` (append; existing tests unchanged).

**Interfaces — Consumes:** Tasks 1-5, 8. **Produces:** builders accept an optional `output_chooser: OutputDeviceChooser | None = None`; `build_ui(...)` accepts the extra callables for the Settings window.

Behaviour: `main()` creates ONE `OutputDeviceChooser(preference=settings.audio_output_device)` (real `sounddevice` defaults) right after `LiveSettings`, and passes it to `build_quick_reader` and `build_doc_reader`; each builds its `AudioPlayer(device_provider=chooser.current, on_device_failed=hook)` where `hook` marks the device failed on the chooser and sends ONE notification "Sound output changed because it stopped working" (guard with a per-chooser flag so repeated failures notify once per switch, not per call) — with `output_chooser=None` (old signature) the builders keep `AudioPlayer()` exactly as today. `speed_provider=settings.speaking_speed` is passed to the Quick Reader and the Document Reader when `settings` is given. `build_ui` passes to `SettingsWindow`: `output_names=chooser.output_names`; `on_test_sound` = a function that (1) refuses politely with "A reading is playing right now — try again when it has finished." if the shared Document/Quick reader is active (check `document_reader.is_reading`), otherwise (2) resolves the selected value (`"auto"` -> `chooser.current()`, a name -> a one-off probe of that name via the chooser's probe) and plays a 0.4 s quiet 440 Hz tone (amplitude 0.08) through an `AudioPlayer` using that device, waits for it (bounded), and returns "Sound test OK (<device name>)" or "Sound test failed: no working output found" / the error text; `on_preview_voice` = `document_reader.read_text(PREVIEW_SENTENCE)` with `PREVIEW_SENTENCE = "This is how Vocalis sounds with your current voice and speed."` (it preempts any current reading through the coordinator and shows the widget as usual). Constants live at module level; each function has a docstring; no real sound in tests.

- [ ] **Step 1: Failing tests** (patch `sounddevice` pieces and use fakes; follow the existing test_cli patterns): with `output_chooser` given, the quick/doc players have a device provider that returns the chooser's current device and a failure hook that calls `chooser.mark_failed` and notifies once per switch (fake notifier); with `output_chooser=None` the player is exactly the default; `speed_provider` is the live `settings.speaking_speed` for both readers (change the setting, assert the provider's value changes); the preview callback calls `document_reader.read_text` with `PREVIEW_SENTENCE`; the test-sound callback returns the busy message when a reading is active, returns "Sound test OK …" with a fake player/chooser when idle, and "Sound test failed …" when the chooser returns None or the player raises; `main()` still passes all existing tests and builds one shared chooser (assert both builders received the same object via patched builders).
- [ ] **Step 2: Run, verify FAIL; Step 3: Implement; Step 4: PASS** — `.venv/bin/pytest tests/test_cli.py -q --no-cov && .venv/bin/mypy`, 10x loop of test_cli.py, then the full suite three times sequentially (no segfault, zero warnings).
- [ ] **Step 5: Commit** — `git add src/vocalis/cli.py tests/test_cli.py && git commit -m "feat: use the working sound output, the speaking pace and the new settings in the app"`

---

### Task 10: Docs, change log, verification

**Files:** Modify `docs/user-docs.md`, `docs/architecture.md`, `docs/workflow.md`, `docs/file-purpose.md`, `docs/folder-structure.md`, `docs/developer-docs.md`, the spec's Stage 1 amendment (implementation deviations list); Create `.claude/<YYYY-MM-DD_HHMM>_CHANGELOG.md`.

- [ ] **Step 1: Docs** (verify every sentence against the code): the Sound output setting (Automatic vs a named output, what the probe does, that it plays nothing audible, the automatic switch + one notification when an output stops working), Test sound, Speaking speed (0.7–1.6, default 1.15, applies on the next reading, same for the backup voice), Preview voice (uses SAVED settings; preempts a current reading), the style setting (only if implemented; otherwise one sentence in developer docs explaining the measured spike result and that the style hint is not offered), how a developer adds a new audio device behaviour, the removal of the "silent default route" known issue from the docs/change log (replace with the new behaviour), the config fields and their limits, and a note that librosa is imported lazily.
- [ ] **Step 2: Change log** with files, tests (counts, coverage), the spike verdict and numbers, rulings from the ledger, deferred minors, and `- [ ] Real-world check pending: Test sound, speed slider, preview voice, automatic output switch — done by the user after installing with the Stage 3 installer`.
- [ ] **Step 3: Verification:** full suite three times sequentially (all pass, zero warnings, coverage >= 85%, no segfault), `.venv/bin/mypy` (Success), and loop `tests/test_pace.py tests/test_output_device.py tests/test_playback.py tests/test_settings_window.py tests/test_cli.py` 10x (all green every time).
- [ ] **Step 4: Offscreen smoke script** in the scratchpad (never in the repo, no real sound): build a `SettingsWindow` with a fake chooser/callbacks, change speed and output, save through the window, print the saved file, call `apply_speed` on a generated sine for 0.8/1.15/1.5 and print the resulting lengths, and run an `AudioPlayer` with a fake stream factory and a fake chooser to show the stalled-device switch. Confirm the user's real config was not touched.
- [ ] **Step 5: Commit** — `git add docs .claude/*CHANGELOG.md && git commit -m "docs: document the sound output, speaking pace and voice controls"`
