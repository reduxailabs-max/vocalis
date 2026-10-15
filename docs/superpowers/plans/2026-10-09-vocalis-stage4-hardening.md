# Vocalis Stage 4 — Phase 9 Testing & Docs Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove that Vocalis's features behave correctly *together* in one running process (not just individually, as every existing test already does), and give the project a proper front door (`README.md`) plus docs that match the code exactly, as the last stage before the user installs and tests the real app.

**Architecture:** One new test file, `tests/test_cross_feature_integration.py`, reuses `cli.py`'s own private builder functions (`build_dictation`, `build_quick_reader`, `build_doc_reader`, `build_ocr_reader`) — never reinvented wiring — against one shared `ModelManager`/`ReadingCoordinator`, with every external dependency (microphone, speaker, clipboard, screen capture, OCR, the two AI engine classes) replaced by the same kind of fakes every other test file in this project already uses. A new root `README.md` links out to the installer and the existing `docs/user-docs.md` rather than duplicating them. A documentation accuracy pass re-reads every `docs/*.md` file against the current code and fixes any drift.

**Tech Stack:** Python 3.13, pytest + pytest-qt (offscreen Qt), `unittest.mock.patch` on `vocalis.cli`'s imported names (the same idiom already used throughout `tests/test_cli.py`), mypy strict.

**Spec:** `docs/superpowers/specs/2026-09-11-vocalis-design.md` — section "Stage 4 amendment — Phase 9, Testing & docs hardening".

## Global Constraints

- Work in a fresh git worktree on branch `worktree-vocalis-stage4-hardening`; run everything with `export PYTHONPATH=$PWD/src` and `.venv/bin/pytest` / `.venv/bin/mypy` (link `.venv` to the main checkout's `.venv`; never `git add` it).
- Conventional commits, **no Co-Authored-By or attribution lines**, explicit `git add` paths, never touch `old/`.
- Every function/class keeps a docstring. `mypy` strict **zero errors** project-wide. Full suite **zero warnings**, coverage **>= 85%** (baseline at plan time: 602 tests — confirm the exact current count by running the suite once before Task 1).
- **Never touch real hardware, a real GPU model, or the real network in any automated test** — every external dependency the whole-system tests exercise (`ParakeetSTT`, `Qwen3TTS`, `PiperTTS`, `MicRecorder`, `AudioPlayer`, `ClipboardReader`, `ClipboardWriter`, `ScreenRegionCapture`, `TesseractOCR`, and the three hotkey-device-enumerator functions) is patched inside `vocalis.cli`'s own namespace — the same way every existing test in `tests/test_cli.py` already does it. Building a real `DocumentReaderPopup` needs a real `QApplication` instance first (it's a `QWidget`) — use the existing `qapp`/`qtbot` pytest-qt fixtures already used elsewhere in this test suite, never construct one by hand in a new test file.
- No new product feature, config field, or CLI behavior in this stage. This stage only adds tests and documentation.
- No hands-on testing in this stage either — per the user's standing instruction, the first real, hands-on use of Vocalis happens only once the user runs the installer themselves, after this stage merges.
- Record the change log in `.claude/<YYYY-MM-DD_HHMM>_CHANGELOG.md`; keep `docs/` in sync.

## Review Focus

1. A whole-system test that only asserts `status == 0` or a loosely-matching substring, the way two tests earlier in this project's own history turned out to (caught during Stage 3's reviews), would pass even if the thing it claims to prove were broken — every new assertion here must be checked against what it would do if the underlying fix were reverted, not just read for plausibility (Tasks 1-3).
2. The two TTS features (Quick Reader, Document Reader) **share one engine id** (`TTS_ENGINE_ID = "qwen3_tts"`) on purpose — switching between them must NOT force a reload, only switching to/from Dictation's `"parakeet"` id should. A scenario that asserts a reload on every feature switch would be testing the wrong thing (Task 1).
3. `ModelEngine.load()`/`unload()` being called the *expected number of times* matters more than just checking `current_engine_id`'s final value — a bug that reloads needlessly, or never unloads at all, can still leave `current_engine_id` looking correct at the end (Task 1).
4. The background-thread nature of `on_trigger()`/`read_text()` (each spawns a daemon thread) means a test that doesn't wait on a deterministic signal (a `threading.Event` the fake player/engine sets) before asserting final state is flaky by construction — no `time.sleep`-based waiting anywhere in the new file (Tasks 1-3).
5. The documentation sweep (Task 5) must find and fix REAL drift, not just skim — each doc file's claims (file paths, function names, behavior descriptions) must be checked against the actual current code, not reproduced from an earlier stage's version of this same file.

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `tests/test_cross_feature_integration.py` | create | shared fakes + all 5 whole-system scenarios |
| `README.md` | create | root-level front door, links out, no duplication |
| `docs/user-docs.md`, `docs/architecture.md`, `docs/workflow.md`, `docs/file-purpose.md`, `docs/folder-structure.md`, `docs/developer-docs.md`, `docs/algorithms.md` | modify (as needed) | accuracy sweep — fix any drift found |
| `.claude/<timestamp>_CHANGELOG.md` | create | what this stage did, final verification numbers |

---

### Task 1: Shared fakes + Scenario 1 (one-GPU-at-a-time) + Scenario 2 (hotkeys don't cross-trigger)

**Files:**
- Create: `tests/test_cross_feature_integration.py`

**Interfaces:**
- Consumes: `vocalis.cli.build_dictation`, `build_quick_reader`, `build_doc_reader`, `build_ocr_reader` (all already exist, unchanged); `vocalis.model_manager.ModelManager`, `TTS_ENGINE_ID` ("qwen3_tts"); `vocalis.features.reading_coordinator.ReadingCoordinator`; `vocalis.config.VocalisConfig`.
- Produces (used by Tasks 2-3): the module-level fake classes `_FakeEngine`, `_FakeRecorder`, `_FakePlayer`, `_FakeClipboardReader` defined in this file — later tasks append to this same file and reuse them directly (no import needed, same module).

The fakes below and both scenarios' test code have already been written and run successfully against the real codebase while writing this plan — this is verified, working code, not a sketch.

- [ ] **Step 1: Write the file with its shared fakes and both scenarios**

```python
"""Whole-system tests: do multiple Vocalis features behave correctly
TOGETHER in one process? Every existing test in this project (602 at
last count) exercises one feature at a time, heavily faked. These tests
reuse cli.py's own private builder functions (never reinvented wiring)
against one shared ModelManager/ReadingCoordinator, with every external
dependency (microphone, speaker, clipboard, screen capture, OCR, the two
AI engine classes) faked the same way tests/test_cli.py already fakes
them -- never real hardware, a real GPU model, or the real network.
"""
import threading
from unittest.mock import patch

import numpy as np
import numpy.typing as npt

from vocalis.cli import build_dictation, build_doc_reader, build_ocr_reader, build_quick_reader
from vocalis.config import VocalisConfig
from vocalis.features.reading_coordinator import ReadingCoordinator
from vocalis.model_manager import ModelManager, TTS_ENGINE_ID


class _FakeEngine:
    """Fake ModelEngine (STT or TTS); records load/unload calls."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.load_count = 0
        self.unload_count = 0
        self._loaded = False

    def load(self) -> None:
        """Mark this fake engine loaded and count the call."""
        self.load_count += 1
        self._loaded = True

    def unload(self) -> None:
        """Mark this fake engine unloaded and count the call."""
        self.unload_count += 1
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Whether load() has run more recently than unload()."""
        return self._loaded

    def transcribe(self, audio: "npt.NDArray[np.float32]", sample_rate: int) -> str:
        """Fake transcription result, ignoring the input."""
        return "fake transcription"

    def synthesize(self, text: str) -> "tuple[npt.NDArray[np.float32], int]":
        """Fake synthesis result: a fixed 10-sample silent clip at 24kHz."""
        return np.zeros(10, dtype=np.float32), 24000


class _FakeRecorder:
    """Fake MicRecorder: start() does nothing, stop() returns fixed audio."""

    def start(self) -> None:
        """No-op: no real microphone is ever opened."""

    def stop(self) -> "npt.NDArray[np.float32]":
        """Return a fixed 100-sample buffer, never real audio."""
        return np.ones(100, dtype=np.float32)


class _FakePlayer:
    """Fake AudioPlayer: records play() calls and signals completion via
    a real threading.Event, so a test can wait() on it deterministically
    instead of sleeping -- on_trigger()/read_text() each run on a
    background thread, so this is the only reliable way to know the
    fake "playback" has actually finished."""

    def __init__(self) -> None:
        self.done_event = threading.Event()
        self.played: "list[tuple[npt.NDArray[np.float32], int]]" = []

    def play(self, audio: "npt.NDArray[np.float32]", sample_rate: int) -> None:
        """Record what was "played"."""
        self.played.append((audio, sample_rate))

    def wait_until_done(self) -> bool:
        """Signal done_event, then report a normal (non-stopped) finish."""
        self.done_event.set()
        return True

    def stop(self) -> None:
        """No-op: nothing real is playing to stop."""


class _FakeClipboardReader:
    """Fake ClipboardReader: always returns the same fixed text."""

    def read(self) -> str:
        """Return fixed clipboard text, never touching a real clipboard."""
        return "clipboard text"


class _FakePiper:
    """Fake PiperTTS: never checks for a real binary/voice file."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Accept any constructor arguments; store nothing real."""

    def load(self) -> None:
        """No-op: no real file-existence check."""

    def synthesize(self, text: str) -> "tuple[npt.NDArray[np.float32], int]":
        """Fake synthesis result, used only if the primary engine fails."""
        return np.zeros(5, dtype=np.float32), 22050


def test_one_gpu_model_at_a_time_holds_across_features(qapp) -> None:  # type: ignore[no-untyped-def]
    """Dictate, then Quick Reader. ModelManager must unload the STT
    engine and load the TTS engine exactly once each -- proving the
    single-GPU-residency rule holds ACROSS features, not just within
    one feature's own already-tested local wiring."""
    config = VocalisConfig()
    manager = ModelManager(idle_timeout_seconds=9999.0)
    coordinator = ReadingCoordinator()
    parakeet_engine = _FakeEngine("parakeet")
    qwen_engine = _FakeEngine("qwen")

    with (
        patch("vocalis.cli.ParakeetSTT", return_value=parakeet_engine),
        patch("vocalis.cli.MicRecorder", return_value=_FakeRecorder()),
        patch("vocalis.cli.Qwen3TTS", return_value=qwen_engine),
        patch("vocalis.cli.PiperTTS", _FakePiper),
        patch("vocalis.cli.AudioPlayer") as mock_player_cls,
        patch("vocalis.cli.ClipboardReader", return_value=_FakeClipboardReader()),
    ):
        fake_player = _FakePlayer()
        mock_player_cls.return_value = fake_player

        dictation, _dictation_listener = build_dictation(config=config, model_manager=manager)
        quick_reader, _quick_listener = build_quick_reader(
            config=config, model_manager=manager, reading_coordinator=coordinator
        )

        dictation.on_press()
        dictation.on_release()
        assert manager.current_engine_id == "parakeet"
        assert parakeet_engine.load_count == 1
        assert parakeet_engine.unload_count == 0

        quick_reader.on_trigger()
        assert fake_player.done_event.wait(timeout=2.0)
        assert manager.current_engine_id == TTS_ENGINE_ID
        assert parakeet_engine.unload_count == 1
        assert qwen_engine.load_count == 1


def test_switching_between_quick_reader_and_document_reader_reuses_the_loaded_engine(qapp) -> None:  # type: ignore[no-untyped-def]
    """Quick Reader and Document Reader share TTS_ENGINE_ID on purpose
    (see model_manager.py's own comment) -- switching between them must
    NOT reload the engine. This is the one case in Scenario 1 where "no
    reload happened" is the correct, desired behavior, not a bug."""
    config = VocalisConfig()
    manager = ModelManager(idle_timeout_seconds=9999.0)
    coordinator = ReadingCoordinator()
    qwen_engine = _FakeEngine("qwen")

    with (
        patch("vocalis.cli.Qwen3TTS", return_value=qwen_engine),
        patch("vocalis.cli.PiperTTS", _FakePiper),
        patch("vocalis.cli.AudioPlayer") as mock_player_cls,
        patch("vocalis.cli.ClipboardReader", return_value=_FakeClipboardReader()),
    ):
        first_player = _FakePlayer()
        mock_player_cls.return_value = first_player
        quick_reader, _quick_listener = build_quick_reader(
            config=config, model_manager=manager, reading_coordinator=coordinator
        )
        quick_reader.on_trigger()
        assert first_player.done_event.wait(timeout=2.0)
        assert qwen_engine.load_count == 1

        second_player = _FakePlayer()
        mock_player_cls.return_value = second_player
        doc_reader, _popup, _doc_listener = build_doc_reader(
            config=config, model_manager=manager, reading_coordinator=coordinator
        )
        doc_reader.read_text("some document text")
        assert second_player.done_event.wait(timeout=2.0)

    assert manager.current_engine_id == TTS_ENGINE_ID
    assert qwen_engine.load_count == 1  # NOT reloaded -- same engine id, same cached instance
    assert qwen_engine.unload_count == 0


def test_hotkeys_never_cross_trigger_another_features_handler(qapp) -> None:  # type: ignore[no-untyped-def]
    """All four shortcut listeners, built together against one shared
    ModelManager, must each be bound to their OWN feature's trigger
    method -- never another feature's."""
    config = VocalisConfig()
    manager = ModelManager(idle_timeout_seconds=9999.0)
    coordinator = ReadingCoordinator()

    with (
        patch("vocalis.cli.ParakeetSTT", return_value=_FakeEngine("p")),
        patch("vocalis.cli.MicRecorder", return_value=_FakeRecorder()),
        patch("vocalis.cli.Qwen3TTS", return_value=_FakeEngine("q")),
        patch("vocalis.cli.PiperTTS", _FakePiper),
        patch("vocalis.cli.AudioPlayer", return_value=_FakePlayer()),
        patch("vocalis.cli.ClipboardReader", return_value=_FakeClipboardReader()),
        patch("vocalis.cli.ClipboardWriter"),
        patch("vocalis.cli.ScreenRegionCapture"),
        patch("vocalis.cli.TesseractOCR"),
        patch("vocalis.cli.list_numlock_capable_devices", return_value=[]),
        patch("vocalis.cli.list_hotkey_capable_devices", return_value=[]),
        patch("vocalis.cli.list_devices_with_key", return_value=[]),
    ):
        dictation, dictation_listener = build_dictation(config=config, model_manager=manager)
        quick_reader, quick_listener = build_quick_reader(
            config=config, model_manager=manager, reading_coordinator=coordinator
        )
        doc_reader, popup, doc_listener = build_doc_reader(
            config=config, model_manager=manager, reading_coordinator=coordinator
        )
        ocr_reader, ocr_listener = build_ocr_reader(
            config=config, document_reader=doc_reader, popup=popup
        )

        assert dictation_listener._on_press == dictation.on_press  # noqa: SLF001
        assert dictation_listener._on_release == dictation.on_release  # noqa: SLF001
        assert quick_listener._on_trigger == quick_reader.on_trigger  # noqa: SLF001
        assert doc_listener._on_trigger == popup.hotkey_pressed.emit  # noqa: SLF001
        assert ocr_listener._on_trigger == ocr_reader.on_trigger  # noqa: SLF001

        bound_callback_owners = [
            dictation_listener._on_press.__self__,  # noqa: SLF001
            quick_listener._on_trigger.__self__,  # noqa: SLF001
            ocr_listener._on_trigger.__self__,  # noqa: SLF001
        ]
        assert len(set(id(owner) for owner in bound_callback_owners)) == 3
```

(`doc_listener._on_trigger` is a bound Qt signal emit, not a plain Python
method — it has no `__self__` the same way, so it is checked by identity
against `popup.hotkey_pressed.emit` directly above instead of being
folded into the `bound_callback_owners` list.)

- [ ] **Step 2: Run, verify PASS** — `export PYTHONPATH=$PWD/src; .venv/bin/pytest tests/test_cross_feature_integration.py -q --no-cov` (needs the `qapp` fixture from pytest-qt, already a dependency; all 3 tests pass).

- [ ] **Step 3: Run mypy** — `.venv/bin/mypy` → zero errors.

- [ ] **Step 4: Commit**

```bash
git add tests/test_cross_feature_integration.py
git commit -m "test: add whole-system tests for GPU residency and hotkey isolation"
```

---

### Task 2: Scenario 3 (settings propagation) + Scenario 4 (shared notifications)

**Files:**
- Modify: `tests/test_cross_feature_integration.py` (append)

**Interfaces:**
- Consumes: Task 1's fakes (`_FakeEngine`, `_FakePlayer`, `_FakeClipboardReader`, `_FakePiper`), plus `vocalis.settings.LiveSettings` (`LiveSettings(config, path, saver=...)`, `.update(new_config)`, `.speaking_speed()` returns `float`), `vocalis.audio.pace.apply_speed` (already used for real — not faked — since it's a pure numpy/librosa function with no hardware dependency; the test checks its REAL effect on output length, the same style of assertion Stage 1's own tests already use), `vocalis.notify.Notifier.notify`.

- [ ] **Step 1: Append the failing tests**

Add these two imports to the file's EXISTING top-of-file import block (next to the Task 1 imports — Python imports belong at the top of the file, not inline mid-file, matching every other file in this project):

```python
from pathlib import Path

from vocalis.settings import LiveSettings
```

Then append the two test functions below at the end of the file:

```python

def test_a_settings_change_is_picked_up_by_a_different_features_next_reading(
    qapp, tmp_path: Path  # type: ignore[no-untyped-def]
) -> None:
    """Change speaking_speed via LiveSettings.update() (the same path the
    real Settings window uses), then start a DIFFERENT feature (Document
    Reader) and confirm its output audio is shorter -- proving the
    shared settings path reaches a feature that never set the value
    itself, not just the one feature whose own tests already cover this
    (Stage 1)."""
    config = VocalisConfig()
    manager = ModelManager(idle_timeout_seconds=9999.0)
    coordinator = ReadingCoordinator()
    settings = LiveSettings(config, tmp_path / "config.toml", saver=lambda cfg, path: None)

    fixed_audio_length = 24000  # 1 second at 24kHz, long enough to stretch measurably
    qwen_engine = _FakeEngine("qwen")

    def fixed_length_synthesize(text: str) -> "tuple[npt.NDArray[np.float32], int]":
        return np.zeros(fixed_audio_length, dtype=np.float32), 24000

    qwen_engine.synthesize = fixed_length_synthesize  # type: ignore[method-assign]

    with (
        patch("vocalis.cli.Qwen3TTS", return_value=qwen_engine),
        patch("vocalis.cli.PiperTTS", _FakePiper),
        patch("vocalis.cli.AudioPlayer") as mock_player_cls,
    ):
        fast_player = _FakePlayer()
        mock_player_cls.return_value = fast_player
        settings.update(config.model_copy(update={"speaking_speed": 1.6}))
        doc_reader, _popup, _listener = build_doc_reader(
            config=config, model_manager=manager, reading_coordinator=coordinator, settings=settings
        )
        doc_reader.read_text("some document text")
        assert fast_player.done_event.wait(timeout=2.0)
        fast_audio, _sr = fast_player.played[0]

    with (
        patch("vocalis.cli.Qwen3TTS", return_value=qwen_engine),
        patch("vocalis.cli.PiperTTS", _FakePiper),
        patch("vocalis.cli.AudioPlayer") as mock_player_cls,
    ):
        slow_player = _FakePlayer()
        mock_player_cls.return_value = slow_player
        settings.update(config.model_copy(update={"speaking_speed": 0.7}))
        doc_reader2, _popup2, _listener2 = build_doc_reader(
            config=config, model_manager=manager, reading_coordinator=coordinator, settings=settings
        )
        doc_reader2.read_text("some document text")
        assert slow_player.done_event.wait(timeout=2.0)
        slow_audio, _sr2 = slow_player.played[0]

    assert len(fast_audio) < fixed_audio_length < len(slow_audio)


def test_notifications_from_two_different_features_both_reach_the_shared_notifier(
    qapp,  # type: ignore[no-untyped-def]
) -> None:
    """Dictate, then Quick Reader, in sequence. Every notification from
    BOTH features must reach the one shared Notifier, in order,
    un-garbled by the other feature's notifications."""
    config = VocalisConfig()
    manager = ModelManager(idle_timeout_seconds=9999.0)
    coordinator = ReadingCoordinator()
    sent: "list[tuple[str, str]]" = []

    def fake_notify(self: object, title: str, message: str = "") -> None:  # noqa: ANN001
        sent.append((title, message))

    with (
        patch("vocalis.cli.ParakeetSTT", return_value=_FakeEngine("p")),
        patch("vocalis.cli.MicRecorder", return_value=_FakeRecorder()),
        patch("vocalis.cli.Qwen3TTS", return_value=_FakeEngine("q")),
        patch("vocalis.cli.PiperTTS", _FakePiper),
        patch("vocalis.cli.AudioPlayer") as mock_player_cls,
        patch("vocalis.cli.ClipboardReader", return_value=_FakeClipboardReader()),
        patch("vocalis.cli.Notifier.notify", fake_notify),
    ):
        fake_player = _FakePlayer()
        mock_player_cls.return_value = fake_player
        dictation, _dictation_listener = build_dictation(config=config, model_manager=manager)
        quick_reader, _quick_listener = build_quick_reader(
            config=config, model_manager=manager, reading_coordinator=coordinator
        )

        dictation.on_press()
        dictation.on_release()
        quick_reader.on_trigger()
        assert fake_player.done_event.wait(timeout=2.0)

    # Confirmed live against the real code while writing this plan:
    # dictation.py's _TITLE == "Vocalis — Dictation", quick_reader.py's
    # _TITLE == "Vocalis — Quick Reader". The real sequence recorded for
    # this exact scenario is:
    #   [("Vocalis — Dictation", "Listening..."),
    #    ("Vocalis — Dictation", "Transcribing..."),
    #    ("Vocalis — Dictation", "fake transcription"),
    #    ("Vocalis — Dictation", "Done"),
    #    ("Vocalis — Quick Reader", "Reading..."),
    #    ("Vocalis — Quick Reader", "clipboard text"),
    #    ("Vocalis — Quick Reader", "Done")]
    dictation_titles = [title for title, _ in sent if title == "Vocalis — Dictation"]
    quick_reader_titles = [title for title, _ in sent if title == "Vocalis — Quick Reader"]
    assert len(dictation_titles) >= 2
    assert len(quick_reader_titles) >= 2
    dictation_last_index = max(i for i, (title, _) in enumerate(sent) if title == "Vocalis — Dictation")
    quick_reader_first_index = min(i for i, (title, _) in enumerate(sent) if title == "Vocalis — Quick Reader")
    assert dictation_last_index < quick_reader_first_index  # dictation's notifications all came first, in order
```

- [ ] **Step 2: Run, verify PASS** — `.venv/bin/pytest tests/test_cross_feature_integration.py -q --no-cov` → all 5 tests pass.

- [ ] **Step 3: Run mypy** — `.venv/bin/mypy` → zero errors.

- [ ] **Step 4: Commit**

```bash
git add tests/test_cross_feature_integration.py
git commit -m "test: add whole-system tests for settings propagation and shared notifications"
```

---

### Task 3: Scenario 5 (full multi-feature lifecycle with clean shutdown)

**Files:**
- Modify: `tests/test_cross_feature_integration.py` (append)

**Interfaces:**
- Consumes: Task 1's fakes; `ModelManager.shutdown()` (already exists, unloads the current engine and clears `current_engine_id`).

- [ ] **Step 1: Append the failing test**

```python
def test_full_lifecycle_two_features_then_clean_shutdown(qapp) -> None:  # type: ignore[no-untyped-def]
    """Build dictation and quick reader together (mirroring main()'s own
    construction order), have each handle one request, then shut the
    shared ModelManager down -- the same call main()'s daemon.stop()
    makes on the way out. The engine must end up unloaded and
    current_engine_id cleared, regardless of which feature used it last."""
    config = VocalisConfig()
    manager = ModelManager(idle_timeout_seconds=9999.0)
    coordinator = ReadingCoordinator()
    parakeet_engine = _FakeEngine("parakeet")
    qwen_engine = _FakeEngine("qwen")

    with (
        patch("vocalis.cli.ParakeetSTT", return_value=parakeet_engine),
        patch("vocalis.cli.MicRecorder", return_value=_FakeRecorder()),
        patch("vocalis.cli.Qwen3TTS", return_value=qwen_engine),
        patch("vocalis.cli.PiperTTS", _FakePiper),
        patch("vocalis.cli.AudioPlayer") as mock_player_cls,
        patch("vocalis.cli.ClipboardReader", return_value=_FakeClipboardReader()),
    ):
        fake_player = _FakePlayer()
        mock_player_cls.return_value = fake_player
        dictation, _dictation_listener = build_dictation(config=config, model_manager=manager)
        quick_reader, _quick_listener = build_quick_reader(
            config=config, model_manager=manager, reading_coordinator=coordinator
        )

        dictation.on_press()
        dictation.on_release()
        quick_reader.on_trigger()
        assert fake_player.done_event.wait(timeout=2.0)

    assert manager.current_engine_id == TTS_ENGINE_ID  # still resident, nothing has released it yet

    manager.shutdown()

    assert manager.current_engine_id is None
    assert qwen_engine.unload_count == 1
```

- [ ] **Step 2: Run, verify PASS** — `.venv/bin/pytest tests/test_cross_feature_integration.py -q --no-cov` → all 6 tests pass.

- [ ] **Step 3: Run the full suite once** — `.venv/bin/pytest -q` → zero failures, coverage >= 85%, zero warnings. Record the exact test count and coverage percentage for the change log in Task 6.

- [ ] **Step 4: Run mypy** — `.venv/bin/mypy` → zero errors.

- [ ] **Step 5: Commit**

```bash
git add tests/test_cross_feature_integration.py
git commit -m "test: add whole-system test for multi-feature lifecycle and clean shutdown"
```

---

### Task 4: Root `README.md`

**Files:**
- Create: `README.md`

**Interfaces:** none — pure documentation, links to existing files only.

- [ ] **Step 1: Write the file**

Read `packaging/installer/README.md`, `docs/user-docs.md`'s first few headings, and `docs/developer-docs.md`'s first few headings first, so every link target and claim below is accurate at the moment you write it (not copied from this plan without checking).

```markdown
# Vocalis

An offline voice AI workstation: dictation, a quick-reader for anything
you've copied, a document reader (with pause/resume), and an OCR reader
that speaks text straight off your screen. Runs entirely on your own
machine — no cloud, no account, nothing sent anywhere.

## Installing

See [`packaging/installer/README.md`](packaging/installer/README.md) for
the one-command installer (works on Debian/Ubuntu-family Linux with an
NVIDIA GPU).

## Using it

The full guide to every feature, the tray icon, the floating widget, and
Settings lives in [`docs/user-docs.md`](docs/user-docs.md).

## Contributing / developing

Architecture, folder layout, and how to run the test suite are in
[`docs/developer-docs.md`](docs/developer-docs.md).
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add a root-level README"
```

---

### Task 5: Documentation accuracy sweep

**Files:**
- Modify (as needed, only where real drift is found): `docs/user-docs.md`, `docs/architecture.md`, `docs/workflow.md`, `docs/file-purpose.md`, `docs/folder-structure.md`, `docs/developer-docs.md`, `docs/algorithms.md`

**Interfaces:** none — pure documentation.

No placeholder steps here: the actual work is reading each file side by
side with the code it describes and fixing what's wrong. There is no way
to pre-write the diffs in this plan, since they depend on whatever drift
is actually found — but the checklist below is concrete and exhaustive
for what counts as "done".

- [ ] **Step 1: `docs/user-docs.md`** — re-read every section against the current code. In particular: confirm the "Installing Vocalis" section matches `packaging/installer/install.sh`'s real behavior (it was written mid-Stage-3, before the final-review fix wave added `--purge`'s Piper removal, the `--help` flag, and the `-- --purge` syntax note for `.run` files — confirm all three are now correctly described). Confirm every keyboard shortcut, config field name, and file path mentioned still exists exactly as named.
- [ ] **Step 2: `docs/architecture.md`** — confirm the described component relationships (ModelManager single-residency, ReadingCoordinator, StatusHub, the daemon) match `cli.py`'s actual `main()` wiring read in Task 1 above. Confirm the Stage 3 installer section (added at the end of the file during Stage 3) is accurate against the final, fixed `install.sh`/`uninstall.sh` (not the pre-fix-wave version).
- [ ] **Step 3: `docs/workflow.md`** — confirm the described user-facing and internal flows for all four features still match the current code paths.
- [ ] **Step 4: `docs/file-purpose.md`** and **`docs/folder-structure.md`** — confirm every file/folder listed still exists at the path given, and that nothing created since (e.g. anything under `packaging/installer/`, `tests/installer/`, `tests/test_cross_feature_integration.py`, the root `README.md` from Task 4) is missing from these two files. Add entries for anything missing.
- [ ] **Step 5: `docs/developer-docs.md`** — confirm the described test-running instructions, the `*_run_*`/override testing idiom, and the installer rebuild instructions are accurate. Add a short note about the new whole-system test file and what makes it different from every other test file (reuses `cli.py`'s builders against one shared `ModelManager`, rather than testing one feature in isolation).
- [ ] **Step 6: `docs/algorithms.md`** — confirm every algorithm described (time-stretch, text chunking, the single-instance socket probe, etc.) still matches its current implementation.
- [ ] **Step 7: Run the full suite once more** — `.venv/bin/pytest -q && .venv/bin/mypy` (docs changes should never affect this, but confirm nothing was accidentally broken while reading code).
- [ ] **Step 8: Commit**

```bash
git add docs
git commit -m "docs: fix documentation drift found during the Stage 4 accuracy sweep"
```

(If Step 1-6 found zero drift in a given file, say so explicitly in the commit message or change log rather than silently skipping it — "found zero drift in docs/algorithms.md, confirmed against the current code" is a valid, complete outcome for that file.)

---

### Task 6: Final verification and change log

**Files:**
- Create: `.claude/<YYYY-MM-DD_HHMM>_CHANGELOG.md`

- [ ] **Step 1: Full verification sweep** — `export PYTHONPATH=$PWD/src; .venv/bin/pytest -q` three times sequentially (all pass, zero warnings, coverage >= 85%, no segfault — a segfault on one run that clears on rerun is a known, pre-existing flake in this project's suite, confirmed harmless by rerunning; record which happened), `.venv/bin/mypy` (Success).
- [ ] **Step 2: Change log** — files touched across all 6 tasks, final test count/coverage from your own runs, what drift (if any) Task 5 found and fixed in each of the 7 doc files, and the explicit note: `- [ ] Real-world check pending: this was the last development stage. The user's own first hands-on run of the installer (install.sh/.run, then actually using dictation/Quick Reader/Document Reader/OCR Reader together) is the first real-world test of everything built across all four stages.`
- [ ] **Step 3: Commit**

```bash
git add .claude/*CHANGELOG.md
git commit -m "docs: add Stage 4 change log"
```
