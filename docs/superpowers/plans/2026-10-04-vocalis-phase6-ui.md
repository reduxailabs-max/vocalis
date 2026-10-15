# Vocalis Phase 6 — UI (tray, floating widget, settings) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a tray icon, a floating mini-widget shown while something is read aloud, and a small Settings window (notifications switch, widget switch, four shortcuts, voice), with the switches and voice applying immediately and shortcuts applying after a restart.

**Architecture:** One shared `LiveSettings` holder (read by features at the moment of use; written only by the Settings window, atomically) plus one `StatusHub` (a Qt object with signals) that the readers and dictation report into; the tray and the widget only listen to the hub. Shortcuts get a real parser so the (today ignored) `[shortcuts]` config is used.

**Tech Stack:** Python 3.13, PySide6 (offscreen in tests, `qtbot`), evdev key codes, pydantic config, pytest + pytest-qt, mypy strict.

**Spec:** `docs/superpowers/specs/2026-09-11-vocalis-design.md` — section "Phase 6 amendment".

## Global Constraints

- Work in a fresh git worktree on branch `worktree-vocalis-phase6-ui`; run everything with `export PYTHONPATH=$PWD/src` and `.venv/bin/pytest` / `.venv/bin/mypy` (the shared `.venv` editable pointer is stale; link `.venv` to the main checkout's `.venv`, do not `git add` it).
- Conventional commits (`feat:`, `fix:`, `test:`, `docs:`), **no Co-Authored-By or attribution lines**, explicit `git add` paths, never touch `old/`.
- Every function, method and class has a docstring. `mypy` strict: **zero errors project-wide**. Full suite **zero warnings**, coverage **>= 85%** (baseline at plan time: 335 tests, 95.79%).
- Never hold a lock across a blocking call; hub/settings methods are callable from any thread and emit Qt signals outside their locks. Tests deterministic (no guessed sleeps; Qt events via `qtbot`).
- Never run the real program, real input devices, the real sound device or the real `notify-send` in tests; do not overwrite the user's real `~/.config/vocalis/config.toml` (tests use `tmp_path`).
- Existing tests keep passing unchanged: new constructor parameters are **optional keyword parameters with defaults** and existing names imported in `cli.py` stay (tests patch `vocalis.cli.list_numlock_capable_devices`, `list_hotkey_capable_devices`, `list_devices_with_key`). Never weaken or delete an existing assertion.
- Voices (exact, capitalised official preset names): `Serena, Vivian, Uncle_Fu, Ryan, Aiden, Ono_Anna, Sohee, Eric, Dylan`; default `Ryan`.
- evdev codes: A-Z and 0-9 via `evdev.ecodes.KEY_<X>` (R=19, D=32, O=24, 1=2); push-to-talk choices `NUMLOCK`(69, default), `SCROLLLOCK`(70), `PAUSE`(119).
- Models and audio path are unchanged (audio device / speed / streaming / voice quality are out of scope).
- Record the change log in `.claude/<YYYY-MM-DD_HHMM>_CHANGELOG.md`; keep `docs/` in sync.

## Review Focus

1. A hand-edited bad shortcut in the config file (e.g. `Ctrl+Alt+Banana`) must not crash startup: log a clear error and use the default shortcuts (Task 1, 9).
2. Two features given the same shortcut in Settings are refused with the reason shown and nothing is saved (Tasks 1, 2, 8).
3. A failing save (disk error) leaves the old file intact and the old settings in memory, with no leftover temp file (Tasks 2, 8).
4. Quick Reader pressed twice quickly (stop then start) must leave the hub in the right state — the stop is reported synchronously, not from the late stop thread (Task 5).
5. Closing the Settings window or the Document Reader window must NOT quit Vocalis (`setQuitOnLastWindowClosed(False)`) (Task 10).
6. No system tray available: one warning, no crash, widget still works (Task 7, 10).
7. The widget never takes keyboard focus and hides when the reading ends or the switch is turned off mid-reading (Task 6).
8. A voice changed in Settings applies to the next reading even while the TTS engine is still loaded in memory (Tasks 2, 3, 9).

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/vocalis/hotkeys/shortcut_parser.py` | create | shortcut text <-> evdev code, validation, duplicates |
| `src/vocalis/config.py` | modify | atomic `save_config` |
| `src/vocalis/settings.py` | create | `LiveSettings`, `VOICES` |
| `src/vocalis/notify.py` | modify | `enabled` may be a callable |
| `src/vocalis/engines/qwen3_tts.py` | modify | `speaker` may be a callable |
| `src/vocalis/ui/status_hub.py` | create | `StatusHub`, `ReadingControls` |
| `src/vocalis/features/doc_reader.py`, `quick_reader.py`, `dictation.py` | modify | optional hub hooks (+ `QuickReaderFeature.stop`) |
| `src/vocalis/ui/floating_widget.py`, `tray.py`, `settings_window.py` | create | the three surfaces |
| `src/vocalis/hotkeys/numlock.py` | modify | configurable key code |
| `src/vocalis/cli.py` | modify | use parsed shortcuts, live settings, hub, UI, OCR popup-vs-widget, quit rules |
| `tests/…`, `docs/…`, `.claude/..._CHANGELOG.md` | create/modify | tests, docs |

---

### Task 1: Shortcut parser

**Files:** Create `src/vocalis/hotkeys/shortcut_parser.py`; Test `tests/test_shortcut_parser.py`.

**Interfaces:**
- Produces: `ShortcutError(ValueError)`; `CTRL_ALT_CHOICES: tuple[str, ...]` (`"A".."Z","0".."9"`); `PUSH_TO_TALK_CHOICES = ("NUMLOCK","SCROLLLOCK","PAUSE")`; `parse_ctrl_alt_shortcut(text: str) -> int`; `format_ctrl_alt_shortcut(choice: str) -> str`; `ctrl_alt_choice(text: str) -> str`; `parse_push_to_talk_key(text: str) -> int`; `@dataclass(frozen=True) ShortcutSet(push_to_talk: int, quick_reader: int, document_reader: int, ocr_reader: int)`; `parse_shortcuts(shortcuts: ShortcutsConfig) -> ShortcutSet`; `DEFAULT_SHORTCUTS: ShortcutSet`.

- [ ] **Step 1: Failing tests** — create `tests/test_shortcut_parser.py`:

```python
"""Tests for vocalis.hotkeys.shortcut_parser."""
import pytest

from vocalis.config import ShortcutsConfig
from vocalis.hotkeys.shortcut_parser import (
    CTRL_ALT_CHOICES,
    DEFAULT_SHORTCUTS,
    PUSH_TO_TALK_CHOICES,
    ShortcutError,
    ShortcutSet,
    ctrl_alt_choice,
    format_ctrl_alt_shortcut,
    parse_ctrl_alt_shortcut,
    parse_push_to_talk_key,
    parse_shortcuts,
)


def test_known_codes_match_evdev() -> None:
    assert parse_ctrl_alt_shortcut("Ctrl+Alt+R") == 19
    assert parse_ctrl_alt_shortcut("Ctrl+Alt+D") == 32
    assert parse_ctrl_alt_shortcut("Ctrl+Alt+O") == 24
    assert parse_ctrl_alt_shortcut("Ctrl+Alt+1") == 2
    assert parse_push_to_talk_key("NUMLOCK") == 69
    assert parse_push_to_talk_key("SCROLLLOCK") == 70
    assert parse_push_to_talk_key("PAUSE") == 119


def test_case_and_spaces_are_tolerated() -> None:
    assert parse_ctrl_alt_shortcut(" ctrl + alt + r ") == 19
    assert parse_push_to_talk_key("numlock") == 69


@pytest.mark.parametrize(
    "bad", ["", "Ctrl+R", "Alt+Ctrl+R", "Ctrl+Alt+", "Ctrl+Alt+Banana", "Ctrl+Alt+RR", "Ctrl+Alt+R+X", "Ctrl+Alt+-", "Shift+Alt+R"]
)
def test_bad_ctrl_alt_shortcuts_are_rejected_with_the_expected_form(bad: str) -> None:
    with pytest.raises(ShortcutError, match="Ctrl\\+Alt\\+"):
        parse_ctrl_alt_shortcut(bad)


@pytest.mark.parametrize("bad", ["", "A", "F13", "CAPSLOCK", "Ctrl+Alt+R"])
def test_bad_push_to_talk_keys_are_rejected(bad: str) -> None:
    with pytest.raises(ShortcutError, match="NUMLOCK"):
        parse_push_to_talk_key(bad)


def test_choices_round_trip() -> None:
    assert len(CTRL_ALT_CHOICES) == 36
    for choice in CTRL_ALT_CHOICES:
        text = format_ctrl_alt_shortcut(choice)
        assert ctrl_alt_choice(text) == choice
        parse_ctrl_alt_shortcut(text)
    for key in PUSH_TO_TALK_CHOICES:
        parse_push_to_talk_key(key)


def test_format_rejects_a_choice_that_is_not_offered() -> None:
    with pytest.raises(ShortcutError):
        format_ctrl_alt_shortcut("%")


def test_defaults_parse_to_the_codes_the_app_used_before() -> None:
    assert DEFAULT_SHORTCUTS == ShortcutSet(push_to_talk=69, quick_reader=19, document_reader=32, ocr_reader=24)
    assert parse_shortcuts(ShortcutsConfig()) == DEFAULT_SHORTCUTS


def test_bad_value_names_the_field() -> None:
    with pytest.raises(ShortcutError, match="ocr_reader"):
        parse_shortcuts(ShortcutsConfig(ocr_reader="Ctrl+Alt+Banana"))
    with pytest.raises(ShortcutError, match="numlock_key"):
        parse_shortcuts(ShortcutsConfig(numlock_key="F13"))


def test_two_features_may_not_share_a_shortcut() -> None:
    with pytest.raises(ShortcutError, match="quick_reader.*ocr_reader|ocr_reader.*quick_reader"):
        parse_shortcuts(ShortcutsConfig(quick_reader="Ctrl+Alt+O", ocr_reader="Ctrl+Alt+o"))
```

- [ ] **Step 2: Run, verify FAIL** — `export PYTHONPATH=$PWD/src; .venv/bin/pytest tests/test_shortcut_parser.py -q --no-cov` → `ModuleNotFoundError`.

- [ ] **Step 3: Implement** — create `src/vocalis/hotkeys/shortcut_parser.py`:

```python
"""Turns shortcut text from the config file ("Ctrl+Alt+R", "NUMLOCK") into
evdev key codes, and validates it.

The three Ctrl+Alt features accept exactly `Ctrl+Alt+<letter or digit>`;
the push-to-talk key accepts a short list of safe keys. The Settings
window offers these as drop-down lists so typos cannot happen there;
values typed by hand into the config file are checked here.
"""
from dataclasses import dataclass

from evdev import ecodes

from vocalis.config import ShortcutsConfig

CTRL_ALT_CHOICES: tuple[str, ...] = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
PUSH_TO_TALK_CHOICES: tuple[str, ...] = ("NUMLOCK", "SCROLLLOCK", "PAUSE")

_EXPECTED_CTRL_ALT = "Ctrl+Alt+<one letter or digit>, for example Ctrl+Alt+R"
_EXPECTED_PUSH_TO_TALK = "one of NUMLOCK, SCROLLLOCK, PAUSE"


class ShortcutError(ValueError):
    """A shortcut could not be understood or conflicts with another one.

    The message is written for the end user (shown in the Settings window
    and in the log).
    """


@dataclass(frozen=True)
class ShortcutSet:
    """The evdev key codes for every configurable shortcut."""

    push_to_talk: int
    quick_reader: int
    document_reader: int
    ocr_reader: int


def _code_for(choice: str) -> int:
    """evdev code for a single letter/digit key name."""
    return int(getattr(ecodes, f"KEY_{choice}"))


def ctrl_alt_choice(text: str) -> str:
    """Extract the letter/digit from "Ctrl+Alt+R" text (upper-cased).

    Raises:
        ShortcutError: If the text is not `Ctrl+Alt+<one letter or digit>`.
    """
    parts = [part.strip() for part in text.split("+")]
    if (
        len(parts) != 3
        or parts[0].casefold() != "ctrl"
        or parts[1].casefold() != "alt"
        or parts[2].upper() not in CTRL_ALT_CHOICES
    ):
        raise ShortcutError(f"'{text}' is not a valid shortcut; expected {_EXPECTED_CTRL_ALT}")
    return parts[2].upper()


def parse_ctrl_alt_shortcut(text: str) -> int:
    """Return the evdev key code for a `Ctrl+Alt+<key>` shortcut.

    Raises:
        ShortcutError: If the text is not `Ctrl+Alt+<one letter or digit>`.
    """
    return _code_for(ctrl_alt_choice(text))


def format_ctrl_alt_shortcut(choice: str) -> str:
    """Return "Ctrl+Alt+<choice>" for a letter/digit offered in the Settings window.

    Raises:
        ShortcutError: If `choice` is not one of CTRL_ALT_CHOICES.
    """
    if choice not in CTRL_ALT_CHOICES:
        raise ShortcutError(f"'{choice}' cannot be used as a shortcut key; expected {_EXPECTED_CTRL_ALT}")
    return f"Ctrl+Alt+{choice}"


def parse_push_to_talk_key(text: str) -> int:
    """Return the evdev key code for the push-to-talk key.

    Raises:
        ShortcutError: If the key is not one of PUSH_TO_TALK_CHOICES.
    """
    name = text.strip().upper()
    if name not in PUSH_TO_TALK_CHOICES:
        raise ShortcutError(f"'{text}' cannot be used as the push-to-talk key; expected {_EXPECTED_PUSH_TO_TALK}")
    return int(getattr(ecodes, f"KEY_{name}"))


def parse_shortcuts(shortcuts: ShortcutsConfig) -> ShortcutSet:
    """Validate and convert every shortcut in the config.

    Raises:
        ShortcutError: A value is invalid (the message names the config
            field) or two Ctrl+Alt features share the same shortcut.
    """
    try:
        push_to_talk = parse_push_to_talk_key(shortcuts.numlock_key)
    except ShortcutError as exc:
        raise ShortcutError(f"numlock_key: {exc}") from exc
    codes: dict[str, int] = {}
    for name in ("quick_reader", "document_reader", "ocr_reader"):
        try:
            codes[name] = parse_ctrl_alt_shortcut(getattr(shortcuts, name))
        except ShortcutError as exc:
            raise ShortcutError(f"{name}: {exc}") from exc
    seen: dict[int, str] = {}
    for name, code in codes.items():
        if code in seen:
            raise ShortcutError(f"{seen[code]} and {name} both use the same shortcut; each feature needs its own")
        seen[code] = name
    return ShortcutSet(
        push_to_talk=push_to_talk,
        quick_reader=codes["quick_reader"],
        document_reader=codes["document_reader"],
        ocr_reader=codes["ocr_reader"],
    )


DEFAULT_SHORTCUTS = parse_shortcuts(ShortcutsConfig())
```

- [ ] **Step 4: Run, verify PASS** — `.venv/bin/pytest tests/test_shortcut_parser.py -q --no-cov && .venv/bin/mypy` → all pass, mypy `Success`.
- [ ] **Step 5: Commit** — `git add src/vocalis/hotkeys/shortcut_parser.py tests/test_shortcut_parser.py && git commit -m "feat: add shortcut parser for the configurable hotkeys"`

---

### Task 2: Atomic config save and LiveSettings

**Files:** Modify `src/vocalis/config.py` (`save_config`); Create `src/vocalis/settings.py`; Test `tests/test_config.py` (append), `tests/test_settings.py`.

**Interfaces:**
- Consumes: Task 1 `parse_shortcuts`, `ShortcutError`.
- Produces: `save_config` now atomic (same signature, still raises `ConfigError`); `VOICES: tuple[str, ...]`; `LiveSettings(config: VocalisConfig, path: Path, saver: Callable[[VocalisConfig, Path], None] = save_config)` with `.config`, `.notifications_enabled() -> bool`, `.floating_widget_enabled() -> bool`, `.voice() -> str`, `.update(new_config) -> None` (raises `ShortcutError` or `ConfigError`, in both cases changing nothing), `.shortcuts_need_restart() -> bool` (True when the saved shortcuts differ from those the program started with).

- [ ] **Step 1: Failing tests.**

Append to `tests/test_config.py` (add imports `import os`, `from unittest.mock import patch`, and `ConfigError`/`save_config`/`load_config` if not already imported):

```python
def test_save_config_is_atomic_and_leaves_no_temp_files(tmp_path: Path) -> None:
    path = tmp_path / "vocalis" / "config.toml"
    save_config(VocalisConfig(notifications_enabled=False), path)
    assert load_config(path).notifications_enabled is False
    assert [p.name for p in path.parent.iterdir()] == ["config.toml"]


def test_failed_save_keeps_the_old_file_and_leaves_no_temp_file(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    save_config(VocalisConfig(notifications_enabled=True), path)
    with patch("vocalis.config.os.replace", side_effect=OSError("disk full")):
        with pytest.raises(ConfigError):
            save_config(VocalisConfig(notifications_enabled=False), path)
    assert load_config(path).notifications_enabled is True
    assert [p.name for p in tmp_path.iterdir()] == ["config.toml"]
```

Create `tests/test_settings.py`:

```python
"""Tests for vocalis.settings.LiveSettings."""
import threading
from pathlib import Path

import pytest

from vocalis.config import ConfigError, ShortcutsConfig, VocalisConfig, load_config
from vocalis.hotkeys.shortcut_parser import ShortcutError
from vocalis.settings import VOICES, LiveSettings


def make(tmp_path: Path, config: VocalisConfig | None = None) -> LiveSettings:
    return LiveSettings(config or VocalisConfig(), tmp_path / "config.toml")


def test_voices_are_the_nine_presets_with_ryan_default() -> None:
    assert VOICES == ("Serena", "Vivian", "Uncle_Fu", "Ryan", "Aiden", "Ono_Anna", "Sohee", "Eric", "Dylan")
    assert make(Path("/nonexistent")).voice() == "Ryan"


def test_update_saves_the_file_then_changes_what_features_read(tmp_path: Path) -> None:
    settings = make(tmp_path)
    new = settings.config.model_copy(
        update={"notifications_enabled": False, "floating_widget_enabled": False,
                "engines": settings.config.engines.model_copy(update={"qwen3_tts_speaker": "Dylan"})}
    )
    settings.update(new)
    assert settings.notifications_enabled() is False
    assert settings.floating_widget_enabled() is False
    assert settings.voice() == "Dylan"
    assert load_config(tmp_path / "config.toml").engines.qwen3_tts_speaker == "Dylan"


def test_duplicate_shortcuts_are_refused_and_nothing_changes(tmp_path: Path) -> None:
    settings = make(tmp_path)
    bad = settings.config.model_copy(update={"shortcuts": ShortcutsConfig(quick_reader="Ctrl+Alt+O")})
    with pytest.raises(ShortcutError):
        settings.update(bad)
    assert settings.config.shortcuts.quick_reader == "Ctrl+Alt+R"
    assert not (tmp_path / "config.toml").exists()


def test_failed_save_keeps_the_old_settings(tmp_path: Path) -> None:
    def failing_saver(config: VocalisConfig, path: Path) -> None:
        raise ConfigError("disk full")

    settings = LiveSettings(VocalisConfig(), tmp_path / "config.toml", saver=failing_saver)
    with pytest.raises(ConfigError):
        settings.update(settings.config.model_copy(update={"notifications_enabled": False}))
    assert settings.notifications_enabled() is True


def test_shortcut_change_needs_a_restart_and_only_then(tmp_path: Path) -> None:
    settings = make(tmp_path)
    assert settings.shortcuts_need_restart() is False
    settings.update(settings.config.model_copy(update={"notifications_enabled": False}))
    assert settings.shortcuts_need_restart() is False
    settings.update(settings.config.model_copy(update={"shortcuts": ShortcutsConfig(ocr_reader="Ctrl+Alt+P")}))
    assert settings.shortcuts_need_restart() is True


def test_readers_on_other_threads_never_see_a_half_updated_config(tmp_path: Path) -> None:
    settings = make(tmp_path)
    seen: list[str] = []
    stop = threading.Event()

    def reader() -> None:
        while not stop.is_set():
            seen.append(settings.voice())

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    for voice in ("Dylan", "Eric", "Ryan"):
        settings.update(settings.config.model_copy(update={"engines": settings.config.engines.model_copy(update={"qwen3_tts_speaker": voice})}))
    stop.set()
    thread.join(timeout=5.0)
    assert not thread.is_alive()
    assert set(seen) <= {"Ryan", "Dylan", "Eric"}
```

- [ ] **Step 2: Run, verify FAIL** — `.venv/bin/pytest tests/test_config.py tests/test_settings.py -q --no-cov` (atomic tests fail: no `os` in config, `settings` module missing).
- [ ] **Step 3: Implement.**

`src/vocalis/config.py`: add `import os` and `import tempfile`; replace the body of `save_config` with:

```python
    tmp_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=path.parent)
        tmp_path = Path(tmp_name)
        with os.fdopen(fd, "wb") as handle:
            tomli_w.dump(config.model_dump(mode="json"), handle)
        os.replace(tmp_path, path)
    except OSError as exc:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise ConfigError(f"Could not write config file {path}: {exc}") from exc
```

and extend the docstring: "The file is written to a temporary file in the same folder and renamed into place, so a crash while saving cannot leave a half-written config."

Create `src/vocalis/settings.py`:

```python
"""The one shared, live copy of the user's settings.

Created once at startup from the config file. The Settings window is the
only writer (via update()); features read it at the moment they need a
value (notifications on/off, widget on/off, voice) instead of copying it
at construction, so those settings apply immediately. Shortcuts are read
once at startup; changing them applies after a restart.
"""
import threading
from pathlib import Path
from typing import Callable

from vocalis.config import VocalisConfig, save_config
from vocalis.hotkeys.shortcut_parser import parse_shortcuts

VOICES: tuple[str, ...] = (
    "Serena", "Vivian", "Uncle_Fu", "Ryan", "Aiden", "Ono_Anna", "Sohee", "Eric", "Dylan",
)


class LiveSettings:
    """Thread-safe holder of the current VocalisConfig."""

    def __init__(
        self,
        config: VocalisConfig,
        path: Path,
        saver: Callable[[VocalisConfig, Path], None] = save_config,
    ) -> None:
        """Create the holder.

        Args:
            config: The config loaded at startup.
            path: Where update() saves the config file.
            saver: Writes a config to a path (default: the atomic
                save_config). Injectable for tests.
        """
        self._lock = threading.Lock()
        self._config = config
        self._started_with = config.shortcuts
        self._path = path
        self._saver = saver

    @property
    def config(self) -> VocalisConfig:
        """The current config (an immutable-by-convention snapshot)."""
        with self._lock:
            return self._config

    def notifications_enabled(self) -> bool:
        """Whether desktop notifications are currently switched on."""
        return self.config.notifications_enabled

    def floating_widget_enabled(self) -> bool:
        """Whether the floating widget is currently switched on."""
        return self.config.floating_widget_enabled

    def voice(self) -> str:
        """The TTS voice to use for the next reading."""
        return self.config.engines.qwen3_tts_speaker

    def shortcuts_need_restart(self) -> bool:
        """Whether the saved shortcuts differ from those the program started with."""
        return self.config.shortcuts != self._started_with

    def update(self, new_config: VocalisConfig) -> None:
        """Validate, save and then adopt a new config.

        The new config replaces the current one only after it has been
        validated and saved, so any failure leaves both the file and the
        in-memory settings as they were.

        Raises:
            ShortcutError: A shortcut is invalid or two features share one.
            ConfigError: The file could not be written.
        """
        parse_shortcuts(new_config.shortcuts)
        self._saver(new_config, self._path)
        with self._lock:
            self._config = new_config
```

- [ ] **Step 4: Run, verify PASS** — `.venv/bin/pytest tests/test_config.py tests/test_settings.py -q --no-cov && .venv/bin/mypy`.
- [ ] **Step 5: Commit** — `git add src/vocalis/config.py src/vocalis/settings.py tests/test_config.py tests/test_settings.py && git commit -m "feat: add live settings holder and make config saving atomic"`

---

### Task 3: Notifier and Qwen3TTS read live values

**Files:** Modify `src/vocalis/notify.py`, `src/vocalis/engines/qwen3_tts.py`; Test `tests/test_notify.py`, `tests/test_qwen3_tts.py` (append).

**Interfaces:** Produces `Notifier(enabled: bool | Callable[[], bool], …)` (callable evaluated on every `notify`); `Qwen3TTS(speaker: str | Callable[[], str] = "Ryan", …)` (callable evaluated on every `synthesize`). Existing positional/keyword uses with plain values are unchanged.

- [ ] **Step 1: Failing tests.** In `tests/test_notify.py` append (reuse that file's fake runner; read the file first to copy its fake's name/shape):

```python
def test_enabled_can_be_a_function_that_is_checked_on_every_notification() -> None:
    state = {"on": True}
    runner = FakeCommandRunner()  # use the file's existing fake runner class
    notifier = Notifier(enabled=lambda: state["on"], command_runner=runner)
    notifier.notify("T", "one")
    state["on"] = False
    notifier.notify("T", "two")
    state["on"] = True
    notifier.notify("T", "three")
    assert [call[-1] for call in runner.calls] == ["one", "three"]
```

In `tests/test_qwen3_tts.py` append (follow that file's existing way of faking the loaded model — read its tests first and reuse its fake model that records `generate_custom_voice` kwargs):

```python
def test_speaker_can_be_a_function_read_on_every_synthesis() -> None:
    voice = {"name": "Ryan"}
    engine = Qwen3TTS(model_dir=Path("/x"), speaker=lambda: voice["name"], use_gpu=False)
    fake = FakeModel()  # the file's existing fake that records kwargs
    engine._model = fake  # noqa: SLF001
    engine.synthesize("hello")
    voice["name"] = "Dylan"
    engine.synthesize("hello")
    assert [call["speaker"] for call in fake.calls] == ["Ryan", "Dylan"]
```

- [ ] **Step 2: Run, verify FAIL** (`'function' object has no attribute…` / speaker is the lambda object).
- [ ] **Step 3: Implement.**

`notify.py`: signature `enabled: bool | Callable[[], bool]`; in `__init__` store `self._enabled = enabled`; in `notify` replace `if not self._enabled: return` with:

```python
        enabled = self._enabled() if callable(self._enabled) else self._enabled
        if not enabled:
            return
```
(update the docstring: "enabled: Whether to send notifications, or a function returning that, checked on every notification so a Settings change applies immediately.")

`qwen3_tts.py`: `speaker: str | Callable[[], str] = "Ryan"` (import `Callable`); keep `self._speaker = speaker`; add

```python
    def _current_speaker(self) -> str:
        """The voice to use right now (re-read on every synthesis so a change in Settings applies to the next reading)."""
        return self._speaker() if callable(self._speaker) else self._speaker
```
and use `speaker=self._current_speaker()` in `generate_custom_voice`. Update the `speaker` docstring.

- [ ] **Step 4: Run, verify PASS** — `.venv/bin/pytest tests/test_notify.py tests/test_qwen3_tts.py -q --no-cov && .venv/bin/mypy`.
- [ ] **Step 5: Commit** — `git add src/vocalis/notify.py src/vocalis/engines/qwen3_tts.py tests/test_notify.py tests/test_qwen3_tts.py && git commit -m "feat: let the notifier and voice read live settings"`

---

### Task 4: StatusHub

**Files:** Create `src/vocalis/ui/status_hub.py`; Test `tests/test_status_hub.py`.

**Interfaces:** Produces `ReadingControls(stop: Callable[[], None], pause: Callable[[], None] | None = None, resume: Callable[[], None] | None = None)` (frozen dataclass; `can_pause` property = both pause and resume are set); `StatusHub(QObject)` with signals `state_changed(str)` ("idle"|"listening"|"reading"), `status_changed(str)`, `progress_changed(int, int)`, `reading_started(str, bool)` (source, can_pause), `reading_finished()`, `paused_changed(bool)`; thread-safe methods `set_listening(bool)`, `set_next_source(label)`, `begin_reading(source, controls)` (idempotent while already reading; a pending next-source label replaces `source` once), `end_reading()` (idempotent when not reading), `report_status(text)`, `report_progress(current, total)`, `set_paused(bool)`; GUI-side commands `pause()`, `resume()`, `stop()`, `toggle_pause()` (call the controls of the active reading, no-ops when none); properties `state`, `is_reading`, `is_paused`, `source`.

- [ ] **Step 1: Failing tests** — create `tests/test_status_hub.py`:

```python
"""Tests for vocalis.ui.status_hub.StatusHub (synchronous: same-thread signal emits are direct)."""
import threading

from vocalis.ui.status_hub import ReadingControls, StatusHub


class Recorder:
    def __init__(self, hub: StatusHub) -> None:
        self.events: "list[tuple[object, ...]]" = []
        hub.state_changed.connect(lambda s: self.events.append(("state", s)))
        hub.reading_started.connect(lambda src, can: self.events.append(("started", src, can)))
        hub.reading_finished.connect(lambda: self.events.append(("finished",)))
        hub.paused_changed.connect(lambda p: self.events.append(("paused", p)))
        hub.status_changed.connect(lambda t: self.events.append(("status", t)))
        hub.progress_changed.connect(lambda c, t: self.events.append(("progress", c, t)))


class Calls:
    def __init__(self) -> None:
        self.log: "list[str]" = []

    def controls(self, pausable: bool = True) -> ReadingControls:
        return ReadingControls(
            stop=lambda: self.log.append("stop"),
            pause=(lambda: self.log.append("pause")) if pausable else None,
            resume=(lambda: self.log.append("resume")) if pausable else None,
        )


def test_starts_idle() -> None:
    hub = StatusHub()
    assert hub.state == "idle" and not hub.is_reading and not hub.is_paused and hub.source == ""


def test_reading_lifecycle_emits_in_order() -> None:
    hub = StatusHub()
    rec = Recorder(hub)
    hub.begin_reading("Document", Calls().controls())
    hub.report_status("Reading...")
    hub.report_progress(1, 3)
    hub.end_reading()
    assert rec.events == [
        ("state", "reading"), ("started", "Document", True),
        ("status", "Reading..."), ("progress", 1, 3),
        ("finished",), ("state", "idle"),
    ]


def test_begin_while_already_reading_and_end_while_idle_are_ignored() -> None:
    hub = StatusHub()
    rec = Recorder(hub)
    hub.end_reading()
    hub.begin_reading("Quick read", Calls().controls(pausable=False))
    hub.begin_reading("Document", Calls().controls())
    assert hub.source == "Quick read"
    assert [e for e in rec.events if e[0] == "started"] == [("started", "Quick read", False)]
    assert [e for e in rec.events if e[0] == "finished"] == []


def test_next_source_label_is_used_once() -> None:
    hub = StatusHub()
    hub.set_next_source("Screen text")
    hub.begin_reading("Document", Calls().controls())
    assert hub.source == "Screen text"
    hub.end_reading()
    hub.begin_reading("Document", Calls().controls())
    assert hub.source == "Document"


def test_listening_state_and_reading_priority() -> None:
    hub = StatusHub()
    rec = Recorder(hub)
    hub.set_listening(True)
    assert hub.state == "listening"
    hub.begin_reading("Document", Calls().controls())
    assert hub.state == "reading"
    hub.end_reading()
    assert hub.state == "listening"
    hub.set_listening(False)
    assert hub.state == "idle"
    assert [e for e in rec.events if e[0] == "state"] == [
        ("state", "listening"), ("state", "reading"), ("state", "listening"), ("state", "idle"),
    ]


def test_commands_call_the_active_readings_controls() -> None:
    hub = StatusHub()
    calls = Calls()
    hub.pause(); hub.stop()  # no active reading: harmless
    hub.begin_reading("Document", calls.controls())
    hub.pause()
    hub.resume()
    hub.stop()
    assert calls.log == ["pause", "resume", "stop"]


def test_toggle_pause_follows_the_paused_flag_and_set_paused_reports_it() -> None:
    hub = StatusHub()
    rec = Recorder(hub)
    calls = Calls()
    hub.begin_reading("Document", calls.controls())
    hub.toggle_pause()
    hub.set_paused(True)   # the reader confirms it paused
    hub.toggle_pause()
    hub.set_paused(False)
    assert calls.log == ["pause", "resume"]
    assert [e for e in rec.events if e[0] == "paused"] == [("paused", True), ("paused", False)]


def test_pause_is_ignored_for_a_stop_only_reading() -> None:
    hub = StatusHub()
    calls = Calls()
    hub.begin_reading("Quick read", calls.controls(pausable=False))
    hub.pause(); hub.toggle_pause(); hub.stop()
    assert calls.log == ["stop"]


def test_end_reading_clears_a_paused_state() -> None:
    hub = StatusHub()
    rec = Recorder(hub)
    hub.begin_reading("Document", Calls().controls())
    hub.set_paused(True)
    hub.end_reading()
    assert not hub.is_paused
    assert ("paused", False) in rec.events


def test_calls_from_another_thread_are_safe() -> None:
    hub = StatusHub()
    done = threading.Event()

    def worker() -> None:
        for _ in range(200):
            hub.begin_reading("Document", Calls().controls())
            hub.end_reading()
        done.set()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    assert done.wait(10.0)
    assert hub.state == "idle"
```

- [ ] **Step 2: Run, verify FAIL** — `ModuleNotFoundError: vocalis.ui.status_hub`.
- [ ] **Step 3: Implement** — create `src/vocalis/ui/status_hub.py`:

```python
"""The status board that the tray icon and the floating widget listen to.

Features (dictation, Quick Reader, Document Reader, OCR Reader) report what
they are doing; the tray and widget connect to the signals and never reach
into the features. All reporting methods are safe to call from any thread:
state changes happen under a short lock, and signals are emitted after the
lock is released (Qt delivers them queued to widgets living on the GUI
thread).
"""
import threading
from dataclasses import dataclass
from typing import Callable

from PySide6.QtCore import QObject, Signal

IDLE = "idle"
LISTENING = "listening"
READING = "reading"


@dataclass(frozen=True)
class ReadingControls:
    """What the widget/tray may do to the active reading."""

    stop: Callable[[], None]
    pause: Callable[[], None] | None = None
    resume: Callable[[], None] | None = None

    @property
    def can_pause(self) -> bool:
        """Whether this reading supports pause and resume."""
        return self.pause is not None and self.resume is not None


class StatusHub(QObject):
    """Thread-safe status board with Qt signals."""

    state_changed = Signal(str)
    status_changed = Signal(str)
    progress_changed = Signal(int, int)
    reading_started = Signal(str, bool)
    reading_finished = Signal()
    paused_changed = Signal(bool)

    def __init__(self, parent: QObject | None = None) -> None:
        """Create an idle hub."""
        super().__init__(parent)
        self._lock = threading.Lock()
        self._listening = False
        self._controls: ReadingControls | None = None
        self._source = ""
        self._next_source: str | None = None
        self._paused = False
        self._last_state = IDLE

    # -- queries -----------------------------------------------------------
    @property
    def state(self) -> str:
        """"reading", else "listening", else "idle"."""
        with self._lock:
            return self._compute_state_locked()

    @property
    def is_reading(self) -> bool:
        """Whether a reading is active (including while paused)."""
        with self._lock:
            return self._controls is not None

    @property
    def is_paused(self) -> bool:
        """Whether the active reading is paused."""
        with self._lock:
            return self._paused

    @property
    def source(self) -> str:
        """Label of the active reading ("" when none)."""
        with self._lock:
            return self._source

    def _compute_state_locked(self) -> str:
        """State derived from the fields; caller holds the lock."""
        if self._controls is not None:
            return READING
        return LISTENING if self._listening else IDLE

    def _take_state_change_locked(self) -> str | None:
        """If the derived state changed, remember and return it (else None); caller holds the lock."""
        state = self._compute_state_locked()
        if state == self._last_state:
            return None
        self._last_state = state
        return state

    # -- reporting (any thread) -------------------------------------------
    def set_listening(self, active: bool) -> None:
        """Dictation started/stopped listening to the microphone."""
        with self._lock:
            self._listening = active
            changed = self._take_state_change_locked()
        if changed is not None:
            self.state_changed.emit(changed)

    def set_next_source(self, label: str) -> None:
        """Name the next reading differently from the default (used once)."""
        with self._lock:
            self._next_source = label

    def begin_reading(self, source: str, controls: ReadingControls) -> None:
        """A reading started. Ignored if one is already being shown."""
        with self._lock:
            if self._controls is not None:
                return
            self._controls = controls
            self._source = self._next_source or source
            self._next_source = None
            self._paused = False
            label = self._source
            changed = self._take_state_change_locked()
        if changed is not None:
            self.state_changed.emit(changed)
        self.reading_started.emit(label, controls.can_pause)

    def end_reading(self) -> None:
        """The reading ended for any reason. Ignored if none is active."""
        with self._lock:
            if self._controls is None:
                return
            was_paused = self._paused
            self._controls = None
            self._source = ""
            self._paused = False
            changed = self._take_state_change_locked()
        if was_paused:
            self.paused_changed.emit(False)
        self.reading_finished.emit()
        if changed is not None:
            self.state_changed.emit(changed)

    def report_status(self, text: str) -> None:
        """A human-readable status line (e.g. "Reading...")."""
        self.status_changed.emit(text)

    def report_progress(self, current: int, total: int) -> None:
        """Progress through the text: piece `current` of `total`."""
        self.progress_changed.emit(current, total)

    def set_paused(self, paused: bool) -> None:
        """The reader confirms it paused/resumed."""
        with self._lock:
            if self._controls is None or self._paused == paused:
                return
            self._paused = paused
        self.paused_changed.emit(paused)

    # -- commands (GUI thread) ---------------------------------------------
    def pause(self) -> None:
        """Ask the active reading to pause (no-op if none or not pausable)."""
        with self._lock:
            controls = self._controls
        if controls is not None and controls.pause is not None and controls.can_pause:
            controls.pause()

    def resume(self) -> None:
        """Ask the active reading to resume (no-op if none or not pausable)."""
        with self._lock:
            controls = self._controls
        if controls is not None and controls.resume is not None and controls.can_pause:
            controls.resume()

    def toggle_pause(self) -> None:
        """Pause if playing, resume if paused."""
        if self.is_paused:
            self.resume()
        else:
            self.pause()

    def stop(self) -> None:
        """Ask the active reading to stop (no-op if none)."""
        with self._lock:
            controls = self._controls
        if controls is not None:
            controls.stop()
```

- [ ] **Step 4: Run, verify PASS** — `.venv/bin/pytest tests/test_status_hub.py -q --no-cov && .venv/bin/mypy`; loop 10x.
- [ ] **Step 5: Commit** — `git add src/vocalis/ui/status_hub.py tests/test_status_hub.py && git commit -m "feat: add the status hub that the tray and widget listen to"`

---

### Task 5: Optional hub hooks in the readers and dictation

**Files:** Modify `src/vocalis/features/doc_reader.py`, `quick_reader.py`, `dictation.py`; Test `tests/test_doc_reader.py`, `tests/test_quick_reader.py`, `tests/test_dictation.py` (append only).

**Interfaces:** Produces: `DocumentReaderFeature(..., on_started: Callable[[], None] | None = None)` (last parameter; called once at the start of every `read_file`/`read_text`, right after the new reading is marked active); `QuickReaderFeature(..., on_reading_changed: Callable[[bool], None] | None = None)` (last parameter; `True` when a fresh reading starts, `False` synchronously when it is stopped by a second press or by the coordinator, and `False` from `_deactivate` only when that generation is still current) plus public `stop() -> None` (same behaviour as the existing coordinator stop: no-op when idle); `DictationFeature(..., on_listening_changed: Callable[[bool], None] | None = None)` (last parameter; `True` after recording started OK, `False` when the key is released or starting failed).

- [ ] **Step 1: Failing tests** (read each test file first; reuse its builders/fakes — names below are the pattern, adapt to the real helper names):

`tests/test_doc_reader.py`:
```python
def test_on_started_is_called_once_per_reading() -> None:
    started: "list[None]" = []
    feature, _s, _p, finishes = _make_feature(FakeSynthesizer(), FakeSynthesizer(), RecordingNotifier())
    feature._on_started = lambda: started.append(None)  # noqa: SLF001  (or pass via a new _make_feature kwarg)
    feature.read_text("Hello there.")
    assert len(started) == 1
    wait_until(lambda: len(finishes) == 1)
```
(Prefer extending `_make_feature` with an `on_started` kwarg defaulting to None, if that does not change existing calls.)

`tests/test_quick_reader.py` (use the file's existing feature builder with a new optional `on_reading_changed` list-collector; the assertions):
```python
def test_reading_changes_are_reported_start_stop_and_natural_end() -> None: ...
    # press -> [True]; press again (stop) -> [True, False] immediately (synchronously, before the stop thread finishes)
def test_press_stop_press_quickly_ends_with_active_true() -> None: ...
    # press, press (stop), press (start): the recorded sequence ends with True and has no stray False after the final True
def test_public_stop_ends_an_active_reading_and_is_a_noop_when_idle() -> None: ...
def test_natural_end_reports_false_once() -> None: ...
```
`tests/test_dictation.py`:
```python
def test_listening_is_reported_true_on_press_and_false_on_release() -> None: ...
def test_listening_false_when_recording_fails_to_start() -> None: ...
def test_duplicate_press_does_not_report_twice() -> None: ...
```
Each test must assert exact recorded sequences (e.g. `== [True, False]`).

- [ ] **Step 2: Run, verify FAIL** (unexpected keyword argument).
- [ ] **Step 3: Implement.**
  - `doc_reader.py`: add the `on_started` constructor parameter + docstring ("Called with no arguments when a reading starts; the app uses it to show the widget"); store `self._on_started = on_started`; in both `read_file` and `read_text`, immediately after `generation = self._start_new_reading()` add `if self._on_started is not None: self._on_started()`. (Before `self._reading_coordinator.start_reading(...)` so the hub's begin comes after any preempted reading's end — check the order against `ocr_reader`: the OCR feature calls `reader.stop()` first, so the hub is already idle.)
  - `quick_reader.py`: add `on_reading_changed`; helper `_report_reading(self, active: bool) -> None` (calls the callback if set, swallowing and logging exceptions so a UI bug can never break reading); call `_report_reading(True)` in `on_trigger` right after a fresh reading is started (after `start_reading` coordination, before the thread starts); call `_report_reading(False)` in `on_trigger`'s interrupt branch **before** spawning the stop thread; in `_stop_for_coordinator` after clearing `_active` and before spawning the thread; in `_deactivate` only when `self._generation == generation` (inside the existing check, but call the callback after releasing the lock). Add:
    ```python
    def stop(self) -> None:
        """Stop the reading in progress, if any (used by the floating widget and the tray)."""
        self._stop_for_coordinator()
    ```
  - `dictation.py`: add `on_listening_changed`; helper `_report_listening(self, active)` (same safe pattern); in `on_press` after `self._recorder.start()` succeeds call `_report_listening(True)`; in the start-failure branch nothing is reported (it was never true) — but make sure no `True` was emitted; in `on_release` after flipping `_active` to False (before stopping the recorder) call `_report_listening(False)`.
- [ ] **Step 4: Run, verify PASS** — `.venv/bin/pytest tests/test_doc_reader.py tests/test_quick_reader.py tests/test_dictation.py -q --no-cov && .venv/bin/mypy`; run the three files 10x.
- [ ] **Step 5: Commit** — `git add src/vocalis/features/doc_reader.py src/vocalis/features/quick_reader.py src/vocalis/features/dictation.py tests/test_doc_reader.py tests/test_quick_reader.py tests/test_dictation.py && git commit -m "feat: let the readers and dictation report their state to a status hub"`

---

### Task 6: FloatingWidget

**Files:** Create `src/vocalis/ui/floating_widget.py`; Test `tests/test_floating_widget.py`.

**Interfaces:** Consumes `StatusHub`. Produces `FloatingWidget(hub: StatusHub, enabled: Callable[[], bool])` (a `QWidget`) with `apply_enabled()` (hides the widget immediately if the switch is now off while it is visible).

Behaviour: window flags `Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.WindowDoesNotAcceptFocus`, attribute `WA_ShowWithoutActivating`; contents: source label, status label, progress label, a Pause/Resume button (hidden when `can_pause` is False) and a Stop button; on `reading_started(source, can_pause)` → set labels, show the widget placed at the bottom-right of the primary screen's `availableGeometry()` with a 16 px margin **only if `enabled()`**; on `reading_finished` → hide and clear labels; `status_changed`/`progress_changed` update labels (`"piece {c} of {t}"`); `paused_changed(p)` → button text `"Resume"`/`"Pause"`; Pause button → `hub.toggle_pause()`; Stop button → `hub.stop()`.

- [ ] **Step 1: Failing tests** (`qtbot`; the hub is real; signals are same-thread so they run synchronously):

```python
"""Tests for vocalis.ui.floating_widget.FloatingWidget."""
from PySide6.QtCore import Qt

from vocalis.ui.floating_widget import FloatingWidget
from vocalis.ui.status_hub import ReadingControls, StatusHub


def make(qtbot, enabled=lambda: True):  # type: ignore[no-untyped-def]
    hub = StatusHub()
    widget = FloatingWidget(hub, enabled)
    qtbot.addWidget(widget)
    return hub, widget


def pausable(log):  # type: ignore[no-untyped-def]
    return ReadingControls(stop=lambda: log.append("stop"), pause=lambda: log.append("pause"), resume=lambda: log.append("resume"))


def test_never_takes_focus_and_stays_on_top(qtbot) -> None:  # type: ignore[no-untyped-def]
    _hub, widget = make(qtbot)
    flags = widget.windowFlags()
    assert flags & Qt.WindowStaysOnTopHint
    assert flags & Qt.WindowDoesNotAcceptFocus
    assert flags & Qt.FramelessWindowHint
    assert widget.testAttribute(Qt.WA_ShowWithoutActivating)


def test_hidden_until_a_reading_starts_and_hidden_again_when_it_ends(qtbot) -> None:  # type: ignore[no-untyped-def]
    hub, widget = make(qtbot)
    assert not widget.isVisible()
    hub.begin_reading("Document", pausable([]))
    assert widget.isVisible()
    assert widget._source_label.text() == "Document"  # noqa: SLF001
    hub.end_reading()
    assert not widget.isVisible()


def test_status_and_progress_labels_update(qtbot) -> None:  # type: ignore[no-untyped-def]
    hub, widget = make(qtbot)
    hub.begin_reading("Document", pausable([]))
    hub.report_status("Reading...")
    hub.report_progress(3, 8)
    assert widget._status_label.text() == "Reading..."  # noqa: SLF001
    assert widget._progress_label.text() == "piece 3 of 8"  # noqa: SLF001


def test_buttons_drive_the_hub_and_pause_label_follows_the_reader(qtbot) -> None:  # type: ignore[no-untyped-def]
    log: "list[str]" = []
    hub, widget = make(qtbot)
    hub.begin_reading("Document", pausable(log))
    widget._pause_button.click()  # noqa: SLF001
    assert log == ["pause"]
    assert widget._pause_button.text() == "Pause"  # noqa: SLF001  (label changes only when the reader confirms)
    hub.set_paused(True)
    assert widget._pause_button.text() == "Resume"  # noqa: SLF001
    widget._pause_button.click()  # noqa: SLF001
    assert log == ["pause", "resume"]
    widget._stop_button.click()  # noqa: SLF001
    assert log[-1] == "stop"


def test_stop_only_reading_hides_the_pause_button(qtbot) -> None:  # type: ignore[no-untyped-def]
    hub, widget = make(qtbot)
    hub.begin_reading("Quick read", ReadingControls(stop=lambda: None))
    assert widget._pause_button.isHidden()  # noqa: SLF001
    assert not widget._stop_button.isHidden()  # noqa: SLF001


def test_switched_off_means_it_never_appears(qtbot) -> None:  # type: ignore[no-untyped-def]
    hub, widget = make(qtbot, enabled=lambda: False)
    hub.begin_reading("Document", pausable([]))
    assert not widget.isVisible()


def test_turning_the_switch_off_mid_reading_hides_it(qtbot) -> None:  # type: ignore[no-untyped-def]
    state = {"on": True}
    hub, widget = make(qtbot, enabled=lambda: state["on"])
    hub.begin_reading("Document", pausable([]))
    assert widget.isVisible()
    state["on"] = False
    widget.apply_enabled()
    assert not widget.isVisible()


def test_placed_in_the_bottom_right_corner_of_the_screen(qtbot) -> None:  # type: ignore[no-untyped-def]
    from PySide6.QtGui import QGuiApplication

    hub, widget = make(qtbot)
    hub.begin_reading("Document", pausable([]))
    area = QGuiApplication.primaryScreen().availableGeometry()
    assert widget.x() + widget.width() <= area.right() + 1
    assert widget.y() + widget.height() <= area.bottom() + 1
    assert widget.x() >= area.center().x() or widget.width() > area.width() // 2
```

- [ ] **Step 2: Run, verify FAIL** — `ModuleNotFoundError`.
- [ ] **Step 3: Implement** `src/vocalis/ui/floating_widget.py` (class docstring: "Small always-on-top box shown only while something is read aloud."; give every method a docstring):

```python
"""The floating mini-widget: shown only while something is read aloud."""
from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from vocalis.ui.status_hub import StatusHub

_MARGIN = 16


class FloatingWidget(QWidget):
    """Always-on-top, never-focused mini control for the active reading."""

    def __init__(self, hub: StatusHub, enabled: Callable[[], bool]) -> None:
        super().__init__(
            None,
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowDoesNotAcceptFocus,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self._hub = hub
        self._enabled = enabled
        self._source_label = QLabel("")
        self._status_label = QLabel("")
        self._progress_label = QLabel("")
        self._pause_button = QPushButton("Pause")
        self._stop_button = QPushButton("Stop")
        for button in (self._pause_button, self._stop_button):
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        row = QHBoxLayout()
        row.addWidget(self._pause_button)
        row.addWidget(self._stop_button)
        layout = QVBoxLayout()
        layout.addWidget(self._source_label)
        layout.addWidget(self._status_label)
        layout.addWidget(self._progress_label)
        layout.addLayout(row)
        self.setLayout(layout)
        hub.reading_started.connect(self._on_reading_started)
        hub.reading_finished.connect(self._on_reading_finished)
        hub.status_changed.connect(self._status_label.setText)
        hub.progress_changed.connect(self._on_progress)
        hub.paused_changed.connect(self._on_paused_changed)
        self._pause_button.clicked.connect(hub.toggle_pause)
        self._stop_button.clicked.connect(hub.stop)

    # slots: each with a one-line docstring in the real file
    def _on_reading_started(self, source: str, can_pause: bool) -> None: ...
    def _on_reading_finished(self) -> None: ...
    def _on_progress(self, current: int, total: int) -> None: ...
    def _on_paused_changed(self, paused: bool) -> None: ...
    def apply_enabled(self) -> None: ...
    def _place_bottom_right(self) -> None: ...
```
Fill the bodies: `_on_reading_started` sets `_source_label`, resets `_status_label`/`_progress_label` to `""`, `_pause_button.setText("Pause")`, `_pause_button.setVisible(can_pause)`, and if `self._enabled()`: `self.adjustSize(); self._place_bottom_right(); self.show()` (never `raise_`/`activateWindow`); `_on_reading_finished` hides and clears labels; `_on_progress` sets `f"piece {current} of {total}"`; `_on_paused_changed` sets `"Resume"` if paused else `"Pause"`; `apply_enabled` hides if visible and not enabled; `_place_bottom_right` uses `QGuiApplication.primaryScreen().availableGeometry()` (guard `None`) and moves to `(right - width - _MARGIN, bottom - height - _MARGIN)`.

- [ ] **Step 4: Run, verify PASS** — `.venv/bin/pytest tests/test_floating_widget.py -q --no-cov && .venv/bin/mypy`.
- [ ] **Step 5: Commit** — `git add src/vocalis/ui/floating_widget.py tests/test_floating_widget.py && git commit -m "feat: add the floating reading widget"`

---

### Task 7: TrayIcon

**Files:** Create `src/vocalis/ui/tray.py`; Test `tests/test_tray.py`.

**Interfaces:** Consumes `StatusHub`. Produces `TrayIcon(hub, on_open_document_reader: Callable[[], None], on_open_settings: Callable[[], None], on_quit: Callable[[], None])` (a `QObject`) with `show() -> None` (shows the icon if `QSystemTrayIcon.isSystemTrayAvailable()`, otherwise logs ONE warning "No system tray available; continuing without the tray icon" and does nothing), `hide() -> None`, `available` property, and the attributes used by tests: `_status_action`, `_pause_action`, `_stop_action`, `_docs_action`, `_settings_action`, `_quit_action`, `_tray` (the `QSystemTrayIcon`).

Behaviour: icon = a 22x22 pixmap with a filled circle drawn by `QPainter` (grey `#9e9e9e` idle, red `#e53935` listening, blue `#1e88e5` reading); tooltip `"Vocalis — idle"` / `"listening"` / `"reading: <source>"`; the menu's first (disabled) action shows the same text; `Pause`/`Resume` action enabled only while a reading with `can_pause` is active and labelled `"Resume"` when paused; `Stop` enabled only while reading; the other three actions call the injected callbacks; the menu is built once and updated from hub signals.

- [ ] **Step 1: Failing tests** (`qtbot` + real `StatusHub`; the tray icon is created but `show()` is only called in the availability tests with `QSystemTrayIcon.isSystemTrayAvailable` patched):

```python
"""Tests for vocalis.ui.tray.TrayIcon."""
import logging
from unittest.mock import patch

from vocalis.ui.status_hub import ReadingControls, StatusHub
from vocalis.ui.tray import TrayIcon


def make(qtbot):  # type: ignore[no-untyped-def]
    hub = StatusHub()
    calls: "list[str]" = []
    tray = TrayIcon(hub, on_open_document_reader=lambda: calls.append("docs"),
                    on_open_settings=lambda: calls.append("settings"), on_quit=lambda: calls.append("quit"))
    return hub, tray, calls


def test_idle_menu_state(qtbot) -> None:  # type: ignore[no-untyped-def]
    _hub, tray, _calls = make(qtbot)
    assert tray._status_action.text() == "Vocalis — idle"  # noqa: SLF001
    assert not tray._status_action.isEnabled()  # noqa: SLF001
    assert not tray._pause_action.isEnabled() and not tray._stop_action.isEnabled()  # noqa: SLF001
    assert tray._tray.toolTip() == "Vocalis — idle"  # noqa: SLF001


def test_reading_enables_pause_and_stop_and_names_the_source(qtbot) -> None:  # type: ignore[no-untyped-def]
    hub, tray, _calls = make(qtbot)
    hub.begin_reading("Screen text", ReadingControls(stop=lambda: None, pause=lambda: None, resume=lambda: None))
    assert tray._tray.toolTip() == "Vocalis — reading: Screen text"  # noqa: SLF001
    assert tray._pause_action.isEnabled() and tray._stop_action.isEnabled()  # noqa: SLF001
    hub.set_paused(True)
    assert tray._pause_action.text() == "Resume"  # noqa: SLF001
    hub.end_reading()
    assert not tray._pause_action.isEnabled() and not tray._stop_action.isEnabled()  # noqa: SLF001
    assert tray._pause_action.text() == "Pause"  # noqa: SLF001


def test_stop_only_reading_leaves_pause_disabled(qtbot) -> None:  # type: ignore[no-untyped-def]
    hub, tray, _calls = make(qtbot)
    hub.begin_reading("Quick read", ReadingControls(stop=lambda: None))
    assert not tray._pause_action.isEnabled() and tray._stop_action.isEnabled()  # noqa: SLF001


def test_listening_changes_the_text_and_the_icon_colour(qtbot) -> None:  # type: ignore[no-untyped-def]
    hub, tray, _calls = make(qtbot)
    idle_icon = tray._tray.icon().pixmap(22, 22).toImage().pixelColor(11, 11).name()  # noqa: SLF001
    hub.set_listening(True)
    assert tray._tray.toolTip() == "Vocalis — listening"  # noqa: SLF001
    listening = tray._tray.icon().pixmap(22, 22).toImage().pixelColor(11, 11).name()  # noqa: SLF001
    assert listening == "#e53935" and idle_icon == "#9e9e9e"
    hub.begin_reading("Document", ReadingControls(stop=lambda: None))
    assert tray._tray.icon().pixmap(22, 22).toImage().pixelColor(11, 11).name() == "#1e88e5"  # noqa: SLF001


def test_menu_actions_call_the_callbacks_and_controls(qtbot) -> None:  # type: ignore[no-untyped-def]
    hub, tray, calls = make(qtbot)
    log: "list[str]" = []
    hub.begin_reading("Document", ReadingControls(stop=lambda: log.append("stop"),
                                                  pause=lambda: log.append("pause"), resume=lambda: log.append("resume")))
    tray._pause_action.trigger()  # noqa: SLF001
    tray._stop_action.trigger()  # noqa: SLF001
    tray._docs_action.trigger()  # noqa: SLF001
    tray._settings_action.trigger()  # noqa: SLF001
    tray._quit_action.trigger()  # noqa: SLF001
    assert log == ["pause", "stop"]
    assert calls == ["docs", "settings", "quit"]


def test_show_without_a_system_tray_warns_once_and_does_not_crash(qtbot, caplog) -> None:  # type: ignore[no-untyped-def]
    _hub, tray, _calls = make(qtbot)
    with patch("vocalis.ui.tray.QSystemTrayIcon.isSystemTrayAvailable", return_value=False):
        with caplog.at_level(logging.WARNING, logger="vocalis"):
            tray.show()
            tray.show()
    assert [r.getMessage() for r in caplog.records].count("No system tray available; continuing without the tray icon") == 1
    assert tray.available is False


def test_show_with_a_system_tray_shows_the_icon(qtbot) -> None:  # type: ignore[no-untyped-def]
    _hub, tray, _calls = make(qtbot)
    with patch("vocalis.ui.tray.QSystemTrayIcon.isSystemTrayAvailable", return_value=True):
        tray.show()
    assert tray.available is True
    tray.hide()
```

- [ ] **Step 2: Run, verify FAIL**; **Step 3: Implement** `tray.py` (module docstring, class + method docstrings; the icon pixmap built by `_make_icon(colour: str) -> QIcon` — transparent 22x22 `QPixmap`, `QPainter` with `Antialiasing`, `setBrush(QColor(colour))`, `setPen(Qt.NoPen)`, `drawEllipse(3, 3, 16, 16)`; the pixel at (11, 11) is the dot's colour). Menu: `QMenu` created and kept as an attribute (parent the tray so it is not garbage-collected), actions in this order: status (disabled), Pause, Stop, separator, Open Document Reader, Settings…, separator, Quit. Hub connections: `state_changed` → refresh text/icon; `reading_started(source, can_pause)` → remember source and can_pause, enable Stop (and Pause if can_pause), refresh; `reading_finished` → disable both, reset Pause label; `paused_changed(p)` → label. The pause action calls `hub.toggle_pause`, stop calls `hub.stop`.
- [ ] **Step 4: Run, verify PASS** — `.venv/bin/pytest tests/test_tray.py -q --no-cov && .venv/bin/mypy`.
- [ ] **Step 5: Commit** — `git add src/vocalis/ui/tray.py tests/test_tray.py && git commit -m "feat: add the tray icon"`

---

### Task 8: SettingsWindow

**Files:** Create `src/vocalis/ui/settings_window.py`; Test `tests/test_settings_window.py`.

**Interfaces:** Consumes `LiveSettings` (Task 2), parser helpers (Task 1), `VOICES`. Produces `SettingsWindow(settings: LiveSettings, on_applied: Callable[[], None] = lambda: None)` (a `QWidget`, title "Vocalis — Settings") with the widgets listed below and `load_from_settings() -> None`.

Widgets (names used by tests): `_notifications_box` and `_widget_box` (`QCheckBox`, "Show desktop notifications" / "Show the floating widget while reading"); `_ptt_combo` (items `PUSH_TO_TALK_CHOICES`), `_quick_combo`, `_doc_combo`, `_ocr_combo` (items `CTRL_ALT_CHOICES`, label text for the rows "Dictation (hold)", "Quick Reader  Ctrl+Alt+", "Document Reader  Ctrl+Alt+", "OCR Reader  Ctrl+Alt+"); `_voice_combo` (items `VOICES`, plus the configured voice if it is not in the list); `_save_button` ("Save"), `_close_button` ("Close"); `_message_label` (`QLabel`).

Behaviour of `_on_save_clicked`: build `new = settings.config.model_copy(update={...})` from the widgets (shortcuts via `format_ctrl_alt_shortcut`, `numlock_key` = `_ptt_combo.currentText()`, voice into `engines.qwen3_tts_speaker`); call `settings.update(new)`; on `ShortcutError` → `_message_label` = `"Not saved: <reason>"`; on `ConfigError` → `"Not saved: <reason>"` (the live settings are unchanged); on success → call `on_applied()`, message `"Saved."`, plus `" Restart Vocalis to use the new shortcuts."` when `settings.shortcuts_need_restart()`. `load_from_settings` fills the widgets from `settings.config`; a stored shortcut the parser cannot read falls back to the default choice for that row (never raises). `_close_button` hides the window; closing the window only hides it.

- [ ] **Step 1: Failing tests** (`qtbot`; `LiveSettings(VocalisConfig(), tmp_path / "config.toml")`, optionally with a failing `saver`):

```python
def test_widgets_show_the_current_settings(qtbot, tmp_path) -> None: ...
    # defaults: both boxes checked, ptt "NUMLOCK", quick "R", doc "D", ocr "O", voice "Ryan"
def test_save_writes_the_file_applies_settings_and_calls_on_applied(qtbot, tmp_path) -> None: ...
    # untick notifications, voice "Dylan", click save -> load_config(file) matches, settings.voice()=="Dylan",
    # notifications_enabled False, on_applied called once, message == "Saved."
def test_changing_a_shortcut_tells_the_user_to_restart(qtbot, tmp_path) -> None: ...
    # ocr combo -> "P": message endswith "Restart Vocalis to use the new shortcuts."
def test_duplicate_shortcuts_are_refused_with_the_reason(qtbot, tmp_path) -> None: ...
    # quick combo -> "O" (same as OCR): message startswith "Not saved:"; no file written; settings unchanged; on_applied not called
def test_save_failure_is_shown_and_nothing_changes(qtbot, tmp_path) -> None: ...
    # saver raises ConfigError("disk full"): message contains "disk full", settings unchanged
def test_unknown_stored_voice_is_still_selectable_and_kept(qtbot, tmp_path) -> None: ...
    # config voice "Custom": combo currentText "Custom"; saving without touching it keeps "Custom"
def test_unreadable_stored_shortcut_falls_back_to_the_default_choice(qtbot, tmp_path) -> None: ...
    # config.shortcuts.ocr_reader="Ctrl+Alt+Banana": no exception; ocr combo shows "O"
def test_close_button_hides_without_quitting(qtbot, tmp_path) -> None: ...
```
Write each test fully (arrange via the combos' `setCurrentText`, click via `_save_button.click()`).

- [ ] **Step 2: Run, verify FAIL**; **Step 3: Implement** the window as specified (layout: `QFormLayout` for the rows, checkboxes above, message label and a button row below; docstrings everywhere; `closeEvent` → `event.ignore(); self.hide()`? — no: the default close just hides a top-level `QWidget` that is not deleted on close; do not override; rely on `setQuitOnLastWindowClosed(False)` in Task 10).
- [ ] **Step 4: Run, verify PASS** — `.venv/bin/pytest tests/test_settings_window.py -q --no-cov && .venv/bin/mypy`.
- [ ] **Step 5: Commit** — `git add src/vocalis/ui/settings_window.py tests/test_settings_window.py && git commit -m "feat: add the settings window"`

---

### Task 9: Configurable hotkeys and live settings in the builders

**Files:** Modify `src/vocalis/hotkeys/numlock.py`, `src/vocalis/cli.py` (builders only); Test `tests/test_numlock.py` (append), `tests/test_cli.py` (append).

**Interfaces:**
- Consumes: Tasks 1-5.
- Produces: `NumlockListener(on_press, on_release, device_enumerator, key_code: int = 69)` (watches `key_code` instead of the hard-coded NUMLOCK); `list_numlock_capable_devices(key_code: int = 69)`; builders gain **optional keyword parameters** (defaults keep all existing calls/tests valid): `build_dictation(config, model_manager, settings: LiveSettings | None = None, hub: StatusHub | None = None, key_code: int = 69)`, `build_quick_reader(config, model_manager, reading_coordinator, settings=None, hub=None, key_code: int = 19)`, `build_doc_reader(config, model_manager, reading_coordinator, settings=None, hub=None, key_code: int = 32)`, `build_ocr_reader(config, document_reader, popup, settings=None, hub=None, key_code: int = 24)`. Module function `resolve_shortcuts(config: VocalisConfig) -> ShortcutSet` (invalid/duplicate shortcuts: `logger.error("Invalid shortcut in the config (%s); using the default shortcuts", exc)` and return `DEFAULT_SHORTCUTS`).

Behaviour when `settings` is given: `Notifier(enabled=settings.notifications_enabled)` (the bound method, i.e. live) and `Qwen3TTS(..., speaker=settings.voice)` (live); when `settings` is None: exactly today's behaviour (`config.notifications_enabled`, `config.engines.qwen3_tts_speaker`). When `hub` is given: dictation passes `on_listening_changed=hub.set_listening`; Quick Reader passes `on_reading_changed` = a closure that calls `hub.begin_reading("Quick read", ReadingControls(stop=feature.stop))` on `True` and `hub.end_reading()` on `False` (closure over the `feature` variable assigned just after); Document Reader passes `on_started` = closure calling `hub.begin_reading("Document", ReadingControls(stop=feature.stop, pause=feature.pause, resume=feature.resume))`, and fans its three callbacks out to both the popup signals **and** the hub (`on_status(msg)`: popup signal + `hub.report_status(msg)`, and `hub.set_paused(True)` when `msg == "Paused"` / `hub.set_paused(False)` when `msg == "Resuming..."` — verify these exact strings in `DocumentReaderFeature.pause()/resume()` before relying on them; `on_progress`: popup + `hub.report_progress`; `on_finished`: popup signal + `hub.end_reading()`). Quick Reader keeps `CtrlAltRListener(device_enumerator=list_hotkey_capable_devices)` when `key_code == 19` and otherwise uses `CtrlAltKeyListener(key_code, on_trigger, lambda: list_devices_with_key(key_code))` (so the imported names tests patch stay used). Dictation uses `device_enumerator=lambda: list_numlock_capable_devices(key_code)`; Document and OCR listeners use their `key_code`. OCR: `on_reading_starting` becomes a closure: `hub.set_next_source("Screen text")` when a hub is given; then **if `settings` is not None and `settings.floating_widget_enabled()` the popup is not opened** (the widget shows), otherwise `popup.external_reading_started.emit(text)` as today (so with `settings=None` the behaviour and the existing test are unchanged).

- [ ] **Step 1: Failing tests.**
  - `tests/test_numlock.py`: a listener built with `key_code=70` and fake devices calls `on_press`/`on_release` for events with code 70 and ignores code 69; default (no `key_code`) still reacts to 69 only; `_is_watchable("kbd", {70}, 70)` is True and `_is_watchable("kbd", {69}, 70)` is False (extend `_is_watchable(name, capabilities, key_code=69)` the same way, keeping the ydotoold exclusion).
  - `tests/test_cli.py` (use the file's existing patterns, `qtbot`/`qapp` where widgets are built):
    - `test_resolve_shortcuts_returns_defaults_for_a_bad_config(caplog)`: `ShortcutsConfig(ocr_reader="Ctrl+Alt+Banana")` → `DEFAULT_SHORTCUTS` and one ERROR record; valid custom config (`ocr_reader="Ctrl+Alt+P"`) → `ocr_reader == 25`.
    - `test_build_dictation_uses_live_notifications_and_the_given_key`: with `LiveSettings`, `feature._notifier` evaluates `settings.notifications_enabled` live (flip the setting and assert the notifier's `_enabled()` result changes) and `listener._key_code == 70` when `key_code=70`.
    - `test_primary_engine_reads_the_live_voice`: for Quick Reader and Document Reader builders with a `LiveSettings`: `feature._primary_engine_factory()` returns a `Qwen3TTS` whose `_current_speaker()` equals `settings.voice()` **after** the setting is changed to `"Dylan"` (no engine reload needed).
    - `test_quick_reader_reports_to_the_hub`: building with a hub, calling `feature._on_reading_changed(True)` begins a reading with source `"Quick read"` and stop-only controls; `False` ends it. (Access the stored callback attribute set in Task 5.)
    - `test_doc_reader_reports_status_progress_finish_and_pause_to_the_hub(qtbot)`: call `feature._on_started()`, `feature._on_status("Paused")`, `feature._on_progress(2, 5)`, `feature._on_status("Resuming...")`, `feature._on_finished()`; assert the hub's recorded signals (source `"Document"`, `can_pause` True, paused True then False, progress, finished) **and** that the popup still got the status/progress/finished (existing behaviour, via `popup._status_label.text()`).
    - `test_ocr_shows_the_widget_not_the_popup_when_the_widget_is_enabled(qtbot)`: with `settings` (widget enabled) and a hub, invoking `feature._on_reading_starting("hi")` leaves the popup hidden and sets the hub's next source so a following `begin_reading("Document", …)` has source `"Screen text"`; with `floating_widget_enabled=False` the popup shows with the text (as in the existing test).
- [ ] **Step 2: Run, verify FAIL**; **Step 3: Implement** as specified above (keep every existing import in `cli.py`; docstrings updated).
- [ ] **Step 4: Run, verify PASS** — `.venv/bin/pytest tests/test_numlock.py tests/test_cli.py -q --no-cov && .venv/bin/mypy`, then the full suite once (existing tests must all still pass).
- [ ] **Step 5: Commit** — `git add src/vocalis/hotkeys/numlock.py src/vocalis/cli.py tests/test_numlock.py tests/test_cli.py && git commit -m "feat: read shortcuts and live settings in the feature builders"`

---

### Task 10: Assemble the UI in main()

**Files:** Modify `src/vocalis/cli.py` (`main()` and a new `build_ui`); Test `tests/test_cli.py` (append).

**Interfaces:**
- Consumes: Tasks 4, 6, 7, 8, 9.
- Produces: `build_ui(settings: LiveSettings, hub: StatusHub, doc_popup: DocumentReaderPopup, on_quit: Callable[[], None]) -> tuple[FloatingWidget, SettingsWindow, TrayIcon]`.

`main()` flow after loading the config: `settings = LiveSettings(config, default_config_path())`; `codes = resolve_shortcuts(config)`; `hub = StatusHub()` (created after the `QApplication`); pass `settings`, `hub` and the matching `key_code` to the four builders; `app.setQuitOnLastWindowClosed(False)` right after the `QApplication` is obtained (closing the Settings or Document Reader window must not quit Vocalis); `widget, settings_window, tray = build_ui(settings, hub, doc_popup, app.quit)`; `tray.show()` before `app.exec()`; in the `finally` block, `tray.hide()` and `widget.hide()` run inside `_stop_all_independently` entries (`("tray icon", tray.hide)`, `("floating widget", widget.hide)`) placed first so a failure in them cannot stop the listeners from stopping. `build_ui` wires: `FloatingWidget(hub, settings.floating_widget_enabled)`; `SettingsWindow(settings, on_applied=widget.apply_enabled)`; `TrayIcon(hub, on_open_document_reader=lambda: (doc_popup.show(), doc_popup.raise_()), on_open_settings=lambda: (settings_window.load_from_settings(), settings_window.show(), settings_window.raise_()), on_quit=on_quit)`.

- [ ] **Step 1: Failing tests.**
  - `test_build_ui_wires_tray_widget_and_settings(qtbot)`: call `build_ui` with a real `LiveSettings(VocalisConfig(), tmp_path / "c.toml")`, `StatusHub`, a `DocumentReaderPopup(FakeFeature)`, and an `on_quit` recorder; assert `tray._quit_action.trigger()` calls `on_quit`; `tray._settings_action.trigger()` shows the settings window with the current values; `tray._docs_action.trigger()` shows the popup; a hub `begin_reading` shows the widget and a settings-window save with the widget box unticked hides it (via `on_applied`).
  - `test_main_does_not_quit_when_the_last_window_closes(qapp)`: run `main()` with the usual patches (`QApplication.exec` patched to a fake that asserts `QApplication.instance().quitOnLastWindowClosed() is False` and returns 0).
  - `test_main_creates_live_settings_from_the_config_and_hides_the_tray_on_exit(qapp)`: patches as in the existing main tests plus `vocalis.cli.TrayIcon` mocked; assert `tray.show` called once, `tray.hide` called once, and that none of the existing stop calls was skipped when `tray.hide` raises (make the mock raise → daemon.stop still called).
  - Existing `main()` tests must pass unchanged (they run under the offscreen platform where no tray exists → one warning only).
- [ ] **Step 2: Run, verify FAIL**; **Step 3: Implement**; **Step 4: Run, verify PASS** — `.venv/bin/pytest tests/test_cli.py -q --no-cov && .venv/bin/mypy`, then the full suite once.
- [ ] **Step 5: Commit** — `git add src/vocalis/cli.py tests/test_cli.py && git commit -m "feat: show the tray icon, floating widget and settings window"`

---

### Task 11: Docs, change log, full verification, hands-on check

**Files:** Modify `docs/user-docs.md`, `docs/architecture.md`, `docs/workflow.md`, `docs/file-purpose.md`, `docs/folder-structure.md`, `docs/developer-docs.md`, the Phase 6 amendment in the spec (planning deviations); Create `.claude/<YYYY-MM-DD_HHMM>_CHANGELOG.md`.

- [ ] **Step 1: Docs** — verify every sentence against the code. User docs: the tray icon (colours, menu, Quit), the floating widget (when it shows, buttons, Quick Reader is Stop-only, OCR shows the widget instead of the big window unless the widget is off), the Settings window (what each control does, which apply immediately, shortcuts need a restart, allowed shortcut keys, what "Not saved" means), the voice list. Architecture/workflow: the `LiveSettings` + `StatusHub` design and the data flow. File purpose/folder structure: the new files. Developer docs: how to add a new setting (field in config, read via `LiveSettings` at point of use, control in the window), how the hub is fed. Spec: list any deviations from the amendment decided during planning/implementation.
- [ ] **Step 2: Change log** — new file with what was added, files, tests, decisions, rollback (`git revert` the merge), and the hands-on result placeholder `[ ]` until Step 5.
- [ ] **Step 3: Full verification** — `.venv/bin/pytest -q` (all pass, zero warnings, coverage >= 85%), `.venv/bin/mypy` (Success), and loop `tests/test_status_hub.py tests/test_floating_widget.py tests/test_tray.py tests/test_settings_window.py tests/test_cli.py` 10x (all green every time).
- [ ] **Step 4: Offscreen smoke** — run a small script with `QT_QPA_PLATFORM=offscreen` that builds `StatusHub`, `FloatingWidget`, `SettingsWindow`, `TrayIcon` with a temp config path, drives a fake reading through the hub, saves a changed config through the window, and prints what happened; confirm the temp file contains the new values and the user's real config was not touched (`stat` its mtime before/after).
- [ ] **Step 5: Real-hardware check with the user** (the controller does this, not the implementer): start `vocalis`; ask the user to (a) see the tray icon and read its tooltip/menu, (b) hold NUMLOCK to dictate and watch the dot turn red, (c) read something and see the widget (Pause/Resume/Stop, progress), (d) press Ctrl+Alt+O on screen text and see the widget (not the big window), (e) open Settings from the tray, untick the widget and notifications and change the voice, read again and hear/see the change without restarting, (f) change the OCR shortcut, see the restart notice, restart, and use the new shortcut, (g) close the Settings window (Vocalis must keep running), (h) Quit from the tray menu and via Ctrl+C: clean exit, no ERROR/Traceback lines. Note: the ALSA `default` audio route is silent on this machine; for the hands-on run start Vocalis with `sd.default.device = (sd.default.device[0], 'pulse')` as in the Phase 5 check. Record the outcome in the change log.
- [ ] **Step 6: Commit** — `git add docs .claude/*CHANGELOG.md && git commit -m "docs: document the ui phase and record the phase 6 change log"`
