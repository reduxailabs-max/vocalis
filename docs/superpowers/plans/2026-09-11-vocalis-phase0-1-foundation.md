# Vocalis Phase 0+1: Foundation (Scaffold + GPU Model Manager) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the Vocalis project skeleton — package layout, typed config,
logging, a daemon that starts/stops cleanly over a Unix socket, and a fully
tested Model Manager that enforces "only one AI model in VRAM at a time,
released a few seconds after it goes idle." This is the foundation every
later feature (Dictation, Quick Reader, Document Reader, OCR) will be built
on top of.

**Architecture:** A `VocalisDaemon` process owns a `ModelManager`. Feature
code (added in later plans) will call `model_manager.acquire(engine_id,
factory)` to get a loaded engine, use it, then call `release(engine_id)` —
the manager itself decides when to actually unload from VRAM. Config is a
`pydantic` model loaded from `~/.config/vocalis/config.toml`. No real AI
model or GPU code is wired in yet — this plan proves the *mechanism* with
fake engines; Phase 2 plugs in the real Parakeet engine and verifies actual
VRAM behavior with `nvidia-smi`.

**Tech Stack:** Python 3.13, `pydantic` v2, `tomli-w` (TOML writing; reading
uses stdlib `tomllib`), `pytest` + `pytest-cov`, `mypy` (strict).

**Spec:** `docs/superpowers/specs/2026-09-11-vocalis-design.md`

## Global Constraints

- Fully offline/local — nothing in this plan makes network calls.
- Only one AI model may be resident in VRAM at a time; idle models must be
  released after `model_idle_timeout_seconds` (default 30.0) of inactivity.
- Every function gets a docstring; structured error handling with specific
  exception types (no bare `except:`); no `eval`/unsanitized shell calls.
- Structured logging via the `logging` module, logger name `"vocalis"`,
  never logging secrets or full file contents.
- ≥85% test coverage (enforced by `pytest-cov --cov-fail-under=85`).
- `old/` is never modified.
- Conventional-commit messages (`feat:`, `fix:`, `test:`, `docs:`, `chore:`).
- One sub-task at a time — stop after each task and wait for explicit user
  "yes"/"y" before continuing (per project task-list protocol).

---

### Task 1: Project Scaffold & Tooling Config

**Files:**
- Create: `pyproject.toml`
- Create: `src/vocalis/__init__.py`
- Create: `src/vocalis/py.typed`
- Create: `tests/__init__.py`
- Create: `tests/test_package.py`
- Create: `.gitignore`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: an installable `vocalis` package with `vocalis.__version__: str`,
  and the mypy/pytest/coverage tool configuration every later task relies on.

- [ ] **Step 1: Create the directory skeleton**

```bash
mkdir -p src/vocalis tests
touch src/vocalis/py.typed tests/__init__.py
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "vocalis"
version = "0.1.0"
description = "Offline voice AI workstation: dictation, quick reader, document reader, OCR reader"
requires-python = ">=3.13,<3.14"
dependencies = [
    "pydantic>=2.6,<3",
    "tomli-w>=1.0,<2",
]

[project.optional-dependencies]
dev = [
    "pytest>=8,<9",
    "pytest-cov>=5,<6",
    "mypy>=1.10,<2",
]

[project.scripts]
vocalis = "vocalis.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["src/vocalis"]

[tool.mypy]
python_version = "3.13"
strict = true
mypy_path = "src"
files = ["src/vocalis", "tests"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
addopts = "--cov=vocalis --cov-report=term-missing --cov-fail-under=85"

[tool.coverage.run]
source = ["src/vocalis"]
branch = true
```

- [ ] **Step 3: Write `.gitignore`**

```
__pycache__/
*.pyc
.mypy_cache/
.pytest_cache/
.coverage
htmlcov/
*.egg-info/
build/
dist/
.venv/
venv/
```

- [ ] **Step 4: Write the failing smoke test**

```python
# tests/test_package.py
"""Smoke test confirming the vocalis package is importable and versioned."""
import vocalis


def test_package_exposes_version() -> None:
    assert isinstance(vocalis.__version__, str)
    assert vocalis.__version__ == "0.1.0"
```

- [ ] **Step 5: Run the test, verify it fails**

Run: `python3.13 -m venv .venv && .venv/bin/pip install -e ".[dev]" && .venv/bin/pytest tests/test_package.py -v`
Expected: FAIL — `AttributeError: module 'vocalis' has no attribute '__version__'`
(or `ImportError` if `src/vocalis/__init__.py` doesn't exist yet — create it
empty first if needed, then re-run to confirm the `AttributeError` failure
specifically).

- [ ] **Step 6: Write the minimal implementation**

```python
# src/vocalis/__init__.py
"""Vocalis: offline voice AI workstation (dictation, quick reader, document reader, OCR)."""

__version__ = "0.1.0"
```

- [ ] **Step 7: Run the test, verify it passes**

Run: `.venv/bin/pytest tests/test_package.py -v`
Expected: PASS

- [ ] **Step 8: Run mypy, verify it is clean**

Run: `.venv/bin/mypy`
Expected: `Success: no issues found`

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml .gitignore src/vocalis/__init__.py src/vocalis/py.typed tests/__init__.py tests/test_package.py
git commit -m "chore: scaffold vocalis package with mypy/pytest/coverage config" -m "- Adds pyproject.toml (hatchling build, strict mypy, pytest+coverage gated at 85%)" -m "- Adds installable src/vocalis package with __version__" -m "Phase 0 of vocalis foundation plan" -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01GMitM6j81rZ9TN8k3Q5tAh"
```

---

### Task 2: Structured Logging Setup

**Files:**
- Create: `src/vocalis/logging_setup.py`
- Create: `tests/test_logging_setup.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `configure_logging(level: str = "INFO", log_file: Path | None = None) -> logging.Logger`
  — returns the `"vocalis"` logger, used by every later task instead of
  calling `logging.getLogger` directly.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_logging_setup.py
"""Tests for vocalis.logging_setup.configure_logging."""
import logging
from pathlib import Path

from vocalis.logging_setup import configure_logging


def test_returns_logger_named_vocalis() -> None:
    logger = configure_logging()
    assert logger.name == "vocalis"


def test_sets_requested_level() -> None:
    logger = configure_logging(level="DEBUG")
    assert logger.level == logging.DEBUG


def test_writes_to_log_file(tmp_path: Path) -> None:
    log_file = tmp_path / "vocalis.log"
    logger = configure_logging(level="INFO", log_file=log_file)
    logger.info("hello from test")
    for handler in logger.handlers:
        handler.flush()
    assert log_file.exists()
    assert "hello from test" in log_file.read_text()


def test_idempotent_does_not_duplicate_handlers(tmp_path: Path) -> None:
    log_file = tmp_path / "vocalis.log"
    configure_logging(level="INFO", log_file=log_file)
    logger = configure_logging(level="INFO", log_file=log_file)
    file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) == 1
```

- [ ] **Step 2: Run the tests, verify they fail**

Run: `.venv/bin/pytest tests/test_logging_setup.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vocalis.logging_setup'`

- [ ] **Step 3: Write the minimal implementation**

```python
# src/vocalis/logging_setup.py
"""Central logging configuration for the vocalis daemon and CLI.

Every module in vocalis logs through the single "vocalis" logger configured
here, so log level and destinations are controlled in one place. Never log
secrets, credentials, or full user text content — only status and metadata.
"""
import logging
from pathlib import Path

_LOGGER_NAME = "vocalis"
_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def configure_logging(level: str = "INFO", log_file: Path | None = None) -> logging.Logger:
    """Configure and return the shared "vocalis" logger.

    Safe to call multiple times (e.g. once at daemon startup, once in
    tests) — it will not attach duplicate handlers for the same log file.

    Args:
        level: Standard logging level name, e.g. "INFO", "DEBUG".
        log_file: Optional path to also write logs to a file. Parent
            directories are created if missing.

    Returns:
        The configured logger named "vocalis".
    """
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(getattr(logging, level.upper()))
    formatter = logging.Formatter(_FORMAT)

    has_console_handler = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )
    if not has_console_handler:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file is not None:
        already_has_this_file = any(
            isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_file
            for h in logger.handlers
        )
        if not already_has_this_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
```

- [ ] **Step 4: Run the tests, verify they pass**

Run: `.venv/bin/pytest tests/test_logging_setup.py -v`
Expected: PASS (all 4 tests)

- [ ] **Step 5: Run mypy, verify it is clean**

Run: `.venv/bin/mypy`
Expected: `Success: no issues found`

- [ ] **Step 6: Commit**

```bash
git add src/vocalis/logging_setup.py tests/test_logging_setup.py
git commit -m "feat: add shared vocalis logger configuration" -m "- configure_logging() sets up console + optional file handlers on the 'vocalis' logger" -m "- Idempotent: safe to call from both daemon startup and tests" -m "Phase 0 of vocalis foundation plan" -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01GMitM6j81rZ9TN8k3Q5tAh"
```

---

### Task 3: Pydantic Config Model + TOML Load/Save

**Files:**
- Create: `src/vocalis/config.py`
- Create: `config/default_config.toml`
- Create: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `VocalisConfig` (pydantic `BaseModel`) with fields: `shortcuts:
    ShortcutsConfig`, `engines: EnginesConfig`, `notifications_enabled: bool`,
    `floating_widget_enabled: bool`, `model_idle_timeout_seconds: float`,
    `gpu_vram_ceiling_mb: int`.
  - `ShortcutsConfig`: `numlock_key: str`, `quick_reader: str`,
    `document_reader: str`, `ocr_reader: str`.
  - `EnginesConfig`: `stt: str`, `quick_tts: str`, `doc_tts: str`.
  - `ConfigError(Exception)` — raised on unreadable/invalid config.
  - `default_config_path() -> Path`
  - `load_config(path: Path) -> VocalisConfig`
  - `save_config(config: VocalisConfig, path: Path) -> None`
  - These are what Task 6 (daemon+manager wiring) and every later feature
    plan will import to read shortcut bindings, engine choices, and the
    idle timeout.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_config.py
"""Tests for vocalis.config: pydantic models and TOML load/save."""
from pathlib import Path

import pytest

from vocalis.config import (
    ConfigError,
    VocalisConfig,
    default_config_path,
    load_config,
    save_config,
)


def test_defaults_match_spec() -> None:
    config = VocalisConfig()
    assert config.shortcuts.numlock_key == "NUMLOCK"
    assert config.shortcuts.quick_reader == "Ctrl+Alt+R"
    assert config.shortcuts.document_reader == "Ctrl+Alt+D"
    assert config.shortcuts.ocr_reader == "Ctrl+Alt+O"
    assert config.engines.stt == "parakeet"
    assert config.engines.quick_tts == "qwen3_tts"
    assert config.engines.doc_tts == "fish_speech"
    assert config.notifications_enabled is True
    assert config.floating_widget_enabled is True
    assert config.model_idle_timeout_seconds == 30.0
    assert config.gpu_vram_ceiling_mb == 6144


def test_default_config_path_is_under_home_config() -> None:
    path = default_config_path()
    assert path == Path.home() / ".config" / "vocalis" / "config.toml"


def test_load_missing_file_returns_defaults(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.toml"
    config = load_config(missing)
    assert config == VocalisConfig()


def test_save_then_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    original = VocalisConfig(model_idle_timeout_seconds=45.0, notifications_enabled=False)
    save_config(original, path)
    loaded = load_config(path)
    assert loaded == original


def test_save_creates_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "config.toml"
    save_config(VocalisConfig(), path)
    assert path.exists()


def test_load_invalid_toml_raises_config_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.toml"
    path.write_text("this is not [ valid toml")
    with pytest.raises(ConfigError):
        load_config(path)


def test_load_invalid_field_type_raises_config_error(tmp_path: Path) -> None:
    path = tmp_path / "bad_value.toml"
    path.write_text('model_idle_timeout_seconds = "not-a-number"\n')
    with pytest.raises(ConfigError):
        load_config(path)
```

- [ ] **Step 2: Run the tests, verify they fail**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vocalis.config'`

- [ ] **Step 3: Write the minimal implementation**

```python
# src/vocalis/config.py
"""Typed, validated configuration for vocalis, backed by a TOML file.

All config values that affect behavior (shortcuts, engine choices,
notification/widget toggles, GPU idle policy) live in one pydantic model so
invalid config is rejected loudly at load time instead of causing confusing
failures deep inside a feature.
"""
import tomllib
from pathlib import Path

import tomli_w
from pydantic import BaseModel, ValidationError


class ConfigError(Exception):
    """Raised when the config file cannot be read, parsed, or validated."""


class ShortcutsConfig(BaseModel):
    """Keyboard shortcut bindings for each vocalis feature."""

    numlock_key: str = "NUMLOCK"
    quick_reader: str = "Ctrl+Alt+R"
    document_reader: str = "Ctrl+Alt+D"
    ocr_reader: str = "Ctrl+Alt+O"


class EnginesConfig(BaseModel):
    """Which engine implementation each feature should use."""

    stt: str = "parakeet"
    quick_tts: str = "qwen3_tts"
    doc_tts: str = "fish_speech"


class VocalisConfig(BaseModel):
    """Top-level vocalis configuration."""

    shortcuts: ShortcutsConfig = ShortcutsConfig()
    engines: EnginesConfig = EnginesConfig()
    notifications_enabled: bool = True
    floating_widget_enabled: bool = True
    model_idle_timeout_seconds: float = 30.0
    gpu_vram_ceiling_mb: int = 6144


def default_config_path() -> Path:
    """Return the standard per-user config file location.

    Returns:
        `~/.config/vocalis/config.toml`.
    """
    return Path.home() / ".config" / "vocalis" / "config.toml"


def load_config(path: Path) -> VocalisConfig:
    """Load and validate config from a TOML file.

    If the file does not exist, returns default settings rather than
    raising — a missing config file is a normal first-run state, not an
    error.

    Args:
        path: Location of the TOML config file.

    Returns:
        A validated VocalisConfig.

    Raises:
        ConfigError: If the file exists but is not valid TOML, or its
            contents fail pydantic validation.
    """
    if not path.exists():
        return VocalisConfig()

    try:
        raw_text = path.read_bytes()
        data = tomllib.loads(raw_text.decode("utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in config file {path}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Could not read config file {path}: {exc}") from exc

    try:
        return VocalisConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"Invalid config values in {path}: {exc}") from exc


def save_config(config: VocalisConfig, path: Path) -> None:
    """Write config to a TOML file, creating parent directories as needed.

    Args:
        config: The validated config to persist.
        path: Destination file path.

    Raises:
        ConfigError: If the file cannot be written (e.g. permission denied).
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            tomli_w.dump(config.model_dump(mode="json"), handle)
    except OSError as exc:
        raise ConfigError(f"Could not write config file {path}: {exc}") from exc
```

- [ ] **Step 4: Run the tests, verify they pass**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: PASS (all 7 tests)

- [ ] **Step 5: Write the default config file shipped with the project**

```toml
# config/default_config.toml
# Default Vocalis configuration. Copied to ~/.config/vocalis/config.toml
# on first run if that file does not already exist.

[shortcuts]
numlock_key = "NUMLOCK"
quick_reader = "Ctrl+Alt+R"
document_reader = "Ctrl+Alt+D"
ocr_reader = "Ctrl+Alt+O"

[engines]
stt = "parakeet"
quick_tts = "qwen3_tts"
doc_tts = "fish_speech"

notifications_enabled = true
floating_widget_enabled = true
model_idle_timeout_seconds = 30.0
gpu_vram_ceiling_mb = 6144
```

- [ ] **Step 6: Run mypy, verify it is clean**

Run: `.venv/bin/mypy`
Expected: `Success: no issues found`

- [ ] **Step 7: Commit**

```bash
git add src/vocalis/config.py config/default_config.toml tests/test_config.py
git commit -m "feat: add pydantic config model with TOML load/save" -m "- VocalisConfig validates shortcuts, engine choices, notification/widget toggles, GPU idle policy" -m "- load_config falls back to defaults on missing file, raises ConfigError on invalid TOML or values" -m "- Ships config/default_config.toml as the first-run template" -m "Phase 0 of vocalis foundation plan" -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01GMitM6j81rZ9TN8k3Q5tAh"
```

---

### Task 4: Model Manager — Single-Residency Core

**Files:**
- Create: `src/vocalis/model_manager.py`
- Create: `tests/test_model_manager.py`

**Interfaces:**
- Consumes: nothing new (pure in-memory logic; no dependency on config or
  daemon yet).
- Produces:
  - `ModelEngine` (Protocol): `load(self) -> None`, `unload(self) -> None`,
    `is_loaded` property (`bool`). Every real engine wrapper built in Phase
    2+ (`ParakeetSTT`, `Qwen3TTS`, `FishSpeechTTS`, `TesseractOCR`) must
    satisfy this protocol.
  - `TimerHandle` (Protocol): `cancel(self) -> None`.
  - `TimerFactory = Callable[[float, Callable[[], None]], TimerHandle]`
  - `ModelManager.__init__(self, idle_timeout_seconds: float = 30.0,
    timer_factory: TimerFactory | None = None) -> None`
  - `ModelManager.acquire(self, engine_id: str, factory: Callable[[],
    ModelEngine]) -> ModelEngine`
  - `ModelManager.release(self, engine_id: str) -> None`
  - `ModelManager.shutdown(self) -> None`
  - `ModelManager.current_engine_id -> str | None` (property)
  - Task 5 adds the real (threading.Timer-backed) default `timer_factory`;
    this task uses a fake one so the idle-timer *logic* is proven without
    any real sleeping.

- [ ] **Step 1: Write the failing tests (with fakes)**

```python
# tests/test_model_manager.py
"""Tests for the ModelManager single-residency + idle-unload policy.

Uses FakeModelEngine and FakeTimerFactory so the idle-timeout logic is
proven deterministically, with no real GPU model and no real sleeping.
"""
import logging

import pytest

from vocalis.model_manager import ModelManager


class FakeModelEngine:
    def __init__(self, name: str) -> None:
        self.name = name
        self.load_calls = 0
        self.unload_calls = 0
        self._loaded = False

    def load(self) -> None:
        self.load_calls += 1
        self._loaded = True

    def unload(self) -> None:
        self.unload_calls += 1
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded


class FakeTimerHandle:
    def __init__(self, callback):  # type: ignore[no-untyped-def]
        self.callback = callback
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True

    def fire(self) -> None:
        if not self.cancelled:
            self.callback()


class FakeTimerFactory:
    def __init__(self) -> None:
        self.handles: list[FakeTimerHandle] = []
        self.last_delay: float | None = None

    def __call__(self, delay, callback):  # type: ignore[no-untyped-def]
        self.last_delay = delay
        handle = FakeTimerHandle(callback)
        self.handles.append(handle)
        return handle


def make_manager(idle_timeout: float = 30.0) -> tuple[ModelManager, FakeTimerFactory]:
    fake_timers = FakeTimerFactory()
    manager = ModelManager(idle_timeout_seconds=idle_timeout, timer_factory=fake_timers)
    return manager, fake_timers


def test_acquire_loads_model() -> None:
    manager, _ = make_manager()
    engine = FakeModelEngine("parakeet")
    result = manager.acquire("parakeet", lambda: engine)
    assert result is engine
    assert engine.load_calls == 1
    assert manager.current_engine_id == "parakeet"


def test_acquire_same_engine_reuses_without_reloading() -> None:
    manager, _ = make_manager()
    engine = FakeModelEngine("parakeet")
    manager.acquire("parakeet", lambda: engine)
    manager.acquire("parakeet", lambda: engine)
    assert engine.load_calls == 1


def test_acquire_different_engine_unloads_previous_first() -> None:
    manager, _ = make_manager()
    parakeet = FakeModelEngine("parakeet")
    qwen = FakeModelEngine("qwen3_tts")
    manager.acquire("parakeet", lambda: parakeet)
    manager.acquire("qwen3_tts", lambda: qwen)
    assert parakeet.unload_calls == 1
    assert qwen.load_calls == 1
    assert manager.current_engine_id == "qwen3_tts"


def test_release_starts_idle_timer_with_configured_timeout() -> None:
    manager, fake_timers = make_manager(idle_timeout=45.0)
    engine = FakeModelEngine("parakeet")
    manager.acquire("parakeet", lambda: engine)
    manager.release("parakeet")
    assert fake_timers.last_delay == 45.0
    assert len(fake_timers.handles) == 1


def test_idle_timer_firing_unloads_the_model() -> None:
    manager, fake_timers = make_manager()
    engine = FakeModelEngine("parakeet")
    manager.acquire("parakeet", lambda: engine)
    manager.release("parakeet")
    fake_timers.handles[0].fire()
    assert engine.unload_calls == 1
    assert manager.current_engine_id is None


def test_reacquire_before_idle_fires_cancels_timer_and_reuses_model() -> None:
    manager, fake_timers = make_manager()
    engine = FakeModelEngine("parakeet")
    manager.acquire("parakeet", lambda: engine)
    manager.release("parakeet")
    manager.acquire("parakeet", lambda: engine)
    assert fake_timers.handles[0].cancelled is True
    assert engine.unload_calls == 0
    assert engine.load_calls == 1


def test_release_with_no_model_loaded_logs_warning_and_starts_no_timer(
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager, fake_timers = make_manager()
    with caplog.at_level(logging.WARNING, logger="vocalis"):
        manager.release("parakeet")
    assert len(fake_timers.handles) == 0
    assert any("parakeet" in record.message for record in caplog.records)


def test_release_wrong_engine_id_logs_warning_and_starts_no_timer(
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager, fake_timers = make_manager()
    engine = FakeModelEngine("parakeet")
    manager.acquire("parakeet", lambda: engine)
    with caplog.at_level(logging.WARNING, logger="vocalis"):
        manager.release("qwen3_tts")
    assert len(fake_timers.handles) == 0
    assert engine.unload_calls == 0


def test_shutdown_unloads_immediately_and_cancels_pending_timer() -> None:
    manager, fake_timers = make_manager()
    engine = FakeModelEngine("parakeet")
    manager.acquire("parakeet", lambda: engine)
    manager.release("parakeet")
    manager.shutdown()
    assert fake_timers.handles[0].cancelled is True
    assert engine.unload_calls == 1
    assert manager.current_engine_id is None
```

- [ ] **Step 2: Run the tests, verify they fail**

Run: `.venv/bin/pytest tests/test_model_manager.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vocalis.model_manager'`

- [ ] **Step 3: Write the minimal implementation**

```python
# src/vocalis/model_manager.py
"""Enforces vocalis's GPU policy: only one AI model resident in VRAM at a
time, automatically released a few seconds after it goes idle.

Feature code calls acquire() to get a loaded engine and release() when done
with it for now. The manager decides the actual load/unload timing: it
unloads the previous engine before loading a different one, and starts an
idle timer on release() that unloads the current engine if nothing
reacquires it before the timeout elapses.
"""
import logging
import threading
from typing import Callable, Protocol

logger = logging.getLogger("vocalis")


class ModelEngine(Protocol):
    """Interface every AI model wrapper (STT/TTS/OCR) must implement."""

    def load(self) -> None:
        """Load the model into memory/VRAM. Must be safe to call once per instance."""
        ...

    def unload(self) -> None:
        """Release the model's memory/VRAM. Must be safe to call after load()."""
        ...

    @property
    def is_loaded(self) -> bool:
        """Whether the model is currently loaded."""
        ...


class TimerHandle(Protocol):
    """A cancellable handle returned by a TimerFactory."""

    def cancel(self) -> None:
        """Cancel the pending callback if it has not fired yet."""
        ...


TimerFactory = Callable[[float, Callable[[], None]], TimerHandle]


class ModelManager:
    """Owns at most one loaded ModelEngine and enforces idle-unload."""

    def __init__(
        self,
        idle_timeout_seconds: float = 30.0,
        timer_factory: TimerFactory | None = None,
    ) -> None:
        """Create a ModelManager.

        Args:
            idle_timeout_seconds: Seconds of inactivity after release()
                before the current engine is unloaded.
            timer_factory: Callable(delay_seconds, callback) -> TimerHandle,
                used to schedule the idle-unload. Defaults to a real
                threading.Timer-backed factory (see model_manager_timers).
        """
        self._idle_timeout_seconds = idle_timeout_seconds
        self._timer_factory = timer_factory or _real_timer_factory
        self._lock = threading.Lock()
        self._current_engine_id: str | None = None
        self._current_engine: ModelEngine | None = None
        self._idle_timer: TimerHandle | None = None

    @property
    def current_engine_id(self) -> str | None:
        """The engine_id currently loaded, or None if nothing is loaded."""
        with self._lock:
            return self._current_engine_id

    def acquire(self, engine_id: str, factory: Callable[[], ModelEngine]) -> ModelEngine:
        """Get a loaded engine for engine_id, loading or reusing as needed.

        Cancels any pending idle-unload. If a different engine is currently
        loaded, it is unloaded first so only one model ever sits in VRAM.

        Args:
            engine_id: Stable identifier for the requested engine, e.g.
                "parakeet" or "qwen3_tts".
            factory: Called to construct a new engine instance if one isn't
                already loaded for engine_id.

        Returns:
            A loaded ModelEngine for engine_id.
        """
        with self._lock:
            self._cancel_idle_timer_locked()

            if self._current_engine_id == engine_id and self._current_engine is not None:
                logger.info("Reusing already-loaded engine %r", engine_id)
                return self._current_engine

            if self._current_engine is not None:
                logger.info(
                    "Unloading engine %r to load %r", self._current_engine_id, engine_id
                )
                self._current_engine.unload()

            logger.info("Loading engine %r", engine_id)
            engine = factory()
            engine.load()
            self._current_engine_id = engine_id
            self._current_engine = engine
            return engine

    def release(self, engine_id: str) -> None:
        """Mark engine_id as no longer actively in use.

        Starts the idle-unload timer. If engine_id does not match the
        currently loaded engine (or nothing is loaded), logs a warning and
        does nothing else — this indicates a bug in the caller, not a state
        we should act on destructively.

        Args:
            engine_id: The engine_id previously returned by acquire().
        """
        with self._lock:
            if self._current_engine_id != engine_id:
                logger.warning(
                    "release() called for %r but currently loaded engine is %r",
                    engine_id,
                    self._current_engine_id,
                )
                return

            logger.info(
                "Engine %r released; will unload after %.1fs if not reacquired",
                engine_id,
                self._idle_timeout_seconds,
            )
            self._idle_timer = self._timer_factory(
                self._idle_timeout_seconds, self._unload_current
            )

    def shutdown(self) -> None:
        """Cancel any pending idle timer and unload the current engine now.

        Called when the daemon is stopping.
        """
        with self._lock:
            self._cancel_idle_timer_locked()
            self._unload_current_locked()

    def _unload_current(self) -> None:
        """Idle-timer callback: unload the current engine if still loaded."""
        with self._lock:
            self._unload_current_locked()

    def _unload_current_locked(self) -> None:
        """Unload the current engine. Caller must hold self._lock."""
        if self._current_engine is not None:
            logger.info("Unloading idle engine %r", self._current_engine_id)
            self._current_engine.unload()
        self._current_engine = None
        self._current_engine_id = None

    def _cancel_idle_timer_locked(self) -> None:
        """Cancel the pending idle timer, if any. Caller must hold self._lock."""
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None


def _real_timer_factory(delay_seconds: float, callback: Callable[[], None]) -> TimerHandle:
    """Default TimerFactory backed by threading.Timer (daemon thread)."""
    timer = threading.Timer(delay_seconds, callback)
    timer.daemon = True
    timer.start()
    return timer
```

- [ ] **Step 4: Run the tests, verify they pass**

Run: `.venv/bin/pytest tests/test_model_manager.py -v`
Expected: PASS (all 9 tests)

- [ ] **Step 5: Run mypy, verify it is clean**

Run: `.venv/bin/mypy`
Expected: `Success: no issues found`
(Note: `threading.Timer` satisfies the `TimerHandle` protocol structurally —
it has a `.cancel()` method — so no cast is needed.)

- [ ] **Step 6: Commit**

```bash
git add src/vocalis/model_manager.py tests/test_model_manager.py
git commit -m "feat: add ModelManager enforcing single-residency + idle-unload GPU policy" -m "- acquire()/release() ensure only one ModelEngine is ever loaded at a time" -m "- release() starts an idle timer that unloads the model if not reacquired" -m "- Fully tested with fake engines/timers; real GPU engines land in Phase 2" -m "Phase 1 of vocalis foundation plan" -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01GMitM6j81rZ9TN8k3Q5tAh"
```

---

### Task 5: Model Manager — Real Timer Integration Test

**Files:**
- Create: `tests/test_model_manager_integration.py`

**Interfaces:**
- Consumes: `ModelManager` from Task 4, using its **default** (real,
  `threading.Timer`-backed) timer factory this time instead of a fake.
- Produces: no new public API — this proves the default factory wired up in
  Task 4's `_real_timer_factory` actually unloads a model after real wall-clock
  time, which the fake-timer tests in Task 4 could not prove by themselves.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model_manager_integration.py
"""Integration test proving ModelManager's real (non-fake) timer actually
unloads an idle engine after real wall-clock time passes."""
import time

from vocalis.model_manager import ModelManager


class _RecordingEngine:
    def __init__(self) -> None:
        self.unloaded = False
        self._loaded = False

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self.unloaded = True
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded


def test_real_timer_unloads_engine_after_idle_timeout() -> None:
    manager = ModelManager(idle_timeout_seconds=0.05)
    engine = _RecordingEngine()
    manager.acquire("parakeet", lambda: engine)
    manager.release("parakeet")

    assert engine.unloaded is False

    time.sleep(0.2)

    assert engine.unloaded is True
    assert manager.current_engine_id is None
```

- [ ] **Step 2: Run the test, verify it fails**

Run: `.venv/bin/pytest tests/test_model_manager_integration.py -v`
Expected: At this point `ModelManager()` (no `timer_factory` argument) should
already work from Task 4's implementation, so this test should actually
**PASS** immediately. Run it anyway to confirm — if it fails, it means the
default `timer_factory` parameter in Task 4 is broken and must be fixed
before continuing.

- [ ] **Step 3: Confirm it passes (no implementation change expected)**

Run: `.venv/bin/pytest tests/test_model_manager_integration.py -v`
Expected: PASS

- [ ] **Step 4: Run the full test suite + coverage gate**

Run: `.venv/bin/pytest`
Expected: All tests across the project PASS and coverage is ≥85%.

- [ ] **Step 5: Commit**

```bash
git add tests/test_model_manager_integration.py
git commit -m "test: prove ModelManager's real timer unloads idle engines" -m "- Integration test uses the real threading.Timer-backed default factory (no fakes)" -m "- Confirms the idle-unload policy holds with actual wall-clock time" -m "Phase 1 of vocalis foundation plan" -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01GMitM6j81rZ9TN8k3Q5tAh"
```

---

### Task 6: Daemon Skeleton (Unix Socket Server + Graceful Shutdown)

**Files:**
- Create: `src/vocalis/daemon.py`
- Create: `tests/test_daemon.py`

**Interfaces:**
- Consumes: `ModelManager` (Task 4) — the daemon owns one instance and calls
  `.shutdown()` on it when stopping.
- Produces:
  - `DaemonAlreadyRunningError(Exception)`
  - `VocalisDaemon.__init__(self, socket_path: Path, logger: logging.Logger,
    model_manager: ModelManager) -> None`
  - `VocalisDaemon.start(self) -> None`
  - `VocalisDaemon.stop(self) -> None`
  - `VocalisDaemon.is_running -> bool` (property)
  - Later feature plans (Phase 2+) will extend the accept-loop's command
    handling; for now it only echoes `b"ack:<line>\n"` back to the caller so
    the socket lifecycle itself is provable in isolation.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_daemon.py
"""Tests for VocalisDaemon's Unix socket lifecycle (start/stop/echo)."""
import socket
from pathlib import Path

import pytest

from vocalis.daemon import DaemonAlreadyRunningError, VocalisDaemon
from vocalis.logging_setup import configure_logging
from vocalis.model_manager import ModelManager


def make_daemon(tmp_path: Path) -> VocalisDaemon:
    logger = configure_logging()
    return VocalisDaemon(
        socket_path=tmp_path / "vocalis.sock",
        logger=logger,
        model_manager=ModelManager(),
    )


def test_start_creates_socket_file(tmp_path: Path) -> None:
    daemon = make_daemon(tmp_path)
    daemon.start()
    try:
        assert (tmp_path / "vocalis.sock").exists()
        assert daemon.is_running is True
    finally:
        daemon.stop()


def test_stop_removes_socket_file(tmp_path: Path) -> None:
    daemon = make_daemon(tmp_path)
    daemon.start()
    daemon.stop()
    assert not (tmp_path / "vocalis.sock").exists()
    assert daemon.is_running is False


def test_echo_roundtrip(tmp_path: Path) -> None:
    daemon = make_daemon(tmp_path)
    daemon.start()
    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(str(tmp_path / "vocalis.sock"))
        client.sendall(b"ping\n")
        response = client.recv(1024)
        client.close()
        assert response == b"ack:ping\n"
    finally:
        daemon.stop()


def test_start_removes_stale_socket_file(tmp_path: Path) -> None:
    socket_path = tmp_path / "vocalis.sock"
    socket_path.write_text("stale file from a previous crashed run")
    daemon = VocalisDaemon(
        socket_path=socket_path,
        logger=configure_logging(),
        model_manager=ModelManager(),
    )
    daemon.start()
    try:
        assert daemon.is_running is True
    finally:
        daemon.stop()


def test_double_start_raises_daemon_already_running_error(tmp_path: Path) -> None:
    daemon = make_daemon(tmp_path)
    daemon.start()
    try:
        with pytest.raises(DaemonAlreadyRunningError):
            daemon.start()
    finally:
        daemon.stop()


def test_stop_calls_model_manager_shutdown(tmp_path: Path) -> None:
    manager = ModelManager()
    shutdown_calls = []
    manager.shutdown = lambda: shutdown_calls.append(True)  # type: ignore[method-assign]
    daemon = VocalisDaemon(
        socket_path=tmp_path / "vocalis.sock", logger=configure_logging(), model_manager=manager
    )
    daemon.start()
    daemon.stop()
    assert shutdown_calls == [True]
```

- [ ] **Step 2: Run the tests, verify they fail**

Run: `.venv/bin/pytest tests/test_daemon.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vocalis.daemon'`

- [ ] **Step 3: Write the minimal implementation**

```python
# src/vocalis/daemon.py
"""The vocalis background daemon: a long-lived process listening on a Unix
domain socket for commands from the `vocalis` CLI and hotkey layer.

This module only implements the socket lifecycle (start/stop/accept-loop).
Command handling for real features (dictation, quick reader, document
reader, OCR) is added in their own phases on top of this skeleton.
"""
import logging
import socket
import threading
from pathlib import Path

from vocalis.model_manager import ModelManager


class DaemonAlreadyRunningError(Exception):
    """Raised when start() is called on a daemon that is already running."""


class VocalisDaemon:
    """Owns the Unix socket server and the ModelManager for one vocalis run."""

    def __init__(
        self,
        socket_path: Path,
        logger: logging.Logger,
        model_manager: ModelManager,
    ) -> None:
        """Create a VocalisDaemon.

        Args:
            socket_path: Filesystem path for the Unix domain socket.
            logger: Logger to use for lifecycle and per-connection events.
            model_manager: The ModelManager to shut down when the daemon stops.
        """
        self._socket_path = socket_path
        self._logger = logger
        self._model_manager = model_manager
        self._server_socket: socket.socket | None = None
        self._accept_thread: threading.Thread | None = None
        self._stop_requested = threading.Event()

    @property
    def is_running(self) -> bool:
        """Whether the daemon's accept loop is currently active."""
        return self._server_socket is not None

    def start(self) -> None:
        """Start listening on the Unix socket and accepting connections.

        Removes a stale socket file left over from a previous crashed run
        before binding.

        Raises:
            DaemonAlreadyRunningError: If start() is called while already running.
        """
        if self.is_running:
            raise DaemonAlreadyRunningError(
                f"Daemon already running on {self._socket_path}"
            )

        if self._socket_path.exists():
            self._logger.warning(
                "Removing stale socket file at %s from a previous run", self._socket_path
            )
            self._socket_path.unlink()

        self._socket_path.parent.mkdir(parents=True, exist_ok=True)
        server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server_socket.bind(str(self._socket_path))
        server_socket.listen(5)
        server_socket.settimeout(0.2)
        self._server_socket = server_socket
        self._stop_requested.clear()

        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()
        self._logger.info("Vocalis daemon listening on %s", self._socket_path)

    def stop(self) -> None:
        """Stop accepting connections, unload models, and clean up the socket file."""
        self._stop_requested.set()

        if self._accept_thread is not None:
            self._accept_thread.join(timeout=2.0)
            self._accept_thread = None

        if self._server_socket is not None:
            self._server_socket.close()
            self._server_socket = None

        if self._socket_path.exists():
            self._socket_path.unlink()

        self._model_manager.shutdown()
        self._logger.info("Vocalis daemon stopped")

    def _accept_loop(self) -> None:
        """Accept connections until stop() is called, echoing each command."""
        assert self._server_socket is not None
        while not self._stop_requested.is_set():
            try:
                connection, _ = self._server_socket.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            try:
                self._handle_connection(connection)
            except OSError as exc:
                self._logger.warning("Error handling connection: %s", exc)
            finally:
                connection.close()

    def _handle_connection(self, connection: socket.socket) -> None:
        """Read one line and echo it back as an acknowledgement.

        Real command dispatch (dictation/quick-reader/document-reader/ocr)
        is added by later phases; this skeleton only proves the socket
        round-trip works.
        """
        data = connection.recv(1024)
        if not data:
            return
        command = data.decode("utf-8").strip()
        self._logger.info("Received command: %s", command)
        connection.sendall(f"ack:{command}\n".encode("utf-8"))
```

- [ ] **Step 4: Run the tests, verify they pass**

Run: `.venv/bin/pytest tests/test_daemon.py -v`
Expected: PASS (all 6 tests)

- [ ] **Step 5: Run the full suite + mypy**

Run: `.venv/bin/pytest && .venv/bin/mypy`
Expected: All tests PASS, coverage ≥85%, mypy clean.

- [ ] **Step 6: Commit**

```bash
git add src/vocalis/daemon.py tests/test_daemon.py
git commit -m "feat: add VocalisDaemon Unix socket skeleton with graceful shutdown" -m "- start()/stop() manage a Unix domain socket accept-loop on a background thread" -m "- Removes stale socket files from crashed previous runs before binding" -m "- stop() always calls ModelManager.shutdown() so VRAM is released on exit" -m "Phase 1 of vocalis foundation plan" -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01GMitM6j81rZ9TN8k3Q5tAh"
```

---

### Task 7: CLI Entrypoint Wiring (Config → Logging → Daemon)

**Files:**
- Create: `src/vocalis/cli.py`
- Create: `tests/test_cli.py`

**Interfaces:**
- Consumes: `load_config`/`default_config_path` (Task 3),
  `configure_logging` (Task 2), `ModelManager` (Task 4), `VocalisDaemon`
  (Task 6).
- Produces: `main() -> None` — the `vocalis` console-script entrypoint
  declared in `pyproject.toml` (Task 1). Later phases will extend this to
  register hotkeys and dispatch feature commands; for now it just proves
  the wiring: load config, configure logging, build a ModelManager sized
  from config, build and start a VocalisDaemon, block until interrupted,
  then stop cleanly.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cli.py
"""Tests for vocalis.cli wiring: config -> logging -> ModelManager -> daemon."""
from pathlib import Path
from unittest.mock import patch

from vocalis.cli import build_daemon
from vocalis.config import VocalisConfig


def test_build_daemon_uses_configured_idle_timeout(tmp_path: Path) -> None:
    config = VocalisConfig(model_idle_timeout_seconds=12.5)
    socket_path = tmp_path / "vocalis.sock"

    daemon = build_daemon(config=config, socket_path=socket_path)

    assert daemon._model_manager._idle_timeout_seconds == 12.5  # noqa: SLF001
    assert daemon._socket_path == socket_path  # noqa: SLF001


def test_main_starts_and_stops_daemon_on_keyboard_interrupt(tmp_path: Path) -> None:
    socket_path = tmp_path / "vocalis.sock"

    with (
        patch("vocalis.cli.default_config_path", return_value=tmp_path / "config.toml"),
        patch("vocalis.cli._socket_path", return_value=socket_path),
        patch("vocalis.cli._block_until_interrupted", side_effect=KeyboardInterrupt),
    ):
        from vocalis.cli import main

        main()

    assert not socket_path.exists()
```

- [ ] **Step 2: Run the tests, verify they fail**

Run: `.venv/bin/pytest tests/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vocalis.cli'`

- [ ] **Step 3: Write the minimal implementation**

```python
# src/vocalis/cli.py
"""The `vocalis` command-line entrypoint.

Wires together config loading, logging, the ModelManager, and the
VocalisDaemon, then blocks until interrupted (Ctrl+C, or SIGTERM once this
runs under systemd in Phase 7).
"""
import time
from pathlib import Path

from vocalis.config import VocalisConfig, default_config_path, load_config
from vocalis.daemon import VocalisDaemon
from vocalis.logging_setup import configure_logging
from vocalis.model_manager import ModelManager


def _socket_path() -> Path:
    """Return the standard runtime location for the daemon's Unix socket."""
    return Path.home() / ".local" / "state" / "vocalis" / "vocalis.sock"


def build_daemon(config: VocalisConfig, socket_path: Path) -> VocalisDaemon:
    """Construct a VocalisDaemon wired from the given config.

    Args:
        config: Loaded, validated vocalis configuration.
        socket_path: Where the daemon's Unix socket should live.

    Returns:
        A VocalisDaemon ready to have .start() called on it.
    """
    logger = configure_logging(level="INFO")
    model_manager = ModelManager(idle_timeout_seconds=config.model_idle_timeout_seconds)
    return VocalisDaemon(socket_path=socket_path, logger=logger, model_manager=model_manager)


def _block_until_interrupted() -> None:
    """Sleep forever until interrupted by Ctrl+C or SIGTERM."""
    while True:
        time.sleep(1.0)


def main() -> None:
    """Entrypoint for the `vocalis` console script.

    Loads config, starts the daemon, and blocks until the process is
    interrupted, at which point it stops the daemon cleanly.
    """
    config = load_config(default_config_path())
    daemon = build_daemon(config=config, socket_path=_socket_path())
    daemon.start()
    try:
        _block_until_interrupted()
    except KeyboardInterrupt:
        pass
    finally:
        daemon.stop()
```

- [ ] **Step 4: Run the tests, verify they pass**

Run: `.venv/bin/pytest tests/test_cli.py -v`
Expected: PASS (both tests)

- [ ] **Step 5: Run the full suite + mypy**

Run: `.venv/bin/pytest && .venv/bin/mypy`
Expected: All tests PASS, coverage ≥85%, mypy clean.

- [ ] **Step 6: Manually smoke-test the real entrypoint**

Run in the background (per governance: long-running processes are run in
the background so they can be observed and managed):

```bash
.venv/bin/vocalis &
sleep 1
ls -la ~/.local/state/vocalis/vocalis.sock
kill -INT %1
sleep 1
ls ~/.local/state/vocalis/vocalis.sock 2>&1 || echo "socket cleaned up correctly"
```

Expected: socket file appears after start, and is gone after the interrupt.

- [ ] **Step 7: Commit**

```bash
git add src/vocalis/cli.py tests/test_cli.py
git commit -m "feat: add vocalis CLI entrypoint wiring config, logging, and the daemon" -m "- build_daemon() constructs a ModelManager sized from config.model_idle_timeout_seconds" -m "- main() loads config, starts the daemon, blocks until interrupted, stops cleanly" -m "Phase 1 of vocalis foundation plan" -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01GMitM6j81rZ9TN8k3Q5tAh"
```

---

### Task 8: Governance Docs Stubs + Changelog

**Files:**
- Create: `docs/architecture.md`
- Create: `docs/folder-structure.md`
- Create: `docs/file-purpose.md`
- Create: `docs/workflow.md`
- Create: `docs/algorithms.md`
- Create: `docs/user-docs.md`
- Create: `docs/developer-docs.md`
- Create: `.claude/2026-09-11_1200_CHANGELOG.md`

**Interfaces:**
- Consumes: nothing (documentation only).
- Produces: the governance-required `docs/` structure, describing exactly
  what exists after Tasks 1–7 (not the full end-state app) — later phases
  append their own sections as features land.

- [ ] **Step 1: Write `docs/architecture.md`**

```markdown
# Vocalis Architecture

## Current state (Phase 0+1 foundation)

Vocalis is a single long-lived daemon process (`VocalisDaemon`) that:

1. Loads validated configuration (`VocalisConfig`) from
   `~/.config/vocalis/config.toml`, falling back to defaults on first run.
2. Owns a `ModelManager`, which enforces the core GPU safety rule: **only
   one AI model may be resident in VRAM at a time**, and it is
   automatically unloaded a few seconds (configurable,
   `model_idle_timeout_seconds`) after the last feature releases it.
3. Listens on a Unix domain socket (`~/.local/state/vocalis/vocalis.sock`)
   for commands. Today it only echoes commands back — real feature
   dispatch (dictation, quick reader, document reader, OCR) is added in
   later phases on top of this same socket.

No AI engines (Parakeet, Qwen3-TTS, fish-speech, Tesseract) are wired in
yet. The `ModelManager` is proven with fake engines in tests; Phase 2
plugs in the first real engine (Parakeet) and verifies actual VRAM
behavior with `nvidia-smi`.

## Design rationale

- **Single daemon, socket-based commands**: hotkeys and the CLI are thin
  clients that send a command over the socket rather than each spinning up
  their own model — this is what makes the single-model-residency GPU
  policy possible.
- **ModelManager as its own module**: the GPU idle-unload policy is the
  riskiest and most safety-critical piece of this app (an 8GB card can't
  hold multiple large models), so it is built and tested in isolation
  before any real feature touches it.
- **pydantic config**: invalid config is rejected loudly at load time with
  a clear `ConfigError`, rather than causing a confusing failure deep
  inside a feature later.

See `docs/superpowers/specs/2026-09-11-vocalis-design.md` for the full
product design (all four features, UI, packaging) this architecture is
building toward.
```

- [ ] **Step 2: Write `docs/folder-structure.md`**

```markdown
# Vocalis Folder Structure

```
src/vocalis/
├── __init__.py         # package version
├── py.typed             # marks the package as type-checked (PEP 561)
├── config.py           # VocalisConfig pydantic model + TOML load/save
├── logging_setup.py     # shared "vocalis" logger configuration
├── model_manager.py     # GPU single-residency + idle-unload policy
├── daemon.py            # Unix socket server lifecycle (start/stop)
└── cli.py               # `vocalis` console-script entrypoint

tests/                    # pytest suite mirroring src/vocalis/ (≥85% coverage)
config/
└── default_config.toml   # first-run config template

docs/
├── superpowers/
│   ├── specs/           # brainstorming design docs
│   └── plans/            # writing-plans implementation plans (this file's home)
├── architecture.md
├── folder-structure.md   # this file
├── file-purpose.md
├── workflow.md
├── algorithms.md
├── user-docs.md
└── developer-docs.md

old/                       # untouched legacy Voxtype scripts/docs (reference only)
```

Feature code (`hotkeys/`, `features/`, `engines/`, `text/`, `audio/`,
`output/`, `notify.py`, `ui/`) and `packaging/` are added by later phases
per the spec's project structure.
```

- [ ] **Step 3: Write `docs/file-purpose.md`**

```markdown
# Purpose of Each File (Phase 0+1)

| File | Purpose |
|---|---|
| `src/vocalis/__init__.py` | Exposes `__version__` for the package. |
| `src/vocalis/config.py` | `VocalisConfig`/`ShortcutsConfig`/`EnginesConfig` pydantic models; `load_config`/`save_config`/`default_config_path`; raises `ConfigError` on invalid input. |
| `src/vocalis/logging_setup.py` | `configure_logging()` — sets up the shared `"vocalis"` logger (console + optional file handler), idempotent. |
| `src/vocalis/model_manager.py` | `ModelManager` — enforces single-model VRAM residency and idle-unload; defines the `ModelEngine`/`TimerHandle` protocols later engine wrappers implement. |
| `src/vocalis/daemon.py` | `VocalisDaemon` — Unix socket server lifecycle: start/stop, stale-socket cleanup, graceful shutdown (calls `ModelManager.shutdown()`). |
| `src/vocalis/cli.py` | `vocalis` console-script entrypoint: loads config, builds the daemon, blocks until interrupted. |
| `config/default_config.toml` | First-run configuration template. |
| `pyproject.toml` | Package metadata, dependencies, mypy/pytest/coverage configuration. |
```

- [ ] **Step 4: Write `docs/workflow.md`**

```markdown
# Workflow (Phase 0+1)

## User perspective

At this stage there is no user-facing feature yet — running `vocalis`
starts a background process that listens for commands and does nothing
else. This is the plumbing later phases attach real features to.

## Internal (daemon) perspective

```
main()
  -> load_config(default_config_path())     # ~/.config/vocalis/config.toml, or defaults
  -> build_daemon(config, socket_path)
       -> configure_logging()
       -> ModelManager(idle_timeout_seconds=config.model_idle_timeout_seconds)
       -> VocalisDaemon(socket_path, logger, model_manager)
  -> daemon.start()                          # binds Unix socket, spawns accept-loop thread
  -> blocks until Ctrl+C / SIGTERM
  -> daemon.stop()                           # closes socket, removes socket file,
                                              #   calls model_manager.shutdown()
```

## Model acquire/release lifecycle (used by every future feature)

```
feature code:
  engine = model_manager.acquire("parakeet", lambda: ParakeetSTT(...))
  ... use engine ...
  model_manager.release("parakeet")

ModelManager, internally:
  acquire(id, factory):
    cancel any pending idle-unload timer
    if id already loaded -> reuse it
    else -> unload whatever else is loaded, then factory().load()
  release(id):
    start idle timer (config.model_idle_timeout_seconds)
    if not reacquired before it fires -> unload
```
```

- [ ] **Step 5: Write `docs/algorithms.md`**

```markdown
# Algorithms Used

## Single-residency + idle-unload (`ModelManager`)

**Problem:** an 8GB GPU cannot hold multiple large speech models at once,
and models must not sit in VRAM when nothing is using them.

**Approach:** a simple reference-free "last write wins, idle timeout evicts"
policy:

- `acquire(id, factory)` — O(1). Cancels any pending eviction timer. If the
  requested id is already the loaded one, reuse it (no reload cost). If a
  *different* engine is loaded, unload it synchronously before loading the
  new one, guaranteeing at most one engine loaded at any instant.
- `release(id)` — O(1). Starts a timer (`threading.Timer` by default, wall
  clock `model_idle_timeout_seconds`, default 30s) that unloads the current
  engine when it fires. Calling `acquire()` again before it fires cancels
  the timer, so a burst of quick requests for the same engine only pays the
  load cost once.
- Thread safety: a single `threading.Lock` guards all state transitions, so
  hotkey threads and the idle-timer thread can never race each other into
  an inconsistent state (e.g. unloading an engine that was just reacquired).

**Complexity:** O(1) per acquire/release; one background timer thread at a
time (daemon thread, does not block process exit).

**Why not reference counting?** Vocalis's features are one-at-a-time,
short-lived operations (a dictation utterance, one read-aloud request) —
concurrent use of two different engines simultaneously is out of scope
(YAGNI), so "last acquired engine wins, evict after idle" is simpler and
sufficient.
```

- [ ] **Step 6: Write `docs/user-docs.md`**

```markdown
# User Documentation (Phase 0+1)

There are no user-facing features yet — this phase only builds the
background service skeleton. Once Phase 2 (Dictation) lands, this document
will explain how to hold NUMLOCK to dictate, and so on for each feature as
it is built.

## Running the daemon manually (development only)

```bash
.venv/bin/vocalis
```

Press Ctrl+C to stop it. Once Phase 7 (Service & autostart) lands, this
will instead run automatically via `systemctl --user start vocalis`.
```

- [ ] **Step 7: Write `docs/developer-docs.md`**

```markdown
# Developer Documentation

## Setup

```bash
python3.13 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Running tests

```bash
.venv/bin/pytest            # runs full suite, fails if coverage < 85%
```

## Type checking

```bash
.venv/bin/mypy               # strict mode, must report "no issues found"
```

## Project conventions

- Every function has a docstring explaining what it does.
- Config is validated at the boundary (`vocalis.config`) — never trust raw
  TOML data beyond that module.
- All logging goes through the `"vocalis"` logger
  (`vocalis.logging_setup.configure_logging()`); never log full user text
  content or secrets.
- Commits use conventional-commit prefixes (`feat:`, `fix:`, `test:`,
  `docs:`, `chore:`) and are made only after the relevant test suite passes.
- `old/` is a read-only reference to the legacy bash-script implementation
  and must never be modified.
```

- [ ] **Step 8: Write the changelog entry**

```markdown
<!-- .claude/2026-09-11_1200_CHANGELOG.md -->
# Changelog — 2026-09-11

## Vocalis Phase 0+1: Foundation

Built the initial project skeleton for Vocalis, the new unified voice AI
app replacing the old Voxtype binary + bash scripts (moved to `old/`).

**Added:**
- Installable `vocalis` Python package (3.13) with strict `mypy`, `pytest`
  + coverage (≥85% gate) configured in `pyproject.toml`.
- `vocalis.config` — pydantic-validated TOML config
  (`~/.config/vocalis/config.toml`), covering shortcuts, engine choices,
  notification/widget toggles, and the GPU idle-timeout policy.
- `vocalis.logging_setup` — shared `"vocalis"` logger.
- `vocalis.model_manager.ModelManager` — enforces the core safety rule:
  only one AI model resident in VRAM at a time, auto-released after an
  idle timeout. Fully unit- and integration-tested with fake and real
  timers.
- `vocalis.daemon.VocalisDaemon` — Unix socket server skeleton with
  graceful start/stop, stale-socket cleanup, and GPU shutdown on stop.
- `vocalis.cli.main` — the `vocalis` console-script entrypoint wiring all
  of the above together.
- Governance-required `docs/` (architecture, folder structure, file
  purpose, workflow, algorithms, user docs, developer docs).

**Not yet implemented (future phases):** no real STT/TTS/OCR engines, no
hotkeys, no UI (tray/widget/settings), no systemd service, no packaging.

**Rollback:** this phase is additive only — nothing in `old/` or the
existing system was touched. Reverting is `git revert` back to before this
changelog's commits.
```

- [ ] **Step 9: Commit**

```bash
git add docs/architecture.md docs/folder-structure.md docs/file-purpose.md docs/workflow.md docs/algorithms.md docs/user-docs.md docs/developer-docs.md .claude/2026-09-11_1200_CHANGELOG.md
git commit -m "docs: add governance-required architecture/workflow/algorithm docs for phase 0+1" -m "- Documents the daemon + ModelManager foundation exactly as built so far" -m "- Adds changelog entry summarizing phase 0+1" -m "Phase 0+1 of vocalis foundation plan" -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01GMitM6j81rZ9TN8k3Q5tAh"
```

---

## After This Plan

With the foundation complete and committed, the next implementation plan
(written just before it starts, per the spec's milestone list) is **Phase
2 — Dictation (STT)**: wrap the existing Parakeet ONNX model behind the
`ModelEngine` protocol, add NUMLOCK push-to-talk via `python-evdev`, wire
typing-at-cursor via `ydotool` with clipboard fallback, add start/live/done
notifications, and validate side-by-side against the current Voxtype
binary before it is retired.
