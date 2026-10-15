# Vocalis Stage 2 — Phase 7 Service and Autostart Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Vocalis can run as a `systemd --user` service that starts at desktop login (after the sound system is up), restarts itself after a crash with a burst limit, warns about missing dependencies at startup via notification, shuts down cleanly on SIGTERM exactly like SIGINT, and can never have two real instances fighting over the same socket.

**Architecture:** A new `startup_checks.py` module with four independent, non-fatal checks, each returning `None` or a human message, called once from `main()` and reported through the existing `Notifier`. `VocalisDaemon.start()` gains a real single-instance guard (connect-before-delete instead of unconditional delete). `main()` gains a `SIGTERM` handler that reuses the exact same `app.quit()` path as the existing `SIGINT` handler. A unit file template in `packaging/systemd/` describes the service; it is not installed or enabled by this stage (that is Stage 3's job).

**Tech Stack:** Python 3.13, `systemd --user`, the existing `evdev`/`sounddevice`/`shutil.which` seams, pytest + pytest-qt, mypy strict.

**Spec:** `docs/superpowers/specs/2026-09-11-vocalis-design.md` — section "Stage 2 — Phase 7 amendment: Service and autostart".

## Global Constraints

- Work in a fresh git worktree on branch `worktree-vocalis-stage2-service`; run everything with `export PYTHONPATH=$PWD/src` and `.venv/bin/pytest` / `.venv/bin/mypy` (link `.venv` to the main checkout's `.venv`; never `git add` it).
- Conventional commits, **no Co-Authored-By or attribution lines**, explicit `git add` paths, never touch `old/`.
- Every function, method and class has a docstring. `mypy` strict: **zero errors project-wide**. Full suite **zero warnings**, coverage **>= 85%** (baseline at plan time: 563 tests, 96.50%). Run full-suite runs strictly one after another, never overlapping (it corrupts `.coverage`).
- New constructor/function parameters on existing code are **optional keyword parameters with defaults**; existing tests stay green unchanged; never weaken an existing assertion. Tests deterministic (no guessed sleeps, no real sockets/processes/devices — fakes only).
- **Never start a real systemd unit, never actually enable/install anything, never send a real SIGTERM to a real running Vocalis, never run the real `vocalis` program.** The real-machine check in Task 6 is validation-only (`systemd-analyze --user verify` on the unit file's TEXT, run by the controller, not an implementer) — it does not start the service.
- The user installs and tests everything only at the very end, once Stage 3 (the installer) exists. Nothing in this stage changes what happens when a user runs `vocalis` manually today, except that it now also handles SIGTERM and runs the startup checks.
- Record the change log in `.claude/<YYYY-MM-DD_HHMM>_CHANGELOG.md`; keep `docs/` in sync.

## Review Focus

1. The single-instance fix must not reject the FIRST-EVER start (no existing socket file) and must not falsely call a genuinely-dead socket "alive" (Tasks 1).
2. A startup check whose own detection code raises (e.g. `shutil.which` or a socket connect raising something unexpected) must never crash `main()` — caught, logged, treated as "could not verify", not reported as a hard failure (Task 2).
3. SIGTERM delivered twice in a row (systemd's `stop` then `kill -TERM` again before exit finishes) must not double-run the shutdown teardown or crash (Task 3).
4. The unit file's `Restart=`/`StartLimitBurst=` values match exactly what the design says (3s, 5 times in 60s) — a text-content test, not a running-service test (Task 4).
5. The "gave up after N crashes" notification must fire on an actual systemd restart-limit hit, not on every ordinary restart — this can only be unit-tested at the text/documentation level in this stage (systemd itself decides when to stop restarting); the plan must say plainly that the notification firing in practice is NOT verified by automated tests in this stage and is a Stage 3/real-install concern (Task 5, documented not implemented as runtime code — see Task 5's note).

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/vocalis/startup_checks.py` | create | four non-fatal dependency checks |
| `src/vocalis/daemon.py` | modify | real single-instance guard |
| `src/vocalis/cli.py` | modify | SIGTERM handler, run+report startup checks |
| `packaging/systemd/vocalis.service` | create | the unit file template (placeholders, not real paths) |
| `packaging/systemd/README.md` | create | what Stage 3 must substitute and how to verify the file |
| tests + docs + change log | create/modify | matching tests and documentation |

---

### Task 1: Real single-instance guard in VocalisDaemon

**Files:**
- Modify: `src/vocalis/daemon.py`
- Test: `tests/test_daemon.py` (append; read it first to match its existing fakes/style)

**Interfaces:**
- Consumes: nothing new.
- Produces: `VocalisDaemon.__init__(..., socket_factory: Callable[[], socket.socket] = lambda: socket.socket(socket.AF_UNIX, socket.SOCK_STREAM))` — a new LAST optional keyword parameter, injectable for tests; `start()`'s behaviour changes as below but its signature and `DaemonAlreadyRunningError` contract are unchanged.

Behaviour: when `self._socket_path.exists()` is True, before deleting it, try to CONNECT to it with a short-lived probe socket (`probe = self._socket_factory(); probe.settimeout(0.5); probe.connect(str(self._socket_path))`, always `probe.close()` in a `finally`). A successful connect means a real instance is alive: raise `DaemonAlreadyRunningError(f"Another vocalis instance is already running on {self._socket_path}")` and do NOT delete the file or bind. `ConnectionRefusedError` or `FileNotFoundError` on connect means the file is genuinely stale: log the existing warning and delete it, exactly as today. Any OTHER exception from the probe (e.g. `PermissionError`, `OSError` for an unexpected reason) is logged at WARNING ("Could not verify whether an existing vocalis socket is alive; treating it as stale") and treated the same as stale (delete and proceed) — never block startup on an ambiguous probe result.

- [ ] **Step 1: Write the failing tests** — read `tests/test_daemon.py` first for its existing `tmp_path`/fixture style, then append:

```python
import socket as socket_module

import pytest

from vocalis.daemon import DaemonAlreadyRunningError, VocalisDaemon


class _RefusingSocket:
    """A probe socket whose connect() always refuses, like a dead socket file."""

    def connect(self, _path: str) -> None:
        raise ConnectionRefusedError("no one listening")

    def settimeout(self, _seconds: float) -> None:
        pass

    def close(self) -> None:
        pass


class _AcceptingSocket:
    """A probe socket whose connect() succeeds, like a real live instance."""

    def connect(self, _path: str) -> None:
        return None

    def settimeout(self, _seconds: float) -> None:
        pass

    def close(self) -> None:
        pass


class _OddlyFailingSocket:
    """A probe socket whose connect() raises something unexpected."""

    def connect(self, _path: str) -> None:
        raise PermissionError("unexpected")

    def settimeout(self, _seconds: float) -> None:
        pass

    def close(self) -> None:
        pass


def test_start_with_no_existing_socket_file_works_as_before(tmp_path, model_manager) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "vocalis.sock"
    daemon = VocalisDaemon(socket_path=path, logger=logging.getLogger("vocalis"), model_manager=model_manager)
    daemon.start()
    assert daemon.is_running
    daemon.stop()


def test_a_genuinely_dead_socket_file_is_removed_and_start_succeeds(tmp_path, model_manager) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "vocalis.sock"
    path.write_bytes(b"")  # a leftover file, nothing listening
    daemon = VocalisDaemon(
        socket_path=path, logger=logging.getLogger("vocalis"), model_manager=model_manager,
        socket_factory=lambda: _RefusingSocket(),  # type: ignore[arg-type]
    )
    daemon.start()
    assert daemon.is_running
    daemon.stop()


def test_a_genuinely_alive_socket_refuses_to_start(tmp_path, model_manager) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "vocalis.sock"
    path.write_bytes(b"")
    daemon = VocalisDaemon(
        socket_path=path, logger=logging.getLogger("vocalis"), model_manager=model_manager,
        socket_factory=lambda: _AcceptingSocket(),  # type: ignore[arg-type]
    )
    with pytest.raises(DaemonAlreadyRunningError, match=str(path)):
        daemon.start()
    assert path.exists()  # the live instance's file must survive
    assert not daemon.is_running


def test_an_ambiguous_probe_failure_is_treated_as_stale_not_blocking(tmp_path, model_manager, caplog) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "vocalis.sock"
    path.write_bytes(b"")
    daemon = VocalisDaemon(
        socket_path=path, logger=logging.getLogger("vocalis"), model_manager=model_manager,
        socket_factory=lambda: _OddlyFailingSocket(),  # type: ignore[arg-type]
    )
    daemon.start()
    assert daemon.is_running
    assert "Could not verify" in caplog.text
    daemon.stop()
```

(Add `import logging` to the top of the file if not already present; reuse the file's existing `model_manager` fixture if one exists — if the existing tests construct a `ModelManager()` inline instead of a fixture, do the same inline in each new test rather than inventing a fixture.)

- [ ] **Step 2: Run, verify FAIL** — `export PYTHONPATH=$PWD/src; .venv/bin/pytest tests/test_daemon.py -q --no-cov` → the "genuinely alive" test fails because today's code deletes the file unconditionally and never raises.

- [ ] **Step 3: Implement** — in `src/vocalis/daemon.py`:

Add `import contextlib` and a type import for `Callable` if missing. Change `__init__` to accept and store the new parameter:

```python
    def __init__(
        self,
        socket_path: Path,
        logger: logging.Logger,
        model_manager: ModelManager,
        socket_factory: "Callable[[], socket.socket]" = lambda: socket.socket(
            socket.AF_UNIX, socket.SOCK_STREAM
        ),
    ) -> None:
        """Create a VocalisDaemon.

        Args:
            socket_path: Filesystem path for the Unix domain socket.
            logger: Logger to use for lifecycle and per-connection events.
            model_manager: The ModelManager to shut down when the daemon stops.
            socket_factory: Constructs a fresh socket, used only to probe
                whether an existing socket file is still alive before
                deleting it as stale. Injectable for tests.
        """
        self._socket_path = socket_path
        self._logger = logger
        self._model_manager = model_manager
        self._socket_factory = socket_factory
        self._server_socket: socket.socket | None = None
        self._accept_thread: threading.Thread | None = None
        self._stop_requested = threading.Event()
```

Replace the existing "stale socket" block in `start()`:

```python
        if self._socket_path.exists():
            if self._is_another_instance_alive():
                raise DaemonAlreadyRunningError(
                    f"Another vocalis instance is already running on {self._socket_path}"
                )
            self._logger.warning(
                "Removing stale socket file at %s from a previous run", self._socket_path
            )
            self._socket_path.unlink()
```

Add the new private method (placed above `start()` or just below it):

```python
    def _is_another_instance_alive(self) -> bool:
        """Whether a real process is still listening on the existing socket file.

        Connects a short-lived probe socket to the existing file. A
        successful connect means another instance is genuinely alive.
        ConnectionRefusedError or FileNotFoundError means the file is a
        stale leftover from a crashed run. Any other error is ambiguous
        and is treated as "not alive" so startup is never blocked by an
        inconclusive probe.
        """
        probe = self._socket_factory()
        try:
            probe.settimeout(0.5)
            probe.connect(str(self._socket_path))
            return True
        except (ConnectionRefusedError, FileNotFoundError):
            return False
        except Exception as exc:
            self._logger.warning(
                "Could not verify whether an existing vocalis socket is alive "
                "(%s); treating it as stale",
                exc,
            )
            return False
        finally:
            with contextlib.suppress(Exception):
                probe.close()
```

- [ ] **Step 4: Run, verify PASS** — `.venv/bin/pytest tests/test_daemon.py -q --no-cov && .venv/bin/mypy` → all pass, mypy `Success`.

- [ ] **Step 5: Commit** — `git add src/vocalis/daemon.py tests/test_daemon.py && git commit -m "fix: tell a stale socket file apart from a genuinely running instance"`

---

### Task 2: Startup health checks

**Files:**
- Create: `src/vocalis/startup_checks.py`
- Test: `tests/test_startup_checks.py`

**Interfaces — Produces:**
- `check_parakeet_model(model_dir: Path) -> str | None`
- `check_qwen3_tts_model(model_dir: Path) -> str | None`
- `check_ydotoold(socket_path: Path) -> str | None`
- `check_capture_tools(which: Callable[[str], str | None] = shutil.which) -> str | None`
- `run_startup_checks(config: VocalisConfig) -> list[str]` — calls all four (deriving `model_dir`/`socket_path` from `config` and `os.environ`), catching any unexpected exception from an individual check (logs it, treats that check as passing/silent rather than crashing the whole function), and returns the non-`None` messages in a fixed order (Parakeet, Qwen3-TTS, ydotoold, capture tools).

Each check returns `None` when everything looks fine, or ONE short, user-facing sentence naming what's missing and, where it helps, what to do about it — never raises for a normal "missing" case.

- [ ] **Step 1: Write the failing tests** — create `tests/test_startup_checks.py`:

```python
"""Tests for vocalis.startup_checks — each check is independent and non-fatal."""
import logging
from pathlib import Path

from vocalis.config import VocalisConfig
from vocalis.startup_checks import (
    check_capture_tools,
    check_parakeet_model,
    check_qwen3_tts_model,
    check_ydotoold,
    run_startup_checks,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def test_parakeet_model_present_is_silent(tmp_path: Path) -> None:
    model_dir = tmp_path / "parakeet"
    for name in ("config.json", "encoder-model.int8.onnx", "decoder_joint-model.int8.onnx", "vocab.txt"):
        _touch(model_dir / name)
    assert check_parakeet_model(model_dir) is None


def test_parakeet_model_missing_names_the_folder(tmp_path: Path) -> None:
    model_dir = tmp_path / "does-not-exist"
    message = check_parakeet_model(model_dir)
    assert message is not None and "Parakeet" in message and str(model_dir) in message


def test_parakeet_model_partially_present_is_also_reported(tmp_path: Path) -> None:
    model_dir = tmp_path / "parakeet"
    _touch(model_dir / "config.json")  # only one of the four files
    message = check_parakeet_model(model_dir)
    assert message is not None and "Parakeet" in message


def test_qwen3_tts_model_present_is_silent(tmp_path: Path) -> None:
    model_dir = tmp_path / "qwen"
    _touch(model_dir / "config.json")
    _touch(model_dir / "model.safetensors")
    assert check_qwen3_tts_model(model_dir) is None


def test_qwen3_tts_model_missing_names_the_folder(tmp_path: Path) -> None:
    model_dir = tmp_path / "does-not-exist"
    message = check_qwen3_tts_model(model_dir)
    assert message is not None and "voice" in message.lower() and str(model_dir) in message


def test_ydotoold_socket_present_is_silent(tmp_path: Path) -> None:
    socket_path = tmp_path / ".ydotool_socket"
    _touch(socket_path)
    assert check_ydotoold(socket_path) is None


def test_ydotoold_socket_missing_mentions_typing() -> None:
    message = check_ydotoold(Path("/nonexistent/.ydotool_socket"))
    assert message is not None and "typ" in message.lower()


def test_capture_tools_all_present_is_silent() -> None:
    assert check_capture_tools(which=lambda name: f"/usr/bin/{name}") is None


def test_capture_tools_all_missing_is_reported() -> None:
    message = check_capture_tools(which=lambda name: None)
    assert message is not None and "tesseract" in message and "xclip" in message


def test_capture_tools_partially_missing_names_only_the_missing_ones() -> None:
    message = check_capture_tools(which=lambda name: "/usr/bin/tesseract" if name == "tesseract" else None)
    assert message is not None and "xclip" in message and "tesseract" not in message


def test_run_startup_checks_returns_every_real_problem_in_a_fixed_order(tmp_path: Path) -> None:
    config = VocalisConfig()
    config.engines.parakeet_model_dir = str(tmp_path / "missing-parakeet")
    config.engines.qwen3_tts_model_dir = str(tmp_path / "missing-qwen")
    messages = run_startup_checks(config)
    assert len(messages) >= 2
    assert "Parakeet" in messages[0]


def test_run_startup_checks_is_silent_when_everything_is_fine(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    config = VocalisConfig()
    parakeet_dir = tmp_path / "parakeet"
    for name in ("config.json", "encoder-model.int8.onnx", "decoder_joint-model.int8.onnx", "vocab.txt"):
        _touch(parakeet_dir / name)
    config.engines.parakeet_model_dir = str(parakeet_dir)
    qwen_dir = tmp_path / "qwen"
    _touch(qwen_dir / "config.json")
    _touch(qwen_dir / "model.safetensors")
    config.engines.qwen3_tts_model_dir = str(qwen_dir)
    socket_path = tmp_path / ".ydotool_socket"
    _touch(socket_path)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setattr("vocalis.startup_checks.shutil.which", lambda _name: "/usr/bin/x")
    assert run_startup_checks(config) == []


def test_run_startup_checks_survives_a_check_that_raises(tmp_path: Path, monkeypatch, caplog) -> None:  # type: ignore[no-untyped-def]
    config = VocalisConfig()

    def _boom(_model_dir: Path) -> str | None:
        raise RuntimeError("boom")

    monkeypatch.setattr("vocalis.startup_checks.check_parakeet_model", _boom)
    with caplog.at_level(logging.WARNING, logger="vocalis"):
        messages = run_startup_checks(config)
    assert "boom" in caplog.text
    assert all("Parakeet" not in m for m in messages)
```

- [ ] **Step 2: Run, verify FAIL** — `ModuleNotFoundError: vocalis.startup_checks`.

- [ ] **Step 3: Implement** — create `src/vocalis/startup_checks.py`:

```python
"""Non-fatal checks run once at startup, each reported as a desktop
notification if something looks wrong.

Every check is independent: if one is missing, the others still run and
Vocalis still starts. A missing piece only means the ONE feature that
needs it will fail per-request later, exactly as it does today -- these
checks exist purely to tell the user that up front instead of leaving
them to discover it the first time they use that feature.
"""
import logging
import os
import shutil
from pathlib import Path
from typing import Callable

from vocalis.config import VocalisConfig

logger = logging.getLogger("vocalis")

_PARAKEET_FILES = ("config.json", "encoder-model.int8.onnx", "decoder_joint-model.int8.onnx", "vocab.txt")
_QWEN3_TTS_FILES = ("config.json", "model.safetensors")
_CAPTURE_TOOLS = ("tesseract", "xclip")


def check_parakeet_model(model_dir: Path) -> str | None:
    """Whether the dictation (Parakeet) model files are all present.

    Returns:
        None if every expected file exists; otherwise a message naming
        the model and the folder it was expected in.
    """
    if all((model_dir / name).exists() for name in _PARAKEET_FILES):
        return None
    return (
        f"The Parakeet dictation model is missing or incomplete at {model_dir}. "
        "Dictation (NUMLOCK) will not work until it is installed."
    )


def check_qwen3_tts_model(model_dir: Path) -> str | None:
    """Whether the Qwen3-TTS voice model files are present.

    Returns:
        None if the expected files exist; otherwise a message naming the
        folder. Reading aloud still works via the Piper fallback voice.
    """
    if all((model_dir / name).exists() for name in _QWEN3_TTS_FILES):
        return None
    return (
        f"The primary voice model is missing at {model_dir}. "
        "Vocalis will use the backup voice until it is installed."
    )


def check_ydotoold(socket_path: Path) -> str | None:
    """Whether ydotoold's socket exists, so dictation can type at the cursor.

    Returns:
        None if the socket exists; otherwise a message explaining that
        typed text will fall back to the clipboard.
    """
    if socket_path.exists():
        return None
    return (
        f"ydotoold is not running ({socket_path} not found). "
        "Dictation will copy text to the clipboard instead of typing it."
    )


def check_capture_tools(which: Callable[[str], str | None] = shutil.which) -> str | None:
    """Whether the OCR Reader's text tools (tesseract, xclip) are installed.

    Returns:
        None if both are found; otherwise a message naming only the
        missing ones.
    """
    missing = [name for name in _CAPTURE_TOOLS if which(name) is None]
    if not missing:
        return None
    return f"The OCR Reader needs {', '.join(missing)}, which is not installed."


def run_startup_checks(config: VocalisConfig) -> list[str]:
    """Run every startup check and collect the real problems, in a fixed order.

    A check that itself raises is logged as a warning and treated as
    passing, so one broken check can never stop the others or crash
    startup.

    Args:
        config: Loaded, validated vocalis configuration.

    Returns:
        Zero or more human-readable messages, Parakeet first, then
        Qwen3-TTS, then ydotoold, then the capture tools.
    """
    runtime_dir = Path(os.environ.get("XDG_RUNTIME_DIR", "/run/user/0"))
    checks: "list[Callable[[], str | None]]" = [
        lambda: check_parakeet_model(Path(config.engines.parakeet_model_dir)),
        lambda: check_qwen3_tts_model(Path(config.engines.qwen3_tts_model_dir)),
        lambda: check_ydotoold(runtime_dir / ".ydotool_socket"),
        check_capture_tools,
    ]
    messages: "list[str]" = []
    for check in checks:
        try:
            result = check()
        except Exception:
            logger.warning("A startup check failed to run", exc_info=True)
            continue
        if result is not None:
            messages.append(result)
    return messages
```

- [ ] **Step 4: Run, verify PASS** — `.venv/bin/pytest tests/test_startup_checks.py -q --no-cov && .venv/bin/mypy`.

- [ ] **Step 5: Commit** — `git add src/vocalis/startup_checks.py tests/test_startup_checks.py && git commit -m "feat: add non-fatal startup checks for missing dependencies"`

---

### Task 3: SIGTERM handling in main()

**Files:**
- Modify: `src/vocalis/cli.py`
- Test: `tests/test_cli.py` (append; read the existing `test_main_installs_sigint_handler_that_quits_qt_and_restores_it` test first and follow its exact pattern for SIGTERM)

**Interfaces:** no new parameters; `main()`'s behaviour gains a SIGTERM handler.

Behaviour: in `main()`, install a `SIGTERM` handler with the identical lambda used for `SIGINT` (`lambda *_: app.quit()`), saved and restored the same way (`previous_sigterm_handler`), right next to the existing `SIGINT` line. Update the module and `main()` docstrings (the module docstring's paragraph that says SIGTERM is "not yet implemented" is now wrong — rewrite it to say both signals are handled identically).

- [ ] **Step 1: Write the failing test** — append to `tests/test_cli.py` (copy the existing SIGINT test's structure exactly, substituting SIGTERM):

```python
def test_main_installs_sigterm_handler_that_quits_qt_and_restores_it(qapp) -> None:  # type: ignore[no-untyped-def]
    original_handler = signal.getsignal(signal.SIGTERM)
    seen: dict[str, object] = {}

    def fake_exec(_self: object) -> int:
        handler = signal.getsignal(signal.SIGTERM)
        seen["handler"] = handler
        assert callable(handler)
        assert handler is not original_handler
        handler(signal.SIGTERM, None)
        return 0

    with (
        patch("vocalis.cli.list_numlock_capable_devices", return_value=[]),
        patch("vocalis.cli.list_hotkey_capable_devices", return_value=[]),
        patch("vocalis.cli.list_devices_with_key", return_value=[]),
        patch("vocalis.cli.load_config", return_value=VocalisConfig()),
        patch("vocalis.cli.VocalisDaemon"),
        patch("vocalis.cli.QApplication.exec", fake_exec),
        patch("vocalis.cli.QApplication.quit") as mock_quit,
    ):
        main()

    mock_quit.assert_called_once()
    assert signal.getsignal(signal.SIGTERM) == original_handler
```

(This mirrors the file's existing `test_main_installs_sigint_handler_that_quits_qt_and_restores_it` test exactly — read that one first and copy its imports/patches if they differ from this sketch.)

- [ ] **Step 2: Run, verify FAIL** — the handler is never installed today, so `signal.getsignal(signal.SIGTERM)` never changes and the assertion `handler is not original_handler` fails.

- [ ] **Step 3: Implement** — in `src/vocalis/cli.py`'s `main()`, change:

```python
    previous_sigint_handler: "signal.Handlers | int | Callable[..., object] | None" = None
    keepalive_timer: QTimer | None = None
    try:
        previous_sigint_handler = signal.signal(signal.SIGINT, lambda *_: app.quit())
```

to:

```python
    previous_sigint_handler: "signal.Handlers | int | Callable[..., object] | None" = None
    previous_sigterm_handler: "signal.Handlers | int | Callable[..., object] | None" = None
    keepalive_timer: QTimer | None = None
    try:
        previous_sigint_handler = signal.signal(signal.SIGINT, lambda *_: app.quit())
        previous_sigterm_handler = signal.signal(signal.SIGTERM, lambda *_: app.quit())
```

and in the `finally` block, right after the existing SIGINT restore:

```python
        if previous_sigint_handler is not None:
            signal.signal(signal.SIGINT, previous_sigint_handler)
        if previous_sigterm_handler is not None:
            signal.signal(signal.SIGTERM, previous_sigterm_handler)
```

Update the module docstring (the "Graceful SIGTERM handling ... is not yet implemented" paragraph) to: "Both SIGINT (Ctrl+C) and SIGTERM (systemd `stop`/`restart`) call `app.quit()` through the identical handler pattern, so the shutdown code in `main()`'s `finally` block runs the same way regardless of which signal arrived." Update `main()`'s own docstring sentence about "The previous SIGINT handler is restored on the way out" to mention SIGTERM too.

- [ ] **Step 4: Run, verify PASS** — `.venv/bin/pytest tests/test_cli.py -q --no-cov && .venv/bin/mypy`, then the full suite once (existing SIGINT test and everything else must still pass).

- [ ] **Step 5: Commit** — `git add src/vocalis/cli.py tests/test_cli.py && git commit -m "feat: handle SIGTERM exactly like SIGINT so systemd can stop vocalis cleanly"`

---

### Task 4: Run startup checks in main()

**Files:**
- Modify: `src/vocalis/cli.py`
- Test: `tests/test_cli.py` (append)

**Interfaces:** no new parameters; uses Task 2's `run_startup_checks`.

Behaviour: in `main()`, right after `config = load_config(default_config_path())`, call `messages = run_startup_checks(config)`, then for each message call `Notifier(enabled=config.notifications_enabled).notify("Vocalis", message)` (one fresh Notifier reused across the loop, built once; this runs BEFORE the `QApplication`/hub/listeners exist, so it must not depend on anything built later). Import `run_startup_checks` and `Notifier` is already imported in `cli.py`.

- [ ] **Step 1: Write the failing test** — append to `tests/test_cli.py`:

```python
def test_main_notifies_about_every_real_startup_problem(tmp_path: Path, qapp) -> None:  # type: ignore[no-untyped-def]
    sent: list[tuple[str, str]] = []

    def fake_notify(self: object, title: str, message: str = "") -> None:  # noqa: ANN001
        sent.append((title, message))

    with (
        patch("vocalis.cli.default_config_path", return_value=tmp_path / "config.toml"),
        patch("vocalis.cli._socket_path", return_value=tmp_path / "vocalis.sock"),
        patch("vocalis.cli.list_numlock_capable_devices", return_value=[]),
        patch("vocalis.cli.list_hotkey_capable_devices", return_value=[]),
        patch("vocalis.cli.list_devices_with_key", return_value=[]),
        patch("vocalis.cli.run_startup_checks", return_value=["Problem A", "Problem B"]),
        patch("vocalis.cli.Notifier.notify", fake_notify),
        patch("vocalis.cli.QApplication.exec", side_effect=KeyboardInterrupt),
    ):
        main()

    assert ("Vocalis", "Problem A") in sent
    assert ("Vocalis", "Problem B") in sent


def test_main_sends_no_startup_notification_when_everything_is_fine(tmp_path: Path, qapp) -> None:  # type: ignore[no-untyped-def]
    sent: list[tuple[str, str]] = []

    def fake_notify(self: object, title: str, message: str = "") -> None:  # noqa: ANN001
        sent.append((title, message))

    with (
        patch("vocalis.cli.default_config_path", return_value=tmp_path / "config.toml"),
        patch("vocalis.cli._socket_path", return_value=tmp_path / "vocalis.sock"),
        patch("vocalis.cli.list_numlock_capable_devices", return_value=[]),
        patch("vocalis.cli.list_hotkey_capable_devices", return_value=[]),
        patch("vocalis.cli.list_devices_with_key", return_value=[]),
        patch("vocalis.cli.run_startup_checks", return_value=[]),
        patch("vocalis.cli.Notifier.notify", fake_notify),
        patch("vocalis.cli.QApplication.exec", side_effect=KeyboardInterrupt),
    ):
        main()

    assert sent == []
```

- [ ] **Step 2: Run, verify FAIL** — `AttributeError`/`ImportError`-shaped failure (`run_startup_checks` not imported in `cli.py` yet; the patch target does not exist).

- [ ] **Step 3: Implement** — add `from vocalis.startup_checks import run_startup_checks` to `cli.py`'s imports; in `main()`, after `config = load_config(default_config_path())`:

```python
    startup_notifier = Notifier(enabled=config.notifications_enabled)
    for message in run_startup_checks(config):
        startup_notifier.notify("Vocalis", message)
```

Update `main()`'s docstring to mention this step.

- [ ] **Step 4: Run, verify PASS** — `.venv/bin/pytest tests/test_cli.py -q --no-cov && .venv/bin/mypy`, then the full suite once.

- [ ] **Step 5: Commit** — `git add src/vocalis/cli.py tests/test_cli.py && git commit -m "feat: notify about missing dependencies when vocalis starts"`

---

### Task 5: The systemd unit file (template, not installed)

**Files:**
- Create: `packaging/systemd/vocalis.service`
- Create: `packaging/systemd/README.md`
- Test: `tests/test_systemd_unit.py`

No source code changes. This unit file is a TEMPLATE: it uses placeholder tokens (`__VOCALIS_EXEC__`, `__VOCALIS_WORKDIR__`) that Stage 3's installer will substitute with real, installed paths. It is not copied anywhere, enabled, or started by this plan.

- [ ] **Step 1: Write the failing test** — create `tests/test_systemd_unit.py`:

```python
"""Tests for the packaging/systemd/vocalis.service template (text content only;
never starts a real service)."""
from pathlib import Path

UNIT_PATH = Path(__file__).resolve().parent.parent / "packaging" / "systemd" / "vocalis.service"


def _lines() -> "list[str]":
    return UNIT_PATH.read_text().splitlines()


def test_unit_file_exists() -> None:
    assert UNIT_PATH.exists()


def test_restart_policy_matches_the_design() -> None:
    lines = _lines()
    assert "Restart=on-failure" in lines
    assert "RestartSec=3" in lines
    assert "StartLimitBurst=5" in lines
    assert "StartLimitIntervalSec=60" in lines


def test_wants_the_xdg_autostart_target_not_default_target() -> None:
    lines = _lines()
    assert any(line.startswith("WantedBy=") and "xdg-desktop-autostart.target" in line for line in lines)
    assert not any(line.startswith("WantedBy=default.target") for line in lines)


def test_waits_for_the_sound_system() -> None:
    text = UNIT_PATH.read_text()
    for unit in ("pipewire.service", "pipewire-pulse.service", "wireplumber.service"):
        assert unit in text, f"{unit} must appear in After=/Wants="
    assert any(line.startswith("After=") for line in _lines())
    assert any(line.startswith("Wants=") for line in _lines())


def test_execstart_uses_a_placeholder_not_a_real_path() -> None:
    text = UNIT_PATH.read_text()
    assert "__VOCALIS_EXEC__" in text
    assert "/home/" not in text  # never a real, user-specific path baked in


def test_readme_documents_the_placeholders() -> None:
    readme = UNIT_PATH.parent / "README.md"
    assert readme.exists()
    text = readme.read_text()
    assert "__VOCALIS_EXEC__" in text
```

- [ ] **Step 2: Run, verify FAIL** — `FileNotFoundError`-shaped failures (the files don't exist yet).

- [ ] **Step 3: Implement** — create `packaging/systemd/vocalis.service`:

```ini
[Unit]
Description=Vocalis voice AI workstation
After=pipewire.service pipewire-pulse.service wireplumber.service
Wants=pipewire.service pipewire-pulse.service wireplumber.service
StartLimitIntervalSec=60
StartLimitBurst=5

[Service]
Type=simple
ExecStart=__VOCALIS_EXEC__
WorkingDirectory=__VOCALIS_WORKDIR__
Restart=on-failure
RestartSec=3

[Install]
WantedBy=xdg-desktop-autostart.target
```

Create `packaging/systemd/README.md`:

```markdown
# Vocalis systemd user service (template)

`vocalis.service` is a template, not a ready-to-use unit file. Two
placeholders must be substituted before it can be installed:

- `__VOCALIS_EXEC__` — the absolute path to the installed `vocalis`
  command (e.g. `/opt/vocalis/venv/bin/vocalis` or wherever Stage 3's
  installer puts it).
- `__VOCALIS_WORKDIR__` — the directory `vocalis` should run from.

Substituting these values, copying the result to
`~/.config/systemd/user/vocalis.service`, and running
`systemctl --user enable --now vocalis.service` is Stage 3's job (the
installer), not something this stage does or asks the user to do by hand.

## Design notes

- `WantedBy=xdg-desktop-autostart.target` (not `default.target`): this
  account has `Linger=yes`, so a `default.target` unit would start at
  boot, before any desktop session — and before a display — exists.
  `xdg-desktop-autostart.target` starts once a desktop session is up, on
  any desktop environment (XFCE, GNOME, KDE, ...).
- `After=`/`Wants=` on PipeWire's three user services: Vocalis's sound
  output chooser (Stage 1) reads the list of sound devices once at
  startup and never refreshes it. Starting before the sound system is
  ready would mean the real outputs (`pulse`, `pipewire`) are invisible
  for that entire run.
- `Restart=on-failure` + `RestartSec=3` + the `StartLimitBurst=5` /
  `StartLimitIntervalSec=60` pair: a crash is retried quickly, but 5
  crashes inside 60 seconds makes systemd give up rather than loop
  forever. Vocalis's own startup-checks notification (`cli.py`) is a
  separate, best-effort heads-up about missing dependencies — it is not
  what detects systemd's restart-limit; systemd decides that on its own.

## Verifying the file (text only — never starts the real service)

```
systemd-analyze --user verify packaging/systemd/vocalis.service
```

This checks the file parses as a valid unit. It still fails on the
placeholders (they are not real paths) until Stage 3 substitutes them —
that failure is expected and fine; it proves the rest of the syntax is
sound.
```

- [ ] **Step 4: Run, verify PASS** — `.venv/bin/pytest tests/test_systemd_unit.py -q --no-cov`.

- [ ] **Step 5: Commit** — `git add packaging/systemd/vocalis.service packaging/systemd/README.md tests/test_systemd_unit.py && git commit -m "feat: add the systemd user service unit template"`

---

### Task 6: Docs, change log, verification, real-machine text check

**Files:**
- Modify: `docs/user-docs.md`, `docs/architecture.md`, `docs/workflow.md`, `docs/file-purpose.md`, `docs/folder-structure.md`, `docs/developer-docs.md`, the Stage 2 amendment in the spec (deviations list)
- Create: `.claude/<YYYY-MM-DD_HHMM>_CHANGELOG.md`

- [ ] **Step 1: Docs** (verify every sentence against the code): user-docs.md — a short "Running automatically" section stating this is set up by the Stage 3 installer, not something to configure by hand yet; what the "could not start" notification means and that it's rare; what each startup-check notification means in plain words. architecture.md/workflow.md — the single-instance probe-before-delete design; SIGTERM handled identically to SIGINT; where startup checks run relative to the rest of `main()`. file-purpose.md/folder-structure.md — `startup_checks.py`, `packaging/systemd/`. developer-docs.md — how to add a new startup check (the "never raises for a normal case, return a message or None" contract) and how `_is_another_instance_alive` decides stale-vs-alive.
- [ ] **Step 2: Change log** — files touched, test counts/coverage from your own final run, the explicit note that `systemd-analyze --user verify` was run on the unit file and what it reported (expected: valid syntax, placeholder substitution errors only), and `- [ ] Real-world check pending: enabling and running the real systemd service, a real crash-loop hitting the burst limit, a real SIGTERM from systemctl stop — done by the user only after Stage 3's installer exists`.
- [ ] **Step 3: Full verification** — `.venv/bin/pytest -q` three times sequentially (all pass, zero warnings, coverage >= 85%, no segfault), `.venv/bin/mypy` (Success), loop `tests/test_daemon.py tests/test_startup_checks.py tests/test_cli.py tests/test_systemd_unit.py` 10 times (all green every time).
- [ ] **Step 4: Real-machine text check (controller runs this, not an implementer; it never starts the service)** —

```bash
systemd-analyze --user verify packaging/systemd/vocalis.service
```

Record the exact output in the change log. Expected: it complains only about the unresolved `__VOCALIS_EXEC__`/`__VOCALIS_WORKDIR__` placeholders (or accepts them as plain strings with no further complaint, depending on the installed systemd version) — it must NOT report a syntax error in `[Unit]`/`[Service]`/`[Install]` structure, the `Restart=`/`RestartSec=`/`StartLimit*=` keys, or the `After=`/`Wants=`/`WantedBy=` lines.

- [ ] **Step 5: Commit** — `git add docs .claude/*CHANGELOG.md && git commit -m "docs: document the service, autostart and startup checks"`
