# Vocalis Stage 5 — Foundations & Quick Wins Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the MIT license, a hardware probe, clearer microphone-problem messages, and a way for the user to find and open the Vocalis window.

**Architecture:** Five small, independent additions to the existing Python package (`src/vocalis`) and the bash installer (`packaging/installer`). A new pure `hardware.py` module, one new startup check, a daemon command hook plus a tiny client module (`remote.py`) for `vocalis --settings`, a tray click handler, and an installed `.desktop` launcher.

**Tech Stack:** Python 3.13, PySide6 (Qt), pytest + pytest-qt (offscreen), mypy strict, bash + bats + shellcheck, hatchling.

**Spec:** `docs/superpowers/specs/2026-09-11-vocalis-design.md` — section "Stage 5 amendment — Foundations and quick wins".

## Global Constraints

- Run tests with `export PYTHONPATH=$PWD/src` then `.venv/bin/python -m pytest -q -p no:cacheprovider` (coverage must stay >= 85%, zero warnings).
- `.venv/bin/python -m mypy` must report "Success: no issues found" (strict mode; new code fully typed).
- `shellcheck` zero warnings on every changed `.sh`; `bats tests/installer` all green.
- Tests NEVER touch real hardware, GPU, network, real systemctl, real sockets outside `tmp_path`, or real `~/.asoundrc`.
- The working tree is on an NTFS mount: `sed -i` fails ("Operation not permitted"). Edit files with the Edit tool or python3.
- Every function needs a short purpose docstring (project rule). User-facing text is plain, non-technical English.
- Commit on the worktree branch after every task with conventional messages; no `Co-Authored-By` lines.
- Do not change existing exact UI strings that tests assert ("Vocalis — idle", "Vocalis — Settings", etc.).

## Review Focus

1. A path with spaces in `$HOME` must still produce a working `.desktop` `Exec=` line (quote the path).
2. `vocalis --settings` when Vocalis is not running AND the service fails to start must exit 1 with a plain message, never hang longer than the wait limit, never raise a traceback.
3. An unreadable `~/.asoundrc` (permission denied, binary content) must not crash startup or the check (return None).
4. `probe_hardware()` when torch is missing, CUDA unavailable, or `/proc` files are absent must return a `HardwareInfo` with `None` fields, never raise.
5. Clicking the tray icon or sending `show-settings` while the Settings window is already open must just raise it (no duplicate windows, no lost edits reset unexpectedly — `present()` reloads current saved values only when the window was hidden).

---

### Task 1: MIT license

**Files:**
- Create: `LICENSE`
- Modify: `pyproject.toml` (the `[project]` table)
- Modify: `README.md` (add a "License" section at the end)
- Test: `tests/test_package_metadata.py` (new)

**Interfaces:** Consumes nothing. Produces nothing used by other tasks.

- [ ] **Step 1: Write the failing test** — `tests/test_package_metadata.py`:

```python
"""Tests that the project's license metadata is present and consistent."""
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_license_file_is_mit_with_the_copyright_holder() -> None:
    text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert text.startswith("MIT License")
    assert "Copyright (c) 2026 reduxailabs-max" in text


def test_pyproject_declares_mit_and_ships_the_license_file() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["license"] == "MIT"
    assert project["license-files"] == ["LICENSE"]


def test_readme_mentions_the_license() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "## License" in readme and "MIT" in readme
```

- [ ] **Step 2: Run it, expect FAIL** — `.venv/bin/python -m pytest tests/test_package_metadata.py -q -p no:cacheprovider` → FileNotFoundError for LICENSE.

- [ ] **Step 3: Implement**
  - `LICENSE`: the standard MIT text, first line `MIT License`, blank line, `Copyright (c) 2026 reduxailabs-max`, then the standard permission and warranty paragraphs verbatim.
  - `pyproject.toml`, in `[project]` right after `description`: `license = "MIT"` and `license-files = ["LICENSE"]`.
  - `README.md`, append:

```markdown

## License

Vocalis is released under the [MIT License](LICENSE). The voice and speech
models it downloads come with their own licenses (Qwen3-TTS and Parakeet are
Apache-2.0 or similar; see each model's page).
```

- [ ] **Step 4: Run the test, expect PASS.** Also run `.venv/bin/uv build` is NOT required; instead run `.venv/bin/python -c "import tomllib;tomllib.load(open('pyproject.toml','rb'))"` to prove the TOML parses.
- [ ] **Step 5: Commit** — `git add LICENSE pyproject.toml README.md tests/test_package_metadata.py && git commit -m "chore: add MIT license and license metadata"`

---

### Task 2: Hardware probe + one startup log line

**Files:**
- Create: `src/vocalis/hardware.py`
- Modify: `src/vocalis/cli.py` (`main()`, right after the `_warm_up_pace_changer` thread starts)
- Test: `tests/test_hardware.py` (new); `tests/test_cli.py` (autouse patch)

**Interfaces:**
- Produces (used by Stage 6):

```python
@dataclass(frozen=True)
class GpuInfo:
    name: str
    total_mb: int
    free_mb: int

@dataclass(frozen=True)
class HardwareInfo:
    gpu: GpuInfo | None
    ram_total_mb: int | None
    ram_available_mb: int | None
    cpu_count: int | None
    load_1min: float | None

def probe_hardware(
    meminfo_path: Path = Path("/proc/meminfo"),
    loadavg_path: Path = Path("/proc/loadavg"),
    cpu_count: Callable[[], int | None] = os.cpu_count,
    gpu_query: Callable[[], GpuInfo | None] | None = None,  # None -> query_gpu_with_torch
) -> HardwareInfo: ...

def query_gpu_with_torch() -> GpuInfo | None: ...
def describe(info: HardwareInfo) -> str: ...   # one plain-English log line
def log_hardware_summary() -> None: ...         # probe + logger.info(describe(...)); never raises
```

- [ ] **Step 1: Write the failing tests** — `tests/test_hardware.py`:

```python
"""Tests for vocalis.hardware — every data source is faked; no real /proc or GPU."""
import logging
from pathlib import Path

from vocalis.hardware import GpuInfo, HardwareInfo, describe, log_hardware_summary, probe_hardware

MEMINFO = "MemTotal:       16000000 kB\nMemFree:  1000000 kB\nMemAvailable:    8000000 kB\n"


def _files(tmp_path: Path, meminfo: str = MEMINFO, loadavg: str = "1.50 1.00 0.50 2/300 42\n") -> tuple[Path, Path]:
    m, l = tmp_path / "meminfo", tmp_path / "loadavg"
    m.write_text(meminfo)
    l.write_text(loadavg)
    return m, l


def test_reads_ram_load_cpu_and_gpu(tmp_path: Path) -> None:
    m, l = _files(tmp_path)
    info = probe_hardware(m, l, cpu_count=lambda: 8, gpu_query=lambda: GpuInfo("RTX", 8188, 7000))
    assert info == HardwareInfo(GpuInfo("RTX", 8188, 7000), 15625, 7812, 8, 1.5)


def test_missing_proc_files_give_none_not_an_error(tmp_path: Path) -> None:
    info = probe_hardware(tmp_path / "nope", tmp_path / "nope2", cpu_count=lambda: None, gpu_query=lambda: None)
    assert info == HardwareInfo(None, None, None, None, None)


def test_garbage_proc_content_gives_none(tmp_path: Path) -> None:
    m, l = _files(tmp_path, meminfo="garbage\n", loadavg="x y z\n")
    info = probe_hardware(m, l, cpu_count=lambda: 4, gpu_query=lambda: None)
    assert info.ram_total_mb is None and info.ram_available_mb is None and info.load_1min is None


def test_a_raising_gpu_query_means_no_gpu(tmp_path: Path) -> None:
    m, l = _files(tmp_path)

    def boom() -> GpuInfo | None:
        raise RuntimeError("CUDA driver missing")

    assert probe_hardware(m, l, cpu_count=lambda: 4, gpu_query=boom).gpu is None


def test_describe_is_one_plain_line() -> None:
    line = describe(HardwareInfo(GpuInfo("RTX 4060", 8188, 7000), 15625, 7812, 8, 1.5))
    assert "RTX 4060" in line and "8188" in line and "8 CPU" in line and "\n" not in line
    assert "no GPU" in describe(HardwareInfo(None, None, None, None, None))


def test_log_hardware_summary_never_raises(monkeypatch, caplog) -> None:  # type: ignore[no-untyped-def]
    def boom(*_a, **_k):  # type: ignore[no-untyped-def]
        raise RuntimeError("probe blew up")

    monkeypatch.setattr("vocalis.hardware.probe_hardware", boom)
    with caplog.at_level(logging.WARNING, logger="vocalis"):
        log_hardware_summary()
    assert "Could not read hardware details" in caplog.text
```

- [ ] **Step 2: Run, expect FAIL** (ModuleNotFoundError).
- [ ] **Step 3: Implement `src/vocalis/hardware.py`:**

```python
"""Reads what the computer has right now (GPU memory, RAM, CPU load) so
Vocalis can explain itself in the log today and pick a voice engine later.
Every source is optional and injectable; nothing here ever raises."""
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

logger = logging.getLogger("vocalis")

_KB_PER_MB = 1024
_BYTES_PER_MB = 1024 * 1024


@dataclass(frozen=True)
class GpuInfo:
    """One CUDA GPU: its name and memory in megabytes."""

    name: str
    total_mb: int
    free_mb: int


@dataclass(frozen=True)
class HardwareInfo:
    """A snapshot of the machine; a field is None when it could not be read."""

    gpu: GpuInfo | None
    ram_total_mb: int | None
    ram_available_mb: int | None
    cpu_count: int | None
    load_1min: float | None


def query_gpu_with_torch() -> GpuInfo | None:
    """Return the first CUDA GPU's name and memory, or None if there is none."""
    import torch  # slow import, kept local

    if not torch.cuda.is_available():
        return None
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    return GpuInfo(torch.cuda.get_device_name(0), total_bytes // _BYTES_PER_MB, free_bytes // _BYTES_PER_MB)


def _read_meminfo(path: Path) -> tuple[int | None, int | None]:
    """Return (total, available) RAM in MB from a /proc/meminfo-style file."""
    try:
        values: dict[str, int] = {}
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            key, _, rest = line.partition(":")
            parts = rest.split()
            if parts and parts[0].isdigit():
                values[key.strip()] = int(parts[0])
    except OSError:
        return None, None
    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    return (
        total // _KB_PER_MB if total is not None else None,
        available // _KB_PER_MB if available is not None else None,
    )


def _read_load(path: Path) -> float | None:
    """Return the 1-minute load average from a /proc/loadavg-style file."""
    try:
        return float(path.read_text(encoding="utf-8", errors="replace").split()[0])
    except (OSError, ValueError, IndexError):
        return None


def probe_hardware(
    meminfo_path: Path = Path("/proc/meminfo"),
    loadavg_path: Path = Path("/proc/loadavg"),
    cpu_count: Callable[[], int | None] = os.cpu_count,
    gpu_query: Callable[[], GpuInfo | None] | None = None,
) -> HardwareInfo:
    """Take a snapshot of GPU, RAM and CPU; unreadable parts become None."""
    query = gpu_query if gpu_query is not None else query_gpu_with_torch
    try:
        gpu = query()
    except Exception:
        logger.debug("GPU query failed; treating as no GPU", exc_info=True)
        gpu = None
    ram_total, ram_available = _read_meminfo(meminfo_path)
    return HardwareInfo(gpu, ram_total, ram_available, cpu_count(), _read_load(loadavg_path))


def describe(info: HardwareInfo) -> str:
    """One plain-English line summarising a snapshot, for the log."""
    gpu = (
        f"GPU {info.gpu.name} ({info.gpu.free_mb} of {info.gpu.total_mb} MB free)"
        if info.gpu is not None
        else "no GPU"
    )
    ram = f"RAM {info.ram_available_mb} of {info.ram_total_mb} MB free" if info.ram_total_mb else "RAM unknown"
    cpu = f"{info.cpu_count} CPU cores" if info.cpu_count else "CPU cores unknown"
    load = f"load {info.load_1min:.2f}" if info.load_1min is not None else "load unknown"
    return f"Hardware: {gpu}; {ram}; {cpu}; {load}"


def log_hardware_summary() -> None:
    """Probe the machine and write one INFO line; any failure is a warning only."""
    try:
        logger.info(describe(probe_hardware()))
    except Exception:
        logger.warning("Could not read hardware details", exc_info=True)
```

- [ ] **Step 4: Wire it in `cli.main()`**: add `from vocalis.hardware import log_hardware_summary` to the imports and, directly after `threading.Thread(target=_warm_up_pace_changer, daemon=True).start()`, add `threading.Thread(target=log_hardware_summary, daemon=True).start()`. In `tests/test_cli.py` add an autouse fixture so main()-tests never import torch for this:

```python
@pytest.fixture(autouse=True)
def _no_real_hardware_probe() -> Iterator[None]:
    with patch("vocalis.cli.log_hardware_summary"):
        yield
```

Also add the same patch to the autouse guard in `tests/test_cross_feature_integration.py` only if that file calls `main()` (it does not today — check with grep; if not, leave it).
- [ ] **Step 5: Run `tests/test_hardware.py tests/test_cli.py`, then the full suite + mypy; expect green.**
- [ ] **Step 6: Commit** — `git commit -m "feat: add hardware probe and log a one-line hardware summary at startup"`

---

### Task 3: Clearer microphone problems

**Files:**
- Modify: `src/vocalis/startup_checks.py` (new `check_stale_asoundrc`, add it to `run_startup_checks`, update module docstring list)
- Modify: `src/vocalis/features/dictation.py:124` (message text)
- Test: `tests/test_startup_checks.py`, `tests/test_dictation.py`

**Interfaces:** Produces `check_stale_asoundrc(path: Path) -> str | None` and module constant `NO_AUDIO_MESSAGE` in `vocalis.features.dictation`.

- [ ] **Step 1: Failing tests.** Append to `tests/test_startup_checks.py` (and add `check_stale_asoundrc` to its import list):

```python
def test_no_asoundrc_is_silent(tmp_path: Path) -> None:
    assert check_stale_asoundrc(tmp_path / ".asoundrc") is None


def test_unrelated_asoundrc_is_silent(tmp_path: Path) -> None:
    path = tmp_path / ".asoundrc"
    path.write_text("pcm.!default { type pulse }\n")
    assert check_stale_asoundrc(path) is None


def test_old_voxtype_asoundrc_is_reported_in_plain_words(tmp_path: Path) -> None:
    path = tmp_path / ".asoundrc"
    path.write_text("# ALSA configuration for VOXTYPE\npcm.voxtype { type plug }\n")
    message = check_stale_asoundrc(path)
    assert message is not None
    assert "Voxtype" in message and ".asoundrc.old" in message and str(path) in message


def test_unreadable_asoundrc_is_silent_not_a_crash(tmp_path: Path) -> None:
    path = tmp_path / ".asoundrc"
    path.mkdir()  # a directory: read_text raises IsADirectoryError
    assert check_stale_asoundrc(path) is None


def test_binary_asoundrc_is_silent(tmp_path: Path) -> None:
    path = tmp_path / ".asoundrc"
    path.write_bytes(b"\xff\xfe\x00voxtype-not-really")
    assert check_stale_asoundrc(path) is None or "Voxtype" in (check_stale_asoundrc(path) or "")
```

In `test_run_startup_checks_is_silent_when_everything_is_fine` add `monkeypatch.setenv("HOME", str(tmp_path))` so the real `~/.asoundrc` is never read. Append:

```python
def test_run_startup_checks_reports_a_stale_asoundrc_last(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".asoundrc").write_text("pcm.voxtype { }\n")
    messages = run_startup_checks(VocalisConfig())
    assert "Voxtype" in messages[-1]
```

In `tests/test_dictation.py::test_empty_audio_skips_transcription` replace the last assert with
`assert any(message == NO_AUDIO_MESSAGE for _, message in notifier.messages)` and import `NO_AUDIO_MESSAGE` from `vocalis.features.dictation`; add:

```python
def test_no_audio_message_tells_the_user_what_to_check() -> None:
    assert "microphone" in NO_AUDIO_MESSAGE and "muted" in NO_AUDIO_MESSAGE
```

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.** In `startup_checks.py`:

```python
def check_stale_asoundrc(path: Path) -> str | None:
    """Whether an old Voxtype ~/.asoundrc is still overriding the system's sound.

    Returns:
        None if the file is absent, unreadable, or not from Voxtype; otherwise
        a plain message saying what it does and how to retire it.
    """
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if "voxtype" not in content.lower():
        return None
    return (
        f"An old sound settings file from Voxtype is still at {path}. "
        "It can make the microphone silent in other apps. Vocalis works around it, "
        f"but you can rename it to {path.name}.old to fix it everywhere."
    )
```

In `run_startup_checks` append `lambda: check_stale_asoundrc(Path(os.environ.get("HOME", str(Path.home()))) / ".asoundrc"),` as the LAST check, and update the docstring order list ("...then the capture tools, then a stale ~/.asoundrc").
In `dictation.py` add near `_TITLE`:

```python
NO_AUDIO_MESSAGE = (
    "No sound was recorded. Check that your microphone is plugged in, not muted, "
    "and chosen as the input device, then try again."
)
```

and change line 124 to `self._notifier.notify(_TITLE, NO_AUDIO_MESSAGE)`.
- [ ] **Step 4: Run both test files, full suite, mypy → green.**
- [ ] **Step 5: Commit** — `git commit -m "feat: warn about a stale Voxtype .asoundrc and explain empty recordings"`

---

### Task 4: Make the Settings window findable (tray click, daemon command, `vocalis --settings`)

**Files:**
- Modify: `src/vocalis/ui/settings_window.py` (add `present()`)
- Modify: `src/vocalis/ui/tray.py` (connect `activated`)
- Modify: `src/vocalis/daemon.py` (optional `command_handler`)
- Create: `src/vocalis/remote.py` (client: send a command, start the service if needed)
- Modify: `src/vocalis/cli.py` (`build_daemon` passes the handler; `build_ui` uses `present()`; new `_SettingsBridge`; new `entrypoint()`)
- Modify: `pyproject.toml` (`vocalis = "vocalis.cli:entrypoint"`)
- Test: `tests/test_settings_window.py`, `tests/test_tray.py`, `tests/test_daemon.py`, `tests/test_remote.py` (new), `tests/test_cli.py`

**Interfaces:**

```python
# settings_window.py
class SettingsWindow:
    def present(self) -> None: ...  # if hidden: load_from_settings(); then show(), raise_(), activateWindow()

# daemon.py
VocalisDaemon(socket_path, logger, model_manager, socket_factory=..., command_handler: Callable[[str], None] | None = None)

# remote.py
SHOW_SETTINGS = "show-settings"
def send_command(socket_path: Path, command: str, timeout: float = 2.0,
                 socket_factory: Callable[[], socket.socket] = <AF_UNIX stream>) -> bool: ...
def start_user_service() -> bool: ...   # runs ["systemctl", "--user", "start", "vocalis"], True on exit 0
def open_settings(socket_path: Path, start_service: Callable[[], bool] = start_user_service,
                  sleep: Callable[[float], None] = time.sleep,
                  wait_seconds: float = 20.0, poll_seconds: float = 0.5,
                  send: Callable[[Path, str], bool] = send_command) -> int: ...  # 0 ok, 1 failure

# cli.py
def build_daemon(model_manager: ModelManager, socket_path: Path,
                 command_handler: Callable[[str], None] | None = None) -> VocalisDaemon: ...
def entrypoint(argv: list[str] | None = None) -> None: ...  # console script
```

- [ ] **Step 1: Failing tests.**

`tests/test_settings_window.py`:

```python
def test_present_shows_and_reloads_saved_values_when_hidden(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings, window, _applied, _path = make(qtbot, tmp_path)
    window._voice_combo.setCurrentText("Aiden")  # noqa: SLF001 - unsaved edit while hidden
    window.present()
    assert window.isVisible()
    assert window._voice_combo.currentText() == settings.voice()  # noqa: SLF001


def test_present_keeps_unsaved_edits_when_already_open(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    _settings, window, _applied, _path = make(qtbot, tmp_path)
    window.present()
    window._voice_combo.setCurrentText("Aiden")  # noqa: SLF001
    window.present()
    assert window._voice_combo.currentText() == "Aiden"  # noqa: SLF001
```

`tests/test_tray.py`:

```python
from PySide6.QtWidgets import QSystemTrayIcon


def test_left_click_opens_settings(qtbot) -> None:  # type: ignore[no-untyped-def]
    _hub, tray, calls = make(qtbot)
    tray._tray.activated.emit(QSystemTrayIcon.ActivationReason.Trigger)  # noqa: SLF001
    assert calls == ["settings"]


def test_right_click_does_not_open_settings(qtbot) -> None:  # type: ignore[no-untyped-def]
    _hub, tray, calls = make(qtbot)
    tray._tray.activated.emit(QSystemTrayIcon.ActivationReason.Context)  # noqa: SLF001
    assert calls == []
```

`tests/test_daemon.py`:

```python
def test_command_handler_receives_each_command_and_reply_is_unchanged(tmp_path: Path) -> None:
    received: list[str] = []
    daemon = VocalisDaemon(socket_path=tmp_path / "vocalis.sock", logger=configure_logging(),
                           model_manager=ModelManager(), command_handler=received.append)
    daemon.start()
    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(str(tmp_path / "vocalis.sock"))
        client.sendall(b"show-settings\n")
        assert client.recv(1024) == b"ack:show-settings\n"
        client.close()
    finally:
        daemon.stop()
    assert received == ["show-settings"]


def test_a_raising_command_handler_is_logged_and_still_acked(tmp_path: Path, caplog) -> None:  # type: ignore[no-untyped-def]
    def boom(_command: str) -> None:
        raise RuntimeError("handler broke")

    daemon = VocalisDaemon(socket_path=tmp_path / "vocalis.sock", logger=configure_logging(),
                           model_manager=ModelManager(), command_handler=boom)
    daemon.start()
    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(str(tmp_path / "vocalis.sock"))
        client.sendall(b"show-settings\n")
        assert client.recv(1024) == b"ack:show-settings\n"
        client.close()
    finally:
        daemon.stop()
    assert "handler broke" in caplog.text
```

New `tests/test_remote.py`:

```python
"""Tests for vocalis.remote — the `vocalis --settings` client. Real sockets only in tmp_path."""
import socket
import threading
from pathlib import Path

from vocalis.remote import SHOW_SETTINGS, open_settings, send_command


def _serve_once(path: Path, replies: list[bytes]) -> threading.Thread:
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    server.listen(1)

    def run() -> None:
        conn, _ = server.accept()
        replies.append(conn.recv(1024))
        conn.sendall(b"ack:show-settings\n")
        conn.close()
        server.close()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def test_send_command_delivers_and_reads_the_ack(tmp_path: Path) -> None:
    path, got = tmp_path / "v.sock", []
    thread = _serve_once(path, got)
    assert send_command(path, SHOW_SETTINGS) is True
    thread.join(2)
    assert got == [b"show-settings\n"]


def test_send_command_to_nothing_returns_false(tmp_path: Path) -> None:
    assert send_command(tmp_path / "missing.sock", SHOW_SETTINGS) is False


def test_open_settings_when_running_does_not_start_the_service(tmp_path: Path) -> None:
    started: list[int] = []
    code = open_settings(tmp_path / "v.sock", start_service=lambda: started.append(1) or True,
                         send=lambda _p, _c: True, sleep=lambda _s: None)
    assert code == 0 and started == []


def test_open_settings_starts_the_service_then_retries_until_it_answers(tmp_path: Path) -> None:
    answers = iter([False, False, False, True])
    started: list[int] = []
    code = open_settings(tmp_path / "v.sock", start_service=lambda: started.append(1) or True,
                         send=lambda _p, _c: next(answers), sleep=lambda _s: None)
    assert code == 0 and started == [1]


def test_open_settings_gives_up_after_the_wait_limit_with_a_plain_message(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    slept: list[float] = []
    code = open_settings(tmp_path / "v.sock", start_service=lambda: True, send=lambda _p, _c: False,
                         sleep=slept.append, wait_seconds=2.0, poll_seconds=0.5)
    assert code == 1
    assert sum(slept) <= 2.0
    assert "Vocalis did not start" in capsys.readouterr().err


def test_open_settings_reports_a_failed_service_start(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    code = open_settings(tmp_path / "v.sock", start_service=lambda: False,
                         send=lambda _p, _c: False, sleep=lambda _s: None)
    assert code == 1
    assert "could not start" in capsys.readouterr().err.lower()
```

`tests/test_cli.py` (add `entrypoint` to the `from vocalis.cli import ...` line):

```python
def test_entrypoint_without_flags_runs_main() -> None:
    with patch("vocalis.cli.main") as fake_main:
        entrypoint([])
    fake_main.assert_called_once_with()


def test_entrypoint_settings_flag_opens_settings_and_exits_with_its_code() -> None:
    with patch("vocalis.cli.open_settings", return_value=0) as fake_open, patch("vocalis.cli.main") as fake_main:
        with pytest.raises(SystemExit) as exited:
            entrypoint(["--settings"])
    assert exited.value.code == 0
    fake_open.assert_called_once()
    fake_main.assert_not_called()


def test_build_daemon_passes_the_command_handler(tmp_path: Path) -> None:
    handler = MagicMock()
    daemon = build_daemon(model_manager=ModelManager(), socket_path=tmp_path / "s.sock", command_handler=handler)
    assert daemon._command_handler is handler  # noqa: SLF001
```

- [ ] **Step 2: Run them, expect FAIL.**
- [ ] **Step 3: Implement.**
  - `SettingsWindow.present()`:

```python
    def present(self) -> None:
        """Bring the window to the front; refresh it from saved settings only if it was hidden."""
        if not self.isVisible():
            self.load_from_settings()
        self.show()
        self.raise_()
        self.activateWindow()
```

  - `build_ui`: replace the body of `open_settings()` with `settings_window.present()`.
  - `tray.py` `__init__`, after the existing `connect` lines: `self._tray.activated.connect(self._on_activated)` and store `self._on_open_settings = on_open_settings`; add

```python
    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        """A plain left-click on the icon opens Settings; other clicks keep their default."""
        if self._closed:
            return
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self._on_open_settings()
```

  and disconnect `self._tray.activated` in `close()` (try/except like the hub signals).
  - `daemon.py`: add the `command_handler` parameter (documented in the docstring), store it as `self._command_handler`; in `_handle_connection`, after logging the command and BEFORE sending the ack:

```python
        if self._command_handler is not None:
            try:
                self._command_handler(command)
            except Exception:
                self._logger.exception("Command handler failed for %r", command)
```

  Update the module and `_handle_connection` docstrings (they still say "real dispatch is added by later phases" — say the socket answers every line with `ack:` and passes it to the optional handler).
  - `src/vocalis/remote.py`:

```python
"""The client side of `vocalis --settings`: ask the running Vocalis to show its
Settings window, starting the background service first if it is not running."""
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

SHOW_SETTINGS = "show-settings"
_START_TIMEOUT_SECONDS = 10.0


def _unix_socket() -> socket.socket:
    """A fresh AF_UNIX stream socket."""
    return socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)


def send_command(socket_path: Path, command: str, timeout: float = 2.0,
                 socket_factory: Callable[[], socket.socket] = _unix_socket) -> bool:
    """Send one command line; True only if the daemon answered with its ack."""
    client = socket_factory()
    try:
        client.settimeout(timeout)
        client.connect(str(socket_path))
        client.sendall(f"{command}\n".encode("utf-8"))
        return client.recv(1024).decode("utf-8", errors="replace").strip() == f"ack:{command}"
    except OSError:
        return False
    finally:
        client.close()


def start_user_service() -> bool:
    """Start the Vocalis user service; True if systemctl reported success."""
    try:
        result = subprocess.run(["systemctl", "--user", "start", "vocalis"],
                                capture_output=True, timeout=_START_TIMEOUT_SECONDS, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def open_settings(socket_path: Path, start_service: Callable[[], bool] = start_user_service,
                  sleep: Callable[[float], None] = time.sleep, wait_seconds: float = 20.0,
                  poll_seconds: float = 0.5,
                  send: Callable[[Path, str], bool] = send_command) -> int:
    """Show Settings in the running Vocalis; start it if needed. Returns an exit code."""
    if send(socket_path, SHOW_SETTINGS):
        return 0
    if not start_service():
        print("Vocalis is not running and could not start it. "
              "Try: systemctl --user start vocalis", file=sys.stderr)
        return 1
    waited = 0.0
    while waited < wait_seconds:
        sleep(poll_seconds)
        waited += poll_seconds
        if send(socket_path, SHOW_SETTINGS):
            return 0
    print(f"Vocalis did not start within {int(wait_seconds)} seconds. "
          "Check: systemctl --user status vocalis", file=sys.stderr)
    return 1
```

  - `cli.py`: `build_daemon` gains `command_handler: Callable[[str], None] | None = None` and passes it through. Add

```python
class _SettingsBridge(QObject):
    """Carries 'show Settings' requests from the socket thread to the Qt thread."""

    show_settings = Signal()

    def handle_command(self, command: str) -> None:
        """Called on the daemon's socket thread for every received command."""
        if command == SHOW_SETTINGS:
            self.show_settings.emit()
```

  (imports: `from PySide6.QtCore import QObject, QTimer, Signal`, `from vocalis.remote import SHOW_SETTINGS, open_settings`). In `main()`: create `bridge = _SettingsBridge()` right after `hub = StatusHub()`; pass `command_handler=bridge.handle_command` to `build_daemon`; after `build_ui(...)` add `bridge.show_settings.connect(settings_window.present)`. Add

```python
def entrypoint(argv: "list[str] | None" = None) -> None:
    """The `vocalis` console command: run the app, or with --settings open its window."""
    parser = argparse.ArgumentParser(prog="vocalis", description="Offline voice AI workstation.")
    parser.add_argument("--settings", action="store_true",
                        help="open the Settings window of the running Vocalis (starting it if needed)")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.settings:
        sys.exit(open_settings(_socket_path()))
    main()
```

  (import `argparse`). `pyproject.toml`: `vocalis = "vocalis.cli:entrypoint"`.
- [ ] **Step 4: Run the five test files, full suite, mypy → green.**
- [ ] **Step 5: Commit** — `git commit -m "feat: open Settings from a tray click or 'vocalis --settings'"`

---

### Task 5: Application-menu launcher, installer message, docs, changelog

**Files:**
- Create: `packaging/desktop/vocalis.desktop` (template)
- Create: `packaging/installer/lib/desktop.sh`
- Modify: `packaging/installer/install.sh` (source desktop.sh, install launcher, new closing message)
- Modify: `packaging/installer/uninstall.sh` (remove launcher)
- Modify: `packaging/installer/build-run.sh` (stage `packaging/desktop/vocalis.desktop`)
- Test: `tests/installer/test_desktop.bats` (new), `tests/installer/test_install.bats`, `tests/installer/test_uninstall.bats`, `tests/installer/test_build_run.bats`, `tests/installer/fixtures/vocalis.desktop` (new)
- Docs: `docs/user-docs.md`, `docs/file-purpose.md`, `docs/folder-structure.md`, `docs/architecture.md`, `docs/workflow.md`, `docs/developer-docs.md`, `packaging/installer/README.md`; changelog `.claude/2026-10-15_<HHMM>_CHANGELOG.md`

**Interfaces:** Consumes `vocalis --settings` (Task 4). Produces `desktop_install_launcher <template> <exec_path> <output_path>`.

- [ ] **Step 1: Template** `packaging/desktop/vocalis.desktop`:

```ini
[Desktop Entry]
Type=Application
Name=Vocalis
Comment=Offline voice dictation and read-aloud
Exec=__VOCALIS_EXEC__ --settings
Icon=audio-input-microphone
Terminal=false
Categories=Utility;Accessibility;AudioVideo;
```

Copy it to `tests/installer/fixtures/vocalis.desktop`.
- [ ] **Step 2: Failing bats test** `tests/installer/test_desktop.bats`:

```bash
#!/usr/bin/env bats

setup() {
  source "${BATS_TEST_DIRNAME}/../../packaging/installer/lib/desktop.sh"
  WORKDIR="$(mktemp -d)"
  FIXTURE="${BATS_TEST_DIRNAME}/fixtures/vocalis.desktop"
}

teardown() { rm -rf "$WORKDIR"; }

@test "desktop_install_launcher fills in the command and creates the folder" {
  local out="$WORKDIR/share/applications/vocalis.desktop"
  desktop_install_launcher "$FIXTURE" "/home/ex/.local/bin/vocalis" "$out"
  grep -q '^Exec="/home/ex/.local/bin/vocalis" --settings$' "$out"
  ! grep -q "__VOCALIS_EXEC__" "$out"
}

@test "desktop_install_launcher quotes a path that contains spaces" {
  local out="$WORKDIR/vocalis.desktop"
  desktop_install_launcher "$FIXTURE" "/home/my user/.local/bin/vocalis" "$out"
  grep -q '^Exec="/home/my user/.local/bin/vocalis" --settings$' "$out"
}

@test "desktop_install_launcher leaves an existing launcher untouched on failure" {
  local out="$WORKDIR/vocalis.desktop"
  echo "old" > "$out"
  run desktop_install_launcher "$WORKDIR/missing.desktop" "/x/vocalis" "$out"
  [ "$status" -ne 0 ]
  [ "$(cat "$out")" = "old" ]
  [ "$(find "$WORKDIR" -mindepth 1 | wc -l)" -eq 1 ]
}
```

- [ ] **Step 3: Implement `lib/desktop.sh`** (mirror `service_substitute_unit`'s atomic temp-file pattern; the exec path is wrapped in double quotes, and `|`, `&`, `\` in the path are escaped for sed):

```bash
#!/usr/bin/env bash
# Installs the "Vocalis" entry in the desktop's application menu.
set -uo pipefail

# Write output_path from the template with __VOCALIS_EXEC__ replaced by the
# quoted exec_path. Atomic: a failure never leaves a broken launcher behind.
desktop_install_launcher() {
  local template_path="$1"
  local exec_path="$2"
  local output_path="$3"
  local output_dir tmp_path escaped
  output_dir="$(dirname "$output_path")"
  mkdir -p "$output_dir" || return 1
  escaped="$(printf '%s' "\"${exec_path}\"" | sed -e 's/[\\|&]/\\&/g')"
  tmp_path="$(mktemp "${output_dir}/.vocalis.desktop.XXXXXX")" || return 1
  if ! sed -e "s|__VOCALIS_EXEC__|${escaped}|g" "$template_path" > "$tmp_path" \
    || ! chmod 644 "$tmp_path" \
    || ! mv -f "$tmp_path" "$output_path"; then
    rm -f "$tmp_path"
    echo "Could not create the Vocalis menu entry at ${output_path}." >&2
    return 1
  fi
}
```

- [ ] **Step 4: Wire `install.sh`.** Source `lib/desktop.sh` with the same shellcheck comment pattern as the others. After `service_enable_and_start`, locate the template like the unit file (`${script_dir}/packaging/desktop/vocalis.desktop`, else `${repo_root}/packaging/desktop/vocalis.desktop`) and call `desktop_install_launcher "$desktop_template" "${HOME}/.local/bin/vocalis" "${HOME}/.local/share/applications/vocalis.desktop" || echo "(The menu entry could not be added; Vocalis still works.)"`. Replace the closing echo lines with:

```bash
  echo ""
  echo "Vocalis is installed and running."
  echo "It will start automatically the next time you log in."
  echo "To open it: click the small round Vocalis icon in your panel's system tray,"
  echo "or choose \"Vocalis\" in your applications menu, or run: vocalis --settings"
```

In `tests/installer/test_install.bats` `setup()` add the stub `desktop_install_launcher() { echo "desktop_install_launcher: $*" >> "$CALL_LOG"; }` and add:

```bash
@test "main installs the menu launcher and tells the user how to open Vocalis" {
  SCRIPT_DIR="$WORKDIR" REPO_ROOT="$WORKDIR" HOME="$WORKDIR" run main
  [ "$status" -eq 0 ]
  grep -q "desktop_install_launcher: .* $WORKDIR/.local/bin/vocalis $WORKDIR/.local/share/applications/vocalis.desktop" "$CALL_LOG"
  [[ "$output" == *"vocalis --settings"* ]]
}

@test "a failing menu launcher does not fail the whole install" {
  desktop_install_launcher() { return 1; }
  SCRIPT_DIR="$WORKDIR" REPO_ROOT="$WORKDIR" HOME="$WORKDIR" run main
  [ "$status" -eq 0 ]
  [[ "$output" == *"menu entry could not be added"* ]]
}
```

- [ ] **Step 5: `uninstall.sh`**: add `local launcher="${HOME}/.local/share/applications/vocalis.desktop"` and, next to the symlink removal, `if [[ -e "$launcher" ]]; then found_anything=1; rm -f "$launcher"; fi`. Add to `tests/installer/test_uninstall.bats`:

```bash
@test "main removes the application-menu launcher" {
  mkdir -p "$WORKDIR/.local/share/applications"
  touch "$WORKDIR/.local/share/applications/vocalis.desktop"
  HOME="$WORKDIR" run main
  [ "$status" -eq 0 ]
  [ ! -e "$WORKDIR/.local/share/applications/vocalis.desktop" ]
}
```

- [ ] **Step 6: `build-run.sh`**: after the systemd copy add `mkdir -p "$staging_dir/packaging/desktop"` and `cp "${repo_root}/packaging/desktop/vocalis.desktop" "$staging_dir/packaging/desktop/"`. In `tests/installer/test_build_run.bats` `setup()` create `mkdir -p "$WORKDIR/packaging/desktop"` and `touch "$WORKDIR/packaging/desktop/vocalis.desktop"`, and add a test whose `build_run_run_makeself` stub records `[ -f "$2/packaging/desktop/vocalis.desktop" ] && echo DESKTOP_OK >> "$CALL_LOG"`, asserting `DESKTOP_OK` appears twice.
- [ ] **Step 7: Run** `shellcheck packaging/installer/*.sh packaging/installer/lib/*.sh` and `bats tests/installer` → all green, zero warnings.
- [ ] **Step 8: Docs.**
  - `docs/user-docs.md`: new section "## How to open Vocalis" near the top (tray click, application menu "Vocalis", `vocalis --settings`; what the "old Voxtype sound settings" and "No sound was recorded" messages mean and what to do).
  - `docs/file-purpose.md`: rows for `src/vocalis/hardware.py`, `src/vocalis/remote.py`, `packaging/desktop/vocalis.desktop`, `packaging/installer/lib/desktop.sh`, `LICENSE`, `tests/test_hardware.py`, `tests/test_remote.py`, `tests/test_package_metadata.py`, `tests/installer/test_desktop.bats`; update the startup_checks row (five checks).
  - `docs/folder-structure.md`: add `LICENSE`, `packaging/desktop/`, the new lib file and tests.
  - `docs/architecture.md`: short "Stage 5 addition" section (daemon command hook + `remote.py`, tray click, hardware probe).
  - `docs/workflow.md`: line 93 text -> new no-audio message; add the `vocalis --settings` flow.
  - `docs/developer-docs.md`: startup-check order list now ends with the `.asoundrc` check; console script is `vocalis.cli:entrypoint`.
  - `packaging/installer/README.md`: mention the menu launcher and how to open Vocalis.
  - Changelog `.claude/2026-10-15_<HHMM>_CHANGELOG.md` (same header/sections as `.claude/2026-10-09_1004_CHANGELOG.md`).
- [ ] **Step 9: Full gates** — pytest (>=85%, zero warnings), mypy, shellcheck, bats.
- [ ] **Step 10: Commit** — `git commit -m "feat: add a Vocalis application-menu launcher and document Stage 5"`

---

## Self-review notes
- Spec coverage: license (T1), hardware probe + log (T2), asoundrc + no-audio message (T3), tray/present/daemon hook/`--settings` with service start (T4), `.desktop` + installer message + uninstall + docs/changelog (T5). Mic device logging already exists (`capture.py:118`).
- Review Focus mapping: spaces in HOME -> T5 bats; `--settings` failure paths -> T4 test_remote; unreadable asoundrc -> T3; probe failures -> T2; present() twice -> T4 settings-window tests.
