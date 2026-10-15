# Vocalis Phase 5 — OCR Reader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Press Ctrl+Alt+O, drag (and resize) a box around text on screen, and hear that text read aloud; the text is also copied to the clipboard.

**Architecture:** A thin `OcrReaderFeature` runs capture -> Tesseract -> clipboard -> notification on a background thread, then hands the text to the existing `DocumentReaderFeature.read_text()`, which already provides synthesis with fallback, chunking, pause/resume/stop and `ReadingCoordinator` ownership. Capture, OCR and clipboard-write are three small injectable classes (command-runner seams, no real tools in tests). The Document Reader popup gains one signal so it can show the recognised text with Pause/Stop enabled.

**Tech Stack:** Python 3.13, system `tesseract` (CPU only, never via `ModelManager`), system `flameshot` / `maim` / `slurp`+`grim`, `xclip`, PySide6 (existing popup), pytest + pytest-qt, mypy strict.

**Spec:** `docs/superpowers/specs/2026-09-11-vocalis-design.md` — section "Phase 5 amendment". Deviations from that section, decided while planning: (1) the picture is passed to Tesseract in memory (stdout of the capture tool -> stdin of tesseract), so there is no temporary file to clean up; (2) the separate "normalise" step is dropped because `DocumentReaderFeature.read_text()` already runs `normalize_markdown()`; the OCR text only has its whitespace collapsed (like the old script's `tr -s`); (3) to make "Pause" reachable, the popup window is shown with Pause/Stop enabled whenever an OCR reading starts.

## Global Constraints

- Work in a fresh git worktree (superpowers:using-git-worktrees) on a branch named `worktree-vocalis-phase5-ocr-reader`; use the project `.venv` (`.venv/bin/pytest`, `.venv/bin/mypy`).
- Conventional commits (`feat:`, `fix:`, `test:`, `docs:`), **no Co-Authored-By or attribution lines**. Commit after every task. Never touch `old/`.
- Every function, method and class gets a docstring (purpose-driven, concise).
- `mypy` strict: **zero errors project-wide** (`.venv/bin/mypy` prints "Success: no issues found"). Full suite **zero warnings**, coverage **>= 85%**.
- No shell strings: all subprocesses use argv lists. Never hold a lock across a blocking call (subprocess, notify, audio).
- Tests are deterministic: use `threading.Event` and a bounded `wait_until` that FAILS on timeout; no guessed sleeps; never weaken or delete an existing assertion.
- Models unchanged: STT Parakeet TDT 0.6B v3 int8; TTS Qwen3-TTS 0.6B CustomVoice ("Ryan") with Piper `en_US-lessac-medium` fallback; OCR = system Tesseract, English (`-l eng`) only.
- Shortcut config `ShortcutsConfig.ocr_reader = "Ctrl+Alt+O"` already exists; evdev key code for O is **24**.
- Record the change log in `.claude/<YYYY-MM-DD_HHMM>_CHANGELOG.md` and keep `docs/` in sync with the code.

## Review Focus

Inputs or conditions the spec implies but no happy-path test covers; each has its own test in the task that owns the code:

1. Pressing Ctrl+Alt+O again **while the box is still being drawn** must be ignored, not start a second capture (Task 5).
2. Pressing Escape / closing the capture tool must show "Cancelled" and leave the feature usable for the next press (Tasks 3, 5).
3. A box over a blank or tiny area (OCR returns "" or 1-2 characters) must say "No text detected" and read nothing (Task 5).
4. `slurp` printing something that is not a plain `x,y WxH` geometry must be rejected, never passed on to `grim` (Task 3).
5. `xclip` missing or failing must not stop the reading (Tasks 1, 5).
6. A whole screen of text (very long result) must be copied in full and handed over unmodified for chunked reading (Task 5).
7. Ctrl+Alt+O while the Document Reader is reading a **document it opened** must start a new capture, not stop that reading (Task 5).

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/vocalis/output/clipboard.py` | modify | add `ClipboardWriter` next to `ClipboardReader` |
| `src/vocalis/engines/tesseract_ocr.py` | create | `TesseractOCR.recognize(png_bytes) -> str`, `TesseractError` |
| `src/vocalis/output/screen_capture.py` | create | `ScreenRegionCapture.capture() -> bytes \| None`, `ScreenCaptureUnavailable` |
| `src/vocalis/features/doc_reader.py` | modify | add read-only `generation` property |
| `src/vocalis/ui/doc_popup.py` | modify | add `external_reading_started` signal + slot |
| `src/vocalis/features/ocr_reader.py` | create | `OcrReaderFeature` orchestration |
| `src/vocalis/cli.py` | modify | `build_ocr_reader()`, start/stop the Ctrl+Alt+O listener |
| `tests/test_clipboard.py`, `tests/test_tesseract_ocr.py`, `tests/test_screen_capture.py`, `tests/test_doc_reader.py`, `tests/test_doc_popup.py`, `tests/test_ocr_reader.py`, `tests/test_cli.py` | create/modify | matching tests |
| `docs/*.md`, `.claude/..._CHANGELOG.md` | modify/create | documentation + change log |

---

### Task 1: ClipboardWriter

**Files:**
- Modify: `src/vocalis/output/clipboard.py`
- Test: `tests/test_clipboard.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `ClipboardWriter(command_runner: InputCommandRunner | None = None)` with `write(self, text: str) -> bool` (True on success, False on any failure, never raises).

- [ ] **Step 1: Write the failing tests** — append to `tests/test_clipboard.py` (update the import line to `from vocalis.output.clipboard import ClipboardReader, ClipboardWriter`):

```python
class FakeInputRunner:
    def __init__(self, returncode: int = 0, raise_exc: Exception | None = None) -> None:
        self.returncode = returncode
        self.raise_exc = raise_exc
        self.calls: "list[tuple[list[str], bytes | None]]" = []

    def __call__(
        self, argv: list[str], input_bytes: bytes | None = None
    ) -> "subprocess.CompletedProcess[bytes]":
        self.calls.append((argv, input_bytes))
        if self.raise_exc is not None:
            raise self.raise_exc
        return subprocess.CompletedProcess(args=argv, returncode=self.returncode)


def test_write_sends_text_to_xclip_and_returns_true() -> None:
    runner = FakeInputRunner()
    assert ClipboardWriter(command_runner=runner).write("héllo") is True
    assert runner.calls == [(["xclip", "-selection", "clipboard"], "héllo".encode("utf-8"))]


def test_write_returns_false_on_nonzero_exit() -> None:
    assert ClipboardWriter(command_runner=FakeInputRunner(returncode=1)).write("x") is False


def test_write_returns_false_when_xclip_is_missing() -> None:
    runner = FakeInputRunner(raise_exc=FileNotFoundError("xclip"))
    assert ClipboardWriter(command_runner=runner).write("x") is False


def test_write_returns_false_on_timeout() -> None:
    runner = FakeInputRunner(raise_exc=subprocess.TimeoutExpired(cmd="xclip", timeout=5.0))
    assert ClipboardWriter(command_runner=runner).write("x") is False
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_clipboard.py -q --no-cov`
Expected: FAIL — `ImportError: cannot import name 'ClipboardWriter'`.

- [ ] **Step 3: Implement** — in `src/vocalis/output/clipboard.py` add this after `ClipboardReader` and before `_default_command_runner` (and add the module docstring sentence "ClipboardWriter is the write-side counterpart used by the OCR Reader."):

```python
class InputCommandRunner(Protocol):
    """Runs a command, optionally piping bytes to its stdin."""

    def __call__(
        self, argv: list[str], input_bytes: bytes | None = None
    ) -> "subprocess.CompletedProcess[bytes]": ...


class ClipboardWriter:
    """Puts text on the clipboard via xclip."""

    def __init__(self, command_runner: InputCommandRunner | None = None) -> None:
        """Create a ClipboardWriter.

        Args:
            command_runner: Callable(argv, input_bytes) -> CompletedProcess,
                used to run xclip. Defaults to subprocess.run with a
                5-second timeout. Injectable for tests.
        """
        self._run: InputCommandRunner = command_runner or _default_input_runner

    def write(self, text: str) -> bool:
        """Copy `text` to the clipboard.

        Returns:
            True if xclip accepted the text; False on any failure
            (missing binary, timeout, non-zero exit). Never raises.
        """
        try:
            result = self._run(["xclip", "-selection", "clipboard"], input_bytes=text.encode("utf-8"))
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning("Failed to write clipboard: %s", exc)
            return False
        if result.returncode != 0:
            logger.warning("xclip exited with code %d", result.returncode)
            return False
        return True
```

and at the bottom of the file:

```python
def _default_input_runner(
    argv: list[str], input_bytes: bytes | None = None
) -> "subprocess.CompletedProcess[bytes]":
    """Default InputCommandRunner: subprocess.run with a 5-second timeout.

    stdout/stderr go to DEVNULL (not pipes) because xclip stays alive in
    the background to serve the selection; holding its output pipes open
    would make subprocess.run wait for it until the timeout.
    """
    return subprocess.run(
        argv,
        input=input_bytes,
        timeout=5.0,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_clipboard.py -q --no-cov && .venv/bin/mypy`
Expected: all pass, mypy `Success`.

- [ ] **Step 5: Commit**

```bash
git add src/vocalis/output/clipboard.py tests/test_clipboard.py
git commit -m "feat: add ClipboardWriter for copying text to the clipboard"
```

---

### Task 2: TesseractOCR engine

**Files:**
- Create: `src/vocalis/engines/tesseract_ocr.py`
- Test: `tests/test_tesseract_ocr.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `TesseractError(RuntimeError)` (message is user-readable); `TesseractOCR(binary: str = "tesseract", language: str = "eng", command_runner: OcrCommandRunner | None = None)` with `recognize(self, png_bytes: bytes) -> str` returning whitespace-collapsed text (`""` if nothing recognised).

- [ ] **Step 1: Write the failing tests** — create `tests/test_tesseract_ocr.py`:

```python
"""Tests for vocalis.engines.tesseract_ocr.TesseractOCR (fake runner, no real tesseract)."""
import subprocess

import pytest

from vocalis.engines.tesseract_ocr import TesseractError, TesseractOCR


class FakeRunner:
    def __init__(
        self,
        stdout: bytes = b"",
        returncode: int = 0,
        stderr: bytes = b"",
        raise_exc: Exception | None = None,
    ) -> None:
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr
        self.raise_exc = raise_exc
        self.calls: "list[tuple[list[str], bytes]]" = []

    def __call__(self, argv: list[str], input_bytes: bytes) -> "subprocess.CompletedProcess[bytes]":
        self.calls.append((argv, input_bytes))
        if self.raise_exc is not None:
            raise self.raise_exc
        return subprocess.CompletedProcess(
            args=argv, returncode=self.returncode, stdout=self.stdout, stderr=self.stderr
        )


def test_recognize_pipes_the_picture_to_tesseract_and_returns_text() -> None:
    runner = FakeRunner(stdout=b"Hello world\n")
    text = TesseractOCR(command_runner=runner).recognize(b"PNGDATA")
    assert text == "Hello world"
    assert runner.calls == [(["tesseract", "stdin", "stdout", "-l", "eng"], b"PNGDATA")]


def test_recognize_collapses_line_breaks_and_extra_spaces() -> None:
    runner = FakeRunner(stdout=b"  one  two\n\nthree\t four \n")
    assert TesseractOCR(command_runner=runner).recognize(b"x") == "one two three four"


def test_recognize_returns_empty_string_when_nothing_is_found() -> None:
    assert TesseractOCR(command_runner=FakeRunner(stdout=b" \n\n")).recognize(b"x") == ""


def test_missing_binary_gives_an_install_hint() -> None:
    runner = FakeRunner(raise_exc=FileNotFoundError("tesseract"))
    with pytest.raises(TesseractError, match="not installed"):
        TesseractOCR(command_runner=runner).recognize(b"x")


def test_timeout_is_reported() -> None:
    runner = FakeRunner(raise_exc=subprocess.TimeoutExpired(cmd="tesseract", timeout=60.0))
    with pytest.raises(TesseractError, match="too long"):
        TesseractOCR(command_runner=runner).recognize(b"x")


def test_other_os_error_is_reported() -> None:
    runner = FakeRunner(raise_exc=PermissionError("denied"))
    with pytest.raises(TesseractError, match="Could not run Tesseract"):
        TesseractOCR(command_runner=runner).recognize(b"x")


def test_nonzero_exit_is_reported() -> None:
    runner = FakeRunner(returncode=1, stderr=b"Error: bad image")
    with pytest.raises(TesseractError, match="could not read"):
        TesseractOCR(command_runner=runner).recognize(b"x")


def test_custom_binary_and_language_are_used() -> None:
    runner = FakeRunner(stdout=b"hi")
    TesseractOCR(binary="/opt/tess", language="deu", command_runner=runner).recognize(b"x")
    assert runner.calls[0][0] == ["/opt/tess", "stdin", "stdout", "-l", "deu"]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_tesseract_ocr.py -q --no-cov`
Expected: FAIL — `ModuleNotFoundError: vocalis.engines.tesseract_ocr`.

- [ ] **Step 3: Implement** — create `src/vocalis/engines/tesseract_ocr.py`:

```python
"""Tesseract OCR engine — reads text out of a picture for the OCR Reader.

Runs the system `tesseract` binary on the CPU, feeding the picture on
stdin and reading the text from stdout. It never touches the GPU, so it
is deliberately NOT routed through ModelManager (single-residency only
applies to GPU engines).
"""
import logging
import subprocess
from typing import Protocol

logger = logging.getLogger("vocalis")

_OCR_TIMEOUT_SECONDS = 60.0


class TesseractError(RuntimeError):
    """Tesseract could not be run or could not read the picture.

    The message is written for the end user (it is shown in a
    notification as-is).
    """


class OcrCommandRunner(Protocol):
    """Runs a command with bytes piped to its stdin."""

    def __call__(self, argv: list[str], input_bytes: bytes) -> "subprocess.CompletedProcess[bytes]": ...


class TesseractOCR:
    """Recognises text in a PNG picture using the tesseract binary."""

    def __init__(
        self,
        binary: str = "tesseract",
        language: str = "eng",
        command_runner: OcrCommandRunner | None = None,
    ) -> None:
        """Create a TesseractOCR.

        Args:
            binary: Name or path of the tesseract executable.
            language: Tesseract language pack to use (only "eng" is installed).
            command_runner: Callable(argv, input_bytes) -> CompletedProcess.
                Defaults to subprocess.run with a 60-second timeout.
                Injectable for tests.
        """
        self._binary = binary
        self._language = language
        self._run: OcrCommandRunner = command_runner or _default_command_runner

    def recognize(self, png_bytes: bytes) -> str:
        """Read the text in a picture.

        Args:
            png_bytes: The picture, PNG-encoded.

        Returns:
            The recognised text with all runs of whitespace (including line
            breaks) collapsed to single spaces; "" if nothing was found.

        Raises:
            TesseractError: Tesseract is missing, timed out, or failed.
        """
        argv = [self._binary, "stdin", "stdout", "-l", self._language]
        try:
            result = self._run(argv, png_bytes)
        except FileNotFoundError as exc:
            logger.error("tesseract binary not found: %s", exc)
            raise TesseractError(
                "Tesseract is not installed. Install it with: sudo apt install tesseract-ocr tesseract-ocr-eng"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            logger.error("tesseract timed out")
            raise TesseractError("Tesseract took too long to read the picture") from exc
        except OSError as exc:
            logger.error("Failed to run tesseract: %s", exc)
            raise TesseractError(f"Could not run Tesseract: {exc}") from exc
        if result.returncode != 0:
            logger.error("tesseract exited with code %d: %r", result.returncode, result.stderr)
            raise TesseractError("Tesseract could not read the picture")
        return " ".join(result.stdout.decode("utf-8", errors="replace").split())


def _default_command_runner(argv: list[str], input_bytes: bytes) -> "subprocess.CompletedProcess[bytes]":
    """Default OcrCommandRunner: subprocess.run with a 60-second timeout."""
    return subprocess.run(argv, input=input_bytes, timeout=_OCR_TIMEOUT_SECONDS, capture_output=True)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_tesseract_ocr.py -q --no-cov && .venv/bin/mypy`
Expected: 8 passed, mypy `Success`.

- [ ] **Step 5: Commit**

```bash
git add src/vocalis/engines/tesseract_ocr.py tests/test_tesseract_ocr.py
git commit -m "feat: add Tesseract OCR engine"
```

---

### Task 3: ScreenRegionCapture

**Files:**
- Create: `src/vocalis/output/screen_capture.py`
- Test: `tests/test_screen_capture.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `ScreenCaptureUnavailable(RuntimeError)` (user-readable message); `ScreenRegionCapture(command_runner: CaptureRunner | None = None, which: Callable[[str], str | None] = shutil.which)` with `capture(self) -> bytes | None` — PNG bytes of the chosen region, or `None` when the user cancelled.

Tool order (first installed wins, decided once per call, no re-try after the tool was started): `flameshot gui --raw` (box can be resized before confirming) -> `maim -s -u` -> `slurp` then `grim -g <geometry> -`.

- [ ] **Step 1: Write the failing tests** — create `tests/test_screen_capture.py`:

```python
"""Tests for vocalis.output.screen_capture.ScreenRegionCapture (fake tools, no screen)."""
import subprocess
from typing import Callable

import pytest

from vocalis.output.screen_capture import ScreenCaptureUnavailable, ScreenRegionCapture

PNG = b"\x89PNG\r\n\x1a\n" + b"fakeimagebytes"


class FakeRunner:
    """Answers each argv from a dict keyed by the tool name (argv[0])."""

    def __init__(self, answers: "dict[str, subprocess.CompletedProcess[bytes] | Exception]") -> None:
        self.answers = answers
        self.calls: "list[list[str]]" = []

    def __call__(self, argv: list[str]) -> "subprocess.CompletedProcess[bytes]":
        self.calls.append(argv)
        answer = self.answers[argv[0]]
        if isinstance(answer, Exception):
            raise answer
        return answer


def done(stdout: bytes = b"", returncode: int = 0) -> "subprocess.CompletedProcess[bytes]":
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout)


def installed(*names: str) -> Callable[[str], str | None]:
    return lambda name: f"/usr/bin/{name}" if name in names else None


def test_flameshot_is_preferred_because_its_box_can_be_resized() -> None:
    runner = FakeRunner({"flameshot": done(PNG), "maim": done(PNG)})
    capture = ScreenRegionCapture(command_runner=runner, which=installed("flameshot", "maim"))
    assert capture.capture() == PNG
    assert runner.calls == [["flameshot", "gui", "--raw"]]


def test_maim_is_used_when_flameshot_is_missing() -> None:
    runner = FakeRunner({"maim": done(PNG)})
    capture = ScreenRegionCapture(command_runner=runner, which=installed("maim"))
    assert capture.capture() == PNG
    assert runner.calls == [["maim", "-s", "-u"]]


def test_slurp_then_grim_is_used_when_nothing_else_is_installed() -> None:
    runner = FakeRunner({"slurp": done(b"10,20 300x400\n"), "grim": done(PNG)})
    capture = ScreenRegionCapture(command_runner=runner, which=installed("slurp", "grim"))
    assert capture.capture() == PNG
    assert runner.calls == [["slurp"], ["grim", "-g", "10,20 300x400", "-"]]


def test_slurp_alone_is_not_enough() -> None:
    capture = ScreenRegionCapture(command_runner=FakeRunner({}), which=installed("slurp"))
    with pytest.raises(ScreenCaptureUnavailable):
        capture.capture()


def test_no_tool_at_all_raises_with_an_install_hint() -> None:
    capture = ScreenRegionCapture(command_runner=FakeRunner({}), which=installed())
    with pytest.raises(ScreenCaptureUnavailable, match="flameshot"):
        capture.capture()


def test_escape_in_the_tool_means_cancelled() -> None:
    runner = FakeRunner({"flameshot": done(b"", returncode=1)})
    assert ScreenRegionCapture(command_runner=runner, which=installed("flameshot")).capture() is None


def test_exit_zero_with_empty_output_means_cancelled() -> None:
    runner = FakeRunner({"flameshot": done(b"")})
    assert ScreenRegionCapture(command_runner=runner, which=installed("flameshot")).capture() is None


def test_output_that_is_not_a_png_means_cancelled() -> None:
    runner = FakeRunner({"maim": done(b"not a picture")})
    assert ScreenRegionCapture(command_runner=runner, which=installed("maim")).capture() is None


def test_slurp_cancel_never_starts_grim() -> None:
    runner = FakeRunner({"slurp": done(b"", returncode=1)})
    capture = ScreenRegionCapture(command_runner=runner, which=installed("slurp", "grim"))
    assert capture.capture() is None
    assert runner.calls == [["slurp"]]


@pytest.mark.parametrize("bad", ["; rm -rf ~", "10,20 300x400; id", "-o /tmp/x", "1,2 3x4 5", "abc"])
def test_slurp_output_that_is_not_a_plain_geometry_is_rejected(bad: str) -> None:
    runner = FakeRunner({"slurp": done(bad.encode() + b"\n"), "grim": done(PNG)})
    capture = ScreenRegionCapture(command_runner=runner, which=installed("slurp", "grim"))
    assert capture.capture() is None
    assert runner.calls == [["slurp"]]


def test_a_timeout_in_the_tool_means_cancelled() -> None:
    runner = FakeRunner({"flameshot": subprocess.TimeoutExpired(cmd="flameshot", timeout=120.0)})
    assert ScreenRegionCapture(command_runner=runner, which=installed("flameshot")).capture() is None


def test_an_os_error_launching_the_tool_means_cancelled() -> None:
    runner = FakeRunner({"flameshot": PermissionError("nope")})
    assert ScreenRegionCapture(command_runner=runner, which=installed("flameshot")).capture() is None
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_screen_capture.py -q --no-cov`
Expected: FAIL — `ModuleNotFoundError: vocalis.output.screen_capture`.

- [ ] **Step 3: Implement** — create `src/vocalis/output/screen_capture.py`:

```python
"""Lets the user drag a box on the screen and returns that region as a PNG.

Uses whichever of these tools is installed, in this order:
flameshot (the box can be resized by dragging its edges/corners before
confirming), maim (draw once), or slurp + grim (Wayland, draw once).
The picture is kept in memory (the tool's stdout), never written to disk.
"""
import logging
import re
import shutil
import subprocess
from typing import Callable, Protocol

logger = logging.getLogger("vocalis")

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
# slurp prints "x,y WxH"; anything else is refused before it reaches grim.
_GEOMETRY_RE = re.compile(r"^\d+,\d+ \d+x\d+$")
# The user is dragging a box by hand, so allow plenty of time.
_CAPTURE_TIMEOUT_SECONDS = 120.0


class ScreenCaptureUnavailable(RuntimeError):
    """No supported screen-capture tool is installed (message is user-readable)."""


class CaptureRunner(Protocol):
    """Runs a command and captures its output."""

    def __call__(self, argv: list[str]) -> "subprocess.CompletedProcess[bytes]": ...


class ScreenRegionCapture:
    """Asks the user to drag a box on screen and returns the picture."""

    def __init__(
        self,
        command_runner: CaptureRunner | None = None,
        which: Callable[[str], str | None] = shutil.which,
    ) -> None:
        """Create a ScreenRegionCapture.

        Args:
            command_runner: Callable(argv) -> CompletedProcess. Defaults to
                subprocess.run with a 120-second timeout. Injectable for tests.
            which: Finds an executable on PATH (returns None if absent).
                Injectable for tests.
        """
        self._run: CaptureRunner = command_runner or _default_command_runner
        self._which = which

    def capture(self) -> bytes | None:
        """Let the user choose a screen region and return it.

        Returns:
            PNG bytes of the region, or None if the user cancelled (Escape,
            closed the tool, empty or non-PNG output, tool timed out or failed).

        Raises:
            ScreenCaptureUnavailable: None of the supported tools is installed.
        """
        if self._which("flameshot"):
            return self._png_from(["flameshot", "gui", "--raw"])
        if self._which("maim"):
            return self._png_from(["maim", "-s", "-u"])
        if self._which("slurp") and self._which("grim"):
            return self._slurp_then_grim()
        raise ScreenCaptureUnavailable(
            "No screen-capture tool found. Install one of: flameshot, maim, or slurp + grim"
        )

    def _slurp_then_grim(self) -> bytes | None:
        """Pick a region with slurp, then grab exactly that region with grim."""
        result = self._run_or_none(["slurp"])
        if result is None or result.returncode != 0:
            return None
        geometry = result.stdout.decode("utf-8", errors="replace").strip()
        if not _GEOMETRY_RE.match(geometry):
            logger.warning("slurp printed an unexpected region, ignoring it")
            return None
        return self._png_from(["grim", "-g", geometry, "-"])

    def _png_from(self, argv: list[str]) -> bytes | None:
        """Run a capture command and return its stdout if it is a PNG."""
        result = self._run_or_none(argv)
        if result is None or result.returncode != 0:
            return None
        if not result.stdout.startswith(_PNG_MAGIC):
            return None
        return result.stdout

    def _run_or_none(self, argv: list[str]) -> "subprocess.CompletedProcess[bytes] | None":
        """Run a command; None if it could not be run or timed out."""
        try:
            return self._run(argv)
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning("Screen capture command %s failed: %s", argv[0], exc)
            return None


def _default_command_runner(argv: list[str]) -> "subprocess.CompletedProcess[bytes]":
    """Default CaptureRunner: subprocess.run with a 120-second timeout."""
    return subprocess.run(argv, timeout=_CAPTURE_TIMEOUT_SECONDS, capture_output=True)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_screen_capture.py -q --no-cov && .venv/bin/mypy`
Expected: all pass, mypy `Success`.

- [ ] **Step 5: Commit**

```bash
git add src/vocalis/output/screen_capture.py tests/test_screen_capture.py
git commit -m "feat: add screen region capture (flameshot, maim, slurp+grim)"
```

---

### Task 4: Let the popup show a reading that another feature started

**Files:**
- Modify: `src/vocalis/features/doc_reader.py` (next to `is_reading`, around line 131)
- Modify: `src/vocalis/ui/doc_popup.py`
- Test: `tests/test_doc_reader.py`, `tests/test_doc_popup.py`

**Interfaces:**
- Consumes: existing `DocumentReaderFeature`, `DocumentReaderPopup`.
- Produces: `DocumentReaderFeature.generation -> int` (read-only property; increases by 1 every time a reading starts); `DocumentReaderPopup.external_reading_started = Signal(str)` and slot `on_external_reading_started(text: str) -> None` (puts the text in the box, enables Pause/Stop, shows the window without stealing focus).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_doc_reader.py`:

```python
def test_generation_increases_by_one_for_each_reading() -> None:
    feature, _statuses, _progresses, finishes = _make_feature(
        FakeSynthesizer(), FakeSynthesizer(), RecordingNotifier()
    )
    first = feature.generation

    feature.read_text("Hello there.")
    assert feature.generation == first + 1
    wait_until(lambda: len(finishes) == 1)

    feature.read_text("Hello again.")
    assert feature.generation == first + 2
    wait_until(lambda: len(finishes) == 2)
```

Append to `tests/test_doc_popup.py`:

```python
def test_external_reading_shows_the_text_with_pause_and_stop_enabled(qtbot) -> None:  # type: ignore[no-untyped-def]
    feature = FakeFeature()
    popup = DocumentReaderPopup(feature)
    qtbot.addWidget(popup)

    popup.external_reading_started.emit("Text from the screen")

    assert popup._text_box.toPlainText() == "Text from the screen"  # noqa: SLF001
    assert popup.isVisible() is True
    assert popup._pause_button.isEnabled() is True  # noqa: SLF001
    assert popup._stop_button.isEnabled() is True  # noqa: SLF001
    assert popup._read_button.isEnabled() is False  # noqa: SLF001
    assert feature.read_text_calls == []  # the popup must not start a second reading


def test_external_reading_resets_a_leftover_resume_label(qtbot) -> None:  # type: ignore[no-untyped-def]
    popup = DocumentReaderPopup(FakeFeature())
    qtbot.addWidget(popup)
    popup._paused = True  # noqa: SLF001
    popup._pause_button.setText("Resume")  # noqa: SLF001

    popup.external_reading_started.emit("x")

    assert popup._pause_button.text() == "Pause"  # noqa: SLF001
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_doc_reader.py::test_generation_increases_by_one_for_each_reading tests/test_doc_popup.py -q --no-cov`
Expected: FAIL — `AttributeError: ... 'generation'` and `AttributeError: ... 'external_reading_started'`.

- [ ] **Step 3: Implement**

In `src/vocalis/features/doc_reader.py`, directly after the `is_reading` property add:

```python
    @property
    def generation(self) -> int:
        """Number of the most recent reading started (0 before any).

        Lets a caller that started a reading tell later whether the
        reading now in progress is still its own.
        """
        with self._lock:
            return self._generation
```

In `src/vocalis/ui/doc_popup.py`: add the signal beside the others,

```python
    external_reading_started = Signal(str)
```

connect it in `__init__` right after `self.hotkey_pressed.connect(self.on_hotkey_trigger)`:

```python
        self.external_reading_started.connect(self.on_external_reading_started)
```

and add the slot after `on_hotkey_trigger`:

```python
    def on_external_reading_started(self, text: str) -> None:
        """Show a reading that another feature (the OCR Reader) is about to
        start on the shared DocumentReaderFeature: put its text in the box
        and enable Pause/Stop so the user can control it.

        Does not call the feature (the other feature starts the reading)
        and does not take keyboard focus, so it never steals focus from
        the window the user is working in.
        """
        self._text_box.setPlainText(text)
        self._reset_pause_state()
        self._set_reading_controls_enabled(True)
        self.show()
        self.raise_()
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_doc_reader.py tests/test_doc_popup.py -q --no-cov && .venv/bin/mypy`
Expected: all pass (no earlier test changed), mypy `Success`.

- [ ] **Step 5: Commit**

```bash
git add src/vocalis/features/doc_reader.py src/vocalis/ui/doc_popup.py tests/test_doc_reader.py tests/test_doc_popup.py
git commit -m "feat: let the document popup show a reading started by another feature"
```

---

### Task 5: OcrReaderFeature

**Files:**
- Create: `src/vocalis/features/ocr_reader.py`
- Test: `tests/test_ocr_reader.py`

**Interfaces:**
- Consumes (exact): `ScreenRegionCapture.capture() -> bytes | None` and `ScreenCaptureUnavailable` (Task 3); `TesseractOCR.recognize(bytes) -> str` and `TesseractError` (Task 2); `ClipboardWriter.write(str) -> bool` (Task 1); `DocumentReaderFeature.read_text(str)`, `.stop()`, `.is_reading`, `.generation` (Task 4); `Notifier.notify(title, message="")`.
- Produces: `OcrReaderFeature(capture, ocr, clipboard, reader, notifier, on_reading_starting=None)` with `on_trigger(self) -> None` (never blocks the caller) and read-only `is_busy -> bool`.

Behaviour: `on_trigger` — if a capture is already in progress: ignore. Else if the Document Reader is currently reading **a reading this feature started** (`reader.is_reading and reader.generation == recorded generation`): stop it. Otherwise start a background capture. Pipeline: capture -> (cancel/none) -> OCR -> (< 3 chars: "No text detected") -> clipboard (failure only logged) -> notification preview (first 80 chars) -> `on_reading_starting(text)` -> `reader.read_text(text)` -> record `reader.generation`.

- [ ] **Step 1: Write the failing tests** — create `tests/test_ocr_reader.py`:

```python
"""Tests for vocalis.features.ocr_reader.OcrReaderFeature (fakes for every collaborator)."""
import threading
import time
from typing import Callable

from vocalis.engines.tesseract_ocr import TesseractError
from vocalis.features.ocr_reader import OcrReaderFeature
from vocalis.output.screen_capture import ScreenCaptureUnavailable


def wait_until(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out waiting for condition"
        time.sleep(0.005)


class FakeCapture:
    def __init__(self, result: bytes | None = b"PNG", raises: Exception | None = None) -> None:
        self.result = result
        self.raises = raises
        self.gate: threading.Event | None = None
        self.started = threading.Event()
        self.calls = 0
        self.threads: "list[threading.Thread]" = []

    def capture(self) -> bytes | None:
        self.calls += 1
        self.threads.append(threading.current_thread())
        self.started.set()
        if self.gate is not None:
            assert self.gate.wait(5.0), "gate never released"
        if self.raises is not None:
            raise self.raises
        return self.result


class FakeOcr:
    def __init__(self, text: str = "Hello world", raises: Exception | None = None) -> None:
        self.text = text
        self.raises = raises
        self.received: "list[bytes]" = []

    def recognize(self, png_bytes: bytes) -> str:
        self.received.append(png_bytes)
        if self.raises is not None:
            raise self.raises
        return self.text


class FakeClipboard:
    def __init__(self, ok: bool = True) -> None:
        self.ok = ok
        self.written: "list[str]" = []

    def write(self, text: str) -> bool:
        self.written.append(text)
        return self.ok


class FakeReader:
    def __init__(self, events: "list[str] | None" = None) -> None:
        self.is_reading = False
        self.generation = 0
        self.read_text_calls: "list[str]" = []
        self.stop_calls = 0
        self.events = events if events is not None else []

    def read_text(self, text: str) -> None:
        self.events.append("read_text")
        self.read_text_calls.append(text)
        self.generation += 1
        self.is_reading = True

    def stop(self) -> None:
        self.stop_calls += 1
        self.is_reading = False


class RecordingNotifier:
    def __init__(self) -> None:
        self.messages: "list[tuple[str, str]]" = []

    def notify(self, title: str, message: str = "") -> None:
        self.messages.append((title, message))

    @property
    def bodies(self) -> "list[str]":
        return [message for _title, message in self.messages]


def make(
    capture: FakeCapture | None = None,
    ocr: FakeOcr | None = None,
    clipboard: FakeClipboard | None = None,
    reader: FakeReader | None = None,
    on_reading_starting: "Callable[[str], None] | None" = None,
) -> "tuple[OcrReaderFeature, FakeCapture, FakeOcr, FakeClipboard, FakeReader, RecordingNotifier]":
    capture = capture or FakeCapture()
    ocr = ocr or FakeOcr()
    clipboard = clipboard or FakeClipboard()
    reader = reader or FakeReader()
    notifier = RecordingNotifier()
    feature = OcrReaderFeature(
        capture=capture,
        ocr=ocr,
        clipboard=clipboard,
        reader=reader,
        notifier=notifier,  # type: ignore[arg-type]
        on_reading_starting=on_reading_starting,
    )
    return feature, capture, ocr, clipboard, reader, notifier


def run_once(feature: OcrReaderFeature) -> None:
    feature.on_trigger()
    wait_until(lambda: not feature.is_busy)


def test_happy_path_captures_recognises_copies_and_reads() -> None:
    feature, capture, ocr, clipboard, reader, notifier = make()

    run_once(feature)

    assert ocr.received == [b"PNG"]
    assert clipboard.written == ["Hello world"]
    assert reader.read_text_calls == ["Hello world"]
    assert "Hello world" in notifier.bodies


def test_capture_never_runs_on_the_callers_thread() -> None:
    feature, capture, *_ = make()
    run_once(feature)
    assert capture.threads and capture.threads[0] is not threading.current_thread()


def test_cancelled_capture_says_cancelled_and_does_nothing_else() -> None:
    feature, capture, ocr, clipboard, reader, notifier = make(capture=FakeCapture(result=None))

    run_once(feature)

    assert "Cancelled" in notifier.bodies
    assert ocr.received == [] and clipboard.written == [] and reader.read_text_calls == []


def test_a_cancelled_capture_leaves_the_feature_ready_for_the_next_press() -> None:
    capture = FakeCapture(result=None)
    feature, *_ = make(capture=capture)
    run_once(feature)
    run_once(feature)
    assert capture.calls == 2


def test_missing_capture_tool_shows_its_install_hint() -> None:
    capture = FakeCapture(raises=ScreenCaptureUnavailable("No screen-capture tool found. Install one of: x"))
    feature, _c, _o, _cb, reader, notifier = make(capture=capture)

    run_once(feature)

    assert "No screen-capture tool found. Install one of: x" in notifier.bodies
    assert reader.read_text_calls == []


def test_tesseract_failure_shows_its_message() -> None:
    feature, _c, _o, clipboard, reader, notifier = make(ocr=FakeOcr(raises=TesseractError("Tesseract is not installed.")))

    run_once(feature)

    assert "Tesseract is not installed." in notifier.bodies
    assert clipboard.written == [] and reader.read_text_calls == []


def test_blank_area_says_no_text_detected() -> None:
    feature, _c, _o, clipboard, reader, notifier = make(ocr=FakeOcr(text=""))

    run_once(feature)

    assert "No text detected" in notifier.bodies
    assert clipboard.written == [] and reader.read_text_calls == []


def test_one_or_two_stray_characters_count_as_no_text() -> None:
    feature, _c, _o, clipboard, reader, notifier = make(ocr=FakeOcr(text="ab"))

    run_once(feature)

    assert "No text detected" in notifier.bodies
    assert reader.read_text_calls == []


def test_clipboard_failure_does_not_stop_the_reading() -> None:
    feature, _c, _o, clipboard, reader, _n = make(clipboard=FakeClipboard(ok=False))

    run_once(feature)

    assert clipboard.written == ["Hello world"]
    assert reader.read_text_calls == ["Hello world"]


def test_a_whole_screen_of_text_is_copied_and_read_in_full() -> None:
    long_text = "word " * 2000
    feature, _c, _o, clipboard, reader, notifier = make(ocr=FakeOcr(text=long_text))

    run_once(feature)

    assert clipboard.written == [long_text]
    assert reader.read_text_calls == [long_text]
    preview = [body for body in notifier.bodies if body.startswith("word")]
    assert len(preview) == 1 and len(preview[0]) <= 81 and preview[0].endswith("…")


def test_unexpected_error_is_reported_and_the_feature_stays_usable() -> None:
    feature, _c, ocr, _cb, reader, notifier = make(ocr=FakeOcr(raises=ValueError("boom")))

    run_once(feature)

    assert "Something went wrong reading the screen" in notifier.bodies
    assert not feature.is_busy


def test_on_reading_starting_is_called_with_the_text_before_reading_begins() -> None:
    events: "list[str]" = []
    reader = FakeReader(events=events)
    feature, *_ = make(reader=reader, on_reading_starting=lambda text: events.append(f"starting:{text}"))

    run_once(feature)

    assert events == ["starting:Hello world", "read_text"]


def test_pressing_again_while_the_box_is_being_drawn_is_ignored() -> None:
    capture = FakeCapture()
    capture.gate = threading.Event()
    feature, *_ = make(capture=capture)

    feature.on_trigger()
    assert capture.started.wait(5.0)
    feature.on_trigger()  # user presses again while still dragging
    capture.gate.set()
    wait_until(lambda: not feature.is_busy)

    assert capture.calls == 1


def test_pressing_again_while_its_own_reading_plays_stops_it() -> None:
    feature, capture, _o, _cb, reader, _n = make()
    run_once(feature)
    assert reader.is_reading is True

    feature.on_trigger()
    wait_until(lambda: reader.stop_calls == 1)

    assert capture.calls == 1  # no new capture was started


def test_pressing_while_a_document_reading_it_did_not_start_plays_starts_a_capture() -> None:
    reader = FakeReader()
    reader.is_reading = True  # e.g. a file opened from the Ctrl+Alt+D window
    reader.generation = 7
    feature, capture, *_ = make(reader=reader)

    run_once(feature)

    assert reader.stop_calls == 0
    assert capture.calls == 1


def test_a_later_reading_by_someone_else_is_not_stopped_by_the_next_press() -> None:
    feature, capture, _o, _cb, reader, _n = make()
    run_once(feature)  # OCR reading finishes being "its own" ...
    reader.generation += 1  # ... then the user starts a different reading

    run_once(feature)

    assert reader.stop_calls == 0
    assert capture.calls == 2
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_ocr_reader.py -q --no-cov`
Expected: FAIL — `ModuleNotFoundError: vocalis.features.ocr_reader`.

- [ ] **Step 3: Implement** — create `src/vocalis/features/ocr_reader.py`:

```python
"""OCR Reader: Ctrl+Alt+O -> drag a box on screen -> read its text aloud.

This feature only captures, recognises and copies. Speaking is handed to
the existing DocumentReaderFeature.read_text(), which already provides
synthesis with fallback, chunking, pause/resume/stop and
ReadingCoordinator ownership, so no reading-aloud code is duplicated.

The interactive capture (the user dragging a box) can take many seconds,
so it never runs on the hotkey watcher thread: on_trigger() only flips
state under a short lock and starts a background thread.
"""
import logging
import threading
from typing import Callable, Protocol

from vocalis.engines.tesseract_ocr import TesseractError
from vocalis.notify import Notifier
from vocalis.output.screen_capture import ScreenCaptureUnavailable

logger = logging.getLogger("vocalis")

_TITLE = "OCR Reader"
_MIN_TEXT_LENGTH = 3
_PREVIEW_LENGTH = 80


class RegionCapture(Protocol):
    """Lets the user choose a screen region and returns it as PNG bytes."""

    def capture(self) -> bytes | None: ...


class TextRecognizer(Protocol):
    """Turns a picture into text."""

    def recognize(self, png_bytes: bytes) -> str: ...


class ClipboardSink(Protocol):
    """Puts text on the clipboard."""

    def write(self, text: str) -> bool: ...


class SpeechReader(Protocol):
    """The part of DocumentReaderFeature that the OCR Reader uses."""

    @property
    def is_reading(self) -> bool: ...

    @property
    def generation(self) -> int: ...

    def read_text(self, text: str) -> None: ...

    def stop(self) -> None: ...


class OcrReaderFeature:
    """Wires Ctrl+Alt+O to capture -> OCR -> clipboard -> read aloud."""

    def __init__(
        self,
        capture: RegionCapture,
        ocr: TextRecognizer,
        clipboard: ClipboardSink,
        reader: SpeechReader,
        notifier: Notifier,
        on_reading_starting: Callable[[str], None] | None = None,
    ) -> None:
        """Create an OcrReaderFeature.

        Args:
            capture: Lets the user drag a box on screen.
            ocr: Reads the text out of the captured picture.
            clipboard: Receives a copy of the recognised text.
            reader: The shared Document Reader that speaks the text.
            notifier: Sends the status notifications.
            on_reading_starting: Called with the text just before the
                reading starts (the app uses it to show the Document
                Reader window with Pause/Stop). Optional.
        """
        self._capture = capture
        self._ocr = ocr
        self._clipboard = clipboard
        self._reader = reader
        self._notifier = notifier
        self._on_reading_starting = on_reading_starting
        self._lock = threading.Lock()
        self._busy = False
        self._my_generation: int | None = None

    @property
    def is_busy(self) -> bool:
        """Whether a capture/recognise/start-reading run is in progress."""
        with self._lock:
            return self._busy

    def on_trigger(self) -> None:
        """Call when Ctrl+Alt+O is pressed. Never blocks the caller.

        Ignored while a capture run is in progress. If the Document
        Reader is playing a reading that this feature started, the press
        stops it; otherwise it starts a new capture on a background thread.
        """
        with self._lock:
            if self._busy:
                return
            owns_reading = (
                self._my_generation is not None
                and self._reader.is_reading
                and self._reader.generation == self._my_generation
            )
            if not owns_reading:
                self._busy = True
        if owns_reading:
            threading.Thread(target=self._reader.stop, daemon=True).start()
            return
        threading.Thread(target=self._capture_and_read, daemon=True).start()

    def _capture_and_read(self) -> None:
        """Run one full capture -> read pass on a background thread and
        always clear the busy flag, whatever happens."""
        try:
            self._run_pipeline()
        except Exception:
            logger.exception("OCR reader failed")
            self._notifier.notify(_TITLE, "Something went wrong reading the screen")
        finally:
            with self._lock:
                self._busy = False

    def _run_pipeline(self) -> None:
        """Capture a region, recognise its text, copy it, and start reading it."""
        try:
            png = self._capture.capture()
        except ScreenCaptureUnavailable as exc:
            self._notifier.notify(_TITLE, str(exc))
            return
        if png is None:
            self._notifier.notify(_TITLE, "Cancelled")
            return
        try:
            text = self._ocr.recognize(png)
        except TesseractError as exc:
            self._notifier.notify(_TITLE, str(exc))
            return
        if len(text) < _MIN_TEXT_LENGTH:
            self._notifier.notify(_TITLE, "No text detected")
            return
        if not self._clipboard.write(text):
            logger.warning("Could not copy the recognised text to the clipboard")
        self._notifier.notify(_TITLE, _preview(text))
        if self._on_reading_starting is not None:
            self._on_reading_starting(text)
        self._reader.read_text(text)
        with self._lock:
            self._my_generation = self._reader.generation


def _preview(text: str) -> str:
    """The first 80 characters of `text`, with an ellipsis if cut short."""
    if len(text) <= _PREVIEW_LENGTH:
        return text
    return text[:_PREVIEW_LENGTH] + "…"
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_ocr_reader.py -q --no-cov && .venv/bin/mypy`
Expected: all pass, mypy `Success`. Then run it 10 times in a loop (`for i in $(seq 10); do .venv/bin/pytest tests/test_ocr_reader.py -q --no-cov -p no:cacheprovider || break; done`) — all must pass every time.

- [ ] **Step 5: Commit**

```bash
git add src/vocalis/features/ocr_reader.py tests/test_ocr_reader.py
git commit -m "feat: add OCR reader feature"
```

---

### Task 6: Wire the OCR Reader into the app

**Files:**
- Modify: `src/vocalis/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: everything above; `build_doc_reader()` return `(feature, popup, listener)`; `CtrlAltKeyListener(key_code, on_trigger, device_enumerator)`; `list_devices_with_key(code)`.
- Produces: `build_ocr_reader(config: VocalisConfig, document_reader: DocumentReaderFeature, popup: DocumentReaderPopup) -> tuple[OcrReaderFeature, CtrlAltKeyListener]`; `main()` starts and stops the Ctrl+Alt+O listener with the others.

- [ ] **Step 1: Write the failing tests** — in `tests/test_cli.py` change the import to `from vocalis.cli import build_daemon, build_dictation, build_doc_reader, build_ocr_reader, build_quick_reader, main` and append:

```python
def test_build_ocr_reader_shares_the_document_reader_and_listens_for_ctrl_alt_o(qtbot) -> None:  # type: ignore[no-untyped-def]
    config = VocalisConfig()
    manager = ModelManager()
    coordinator = ReadingCoordinator()
    doc_feature, popup, _doc_listener = build_doc_reader(
        config=config, model_manager=manager, reading_coordinator=coordinator
    )
    qtbot.addWidget(popup)

    feature, listener = build_ocr_reader(config=config, document_reader=doc_feature, popup=popup)

    assert feature._reader is doc_feature  # noqa: SLF001
    assert listener._key_code == 24  # noqa: SLF001  # evdev KEY_O

    # The "reading is starting" hook must reach the popup so Pause/Stop work.
    hook = feature._on_reading_starting  # noqa: SLF001
    assert hook is not None
    hook("Hello from the screen")
    assert popup._text_box.toPlainText() == "Hello from the screen"  # noqa: SLF001
    assert popup._stop_button.isEnabled() is True  # noqa: SLF001


def test_main_starts_and_stops_the_ocr_listener(tmp_path: Path, qapp) -> None:  # type: ignore[no-untyped-def]
    with (
        patch("vocalis.cli.default_config_path", return_value=tmp_path / "config.toml"),
        patch("vocalis.cli._socket_path", return_value=tmp_path / "vocalis.sock"),
        patch("vocalis.cli.list_numlock_capable_devices", return_value=[]),
        patch("vocalis.cli.list_hotkey_capable_devices", return_value=[]),
        patch("vocalis.cli.list_devices_with_key", return_value=[]),
        patch("vocalis.cli.build_ocr_reader") as mock_build_ocr,
        patch("vocalis.cli.QApplication.exec", side_effect=KeyboardInterrupt),
    ):
        listener = MagicMock()
        mock_build_ocr.return_value = (MagicMock(), listener)
        main()

    listener.start.assert_called_once_with()
    listener.stop.assert_called_once_with()
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_cli.py -q --no-cov`
Expected: FAIL — `ImportError: cannot import name 'build_ocr_reader'`.

- [ ] **Step 3: Implement** — in `src/vocalis/cli.py`:

Add imports next to the existing ones:

```python
from vocalis.engines.tesseract_ocr import TesseractOCR
from vocalis.features.ocr_reader import OcrReaderFeature
from vocalis.output.clipboard import ClipboardWriter
from vocalis.output.screen_capture import ScreenRegionCapture
```

Add the key constant next to `_KEY_D = 32`:

```python
_KEY_O = 24
```

Add after `build_doc_reader`:

```python
def build_ocr_reader(
    config: VocalisConfig, document_reader: DocumentReaderFeature, popup: DocumentReaderPopup
) -> tuple[OcrReaderFeature, CtrlAltKeyListener]:
    """Construct the OCR Reader feature and its Ctrl+Alt+O listener.

    The OCR Reader speaks through the already-built Document Reader, so it
    shares that reader's voices, pause/stop and ReadingCoordinator. Its
    "reading is starting" hook is wired to the popup's
    `external_reading_started` signal so the Document Reader window shows
    the recognised text with Pause/Stop enabled. The hook is a signal
    emit, so it is safe to call from the OCR background thread.

    Args:
        config: Loaded, validated vocalis configuration.
        document_reader: The Document Reader built by build_doc_reader().
        popup: The Document Reader popup built by build_doc_reader().

    Returns:
        A (feature, listener) pair. The caller starts/stops the listener.
    """
    feature = OcrReaderFeature(
        capture=ScreenRegionCapture(),
        ocr=TesseractOCR(),
        clipboard=ClipboardWriter(),
        reader=document_reader,
        notifier=Notifier(enabled=config.notifications_enabled),
        on_reading_starting=popup.external_reading_started.emit,
    )
    listener = CtrlAltKeyListener(
        key_code=_KEY_O,
        on_trigger=feature.on_trigger,
        device_enumerator=lambda: list_devices_with_key(_KEY_O),
    )
    return feature, listener
```

In `main()`: change `_, _doc_popup, doc_reader_listener = build_doc_reader(...)` to

```python
    doc_reader_feature, doc_popup, doc_reader_listener = build_doc_reader(
        config=config, model_manager=model_manager, reading_coordinator=reading_coordinator
    )
    _, ocr_listener = build_ocr_reader(
        config=config, document_reader=doc_reader_feature, popup=doc_popup
    )
```

add `ocr_listener.start()` after `doc_reader_listener.start()`, and put `("ocr reader listener", ocr_listener.stop),` as the FIRST entry of the `_stop_all_independently([...])` list. Update `main()`'s docstring list of listeners to mention the OCR Reader's Ctrl+Alt+O listener.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_cli.py -q --no-cov && .venv/bin/mypy`
Expected: all pass (existing `main` tests still pass because the OCR listener's enumerator is the already-patched `list_devices_with_key`), mypy `Success`.

- [ ] **Step 5: Commit**

```bash
git add src/vocalis/cli.py tests/test_cli.py
git commit -m "feat: wire the OCR reader and its Ctrl+Alt+O listener into the app"
```

---

### Task 7: Docs, change log, full verification, real-hardware check

**Files:**
- Modify: `docs/user-docs.md`, `docs/architecture.md`, `docs/workflow.md`, `docs/file-purpose.md`, `docs/folder-structure.md`, `docs/developer-docs.md` (only where the text is now inaccurate or missing), `docs/superpowers/specs/2026-09-11-vocalis-design.md` (Phase 5 amendment: note the three deviations listed at the top of this plan)
- Create: `.claude/<YYYY-MM-DD_HHMM>_CHANGELOG.md`

- [ ] **Step 1: Docs** (verify every sentence against the code before writing it):
  - `user-docs.md`: new "Reading text off the screen (OCR Reader)" section: press **Ctrl+Alt+O**; a Flameshot box appears — drag it, **drag its edges or corners to resize**, then press **Enter** (or click the tick) to confirm, **Esc** to cancel; the text is copied to the clipboard and read aloud; the Document Reader window opens with **Pause/Resume/Stop**; pressing **Ctrl+Alt+O** again while it is reading stops it; messages: "Cancelled", "No text detected", missing-tool hints; English only; resizing needs Flameshot (otherwise `maim` / `slurp`+`grim` draw one fixed box).
  - `architecture.md` + `workflow.md`: the Ctrl+Alt+O flow (key -> `OcrReaderFeature` background thread -> capture -> Tesseract -> clipboard -> notify -> popup signal -> `DocumentReaderFeature.read_text`), noting Tesseract is CPU-only and not routed through `ModelManager`.
  - `file-purpose.md` / `folder-structure.md`: rows for `screen_capture.py`, `tesseract_ocr.py`, `ocr_reader.py`, `ClipboardWriter`; mention the `generation` property and the popup's `external_reading_started` signal.
  - Spec amendment: add a short "Planning deviations" list (in-memory picture, normalisation done by `read_text`, popup shown for Pause).
- [ ] **Step 2: Change log** — create `.claude/<timestamp>_CHANGELOG.md` summarising Phase 5 (what was added, files, tests, decisions, rollback = `git revert` the merge).
- [ ] **Step 3: Full verification**

Run: `.venv/bin/pytest -q` and `.venv/bin/mypy`
Expected: all tests pass, **zero warnings**, coverage **>= 85%**, mypy `Success: no issues found`. Also run `tests/test_ocr_reader.py tests/test_screen_capture.py tests/test_cli.py` 10 times in a loop; every run must pass.

- [ ] **Step 4: Real-tool smoke test (no screen needed)** — in the scratchpad directory:

```bash
convert -size 500x100 xc:white -font DejaVu-Sans -pointsize 40 -fill black -annotate +10+65 "Hello OCR world" /tmp/claude-1000/ocr_smoke.png 2>/dev/null || import -window root -crop 500x100+0+0 /tmp/claude-1000/ocr_smoke.png
.venv/bin/python - <<'PY'
from pathlib import Path
from vocalis.engines.tesseract_ocr import TesseractOCR
from vocalis.output.clipboard import ClipboardWriter, ClipboardReader
print("OCR:", TesseractOCR().recognize(Path("/tmp/claude-1000/ocr_smoke.png").read_bytes()))
print("clipboard write:", ClipboardWriter().write("vocalis-ocr-test"))
print("clipboard read :", ClipboardReader().read())
PY
```
Expected: OCR line contains `Hello OCR world`; `clipboard write: True`; `clipboard read : vocalis-ocr-test`. (Delete the PNG afterwards.)

- [ ] **Step 5: Real-hardware check with the user** — run `vocalis` in a terminal (check first that no other `vocalis` process is running), then ask the user to: (a) press Ctrl+Alt+O, draw a box over some text, resize it by its corners, press Enter, and listen; (b) press Pause/Resume/Stop in the window; (c) press Ctrl+Alt+O while it reads (stops); (d) press Ctrl+Alt+O then Esc ("Cancelled"); (e) box over an empty area ("No text detected"); (f) paste the clipboard to confirm the text was copied; (g) Ctrl+C the program: it must exit in a few seconds with no `ERROR`/`Traceback` lines in the log. Record the outcome in the change log.
- [ ] **Step 6: Commit**

```bash
git add docs .claude/*CHANGELOG.md
git commit -m "docs: document the OCR reader and record the phase 5 change log"
```
