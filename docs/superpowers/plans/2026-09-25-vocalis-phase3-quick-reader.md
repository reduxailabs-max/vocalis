# Vocalis Phase 3: Quick Reader (TTS) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Ctrl+Alt+R read the current clipboard text aloud — press it
again while reading to stop immediately. Uses Qwen3-TTS (a real, local,
open-source voice) as the primary voice, automatically falling back to the
already-installed Piper engine if Qwen3-TTS fails for any reason.

**Architecture:** A new `CtrlAltRListener` (evdev, background thread per
keyboard device — the same proven pattern as Phase 2's `NumlockListener`)
calls `QuickReaderFeature.on_trigger()`. On the first press it reads the
clipboard, borrows Qwen3-TTS (or Piper on failure) through the existing
Phase 1 `ModelManager`, generates speech, and plays it via a new
`sounddevice`-based `AudioPlayer` — all in a background thread so the
hotkey listener itself is never blocked and can immediately notice a
second press. On that second press (while still reading), the feature
cancels the in-flight work and stops playback immediately.

**Tech Stack:** `qwen-tts` (Qwen3-TTS Python package, CUDA-only),
`torch`/`torchaudio` (Qwen3-TTS's runtime), the existing `piper` binary +
`en_US-lessac-medium.onnx` voice (already installed, CPU-only, zero new
dependencies), `sounddevice` (already a dependency — used for playback the
same way Phase 2 used it for capture), `evdev` (already a dependency),
`xclip` (already installed, used for reading instead of writing this
time).

**Spec:** `docs/superpowers/specs/2026-09-11-vocalis-design.md` — see the
Phase 3 amendment section (added 2026-09-24, corrected 2026-09-25) for the
full rationale behind every decision below. This plan implements that
amendment exactly; read both together.

## Global Constraints

- Fully offline at runtime — Qwen3-TTS's model files are downloaded once
  (Task 2, a genuine one-time setup step needing internet), then loaded
  from a local directory with no further network calls, exactly like
  Parakeet.
- Only **one** AI model resident in VRAM at any time — Qwen3-TTS is
  borrowed/released through the existing `ModelManager` exactly like
  Parakeet. Piper never touches the GPU at all, so it is **not** routed
  through `ModelManager` — it's a plain, always-available object.
- Every function gets a docstring; structured error handling with
  specific exception types; no bare `except:`.
- Structured logging via the `"vocalis"` logger.
- ≥85% test coverage (enforced by `pytest-cov --cov-fail-under=85`).
- `old/` is never modified.
- Conventional-commit messages, ending with:
  ```
  Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_0116ii4z7tzRCRhVAxecpkYC
  ```
  copied verbatim.
- Desktop notifications for Quick Reader must cover **start** ("Reading..."),
  **live text** (the text being read), and **done** states, matching
  Dictation's existing pattern, toggleable via the existing
  `notifications_enabled` config field.
- One sub-task at a time — stop after each task and wait for explicit
  user "yes"/"y" before continuing.
- Every new evdev-based listener excludes Vocalis's own `ydotoold virtual
  device` from enumeration and wraps callback invocations in their own
  try/except inside the read loop — these are lessons learned the hard
  way in Phase 2 (Tasks 9 and 10) and must not be repeated as bugs in new
  code.
- Every new module that swaps out a mutable resource (an audio stream, a
  loaded model) under a lock must release the lock **before** calling any
  blocking method on that resource — the exact lesson from Phase 2 Task
  9's `MicRecorder` deadlock. Never hold a lock across a call into
  unknown/blocking code.

---

### Task 1: Piper TTS Engine Wrapper (the fallback voice)

Built first because it needs zero new dependencies — the `piper` binary
and `en_US-lessac-medium.onnx` voice are already installed and proven
working (confirmed by hand: `--output_raw` produces raw 16-bit PCM at
22050Hz to stdout).

**Files:**
- Create: `src/vocalis/engines/piper_tts.py`
- Create: `tests/test_piper_tts.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `PiperLoadError(Exception)`
  - `PiperSynthesisError(Exception)`
  - `PiperTTS.__init__(self, binary_path: Path, model_path: Path, sample_rate: int = 22050, command_runner: CommandRunner | None = None) -> None`
  - `PiperTTS.load(self) -> None` (implements `ModelEngine`)
  - `PiperTTS.unload(self) -> None` (implements `ModelEngine`)
  - `PiperTTS.is_loaded -> bool` (property, implements `ModelEngine`)
  - `PiperTTS.synthesize(self, text: str) -> tuple[npt.NDArray[np.float32], int]`
  - Task 6 (QuickReaderFeature) holds one `PiperTTS` instance directly
    (constructed and `.load()`-ed once at startup, never through
    `ModelManager` since it never touches the GPU) and calls
    `.synthesize(text)` on it as the fallback path.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_piper_tts.py
"""Tests for vocalis.engines.piper_tts.PiperTTS.

Uses a FakeCommandRunner instead of a real subprocess, so tests never
touch the real piper binary.
"""
import struct
from pathlib import Path

import numpy as np
import pytest

from vocalis.engines.piper_tts import PiperLoadError, PiperSynthesisError, PiperTTS


class FakeCompletedProcess:
    def __init__(self, returncode: int, stdout: bytes = b"") -> None:
        self.returncode = returncode
        self.stdout = stdout


class FakeCommandRunner:
    def __init__(self, returncode: int = 0, stdout: bytes = b"", raise_exc: Exception | None = None) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.raise_exc = raise_exc
        self.calls: list[tuple[list[str], bytes | None]] = []

    def __call__(self, argv: list[str], input_bytes: bytes | None = None) -> FakeCompletedProcess:
        self.calls.append((argv, input_bytes))
        if self.raise_exc is not None:
            raise self.raise_exc
        return FakeCompletedProcess(returncode=self.returncode, stdout=self.stdout)


def _pcm_bytes(samples: list[int]) -> bytes:
    return struct.pack(f"<{len(samples)}h", *samples)


def test_load_succeeds_when_binary_and_model_exist(tmp_path: Path) -> None:
    binary = tmp_path / "piper"
    binary.write_text("")
    model = tmp_path / "voice.onnx"
    model.write_text("")
    engine = PiperTTS(binary_path=binary, model_path=model)
    assert engine.is_loaded is False
    engine.load()
    assert engine.is_loaded is True


def test_load_raises_when_binary_missing(tmp_path: Path) -> None:
    model = tmp_path / "voice.onnx"
    model.write_text("")
    engine = PiperTTS(binary_path=tmp_path / "missing-piper", model_path=model)
    with pytest.raises(PiperLoadError):
        engine.load()


def test_load_raises_when_model_missing(tmp_path: Path) -> None:
    binary = tmp_path / "piper"
    binary.write_text("")
    engine = PiperTTS(binary_path=binary, model_path=tmp_path / "missing.onnx")
    with pytest.raises(PiperLoadError):
        engine.load()


def test_synthesize_before_load_raises(tmp_path: Path) -> None:
    binary = tmp_path / "piper"
    binary.write_text("")
    model = tmp_path / "voice.onnx"
    model.write_text("")
    engine = PiperTTS(binary_path=binary, model_path=model)
    with pytest.raises(RuntimeError):
        engine.synthesize("hello")


def test_synthesize_converts_raw_pcm_to_float32(tmp_path: Path) -> None:
    binary = tmp_path / "piper"
    binary.write_text("")
    model = tmp_path / "voice.onnx"
    model.write_text("")
    runner = FakeCommandRunner(returncode=0, stdout=_pcm_bytes([0, 16384, -16384, 32767]))
    engine = PiperTTS(binary_path=binary, model_path=model, sample_rate=22050, command_runner=runner)
    engine.load()

    audio, sample_rate = engine.synthesize("hello world")

    assert sample_rate == 22050
    assert audio.dtype == np.float32
    np.testing.assert_array_almost_equal(
        audio, np.array([0.0, 0.5, -0.5, 0.999969], dtype=np.float32), decimal=5
    )
    argv, input_bytes = runner.calls[0]
    assert argv == [str(binary), "--model", str(model), "--output_raw"]
    assert input_bytes == b"hello world"


def test_synthesize_raises_on_nonzero_exit(tmp_path: Path) -> None:
    binary = tmp_path / "piper"
    binary.write_text("")
    model = tmp_path / "voice.onnx"
    model.write_text("")
    runner = FakeCommandRunner(returncode=1, stdout=b"")
    engine = PiperTTS(binary_path=binary, model_path=model, command_runner=runner)
    engine.load()
    with pytest.raises(PiperSynthesisError):
        engine.synthesize("hello")


def test_synthesize_raises_when_runner_raises(tmp_path: Path) -> None:
    binary = tmp_path / "piper"
    binary.write_text("")
    model = tmp_path / "voice.onnx"
    model.write_text("")
    runner = FakeCommandRunner(raise_exc=OSError("piper not found"))
    engine = PiperTTS(binary_path=binary, model_path=model, command_runner=runner)
    engine.load()
    with pytest.raises(PiperSynthesisError):
        engine.synthesize("hello")


def test_unload_clears_loaded_state(tmp_path: Path) -> None:
    binary = tmp_path / "piper"
    binary.write_text("")
    model = tmp_path / "voice.onnx"
    model.write_text("")
    engine = PiperTTS(binary_path=binary, model_path=model)
    engine.load()
    engine.unload()
    assert engine.is_loaded is False
```

- [ ] **Step 2: Run the tests, verify they fail**

Run: `.venv/bin/python -m pytest tests/test_piper_tts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vocalis.engines.piper_tts'`

- [ ] **Step 3: Write the minimal implementation**

```python
# src/vocalis/engines/piper_tts.py
"""Piper text-to-speech engine — Quick Reader's fallback voice.

Implements the ModelEngine protocol for interface consistency with every
other engine, but is never actually routed through ModelManager: Piper
runs entirely on CPU via the existing local `piper` binary (already
installed, proven working — see old/read-selected.sh), so it never
competes for GPU residency the way Qwen3-TTS does. load()/unload() exist
to fail fast if the binary or voice model files go missing, and to match
the shape every other engine wrapper uses; there is no persistent process
to actually hold — each synthesize() call runs a fresh subprocess, the
same pattern TextOutput/Notifier already use for ydotool/xclip/notify-send.
"""
import logging
import subprocess
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

logger = logging.getLogger("vocalis")


class CommandRunner(Protocol):
    """Runs a command, optionally piping bytes to its stdin."""

    def __call__(
        self, argv: list[str], input_bytes: bytes | None = None
    ) -> "subprocess.CompletedProcess[bytes]": ...


class PiperLoadError(Exception):
    """Raised when Piper's binary or voice model files are missing."""


class PiperSynthesisError(Exception):
    """Raised when Piper fails to synthesize speech."""


class PiperTTS:
    """Text-to-speech engine backed by the local Piper binary (CPU-only)."""

    def __init__(
        self,
        binary_path: Path,
        model_path: Path,
        sample_rate: int = 22050,
        command_runner: CommandRunner | None = None,
    ) -> None:
        """Create a PiperTTS engine (does not check files yet).

        Args:
            binary_path: Path to the piper executable.
            model_path: Path to the .onnx voice model file.
            sample_rate: Sample rate of the voice model's audio output, in
                Hz (22050 for en_US-lessac-medium, per its .onnx.json).
            command_runner: Callable(argv, input_bytes=None) -> CompletedProcess,
                used to run piper. Defaults to subprocess.run with a
                30-second timeout. Injectable for tests.
        """
        self._binary_path = binary_path
        self._model_path = model_path
        self._sample_rate = sample_rate
        self._run: CommandRunner = command_runner or _default_command_runner
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Whether load() has succeeded."""
        return self._loaded

    def load(self) -> None:
        """Verify the piper binary and voice model files exist.

        Raises:
            PiperLoadError: If either file is missing.
        """
        if not self._binary_path.exists():
            raise PiperLoadError(f"Piper binary not found at {self._binary_path}")
        if not self._model_path.exists():
            raise PiperLoadError(f"Piper voice model not found at {self._model_path}")
        self._loaded = True

    def unload(self) -> None:
        """Clear the loaded flag. Safe to call even if never loaded."""
        self._loaded = False

    def synthesize(self, text: str) -> "tuple[npt.NDArray[np.float32], int]":
        """Generate speech audio for the given text.

        Args:
            text: The text to speak.

        Returns:
            (audio, sample_rate) — a 1-D float32 numpy array and its rate.

        Raises:
            RuntimeError: If called before load().
            PiperSynthesisError: If the piper subprocess fails or cannot
                be run at all.
        """
        if not self._loaded:
            raise RuntimeError("PiperTTS.synthesize() called before load()")
        try:
            result = self._run(
                [str(self._binary_path), "--model", str(self._model_path), "--output_raw"],
                input_bytes=text.encode("utf-8"),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise PiperSynthesisError(f"Failed to run piper: {exc}") from exc
        if result.returncode != 0:
            raise PiperSynthesisError(f"piper exited with code {result.returncode}")
        samples = np.frombuffer(result.stdout, dtype="<i2").astype(np.float32) / 32768.0
        return samples, self._sample_rate


def _default_command_runner(
    argv: list[str], input_bytes: bytes | None = None
) -> "subprocess.CompletedProcess[bytes]":
    """Default CommandRunner: subprocess.run with a 30-second timeout."""
    return subprocess.run(argv, input=input_bytes, timeout=30.0, capture_output=True)
```

- [ ] **Step 4: Run the tests, verify they pass**

Run: `.venv/bin/python -m pytest tests/test_piper_tts.py -v`
Expected: all 8 tests PASS

- [ ] **Step 5: Run the full suite + mypy**

```bash
.venv/bin/python -m pytest
.venv/bin/python -m mypy src
```
Expected: all existing tests still pass, coverage still ≥85%, mypy clean.

- [ ] **Step 6: Real smoke test with the actual piper binary**

Run this once by hand (not part of the automated suite) to confirm the
real binary/model files actually work end to end through the new wrapper:

```bash
.venv/bin/python -c "
from pathlib import Path
from vocalis.engines.piper_tts import PiperTTS

engine = PiperTTS(
    binary_path=Path.home() / '.local' / 'bin' / 'piper' / 'piper',
    model_path=Path.home() / '.local' / 'share' / 'piper-voices' / 'en_US-lessac-medium.onnx',
)
engine.load()
audio, sr = engine.synthesize('Hello from Vocalis Phase 3.')
print('samples:', len(audio), 'sample_rate:', sr, 'max amplitude:', abs(audio).max())
"
```

Expected: prints a nonzero sample count, `sample_rate: 22050`, and a
nonzero max amplitude. Record the actual output in this task's commit
message or the ledger.

- [ ] **Step 7: Commit**

```bash
git add src/vocalis/engines/piper_tts.py tests/test_piper_tts.py
git commit -m "feat: add PiperTTS fallback engine wrapper" -m "- Wraps the existing local piper binary via subprocess, converting raw PCM to float32" -m "- Implements ModelEngine for interface consistency, but is never routed through ModelManager since it never touches the GPU" -m "- Verified end-to-end against the real piper binary and voice model" -m "Phase 3 of vocalis (quick reader)" -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_0116ii4z7tzRCRhVAxecpkYC"
```

---

### Task 2: Qwen3-TTS Engine Wrapper (the primary voice)

This is the highest-risk task in this plan: it involves installing a real
multi-gigabyte ML stack (`qwen-tts`, `torch`, `transformers`) that has two
**known real dependency-conflict risks** with what's already installed:

1. `qwen-tts` depends on plain `onnxruntime` (CPU-only), but this project
   already has `onnxruntime-gpu` installed for Parakeet — both packages
   provide the same `onnxruntime` importable module and can silently
   clobber each other if both get installed normally.
2. `pip install torch` from PyPI normally pulls in its own copies of
   `nvidia-cudnn-cu12`/`nvidia-cublas-cu12` etc. as transitive
   dependencies, which may conflict with the exact `nvidia-cudnn-cu12`
   version this project already pins for Parakeet's CUDA provider.

**Do not guess your way around these — resolve them for real and record
exactly what worked**, the same way Phase 2 Task 1 confirmed the real
Parakeet loading path by hand before finalizing it, and Phase 2 Task 5
worked through a real venv install failure rather than assuming a fix.

**Files:**
- Modify: `pyproject.toml`
- Create: `src/vocalis/engines/qwen3_tts.py`
- Create: `tests/test_qwen3_tts.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `Qwen3TTSLoadError(Exception)`
  - `Qwen3TTS.__init__(self, model_dir: Path, speaker: str = "Ryan", language: str = "English", use_gpu: bool = True) -> None`
  - `Qwen3TTS.load(self) -> None` (implements `ModelEngine`)
  - `Qwen3TTS.unload(self) -> None` (implements `ModelEngine`)
  - `Qwen3TTS.is_loaded -> bool` (property, implements `ModelEngine`)
  - `Qwen3TTS.synthesize(self, text: str) -> tuple[npt.NDArray[np.float32], int]`
  - Task 6 (QuickReaderFeature) borrows this through `ModelManager.acquire()`/
    `.release()` exactly like `ParakeetSTT` in Phase 2, via an
    `engine_factory` closure built in Task 7's CLI wiring.

- [ ] **Step 1: Install the real dependencies and resolve the real conflicts**

Add to `pyproject.toml`'s `dependencies` list:

```toml
    "qwen-tts==0.1.1",
    "transformers==4.57.3",
    "accelerate==1.12.0",
    "torchaudio>=2.0,<3",
    "soundfile>=0.12,<1",
    "einops>=0.7,<1",
    "librosa>=0.10,<1",
    "huggingface-hub>=0.20,<1",
```

Deliberately **not** added: `onnxruntime` (already satisfied by the
existing `onnxruntime-gpu`), `gradio` (qwen-tts's own demo UI, not needed
for library use), `sox` (a system audio CLI tool `qwen-tts` may reference
in its own demo code, not the library import path we use).

Install with:

```bash
.venv/bin/python -m pip install -e ".[dev]"
```

This will likely try to pull in `onnxruntime` and `gradio`/`sox` anyway as
`qwen-tts`'s own declared dependencies (pip installs a package's full
declared dependency tree unless told otherwise). If that happens:

1. Check whether `onnxruntime` (plain, CPU) got installed alongside
   `onnxruntime-gpu` and broke Parakeet's CUDA provider — run
   `.venv/bin/python -m pytest tests/test_parakeet.py -v` and the real
   smoke test from Phase 2 Task 1's changelog to check.
2. If it did break, reinstall with `pip install --no-deps qwen-tts==0.1.1`
   instead (skips its declared deps entirely — you already installed the
   real runtime deps explicitly above), then re-verify Parakeet still
   works.
3. Check whether `torch`'s own transitive CUDA packages conflict with
   `nvidia-cudnn-cu12`'s already-pinned version — run
   `.venv/bin/python -c "import torch; print(torch.__version__); print(torch.version.cuda)"`
   and confirm it imports without error and reports a CUDA version
   consistent with this machine's driver (check with `nvidia-smi`).
4. Whatever you actually had to do to get a working install — record it
   precisely in this task's changelog entry (Step 8) and its commit
   message. This is exactly the kind of real-environment finding Phase 2
   Task 5's venv install stall was — expected to happen, not a plan
   defect.

Confirm success with:

```bash
.venv/bin/python -c "import torch, qwen_tts; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available())"
```

Expected: prints a torch version and `cuda available True`.

- [ ] **Step 2: One-time model download**

Download the CustomVoice 0.6B model files into a local directory Vocalis
manages itself (matching Parakeet's philosophy: no implicit runtime
network calls, config points at an explicit local path):

```bash
.venv/bin/python -c "
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id='Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice',
    local_dir='$HOME/.local/share/vocalis/models/qwen3-tts-customvoice-0.6b',
)
print('Downloaded to:', path)
"
```

Expected: downloads roughly 1.8GB and prints the local directory path.
This step needs internet access once; nothing after this does.

- [ ] **Step 3: Write the failing tests**

```python
# tests/test_qwen3_tts.py
"""Tests for vocalis.engines.qwen3_tts.Qwen3TTS.

Uses a fake model object in place of the real Qwen3TTSModel (which needs
a GPU and a multi-gigabyte download), so tests never touch real hardware.
"""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from vocalis.engines.qwen3_tts import Qwen3TTS, Qwen3TTSLoadError


class FakeQwenModel:
    def __init__(self, wavs: list[np.ndarray], sample_rate: int, raise_exc: Exception | None = None) -> None:
        self.wavs = wavs
        self.sample_rate = sample_rate
        self.raise_exc = raise_exc
        self.calls: list[dict[str, object]] = []

    def generate_custom_voice(self, text: str, language: str, speaker: str) -> tuple[list[np.ndarray], int]:
        self.calls.append({"text": text, "language": language, "speaker": speaker})
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.wavs, self.sample_rate


def test_is_loaded_false_before_load() -> None:
    engine = Qwen3TTS(model_dir=Path("/fake/model/dir"))
    assert engine.is_loaded is False


def test_synthesize_before_load_raises() -> None:
    engine = Qwen3TTS(model_dir=Path("/fake/model/dir"))
    with pytest.raises(RuntimeError):
        engine.synthesize("hello")


def test_load_wraps_failure_in_load_error() -> None:
    engine = Qwen3TTS(model_dir=Path("/nonexistent/model/dir"))
    with patch("vocalis.engines.qwen3_tts.Qwen3TTSModel") as mock_model_cls:
        mock_model_cls.from_pretrained.side_effect = RuntimeError("CUDA out of memory")
        with pytest.raises(Qwen3TTSLoadError):
            engine.load()
    assert engine.is_loaded is False


def test_load_then_synthesize_calls_generate_custom_voice_with_configured_speaker() -> None:
    fake_wav = np.array([0.1, 0.2, -0.1], dtype=np.float32)
    fake_model = FakeQwenModel(wavs=[fake_wav], sample_rate=24000)
    engine = Qwen3TTS(model_dir=Path("/fake/model/dir"), speaker="Ryan", language="English")
    with patch("vocalis.engines.qwen3_tts.Qwen3TTSModel") as mock_model_cls:
        mock_model_cls.from_pretrained.return_value = fake_model
        engine.load()

    assert engine.is_loaded is True
    audio, sample_rate = engine.synthesize("Hello world")

    assert sample_rate == 24000
    np.testing.assert_array_almost_equal(audio, fake_wav)
    assert fake_model.calls == [{"text": "Hello world", "language": "English", "speaker": "Ryan"}]


def test_synthesize_propagates_generation_failure() -> None:
    fake_model = FakeQwenModel(wavs=[], sample_rate=24000, raise_exc=RuntimeError("generation failed"))
    engine = Qwen3TTS(model_dir=Path("/fake/model/dir"))
    with patch("vocalis.engines.qwen3_tts.Qwen3TTSModel") as mock_model_cls:
        mock_model_cls.from_pretrained.return_value = fake_model
        engine.load()
    with pytest.raises(RuntimeError):
        engine.synthesize("hello")


def test_unload_clears_loaded_state() -> None:
    fake_model = FakeQwenModel(wavs=[np.zeros(1, dtype=np.float32)], sample_rate=24000)
    engine = Qwen3TTS(model_dir=Path("/fake/model/dir"))
    with patch("vocalis.engines.qwen3_tts.Qwen3TTSModel") as mock_model_cls:
        mock_model_cls.from_pretrained.return_value = fake_model
        engine.load()
    engine.unload()
    assert engine.is_loaded is False
```

- [ ] **Step 4: Run the tests, verify they fail**

Run: `.venv/bin/python -m pytest tests/test_qwen3_tts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vocalis.engines.qwen3_tts'`

- [ ] **Step 5: Write the minimal implementation**

```python
# src/vocalis/engines/qwen3_tts.py
"""Qwen3-TTS text-to-speech engine — Quick Reader's primary voice.

Wraps the qwen-tts package's CustomVoice model (a fixed set of 9 preset
voices, selected by name — no reference audio or voice-cloning step
needed, unlike Qwen3-TTS's separate "Base" variant). Implements the
ModelEngine protocol so ModelManager's single-GPU-residency policy
applies to it exactly like Parakeet — this is the only Quick Reader
engine that touches the GPU; PiperTTS (the fallback) does not and is
never routed through ModelManager.
"""
import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from qwen_tts import Qwen3TTSModel

logger = logging.getLogger("vocalis")


class Qwen3TTSLoadError(Exception):
    """Raised when the Qwen3-TTS model fails to load."""


class Qwen3TTS:
    """Text-to-speech engine backed by the local Qwen3-TTS CustomVoice model."""

    def __init__(
        self,
        model_dir: Path,
        speaker: str = "Ryan",
        language: str = "English",
        use_gpu: bool = True,
    ) -> None:
        """Create a Qwen3TTS engine (does not load the model yet).

        Args:
            model_dir: Local directory the CustomVoice model files were
                downloaded into (see this task's one-time download step).
            speaker: Preset voice name (one of the 9 CustomVoice presets).
            language: Spoken language passed to generate_custom_voice().
            use_gpu: If True, load onto CUDA (the only supported device
                for Qwen3-TTS); False is only useful in tests that never
                call load().
        """
        self._model_dir = model_dir
        self._speaker = speaker
        self._language = language
        self._use_gpu = use_gpu
        self._model: Qwen3TTSModel | None = None

    @property
    def is_loaded(self) -> bool:
        """Whether the model is currently loaded."""
        return self._model is not None

    def load(self) -> None:
        """Load the model onto the GPU.

        Raises:
            Qwen3TTSLoadError: If loading fails for any reason (missing
                files, CUDA unavailable, out-of-memory, etc.).
        """
        try:
            self._model = Qwen3TTSModel.from_pretrained(
                str(self._model_dir),
                device_map="cuda:0" if self._use_gpu else "cpu",
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
        except Exception as exc:
            raise Qwen3TTSLoadError(
                f"Failed to load Qwen3-TTS model from {self._model_dir}: {exc}"
            ) from exc

    def unload(self) -> None:
        """Release the model reference. Safe to call even if never loaded."""
        self._model = None

    def synthesize(self, text: str) -> "tuple[npt.NDArray[np.float32], int]":
        """Generate speech audio for the given text using the preset voice.

        Args:
            text: The text to speak.

        Returns:
            (audio, sample_rate) — a 1-D float32 numpy array and its rate.

        Raises:
            RuntimeError: If called before load().
        """
        if self._model is None:
            raise RuntimeError("Qwen3TTS.synthesize() called before load()")
        wavs, sample_rate = self._model.generate_custom_voice(
            text=text, language=self._language, speaker=self._speaker
        )
        audio = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
        return audio, int(sample_rate)
```

Add this override to `pyproject.toml` (`qwen_tts` ships no type stubs,
matching the existing `sounddevice` override):

```toml
[[tool.mypy.overrides]]
module = "qwen_tts"
ignore_missing_imports = true
```

- [ ] **Step 6: Run the tests, verify they pass**

Run: `.venv/bin/python -m pytest tests/test_qwen3_tts.py -v`
Expected: all 6 tests PASS

- [ ] **Step 7: Run the full suite + mypy**

```bash
.venv/bin/python -m pytest
.venv/bin/python -m mypy src
```
Expected: all existing tests still pass (including Parakeet's — this is
the real regression check that the new dependencies didn't break
anything), coverage still ≥85%, mypy clean.

- [ ] **Step 8: Real smoke test with the actual model, and the changelog**

Run this once by hand:

```bash
.venv/bin/python -c "
from pathlib import Path
from vocalis.engines.qwen3_tts import Qwen3TTS

engine = Qwen3TTS(
    model_dir=Path.home() / '.local' / 'share' / 'vocalis' / 'models' / 'qwen3-tts-customvoice-0.6b',
)
engine.load()
audio, sr = engine.synthesize('Hello from Vocalis Phase 3, this is the primary voice.')
print('samples:', len(audio), 'sample_rate:', sr, 'max amplitude:', abs(audio).max())
"
```

Expected: prints a nonzero sample count, a sample rate, and a nonzero max
amplitude. Watch `nvidia-smi` during this call to confirm VRAM actually
rises. Write a changelog entry now, `.claude/2026-09-25_0200_CHANGELOG.md`,
covering: the qwen-tts/torch install (including whatever conflict
resolution Step 1 actually required — the real one, not a guess), the
one-time model download, and this smoke test's real output numbers.

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml src/vocalis/engines/qwen3_tts.py tests/test_qwen3_tts.py .claude/2026-09-25_0200_CHANGELOG.md
git commit -m "feat: add Qwen3TTS primary engine wrapper" -m "- Wraps qwen-tts's CustomVoice model (preset voice, no cloning) via generate_custom_voice()" -m "- Implements ModelEngine so it shares ModelManager's single-GPU-residency policy with Parakeet" -m "- Resolves the onnxruntime/torch dependency conflicts found during install (see changelog for exact resolution)" -m "- Verified end-to-end against the real downloaded model on GPU" -m "Phase 3 of vocalis (quick reader)" -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_0116ii4z7tzRCRhVAxecpkYC"
```

---

### Task 3: Audio Playback

**Files:**
- Create: `src/vocalis/audio/playback.py`
- Create: `tests/test_playback.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `PlaybackAlreadyActiveError(Exception)`
  - `AudioPlayer.__init__(self, stream_factory: Callable[..., Any] | None = None) -> None`
  - `AudioPlayer.is_playing -> bool` (property)
  - `AudioPlayer.play(self, audio: npt.NDArray[np.float32], sample_rate: int) -> None` — starts playback, returns immediately.
  - `AudioPlayer.wait_until_done(self) -> bool` — blocks until playback finishes naturally or `stop()` is called; returns `True` if it finished naturally, `False` if it was stopped early.
  - `AudioPlayer.stop(self) -> None` — stops playback immediately if in progress; no-op otherwise.
  - Task 6 (QuickReaderFeature) calls `play()` then `wait_until_done()` in
    its background worker thread, and calls `stop()` from `on_trigger()`
    when interrupting an in-progress reading.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_playback.py
"""Tests for vocalis.audio.playback.AudioPlayer.

Uses a FakeOutputStream in place of sounddevice.OutputStream so tests
never touch real speakers. Mirrors test_capture.py's FakeStream pattern.
"""
import threading
import time

import numpy as np
import pytest

from vocalis.audio.playback import AudioPlayer, PlaybackAlreadyActiveError


class FakeOutputStream:
    def __init__(self, samplerate, channels, dtype, callback, finished_callback=None) -> None:  # type: ignore[no-untyped-def]
        self.samplerate = samplerate
        self.callback = callback
        self.finished_callback = finished_callback
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True


def make_player() -> tuple[AudioPlayer, list[FakeOutputStream]]:
    streams: list[FakeOutputStream] = []

    def factory(**kwargs):  # type: ignore[no-untyped-def]
        stream = FakeOutputStream(**kwargs)
        streams.append(stream)
        return stream

    return AudioPlayer(stream_factory=factory), streams


def _run_callback_to_completion(stream: FakeOutputStream, audio: np.ndarray) -> None:
    """Drives a fake stream's callback exactly like real sounddevice would:
    repeatedly asks for chunks until the audio is exhausted, then calls
    finished_callback() (as real PortAudio does once the stream drains
    after the callback raises CallbackStop)."""
    frame_size = 4
    outdata = np.zeros((frame_size, 1), dtype=np.float32)
    while True:
        try:
            stream.callback(outdata, frame_size, None, None)
        except Exception:
            break
    if stream.finished_callback is not None:
        stream.finished_callback()


def test_play_starts_a_stream() -> None:
    player, streams = make_player()
    player.play(np.array([0.1, 0.2, 0.3], dtype=np.float32), 22050)
    assert len(streams) == 1
    assert streams[0].started is True
    assert player.is_playing is True


def test_play_while_already_playing_raises() -> None:
    player, _ = make_player()
    player.play(np.array([0.1], dtype=np.float32), 22050)
    with pytest.raises(PlaybackAlreadyActiveError):
        player.play(np.array([0.2], dtype=np.float32), 22050)


def test_wait_until_done_returns_true_after_natural_completion() -> None:
    player, streams = make_player()
    audio = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    player.play(audio, 22050)

    driver = threading.Thread(target=_run_callback_to_completion, args=(streams[0], audio))
    driver.start()
    finished_naturally = player.wait_until_done()
    driver.join()

    assert finished_naturally is True
    assert player.is_playing is False
    assert streams[0].closed is True


def test_stop_stops_playback_and_wait_until_done_returns_false() -> None:
    player, streams = make_player()
    player.play(np.array([0.1, 0.2, 0.3], dtype=np.float32), 22050)

    def stop_soon() -> None:
        time.sleep(0.05)
        player.stop()

    stopper = threading.Thread(target=stop_soon)
    stopper.start()
    finished_naturally = player.wait_until_done()
    stopper.join()

    assert finished_naturally is False
    assert player.is_playing is False
    assert streams[0].stopped is True


def test_stop_when_not_playing_is_a_no_op() -> None:
    player, _ = make_player()
    player.stop()  # must not raise
    assert player.is_playing is False


def test_callback_writes_audio_chunks_correctly() -> None:
    player, streams = make_player()
    audio = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    player.play(audio, 22050)
    stream = streams[0]

    outdata = np.zeros((3, 1), dtype=np.float32)
    stream.callback(outdata, 3, None, None)
    np.testing.assert_array_almost_equal(outdata[:, 0], audio[0:3])

    outdata2 = np.zeros((3, 1), dtype=np.float32)
    with pytest.raises(Exception):  # sd.CallbackStop signals natural end
        stream.callback(outdata2, 3, None, None)
    np.testing.assert_array_almost_equal(outdata2[:2, 0], audio[3:5])
```

- [ ] **Step 2: Run the tests, verify they fail**

Run: `.venv/bin/python -m pytest tests/test_playback.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vocalis.audio.playback'`

- [ ] **Step 3: Write the minimal implementation**

```python
# src/vocalis/audio/playback.py
"""Audio playback for Quick Reader.

Plays a float32 waveform through the default output device via
sounddevice (talks to PipeWire, the same way MicRecorder already
captures audio) instead of shelling out to a separate player process —
this makes playback easy to interrupt cleanly and easy to unit test with
an injectable fake, mirroring MicRecorder's stream_factory pattern.

Lock discipline: exactly like the fix for Phase 2's MicRecorder deadlock
(Task 9), the lock here only ever guards the swap of self._stream to/from
None — it is never held across a call into stream.stop()/close(), which
themselves block waiting on the audio callback thread.
"""
import logging
import threading
from typing import Any, Callable

import numpy as np
import numpy.typing as npt
import sounddevice as sd

logger = logging.getLogger("vocalis")


class PlaybackAlreadyActiveError(Exception):
    """Raised when play() is called while playback is already active."""


class AudioPlayer:
    """Plays a float32 waveform through the default output device."""

    def __init__(self, stream_factory: Callable[..., Any] | None = None) -> None:
        """Create an AudioPlayer.

        Args:
            stream_factory: Callable(**kwargs) -> stream object with
                start()/stop()/close() methods, used to construct the
                output stream. Defaults to sounddevice.OutputStream;
                injectable for tests.
        """
        self._stream_factory = stream_factory or sd.OutputStream
        self._stream: Any | None = None
        self._lock = threading.Lock()
        self._done_event = threading.Event()
        self._stopped_early = False

    @property
    def is_playing(self) -> bool:
        """Whether a playback stream is currently open."""
        return self._stream is not None

    def play(self, audio: npt.NDArray[np.float32], sample_rate: int) -> None:
        """Begin playing audio. Returns immediately; does not block.

        Args:
            audio: 1-D float32 numpy array of audio samples.
            sample_rate: Sample rate of `audio` in Hz.

        Raises:
            PlaybackAlreadyActiveError: If playback is already active.
        """
        if self.is_playing:
            raise PlaybackAlreadyActiveError("AudioPlayer.play() called while already playing")

        self._done_event.clear()
        self._stopped_early = False
        position = {"i": 0}

        def callback(
            outdata: npt.NDArray[np.float32], frames: int, time_info: object, status: object
        ) -> None:
            if status:
                logger.warning("Audio output status: %s", status)
            start = position["i"]
            end = start + frames
            chunk = audio[start:end]
            outdata[: len(chunk), 0] = chunk
            if len(chunk) < frames:
                outdata[len(chunk) :, 0] = 0
                position["i"] = len(audio)
                raise sd.CallbackStop()
            position["i"] = end

        def finished_callback() -> None:
            self._done_event.set()

        stream = self._stream_factory(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            callback=callback,
            finished_callback=finished_callback,
        )
        stream.start()
        self._stream = stream
        logger.info("Playback started (%.2fs, sample_rate=%d)", len(audio) / sample_rate, sample_rate)

    def wait_until_done(self) -> bool:
        """Block until playback finishes naturally or stop() is called.

        Returns:
            True if playback finished naturally; False if it was stopped
            early via stop().
        """
        self._done_event.wait()
        with self._lock:
            stream, self._stream = self._stream, None
        if stream is not None:
            stream.close()
        return not self._stopped_early

    def stop(self) -> None:
        """Stop playback immediately if active; no-op otherwise."""
        with self._lock:
            stream, self._stream = self._stream, None
        if stream is not None:
            self._stopped_early = True
            stream.stop()
            stream.close()
            self._done_event.set()
            logger.info("Playback stopped early")
```

- [ ] **Step 4: Run the tests, verify they pass**

Run: `.venv/bin/python -m pytest tests/test_playback.py -v`
Expected: all 6 tests PASS

- [ ] **Step 5: Run the full suite + mypy**

```bash
.venv/bin/python -m pytest
.venv/bin/python -m mypy src
```
Expected: all tests pass, coverage ≥85%, mypy clean.

- [ ] **Step 6: Real smoke test with real speakers**

```bash
.venv/bin/python -c "
import numpy as np
import time
from vocalis.audio.playback import AudioPlayer

player = AudioPlayer()
tone = (0.2 * np.sin(2 * np.pi * 440 * np.arange(22050) / 22050)).astype(np.float32)
player.play(tone, 22050)
finished = player.wait_until_done()
print('finished naturally:', finished)
"
```

Expected: you hear a one-second 440Hz tone, and it prints
`finished naturally: True`.

- [ ] **Step 7: Commit**

```bash
git add src/vocalis/audio/playback.py tests/test_playback.py
git commit -m "feat: add AudioPlayer for Quick Reader speech playback" -m "- sounddevice-based playback (not a pw-play subprocess), mirroring MicRecorder's injectable stream_factory pattern" -m "- play()/wait_until_done()/stop() lets callers interrupt playback cleanly" -m "- Lock only ever guards the stream-reference swap, never a blocking call, per Phase 2 Task 9's deadlock lesson" -m "Phase 3 of vocalis (quick reader)" -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_0116ii4z7tzRCRhVAxecpkYC"
```

---

### Task 4: Ctrl+Alt+R Hotkey Listener

**Files:**
- Create: `src/vocalis/hotkeys/ctrl_alt_r.py`
- Create: `tests/test_ctrl_alt_r.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `CtrlAltRListener.__init__(self, on_trigger: Callable[[], None], device_enumerator: Callable[[], list[EvdevDevice]]) -> None`
  - `CtrlAltRListener.start(self) -> None`
  - `CtrlAltRListener.stop(self) -> None`
  - `CtrlAltRListener.is_running -> bool` (property)
  - `list_hotkey_capable_devices() -> list[evdev.InputDevice]` — the real
    device enumerator, used by Task 7's CLI wiring.
  - Task 7 (CLI wiring) constructs
    `CtrlAltRListener(on_trigger=feature.on_trigger, device_enumerator=list_hotkey_capable_devices)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ctrl_alt_r.py
"""Tests for vocalis.hotkeys.ctrl_alt_r.CtrlAltRListener.

Uses a FakeDevice implementing the same read_loop()/close() pattern real
evdev.InputDevice objects use, so tests never touch real input hardware.
Mirrors test_numlock.py's FakeDevice/FakeEvent pattern exactly.
"""
import logging
import threading
import time
from typing import Iterator

import pytest

from vocalis.hotkeys.ctrl_alt_r import CtrlAltRListener

_EV_KEY = 1
_KEY_LEFTCTRL = 29
_KEY_RIGHTCTRL = 97
_KEY_LEFTALT = 56
_KEY_RIGHTALT = 100
_KEY_R = 19
_KEY_A = 30


class FakeEvent:
    def __init__(self, type_: int, code: int, value: int) -> None:
        self.type = type_
        self.code = code
        self.value = value


class FakeDevice:
    def __init__(self) -> None:
        self._pending: list[FakeEvent] = []
        self._closed = threading.Event()

    def queue_event(self, event: FakeEvent) -> None:
        self._pending.append(event)

    def read_loop(self) -> Iterator[FakeEvent]:
        while not self._closed.is_set():
            if self._pending:
                yield self._pending.pop(0)
            else:
                time.sleep(0.01)

    def close(self) -> None:
        self._closed.set()


def wait_for(flag: threading.Event, timeout: float = 1.0) -> None:
    assert flag.wait(timeout), "Expected callback was not invoked in time"


def _press(code: int) -> FakeEvent:
    return FakeEvent(_EV_KEY, code, 1)


def _release(code: int) -> FakeEvent:
    return FakeEvent(_EV_KEY, code, 0)


def test_no_devices_found_does_not_crash() -> None:
    triggered = threading.Event()
    listener = CtrlAltRListener(on_trigger=triggered.set, device_enumerator=lambda: [])
    listener.start()
    listener.stop()
    assert not triggered.is_set()
    assert listener.is_running is False


def test_ctrl_alt_r_together_triggers() -> None:
    triggered = threading.Event()
    device = FakeDevice()
    device.queue_event(_press(_KEY_LEFTCTRL))
    device.queue_event(_press(_KEY_LEFTALT))
    device.queue_event(_press(_KEY_R))
    listener = CtrlAltRListener(on_trigger=triggered.set, device_enumerator=lambda: [device])
    listener.start()
    try:
        wait_for(triggered)
    finally:
        listener.stop()


def test_r_alone_does_not_trigger() -> None:
    triggered = threading.Event()
    device = FakeDevice()
    device.queue_event(_press(_KEY_R))
    listener = CtrlAltRListener(on_trigger=triggered.set, device_enumerator=lambda: [device])
    listener.start()
    try:
        time.sleep(0.2)
        assert not triggered.is_set()
    finally:
        listener.stop()


def test_ctrl_r_without_alt_does_not_trigger() -> None:
    triggered = threading.Event()
    device = FakeDevice()
    device.queue_event(_press(_KEY_LEFTCTRL))
    device.queue_event(_press(_KEY_R))
    listener = CtrlAltRListener(on_trigger=triggered.set, device_enumerator=lambda: [device])
    listener.start()
    try:
        time.sleep(0.2)
        assert not triggered.is_set()
    finally:
        listener.stop()


def test_releasing_ctrl_then_pressing_r_does_not_trigger() -> None:
    triggered = threading.Event()
    device = FakeDevice()
    device.queue_event(_press(_KEY_LEFTCTRL))
    device.queue_event(_press(_KEY_LEFTALT))
    device.queue_event(_release(_KEY_LEFTCTRL))
    device.queue_event(_press(_KEY_R))
    listener = CtrlAltRListener(on_trigger=triggered.set, device_enumerator=lambda: [device])
    listener.start()
    try:
        time.sleep(0.2)
        assert not triggered.is_set()
    finally:
        listener.stop()


def test_right_ctrl_and_right_alt_also_trigger() -> None:
    triggered = threading.Event()
    device = FakeDevice()
    device.queue_event(_press(_KEY_RIGHTCTRL))
    device.queue_event(_press(_KEY_RIGHTALT))
    device.queue_event(_press(_KEY_R))
    listener = CtrlAltRListener(on_trigger=triggered.set, device_enumerator=lambda: [device])
    listener.start()
    try:
        wait_for(triggered)
    finally:
        listener.stop()


def test_unrelated_key_is_ignored() -> None:
    triggered = threading.Event()
    device = FakeDevice()
    device.queue_event(_press(_KEY_LEFTCTRL))
    device.queue_event(_press(_KEY_LEFTALT))
    device.queue_event(_press(_KEY_A))
    listener = CtrlAltRListener(on_trigger=triggered.set, device_enumerator=lambda: [device])
    listener.start()
    try:
        time.sleep(0.2)
        assert not triggered.is_set()
    finally:
        listener.stop()


def test_stop_terminates_background_threads() -> None:
    device = FakeDevice()
    listener = CtrlAltRListener(on_trigger=lambda: None, device_enumerator=lambda: [device])
    listener.start()
    assert listener.is_running is True
    listener.stop()
    assert listener.is_running is False


def test_callback_exception_does_not_kill_watcher_thread() -> None:
    triggers_seen: list[int] = []

    def failing_on_trigger() -> None:
        triggers_seen.append(1)
        raise RuntimeError("boom")

    device = FakeDevice()
    device.queue_event(_press(_KEY_LEFTCTRL))
    device.queue_event(_press(_KEY_LEFTALT))
    device.queue_event(_press(_KEY_R))
    device.queue_event(_release(_KEY_R))
    device.queue_event(_press(_KEY_R))
    listener = CtrlAltRListener(on_trigger=failing_on_trigger, device_enumerator=lambda: [device])
    listener.start()
    try:
        deadline = time.time() + 1.0
        while len(triggers_seen) < 2 and time.time() < deadline:
            time.sleep(0.01)
        assert len(triggers_seen) == 2
        assert listener.is_running is True
    finally:
        listener.stop()
```

- [ ] **Step 2: Run the tests, verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ctrl_alt_r.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vocalis.hotkeys.ctrl_alt_r'`

- [ ] **Step 3: Write the minimal implementation**

```python
# src/vocalis/hotkeys/ctrl_alt_r.py
"""Ctrl+Alt+R push-to-trigger hotkey listener for Quick Reader.

Built the same way as NumlockListener (Phase 2): watches raw evdev key
events without suppressing them, one background thread per
keyboard-capable device, firing on_trigger() when R is pressed while
both Ctrl and Alt are currently held on that same device. Held-state for
Ctrl/Alt is tracked per-device (inside each device's own watcher thread)
since a real combo is normally pressed on a single physical keyboard.

Two lessons from Phase 2 are applied here from the start rather than
discovered as bugs later (Tasks 9 and 10): the on_trigger() invocation is
wrapped in its own try/except inside the read loop (an exception must
never permanently kill the watcher thread), and Vocalis's own ydotoold
virtual output device is excluded from enumeration (the app must never
watch the device it uses to type/inject its own output).
"""
import logging
import threading
from typing import Callable, Iterable, Protocol

logger = logging.getLogger("vocalis")

_EV_KEY = 1
_KEY_LEFTCTRL = 29
_KEY_RIGHTCTRL = 97
_KEY_LEFTALT = 56
_KEY_RIGHTALT = 100
_KEY_R = 19
_EXCLUDED_DEVICE_NAMES = frozenset({"ydotoold virtual device"})


class EvdevEvent(Protocol):
    """The subset of an evdev InputEvent this module depends on."""

    type: int
    code: int
    value: int


class EvdevDevice(Protocol):
    """The subset of evdev.InputDevice this module depends on."""

    def read_loop(self) -> Iterable[EvdevEvent]:
        """Blocks, yielding events as they arrive, until close() is called."""
        ...

    def close(self) -> None:
        """Close the device, causing a blocked read_loop() to raise and exit."""
        ...


DeviceEnumerator = Callable[[], "list[EvdevDevice]"]


class CtrlAltRListener:
    """Calls on_trigger() when Ctrl+Alt+R is pressed."""

    def __init__(self, on_trigger: Callable[[], None], device_enumerator: DeviceEnumerator) -> None:
        """Create a CtrlAltRListener.

        Args:
            on_trigger: Called when Ctrl+Alt+R is pressed.
            device_enumerator: Returns the list of evdev devices to watch.
                Injectable so tests can supply fake devices instead of
                real hardware.
        """
        self._on_trigger = on_trigger
        self._device_enumerator = device_enumerator
        self._devices: list[EvdevDevice] = []
        self._threads: list[threading.Thread] = []

    @property
    def is_running(self) -> bool:
        """Whether any watcher thread is currently active."""
        return any(thread.is_alive() for thread in self._threads)

    def start(self) -> None:
        """Enumerate hotkey-capable devices and start watching each."""
        self._devices = self._device_enumerator()
        if not self._devices:
            logger.warning("No keyboard-capable input devices found")
            return
        self._threads = [
            threading.Thread(target=self._watch_device, args=(device,), daemon=True)
            for device in self._devices
        ]
        for thread in self._threads:
            thread.start()
        logger.info("Watching %d device(s) for Ctrl+Alt+R", len(self._devices))

    def stop(self) -> None:
        """Close all watched devices and wait for their threads to exit."""
        for device in self._devices:
            try:
                device.close()
            except Exception:
                logger.exception("Error closing input device")
        for thread in self._threads:
            thread.join(timeout=1.0)
            if thread.is_alive():
                logger.warning("Device watcher thread did not exit within timeout")
        self._devices = []
        self._threads = []

    def _watch_device(self, device: EvdevDevice) -> None:
        """Read events from one device until it is closed."""
        ctrl_held = False
        alt_held = False
        try:
            for event in device.read_loop():
                if event.type != _EV_KEY:
                    continue
                try:
                    if event.code in (_KEY_LEFTCTRL, _KEY_RIGHTCTRL):
                        ctrl_held = event.value != 0
                    elif event.code in (_KEY_LEFTALT, _KEY_RIGHTALT):
                        alt_held = event.value != 0
                    elif event.code == _KEY_R and event.value == 1 and ctrl_held and alt_held:
                        self._on_trigger()
                except Exception:
                    logger.exception("Error in Ctrl+Alt+R callback; continuing to watch device")
        except OSError:
            pass  # device closed by stop(); expected exit path
        except Exception:
            logger.exception("Error reading input device events")


def list_hotkey_capable_devices() -> "list[EvdevDevice]":
    """Enumerate real input devices that report Ctrl+Alt+R key capability.

    Excludes Vocalis's own ydotoold virtual output device, closing any
    device that doesn't qualify immediately since it won't be watched.
    """
    import evdev

    devices: list[EvdevDevice] = []
    needed = {_KEY_LEFTCTRL, _KEY_LEFTALT, _KEY_R}
    for path in evdev.list_devices():
        device = evdev.InputDevice(path)
        capabilities = set(device.capabilities().get(evdev.ecodes.EV_KEY, []))
        if needed.issubset(capabilities) and device.name not in _EXCLUDED_DEVICE_NAMES:
            devices.append(device)
        else:
            device.close()
    return devices
```

- [ ] **Step 4: Run the tests, verify they pass**

Run: `.venv/bin/python -m pytest tests/test_ctrl_alt_r.py -v`
Expected: all 9 tests PASS

- [ ] **Step 5: Run the full suite + mypy**

```bash
.venv/bin/python -m pytest
.venv/bin/python -m mypy src
```
Expected: all tests pass, coverage ≥85%, mypy clean.

- [ ] **Step 6: Real-hardware smoke test**

```bash
.venv/bin/python -c "
from vocalis.hotkeys.ctrl_alt_r import list_hotkey_capable_devices
devices = list_hotkey_capable_devices()
print('found', len(devices), 'device(s):', [d.name for d in devices])
"
```

Expected: lists your real keyboard device(s) and does **not** list
`ydotoold virtual device`. Then, with `ydotoold` running, inject a
synthetic combo and confirm it fires:

```bash
ydotool key 29:1 56:1 19:1 19:0 56:0 29:0
```

(codes: 29=leftctrl, 56=leftalt, 19=r; `:1`=press, `:0`=release) while a
small script using `CtrlAltRListener` with the real
`list_hotkey_capable_devices` and a `print`-based `on_trigger` is running,
and confirm it prints once. Record the result in this task's commit.

- [ ] **Step 7: Commit**

```bash
git add src/vocalis/hotkeys/ctrl_alt_r.py tests/test_ctrl_alt_r.py
git commit -m "feat: add CtrlAltRListener for Quick Reader hotkey detection" -m "- evdev-based, same proven pattern as NumlockListener, per-device Ctrl/Alt held-state tracking" -m "- Applies Phase 2's Task 9/Task 10 lessons from the start: per-callback exception isolation, ydotoold device exclusion" -m "- Verified against real hardware: real device enumeration excludes ydotoold, synthetic ydotool-injected combo fires correctly" -m "Phase 3 of vocalis (quick reader)" -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_0116ii4z7tzRCRhVAxecpkYC"
```

---

### Task 5: Clipboard Reader

**Files:**
- Create: `src/vocalis/output/clipboard.py`
- Create: `tests/test_clipboard.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `ClipboardReader.__init__(self, command_runner: CommandRunner | None = None) -> None`
  - `ClipboardReader.read(self) -> str` — never raises; returns `""` on
    any failure (empty clipboard, xclip missing, timeout).
  - Task 6 (QuickReaderFeature) calls `.read()` at the start of each
    trigger cycle.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_clipboard.py
"""Tests for vocalis.output.clipboard.ClipboardReader.

Uses a FakeCommandRunner instead of a real xclip subprocess.
"""
import subprocess

from vocalis.output.clipboard import ClipboardReader


class FakeCompletedProcess:
    def __init__(self, returncode: int, stdout: bytes = b"") -> None:
        self.returncode = returncode
        self.stdout = stdout


class FakeCommandRunner:
    def __init__(self, returncode: int = 0, stdout: bytes = b"", raise_exc: Exception | None = None) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.raise_exc = raise_exc
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str]) -> FakeCompletedProcess:
        self.calls.append(argv)
        if self.raise_exc is not None:
            raise self.raise_exc
        return FakeCompletedProcess(returncode=self.returncode, stdout=self.stdout)


def test_read_returns_clipboard_text() -> None:
    runner = FakeCommandRunner(returncode=0, stdout=b"hello world")
    reader = ClipboardReader(command_runner=runner)
    assert reader.read() == "hello world"
    assert runner.calls == [["xclip", "-o", "-selection", "clipboard"]]


def test_read_returns_empty_string_on_nonzero_exit() -> None:
    # xclip exits nonzero when the clipboard is empty/unavailable.
    runner = FakeCommandRunner(returncode=1, stdout=b"")
    reader = ClipboardReader(command_runner=runner)
    assert reader.read() == ""


def test_read_returns_empty_string_when_runner_raises() -> None:
    runner = FakeCommandRunner(raise_exc=OSError("xclip not found"))
    reader = ClipboardReader(command_runner=runner)
    assert reader.read() == ""


def test_read_returns_empty_string_on_timeout() -> None:
    runner = FakeCommandRunner(raise_exc=subprocess.TimeoutExpired(cmd="xclip", timeout=5.0))
    reader = ClipboardReader(command_runner=runner)
    assert reader.read() == ""


def test_read_decodes_utf8_with_replacement_on_bad_bytes() -> None:
    runner = FakeCommandRunner(returncode=0, stdout=b"caf\xc3\xa9")
    reader = ClipboardReader(command_runner=runner)
    assert reader.read() == "café"
```

- [ ] **Step 2: Run the tests, verify they fail**

Run: `.venv/bin/python -m pytest tests/test_clipboard.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vocalis.output.clipboard'`

- [ ] **Step 3: Write the minimal implementation**

```python
# src/vocalis/output/clipboard.py
"""Reads clipboard text for Quick Reader — the read-side counterpart to
TextOutput's clipboard-write fallback, using the same xclip-based
CommandRunner seam.
"""
import logging
import subprocess
from typing import Protocol

logger = logging.getLogger("vocalis")


class CommandRunner(Protocol):
    """Runs a command and captures its output."""

    def __call__(self, argv: list[str]) -> "subprocess.CompletedProcess[bytes]": ...


class ClipboardReader:
    """Reads the current clipboard contents as text."""

    def __init__(self, command_runner: CommandRunner | None = None) -> None:
        """Create a ClipboardReader.

        Args:
            command_runner: Callable(argv) -> CompletedProcess, used to
                run xclip. Defaults to subprocess.run with a 5-second
                timeout. Injectable for tests.
        """
        self._run: CommandRunner = command_runner or _default_command_runner

    def read(self) -> str:
        """Read the current clipboard contents.

        Returns:
            The clipboard's text contents, or "" if the clipboard is
            empty or xclip fails for any reason. Never raises.
        """
        try:
            result = self._run(["xclip", "-o", "-selection", "clipboard"])
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning("Failed to read clipboard: %s", exc)
            return ""
        if result.returncode != 0:
            return ""
        return result.stdout.decode("utf-8", errors="replace")


def _default_command_runner(argv: list[str]) -> "subprocess.CompletedProcess[bytes]":
    """Default CommandRunner: subprocess.run with a 5-second timeout."""
    return subprocess.run(argv, timeout=5.0, capture_output=True)
```

- [ ] **Step 4: Run the tests, verify they pass**

Run: `.venv/bin/python -m pytest tests/test_clipboard.py -v`
Expected: all 5 tests PASS

- [ ] **Step 5: Run the full suite + mypy**

```bash
.venv/bin/python -m pytest
.venv/bin/python -m mypy src
```
Expected: all tests pass, coverage ≥85%, mypy clean.

- [ ] **Step 6: Commit**

```bash
git add src/vocalis/output/clipboard.py tests/test_clipboard.py
git commit -m "feat: add ClipboardReader for Quick Reader text acquisition" -m "- Reads clipboard text via xclip, mirroring TextOutput's write-side CommandRunner seam" -m "- Never raises: empty/unavailable clipboard and any xclip failure both resolve to an empty string" -m "Phase 3 of vocalis (quick reader)" -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_0116ii4z7tzRCRhVAxecpkYC"
```

---

### Task 6: QuickReaderFeature Orchestration

This is the task that ties Tasks 1-5 together, and is the most
architecturally important one: it must **never block the hotkey listener
thread**, so that a second Ctrl+Alt+R press can interrupt an in-progress
reading immediately. All real work (clipboard read, speech generation,
playback) happens in a background thread spawned per trigger; `on_trigger()`
itself only ever does a quick lock-guarded flag check plus, when
interrupting, a call to `AudioPlayer.stop()` (itself non-blocking).

**Files:**
- Create: `src/vocalis/features/quick_reader.py`
- Create: `tests/test_quick_reader.py`

**Interfaces:**
- Consumes:
  - `ModelManager.acquire(engine_id: str, factory: Callable[[], ModelEngine]) -> ModelEngine`
    and `.release(engine_id: str) -> None` (Phase 1, unchanged).
  - `PiperTTS.synthesize(text: str) -> tuple[NDArray[float32], int]` (Task 1).
  - `Qwen3TTS.synthesize(text: str) -> tuple[NDArray[float32], int]` via an
    engine factory, exactly like `ParakeetSTT` in Phase 2 (Task 2).
  - `AudioPlayer.play/wait_until_done/stop` (Task 3).
  - `ClipboardReader.read() -> str` (Task 5).
  - `Notifier.notify(title: str, message: str) -> None` (Phase 2, unchanged).
- Produces:
  - `QuickReaderFeature.__init__(self, model_manager: ModelManager, primary_engine_factory: Callable[[], ModelEngine], fallback_engine: Synthesizer, clipboard_reader: ClipboardReader, player: AudioPlayer, notifier: Notifier, language: str = "English") -> None`
  - `QuickReaderFeature.on_trigger(self) -> None`
  - Task 7 (CLI wiring) constructs this and wires
    `CtrlAltRListener(on_trigger=feature.on_trigger, ...)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_quick_reader.py
"""Tests for vocalis.features.quick_reader.QuickReaderFeature.

Uses fakes for every collaborator — this suite verifies orchestration
logic only, the same approach test_dictation.py already established.
"""
import threading
import time

import numpy as np
import numpy.typing as npt

from vocalis.features.quick_reader import QuickReaderFeature
from vocalis.model_manager import ModelManager


class FakeSynthesizer:
    def __init__(
        self,
        audio: npt.NDArray[np.float32],
        sample_rate: int = 22050,
        raise_exc: Exception | None = None,
        delay_seconds: float = 0.0,
    ) -> None:
        self.audio = audio
        self.sample_rate = sample_rate
        self.raise_exc = raise_exc
        self.delay_seconds = delay_seconds
        self.calls: list[str] = []
        self._loaded = False

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def synthesize(self, text: str):  # type: ignore[no-untyped-def]
        self.calls.append(text)
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.audio, self.sample_rate


class FakeClipboardReader:
    def __init__(self, text: str) -> None:
        self.text = text

    def read(self) -> str:
        return self.text


class FakePlayer:
    def __init__(self, delay_seconds: float = 0.0, raise_on_play: Exception | None = None) -> None:
        self.delay_seconds = delay_seconds
        self.raise_on_play = raise_on_play
        self.play_calls: list[tuple[object, int]] = []
        self.stop_calls = 0
        self._stopped = threading.Event()

    def play(self, audio, sample_rate: int) -> None:  # type: ignore[no-untyped-def]
        self.play_calls.append((audio, sample_rate))
        if self.raise_on_play is not None:
            raise self.raise_on_play

    def wait_until_done(self) -> bool:
        if self._stopped.wait(timeout=self.delay_seconds or 0.05):
            return False
        return True

    def stop(self) -> None:
        self.stop_calls += 1
        self._stopped.set()


class FakeNotifier:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def notify(self, title: str, message: str = "") -> None:
        self.messages.append((title, message))


def make_feature(
    primary: FakeSynthesizer,
    fallback: FakeSynthesizer,
    clipboard_text: str,
    player: FakePlayer | None = None,
):
    manager = ModelManager()
    notifier = FakeNotifier()
    player = player or FakePlayer()
    feature = QuickReaderFeature(
        model_manager=manager,
        primary_engine_factory=lambda: primary,  # type: ignore[arg-type]
        fallback_engine=fallback,  # type: ignore[arg-type]
        clipboard_reader=FakeClipboardReader(clipboard_text),  # type: ignore[arg-type]
        player=player,  # type: ignore[arg-type]
        notifier=notifier,  # type: ignore[arg-type]
    )
    return feature, primary, fallback, player, notifier, manager


def wait_until(predicate, timeout: float = 1.0) -> None:  # type: ignore[no-untyped-def]
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("Condition was not met in time")


def test_trigger_with_empty_clipboard_notifies_and_does_nothing() -> None:
    primary = FakeSynthesizer(audio=np.zeros(1, dtype=np.float32))
    fallback = FakeSynthesizer(audio=np.zeros(1, dtype=np.float32))
    feature, _, _, player, notifier, _ = make_feature(primary, fallback, clipboard_text="")

    feature.on_trigger()
    wait_until(lambda: any("Nothing copied" in msg for _, msg in notifier.messages))

    assert player.play_calls == []
    assert primary.calls == []


def test_trigger_reads_speaks_and_notifies_done() -> None:
    audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    primary = FakeSynthesizer(audio=audio, sample_rate=24000)
    fallback = FakeSynthesizer(audio=np.zeros(1, dtype=np.float32))
    feature, _, _, player, notifier, _ = make_feature(primary, fallback, clipboard_text="hello there")

    feature.on_trigger()
    wait_until(lambda: any(msg == "Done" for _, msg in notifier.messages))

    assert primary.calls == ["hello there"]
    assert player.play_calls == [(audio, 24000)]
    assert any(msg == "Reading..." for _, msg in notifier.messages)
    assert any(msg == "hello there" for _, msg in notifier.messages)


def test_primary_failure_falls_back_to_secondary_engine() -> None:
    primary = FakeSynthesizer(audio=np.zeros(1, dtype=np.float32), raise_exc=RuntimeError("cuda oom"))
    fallback_audio = np.array([0.4, 0.5], dtype=np.float32)
    fallback = FakeSynthesizer(audio=fallback_audio, sample_rate=22050)
    feature, _, _, player, notifier, _ = make_feature(primary, fallback, clipboard_text="fallback please")

    feature.on_trigger()
    wait_until(lambda: any(msg == "Done" for _, msg in notifier.messages))

    assert fallback.calls == ["fallback please"]
    assert player.play_calls == [(fallback_audio, 22050)]
    assert any("backup voice" in msg.lower() for _, msg in notifier.messages)


def test_both_engines_failing_notifies_failure_and_does_not_crash() -> None:
    primary = FakeSynthesizer(audio=np.zeros(1, dtype=np.float32), raise_exc=RuntimeError("cuda oom"))
    fallback = FakeSynthesizer(audio=np.zeros(1, dtype=np.float32), raise_exc=RuntimeError("piper missing"))
    feature, _, _, player, notifier, _ = make_feature(primary, fallback, clipboard_text="uh oh")

    feature.on_trigger()  # must not raise
    wait_until(lambda: any("Could not generate speech" in msg for _, msg in notifier.messages))

    assert player.play_calls == []


def test_second_trigger_while_playing_stops_and_notifies_stopped() -> None:
    audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    primary = FakeSynthesizer(audio=audio)
    fallback = FakeSynthesizer(audio=np.zeros(1, dtype=np.float32))
    player = FakePlayer(delay_seconds=2.0)  # wait_until_done() would block a long time unless stopped
    feature, _, _, _, notifier, _ = make_feature(primary, fallback, clipboard_text="a long passage", player=player)

    feature.on_trigger()
    wait_until(lambda: len(player.play_calls) == 1)
    feature.on_trigger()  # the interrupting press

    wait_until(lambda: any(msg == "Stopped" for _, msg in notifier.messages))
    assert player.stop_calls == 1
    assert not any(msg == "Done" for _, msg in notifier.messages)


def test_second_trigger_while_synthesizing_cancels_before_playback_starts() -> None:
    primary = FakeSynthesizer(audio=np.zeros(1, dtype=np.float32), delay_seconds=0.3)
    fallback = FakeSynthesizer(audio=np.zeros(1, dtype=np.float32))
    feature, _, _, player, notifier, _ = make_feature(primary, fallback, clipboard_text="slow one")

    feature.on_trigger()
    wait_until(lambda: any(msg == "Reading..." for _, msg in notifier.messages))
    feature.on_trigger()  # interrupt while still synthesizing

    wait_until(lambda: any(msg == "Stopped" for _, msg in notifier.messages))
    time.sleep(0.5)  # let the background synthesis finish
    assert player.play_calls == []  # playback must never start after being cancelled


def test_third_trigger_after_stop_starts_a_fresh_reading() -> None:
    audio = np.array([0.1], dtype=np.float32)
    primary = FakeSynthesizer(audio=audio)
    fallback = FakeSynthesizer(audio=np.zeros(1, dtype=np.float32))
    feature, _, _, player, notifier, _ = make_feature(primary, fallback, clipboard_text="again")

    feature.on_trigger()
    wait_until(lambda: len(player.play_calls) == 1)
    feature.on_trigger()  # stop
    wait_until(lambda: player.stop_calls == 1)
    feature.on_trigger()  # fresh start
    wait_until(lambda: len(player.play_calls) == 2)
```

- [ ] **Step 2: Run the tests, verify they fail**

Run: `.venv/bin/python -m pytest tests/test_quick_reader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vocalis.features.quick_reader'`

- [ ] **Step 3: Write the minimal implementation**

```python
# src/vocalis/features/quick_reader.py
"""Orchestrates Quick Reader: on Ctrl+Alt+R, read the clipboard, speak it
via Qwen3-TTS (falling back to Piper on any failure), and play it back —
with desktop notifications at each stage, and press-again-to-stop.

Never blocks the caller (the hotkey listener's watcher thread): on_trigger()
either spawns a background thread to do the real work, or (when
interrupting an in-progress reading) makes only quick, non-blocking calls
itself. This is essential for "press again to stop" to actually work —
evdev's read_loop() processes one event at a time per device, so if
on_trigger() ever blocked until a reading finished, a second press could
never be noticed until the first reading was already over.
"""
import logging
import threading
from typing import Callable, Protocol, cast

import numpy as np
import numpy.typing as npt

from vocalis.audio.playback import AudioPlayer
from vocalis.model_manager import ModelEngine, ModelManager
from vocalis.notify import Notifier
from vocalis.output.clipboard import ClipboardReader

logger = logging.getLogger("vocalis")

_ENGINE_ID = "quick_reader_tts"
_TITLE = "Vocalis — Quick Reader"


class Synthesizer(Protocol):
    """The subset of a TTS engine this module depends on."""

    def synthesize(self, text: str) -> "tuple[npt.NDArray[np.float32], int]": ...


class QuickReaderFeature:
    """Wires Ctrl+Alt+R to read -> speak (with fallback) -> play."""

    def __init__(
        self,
        model_manager: ModelManager,
        primary_engine_factory: Callable[[], ModelEngine],
        fallback_engine: Synthesizer,
        clipboard_reader: ClipboardReader,
        player: AudioPlayer,
        notifier: Notifier,
        language: str = "English",
    ) -> None:
        """Create a QuickReaderFeature.

        Args:
            model_manager: Shared ModelManager enforcing single-residency
                (used only for the primary, GPU-backed engine).
            primary_engine_factory: Constructs a fresh primary engine
                (e.g. Qwen3TTS) when it needs to be (re)loaded. The
                constructed engine must also implement Synthesizer.
            fallback_engine: An already-loaded, CPU-only engine (e.g.
                PiperTTS) used when the primary engine fails. Never
                routed through ModelManager since it never touches the GPU.
            clipboard_reader: Reads the text to speak.
            player: Plays the generated speech and can be interrupted.
            notifier: Sends start/live/done/stopped notifications.
            language: Spoken language passed to the primary engine.
        """
        self._model_manager = model_manager
        self._primary_engine_factory = primary_engine_factory
        self._fallback_engine = fallback_engine
        self._clipboard_reader = clipboard_reader
        self._player = player
        self._notifier = notifier
        self._language = language
        self._lock = threading.Lock()
        self._active = False

    def on_trigger(self) -> None:
        """Call when Ctrl+Alt+R is pressed: start reading, or stop if
        already reading. Never blocks."""
        with self._lock:
            if self._active:
                self._active = False
                self._player.stop()
                self._notifier.notify(_TITLE, "Stopped")
                return
            self._active = True
        threading.Thread(target=self._read_aloud, daemon=True).start()

    def _is_still_active(self) -> bool:
        with self._lock:
            return self._active

    def _deactivate(self) -> None:
        with self._lock:
            self._active = False

    def _read_aloud(self) -> None:
        text = self._clipboard_reader.read()
        if not self._is_still_active():
            return  # cancelled while reading the clipboard
        if not text.strip():
            self._notifier.notify(_TITLE, "Nothing copied")
            self._deactivate()
            return

        self._notifier.notify(_TITLE, "Reading...")
        try:
            audio, sample_rate = self._synthesize_with_fallback(text)
        except Exception:
            logger.exception("Both TTS engines failed")
            self._notifier.notify(_TITLE, "Could not generate speech")
            self._deactivate()
            return
        if not self._is_still_active():
            return  # cancelled while synthesizing; must not start playback

        self._notifier.notify(_TITLE, text)
        try:
            self._player.play(audio, sample_rate)
            finished_naturally = self._player.wait_until_done()
        except Exception:
            logger.exception("Playback failed")
            self._notifier.notify(_TITLE, "Playback failed")
            self._deactivate()
            return

        self._deactivate()
        if finished_naturally:
            self._notifier.notify(_TITLE, "Done")

    def _synthesize_with_fallback(self, text: str) -> "tuple[npt.NDArray[np.float32], int]":
        """Try the primary engine; fall back to the secondary on any failure."""
        try:
            engine = self._model_manager.acquire(_ENGINE_ID, self._primary_engine_factory)
            try:
                return cast(Synthesizer, engine).synthesize(text)
            finally:
                self._model_manager.release(_ENGINE_ID)
        except Exception:
            logger.exception("Primary TTS engine failed; falling back to Piper")
            self._notifier.notify(_TITLE, "Primary voice unavailable — using backup voice")
            return self._fallback_engine.synthesize(text)
```

- [ ] **Step 4: Run the tests, verify they pass**

Run: `.venv/bin/python -m pytest tests/test_quick_reader.py -v`
Expected: all 7 tests PASS

- [ ] **Step 5: Run the full suite + mypy**

```bash
.venv/bin/python -m pytest
.venv/bin/python -m mypy src
```
Expected: all tests pass, coverage ≥85%, mypy clean.

- [ ] **Step 6: Commit**

```bash
git add src/vocalis/features/quick_reader.py tests/test_quick_reader.py
git commit -m "feat: add QuickReaderFeature orchestrating clipboard -> speech -> playback" -m "- Never blocks the hotkey listener thread: real work runs in a spawned background thread per trigger" -m "- Primary engine (Qwen3-TTS, via ModelManager) with automatic fallback to Piper (never GPU-managed) on any failure" -m "- Press-again-to-stop cancels cleanly whether it's still reading the clipboard, still synthesizing, or already playing" -m "Phase 3 of vocalis (quick reader)" -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_0116ii4z7tzRCRhVAxecpkYC"
```

---

### Task 7: Wire Quick Reader into the CLI/Daemon

**Files:**
- Modify: `src/vocalis/config.py`
- Modify: `config/default_config.toml`
- Modify: `src/vocalis/cli.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: everything from Tasks 1-6.
- Produces:
  - `EnginesConfig.qwen3_tts_model_dir: str` (new field)
  - `EnginesConfig.qwen3_tts_speaker: str` (new field, default `"Ryan"`)
  - `EnginesConfig.piper_binary_path: str` (new field)
  - `EnginesConfig.piper_voice_model_path: str` (new field)
  - `build_quick_reader(config: VocalisConfig, model_manager: ModelManager) -> tuple[QuickReaderFeature, CtrlAltRListener]`
  - `main()` now also builds and starts/stops the Quick Reader listener
    alongside the daemon and the dictation listener.

- [ ] **Step 1: Write the failing test for the new config fields**

```python
# tests/test_config.py — add these, alongside the existing tests
def test_engines_config_has_qwen3_tts_and_piper_defaults() -> None:
    config = VocalisConfig()
    assert config.engines.qwen3_tts_speaker == "Ryan"
    assert "qwen3-tts-customvoice-0.6b" in config.engines.qwen3_tts_model_dir
    assert "piper" in config.engines.piper_binary_path
    assert "piper-voices" in config.engines.piper_voice_model_path
```

- [ ] **Step 2: Run the test, verify it fails**

Run: `.venv/bin/python -m pytest tests/test_config.py -k qwen3_tts_and_piper -v`
Expected: FAIL with `AttributeError: 'EnginesConfig' object has no attribute 'qwen3_tts_speaker'`

- [ ] **Step 3: Add the config fields**

In `src/vocalis/config.py`, add these two functions above `EnginesConfig`
(alongside the existing `_default_parakeet_model_dir`):

```python
def _default_qwen3_tts_model_dir() -> str:
    """Default location of the downloaded Qwen3-TTS CustomVoice model
    files (see Phase 3's Task 2 one-time download step)."""
    return str(Path.home() / ".local" / "share" / "vocalis" / "models" / "qwen3-tts-customvoice-0.6b")


def _default_piper_binary_path() -> str:
    """Default location of the existing Piper binary (already installed
    from the legacy Voxtype setup)."""
    return str(Path.home() / ".local" / "bin" / "piper" / "piper")


def _default_piper_voice_model_path() -> str:
    """Default location of the existing Piper voice model (already
    installed from the legacy Voxtype setup)."""
    return str(Path.home() / ".local" / "share" / "piper-voices" / "en_US-lessac-medium.onnx")
```

Then change `EnginesConfig` from:

```python
class EnginesConfig(BaseModel):
    """Which engine implementation each feature should use."""

    stt: str = "parakeet"
    quick_tts: str = "qwen3_tts"
    doc_tts: str = "fish_speech"
    parakeet_model_dir: str = Field(default_factory=_default_parakeet_model_dir)
```

to:

```python
class EnginesConfig(BaseModel):
    """Which engine implementation each feature should use."""

    stt: str = "parakeet"
    quick_tts: str = "qwen3_tts"
    doc_tts: str = "fish_speech"
    parakeet_model_dir: str = Field(default_factory=_default_parakeet_model_dir)
    qwen3_tts_model_dir: str = Field(default_factory=_default_qwen3_tts_model_dir)
    qwen3_tts_speaker: str = "Ryan"
    piper_binary_path: str = Field(default_factory=_default_piper_binary_path)
    piper_voice_model_path: str = Field(default_factory=_default_piper_voice_model_path)
```

- [ ] **Step 4: Run the test, verify it passes**

Run: `.venv/bin/python -m pytest tests/test_config.py -v`
Expected: all config tests PASS

- [ ] **Step 5: Update the default config TOML**

In `config/default_config.toml`, find the `[engines]` section and add:

```toml
qwen3_tts_speaker = "Ryan"
```

(The three path fields are intentionally left out of the shipped
template, exactly like `parakeet_model_dir` already is — they're
computed per-user defaults via `Path.home()`, not baked into the
template.)

- [ ] **Step 6: Run the full config test suite**

Run: `.venv/bin/python -m pytest tests/test_config.py -v`
Expected: all PASS

- [ ] **Step 7: Write the failing tests for the CLI wiring changes**

```python
# tests/test_cli.py — add these, alongside the existing tests
from unittest.mock import patch

def test_build_quick_reader_returns_feature_and_listener() -> None:
    with patch("vocalis.cli.list_hotkey_capable_devices", return_value=[]):
        config = VocalisConfig()
        manager = ModelManager()
        feature, listener = build_quick_reader(config=config, model_manager=manager)
        assert feature is not None
        assert listener is not None


def test_main_starts_and_stops_quick_reader_listener() -> None:
    with (
        patch("vocalis.cli.list_numlock_capable_devices", return_value=[]),
        patch("vocalis.cli.list_hotkey_capable_devices", return_value=[]),
        patch("vocalis.cli.load_config", return_value=VocalisConfig()),
        patch("vocalis.cli.VocalisDaemon") as mock_daemon_cls,
        patch("vocalis.cli._block_until_interrupted", side_effect=KeyboardInterrupt),
    ):
        main()
        mock_daemon_cls.return_value.start.assert_called_once()
        mock_daemon_cls.return_value.stop.assert_called_once()
```

- [ ] **Step 8: Run the tests, verify they fail**

Run: `.venv/bin/python -m pytest tests/test_cli.py -k "quick_reader" -v`
Expected: FAIL with `ImportError: cannot import name 'build_quick_reader'`

- [ ] **Step 9: Add the `build_quick_reader` function and wire it into `main()`**

In `src/vocalis/cli.py`, add these imports alongside the existing ones:

```python
from vocalis.audio.playback import AudioPlayer
from vocalis.engines.piper_tts import PiperTTS
from vocalis.engines.qwen3_tts import Qwen3TTS
from vocalis.features.quick_reader import QuickReaderFeature
from vocalis.hotkeys.ctrl_alt_r import CtrlAltRListener, list_hotkey_capable_devices
from vocalis.output.clipboard import ClipboardReader
```

Add this function, placed after `build_dictation`:

```python
def build_quick_reader(
    config: VocalisConfig, model_manager: ModelManager
) -> tuple[QuickReaderFeature, CtrlAltRListener]:
    """Construct the Quick Reader feature and its Ctrl+Alt+R listener.

    Args:
        config: Loaded, validated vocalis configuration.
        model_manager: The shared ModelManager (same instance passed to
            build_daemon() and build_dictation()) — only the primary
            (Qwen3-TTS) engine is routed through it; the Piper fallback
            never touches the GPU and is constructed/loaded directly.

    Returns:
        A (feature, listener) pair. The caller is responsible for calling
        listener.start()/listener.stop() around the daemon's own lifecycle.
    """
    qwen_model_dir = Path(config.engines.qwen3_tts_model_dir)
    speaker = config.engines.qwen3_tts_speaker

    def primary_engine_factory() -> Qwen3TTS:
        return Qwen3TTS(model_dir=qwen_model_dir, speaker=speaker)

    fallback_engine = PiperTTS(
        binary_path=Path(config.engines.piper_binary_path),
        model_path=Path(config.engines.piper_voice_model_path),
    )
    fallback_engine.load()

    feature = QuickReaderFeature(
        model_manager=model_manager,
        primary_engine_factory=primary_engine_factory,
        fallback_engine=fallback_engine,
        clipboard_reader=ClipboardReader(),
        player=AudioPlayer(),
        notifier=Notifier(enabled=config.notifications_enabled),
    )
    listener = CtrlAltRListener(
        on_trigger=feature.on_trigger,
        device_enumerator=list_hotkey_capable_devices,
    )
    return feature, listener
```

Then change `main()` from:

```python
    config = load_config(default_config_path())
    model_manager = ModelManager(idle_timeout_seconds=config.model_idle_timeout_seconds)
    daemon = build_daemon(model_manager=model_manager, socket_path=_socket_path())
    _, listener = build_dictation(config=config, model_manager=model_manager)
    try:
        daemon.start()
        listener.start()
        _block_until_interrupted()
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()
        daemon.stop()
```

to:

```python
    config = load_config(default_config_path())
    model_manager = ModelManager(idle_timeout_seconds=config.model_idle_timeout_seconds)
    daemon = build_daemon(model_manager=model_manager, socket_path=_socket_path())
    _, dictation_listener = build_dictation(config=config, model_manager=model_manager)
    _, quick_reader_listener = build_quick_reader(config=config, model_manager=model_manager)
    try:
        daemon.start()
        dictation_listener.start()
        quick_reader_listener.start()
        _block_until_interrupted()
    except KeyboardInterrupt:
        pass
    finally:
        quick_reader_listener.stop()
        dictation_listener.stop()
        daemon.stop()
```

(Note: `PiperTTS.load()` here can raise `PiperLoadError` if the binary or
voice model files are somehow missing on this machine — that's an
intentional fail-fast at startup, the same way a missing Parakeet model
directory would surface immediately rather than silently at first use.)

- [ ] **Step 10: Run the tests, verify they pass**

Run: `.venv/bin/python -m pytest tests/test_cli.py -v`
Expected: all tests PASS

- [ ] **Step 11: Run the full suite + mypy**

```bash
.venv/bin/python -m pytest
.venv/bin/python -m mypy src
```
Expected: all tests pass, coverage ≥85%, mypy clean.

- [ ] **Step 12: Manual smoke test — real components, real hardware**

```bash
.venv/bin/vocalis &
```

1. Copy some text (Ctrl+C).
2. Press Ctrl+Alt+R. Confirm: "Reading..." notification → the text
   itself → speech plays through your speakers → "Done".
3. Copy a longer paragraph, press Ctrl+Alt+R, then press it again
   mid-sentence. Confirm playback stops immediately and you see
   "Stopped", not "Done".
4. Copy nothing (clear the clipboard first, e.g. `xclip -i -selection
   clipboard </dev/null`), press Ctrl+Alt+R, confirm "Nothing copied".
5. Compare the primary voice's sound against the Piper fallback
   subjectively — temporarily rename the Qwen3-TTS model directory to
   force a fallback, press Ctrl+Alt+R, confirm you hear Piper's voice and
   see the "using backup voice" notification, then rename the directory
   back.
6. Watch `nvidia-smi -l 1` during a reading: VRAM should rise while
   Qwen3-TTS is active and fall back to the Parakeet/idle floor within
   `model_idle_timeout_seconds` after it finishes.

- [ ] **Step 13: Commit**

```bash
git add src/vocalis/config.py config/default_config.toml src/vocalis/cli.py tests/test_config.py tests/test_cli.py
git commit -m "feat: wire QuickReaderFeature into the vocalis CLI and daemon" -m "- Adds qwen3_tts_model_dir/speaker and piper_binary_path/voice_model_path config fields" -m "- build_quick_reader() constructs the feature and listener from config; main() starts/stops it alongside dictation and the daemon" -m "- Piper fallback is loaded eagerly at startup (fail-fast if files are missing) since it is never GPU-managed" -m "- Verified end-to-end on real hardware: reading, interrupting mid-reading, empty clipboard, and the Piper fallback path" -m "Phase 3 of vocalis (quick reader)" -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_0116ii4z7tzRCRhVAxecpkYC"
```

---

### Task 8: Docs + Changelog

**Files:**
- Modify: `docs/architecture.md`
- Modify: `docs/folder-structure.md`
- Modify: `docs/file-purpose.md`
- Modify: `docs/workflow.md`
- Modify: `docs/user-docs.md`
- Create: `.claude/2026-09-25_0300_CHANGELOG.md`

**Interfaces:**
- Consumes: nothing (documentation only).
- Produces: updated governance docs describing exactly what Phase 3
  adds, matching the actual final code.

- [ ] **Step 1: Append to `docs/architecture.md`**

```markdown

## Phase 3 addition: Quick Reader (TTS)

Pressing Ctrl+Alt+R now works end-to-end:

1. `CtrlAltRListener` (evdev, one background thread per keyboard-capable
   device, built the same way as `NumlockListener`) detects Ctrl+Alt+R
   without suppressing normal key behavior, and excludes Vocalis's own
   ydotoold virtual device from watching.
2. On trigger, `QuickReaderFeature.on_trigger()` either starts a new
   reading (spawning a background thread so the listener itself is never
   blocked) or, if already reading, stops it immediately.
3. The background thread reads the clipboard (`ClipboardReader`), then
   calls `model_manager.acquire("quick_reader_tts", ...)` to get a loaded
   `Qwen3TTS` engine — obeying the same single-residency and idle-unload
   policy every GPU engine uses. If that fails for any reason, it falls
   back to an always-loaded `PiperTTS` instance, which runs entirely on
   CPU and is never routed through `ModelManager`.
4. The generated speech is played via `AudioPlayer` (sounddevice-based,
   the same library `MicRecorder` uses for capture), which can be
   interrupted mid-playback.
5. Notifications fire at each stage: reading, the text itself, done (or
   stopped early, or the specific failure reason).

See `docs/superpowers/specs/2026-09-11-vocalis-design.md`'s Phase 3
amendment for the full rationale, including two corrections made during
brainstorming: the legacy tool actually used Piper (not Qwen3-TTS/
fish-speech as originally assumed), and Qwen3-TTS's "Base" variant turned
out to be a voice-cloning model — "CustomVoice" with a preset voice is
what this feature actually uses.
```

- [ ] **Step 2: Update `docs/folder-structure.md`**

Add these new entries in their alphabetical position within the existing
structure:

```
├── engines/
│   └── qwen3_tts.py        # Qwen3TTS: primary voice, GPU, via ModelManager
│   └── piper_tts.py         # PiperTTS: CPU-only fallback voice, never GPU-managed
├── features/
│   └── quick_reader.py      # QuickReaderFeature: trigger/stop orchestration
├── hotkeys/
│   └── ctrl_alt_r.py         # CtrlAltRListener: evdev Ctrl+Alt+R detection
├── audio/
│   └── playback.py           # AudioPlayer: sounddevice-based interruptible playback
├── output/
│   └── clipboard.py           # ClipboardReader: reads clipboard text via xclip
```

- [ ] **Step 3: Update `docs/file-purpose.md`**

Add these rows to the table:

```markdown
| `src/vocalis/engines/qwen3_tts.py` | `Qwen3TTS` — wraps the `qwen-tts` package's CustomVoice model (preset voice, no cloning); implements `ModelEngine`. |
| `src/vocalis/engines/piper_tts.py` | `PiperTTS` — wraps the existing local `piper` binary via subprocess; CPU-only, never routed through `ModelManager`. |
| `src/vocalis/audio/playback.py` | `AudioPlayer` — sounddevice-based, interruptible speech playback. |
| `src/vocalis/hotkeys/ctrl_alt_r.py` | `CtrlAltRListener` — evdev Ctrl+Alt+R push-to-trigger detection; `list_hotkey_capable_devices()` is the real device enumerator. |
| `src/vocalis/output/clipboard.py` | `ClipboardReader` — reads clipboard text via `xclip`, the read-side counterpart to `TextOutput`. |
| `src/vocalis/features/quick_reader.py` | `QuickReaderFeature` — wires Ctrl+Alt+R trigger/stop to clipboard-read → speak (with fallback) → play, with notifications at each stage. |
```

- [ ] **Step 4: Append to `docs/workflow.md`**

```markdown

## Quick Reader trigger/stop flow (Phase 3)

```
Ctrl+Alt+R pressed (evdev event, CtrlAltRListener)
  -> QuickReaderFeature.on_trigger()
       -> if already reading: player.stop(), notify("Stopped"), return
       -> else: mark active, spawn background thread, return immediately

Background thread:
  -> text = clipboard_reader.read()
  -> if cancelled or empty: notify accordingly, stop
  -> notify("Reading...")
  -> try: engine = model_manager.acquire("quick_reader_tts", qwen3_tts_factory)
          audio, sr = engine.synthesize(text)
          model_manager.release("quick_reader_tts")
     except: notify("using backup voice"), audio, sr = piper.synthesize(text)
  -> if cancelled while synthesizing: stop, never start playback
  -> notify(text)                              # live transcript notification
  -> player.play(audio, sr); finished = player.wait_until_done()
  -> if finished naturally: notify("Done")
```

Every failure path (empty clipboard, both engines failing, playback
failure) notifies a specific reason and returns without crashing the
listener thread or the daemon. A second press is only ever noticed
promptly because on_trigger() itself never blocks — all the real work
happens in the spawned background thread.
```

- [ ] **Step 5: Update `docs/user-docs.md`**

Add this section after the existing "Dictation (Phase 2)" section:

```markdown

## Quick Reader (Phase 3)

1. Copy some text (**Ctrl+C**).
2. Press **Ctrl+Alt+R**. You'll see "Reading...", then the text itself,
   then it plays out loud through your speakers, then "Done".
3. Press **Ctrl+Alt+R** again while it's still talking to stop it
   immediately.
4. If nothing is copied, you'll see "Nothing copied" instead.

Quick Reader uses a local AI voice (Qwen3-TTS) by default. If that voice
ever fails to load or generate speech for any reason, it automatically
switches to a simpler backup voice (Piper) and tells you so — you'll
still hear your text read aloud either way.
```

Also update the Configuration section's bullet list to add:

```markdown
- `[engines] qwen3_tts_speaker` — which of Qwen3-TTS's built-in voices to
  use (default: "Ryan").
```

- [ ] **Step 6: Write the changelog entry**

```markdown
<!-- .claude/2026-09-25_0300_CHANGELOG.md -->
# Changelog — 2026-09-25 (Phase 3)

## Vocalis Phase 3: Quick Reader (TTS)

Pressing Ctrl+Alt+R now reads the clipboard aloud, with a real local AI
voice and an always-available backup voice.

**Added:**
- `vocalis.engines.qwen3_tts.Qwen3TTS` — primary voice, wraps the
  `qwen-tts` package's CustomVoice model (preset voice "Ryan" by
  default, no reference-audio/cloning step). GPU-only, managed through
  the existing `ModelManager` exactly like Parakeet.
- `vocalis.engines.piper_tts.PiperTTS` — fallback voice, wraps the
  already-installed local `piper` binary. CPU-only, never GPU-managed;
  loaded once at startup and always available.
- `vocalis.audio.playback.AudioPlayer` — sounddevice-based, interruptible
  speech playback (not a `pw-play` subprocess).
- `vocalis.hotkeys.ctrl_alt_r.CtrlAltRListener` — evdev-based Ctrl+Alt+R
  detection, built the same way as `NumlockListener`, applying Phase 2's
  Task 9/Task 10 lessons (per-callback exception isolation, ydotoold
  device exclusion) from the start.
- `vocalis.output.clipboard.ClipboardReader` — reads clipboard text via
  `xclip`.
- `vocalis.features.quick_reader.QuickReaderFeature` — orchestrates
  trigger → read → speak (with automatic fallback) → play, with
  press-again-to-stop and notifications at every stage.
- `EnginesConfig.qwen3_tts_model_dir`/`qwen3_tts_speaker`/
  `piper_binary_path`/`piper_voice_model_path` — new config fields.

**Design corrections made along the way (recorded in the spec, not just
here):**
- The legacy `old/read-selected.sh` script actually used **Piper**, not
  Qwen3-TTS/fish-speech as this project's original design spec assumed —
  discovered by reading the actual old script before finalizing the
  design, the same way Phase 2 checked real feasibility before finalizing
  its plan.
- Qwen3-TTS's "Base" variant is a voice-*cloning* model requiring
  reference audio, not a plain default-voice TTS mode as first assumed;
  "CustomVoice" (preset voices, no cloning) is the variant actually used.

**Not yet implemented (future phases):** Document Reader, OCR, UI (tray/
widget/settings), systemd service, packaging.

**Rollback:** additive only — Phase 0/1/2 code untouched; the only
existing-file changes are the new `EnginesConfig` fields (additive,
default-valued) and `main()`'s addition of a third listener alongside the
two that already existed. Revert via `git revert` back to before this
phase's commits.
```

- [ ] **Step 7: Commit**

```bash
git add docs/architecture.md docs/folder-structure.md docs/file-purpose.md docs/workflow.md docs/user-docs.md .claude/2026-09-25_0300_CHANGELOG.md
git commit -m "docs: document phase 3 quick reader feature" -m "- Architecture, folder structure, file purpose, workflow, and user docs updated to describe the built quick reader pipeline" -m "- Adds changelog entry summarizing phase 3, including the two design corrections found along the way" -m "Phase 3 of vocalis (quick reader)" -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_0116ii4z7tzRCRhVAxecpkYC"
```

---

## After This Plan — Manual Verification (you, not an agent)

An AI agent cannot physically press a key combo and listen to speakers,
so the final proof this phase works is a manual check — most of it is
already folded into Task 7 Step 12 above, but as a final pass once all 8
tasks are done:

1. Run `.venv/bin/vocalis` in the background.
2. Copy a sentence, press Ctrl+Alt+R, confirm it's read aloud correctly
   start to finish.
3. Copy a longer paragraph, press Ctrl+Alt+R, press it again mid-read to
   confirm the stop works, then a third time to confirm a fresh reading
   starts cleanly.
4. Confirm NUMLOCK dictation (Phase 2) still works exactly as before —
   this phase must not have regressed it.
5. Watch `nvidia-smi -l 1` across a few reading cycles: VRAM should rise
   only while Qwen3-TTS is actively generating, and fall back to the
   established idle floor afterward.

Once you're satisfied, the next implementation plan (per the spec's
milestone list) is **Phase 4 — Text pipeline + Document Reader**.
