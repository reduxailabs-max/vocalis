# Vocalis Phase 2: Dictation (STT) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make holding NUMLOCK work again — this time inside Vocalis. Speak
while NUMLOCK is held, release it, and the transcribed text is typed at the
cursor (or copied to the clipboard if typing fails), with desktop
notifications at each stage. Reuses the exact Parakeet model files the
existing Voxtype binary already uses.

**Architecture:** A `NumlockListener` (evdev, background thread) calls
`DictationFeature.on_press()`/`on_release()`. `on_press` starts a
`MicRecorder`; `on_release` stops it, acquires a `ParakeetSTT` engine
through the Phase 1 `ModelManager` (so it obeys the single-residency +
idle-unload GPU policy), transcribes, releases the engine, and delivers the
text through `TextOutput` (ydotool, clipboard fallback). A `Notifier` reports
start/transcribing/done at each stage. `cli.py`'s `main()` wires a single
shared `ModelManager` into both the daemon and the dictation feature and
starts/stops the listener alongside the daemon.

**Tech Stack:** `onnx-asr[gpu]` (wraps ONNX Runtime + the existing Parakeet
TDT ONNX files), `nvidia-cudnn-cu12` (cuDNN, required for the CUDA
execution provider — not bundled with `onnxruntime-gpu`), `evdev`
(NUMLOCK push-to-talk), `sounddevice` (mic capture via PortAudio, already
installed as a system library), `ydotool`/`xclip` (already installed,
subprocess-invoked), `notify-send` (already installed).

**Spec:** `docs/superpowers/specs/2026-09-11-vocalis-design.md` (see the
Phase 2 milestone note and the Hard Constraint #2 amendment, both added
2026-09-11 after direct feasibility testing on this machine)

## Global Constraints

- Fully offline/local — the `onnx-asr[hub]` extra (HuggingFace download) is
  NOT used; the model is always loaded from the existing local directory.
- Only one AI model may be resident in VRAM at a time; `ParakeetSTT` must be
  acquired/released through the Phase 1 `ModelManager`, never held directly.
- Every function gets a docstring; structured error handling with specific
  exception types; no bare `except:`.
- Structured logging via the `"vocalis"` logger; never log the transcribed
  text content at INFO level in a way that could leak into shared logs at
  default verbosity beyond what notifications already show the user
  on-screen (notifications are fine — they're local and user-facing).
- ≥85% test coverage (enforced by `pytest-cov --cov-fail-under=85`).
- `old/` is never modified.
- Conventional-commit messages (`feat:`, `fix:`, `test:`, `docs:`, `chore:`),
  ending with:
  ```
  Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01GMitM6j81rZ9TN8k3Q5tAh
  ```
  copied verbatim — do not substitute a different model's name regardless
  of which model executes a task.
- Desktop notifications for dictation must cover **start**, **live
  transcript**, and **done** by default (per spec), and must never crash
  the feature if `notify-send` itself fails.
- Accept the small (~150MB) fixed GPU residual described in the spec's
  Hard Constraint #2 amendment — do not attempt to "fix" it; it is not a
  defect in this plan's code.
- One sub-task at a time — stop after each task and wait for explicit user
  "yes"/"y" before continuing (per project task-list protocol).

---

### Task 1: Parakeet STT Engine Wrapper ✅

**Files:**
- Create: `src/vocalis/engines/__init__.py`
- Create: `src/vocalis/engines/parakeet.py`
- Create: `tests/test_parakeet.py`
- Modify: `pyproject.toml` (add `onnx-asr[gpu]`, `nvidia-cudnn-cu12`, `numpy` dependencies)

**Interfaces:**
- Consumes: nothing new (implements the `ModelEngine` protocol from
  `src/vocalis/model_manager.py`: `load() -> None`, `unload() -> None`,
  `is_loaded: bool` property — so it can be managed by `ModelManager`).
- Produces:
  - `ParakeetLoadError(Exception)`
  - `ParakeetSTT.__init__(self, model_dir: Path, quantization: str = "int8", use_gpu: bool = True) -> None`
  - `ParakeetSTT.load(self) -> None`
  - `ParakeetSTT.unload(self) -> None`
  - `ParakeetSTT.is_loaded -> bool` (property)
  - `ParakeetSTT.transcribe(self, audio: npt.NDArray[np.float32], sample_rate: int) -> str`
  - Task 6 (DictationFeature) calls `transcribe()`; Task 7 (CLI wiring)
    constructs `ParakeetSTT` instances via a factory pointed at
    `config.engines.parakeet_model_dir`.

- [x] **Step 1: Add dependencies to `pyproject.toml`**

Change the `dependencies` list from:

```toml
dependencies = [
    "pydantic>=2.6,<3",
    "tomli-w>=1.0,<2",
]
```

to:

```toml
dependencies = [
    "pydantic>=2.6,<3",
    "tomli-w>=1.0,<2",
    "numpy>=1.26,<3",
    "onnx-asr[gpu]>=0.12,<1",
    "nvidia-cudnn-cu12>=9.0,<10",
]
```

Then run: `.venv/bin/pip install -e ".[dev]"` to install the new dependencies
(this downloads ~1-2GB for `nvidia-cudnn-cu12` and `onnxruntime-gpu` — run it
in the background and wait for it to finish before continuing).

- [x] **Step 2: Create the package `__init__.py`**

```python
# src/vocalis/engines/__init__.py
"""AI model engine wrappers implementing vocalis.model_manager.ModelEngine."""
```

- [x] **Step 3: Write the failing tests**

```python
# tests/test_parakeet.py
"""Tests for vocalis.engines.parakeet.ParakeetSTT.

The real Parakeet model files already exist on this development machine
(inherited from the existing Voxtype installation). Tests that need the
real model are skipped automatically on a machine that doesn't have them,
so this suite stays portable while still proving real correctness here.
"""
from pathlib import Path

import numpy as np
import pytest

from vocalis.engines.parakeet import ParakeetLoadError, ParakeetSTT

REAL_MODEL_DIR = Path("/home/redx/.local/share/voxtype/models/parakeet-tdt-0.6b-v3-int8")
_real_model_available = REAL_MODEL_DIR.exists()


def test_is_loaded_false_before_load() -> None:
    engine = ParakeetSTT(model_dir=REAL_MODEL_DIR)
    assert engine.is_loaded is False


def test_transcribe_before_load_raises() -> None:
    engine = ParakeetSTT(model_dir=REAL_MODEL_DIR)
    with pytest.raises(RuntimeError):
        engine.transcribe(np.zeros(16000, dtype=np.float32), sample_rate=16000)


def test_load_missing_model_dir_raises_parakeet_load_error(tmp_path: Path) -> None:
    engine = ParakeetSTT(model_dir=tmp_path / "does_not_exist", use_gpu=False)
    with pytest.raises(ParakeetLoadError):
        engine.load()


def test_unload_before_load_is_safe() -> None:
    engine = ParakeetSTT(model_dir=REAL_MODEL_DIR)
    engine.unload()  # must not raise
    assert engine.is_loaded is False


@pytest.mark.skipif(
    not _real_model_available, reason="Real Parakeet model files not present on this machine"
)
def test_load_and_transcribe_real_model_on_cpu() -> None:
    engine = ParakeetSTT(model_dir=REAL_MODEL_DIR, use_gpu=False)
    engine.load()
    try:
        assert engine.is_loaded is True
        audio = (np.random.randn(16000) * 0.001).astype(np.float32)
        result = engine.transcribe(audio, sample_rate=16000)
        assert isinstance(result, str)
    finally:
        engine.unload()
    assert engine.is_loaded is False


@pytest.mark.skipif(
    not _real_model_available, reason="Real Parakeet model files not present on this machine"
)
def test_load_and_transcribe_real_model_on_gpu() -> None:
    engine = ParakeetSTT(model_dir=REAL_MODEL_DIR, use_gpu=True)
    engine.load()
    try:
        assert engine.is_loaded is True
        audio = (np.random.randn(16000) * 0.001).astype(np.float32)
        result = engine.transcribe(audio, sample_rate=16000)
        assert isinstance(result, str)
    finally:
        engine.unload()
```

- [x] **Step 4: Run the tests, verify they fail**

Run: `.venv/bin/pytest tests/test_parakeet.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vocalis.engines.parakeet'`

- [x] **Step 5: Write the minimal implementation**

```python
# src/vocalis/engines/parakeet.py
"""Parakeet speech-to-text engine, implementing the ModelEngine protocol so
it can be managed by ModelManager's single-residency/idle-unload policy.

Uses the `onnx-asr` package to run NVIDIA's Parakeet TDT model (the same
ONNX files the existing Voxtype installation already uses) via ONNX
Runtime, on GPU (CUDA) by default with a CPU fallback available for
testing/debugging.

cuDNN is required for the CUDA execution provider but is not bundled with
onnxruntime-gpu; it's installed as the `nvidia-cudnn-cu12` pip package and
must be preloaded via ctypes before the CUDA session is created, since its
.so files live inside site-packages rather than the system library path.
"""
import ctypes
import glob
import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnx_asr

logger = logging.getLogger("vocalis")

_MODEL_TYPE = "nemo-parakeet-tdt-0.6b-v3"


class ParakeetLoadError(Exception):
    """Raised when the Parakeet model fails to load."""


class ParakeetSTT:
    """Speech-to-text engine backed by the local Parakeet TDT ONNX model."""

    def __init__(
        self,
        model_dir: Path,
        quantization: str = "int8",
        use_gpu: bool = True,
    ) -> None:
        """Create a ParakeetSTT engine (does not load the model yet).

        Args:
            model_dir: Directory containing the encoder/decoder ONNX files,
                vocab.txt, and config.json (the existing Voxtype Parakeet
                model directory).
            quantization: Quantization suffix matching the model files,
                e.g. "int8".
            use_gpu: If True, use CUDAExecutionProvider; if False, force
                CPUExecutionProvider (useful for testing without a GPU).
        """
        self._model_dir = model_dir
        self._quantization = quantization
        self._use_gpu = use_gpu
        self._model: onnx_asr.adapters.TextResultsAsrAdapter | None = None

    @property
    def is_loaded(self) -> bool:
        """Whether the model is currently loaded."""
        return self._model is not None

    def load(self) -> None:
        """Load the ONNX model into memory (and GPU VRAM if use_gpu=True).

        Raises:
            ParakeetLoadError: If the model files are missing or loading
                otherwise fails.
        """
        if self._use_gpu:
            _preload_cudnn()
        providers = ["CUDAExecutionProvider"] if self._use_gpu else ["CPUExecutionProvider"]
        try:
            self._model = onnx_asr.load_model(
                _MODEL_TYPE,
                str(self._model_dir),
                quantization=self._quantization,
                providers=providers,
            )
        except Exception as exc:
            raise ParakeetLoadError(
                f"Failed to load Parakeet model from {self._model_dir}: {exc}"
            ) from exc

    def unload(self) -> None:
        """Release the model reference. Safe to call even if never loaded."""
        self._model = None

    def transcribe(self, audio: npt.NDArray[np.float32], sample_rate: int) -> str:
        """Transcribe a float32 audio waveform to text.

        Args:
            audio: 1-D float32 numpy array of audio samples.
            sample_rate: Sample rate of `audio` in Hz.

        Returns:
            The transcribed text (may be an empty string for silence).

        Raises:
            RuntimeError: If called before load().
        """
        if self._model is None:
            raise RuntimeError("ParakeetSTT.transcribe() called before load()")
        result = self._model.recognize(audio, sample_rate=sample_rate)
        return str(result)


def _preload_cudnn() -> None:
    """Preload cuDNN shared libraries so onnxruntime's CUDA provider can
    find them, since they live inside the nvidia-cudnn-cu12 pip package's
    site-packages directory rather than the system library path.

    Safe to call multiple times (loading an already-loaded .so is a no-op).
    """
    try:
        import nvidia.cudnn  # type: ignore[import-not-found]
    except ImportError:
        logger.warning(
            "nvidia-cudnn-cu12 not installed; CUDAExecutionProvider will likely fail"
        )
        return

    lib_dir = Path(nvidia.cudnn.__path__[0]) / "lib"
    for lib_path in sorted(glob.glob(str(lib_dir / "libcudnn*.so.9"))):
        ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
```

- [x] **Step 6: Run the tests, verify they pass**

Run: `.venv/bin/pytest tests/test_parakeet.py -v`
Expected: PASS (all 6 tests — the two GPU/CPU real-model tests will
actually execute on this machine since the model files are present; they
take a few seconds each due to model loading).

- [x] **Step 7: Run the full suite + mypy**

Run: `.venv/bin/pytest && .venv/bin/mypy`
Expected: All tests PASS, coverage ≥85%, mypy clean.

- [x] **Step 8: Commit**

```bash
git add pyproject.toml src/vocalis/engines/__init__.py src/vocalis/engines/parakeet.py tests/test_parakeet.py
git commit -m "feat: add ParakeetSTT engine wrapper using onnx-asr" -m "- Wraps the existing Voxtype Parakeet TDT ONNX model via onnx-asr, no re-download needed" -m "- Implements the ModelEngine protocol so it can be managed by ModelManager" -m "- Preloads cuDNN from the nvidia-cudnn-cu12 pip package since onnxruntime-gpu doesn't bundle it" -m "- Tested against the real local model on both CPU and GPU providers" -m "Phase 2 of vocalis (dictation)" -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01GMitM6j81rZ9TN8k3Q5tAh"
```

---

### Task 2: Microphone Capture ✅

**Files:**
- Create: `src/vocalis/audio/__init__.py`
- Create: `src/vocalis/audio/capture.py`
- Create: `tests/test_capture.py`
- Modify: `pyproject.toml` (add `sounddevice` dependency)

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `RecordingAlreadyActiveError(Exception)`
  - `MicRecorder.__init__(self, sample_rate: int = 16000, max_duration_seconds: float = 60.0, stream_factory: Callable[..., Any] | None = None) -> None`
  - `MicRecorder.start(self) -> None`
  - `MicRecorder.stop(self) -> npt.NDArray[np.float32]`
  - `MicRecorder.is_recording -> bool` (property)
  - `DEFAULT_SAMPLE_RATE = 16000` (module constant)
  - Task 6 (DictationFeature) calls `start()`/`stop()`; Task 7 (CLI wiring)
    constructs a real `MicRecorder`.

- [x] **Step 1: Add dependency to `pyproject.toml`**

Add `"sounddevice>=0.4.6,<1"` to the `dependencies` list (alongside the
Task 1 additions). Run `.venv/bin/pip install -e ".[dev]"`.

- [x] **Step 2: Create the package `__init__.py`**

```python
# src/vocalis/audio/__init__.py
"""Audio capture and playback for vocalis."""
```

- [x] **Step 3: Write the failing tests**

```python
# tests/test_capture.py
"""Tests for vocalis.audio.capture.MicRecorder.

Uses a FakeStream in place of sounddevice.InputStream so tests never touch
real microphone hardware.
"""
import time

import numpy as np
import pytest

from vocalis.audio.capture import MicRecorder, RecordingAlreadyActiveError


class FakeStream:
    def __init__(self, samplerate, channels, dtype, callback) -> None:
        self.samplerate = samplerate
        self.callback = callback
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True


def make_recorder(max_duration_seconds: float = 60.0):
    streams: list[FakeStream] = []

    def factory(**kwargs):
        stream = FakeStream(**kwargs)
        streams.append(stream)
        return stream

    recorder = MicRecorder(max_duration_seconds=max_duration_seconds, stream_factory=factory)
    return recorder, streams


def test_start_creates_and_starts_a_stream() -> None:
    recorder, streams = make_recorder()
    recorder.start()
    assert len(streams) == 1
    assert streams[0].started is True
    assert recorder.is_recording is True


def test_stop_without_any_audio_returns_empty_array() -> None:
    recorder, _ = make_recorder()
    recorder.start()
    audio = recorder.stop()
    assert audio.shape == (0,)
    assert audio.dtype == np.float32
    assert recorder.is_recording is False


def test_stop_returns_captured_audio() -> None:
    recorder, streams = make_recorder()
    recorder.start()
    chunk1 = np.array([[0.1], [0.2]], dtype=np.float32)
    chunk2 = np.array([[0.3]], dtype=np.float32)
    streams[0].callback(chunk1, 2, None, None)
    streams[0].callback(chunk2, 1, None, None)
    audio = recorder.stop()
    np.testing.assert_array_almost_equal(audio, np.array([0.1, 0.2, 0.3], dtype=np.float32))


def test_double_start_raises() -> None:
    recorder, _ = make_recorder()
    recorder.start()
    with pytest.raises(RecordingAlreadyActiveError):
        recorder.start()
    recorder.stop()


def test_stream_stopped_and_closed_on_stop() -> None:
    recorder, streams = make_recorder()
    recorder.start()
    recorder.stop()
    assert streams[0].stopped is True
    assert streams[0].closed is True


def test_safety_timeout_stops_stream_but_stop_still_returns_captured_audio() -> None:
    recorder, streams = make_recorder(max_duration_seconds=0.05)
    recorder.start()
    streams[0].callback(np.array([[0.5]], dtype=np.float32), 1, None, None)
    time.sleep(0.2)
    assert recorder.is_recording is False  # safety cutoff already stopped the stream
    audio = recorder.stop()
    np.testing.assert_array_almost_equal(audio, np.array([0.5], dtype=np.float32))
```

- [x] **Step 4: Run the tests, verify they fail**

Run: `.venv/bin/pytest tests/test_capture.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vocalis.audio.capture'`

- [x] **Step 5: Write the minimal implementation**

```python
# src/vocalis/audio/capture.py
"""Microphone audio capture for dictation.

Records float32 mono audio at a fixed sample rate while the caller holds
the stream open (e.g. NUMLOCK held down), with a hard safety cutoff so a
stuck hotkey can never record forever. The safety cutoff only stops the
underlying stream — it does not discard already-captured audio, so a
later stop() call still returns everything captured up to the cutoff.
"""
import logging
import threading
from typing import Any, Callable

import numpy as np
import numpy.typing as npt
import sounddevice as sd

logger = logging.getLogger("vocalis")

DEFAULT_SAMPLE_RATE = 16000


class RecordingAlreadyActiveError(Exception):
    """Raised when start() is called while a recording is already active."""


class MicRecorder:
    """Captures microphone audio into an in-memory float32 buffer."""

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        max_duration_seconds: float = 60.0,
        stream_factory: Callable[..., Any] | None = None,
    ) -> None:
        """Create a MicRecorder.

        Args:
            sample_rate: Capture sample rate in Hz.
            max_duration_seconds: Hard safety cutoff — recording auto-stops
                the audio stream after this many seconds even if stop() is
                never called.
            stream_factory: Callable(**kwargs) -> stream object with
                start()/stop()/close() methods, used to construct the audio
                stream. Defaults to sounddevice.InputStream; injectable for
                tests.
        """
        self._sample_rate = sample_rate
        self._max_duration_seconds = max_duration_seconds
        self._stream_factory = stream_factory or sd.InputStream
        self._chunks: list[npt.NDArray[np.float32]] = []
        self._stream: Any | None = None
        self._lock = threading.Lock()
        self._safety_timer: threading.Timer | None = None

    @property
    def is_recording(self) -> bool:
        """Whether the underlying audio stream is currently open."""
        return self._stream is not None

    def start(self) -> None:
        """Begin recording from the default microphone.

        Raises:
            RecordingAlreadyActiveError: If a recording is already active.
        """
        if self.is_recording:
            raise RecordingAlreadyActiveError("MicRecorder.start() called while already recording")

        self._chunks = []
        stream = self._stream_factory(
            samplerate=self._sample_rate,
            channels=1,
            dtype="float32",
            callback=self._on_audio,
        )
        stream.start()
        self._stream = stream

        self._safety_timer = threading.Timer(self._max_duration_seconds, self._stop_stream_only)
        self._safety_timer.daemon = True
        self._safety_timer.start()
        logger.info("Recording started (sample_rate=%d)", self._sample_rate)

    def stop(self) -> npt.NDArray[np.float32]:
        """Stop recording and return everything captured.

        Returns:
            A 1-D float32 numpy array of all captured samples (empty array
            if nothing was captured).
        """
        if self._safety_timer is not None:
            self._safety_timer.cancel()
            self._safety_timer = None

        self._stop_stream_only()

        with self._lock:
            chunks = self._chunks
            self._chunks = []

        if not chunks:
            logger.info("Recording stopped: no audio captured")
            return np.zeros(0, dtype=np.float32)

        audio = np.concatenate(chunks).astype(np.float32).reshape(-1)
        logger.info("Recording stopped: %.2fs captured", len(audio) / self._sample_rate)
        return audio

    def _stop_stream_only(self) -> None:
        """Stop and close the underlying stream without clearing captured
        chunks, so a later stop() call can still return them. Safe to call
        multiple times (e.g. once from the safety timer, once from stop()).
        """
        with self._lock:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
                self._stream = None
                logger.warning(
                    "Recording stream stopped (safety cutoff or explicit stop)"
                )

    def _on_audio(
        self, indata: npt.NDArray[np.float32], frames: int, time_info: object, status: object
    ) -> None:
        """sounddevice callback: append each incoming audio block."""
        if status:
            logger.warning("Audio input status: %s", status)
        with self._lock:
            self._chunks.append(indata.copy().reshape(-1))
```

Note: `_stop_stream_only`'s warning log fires on every call including the
normal explicit `stop()` path, which is a little noisy — that's intentional
for this task (simplicity over polish); if it proves too noisy in practice
it can be quieted in a later phase.

- [x] **Step 6: Run the tests, verify they pass**

Run: `.venv/bin/pytest tests/test_capture.py -v`
Expected: PASS (all 6 tests)

- [x] **Step 7: Run the full suite + mypy**

Run: `.venv/bin/pytest && .venv/bin/mypy`
Expected: All tests PASS, coverage ≥85%, mypy clean.

- [x] **Step 8: Manual smoke test with a real microphone**

Run this from `.venv/bin/python` interactively or as a throwaway script,
in the background if it's turned into a long-running script (it isn't
here — it's a short synchronous check):

```python
from vocalis.audio.capture import MicRecorder
import time

recorder = MicRecorder()
recorder.start()
time.sleep(2)
audio = recorder.stop()
print(f"Captured {len(audio)} samples ({len(audio)/16000:.2f}s), "
      f"max amplitude: {abs(audio).max() if len(audio) else 0}")
```

Expected: prints a sample count around `32000` (2 seconds × 16000Hz) and a
non-zero max amplitude if you made any sound.

- [x] **Step 9: Commit**

```bash
git add pyproject.toml src/vocalis/audio/__init__.py src/vocalis/audio/capture.py tests/test_capture.py
git commit -m "feat: add MicRecorder for microphone capture" -m "- Captures float32 mono audio via sounddevice with an injectable stream factory for testing" -m "- Safety cutoff (max_duration_seconds) stops the stream without discarding already-captured audio" -m "- Verified against a real microphone" -m "Phase 2 of vocalis (dictation)" -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01GMitM6j81rZ9TN8k3Q5tAh"
```

---

### Task 3: NUMLOCK Hotkey Listener ✅

**Files:**
- Create: `src/vocalis/hotkeys/__init__.py`
- Create: `src/vocalis/hotkeys/numlock.py`
- Create: `tests/test_numlock.py`
- Modify: `pyproject.toml` (add `evdev` dependency)

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `NumlockListener.__init__(self, on_press: Callable[[], None], on_release: Callable[[], None], device_enumerator: Callable[[], list[EvdevDevice]]) -> None`
  - `NumlockListener.start(self) -> None`
  - `NumlockListener.stop(self) -> None`
  - `NumlockListener.is_running -> bool` (property)
  - `list_numlock_capable_devices() -> list[evdev.InputDevice]` — the real
    device enumerator, used by Task 7's CLI wiring.
  - Task 7 (CLI wiring) constructs `NumlockListener(on_press=feature.on_press,
    on_release=feature.on_release, device_enumerator=list_numlock_capable_devices)`.

- [x] **Step 1: Add dependency to `pyproject.toml`**

Add `"evdev>=1.7,<2"` to the `dependencies` list. Run
`.venv/bin/pip install -e ".[dev]"`.

- [x] **Step 2: Create the package `__init__.py`**

```python
# src/vocalis/hotkeys/__init__.py
"""Global hotkey listeners for vocalis."""
```

- [x] **Step 3: Write the failing tests**

```python
# tests/test_numlock.py
"""Tests for vocalis.hotkeys.numlock.NumlockListener.

Uses a FakeDevice implementing the same read_loop()/close() pattern real
evdev.InputDevice objects use, so tests never touch real input hardware.
"""
import threading
import time

from vocalis.hotkeys.numlock import NumlockListener

_EV_KEY = 1
_KEY_NUMLOCK = 69
_KEY_A = 30


class FakeEvent:
    def __init__(self, type_: int, code: int, value: int) -> None:
        self.type = type_
        self.code = code
        self.value = value


class FakeDevice:
    """Mimics evdev.InputDevice's read_loop()/close() contract:
    read_loop() blocks yielding queued events until close() is called,
    at which point it stops.
    """

    def __init__(self) -> None:
        self._pending: list[FakeEvent] = []
        self._closed = threading.Event()

    def queue_event(self, event: FakeEvent) -> None:
        self._pending.append(event)

    def read_loop(self):
        while not self._closed.is_set():
            if self._pending:
                yield self._pending.pop(0)
            else:
                time.sleep(0.01)

    def close(self) -> None:
        self._closed.set()


def wait_for(flag: threading.Event, timeout: float = 1.0) -> None:
    assert flag.wait(timeout), "Expected callback was not invoked in time"


def test_no_devices_found_does_not_crash() -> None:
    press_called = threading.Event()
    listener = NumlockListener(
        on_press=press_called.set, on_release=lambda: None, device_enumerator=lambda: []
    )
    listener.start()
    listener.stop()
    assert not press_called.is_set()
    assert listener.is_running is False


def test_key_down_triggers_on_press() -> None:
    press_called = threading.Event()
    device = FakeDevice()
    device.queue_event(FakeEvent(_EV_KEY, _KEY_NUMLOCK, 1))
    listener = NumlockListener(
        on_press=press_called.set, on_release=lambda: None, device_enumerator=lambda: [device]
    )
    listener.start()
    try:
        wait_for(press_called)
    finally:
        listener.stop()


def test_key_up_triggers_on_release() -> None:
    release_called = threading.Event()
    device = FakeDevice()
    device.queue_event(FakeEvent(_EV_KEY, _KEY_NUMLOCK, 0))
    listener = NumlockListener(
        on_press=lambda: None, on_release=release_called.set, device_enumerator=lambda: [device]
    )
    listener.start()
    try:
        wait_for(release_called)
    finally:
        listener.stop()


def test_unrelated_key_is_ignored() -> None:
    press_called = threading.Event()
    device = FakeDevice()
    device.queue_event(FakeEvent(_EV_KEY, _KEY_A, 1))
    listener = NumlockListener(
        on_press=press_called.set, on_release=lambda: None, device_enumerator=lambda: [device]
    )
    listener.start()
    try:
        time.sleep(0.2)
        assert not press_called.is_set()
    finally:
        listener.stop()


def test_auto_repeat_value_two_is_ignored() -> None:
    press_called = threading.Event()
    device = FakeDevice()
    device.queue_event(FakeEvent(_EV_KEY, _KEY_NUMLOCK, 2))
    listener = NumlockListener(
        on_press=press_called.set, on_release=lambda: None, device_enumerator=lambda: [device]
    )
    listener.start()
    try:
        time.sleep(0.2)
        assert not press_called.is_set()
    finally:
        listener.stop()


def test_stop_terminates_background_threads() -> None:
    device = FakeDevice()
    listener = NumlockListener(
        on_press=lambda: None, on_release=lambda: None, device_enumerator=lambda: [device]
    )
    listener.start()
    assert listener.is_running is True
    listener.stop()
    assert listener.is_running is False
```

- [x] **Step 4: Run the tests, verify they fail**

Run: `.venv/bin/pytest tests/test_numlock.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vocalis.hotkeys.numlock'`

- [x] **Step 5: Write the minimal implementation**

```python
# src/vocalis/hotkeys/numlock.py
"""NUMLOCK push-to-talk hotkey listener using evdev.

Watches all keyboard-capable input devices for NUMLOCK key-down/key-up
events and invokes callbacks accordingly. Each device gets its own
background thread using evdev's read_loop()/close() idiom: read_loop()
blocks until the device is closed, at which point it raises OSError in the
reading thread — the standard, documented way to stop an evdev read loop
from another thread. Does not grab() or suppress the key, so the OS's own
NumLock LED/state toggling is unaffected (matching the previous Voxtype
behavior).
"""
import logging
import threading
from typing import Callable, Iterable, Protocol

logger = logging.getLogger("vocalis")

_EV_KEY = 1
_KEY_NUMLOCK = 69


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


DeviceEnumerator = Callable[[], list[EvdevDevice]]


class NumlockListener:
    """Calls on_press()/on_release() when NUMLOCK is held down/released."""

    def __init__(
        self,
        on_press: Callable[[], None],
        on_release: Callable[[], None],
        device_enumerator: DeviceEnumerator,
    ) -> None:
        """Create a NumlockListener.

        Args:
            on_press: Called when NUMLOCK transitions to pressed.
            on_release: Called when NUMLOCK transitions to released.
            device_enumerator: Returns the list of evdev devices to watch.
                Injectable so tests can supply fake devices instead of
                real hardware.
        """
        self._on_press = on_press
        self._on_release = on_release
        self._device_enumerator = device_enumerator
        self._devices: list[EvdevDevice] = []
        self._threads: list[threading.Thread] = []

    @property
    def is_running(self) -> bool:
        """Whether any watcher thread is currently active."""
        return any(thread.is_alive() for thread in self._threads)

    def start(self) -> None:
        """Enumerate NUMLOCK-capable devices and start watching each."""
        self._devices = self._device_enumerator()
        if not self._devices:
            logger.warning("No NUMLOCK-capable input devices found")
            return

        self._threads = [
            threading.Thread(target=self._watch_device, args=(device,), daemon=True)
            for device in self._devices
        ]
        for thread in self._threads:
            thread.start()
        logger.info("Watching %d device(s) for NUMLOCK", len(self._devices))

    def stop(self) -> None:
        """Close all watched devices and wait for their threads to exit."""
        for device in self._devices:
            try:
                device.close()
            except Exception:
                logger.exception("Error closing input device")
        for thread in self._threads:
            thread.join(timeout=1.0)
        self._devices = []
        self._threads = []

    def _watch_device(self, device: EvdevDevice) -> None:
        """Read events from one device until it is closed."""
        try:
            for event in device.read_loop():
                if event.type != _EV_KEY or event.code != _KEY_NUMLOCK:
                    continue
                if event.value == 1:
                    self._on_press()
                elif event.value == 0:
                    self._on_release()
                # value == 2 is auto-repeat; ignored
        except OSError:
            pass  # device closed by stop(); expected exit path
        except Exception:
            logger.exception("Error reading input device events")


def list_numlock_capable_devices() -> "list[EvdevDevice]":
    """Enumerate real input devices that report NUMLOCK key capability.

    Closes any device that doesn't have NUMLOCK capability immediately,
    since it won't be watched.
    """
    import evdev

    devices: list[EvdevDevice] = []
    for path in evdev.list_devices():
        device = evdev.InputDevice(path)
        capabilities = device.capabilities().get(evdev.ecodes.EV_KEY, [])
        if evdev.ecodes.KEY_NUMLOCK in capabilities:
            devices.append(device)
        else:
            device.close()
    return devices
```

- [x] **Step 6: Run the tests, verify they pass**

Run: `.venv/bin/pytest tests/test_numlock.py -v`
Expected: PASS (all 6 tests)

- [x] **Step 7: Run the full suite + mypy**

Run: `.venv/bin/pytest && .venv/bin/mypy`
Expected: All tests PASS, coverage ≥85%, mypy clean.

- [x] **Step 8: Manual smoke test with real hardware**

Run this as a short interactive script (it will run until you press
Ctrl+C — this is a quick manual check, not something to background):

```python
from vocalis.hotkeys.numlock import NumlockListener, list_numlock_capable_devices

listener = NumlockListener(
    on_press=lambda: print("NUMLOCK pressed"),
    on_release=lambda: print("NUMLOCK released"),
    device_enumerator=list_numlock_capable_devices,
)
listener.start()
print("Hold and release NUMLOCK a few times, then Ctrl+C to stop.")
try:
    while True:
        import time
        time.sleep(1)
except KeyboardInterrupt:
    listener.stop()
```

Expected: pressing and releasing NUMLOCK prints the corresponding message
each time; the actual keyboard NumLock LED still toggles normally (this
listener does not suppress it).

- [x] **Step 9: Commit**

```bash
git add pyproject.toml src/vocalis/hotkeys/__init__.py src/vocalis/hotkeys/numlock.py tests/test_numlock.py
git commit -m "feat: add NumlockListener for push-to-talk hotkey detection" -m "- Watches evdev devices for NUMLOCK key-down/key-up using the read_loop()/close() idiom" -m "- Does not grab() the key, so normal NumLock LED behavior is preserved" -m "- Verified against real keyboard hardware" -m "Phase 2 of vocalis (dictation)" -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01GMitM6j81rZ9TN8k3Q5tAh"
```

---

### Task 4: Text Output (Type via ydotool, Clipboard Fallback) ✅

**Files:**
- Create: `src/vocalis/output/__init__.py`
- Create: `src/vocalis/output/typer.py`
- Create: `tests/test_typer.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `OutputResult.__init__(self, method: str, text: str) -> None` (attrs:
    `method: str` — `"typed"` or `"clipboard"` — and `text: str`)
  - `TextOutput.__init__(self, command_runner: CommandRunner | None = None) -> None`
  - `TextOutput.deliver(self, text: str) -> OutputResult`
  - Task 6 (DictationFeature) calls `deliver()`; Task 7 (CLI wiring)
    constructs a real `TextOutput()`.

- [x] **Step 1: Create the package `__init__.py`**

```python
# src/vocalis/output/__init__.py
"""Text delivery (typing at the cursor, clipboard fallback) for vocalis."""
```

- [x] **Step 2: Write the failing tests**

```python
# tests/test_typer.py
"""Tests for vocalis.output.typer.TextOutput."""
import subprocess

from vocalis.output.typer import TextOutput


class FakeRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], bytes | None]] = []
        self.ydotool_returncode = 0
        self.ydotool_raises: Exception | None = None
        self.xclip_returncode = 0

    def __call__(self, argv: list[str], input_bytes: bytes | None = None):
        self.calls.append((argv, input_bytes))
        if argv[0] == "ydotool":
            if self.ydotool_raises is not None:
                raise self.ydotool_raises
            return subprocess.CompletedProcess(argv, self.ydotool_returncode)
        if argv[0] == "xclip":
            return subprocess.CompletedProcess(argv, self.xclip_returncode)
        raise AssertionError(f"Unexpected command: {argv}")


def test_empty_text_is_a_noop() -> None:
    runner = FakeRunner()
    output = TextOutput(command_runner=runner)
    result = output.deliver("")
    assert result.method == "typed"
    assert result.text == ""
    assert runner.calls == []


def test_successful_typing_does_not_touch_clipboard() -> None:
    runner = FakeRunner()
    output = TextOutput(command_runner=runner)
    result = output.deliver("hello world")
    assert result.method == "typed"
    assert result.text == "hello world"
    assert len(runner.calls) == 1
    assert runner.calls[0][0][:2] == ["ydotool", "type"]


def test_ydotool_nonzero_exit_falls_back_to_clipboard() -> None:
    runner = FakeRunner()
    runner.ydotool_returncode = 1
    output = TextOutput(command_runner=runner)
    result = output.deliver("hello world")
    assert result.method == "clipboard"
    assert len(runner.calls) == 2
    assert runner.calls[1][0][0] == "xclip"
    assert runner.calls[1][1] == b"hello world"


def test_ydotool_missing_binary_falls_back_to_clipboard() -> None:
    runner = FakeRunner()
    runner.ydotool_raises = FileNotFoundError("ydotool not found")
    output = TextOutput(command_runner=runner)
    result = output.deliver("hello world")
    assert result.method == "clipboard"
    assert len(runner.calls) == 2


def test_ydotool_timeout_falls_back_to_clipboard() -> None:
    runner = FakeRunner()
    runner.ydotool_raises = subprocess.TimeoutExpired(cmd="ydotool", timeout=5.0)
    output = TextOutput(command_runner=runner)
    result = output.deliver("hello world")
    assert result.method == "clipboard"


def test_clipboard_also_failing_does_not_raise() -> None:
    runner = FakeRunner()
    runner.ydotool_returncode = 1
    runner.xclip_returncode = 1
    output = TextOutput(command_runner=runner)
    result = output.deliver("hello world")  # must not raise
    assert result.method == "clipboard"
```

- [x] **Step 3: Run the tests, verify they fail**

Run: `.venv/bin/pytest tests/test_typer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vocalis.output.typer'`

- [x] **Step 4: Write the minimal implementation**

```python
# src/vocalis/output/typer.py
"""Delivers transcribed text at the cursor via ydotool, falling back to the
clipboard (via xclip) if typing fails for any reason (ydotoold not running,
binary missing, timeout, non-zero exit).
"""
import logging
import subprocess
from typing import Protocol

logger = logging.getLogger("vocalis")


class CommandRunner(Protocol):
    """Runs a command, optionally piping bytes to its stdin."""

    def __call__(
        self, argv: list[str], input_bytes: bytes | None = None
    ) -> "subprocess.CompletedProcess[bytes]": ...


class OutputResult:
    """Outcome of attempting to deliver transcribed text to the user."""

    def __init__(self, method: str, text: str) -> None:
        """Create an OutputResult.

        Args:
            method: "typed" or "clipboard" — how the text was delivered.
            text: The text that was delivered.
        """
        self.method = method
        self.text = text


class TextOutput:
    """Delivers text via ydotool typing, falling back to clipboard."""

    def __init__(self, command_runner: CommandRunner | None = None) -> None:
        """Create a TextOutput.

        Args:
            command_runner: Callable(argv, input_bytes=None) -> CompletedProcess,
                used to run ydotool/xclip. Defaults to subprocess.run with a
                5-second timeout. Injectable for tests.
        """
        self._run: CommandRunner = command_runner or _default_command_runner

    def deliver(self, text: str) -> OutputResult:
        """Type text at the cursor via ydotool; fall back to clipboard on failure.

        Args:
            text: The text to deliver. Empty text is a no-op.

        Returns:
            An OutputResult describing how the text was delivered.
        """
        if not text:
            return OutputResult(method="typed", text="")

        try:
            result = self._run(["ydotool", "type", "--", text])
            if result.returncode == 0:
                return OutputResult(method="typed", text=text)
            logger.warning(
                "ydotool exited with code %d; falling back to clipboard", result.returncode
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning("ydotool failed (%s); falling back to clipboard", exc)

        try:
            clip_result = self._run(
                ["xclip", "-selection", "clipboard"], input_bytes=text.encode("utf-8")
            )
            if clip_result.returncode != 0:
                logger.error("xclip exited with code %d", clip_result.returncode)
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.error("Clipboard fallback also failed: %s", exc)

        return OutputResult(method="clipboard", text=text)


def _default_command_runner(
    argv: list[str], input_bytes: bytes | None = None
) -> "subprocess.CompletedProcess[bytes]":
    """Default CommandRunner: subprocess.run with a 5-second timeout."""
    return subprocess.run(argv, input=input_bytes, timeout=5.0, capture_output=True)
```

- [x] **Step 5: Run the tests, verify they pass**

Run: `.venv/bin/pytest tests/test_typer.py -v`
Expected: PASS (all 6 tests)

- [x] **Step 6: Run the full suite + mypy**

Run: `.venv/bin/pytest && .venv/bin/mypy`
Expected: All tests PASS, coverage ≥85%, mypy clean.

- [x] **Step 7: Commit**

```bash
git add src/vocalis/output/__init__.py src/vocalis/output/typer.py tests/test_typer.py
git commit -m "feat: add TextOutput with ydotool typing and clipboard fallback" -m "- deliver() types via ydotool, falling back to xclip clipboard on any failure (missing binary, timeout, non-zero exit)" -m "- Single injectable CommandRunner seam covers both ydotool and xclip calls" -m "Phase 2 of vocalis (dictation)" -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01GMitM6j81rZ9TN8k3Q5tAh"
```

---

### Task 5: Desktop Notifier ✅

**Files:**
- Create: `src/vocalis/notify.py`
- Create: `tests/test_notify.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `Notifier.__init__(self, enabled: bool, command_runner: CommandRunner | None = None) -> None`
  - `Notifier.notify(self, title: str, message: str = "") -> None`
  - Task 6 (DictationFeature) calls `notify()`; Task 7 (CLI wiring)
    constructs `Notifier(enabled=config.notifications_enabled)`.

- [x] **Step 1: Write the failing tests**

```python
# tests/test_notify.py
"""Tests for vocalis.notify.Notifier."""
import subprocess

from vocalis.notify import Notifier


class FakeRunner:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.raises: Exception | None = None

    def __call__(self, argv: list[str]):
        self.calls.append(argv)
        if self.raises is not None:
            raise self.raises
        return subprocess.CompletedProcess(argv, 0)


def test_disabled_notifier_sends_nothing() -> None:
    runner = FakeRunner()
    notifier = Notifier(enabled=False, command_runner=runner)
    notifier.notify("Vocalis", "Started")
    assert runner.calls == []


def test_enabled_notifier_sends_title_and_message() -> None:
    runner = FakeRunner()
    notifier = Notifier(enabled=True, command_runner=runner)
    notifier.notify("Vocalis", "Started")
    assert len(runner.calls) == 1
    assert "Vocalis" in runner.calls[0]
    assert "Started" in runner.calls[0]


def test_message_omitted_when_empty() -> None:
    runner = FakeRunner()
    notifier = Notifier(enabled=True, command_runner=runner)
    notifier.notify("Vocalis")
    assert len(runner.calls[0]) == 3  # notify-send, --app-name=Vocalis, title only


def test_notify_send_failure_does_not_raise() -> None:
    runner = FakeRunner()
    runner.raises = FileNotFoundError("notify-send not found")
    notifier = Notifier(enabled=True, command_runner=runner)
    notifier.notify("Vocalis", "Started")  # must not raise
```

- [x] **Step 2: Run the tests, verify they fail**

Run: `.venv/bin/pytest tests/test_notify.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vocalis.notify'`

- [x] **Step 3: Write the minimal implementation**

```python
# src/vocalis/notify.py
"""Desktop notifications for vocalis features, via notify-send. Respects
VocalisConfig.notifications_enabled — when disabled, notify() is a no-op.
"""
import logging
import subprocess
from typing import Callable

logger = logging.getLogger("vocalis")

CommandRunner = Callable[[list[str]], "subprocess.CompletedProcess[bytes]"]


class Notifier:
    """Sends desktop notifications, or does nothing if disabled."""

    def __init__(self, enabled: bool, command_runner: CommandRunner | None = None) -> None:
        """Create a Notifier.

        Args:
            enabled: Whether to actually send notifications.
            command_runner: Callable(argv) -> CompletedProcess, used to run
                notify-send. Defaults to subprocess.run with a 5-second
                timeout. Injectable for tests.
        """
        self._enabled = enabled
        self._run: CommandRunner = command_runner or _default_command_runner

    def notify(self, title: str, message: str = "") -> None:
        """Send a desktop notification, if enabled.

        A failure to send never raises — a missing notification is not
        worth crashing a feature over.

        Args:
            title: Notification title.
            message: Notification body (optional).
        """
        if not self._enabled:
            return
        argv = ["notify-send", "--app-name=Vocalis", title]
        if message:
            argv.append(message)
        try:
            self._run(argv)
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning("notify-send failed: %s", exc)


def _default_command_runner(argv: list[str]) -> "subprocess.CompletedProcess[bytes]":
    """Default CommandRunner: subprocess.run with a 5-second timeout."""
    return subprocess.run(argv, timeout=5.0, capture_output=True)
```

- [x] **Step 4: Run the tests, verify they pass**

Run: `.venv/bin/pytest tests/test_notify.py -v`
Expected: PASS (all 4 tests)

- [x] **Step 5: Run the full suite + mypy**

Run: `.venv/bin/pytest && .venv/bin/mypy`
Expected: All tests PASS, coverage ≥85%, mypy clean.

- [x] **Step 6: Commit**

```bash
git add src/vocalis/notify.py tests/test_notify.py
git commit -m "feat: add Notifier for desktop notifications" -m "- notify() wraps notify-send, no-ops when disabled, never raises on failure" -m "Phase 2 of vocalis (dictation)" -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01GMitM6j81rZ9TN8k3Q5tAh"
```

---

### Task 6: DictationFeature Orchestration ✅

**Files:**
- Create: `src/vocalis/features/__init__.py`
- Create: `src/vocalis/features/dictation.py`
- Create: `tests/test_dictation.py`

**Interfaces:**
- Consumes: `ModelManager`, `ModelEngine` (Task 4 of Phase 0+1); `Notifier.notify` (Task 5); `TextOutput.deliver` (Task 4); the `Recorder`/`Transcriber` shapes match `MicRecorder`/`ParakeetSTT` (Tasks 1-2) but are declared as local Protocols here so this module doesn't import GPU/audio libraries directly.
- Produces:
  - `DictationFeature.__init__(self, model_manager: ModelManager, engine_factory: Callable[[], ModelEngine], recorder: Recorder, text_output: TextOutput, notifier: Notifier, sample_rate: int = 16000) -> None`
  - `DictationFeature.on_press(self) -> None`
  - `DictationFeature.on_release(self) -> None`
  - Task 7 (CLI wiring) constructs `DictationFeature` with real
    collaborators and wires its `on_press`/`on_release` to a
    `NumlockListener`.

- [x] **Step 1: Create the package `__init__.py`**

```python
# src/vocalis/features/__init__.py
"""Feature orchestration modules — wire hotkeys/UI to engines/output."""
```

- [x] **Step 2: Write the failing tests**

```python
# tests/test_dictation.py
"""Tests for vocalis.features.dictation.DictationFeature.

Uses fakes for every collaborator (engine, recorder, text output,
notifier) — this suite verifies orchestration logic only. Each
collaborator's own real-world correctness is tested in its own test file
(test_parakeet.py, test_capture.py, test_typer.py, test_notify.py).
"""
import numpy as np

from vocalis.features.dictation import DictationFeature
from vocalis.model_manager import ModelManager


class FakeEngine:
    def __init__(self, transcript: str = "hello world", raise_on_transcribe: Exception | None = None) -> None:
        self.transcript = transcript
        self.raise_on_transcribe = raise_on_transcribe
        self._loaded = False
        self.load_calls = 0
        self.unload_calls = 0
        self.transcribe_calls: list[tuple] = []

    def load(self) -> None:
        self.load_calls += 1
        self._loaded = True

    def unload(self) -> None:
        self.unload_calls += 1
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def transcribe(self, audio, sample_rate):
        self.transcribe_calls.append((audio, sample_rate))
        if self.raise_on_transcribe is not None:
            raise self.raise_on_transcribe
        return self.transcript


class FakeRecorder:
    def __init__(
        self,
        audio_to_return,
        raise_on_start: Exception | None = None,
        raise_on_stop: Exception | None = None,
    ) -> None:
        self.audio_to_return = audio_to_return
        self.raise_on_start = raise_on_start
        self.raise_on_stop = raise_on_stop
        self.start_calls = 0
        self.stop_calls = 0

    def start(self) -> None:
        self.start_calls += 1
        if self.raise_on_start is not None:
            raise self.raise_on_start

    def stop(self):
        self.stop_calls += 1
        if self.raise_on_stop is not None:
            raise self.raise_on_stop
        return self.audio_to_return


class FakeOutputResult:
    def __init__(self, method: str, text: str) -> None:
        self.method = method
        self.text = text


class FakeTextOutput:
    def __init__(self, method: str = "typed") -> None:
        self.method = method
        self.delivered: list[str] = []

    def deliver(self, text: str):
        self.delivered.append(text)
        return FakeOutputResult(method=self.method, text=text)


class FakeNotifier:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def notify(self, title: str, message: str = "") -> None:
        self.messages.append((title, message))


def make_feature(engine, audio, output_method: str = "typed"):
    manager = ModelManager()
    recorder = FakeRecorder(audio_to_return=audio)
    text_output = FakeTextOutput(method=output_method)
    notifier = FakeNotifier()
    feature = DictationFeature(
        model_manager=manager,
        engine_factory=lambda: engine,
        recorder=recorder,
        text_output=text_output,
        notifier=notifier,
    )
    return feature, recorder, text_output, notifier, manager


def test_press_starts_recording_and_notifies() -> None:
    engine = FakeEngine()
    feature, recorder, _, notifier, _ = make_feature(engine, np.zeros(0, dtype=np.float32))
    feature.on_press()
    assert recorder.start_calls == 1
    assert any("Listening" in message for _, message in notifier.messages)


def test_release_transcribes_and_delivers_result() -> None:
    engine = FakeEngine(transcript="hello world")
    audio = np.ones(1600, dtype=np.float32)
    feature, recorder, text_output, notifier, manager = make_feature(engine, audio)
    feature.on_release()
    assert recorder.stop_calls == 1
    assert len(engine.transcribe_calls) == 1
    called_audio, called_sample_rate = engine.transcribe_calls[0]
    assert called_sample_rate == 16000
    assert np.array_equal(called_audio, audio)
    assert text_output.delivered == ["hello world"]
    assert manager.current_engine_id is None  # released back after use
    assert any(message == "Done" for _, message in notifier.messages)


def test_empty_audio_skips_transcription() -> None:
    engine = FakeEngine()
    feature, recorder, text_output, notifier, _ = make_feature(engine, np.zeros(0, dtype=np.float32))
    feature.on_release()
    assert engine.transcribe_calls == []
    assert text_output.delivered == []
    assert any("No audio" in message for _, message in notifier.messages)


def test_blank_transcript_skips_output() -> None:
    engine = FakeEngine(transcript="   ")
    audio = np.ones(1600, dtype=np.float32)
    feature, _, text_output, notifier, _ = make_feature(engine, audio)
    feature.on_release()
    assert text_output.delivered == []
    assert any("no speech" in message.lower() for _, message in notifier.messages)


def test_transcription_failure_notifies_and_does_not_crash() -> None:
    engine = FakeEngine(raise_on_transcribe=RuntimeError("boom"))
    audio = np.ones(1600, dtype=np.float32)
    feature, _, text_output, notifier, manager = make_feature(engine, audio)
    feature.on_release()  # must not raise
    assert text_output.delivered == []
    assert any("Transcription failed" in message for _, message in notifier.messages)
    assert manager.current_engine_id is None  # still released despite failure


def test_clipboard_fallback_notifies_accordingly() -> None:
    engine = FakeEngine(transcript="hello")
    audio = np.ones(1600, dtype=np.float32)
    feature, _, _, notifier, _ = make_feature(engine, audio, output_method="clipboard")
    feature.on_release()
    assert any("clipboard" in message.lower() for _, message in notifier.messages)


def test_recorder_stop_failure_notifies_and_does_not_crash() -> None:
    engine = FakeEngine()
    manager = ModelManager()
    recorder = FakeRecorder(
        audio_to_return=np.zeros(0, dtype=np.float32), raise_on_stop=RuntimeError("mic error")
    )
    text_output = FakeTextOutput()
    notifier = FakeNotifier()
    feature = DictationFeature(
        model_manager=manager,
        engine_factory=lambda: engine,
        recorder=recorder,
        text_output=text_output,
        notifier=notifier,
    )
    feature.on_release()  # must not raise
    assert any("Recording failed" in message for _, message in notifier.messages)


def test_recorder_start_failure_notifies_and_does_not_crash() -> None:
    engine = FakeEngine()
    manager = ModelManager()
    recorder = FakeRecorder(
        audio_to_return=np.zeros(0, dtype=np.float32), raise_on_start=RuntimeError("mic busy")
    )
    text_output = FakeTextOutput()
    notifier = FakeNotifier()
    feature = DictationFeature(
        model_manager=manager,
        engine_factory=lambda: engine,
        recorder=recorder,
        text_output=text_output,
        notifier=notifier,
    )
    feature.on_press()  # must not raise
    assert any("Could not start recording" in message for _, message in notifier.messages)
```

- [x] **Step 3: Run the tests, verify they fail**

Run: `.venv/bin/pytest tests/test_dictation.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vocalis.features.dictation'`

- [x] **Step 4: Write the minimal implementation**

```python
# src/vocalis/features/dictation.py
"""Orchestrates the dictation feature: on NUMLOCK press, start recording;
on release, stop recording, transcribe via the Parakeet engine (borrowed
from ModelManager), and deliver the text (typed, or clipboard fallback) —
with desktop notifications at each stage.
"""
import logging
from typing import Callable, Protocol, cast

import numpy as np
import numpy.typing as npt

from vocalis.model_manager import ModelEngine, ModelManager
from vocalis.notify import Notifier
from vocalis.output.typer import TextOutput

logger = logging.getLogger("vocalis")

_ENGINE_ID = "parakeet"
_TITLE = "Vocalis — Dictation"


class Transcriber(Protocol):
    """The subset of ParakeetSTT this module depends on."""

    def transcribe(self, audio: npt.NDArray[np.float32], sample_rate: int) -> str: ...


class Recorder(Protocol):
    """The subset of MicRecorder this module depends on."""

    def start(self) -> None: ...
    def stop(self) -> npt.NDArray[np.float32]: ...


class DictationFeature:
    """Wires NUMLOCK press/release to record -> transcribe -> deliver."""

    def __init__(
        self,
        model_manager: ModelManager,
        engine_factory: Callable[[], ModelEngine],
        recorder: Recorder,
        text_output: TextOutput,
        notifier: Notifier,
        sample_rate: int = 16000,
    ) -> None:
        """Create a DictationFeature.

        Args:
            model_manager: Shared ModelManager enforcing single-residency.
            engine_factory: Constructs a fresh engine (e.g. ParakeetSTT)
                when the model needs to be (re)loaded. The constructed
                engine must also implement Transcriber (have a
                transcribe() method) — ModelManager itself only requires
                the ModelEngine protocol, so this is a documented
                requirement on the factory, not something the type system
                enforces here.
            recorder: Microphone recorder.
            text_output: Delivers transcribed text (typed or clipboard).
            notifier: Sends start/transcribing/done notifications.
            sample_rate: Sample rate to record and transcribe at.
        """
        self._model_manager = model_manager
        self._engine_factory = engine_factory
        self._recorder = recorder
        self._text_output = text_output
        self._notifier = notifier
        self._sample_rate = sample_rate

    def on_press(self) -> None:
        """Call when NUMLOCK is pressed: begin recording."""
        self._notifier.notify(_TITLE, "Listening...")
        try:
            self._recorder.start()
        except Exception:
            logger.exception("Failed to start recording")
            self._notifier.notify(_TITLE, "Could not start recording")

    def on_release(self) -> None:
        """Call when NUMLOCK is released: transcribe and deliver the text."""
        try:
            audio = self._recorder.stop()
        except Exception:
            logger.exception("Failed to stop recording")
            self._notifier.notify(_TITLE, "Recording failed")
            return

        if audio.size == 0:
            self._notifier.notify(_TITLE, "No audio captured")
            return

        self._notifier.notify(_TITLE, "Transcribing...")
        try:
            engine = self._model_manager.acquire(_ENGINE_ID, self._engine_factory)
            try:
                text = cast(Transcriber, engine).transcribe(audio, sample_rate=self._sample_rate)
            finally:
                self._model_manager.release(_ENGINE_ID)
        except Exception:
            logger.exception("Transcription failed")
            self._notifier.notify(_TITLE, "Transcription failed")
            return

        if not text.strip():
            self._notifier.notify(_TITLE, "(no speech detected)")
            return

        self._notifier.notify(_TITLE, text)
        result = self._text_output.deliver(text)
        if result.method == "clipboard":
            self._notifier.notify(_TITLE, "Typing failed — copied to clipboard instead")
        else:
            self._notifier.notify(_TITLE, "Done")
```

- [x] **Step 5: Run the tests, verify they pass**

Run: `.venv/bin/pytest tests/test_dictation.py -v`
Expected: PASS (all 8 tests)

- [x] **Step 6: Run the full suite + mypy**

Run: `.venv/bin/pytest && .venv/bin/mypy`
Expected: All tests PASS, coverage ≥85%, mypy clean.

- [x] **Step 7: Commit**

```bash
git add src/vocalis/features/__init__.py src/vocalis/features/dictation.py tests/test_dictation.py
git commit -m "feat: add DictationFeature orchestrating record -> transcribe -> deliver" -m "- on_press()/on_release() wire NUMLOCK events to MicRecorder, ModelManager-acquired Parakeet engine, and TextOutput" -m "- Notifies at listening/transcribing/done, and on every failure path (recording, transcription, clipboard fallback)" -m "- Fully tested with fakes for every collaborator; real ModelManager used to prove acquire/release wiring" -m "Phase 2 of vocalis (dictation)" -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01GMitM6j81rZ9TN8k3Q5tAh"
```

---

### Task 7: Wire Dictation into the CLI/Daemon ✅

**Files:**
- Modify: `src/vocalis/config.py` (add `parakeet_model_dir` to `EnginesConfig`)
- Modify: `config/default_config.toml` (add the corresponding entry)
- Modify: `src/vocalis/cli.py` (share one `ModelManager` between the daemon
  and the dictation feature; add `build_dictation`; start/stop the listener
  in `main()`)
- Modify: `tests/test_config.py` (add a test for the new field)
- Modify: `tests/test_cli.py` (update for `build_daemon`'s new signature;
  add a test for `build_dictation`; update the `main()` test to mock the
  new device enumerator)

**Interfaces:**
- Consumes: `ParakeetSTT` (Task 1), `MicRecorder` (Task 2), `NumlockListener`
  + `list_numlock_capable_devices` (Task 3), `TextOutput` (Task 4),
  `Notifier` (Task 5), `DictationFeature` (Task 6).
- Produces:
  - `EnginesConfig.parakeet_model_dir: str` (new field, default computed
    from `Path.home()`, pointing at the existing Voxtype model directory).
  - `build_daemon(model_manager: ModelManager, socket_path: Path) -> VocalisDaemon`
    (signature CHANGES from Phase 0+1: now takes an already-constructed
    `ModelManager` instead of a `VocalisConfig`, since the `ModelManager`
    must now be shared with `build_dictation` too).
  - `build_dictation(config: VocalisConfig, model_manager: ModelManager) -> tuple[DictationFeature, NumlockListener]`
  - `main()` now also starts/stops the `NumlockListener` alongside the daemon.

- [x] **Step 1: Write the failing test for the new config field**

Add this test to `tests/test_config.py` (append near the end, after the
existing tests):

```python
def test_parakeet_model_dir_defaults_under_home() -> None:
    config = VocalisConfig()
    assert config.engines.parakeet_model_dir.endswith(
        "/.local/share/voxtype/models/parakeet-tdt-0.6b-v3-int8"
    )
    assert str(Path.home()) in config.engines.parakeet_model_dir
```

- [x] **Step 2: Run the test, verify it fails**

Run: `.venv/bin/pytest tests/test_config.py::test_parakeet_model_dir_defaults_under_home -v`
Expected: FAIL — `AttributeError: 'EnginesConfig' object has no attribute 'parakeet_model_dir'`

- [x] **Step 3: Add the config field**

In `src/vocalis/config.py`, add this function above the `EnginesConfig`
class:

```python
def _default_parakeet_model_dir() -> str:
    """Default location of the Parakeet model files.

    Points at the existing Voxtype installation's model directory — Phase 2
    reuses those exact files rather than re-downloading them. A future
    packaging phase will add a proper first-run download for machines that
    don't already have a Voxtype install.
    """
    return str(Path.home() / ".local" / "share" / "voxtype" / "models" / "parakeet-tdt-0.6b-v3-int8")
```

Then change the `EnginesConfig` class from:

```python
class EnginesConfig(BaseModel):
    """Which engine implementation each feature should use."""

    stt: str = "parakeet"
    quick_tts: str = "qwen3_tts"
    doc_tts: str = "fish_speech"
```

to:

```python
class EnginesConfig(BaseModel):
    """Which engine implementation each feature should use."""

    stt: str = "parakeet"
    quick_tts: str = "qwen3_tts"
    doc_tts: str = "fish_speech"
    parakeet_model_dir: str = Field(default_factory=_default_parakeet_model_dir)
```

(`Field` is already imported in this file from the Phase 0+1 final-review
fix wave.)

- [x] **Step 4: Run the test, verify it passes**

Run: `.venv/bin/pytest tests/test_config.py::test_parakeet_model_dir_defaults_under_home -v`
Expected: PASS

- [x] **Step 5: Update the default config TOML**

In `config/default_config.toml`, add `parakeet_model_dir` to the
`[engines]` section:

```toml
[engines]
stt = "parakeet"
quick_tts = "qwen3_tts"
doc_tts = "fish_speech"
parakeet_model_dir = "/home/redx/.local/share/voxtype/models/parakeet-tdt-0.6b-v3-int8"
```

(This TOML file is a reference copy for documentation/drift-protection
only — `load_config()` never reads it directly; see its own header
comment. The existing `test_default_toml_matches_model_defaults` test in
`tests/test_config.py` will confirm this value matches the model's
computed default on this machine.)

- [x] **Step 6: Run the full config test suite**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: PASS (all tests, including `test_default_toml_matches_model_defaults`)

- [x] **Step 7: Write the failing tests for the CLI wiring changes**

Replace the entire contents of `tests/test_cli.py` with:

```python
"""Tests for vocalis.cli wiring: config -> logging -> ModelManager -> daemon
-> dictation feature."""
from pathlib import Path
from unittest.mock import patch

from vocalis.cli import build_daemon, build_dictation
from vocalis.config import VocalisConfig
from vocalis.model_manager import ModelManager


def test_build_daemon_uses_the_given_model_manager(tmp_path: Path) -> None:
    manager = ModelManager(idle_timeout_seconds=12.5)
    socket_path = tmp_path / "vocalis.sock"

    daemon = build_daemon(model_manager=manager, socket_path=socket_path)

    assert daemon._model_manager is manager  # noqa: SLF001
    assert daemon._socket_path == socket_path  # noqa: SLF001


def test_build_dictation_wires_engine_factory_to_configured_model_dir() -> None:
    config = VocalisConfig()
    manager = ModelManager()

    feature, listener = build_dictation(config=config, model_manager=manager)

    engine = feature._engine_factory()  # noqa: SLF001
    assert str(engine._model_dir) == config.engines.parakeet_model_dir  # noqa: SLF001
    assert listener._on_press == feature.on_press  # noqa: SLF001
    assert listener._on_release == feature.on_release  # noqa: SLF001


def test_main_starts_and_stops_daemon_on_keyboard_interrupt(tmp_path: Path) -> None:
    socket_path = tmp_path / "vocalis.sock"

    with (
        patch("vocalis.cli.default_config_path", return_value=tmp_path / "config.toml"),
        patch("vocalis.cli._socket_path", return_value=socket_path),
        patch("vocalis.cli.list_numlock_capable_devices", return_value=[]),
        patch("vocalis.cli._block_until_interrupted", side_effect=KeyboardInterrupt),
    ):
        from vocalis.cli import main

        main()

    assert not socket_path.exists()
```

- [x] **Step 8: Run the tests, verify they fail**

Run: `.venv/bin/pytest tests/test_cli.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_dictation' from 'vocalis.cli'`
(and `build_daemon`'s old signature won't match the new test's kwargs yet).

- [x] **Step 9: Rewrite `src/vocalis/cli.py`**

Replace its entire contents with:

```python
"""The `vocalis` command-line entrypoint.

Wires together config loading, logging, a single shared ModelManager, the
VocalisDaemon, and the dictation feature, then blocks until interrupted via
Ctrl+C (KeyboardInterrupt).

Graceful SIGTERM handling (needed once this runs under systemd in Phase 7)
is not yet implemented: SIGTERM currently terminates the process via
Python's default disposition, without running daemon.stop().
"""
import time
from pathlib import Path

from vocalis.config import VocalisConfig, default_config_path, load_config
from vocalis.daemon import VocalisDaemon
from vocalis.engines.parakeet import ParakeetSTT
from vocalis.features.dictation import DictationFeature
from vocalis.hotkeys.numlock import NumlockListener, list_numlock_capable_devices
from vocalis.logging_setup import configure_logging
from vocalis.model_manager import ModelManager
from vocalis.notify import Notifier
from vocalis.output.typer import TextOutput
from vocalis.audio.capture import MicRecorder


def _socket_path() -> Path:
    """Return the standard runtime location for the daemon's Unix socket."""
    return Path.home() / ".local" / "state" / "vocalis" / "vocalis.sock"


def build_daemon(model_manager: ModelManager, socket_path: Path) -> VocalisDaemon:
    """Construct a VocalisDaemon using the given shared ModelManager.

    Args:
        model_manager: The ModelManager this daemon (and any features
            sharing it) will use. Constructed once by the caller so it can
            also be passed to build_dictation() — only one ModelManager
            may exist per running vocalis process, since it enforces
            single-model GPU residency across every feature.
        socket_path: Where the daemon's Unix socket should live.

    Returns:
        A VocalisDaemon ready to have .start() called on it.
    """
    logger = configure_logging(level="INFO")
    return VocalisDaemon(socket_path=socket_path, logger=logger, model_manager=model_manager)


def build_dictation(
    config: VocalisConfig, model_manager: ModelManager
) -> tuple[DictationFeature, NumlockListener]:
    """Construct the dictation feature and its NUMLOCK listener from config.

    Args:
        config: Loaded, validated vocalis configuration.
        model_manager: The shared ModelManager (same instance passed to
            build_daemon()).

    Returns:
        A (feature, listener) pair. The caller is responsible for calling
        listener.start()/listener.stop() around the daemon's own lifecycle.
    """
    model_dir = Path(config.engines.parakeet_model_dir)

    def engine_factory() -> ParakeetSTT:
        return ParakeetSTT(model_dir=model_dir)

    recorder = MicRecorder(max_duration_seconds=60.0)
    text_output = TextOutput()
    notifier = Notifier(enabled=config.notifications_enabled)
    feature = DictationFeature(
        model_manager=model_manager,
        engine_factory=engine_factory,
        recorder=recorder,
        text_output=text_output,
        notifier=notifier,
    )
    listener = NumlockListener(
        on_press=feature.on_press,
        on_release=feature.on_release,
        device_enumerator=list_numlock_capable_devices,
    )
    return feature, listener


def _block_until_interrupted() -> None:
    """Sleep forever until interrupted by Ctrl+C (KeyboardInterrupt).

    Does not currently handle SIGTERM — see the module docstring.
    """
    while True:
        time.sleep(1.0)


def main() -> None:
    """Entrypoint for the `vocalis` console script.

    Loads config, starts the daemon and the dictation hotkey listener, and
    blocks until the process is interrupted, at which point it stops both
    cleanly. Both start()/stop() calls are inside the try/finally, so
    daemon.stop() (and therefore ModelManager.shutdown()) always runs on
    the way out — including if either start() call itself fails partway
    through, since both components' stop() methods are safe to call
    regardless of how much of start() completed.
    """
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

- [x] **Step 10: Run the tests, verify they pass**

Run: `.venv/bin/pytest tests/test_cli.py -v`
Expected: PASS (all 3 tests)

- [x] **Step 11: Run the full suite + mypy**

Run: `.venv/bin/pytest && .venv/bin/mypy`
Expected: All tests PASS, coverage ≥85%, mypy clean.

- [x] **Step 12: Manual smoke test — real components, no real NUMLOCK hardware needed**

This proves the whole pipeline (real Parakeet model, real microphone, real
ydotool/notify-send) works end-to-end without needing an AI agent to
physically hold a key. Run it in the background so it can be observed, per
this project's convention for long-running processes — though this
particular script finishes in a few seconds, so a plain foreground run is
fine too:

```python
# Run interactively or as a throwaway script.
from pathlib import Path
from vocalis.model_manager import ModelManager
from vocalis.cli import build_dictation
from vocalis.config import VocalisConfig
import time

config = VocalisConfig()
manager = ModelManager()
feature, _ = build_dictation(config=config, model_manager=manager)

print("Speak now for 3 seconds...")
feature.on_press()
time.sleep(3)
feature.on_release()
print("Check for notifications and typed/clipboard text.")
```

Expected: a "Listening..." notification appears, then after 3 seconds a
"Transcribing..." notification, then the transcribed text appears as a
notification and is typed wherever your cursor was focused (or copied to
the clipboard if `ydotool type` failed) followed by a "Done" notification.

- [x] **Step 13: Commit**

```bash
git add src/vocalis/config.py config/default_config.toml src/vocalis/cli.py tests/test_config.py tests/test_cli.py
git commit -m "feat: wire DictationFeature into the vocalis CLI and daemon" -m "- build_daemon() now takes an already-constructed ModelManager so it can be shared with build_dictation()" -m "- build_dictation() wires ParakeetSTT, MicRecorder, TextOutput, Notifier, and NumlockListener from config" -m "- main() starts/stops the NUMLOCK listener alongside the daemon" -m "- Adds engines.parakeet_model_dir config field, pointing at the existing Voxtype model files by default" -m "- Verified end-to-end with real model/mic/ydotool/notify-send components" -m "Phase 2 of vocalis (dictation)" -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01GMitM6j81rZ9TN8k3Q5tAh"
```

---

### Task 8: Docs + Changelog ✅

**Files:**
- Modify: `docs/architecture.md` (add a Phase 2 section)
- Modify: `docs/folder-structure.md` (add the new directories/files)
- Modify: `docs/file-purpose.md` (add rows for the new files)
- Modify: `docs/workflow.md` (add the dictation press/release flow)
- Modify: `docs/user-docs.md` (document how to use dictation)
- Create: `.claude/2026-09-11_1300_CHANGELOG.md`

**Interfaces:**
- Consumes: nothing (documentation only).
- Produces: updated governance docs describing exactly what Phase 2 adds,
  matching the actual final code (including the `build_daemon` signature
  change and the config field addition from Task 7).

- [x] **Step 1: Append to `docs/architecture.md`**

Add this section at the end of the file:

```markdown

## Phase 2 addition: Dictation (STT)

Holding NUMLOCK now works end-to-end:

1. `NumlockListener` (evdev, one background thread per keyboard-capable
   device) detects NUMLOCK press/release without suppressing the OS's own
   NumLock handling.
2. On press, `DictationFeature.on_press()` notifies "Listening..." and
   starts `MicRecorder` (sounddevice, float32 @ 16kHz, 60s safety cutoff).
3. On release, `DictationFeature.on_release()` stops the recorder, notifies
   "Transcribing...", and calls `model_manager.acquire("parakeet", ...)` to
   get a loaded `ParakeetSTT` engine — obeying the same single-residency
   and idle-unload policy every future engine will use.
4. `ParakeetSTT` wraps the `onnx-asr` package running the existing Voxtype
   Parakeet TDT ONNX model, GPU (CUDA) by default.
5. The transcribed text is delivered via `TextOutput` (ydotool typing,
   clipboard fallback on any failure) and the engine is released back to
   `ModelManager` (which starts its idle-unload timer).
6. Notifications fire at each stage: listening, transcribing, the
   transcript itself, and done (or the specific failure reason).

`build_daemon()` changed signature in this phase: it now takes an
already-constructed `ModelManager` instead of building its own from
config, because `main()` needs one shared `ModelManager` instance across
both the daemon and the dictation feature — only one may exist per
process, since it enforces single-model GPU residency app-wide.

See `docs/algorithms.md` for how `ModelManager`'s policy applies here
unchanged, and the design spec's Hard Constraint #2 amendment for the
confirmed small (~150MB) fixed GPU residual once any engine has been used.
```

- [x] **Step 2: Update `docs/folder-structure.md`**

In the `src/vocalis/` tree, add these new entries in their alphabetical
position within the existing structure:

```
├── audio/
│   └── capture.py        # MicRecorder: sounddevice-based mic capture
├── engines/
│   └── parakeet.py        # ParakeetSTT: onnx-asr wrapper implementing ModelEngine
├── features/
│   └── dictation.py       # DictationFeature: press/release orchestration
├── hotkeys/
│   └── numlock.py         # NumlockListener: evdev NUMLOCK push-to-talk
├── notify.py               # Notifier: desktop notifications (notify-send)
├── output/
│   └── typer.py            # TextOutput: ydotool typing + clipboard fallback
```

(These replace the "Feature code ... added by later phases" placeholder
note for these specific directories; other still-unbuilt directories like
`ui/` remain noted as future work.)

- [x] **Step 3: Update `docs/file-purpose.md`**

Add these rows to the table:

```markdown
| `src/vocalis/engines/parakeet.py` | `ParakeetSTT` — wraps `onnx-asr` running the existing Voxtype Parakeet TDT ONNX model; implements `ModelEngine`. |
| `src/vocalis/audio/capture.py` | `MicRecorder` — sounddevice-based microphone capture with a safety-duration cutoff. |
| `src/vocalis/hotkeys/numlock.py` | `NumlockListener` — evdev NUMLOCK push-to-talk detection; `list_numlock_capable_devices()` is the real device enumerator. |
| `src/vocalis/output/typer.py` | `TextOutput` — types text via ydotool, falls back to clipboard (xclip) on any failure. |
| `src/vocalis/notify.py` | `Notifier` — desktop notifications via notify-send, no-op when disabled. |
| `src/vocalis/features/dictation.py` | `DictationFeature` — wires NUMLOCK press/release to record → transcribe → deliver, with notifications at each stage. |
```

- [x] **Step 4: Append to `docs/workflow.md`**

Add this section at the end of the file:

```markdown

## Dictation press/release flow (Phase 2)

```
NUMLOCK pressed (evdev event, NumlockListener)
  -> DictationFeature.on_press()
       -> notify("Listening...")
       -> recorder.start()                      # begins capturing mic audio

NUMLOCK released (evdev event, NumlockListener)
  -> DictationFeature.on_release()
       -> audio = recorder.stop()
       -> if no audio captured: notify("No audio captured"), stop
       -> notify("Transcribing...")
       -> engine = model_manager.acquire("parakeet", parakeet_factory)
       -> text = engine.transcribe(audio, sample_rate=16000)
       -> model_manager.release("parakeet")      # starts the idle-unload timer
       -> if blank transcript: notify("(no speech detected)"), stop
       -> notify(text)                            # live transcript notification
       -> result = text_output.deliver(text)      # ydotool, or clipboard on failure
       -> notify("Done" | "Typing failed — copied to clipboard instead")
```

Every failure path (recording start/stop, transcription, clipboard-also-
failing) notifies a specific reason and returns without crashing the
listener thread or the daemon.
```

- [x] **Step 5: Rewrite `docs/user-docs.md`**

Replace its entire contents with:

```markdown
# User Documentation

## Dictation (Phase 2)

1. Hold **NUMLOCK**.
2. Speak. A "Listening..." notification confirms recording started.
3. Release NUMLOCK. You'll see "Transcribing...", then the transcribed
   text, then either the text typed at your cursor or (if typing failed —
   e.g. `ydotoold` isn't running) a note that it was copied to your
   clipboard instead.

Your normal NumLock LED/state is unaffected — this only listens for the
key, it doesn't suppress it.

## Running the daemon manually (development only)

```bash
.venv/bin/vocalis
```

Press Ctrl+C to stop it. Once Phase 7 (Service & autostart) lands, this
will instead run automatically via `systemctl --user start vocalis`.

## Configuration

Edit `~/.config/vocalis/config.toml` (created on first save, or copy
`config/default_config.toml`). Relevant Phase 2 settings:

- `notifications_enabled` — set to `false` to silence all notifications.
- `model_idle_timeout_seconds` — how long the Parakeet model stays loaded
  in VRAM after you stop dictating before it's automatically released.
- `[engines] parakeet_model_dir` — path to the Parakeet ONNX model files
  (defaults to the existing Voxtype installation's model directory).
```

- [x] **Step 6: Write the changelog entry**

```markdown
<!-- .claude/2026-09-11_1300_CHANGELOG.md -->
# Changelog — 2026-09-11 (Phase 2)

## Vocalis Phase 2: Dictation (STT)

Holding NUMLOCK now works inside Vocalis, reusing the exact Parakeet model
files the existing Voxtype binary already uses.

**Added:**
- `vocalis.engines.parakeet.ParakeetSTT` — wraps the `onnx-asr` package
  running the local Parakeet TDT ONNX model (GPU by default, CPU fallback).
  Confirmed via direct testing: no model re-download or conversion needed.
- `vocalis.audio.capture.MicRecorder` — sounddevice-based microphone
  capture with a safety-duration cutoff that preserves already-captured
  audio.
- `vocalis.hotkeys.numlock.NumlockListener` — evdev-based NUMLOCK
  push-to-talk detection, one thread per keyboard device, does not
  suppress normal NumLock behavior.
- `vocalis.output.typer.TextOutput` — types via ydotool, falls back to
  clipboard (xclip) on any failure.
- `vocalis.notify.Notifier` — desktop notifications via notify-send.
- `vocalis.features.dictation.DictationFeature` — orchestrates the full
  press → record → release → transcribe → deliver flow with notifications
  at every stage and failure path.
- `EnginesConfig.parakeet_model_dir` — new config field pointing at the
  Parakeet model directory (default: the existing Voxtype install).

**Changed:**
- `vocalis.cli.build_daemon()` now takes an already-constructed
  `ModelManager` instead of building its own, so the same instance can be
  shared with the new `build_dictation()` — only one `ModelManager` may
  exist per process (it enforces single-model GPU residency app-wide).

**Spec amendments (recorded in the design spec, not just here):**
- Confirmed `onnx-asr` as the STT library and documented the exact
  loading/GPU setup (cuDNN preload workaround).
- Confirmed and user-approved a small fixed (~150MB) GPU memory floor once
  the process has used the GPU at all — a CUDA/ONNX Runtime characteristic
  ModelManager's idle-unload cannot eliminate, not a defect in this phase's
  code. Applies to every future GPU-backed engine too.

**Not yet implemented (future phases):** Quick Reader, Document Reader,
OCR, UI (tray/widget/settings), systemd service, packaging.

**Rollback:** additive only — `old/` and Phase 0+1 code untouched beyond
the documented `build_daemon()` signature change (a same-phase-cycle
refactor, not a breaking change to already-shipped behavior). Revert via
`git revert` back to before this phase's commits.
```

- [x] **Step 7: Commit**

```bash
git add docs/architecture.md docs/folder-structure.md docs/file-purpose.md docs/workflow.md docs/user-docs.md .claude/2026-09-11_1300_CHANGELOG.md
git commit -m "docs: document phase 2 dictation feature" -m "- Architecture, folder structure, file purpose, workflow, and user docs updated to describe the built dictation pipeline" -m "- Adds changelog entry summarizing phase 2" -m "Phase 2 of vocalis (dictation)" -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01GMitM6j81rZ9TN8k3Q5tAh"
```

---

## After This Plan — Manual Verification (you, not an agent)

An AI agent cannot physically hold a key and speak, so the final proof
this phase actually replaces Voxtype is a manual check:

1. Run `.venv/bin/vocalis` in the background (`&` or a separate terminal).
2. Hold NUMLOCK, say a full sentence, release it.
3. Confirm: "Listening..." → "Transcribing..." → the transcript appears →
   text is typed at your cursor (test with a text editor or terminal
   focused) → "Done".
4. Try it with no text field focused (or `ydotoold` briefly stopped) to
   confirm the clipboard fallback and its notification.
5. Compare transcription accuracy/latency subjectively against the
   existing Voxtype binary (`~/.local/bin/start-voxtype.sh`) for a few
   sentences — they use the identical model, so results should be very
   close.
6. Watch `nvidia-smi -l 1` during a few dictation cycles: VRAM should rise
   to a few hundred MB while transcribing and settle back to the ~150MB
   floor (not 15MB — see the spec's Hard Constraint #2 amendment) within
   `model_idle_timeout_seconds` of the last release.

Once you're satisfied dictation works, the next implementation plan (per
the spec's milestone list) is **Phase 3 — Quick Reader (TTS)**: wrap
Qwen3-TTS behind the `ModelEngine` protocol, grab PRIMARY-selection/
CLIPBOARD text, add the Ctrl+Alt+R desktop shortcut, and stream playback
through PipeWire.
