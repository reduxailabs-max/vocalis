# Stage 6c (Clarity) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Each voice engine paces itself with its own native speed control instead of one shared lossy stretch; Qwen3-TTS's own generation is tuned for steadier, faster output; a non-English-native Qwen3-TTS speaker choice is flagged; and Quick Reader's and Document Reader's text gets a basic normalization pass (URLs, currency, percentages, a couple of symbols) before it is spoken.

**Architecture:** Pacing moves out of the shared `ChunkedSpeaker`/`paced_audio()` path and into each engine itself: `KokoroTTS` and `PiperTTS` gain a `speed` constructor argument wired to their own native controls (Kokoro's `speed=`, Piper's `--length_scale`); `Qwen3TTS` (which has no native control) keeps the existing `librosa` stretch, but only ever applies it in a narrow, quality-safe range. `QuickReaderFeature`/`DocumentReaderFeature` and `cli.py`'s engine factories are updated to match (the same `str | Callable[[], str]` "read live, else fixed" pattern already used for `voice`/`speaker`). A `pydantic` validator and a pure text-normalization function are both additions with no architectural ripple.

**Tech Stack:** Python 3.13, `qwen-tts` (installed at `.venv/lib/python3.13/site-packages/qwen_tts`), `kokoro-onnx`, the local `piper` binary, `librosa`, `pydantic` v2.

**Spec:** `docs/superpowers/specs/2026-09-11-vocalis-design.md`, section "**6c — Clarity.**" (Stage 6 design). 6a merged at `b6bd8eb`, 6b merged at `fb43b0d`, both on `main`.

## Global Constraints

- Tests must stay at or above 85% coverage (enforced: `pyproject.toml:80`, `--cov-fail-under=85`).
- mypy strict must stay clean (zero errors) across the whole `src/` tree.
- Every function/method gets a concise, purpose-driven docstring (project `CLAUDE.md`).
- No hardcoded secrets; nothing here touches secrets, but the rule stands.
- Speed values stay inside `MIN_SPEED`/`MAX_SPEED` (`0.7`/`1.6`, `vocalis/audio/pace.py`) — every engine's own speed handling must clamp through `clamp_speed()`, never trust a raw float.
- No attribution/trailer lines in commit messages (session-standing rule).
- `old/` stays read-only; nothing in this plan touches it.
- A pacing or validation failure must never break a reading or crash config load — swallow and log, matching the existing `paced_audio()` convention.

## Review Focus

1. **Qwen3-TTS at a speed outside 1.0-1.2x.** The Settings slider still spans 0.7x-1.6x and a user has no reason to expect Qwen3-TTS to behave differently from Kokoro/Piper at the edges — the capped range must be a deliberate, tested choice (Task 1), not a silent no-op the user discovers by ear.
2. **Quick Reader's copied text, not just Document Reader's.** The spec's normalization item does not say "Document Reader only"; clipboard text just as often has a URL or a dollar amount, and Quick Reader currently has no normalization step at all (Task 6).
3. **Piper's `--length_scale` is inverse of speed.** A faster speed must produce a *smaller* `--length_scale`, not a larger one — an easy sign mix-up that would make "faster" sound slower (Task 3).
4. **A non-English-native speaker configured today or later** (hand-edited TOML, or a future Settings dropdown) should be flagged, not silently mismatched forever with no signal to the user (Task 5).
5. **A very short chunk handed to Qwen3-TTS.** `max_new_tokens` scaling must floor high enough that a short sentence is never cut off by an under-sized token budget (Task 1).

---

### Task 1: Qwen3-TTS generation tuning and its own capped speed stretch

**Files:**
- Modify: `src/vocalis/engines/qwen3_tts.py`
- Test: `tests/test_qwen3_tts.py`

**Interfaces:**
- Consumes: `vocalis.audio.pace.clamp_speed`, `vocalis.audio.pace.paced_audio` (both exist, unchanged).
- Produces: `Qwen3TTS.__init__(..., speed: float | Callable[[], float] = 1.0)`; `Qwen3TTS._current_speed() -> float`; module-level `_max_new_tokens_for(text: str) -> int` (used again by Task 4's cli.py wiring only indirectly — no other task calls it directly, but it must exist at this exact name for Task 1's own tests).

- [ ] **Step 1: Write the new/changed tests**

Open `tests/test_qwen3_tts.py`. First, replace `FakeQwenModel` so it accepts (and records) the new generation keyword arguments instead of rejecting them with a `TypeError`:

```python
class FakeQwenModel:
    def __init__(self, wavs: list[np.ndarray], sample_rate: int, raise_exc: Exception | None = None) -> None:
        self.wavs = wavs
        self.sample_rate = sample_rate
        self.raise_exc = raise_exc
        self.calls: list[dict[str, object]] = []

    def generate_custom_voice(
        self, text: str, language: str, speaker: str, **kwargs: object
    ) -> tuple[list[np.ndarray], int]:
        self.calls.append({"text": text, "language": language, "speaker": speaker, **kwargs})
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.wavs, self.sample_rate
```

Replace `test_load_then_synthesize_calls_generate_custom_voice_with_configured_speaker` (it asserted an exact `calls` dict that is now missing the new keys) with:

```python
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
    assert len(fake_model.calls) == 1
    call = fake_model.calls[0]
    assert call["text"] == "Hello world"
    assert call["language"] == "English"
    assert call["speaker"] == "Ryan"
    assert call["temperature"] == 0.65
    assert call["top_k"] == 20
    assert call["max_new_tokens"] == 200  # "Hello world" is 11 chars * 6 = 66, floored to 200
```

Add these new tests at the end of the file (they need `from vocalis.engines.qwen3_tts import Qwen3TTS, Qwen3TTSLoadError, _max_new_tokens_for` — add `_max_new_tokens_for` to the existing import line — and a small local sine helper):

```python
def _sine(seconds: float = 2.0, rate: int = 24000) -> np.ndarray:
    """A real, long-enough-to-stretch waveform (librosa needs >4096 samples)."""
    t = np.arange(int(rate * seconds)) / rate
    return (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)


def test_max_new_tokens_scales_with_text_length_within_bounds() -> None:
    assert _max_new_tokens_for("hi") == 200  # floor
    assert _max_new_tokens_for("x" * 100) == 600  # 100 * 6, well inside bounds
    assert _max_new_tokens_for("x" * 1000) == 2048  # ceiling


def test_speed_below_one_is_not_applied() -> None:
    """Qwen3-TTS's own stretch only ever speeds up -- never slows down."""
    audio = _sine()
    fake_model = FakeQwenModel(wavs=[audio], sample_rate=24000)
    engine = Qwen3TTS(model_dir=Path("/fake/model/dir"), speed=0.7)
    with patch("vocalis.engines.qwen3_tts.Qwen3TTSModel") as mock_model_cls:
        mock_model_cls.from_pretrained.return_value = fake_model
        engine.load()

    result, _ = engine.synthesize("hello there, this sentence needs to be long enough to stretch")

    assert len(result) == len(audio)


def test_speed_above_one_point_two_is_capped() -> None:
    """A requested speed above 1.2x is capped there, not applied in full."""
    audio = _sine()
    fake_model = FakeQwenModel(wavs=[audio], sample_rate=24000)
    engine = Qwen3TTS(model_dir=Path("/fake/model/dir"), speed=1.6)
    with patch("vocalis.engines.qwen3_tts.Qwen3TTSModel") as mock_model_cls:
        mock_model_cls.from_pretrained.return_value = fake_model
        engine.load()

    result, _ = engine.synthesize("hello there, this sentence needs to be long enough to stretch")

    assert abs(len(result) - len(audio) / 1.2) < 0.03 * len(audio)


def test_speed_between_one_and_one_point_two_is_applied_as_given() -> None:
    audio = _sine()
    fake_model = FakeQwenModel(wavs=[audio], sample_rate=24000)
    engine = Qwen3TTS(model_dir=Path("/fake/model/dir"), speed=1.1)
    with patch("vocalis.engines.qwen3_tts.Qwen3TTSModel") as mock_model_cls:
        mock_model_cls.from_pretrained.return_value = fake_model
        engine.load()

    result, _ = engine.synthesize("hello there, this sentence needs to be long enough to stretch")

    assert abs(len(result) - len(audio) / 1.1) < 0.03 * len(audio)


def test_speed_can_be_a_live_callable() -> None:
    """A callable speed is re-read on every synthesis call."""
    audio = _sine()
    fake_model = FakeQwenModel(wavs=[audio], sample_rate=24000)
    speeds = iter([1.1, 1.2])
    engine = Qwen3TTS(model_dir=Path("/fake/model/dir"), speed=lambda: next(speeds))
    with patch("vocalis.engines.qwen3_tts.Qwen3TTSModel") as mock_model_cls:
        mock_model_cls.from_pretrained.return_value = fake_model
        engine.load()

    first, _ = engine.synthesize("hello there, this sentence needs to be long enough to stretch")
    second, _ = engine.synthesize("hello there, this sentence needs to be long enough to stretch")

    assert abs(len(first) - len(audio) / 1.1) < 0.03 * len(audio)
    assert abs(len(second) - len(audio) / 1.2) < 0.03 * len(audio)
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `.venv/bin/python -m pytest tests/test_qwen3_tts.py -v --no-cov`
Expected: FAIL — `_max_new_tokens_for` does not exist yet (ImportError), and the generation-param assertions fail against the current `generate_custom_voice()` call (which passes none of `temperature`/`top_k`/`max_new_tokens`).

- [ ] **Step 3: Implement**

In `src/vocalis/engines/qwen3_tts.py`, add the import and module-level constants right after the existing imports (keep the `os.environ.setdefault(...)` line and its comment exactly where they are, at the top):

```python
from vocalis.audio.pace import clamp_speed, paced_audio
```

```python
_TEMPERATURE = 0.65
_TOP_K = 20
_TOKENS_PER_CHAR = 6
_MIN_NEW_TOKENS = 200
_MAX_NEW_TOKENS = 2048
_STRETCH_MIN_SPEED = 1.0
_STRETCH_MAX_SPEED = 1.2


def _max_new_tokens_for(text: str) -> int:
    """How many codec tokens to let Qwen3-TTS generate for `text`.

    Scaled to the text's length instead of always requesting the
    library's full default budget (2048): a short chunk then generates
    faster and is less likely to wander or repeat near the end. A floor
    keeps very short text from being cut off; a ceiling never asks for
    more than the library's own tuned default.
    """
    return max(_MIN_NEW_TOKENS, min(_MAX_NEW_TOKENS, len(text) * _TOKENS_PER_CHAR))
```

Update the constructor to accept `speed`:

```python
    def __init__(
        self,
        model_dir: Path,
        speaker: str | Callable[[], str] = "Ryan",
        language: str = "English",
        use_gpu: bool = True,
        speed: float | Callable[[], float] = 1.0,
    ) -> None:
        """Create a Qwen3TTS engine (does not load the model yet).

        Args:
            model_dir: Local directory the CustomVoice model files were
                downloaded into (see this task's one-time download step).
            speaker: Preset voice name (one of the 9 CustomVoice presets),
                or a function returning it, re-read on every synthesis so a
                change in Settings applies to the next reading.
            language: Spoken language passed to generate_custom_voice().
            use_gpu: If True, load onto CUDA (the only supported device
                for Qwen3-TTS); False is only useful in tests that never
                call load().
            speed: Desired speaking pace, or a function returning it,
                re-read on every synthesis. Qwen3-TTS has no native speed
                control, so this is approximated with a mild (1.0-1.2x)
                time-stretch of the finished audio -- see
                _capped_speed(): a bigger stretch, in either direction,
                audibly hurts this voice, so anything outside that range
                is simply played at the nearest end of it instead.
        """
        self._model_dir = model_dir
        self._speaker = speaker
        self._language = language
        self._use_gpu = use_gpu
        self._speed = speed
        self._model: Qwen3TTSModel | None = None
```

Add the two speed helpers right after `_current_speaker()`:

```python
    def _current_speed(self) -> float:
        """The requested speaking pace right now (re-read on every synthesis)."""
        return self._speed() if callable(self._speed) else self._speed

    def _capped_speed(self) -> float:
        """The speed actually applied: clamped to the supported range, then
        narrowed to 1.0-1.2x (see __init__'s `speed` docstring)."""
        speed = clamp_speed(self._current_speed())
        return min(max(speed, _STRETCH_MIN_SPEED), _STRETCH_MAX_SPEED)
```

Update `synthesize()`:

```python
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
            text=text,
            language=self._language,
            speaker=self._current_speaker(),
            temperature=_TEMPERATURE,
            top_k=_TOP_K,
            max_new_tokens=_max_new_tokens_for(text),
        )
        audio = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
        sample_rate = int(sample_rate)
        return paced_audio(audio, sample_rate, self._capped_speed), sample_rate
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `.venv/bin/python -m pytest tests/test_qwen3_tts.py -v --no-cov`
Expected: PASS, all tests.

- [ ] **Step 5: Commit**

```bash
git add src/vocalis/engines/qwen3_tts.py tests/test_qwen3_tts.py
git commit -m "feat: tune Qwen3-TTS generation and cap its own speed stretch"
```

---

### Task 2: Kokoro native speed control

**Files:**
- Modify: `src/vocalis/engines/kokoro_tts.py`
- Test: `tests/test_kokoro_tts.py`

**Interfaces:**
- Consumes: `vocalis.audio.pace.clamp_speed`.
- Produces: `KokoroTTS.__init__(..., speed: float | Callable[[], float] = 1.0)`; `KokoroTTS._current_speed() -> float`.

- [ ] **Step 1: Write the new tests**

Append to `tests/test_kokoro_tts.py` (its existing `_engine(tmp_path, kokoro=None, **kwargs)` helper already forwards `**kwargs` straight into `KokoroTTS(...)`, and `_FakeKokoro.create()` already records `(text, voice, speed, lang)` tuples — no helper changes needed):

```python
from vocalis.audio.pace import MAX_SPEED


def test_default_speed_is_neutral(tmp_path: Path) -> None:
    """Without a configured speed, Kokoro is asked for its natural pace."""
    engine, _, fake = _engine(tmp_path)
    engine.load()
    engine.synthesize("hello")
    assert fake.calls[0][2] == 1.0


def test_synthesize_passes_the_configured_speed(tmp_path: Path) -> None:
    engine, _, fake = _engine(tmp_path, speed=1.3)
    engine.load()
    engine.synthesize("hello")
    assert fake.calls[0][2] == 1.3


def test_speed_is_clamped_to_the_supported_range(tmp_path: Path) -> None:
    engine, _, fake = _engine(tmp_path, speed=9.0)
    engine.load()
    engine.synthesize("hello")
    assert fake.calls[0][2] == MAX_SPEED


def test_speed_can_be_a_live_callable(tmp_path: Path) -> None:
    """A callable speed is re-read on every synthesis call."""
    speeds = iter([0.8, 1.4])
    engine, _, fake = _engine(tmp_path, speed=lambda: next(speeds))
    engine.load()
    engine.synthesize("one")
    engine.synthesize("two")
    assert [call[2] for call in fake.calls] == [0.8, 1.4]
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `.venv/bin/python -m pytest tests/test_kokoro_tts.py -v --no-cov`
Expected: FAIL — `KokoroTTS.__init__()` has no `speed` keyword, and `kokoro.create()` is always called with the hardcoded `speed=1.0`.

- [ ] **Step 3: Implement**

In `src/vocalis/engines/kokoro_tts.py`, add the import near the top (alongside the existing `numpy`/`numpy.typing` imports):

```python
from vocalis.audio.pace import clamp_speed
```

Update the constructor:

```python
    def __init__(
        self,
        model_path: Path,
        voices_path: Path,
        voice: "str | Callable[[], str]" = "af_heart",
        use_gpu: "bool | Callable[[], bool]" = False,
        speed: "float | Callable[[], float]" = 1.0,
        session_builder: Callable[[Path, bool], Any] = _default_session_builder,
        kokoro_factory: Callable[[Any, Path], Any] = _default_kokoro_factory,
    ) -> None:
        """Create a KokoroTTS engine (does not load the model yet).

        Args:
            model_path: Path to kokoro-v1.0.onnx.
            voices_path: Path to voices-v1.0.bin.
            voice: Voice name, or a function returning it (re-read on every
                synthesis so a Settings change applies to the next reading).
            use_gpu: Whether to run on the graphics card, or a function
                answering that at load time.
            speed: Speaking pace, or a function returning it, re-read on
                every synthesis and clamped to the supported range.
                Passed straight to Kokoro's own `speed=` parameter.
            session_builder: Builds the ONNX session; injectable for tests.
            kokoro_factory: Wraps a session into a synthesizer; injectable for tests.
        """
        self._model_path = model_path
        self._voices_path = voices_path
        self._voice = voice
        self._use_gpu = use_gpu
        self._speed = speed
        self._session_builder = session_builder
        self._kokoro_factory = kokoro_factory
        self._kokoro: Any | None = None
```

Add the speed helper right after `_current_voice()`:

```python
    def _current_speed(self) -> float:
        """The speaking pace to use right now, clamped to the supported range."""
        raw = self._speed() if callable(self._speed) else self._speed
        return clamp_speed(raw)
```

In `synthesize()`, read the speed once and pass it instead of the hardcoded `1.0`:

```python
        try:
            voice = self._current_voice()
            lang = _language_for(voice)
            speed = self._current_speed()
            clips: list[npt.NDArray[np.float32]] = []
            sample_rate = 0
            for piece in _pieces_for_kokoro(text):
                samples, rate = kokoro.create(piece, voice=voice, speed=speed, lang=lang)
```

(The rest of `synthesize()`'s body is unchanged.)

- [ ] **Step 4: Run the tests to see them pass**

Run: `.venv/bin/python -m pytest tests/test_kokoro_tts.py -v --no-cov`
Expected: PASS, all tests.

- [ ] **Step 5: Commit**

```bash
git add src/vocalis/engines/kokoro_tts.py tests/test_kokoro_tts.py
git commit -m "feat: give Kokoro a native speed control"
```

---

### Task 3: Piper native `--length_scale` speed control

**Files:**
- Modify: `src/vocalis/engines/piper_tts.py`
- Test: `tests/test_piper_tts.py`

**Interfaces:**
- Consumes: `vocalis.audio.pace.clamp_speed`.
- Produces: `PiperTTS.__init__(..., speed: float | Callable[[], float] = 1.0)`; `PiperTTS._current_speed() -> float`.

- [ ] **Step 1: Write the new/changed tests**

In `tests/test_piper_tts.py`, update the existing `test_synthesize_converts_raw_pcm_to_float32`'s argv assertion (Piper's CLI now always gets a `--length_scale` argument, including at the default speed, where `length_scale` is `1.0`):

```python
    argv, input_bytes = runner.calls[0]
    assert argv == [str(binary), "--model", str(model), "--length_scale", "1.0000", "--output_raw"]
    assert input_bytes == b"hello world"
```

Add these new tests at the end of the file:

```python
def test_synthesize_passes_length_scale_for_a_faster_speed(tmp_path: Path) -> None:
    """A speed above 1.0 is the INVERSE as --length_scale (Piper's convention: smaller = faster)."""
    binary = tmp_path / "piper"
    binary.write_text("")
    model = tmp_path / "voice.onnx"
    model.write_text("")
    runner = FakeCommandRunner(returncode=0, stdout=_pcm_bytes([0]))
    engine = PiperTTS(binary_path=binary, model_path=model, command_runner=runner, speed=1.25)
    engine.load()

    engine.synthesize("hello")

    argv, _ = runner.calls[0]
    assert argv == [str(binary), "--model", str(model), "--length_scale", "0.8000", "--output_raw"]


def test_speed_is_clamped_to_the_supported_range(tmp_path: Path) -> None:
    binary = tmp_path / "piper"
    binary.write_text("")
    model = tmp_path / "voice.onnx"
    model.write_text("")
    runner = FakeCommandRunner(returncode=0, stdout=_pcm_bytes([0]))
    engine = PiperTTS(binary_path=binary, model_path=model, command_runner=runner, speed=9.0)
    engine.load()

    engine.synthesize("hello")

    argv, _ = runner.calls[0]
    assert argv == [str(binary), "--model", str(model), "--length_scale", "0.6250", "--output_raw"]


def test_speed_can_be_a_live_callable(tmp_path: Path) -> None:
    """A callable speed is re-read on every synthesis call."""
    binary = tmp_path / "piper"
    binary.write_text("")
    model = tmp_path / "voice.onnx"
    model.write_text("")
    runner = FakeCommandRunner(returncode=0, stdout=_pcm_bytes([0]))
    speeds = iter([1.0, 1.25])
    engine = PiperTTS(binary_path=binary, model_path=model, command_runner=runner, speed=lambda: next(speeds))
    engine.load()

    engine.synthesize("one")
    engine.synthesize("two")

    assert runner.calls[0][0][-2] == "1.0000"
    assert runner.calls[1][0][-2] == "0.8000"
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `.venv/bin/python -m pytest tests/test_piper_tts.py -v --no-cov`
Expected: FAIL — `PiperTTS.__init__()` has no `speed` keyword, and the existing argv assertion no longer matches (no `--length_scale` is passed at all yet).

- [ ] **Step 3: Implement**

In `src/vocalis/engines/piper_tts.py`, add `Callable` to the existing `from typing import Protocol` line (making it `from typing import Callable, Protocol`), and add this import near the top:

```python
from vocalis.audio.pace import clamp_speed
```

Update the constructor:

```python
    def __init__(
        self,
        binary_path: Path,
        model_path: Path,
        sample_rate: int = 22050,
        speed: float | Callable[[], float] = 1.0,
        command_runner: CommandRunner | None = None,
    ) -> None:
        """Create a PiperTTS engine (does not check files yet).

        Args:
            binary_path: Path to the piper executable.
            model_path: Path to the .onnx voice model file.
            sample_rate: Sample rate of the voice model's audio output, in
                Hz (22050 for en_US-lessac-medium, per its .onnx.json).
            speed: Speaking pace, or a function returning it, re-read on
                every synthesis and clamped to the supported range.
                Converted to Piper's own `--length_scale` argument
                (`1 / speed`: a bigger length means slower speech).
            command_runner: Callable(argv, input_bytes=None) -> CompletedProcess,
                used to run piper. Defaults to subprocess.run with a
                30-second timeout. Injectable for tests.
        """
        self._binary_path = binary_path
        self._model_path = model_path
        self._sample_rate = sample_rate
        self._speed = speed
        self._run: CommandRunner = command_runner or _default_command_runner
        self._loaded = False
```

Add the speed helper right after `is_loaded`:

```python
    def _current_speed(self) -> float:
        """The speaking pace to use right now, clamped to the supported range."""
        raw = self._speed() if callable(self._speed) else self._speed
        return clamp_speed(raw)
```

Update `synthesize()` to add `--length_scale`:

```python
        if not self._loaded:
            raise RuntimeError("PiperTTS.synthesize() called before load()")
        length_scale = 1.0 / self._current_speed()
        try:
            result = self._run(
                [
                    str(self._binary_path),
                    "--model",
                    str(self._model_path),
                    "--length_scale",
                    f"{length_scale:.4f}",
                    "--output_raw",
                ],
                input_bytes=text.encode("utf-8"),
            )
```

(The rest of `synthesize()` is unchanged.)

- [ ] **Step 4: Run the tests to see them pass**

Run: `.venv/bin/python -m pytest tests/test_piper_tts.py -v --no-cov`
Expected: PASS, all tests.

- [ ] **Step 5: Commit**

```bash
git add src/vocalis/engines/piper_tts.py tests/test_piper_tts.py
git commit -m "feat: give Piper a native speed control via --length_scale"
```

---

### Task 4: Move pacing out of ChunkedSpeaker/the readers and into cli.py's engine factories

**Files:**
- Modify: `src/vocalis/features/chunked_speaker.py`
- Modify: `src/vocalis/features/quick_reader.py`
- Modify: `src/vocalis/features/doc_reader.py`
- Modify: `src/vocalis/cli.py`
- Test: `tests/test_chunked_speaker.py`, `tests/test_quick_reader.py`, `tests/test_doc_reader.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `Qwen3TTS(..., speed=...)`, `KokoroTTS(..., speed=...)`, `PiperTTS(..., speed=...)` from Tasks 1-3.
- Produces: `ChunkedSpeaker.__init__()` loses `speed_provider`; `QuickReaderFeature.__init__()` and `DocumentReaderFeature.__init__()` lose `speed_provider`. `cli.py`'s `_make_qwen_factory()` and `_make_kokoro_factory()` gain a live `speed`; both `PiperTTS(...)` construction sites in `build_quick_reader()`/`build_doc_reader()` gain `speed=`.

- [ ] **Step 1: Remove `speed_provider` from `ChunkedSpeaker`**

In `src/vocalis/features/chunked_speaker.py`:
- Remove the import `from vocalis.audio.pace import paced_audio`.
- Remove the `speed_provider: Callable[[], float] | None = None,` constructor parameter and its `Args:` docstring line ("speed_provider: Current speaking pace, read once per chunk; None = no pacing.").
- Remove `self._speed_provider = speed_provider`.
- In `_speak()`, delete the line `audio = paced_audio(audio, sample_rate, self._speed_provider)` (the `audio, sample_rate = source.synthesize(text)` line right above it is unchanged; `audio` now flows straight from `synthesize()` into the `if not opened: ...` block below).

- [ ] **Step 2: Remove `speed_provider` from `QuickReaderFeature`**

In `src/vocalis/features/quick_reader.py`:
- Remove the `speed_provider: Callable[[], float] | None = None,` constructor parameter and its `Args:` docstring block (the "speed_provider: Called (with no arguments)..." paragraph).
- Remove `self._speed_provider = speed_provider`.
- In `_read_aloud()`, remove the line `speed_provider=self._speed_provider,` from the `ChunkedSpeaker(...)` call (it becomes just `player=self._player, source_factory=self._new_source,`).

- [ ] **Step 3: Remove `speed_provider` from `DocumentReaderFeature`**

In `src/vocalis/features/doc_reader.py`:
- Remove the `speed_provider: Callable[[], float] | None = None,` constructor parameter and its `Args:` docstring block (the "speed_provider: Called (with no arguments)..." paragraph).
- Remove `self._speed_provider = speed_provider`.
- In `_read_worker()`, remove the line `speed_provider=self._speed_provider,` from the `ChunkedSpeaker(...)` call.

- [ ] **Step 4: Wire live speed into cli.py's engine factories**

In `src/vocalis/cli.py`, update `_make_qwen_factory`:

```python
def _make_qwen_factory(
    config: VocalisConfig, settings: LiveSettings | None
) -> Callable[[], Qwen3TTS]:
    """A factory building Qwen3TTS; the voice and speed are read live when settings are given."""
    qwen_model_dir = Path(config.engines.qwen3_tts_model_dir)
    speaker: "str | Callable[[], str]" = (
        settings.voice if settings is not None else config.engines.qwen3_tts_speaker
    )
    speed: "float | Callable[[], float]" = settings.speaking_speed if settings is not None else config.speaking_speed

    def primary_engine_factory() -> Qwen3TTS:
        """Build the primary text-to-speech engine."""
        return Qwen3TTS(model_dir=qwen_model_dir, speaker=speaker, speed=speed)

    return primary_engine_factory
```

Update `_make_kokoro_factory`:

```python
def _make_kokoro_factory(
    config: VocalisConfig, settings: LiveSettings | None, selector: EngineSelector
) -> Callable[[], KokoroTTS]:
    """A factory building KokoroTTS; voice and speed are read live, graphics-card use is decided at load."""
    model_path = Path(config.engines.kokoro_model_path)
    voices_path = Path(config.engines.kokoro_voices_path)
    voice: "str | Callable[[], str]" = settings.kokoro_voice if settings is not None else config.engines.kokoro_voice
    speed: "float | Callable[[], float]" = settings.speaking_speed if settings is not None else config.speaking_speed

    def kokoro_factory() -> KokoroTTS:
        """Build the light voice engine."""
        return KokoroTTS(
            model_path=model_path, voices_path=voices_path, voice=voice, use_gpu=selector.kokoro_use_gpu, speed=speed
        )

    return kokoro_factory
```

In `build_quick_reader()` and `build_doc_reader()`, update **both** `fallback_engine = PiperTTS(...)` construction sites (they are currently identical in each function) to:

```python
    fallback_engine = PiperTTS(
        binary_path=Path(config.engines.piper_binary_path),
        model_path=Path(config.engines.piper_voice_model_path),
        speed=settings.speaking_speed if settings is not None else config.speaking_speed,
    )
```

In the same two functions, remove the line `speed_provider=settings.speaking_speed if settings is not None else None,` from the `QuickReaderFeature(...)` / `DocumentReaderFeature(...)` construction calls.

- [ ] **Step 5: Update `tests/test_chunked_speaker.py`**

Delete the test `test_the_speaking_pace_is_applied_to_each_chunk` (it tested the now-removed `speed_provider` path end to end; the per-engine speed behavior is covered by Tasks 1-3's own tests). Delete the blank line immediately above it too, so the file ends cleanly on the previous test.

- [ ] **Step 6: Update `tests/test_quick_reader.py`**

Remove the `speed_provider: "Callable[[], float] | None" = None,` parameter from `make_feature()`'s signature, and remove the `speed_provider=speed_provider,` line from its `QuickReaderFeature(...)` call.

Delete the `_played_length()` helper and these four tests entirely (they tested the removed `speed_provider` path, and `_played_length()` is used by nothing else in the file): `test_no_speed_provider_leaves_audio_unchanged`, `test_speed_provider_returning_one_leaves_audio_unchanged`, `test_speed_provider_shortens_the_audio_handed_to_the_player`, `test_a_raising_speed_provider_still_plays_the_original_audio`. This whole block runs from the `_played_length` helper through the end of the last test, right before the `_build_with_selector` helper that follows it.

- [ ] **Step 7: Update `tests/test_doc_reader.py`**

Remove the `speed_provider: "Callable[[], float] | None" = None,` parameter from `_make_feature()`'s signature, and remove the `speed_provider=speed_provider,` line from its `DocumentReaderFeature(...)` call.

Delete the `_two_chunk_text()`, `_speed_feature()`, and `_non_silent_count()` helpers and these three tests entirely — all are used only by this block: `test_no_speed_provider_leaves_chunk_audio_unchanged`, `test_speed_provider_is_read_per_chunk_and_produces_two_lengths`, `test_a_raising_speed_provider_still_plays_the_original_chunk`. This whole block runs from the `_two_chunk_text` helper through the end of the last test, right before the `_make_selector_feature` helper that follows it.

- [ ] **Step 8: Update `tests/test_cli.py`**

Replace the three tests `test_build_quick_reader_speed_provider_follows_live_settings`, `test_build_doc_reader_speed_provider_follows_live_settings`, and `test_build_quick_reader_without_settings_has_no_speed_provider` with:

```python
def test_primary_engine_reads_the_live_speed(tmp_path: Path, qtbot) -> None:  # type: ignore[no-untyped-def]
    settings = _live_settings(tmp_path)
    manager = ModelManager()
    with patch("vocalis.cli.list_hotkey_capable_devices", return_value=[]):
        quick, _ = build_quick_reader(
            config=settings.config, model_manager=manager,
            reading_coordinator=ReadingCoordinator(), settings=settings,
        )
    doc, popup, _ = build_doc_reader(
        config=settings.config, model_manager=manager,
        reading_coordinator=ReadingCoordinator(), settings=settings,
    )
    qtbot.addWidget(popup)
    settings.update(settings.config.model_copy(update={"speaking_speed": 1.4}))
    for feature in (quick, doc):
        engine = feature._primary_engine_factory()  # noqa: SLF001
        assert isinstance(engine, Qwen3TTS)
        assert engine._current_speed() == 1.4  # noqa: SLF001


def test_kokoro_factory_reads_the_live_speed(tmp_path: Path) -> None:
    settings = _live_settings(tmp_path)
    with patch("vocalis.cli.list_hotkey_capable_devices", return_value=[]):
        feature, _ = build_quick_reader(
            config=settings.config, model_manager=ModelManager(),
            reading_coordinator=ReadingCoordinator(), settings=settings,
        )
    kokoro = (feature._engine_factories or {})["kokoro"]()  # noqa: SLF001
    assert isinstance(kokoro, KokoroTTS)
    assert kokoro._current_speed() == settings.speaking_speed()  # noqa: SLF001
    settings.update(settings.config.model_copy(update={"speaking_speed": 1.3}))
    assert kokoro._current_speed() == 1.3  # noqa: SLF001


def test_fallback_engine_reads_the_live_speed(tmp_path: Path) -> None:
    settings = _live_settings(tmp_path)
    with patch("vocalis.cli.list_hotkey_capable_devices", return_value=[]):
        feature, _ = build_quick_reader(
            config=settings.config, model_manager=ModelManager(),
            reading_coordinator=ReadingCoordinator(), settings=settings,
        )
    piper = feature._fallback_engine  # noqa: SLF001
    assert piper._current_speed() == settings.speaking_speed()  # noqa: SLF001
    settings.update(settings.config.model_copy(update={"speaking_speed": 0.9}))
    assert piper._current_speed() == 0.9  # noqa: SLF001


def test_build_quick_reader_without_settings_uses_the_configured_speed() -> None:
    with patch("vocalis.cli.list_hotkey_capable_devices", return_value=[]):
        feature, _ = build_quick_reader(
            config=VocalisConfig(), model_manager=ModelManager(), reading_coordinator=ReadingCoordinator(),
        )
    engine = feature._primary_engine_factory()  # noqa: SLF001
    assert engine._current_speed() == VocalisConfig().speaking_speed  # noqa: SLF001
```

This needs `KokoroTTS` imported in `tests/test_cli.py`, which does not import it yet: add `from vocalis.engines.kokoro_tts import KokoroTTS  # noqa: E402` right after the existing `from vocalis.engines.qwen3_tts import Qwen3TTS  # noqa: E402` line (around line 457).

- [ ] **Step 9: Run every touched test file**

Run: `.venv/bin/python -m pytest tests/test_chunked_speaker.py tests/test_quick_reader.py tests/test_doc_reader.py tests/test_cli.py -v --no-cov`
Expected: PASS, all tests (some failures are expected before this step if Steps 1-4's implementation and Steps 5-8's test edits are applied together, which they are — this is the one run that should be green).

- [ ] **Step 10: Commit**

```bash
git add src/vocalis/features/chunked_speaker.py src/vocalis/features/quick_reader.py src/vocalis/features/doc_reader.py src/vocalis/cli.py tests/test_chunked_speaker.py tests/test_quick_reader.py tests/test_doc_reader.py tests/test_cli.py
git commit -m "refactor: move speech pacing from ChunkedSpeaker into each engine"
```

---

### Task 5: Warn on a non-English-native Qwen3-TTS speaker

**Files:**
- Modify: `src/vocalis/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: none new.
- Produces: `vocalis.config.ENGLISH_NATIVE_QWEN_SPEAKERS: frozenset[str]`; a `pydantic` `model_validator` on `EnginesConfig` (no new public method — it runs automatically on every construction/validation of `EnginesConfig`).

- [ ] **Step 1: Write the new tests**

Add to `tests/test_config.py` (add `import logging` to its imports):

```python
def test_default_speaker_is_english_native() -> None:
    """Ryan (English) is the configured default; this pins that choice."""
    assert EnginesConfig().qwen3_tts_speaker == "Ryan"


def test_a_non_english_native_speaker_logs_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="vocalis"):
        EnginesConfig(qwen3_tts_speaker="Serena")
    assert any("Serena" in record.message for record in caplog.records)


def test_english_native_speakers_log_no_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="vocalis"):
        EnginesConfig(qwen3_tts_speaker="Aiden")
    assert not caplog.records


def test_speaker_check_is_case_insensitive(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="vocalis"):
        EnginesConfig(qwen3_tts_speaker="ryan")
    assert not caplog.records
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `.venv/bin/python -m pytest tests/test_config.py -v --no-cov`
Expected: FAIL — `test_a_non_english_native_speaker_logs_a_warning` finds no warning logged (nothing validates `qwen3_tts_speaker` today).

- [ ] **Step 3: Implement**

In `src/vocalis/config.py`, add `import logging` to the imports (alongside the existing `os`, `tempfile`, `tomllib` imports), add `model_validator` to the `pydantic` import line, and add a module logger plus the speaker set right after the module docstring's imports, before `class ConfigError`:

```python
import logging
...
from pydantic import BaseModel, Field, StringConstraints, ValidationError, model_validator
...

logger = logging.getLogger("vocalis")

ENGLISH_NATIVE_QWEN_SPEAKERS = frozenset({"ryan", "aiden"})
```

Add the validator to `EnginesConfig`, right after its field declarations (after `kokoro_voice: str = "af_heart"`):

```python
    @model_validator(mode="after")
    def _warn_if_speaker_is_not_english_native(self) -> "EnginesConfig":
        """Qwen3-TTS always speaks as English (its spoken language is not
        yet separately configurable), but most of its nine preset voices
        are native to Chinese, Japanese or Korean and sound noticeably
        worse when forced to speak English. This never blocks the
        config -- it only logs, so a deliberate choice still works.
        """
        if self.qwen3_tts_speaker.strip().lower() not in ENGLISH_NATIVE_QWEN_SPEAKERS:
            logger.warning(
                "Qwen3-TTS speaker %r is not English-native (English-native choices: Ryan, Aiden); "
                "it will still be asked to speak English, which may sound worse than its own language.",
                self.qwen3_tts_speaker,
            )
        return self
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `.venv/bin/python -m pytest tests/test_config.py -v --no-cov`
Expected: PASS, all tests.

- [ ] **Step 5: Commit**

```bash
git add src/vocalis/config.py tests/test_config.py
git commit -m "feat: warn when the Qwen3-TTS speaker is not English-native"
```

---

### Task 6: Basic text normalization (numbers, URLs, symbols)

**Files:**
- Modify: `src/vocalis/text/normalizer.py`
- Modify: `src/vocalis/features/quick_reader.py`
- Test: `tests/test_normalizer.py`, `tests/test_quick_reader.py`

**Interfaces:**
- Consumes: none new.
- Produces: `vocalis.text.normalizer.normalize_text(text: str) -> str` (new, public); `normalize_markdown()`'s return value now also passes through `normalize_text()`.

- [ ] **Step 1: Write the new tests**

Add to `tests/test_normalizer.py` (add `normalize_text` to its existing `from vocalis.text.normalizer import normalize_markdown` import line):

```python
def test_normalize_text_speaks_urls() -> None:
    assert normalize_text("Visit https://example.com/docs for info") == "Visit example dot com slash docs for info"


def test_normalize_text_speaks_www_urls() -> None:
    assert normalize_text("See www.example.com") == "See www dot example dot com"


def test_normalize_text_strips_a_trailing_slash() -> None:
    assert normalize_text("Go to https://example.com/") == "Go to example dot com"


def test_normalize_text_speaks_currency() -> None:
    assert normalize_text("It costs $42") == "It costs 42 dollars"


def test_normalize_text_speaks_percent() -> None:
    assert normalize_text("Up 50% today") == "Up 50 percent today"


def test_normalize_text_expands_ampersand_and_at() -> None:
    assert normalize_text("Rock & Roll, email me @home") == "Rock and Roll, email me at home"


def test_normalize_text_leaves_plain_numbers_alone() -> None:
    assert normalize_text("I have 42 apples and 3.14 pie") == "I have 42 apples and 3.14 pie"


def test_normalize_text_handles_empty_string() -> None:
    assert normalize_text("") == ""


def test_normalize_markdown_also_normalizes_urls_and_symbols() -> None:
    assert normalize_markdown("Cost is $5 & visit https://example.com") == "Cost is 5 dollars and visit example dot com"
```

Add to `tests/test_quick_reader.py` (needs no new imports — `np` is already imported):

```python
def test_clipboard_text_is_normalized_before_being_spoken() -> None:
    """A URL or symbol in the copied text is spoken, not read as symbols;
    the desktop notification still shows the original, un-normalized text."""
    primary = FakeSynthesizer(audio=np.zeros(100, dtype=np.float32))
    fallback = FakeSynthesizer(audio=np.zeros(1, dtype=np.float32))
    feature, _, _, _, notifier, _ = make_feature(
        primary, fallback, clipboard_text="Cost is $5 & visit https://example.com"
    )

    feature.on_trigger()
    wait_until(lambda: any(msg == "Done" for _, msg in notifier.messages))

    assert primary.calls == ["Cost is 5 dollars and visit example dot com"]
    assert ("Vocalis — Quick Reader", "Cost is $5 & visit https://example.com") in notifier.messages
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `.venv/bin/python -m pytest tests/test_normalizer.py tests/test_quick_reader.py -v --no-cov`
Expected: FAIL — `normalize_text` does not exist (ImportError in `test_normalizer.py`); `test_clipboard_text_is_normalized_before_being_spoken` fails because the clipboard text is currently spoken verbatim.

- [ ] **Step 3: Implement**

In `src/vocalis/text/normalizer.py`, add these constants and functions right after the existing regex constants (after `_EXCESS_BLANK_LINES_RE = re.compile(r"\n{3,}")`), before `def normalize_markdown`:

```python
_URL_RE = re.compile(r"\bhttps?://[^\s,;:!?)]+|\bwww\.[^\s,;:!?)]+", re.IGNORECASE)
_CURRENCY_RE = re.compile(r"\$(\d+(?:\.\d+)?)")
_PERCENT_RE = re.compile(r"(\d)%")


def _speak_url(match: "re.Match[str]") -> str:
    """Turn one matched URL into a short, spoken-friendly phrase."""
    url = re.sub(r"^https?://", "", match.group(0), flags=re.IGNORECASE)
    url = url.replace(".", " dot ").replace("/", " slash ")
    url = re.sub(r"\s+", " ", url).strip()
    return re.sub(r"\s*slash$", "", url)


def normalize_text(text: str) -> str:
    """Make raw text easier for a TTS engine to pronounce.

    Deliberately basic, and independent of normalize_markdown (Quick
    Reader's clipboard text is never markdown): speaks out URLs
    ("https://example.com/docs" -> "example dot com slash docs"), dollar
    amounts ("$42" -> "42 dollars") and percentages ("50%" -> "50
    percent"), and expands the two symbols a voice engine otherwise
    either misreads or skips (& and @). Plain numbers are left alone --
    the voice engines already read digit sequences correctly, so
    over-normalizing them would do more harm than good.

    Args:
        text: Raw text, already markdown-free (or never markdown).

    Returns:
        The same text with the substitutions above applied. Never raises.
    """
    result = _URL_RE.sub(_speak_url, text)
    result = _CURRENCY_RE.sub(r"\1 dollars", result)
    result = _PERCENT_RE.sub(r"\1 percent", result)
    result = result.replace("&", " and ").replace("@", " at ")
    return re.sub(r"[ \t]+", " ", result).strip()
```

Change `normalize_markdown()`'s final two lines from:

```python
    for i, code_text in enumerate(inline_codes):
        result = result.replace(f"\x00CODE{i}\x00", code_text)

    return result.strip()
```

to:

```python
    for i, code_text in enumerate(inline_codes):
        result = result.replace(f"\x00CODE{i}\x00", code_text)

    return normalize_text(result)
```

In `src/vocalis/features/quick_reader.py`, add the import (alongside the existing `from vocalis.text.chunker import chunk_text` line):

```python
from vocalis.text.normalizer import normalize_text
```

In `_read_aloud()`, change:

```python
        outcome = speaker.run(chunk_text(text))
```

to:

```python
        outcome = speaker.run(chunk_text(normalize_text(text)))
```

(`text_notice` just above it keeps notifying the original, un-normalized `text` — do not change that line.)

- [ ] **Step 4: Run the tests to see them pass**

Run: `.venv/bin/python -m pytest tests/test_normalizer.py tests/test_quick_reader.py -v --no-cov`
Expected: PASS, all tests.

- [ ] **Step 5: Commit**

```bash
git add src/vocalis/text/normalizer.py src/vocalis/features/quick_reader.py tests/test_normalizer.py tests/test_quick_reader.py
git commit -m "feat: speak URLs, currency, percentages and a couple of symbols"
```

---

### Task 7: Docs and changelog

**Files:**
- Modify: `docs/architecture.md`
- Modify: `docs/algorithms.md`
- Modify: `docs/workflow.md`
- Modify: `docs/file-purpose.md`
- Modify: `docs/folder-structure.md`
- Modify: `docs/user-docs.md`
- Create: `.claude/<today's HHMM>_CHANGELOG.md`

**Interfaces:** None (docs only).

- [ ] **Step 1: `docs/algorithms.md`**

Replace this paragraph (currently at the end of the "Changing speaking pace..." section):

```
`paced_audio(audio, sample_rate, provider)` is the shared entry point both
readers call: with `provider` is `None` it returns `audio` unchanged; otherwise
it calls `provider()` for the current speed and applies it, catching any
exception from either call, logging it as a warning, and returning the
original `audio` — a pacing failure must never break a reading. Since Stage 6b both readers call it through `ChunkedSpeaker`, once per chunk,
before the chunk is pushed into the stream; a pause rewinds the heard clip inside
`SpeechStream`, so a resume replays the same already-paced audio.
```

with:

```
`paced_audio(audio, sample_rate, provider)` is the shared entry point: with
`provider` is `None` it returns `audio` unchanged; otherwise it calls
`provider()` for the current speed and applies it, catching any exception
from either call, logging it as a warning, and returning the original
`audio` — a pacing failure must never break a reading. Since Stage 6c it has
exactly one caller, `Qwen3TTS.synthesize()`, capped to a mild 1.0-1.2x (that
engine has no native speed control of its own, and a bigger stretch audibly
hurts its voice). Kokoro and Piper have real speed controls and never call
it: Kokoro passes the speed straight to `kokoro.create(..., speed=...)`, and
Piper converts it to its `--length_scale` argument (`1 / speed`). A pause
still rewinds the heard clip inside `SpeechStream`, so a resume replays the
same already-paced audio.
```

- [ ] **Step 2: `docs/workflow.md`**

Replace (Quick Reader flow):

```
       for each chunk: EngineChain.synthesize (voice held loaded, falls down the
       plan on failure) -> paced_audio -> push into the SpeechStream
```

with:

```
       for each chunk: EngineChain.synthesize (voice held loaded, falls down the
       plan on failure; each engine paces its own output -- see "Reading-time
       pacing flow" below) -> push into the SpeechStream
```

Replace (Document Reader flow):

```
       producer: for each chunk, EngineChain.synthesize (voice held loaded for
                 the whole reading; falls down Qwen3-TTS -> Kokoro -> Piper, or
                 the user's choice, notifying once) -> paced_audio -> push
```

with:

```
       producer: for each chunk, EngineChain.synthesize (voice held loaded for
                 the whole reading; falls down Qwen3-TTS -> Kokoro -> Piper, or
                 the user's choice, notifying once; each engine paces its own
                 output) -> push
```

Replace the whole "Reading-time pacing flow" block:

```
Reading-time pacing flow (Quick Reader and Document Reader; since Stage 6b
`ChunkedSpeaker` applies it once per chunk before the chunk is pushed into the
stream; the last line below is the old `player.play()` call):

```
synthesize text -> audio, sample_rate
-> audio = paced_audio(audio, sample_rate, settings.speaking_speed)
     -> provider() reads the CURRENT speed at the moment of synthesis
     -> apply_speed() time-stretches unless speed is within 1% of 1.0
     -> any failure here is logged and the original audio is played instead
-> player.play(audio, sample_rate)
```
```

with:

```
Reading-time pacing flow (Quick Reader and Document Reader; since Stage 6c
each voice engine applies its own pace inside synthesize(), instead of a
shared post-processing step):

```
EngineChain.synthesize(text)
  -> KokoroTTS / PiperTTS: pass the configured speed straight to the engine's
     own speed knob (Kokoro's `speed=`, Piper's `--length_scale`) -- no
     separate stretch, no quality loss
  -> Qwen3TTS: no native speed control, so it stretches its OWN output with
     apply_speed()/paced_audio(), capped to 1.0-1.2x (never slower than
     natural speech, never sped up past 1.2x, to protect its voice quality);
     any failure here is logged and the original audio is used instead
-> ChunkedSpeaker pushes the (already-paced) audio into the SpeechStream
```
```

- [ ] **Step 3: `docs/architecture.md`**

Replace this bullet (in the Stage 6b "producer/consumer picture" list):

```
- `features/chunked_speaker.py` — `ChunkedSpeaker` is made once per
  reading. Its `run(chunks)` is a producer loop on the caller's worker
  thread: for each chunk it asks a `SpeechSource` (in practice an
  `EngineChain`) for audio, applies the speaking pace (`paced_audio`) and
  pushes the result into the stream. It returns a `SpeakOutcome`
```

with:

```
- `features/chunked_speaker.py` — `ChunkedSpeaker` is made once per
  reading. Its `run(chunks)` is a producer loop on the caller's worker
  thread: for each chunk it asks a `SpeechSource` (in practice an
  `EngineChain`, whose chosen engine already applied its own pace -- see
  "Stage 6c addition: clarity" below) for audio and pushes it into the
  stream. It returns a `SpeakOutcome`
```

Replace the "### Pacing (`audio/pace.py`): one shared helper, applied once per chunk/reading" section header and its final paragraph. Header, from:

```
### Pacing (`audio/pace.py`): one shared helper, applied once per chunk/reading
```

to:

```
### Pacing (`audio/pace.py`): each engine applies its own
```

And the paragraph, from:

```
`paced_audio(audio, sample_rate, provider)` is the one function Quick
Reader and Document Reader both call: it reads the caller's current speed
from `provider()` and applies it, or returns the audio unchanged if
`provider` is `None`, or if either call raises (logged as a warning —
a pacing failure must never break a reading). Since Stage 6b both readers call it through `ChunkedSpeaker`, once per chunk,
before the chunk is pushed into the stream; a pause rewinds the heard clip inside
`SpeechStream`, so a resume replays the same already-paced audio (see the spec
amendment below for the user-visible effect of that choice).
```

to:

```
`paced_audio(audio, sample_rate, provider)` reads the caller's current speed
from `provider()` and applies it, or returns the audio unchanged if
`provider` is `None`, or if either call raises (logged as a warning — a
pacing failure must never break a reading). Since Stage 6c it has exactly one
caller: `Qwen3TTS.synthesize()`, which has no native speed control of its own
and uses it for a mild, capped (1.0-1.2x) time-stretch — never slower than
natural speech, never sped up past 1.2x, so the stretch never gets heavy
enough to hurt this voice's quality. Kokoro and Piper have real speed
controls and never call `paced_audio` at all: `KokoroTTS.synthesize()`
passes the configured speed straight to `kokoro.create(..., speed=...)`, and
`PiperTTS.synthesize()` converts it to Piper's `--length_scale` argument
(`1 / speed`). A pause still rewinds the heard clip inside `SpeechStream`, so
a resume replays the same already-paced audio.
```

In the Stage 6b "Known limits (accepted)" list, delete this now-fixed bullet (it was resolved by Stage 6b's own final fix wave, before this plan; leaving it would misstate the current behaviour):

```
- An engine that loads but fails to synthesize stays loaded until the
  reading ends.
```

Append this new section at the very end of the file (after the last line, the "Quick Reader can show both..." / final bullet list ends the file):

```markdown

## Stage 6c addition: clarity

Stage 6b made long readings gapless; Stage 6c makes them sound better.
Four independent changes, each scoped to the engine or layer that owns the
problem.

### Each engine paces itself

Before Stage 6c, every engine's output went through one shared
`paced_audio()` step in `ChunkedSpeaker`, which always used a lossy
`librosa` time-stretch — even for engines that have a real speed knob.
Now each engine applies its own pace during `synthesize()`, and
`ChunkedSpeaker` no longer touches pacing at all:

- `KokoroTTS` and `PiperTTS` gained a `speed` constructor argument (a
  float or a live callable, read fresh on every call, the same pattern
  `voice`/`speaker` already use). Kokoro passes it straight through to
  `kokoro.create(..., speed=...)`; Piper converts it to its
  `--length_scale` command-line argument (`1 / speed`). Neither ever
  calls `apply_speed()`.
- `Qwen3TTS` has no native speed control, so it keeps the `librosa`
  stretch — but only for a mild, capped range (**1.0x-1.2x**): a bigger
  stretch, in either direction, audibly hurts this voice, so a slower or
  more-sped-up request is simply played at the nearest end of that range
  instead. `audio/pace.py` is unchanged; `paced_audio()`'s only caller is
  now `Qwen3TTS.synthesize()`.
- `cli.py`'s `_make_qwen_factory()` / `_make_kokoro_factory()` and both
  `PiperTTS(...)` construction sites now resolve `speed` the same way
  they already resolve `voice`/`speaker` (`settings.speaking_speed` when
  live settings exist, else the config value), and pass it to the engine.
  `QuickReaderFeature` and `DocumentReaderFeature` lost their
  `speed_provider` constructor argument entirely — pacing is no longer
  something a reader orchestrates.

### Qwen3-TTS generation tuning

`Qwen3TTS.synthesize()` now passes three extra arguments to
`generate_custom_voice()`: a lower, steadier `temperature` (`0.65` instead
of the library's own `0.9` default), a lower `top_k` (`20` instead of
`50`), and a `max_new_tokens` scaled to the text's length (`6` tokens per
character, floored at `200`, capped at the library's own default of
`2048`) instead of always requesting the full default budget. A typical
~300-character chunk now asks for roughly half the token budget it used
to, which is both faster and less prone to the model wandering or
repeating near the end of a long generation; an occasional very long
chunk is still capped at the library's tuned default rather than cut off.

### English-native speaker, with a warning on a mismatch

The default Qwen3-TTS speaker was already `Ryan` (English-native); Stage
6c adds a check that stays useful if that ever changes. Qwen3-TTS always
generates speech with `language="English"` (configuring the spoken
language itself is not yet exposed), but five of its nine preset voices
are native to Chinese, one to Japanese, and one to Korean. `EnginesConfig`
gained a `model_validator` that logs a warning (never raises — a
mismatched choice still works, it just sounds worse) whenever
`qwen3_tts_speaker` is not one of the two English-native presets (`Ryan`,
`Aiden`).

### Basic text normalization

`text/normalizer.py` gained `normalize_text()`: a small, deliberately
basic pass that speaks out URLs (`https://example.com/docs` -> "example
dot com slash docs"), dollar amounts (`$42` -> "42 dollars") and
percentages (`50%` -> "50 percent"), and expands `&`/`@` into "and"/"at".
Plain numbers are left alone — the voice engines already read digit
sequences correctly, so over-normalizing them would do more harm than
good. `normalize_markdown()` now calls it as its last step, so Document
Reader gets it for free; Quick Reader (whose clipboard text is never
markdown) calls it directly, right before chunking, while still notifying
the user with the original, un-normalized copied text.
```

- [ ] **Step 4: `docs/file-purpose.md`**

Replace this row (Stage 1 "new files" table):

```
| `src/vocalis/audio/pace.py` | `apply_speed()`/`clamp_speed()` — pitch-preserving speed change via `librosa.effects.time_stretch` (imported lazily, inside the function); `paced_audio()` — the one helper Quick Reader and Document Reader both call to apply the current speaking pace, swallowing any pacing failure and returning the original audio unchanged. |
```

with:

```
| `src/vocalis/audio/pace.py` | `apply_speed()`/`clamp_speed()` — pitch-preserving speed change via `librosa.effects.time_stretch` (imported lazily, inside the function); `paced_audio()` — swallows any pacing failure and returns the original audio unchanged (since Stage 6c, its only caller is `Qwen3TTS.synthesize()`, capped to 1.0-1.2x; Kokoro and Piper use their own native speed controls instead — see Stage 6c below). |
```

Replace these two rows (Stage 1 "changed files" table):

```
| `src/vocalis/features/quick_reader.py` | Gained `speed_provider`; pacing applied via `paced_audio()` (since Stage 6b once per chunk, inside `ChunkedSpeaker`). |
| `src/vocalis/features/doc_reader.py` | Gained `speed_provider`; pacing applied via `paced_audio()` (since Stage 6b pacing happens once per chunk inside `ChunkedSpeaker`; the old `_play_chunk_until_done()` helper was removed). |
```

with:

```
| `src/vocalis/features/quick_reader.py` | Gained `speed_provider`; pacing applied via `paced_audio()` (since Stage 6b once per chunk, inside `ChunkedSpeaker`; since Stage 6c `speed_provider` was removed again — each voice engine paces its own output now, see Stage 6c below). |
| `src/vocalis/features/doc_reader.py` | Gained `speed_provider`; pacing applied via `paced_audio()` (since Stage 6b pacing happens once per chunk inside `ChunkedSpeaker`, and the old `_play_chunk_until_done()` helper was removed; since Stage 6c `speed_provider` was removed again — each voice engine paces its own output now, see Stage 6c below). |
```

Append this new section at the end of the file (after the Stage 6b "Changed files" table, which is the file's last section):

```markdown

## Stage 6c: clarity

**Changed files**

| File | Change |
|---|---|
| `src/vocalis/engines/qwen3_tts.py` | Gained a `speed` constructor argument (own capped 1.0-1.2x `librosa` stretch, `paced_audio()`'s only remaining caller); `synthesize()` now tunes `temperature`/`top_k`/`max_new_tokens`. |
| `src/vocalis/engines/kokoro_tts.py` | Gained a `speed` constructor argument, passed straight to Kokoro's own `speed=` parameter. |
| `src/vocalis/engines/piper_tts.py` | Gained a `speed` constructor argument, converted to Piper's own `--length_scale` argument (`1 / speed`). |
| `src/vocalis/features/chunked_speaker.py` | Lost `speed_provider` — pacing is now each engine's own job. |
| `src/vocalis/features/quick_reader.py` | Lost `speed_provider`; clipboard text is now passed through `normalize_text()` before chunking (the desktop notification still shows the original text). |
| `src/vocalis/features/doc_reader.py` | Lost `speed_provider`. |
| `src/vocalis/cli.py` | `_make_qwen_factory()`/`_make_kokoro_factory()` and both `PiperTTS(...)` construction sites now resolve and pass a live `speed`. |
| `src/vocalis/config.py` | `EnginesConfig` gained a `model_validator` that warns (never blocks) when `qwen3_tts_speaker` is not English-native (`Ryan`/`Aiden`). |
| `src/vocalis/text/normalizer.py` | Gained `normalize_text()` (URLs, `$`/`%`/`&`/`@`); `normalize_markdown()` now calls it as its last step. |
| `tests/test_qwen3_tts.py`, `tests/test_kokoro_tts.py`, `tests/test_piper_tts.py`, `tests/test_chunked_speaker.py`, `tests/test_quick_reader.py`, `tests/test_doc_reader.py`, `tests/test_cli.py`, `tests/test_config.py`, `tests/test_normalizer.py` | Extended or ported for the above. |
```

- [ ] **Step 5: `docs/folder-structure.md`**

Replace:

```
│   ├── pace.py            # apply_speed/paced_audio: pitch-preserving speaking-pace change
```

with:

```
│   ├── pace.py            # apply_speed/paced_audio: pitch-preserving speaking-pace change (Stage 6c: Qwen3-TTS's own capped 1.0-1.2x stretch only)
```

Replace:

```
│   ├── kokoro_tts.py      # KokoroTTS: light, natural voice via kokoro-onnx (Stage 6a)
```

with:

```
│   ├── kokoro_tts.py      # KokoroTTS: light, natural voice via kokoro-onnx (Stage 6a; native speed control, Stage 6c)
```

Replace:

```
│   ├── piper_tts.py        # PiperTTS: CPU-only fallback voice, never GPU-managed
```

with:

```
│   ├── piper_tts.py        # PiperTTS: CPU-only fallback voice, never GPU-managed (native --length_scale speed, Stage 6c)
```

Replace:

```
│   └── qwen3_tts.py        # Qwen3TTS: primary voice, GPU, via ModelManager
```

with:

```
│   └── qwen3_tts.py        # Qwen3TTS: primary voice, GPU, via ModelManager (tuned generation params + capped speed stretch, Stage 6c)
```

Replace:

```
│   └── normalizer.py       # normalize_markdown: markdown -> speakable text
```

with:

```
│   └── normalizer.py       # normalize_markdown + normalize_text: markdown/plain text -> speakable text (numbers, URLs, symbols; Stage 6c)
```

- [ ] **Step 6: `docs/user-docs.md`**

Replace:

```
- **Speaking speed** — a slider from **0.7x** (slower) to **1.6x**
  (faster), default **1.15x** (a little faster than the voice's natural
  pace). It applies from the next piece of speech it generates — if you
  change it while something is already mid-sentence, that part keeps its
  old speed and the new speed takes over from the next piece onward. This
  is the same for every voice (Qwen3-TTS, Kokoro and Piper).
  **Note:** if you change the speed while a Document Reader reading is
  *paused*, the new speed takes effect from the next chunk once you
  resume — the paused chunk itself resumes at the speed it was already
  playing at.
```

with:

```
- **Speaking speed** — a slider from **0.7x** (slower) to **1.6x**
  (faster), default **1.15x** (a little faster than the voice's natural
  pace). It applies from the next piece of speech it generates — if you
  change it while something is already mid-sentence, that part keeps its
  old speed and the new speed takes over from the next piece onward.
  Kokoro and Piper follow the slider across its whole range. Qwen3-TTS has
  no speed control of its own, so it only ever speeds up, and only up to
  **1.2x** — asking for slower than natural, or faster than 1.2x, still
  sounds like Qwen3-TTS's normal pace, to protect its voice quality.
  **Note:** if you change the speed while a Document Reader reading is
  *paused*, the new speed takes effect from the next chunk once you
  resume — the paused chunk itself resumes at the speed it was already
  playing at.
```

- [ ] **Step 7: Write the changelog**

Create `.claude/<today's date and time as YYYY-MM-DD_HHMM>_CHANGELOG.md` (match the existing file's naming, e.g. `.claude/2026-10-20_1801_CHANGELOG.md` from Stage 6b) summarizing, in plain beginner-friendly English (per the project's global instructions): what changed (each engine now controls its own speaking speed; Qwen3-TTS's voice generation was tuned to be steadier and faster; a warning appears if a non-English voice is picked for Qwen3-TTS; clipboard and document text now speak URLs/money/percent signs properly instead of reading symbols literally), why, and the before/after test numbers.

- [ ] **Step 8: Run the full test suite, mypy, and commit**

Run: `.venv/bin/python -m pytest -q`
Expected: all tests pass (same count as before this task, plus the new tests from Tasks 1-6), coverage still >= 85%.

Run: `.venv/bin/python -m mypy src`
Expected: no errors.

```bash
git add docs/architecture.md docs/algorithms.md docs/workflow.md docs/file-purpose.md docs/folder-structure.md docs/user-docs.md .claude/*_CHANGELOG.md
git commit -m "docs: document Stage 6c clarity changes"
```
