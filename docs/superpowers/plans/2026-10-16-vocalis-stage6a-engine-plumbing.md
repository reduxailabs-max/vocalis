# Vocalis Stage 6a — Engine Plumbing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the Kokoro voice engine, a pure "Auto" engine-picker that follows the user's rule, a fall-down-the-chain mechanism, and a Settings picker — replacing today's hard-wired Qwen3-TTS-then-Piper.

**Architecture:** A new `EngineSelector` (config preference + hardware snapshot -> ordered list of engine ids) and a new `EngineChain` (walks that list for one reading, remembering failures) replace the duplicated `_synthesize_with_fallback` code in Quick Reader and Document Reader. Engines are still loaded through the existing single-residency `ModelManager`; Piper stays the always-ready, unmanaged last resort. Kokoro is a new `ModelEngine` that wraps the `kokoro-onnx` library.

**Tech Stack:** Python 3.13, pydantic, PySide6 (Settings window), onnxruntime (via `kokoro-onnx` 0.4.7, loaded lazily with `importlib`), pytest, mypy strict, bash + bats + shellcheck.

**Spec:** `docs/superpowers/specs/2026-09-11-vocalis-design.md` — section "Stage 6 amendment — Voice engine overhaul", group 6a (engine plumbing).

## Global Constraints

- Tests NEVER touch real hardware, GPU, network, audio devices, or the real `kokoro_onnx` / `onnxruntime` libraries: those are imported ONLY inside small functions via `importlib.import_module(...)` and replaced by injected fakes in tests. `vocalis.cli.probe_hardware` must be patched in every test that builds a feature (see Task 6).
- `.venv/bin/python -m pytest -q -p no:cacheprovider` must keep coverage >= 85% with zero warnings; `.venv/bin/python -m mypy` must report "Success: no issues found"; `shellcheck` zero warnings and `bats tests/installer/` all green for installer changes. The only tolerated failures are the two pre-existing `tests/test_offline_guard.py` cases (they need `PYTHONPATH` in the environment).
- Write ALL the code for a task first, then run the tests ONCE (user preference — no test-as-you-go loops). Run pytest in the foreground only, never backgrounded, and never while another process is running tests in the same worktree.
- `sed -i` fails on this NTFS mount: edit with the Edit tool or python3.
- Every function/class needs a short purpose docstring. User-facing text is plain, non-technical English. No `Co-Authored-By` lines in commits.
- `TTS_ENGINE_ID = "qwen3_tts"` in `vocalis/model_manager.py` STAYS (tests import it); it is now simply "the id of the Qwen3-TTS engine". New ids: `"kokoro"`. Piper is never routed through `ModelManager`.
- Engine ids used everywhere: `"qwen3_tts"`, `"kokoro"`, `"piper"` (constants in `vocalis/engines/auto_select.py`).
- Existing UI strings asserted by tests must not change; the downshift notification text stays exactly `"Primary voice unavailable — using backup voice"` and the Document Reader status stays `"Using backup voice"`.
- Kokoro facts (verified from the real `kokoro_onnx` 0.4.7 wheel and release files): `Kokoro.from_session(session, voices_path)` accepts a ready `onnxruntime.InferenceSession`; `create(text, voice, speed=1.0, lang="en-us")` returns `(float32 samples, 24000)`; model file `kokoro-v1.0.onnx` is 325,532,387 bytes and `voices-v1.0.bin` is 28,214,398 bytes, both at `https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/<name>`; `kokoro-onnx` hard-requires plain `onnxruntime`, which can silently disable CUDA for Parakeet if installed normally (hence `--no-deps`, Task 8); English voice names beginning `af_ am_ bf_ bm_` exist (e.g. `af_heart`, `bf_emma`).

## Review Focus

1. A forced engine whose model files are missing must fall down the chain with a notification, never crash the reading (Task 4 chain tests).
2. "Auto" when the hardware probe fails or reports unknown (None) fields must still give a deterministic, exception-free plan (Tasks 2 and 4).
3. Quick Reader and Document Reader both on "Auto" must pick the same engine id so switching between them never reloads the model (Task 6 integration test).
4. A Kokoro voice name that is not in the voices file must fail cleanly (down the chain), not produce silence (Task 3).
5. An existing user `config.toml` that still contains the old dead keys `quick_tts` / `doc_tts` must load and mean "Auto" (Task 1).

---

### Task 1: Config — engine choice, Kokoro settings, live getters

**Files:**
- Modify: `src/vocalis/config.py` (`EnginesConfig`, imports, two new default-path helpers)
- Modify: `src/vocalis/settings.py` (new `KOKORO_VOICES` tuple + three getters on `LiveSettings`)
- Modify: `config/default_config.toml`
- Test: `tests/test_config.py`, `tests/test_settings.py`

**Interfaces:**
- Produces: `EngineChoice = Literal["auto", "qwen3_tts", "kokoro", "piper"]` (in `vocalis.config`); `EnginesConfig` fields `quick_reader_engine: EngineChoice = "auto"`, `document_reader_engine: EngineChoice = "auto"`, `kokoro_model_path: str`, `kokoro_voices_path: str`, `kokoro_voice: str = "af_heart"`; `LiveSettings.quick_reader_engine() -> str`, `.document_reader_engine() -> str`, `.kokoro_voice() -> str`; `vocalis.settings.KOKORO_VOICES: tuple[str, ...]`.
- The old dead fields `quick_tts` / `doc_tts` are REMOVED from `EnginesConfig` (pydantic ignores unknown keys, so old config files still load).

- [ ] **Step 1: Write the tests.** In `tests/test_config.py` change the import to `from vocalis.config import (ConfigError, EnginesConfig, VocalisConfig, default_config_path, load_config, save_config)`, add `from pydantic import ValidationError`, DELETE the two assertions on `config.engines.quick_tts` / `doc_tts` in `test_defaults_match_spec` (lines ~23-24) and append:

```python
def test_engine_choice_defaults_to_auto_for_both_readers() -> None:
    """Both readers start on Automatic voice-engine choice."""
    engines = VocalisConfig().engines
    assert engines.quick_reader_engine == "auto"
    assert engines.document_reader_engine == "auto"


def test_engine_choice_rejects_an_unknown_engine_name() -> None:
    """A typo'd engine name is a validation error, not a silent fallback."""
    with pytest.raises(ValidationError):
        EnginesConfig(quick_reader_engine="banana")  # type: ignore[arg-type]


def test_old_config_with_the_dead_quick_tts_keys_loads_and_means_auto(tmp_path: Path) -> None:
    """Config files written by earlier versions still load; their ignored keys mean Auto."""
    path = tmp_path / "config.toml"
    path.write_text('[engines]\nquick_tts = "qwen3_tts"\ndoc_tts = "qwen3_tts"\n', encoding="utf-8")
    config = load_config(path)
    assert config.engines.quick_reader_engine == "auto"
    assert config.engines.document_reader_engine == "auto"


def test_kokoro_defaults_live_under_the_vocalis_models_folder() -> None:
    """Kokoro's model files default to ~/.local/share/vocalis/models/kokoro/."""
    engines = VocalisConfig().engines
    assert engines.kokoro_model_path.endswith("/.local/share/vocalis/models/kokoro/kokoro-v1.0.onnx")
    assert engines.kokoro_voices_path.endswith("/.local/share/vocalis/models/kokoro/voices-v1.0.bin")
    assert engines.kokoro_voice == "af_heart"
```

In `tests/test_settings.py` append:

```python
def test_engine_and_kokoro_voice_getters_follow_the_current_config(tmp_path: Path) -> None:
    """The three new getters read the live config."""
    from vocalis.config import EnginesConfig

    config = VocalisConfig(
        engines=EnginesConfig(quick_reader_engine="kokoro", document_reader_engine="piper", kokoro_voice="bf_emma")
    )
    settings = make(tmp_path, config)
    assert settings.quick_reader_engine() == "kokoro"
    assert settings.document_reader_engine() == "piper"
    assert settings.kokoro_voice() == "bf_emma"


def test_kokoro_voices_are_real_english_voice_names() -> None:
    """The Settings shortlist only offers voices that exist in Kokoro's voices file."""
    from vocalis.settings import KOKORO_VOICES

    assert "af_heart" in KOKORO_VOICES
    assert all(name[:2] in ("af", "am", "bf", "bm") for name in KOKORO_VOICES)
```

- [ ] **Step 2: Implement.** In `src/vocalis/config.py`: change `from typing import Annotated` to `from typing import Annotated, Literal` (add `Literal` to the existing typing import); above `EnginesConfig` add

```python
EngineChoice = Literal["auto", "qwen3_tts", "kokoro", "piper"]


def _default_kokoro_model_path() -> str:
    """Default location of the Kokoro voice model file."""
    return str(Path.home() / ".local" / "share" / "vocalis" / "models" / "kokoro" / "kokoro-v1.0.onnx")


def _default_kokoro_voices_path() -> str:
    """Default location of Kokoro's voices file."""
    return str(Path.home() / ".local" / "share" / "vocalis" / "models" / "kokoro" / "voices-v1.0.bin")
```

and replace the two lines `quick_tts: str = "qwen3_tts"` / `doc_tts: str = "qwen3_tts"` with

```python
    quick_reader_engine: EngineChoice = "auto"
    document_reader_engine: EngineChoice = "auto"
```

and add after `piper_voice_model_path`:

```python
    kokoro_model_path: str = Field(default_factory=_default_kokoro_model_path)
    kokoro_voices_path: str = Field(default_factory=_default_kokoro_voices_path)
    kokoro_voice: str = "af_heart"
```

In `src/vocalis/settings.py` add below `VOICES`:

```python
KOKORO_VOICES: tuple[str, ...] = (
    "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky",
    "am_adam", "am_michael", "bf_emma", "bf_isabella", "bm_george", "bm_lewis",
)
```

and after `voice()` add the three getters:

```python
    def quick_reader_engine(self) -> str:
        """Which voice engine Quick Reader should use ("auto" lets Vocalis pick)."""
        return self.config.engines.quick_reader_engine

    def document_reader_engine(self) -> str:
        """Which voice engine Document Reader should use ("auto" lets Vocalis pick)."""
        return self.config.engines.document_reader_engine

    def kokoro_voice(self) -> str:
        """The Kokoro voice to use for the next reading."""
        return self.config.engines.kokoro_voice
```

In `config/default_config.toml` replace the two lines `quick_tts = "qwen3_tts"` / `doc_tts = "qwen3_tts"` with:

```toml
# Voice engine per reader: "auto" (Vocalis picks from your PC's load), "qwen3_tts", "kokoro", or "piper".
quick_reader_engine = "auto"
document_reader_engine = "auto"
```

- [ ] **Step 3: Run once:** `.venv/bin/python -m pytest tests/test_config.py tests/test_settings.py -q -p no:cacheprovider` — all pass (including `test_default_toml_matches_model_defaults`). Fix anything that fails (also grep `tests/` for any other `quick_tts` use).
- [ ] **Step 4: Commit** — `git add -A && git commit -m "feat: add per-reader voice engine choice and Kokoro settings"`

---

### Task 2: Pure "Auto" engine-picker policy

**Files:**
- Create: `src/vocalis/engines/auto_select.py`
- Test: `tests/test_auto_select.py`

**Interfaces:**
- Consumes: `vocalis.hardware.HardwareInfo`, `GpuInfo` (Stage 5).
- Produces: constants `QWEN3_TTS = "qwen3_tts"`, `KOKORO = "kokoro"`, `PIPER = "piper"`; `@dataclass(frozen=True) AutoPolicy` (fields below); `@dataclass(frozen=True) AutoDecision(chain: tuple[str, ...], reason: str)`; `is_stressed(hw, policy) -> bool`; `decide(hw, policy=AutoPolicy(), held_vram_mb=0) -> AutoDecision`; `kokoro_should_use_gpu(hw, policy=AutoPolicy()) -> bool`.

- [ ] **Step 1: Write the tests** — `tests/test_auto_select.py`:

```python
"""Tests for vocalis.engines.auto_select — the pure rule table behind "Auto" voice choice."""
from vocalis.engines.auto_select import (
    KOKORO, PIPER, QWEN3_TTS, AutoPolicy, decide, is_stressed, kokoro_should_use_gpu,
)
from vocalis.hardware import GpuInfo, HardwareInfo


def _hw(
    total: int | None = 8188, free: int = 7000, ram_total: int | None = 16000,
    ram_avail: int | None = 10000, cores: int | None = 8, load: float | None = 1.0,
) -> HardwareInfo:
    """A hardware snapshot; total=None means no GPU."""
    gpu = GpuInfo("Test GPU", total, free) if total is not None else None
    return HardwareInfo(gpu, ram_total, ram_avail, cores, load)


def test_no_graphics_card_uses_the_lightest_voice_then_kokoro() -> None:
    """With no GPU the plan is Piper first, Kokoro as the backup."""
    decision = decide(_hw(total=None))
    assert decision.chain == (PIPER, KOKORO)
    assert "graphics card" in decision.reason


def test_very_low_memory_uses_the_lightest_voice() -> None:
    """Critically low free RAM picks Piper even on a strong GPU."""
    assert decide(_hw(ram_avail=500)).chain == (PIPER, KOKORO)


def test_strong_idle_pc_uses_qwen3_then_the_lighter_voices() -> None:
    """A big, idle GPU gets the best-quality voice with the others as backups."""
    decision = decide(_hw())
    assert decision.chain == (QWEN3_TTS, KOKORO, PIPER)
    assert "best-quality" in decision.reason


def test_a_four_gb_card_is_not_big_enough() -> None:
    """The rule is 'more than 4 GB': exactly 4096 MB total goes to Kokoro."""
    assert decide(_hw(total=4096, free=4096)).chain == (KOKORO, PIPER)


def test_just_over_four_gb_with_room_is_enough() -> None:
    """4097 MB total with 4096 MB free is the smallest card that gets Qwen3-TTS."""
    assert decide(_hw(total=4097, free=4096)).chain == (QWEN3_TTS, KOKORO, PIPER)


def test_busy_processor_uses_kokoro() -> None:
    """Load per core above 0.85 means the PC is busy."""
    decision = decide(_hw(load=7.0, cores=8))
    assert decision.chain == (KOKORO, PIPER)
    assert "busy" in decision.reason


def test_load_exactly_at_the_threshold_is_not_busy() -> None:
    """6.8 load on 8 cores is exactly 0.85: not stressed."""
    assert decide(_hw(load=6.8, cores=8)).chain[0] == QWEN3_TTS


def test_low_free_memory_counts_as_busy() -> None:
    """Free RAM under max(2 GB, 15% of total) is stress."""
    assert is_stressed(_hw(ram_total=16000, ram_avail=2300), AutoPolicy()) is True
    assert decide(_hw(ram_total=16000, ram_avail=2300)).chain[0] == KOKORO


def test_graphics_memory_used_by_other_programs_uses_kokoro() -> None:
    """Under 4 GB of free graphics memory means Qwen3-TTS will not fit."""
    decision = decide(_hw(free=3000))
    assert decision.chain == (KOKORO, PIPER)
    assert "graphics card" in decision.reason


def test_vocalis_own_loaded_model_is_not_counted_against_free_memory() -> None:
    """Credit for memory Vocalis itself holds prevents flip-flopping off Qwen3-TTS."""
    assert decide(_hw(free=1000), held_vram_mb=4096).chain[0] == QWEN3_TTS
    assert decide(_hw(free=1000), held_vram_mb=0).chain[0] == KOKORO


def test_unknown_load_and_memory_are_not_treated_as_stress() -> None:
    """Missing readings never block the best voice."""
    hw = _hw(ram_total=None, ram_avail=None, cores=None, load=None)
    assert is_stressed(hw, AutoPolicy()) is False
    assert decide(hw).chain[0] == QWEN3_TTS


def test_a_custom_policy_changes_the_decision() -> None:
    """Thresholds are named policy values, not magic numbers."""
    assert decide(_hw(), AutoPolicy(qwen3_need_free_mb=9000)).chain[0] == KOKORO


def test_kokoro_uses_the_gpu_only_when_there_is_room() -> None:
    """Kokoro runs on the graphics card when at least 1 GB is free, else on the processor."""
    assert kokoro_should_use_gpu(_hw(free=2000)) is True
    assert kokoro_should_use_gpu(_hw(free=500)) is False
    assert kokoro_should_use_gpu(_hw(total=None)) is False
```

- [ ] **Step 2: Implement** `src/vocalis/engines/auto_select.py`:

```python
"""Decides which voice engine "Auto" should use right now.

Pure: it only looks at a HardwareInfo snapshot and a policy of named
thresholds, so every rule is unit-testable without a GPU. The user's rule:
a graphics card with more than 4 GB and a PC that is not under stress
-> Qwen3-TTS; a busy PC or too little free graphics memory -> Kokoro;
no graphics card -> Piper.
"""
from dataclasses import dataclass

from vocalis.hardware import HardwareInfo

QWEN3_TTS = "qwen3_tts"
KOKORO = "kokoro"
PIPER = "piper"


@dataclass(frozen=True)
class AutoPolicy:
    """The named thresholds behind Auto. All sizes are in megabytes."""

    qwen3_min_gpu_total_mb: int = 4096
    qwen3_need_free_mb: int = 4096
    kokoro_gpu_min_free_mb: int = 1024
    stress_load_per_core: float = 0.85
    stress_min_free_ram_mb: int = 2048
    stress_min_free_ram_fraction: float = 0.15
    critical_free_ram_mb: int = 1024


@dataclass(frozen=True)
class AutoDecision:
    """Engines to try in order, plus a plain-English reason for the log."""

    chain: tuple[str, ...]
    reason: str


def is_stressed(hw: HardwareInfo, policy: AutoPolicy) -> bool:
    """Whether the processor or memory is busy enough to avoid the heavy voice."""
    if hw.load_1min is not None and hw.cpu_count:
        if hw.load_1min / hw.cpu_count > policy.stress_load_per_core:
            return True
    if hw.ram_available_mb is not None:
        floor = policy.stress_min_free_ram_mb
        if hw.ram_total_mb:
            floor = max(floor, int(hw.ram_total_mb * policy.stress_min_free_ram_fraction))
        if hw.ram_available_mb < floor:
            return True
    return False


def decide(hw: HardwareInfo, policy: AutoPolicy = AutoPolicy(), held_vram_mb: int = 0) -> AutoDecision:
    """Pick the engine order for the current hardware snapshot.

    Args:
        hw: What the PC looks like right now.
        policy: The thresholds.
        held_vram_mb: Graphics memory Vocalis itself currently holds, added
            back to "free" so its own loaded model never counts as stress.
    """
    if hw.gpu is None:
        return AutoDecision((PIPER, KOKORO), "No graphics card was found, so Vocalis is using the lightest voice.")
    if hw.ram_available_mb is not None and hw.ram_available_mb < policy.critical_free_ram_mb:
        return AutoDecision((PIPER, KOKORO), "Your PC is very low on memory, so Vocalis is using the lightest voice.")
    if hw.gpu.total_mb <= policy.qwen3_min_gpu_total_mb:
        return AutoDecision((KOKORO, PIPER), "Your graphics card is small, so Vocalis is using the lighter voice.")
    if is_stressed(hw, policy):
        return AutoDecision((KOKORO, PIPER), "Your PC is busy right now, so Vocalis is using the lighter voice.")
    if hw.gpu.free_mb + held_vram_mb < policy.qwen3_need_free_mb:
        return AutoDecision(
            (KOKORO, PIPER),
            "Other programs are using your graphics card, so Vocalis is using the lighter voice.",
        )
    return AutoDecision(
        (QWEN3_TTS, KOKORO, PIPER),
        "Your PC has room to spare, so Vocalis is using the best-quality voice.",
    )


def kokoro_should_use_gpu(hw: HardwareInfo, policy: AutoPolicy = AutoPolicy()) -> bool:
    """Whether Kokoro should run on the graphics card (enough free memory) or the processor."""
    return hw.gpu is not None and hw.gpu.free_mb >= policy.kokoro_gpu_min_free_mb
```

- [ ] **Step 3: Run once:** `.venv/bin/python -m pytest tests/test_auto_select.py -q -p no:cacheprovider --no-cov`; fix failures.
- [ ] **Step 4: Commit** — `git add -A && git commit -m "feat: add the pure Auto voice-engine decision policy"`

---

### Task 3: Kokoro engine

**Files:**
- Create: `src/vocalis/engines/kokoro_tts.py`
- Test: `tests/test_kokoro_tts.py`

**Interfaces:**
- Produces: `class KokoroLoadError(Exception)`, `class KokoroSynthesisError(Exception)`, and
  `KokoroTTS(model_path: Path, voices_path: Path, voice: str | Callable[[], str] = "af_heart", use_gpu: bool | Callable[[], bool] = False, session_builder: Callable[[Path, bool], Any] = <real>, kokoro_factory: Callable[[Any, Path], Any] = <real>)` with `load()`, `unload()`, `is_loaded`, `synthesize(text) -> tuple[npt.NDArray[np.float32], int]` (satisfies `ModelEngine` and `Synthesizer`). `use_gpu` and `voice` callables are evaluated at `load()` / each `synthesize()` respectively. Speed is NOT applied here (Stage 6c adds native speed); features still apply `paced_audio`.

- [ ] **Step 1: Write the tests** — `tests/test_kokoro_tts.py`:

```python
"""Tests for vocalis.engines.kokoro_tts — the real kokoro_onnx / onnxruntime are replaced by fakes."""
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vocalis.engines.kokoro_tts import KokoroLoadError, KokoroSynthesisError, KokoroTTS


class _FakeKokoro:
    """Records create() calls and returns a short silent clip."""

    def __init__(self, fail: bool = False) -> None:
        self.calls: list[tuple[str, str, float, str]] = []
        self._fail = fail

    def create(self, text: str, voice: str, speed: float, lang: str) -> tuple[Any, int]:
        if self._fail:
            raise AssertionError("Voice not found")
        self.calls.append((text, voice, speed, lang))
        return np.zeros((2400, 1), dtype=np.float64), 24000


def _files(tmp_path: Path) -> tuple[Path, Path]:
    model, voices = tmp_path / "kokoro-v1.0.onnx", tmp_path / "voices-v1.0.bin"
    model.write_bytes(b"x")
    voices.write_bytes(b"x")
    return model, voices


def _engine(tmp_path: Path, kokoro: _FakeKokoro | None = None, **kwargs: Any) -> tuple[KokoroTTS, list[tuple[Path, bool]], _FakeKokoro]:
    model, voices = _files(tmp_path)
    built: list[tuple[Path, bool]] = []
    fake = kokoro or _FakeKokoro()

    def session_builder(path: Path, use_gpu: bool) -> object:
        built.append((path, use_gpu))
        return object()

    engine = KokoroTTS(model, voices, session_builder=session_builder, kokoro_factory=lambda _s, _v: fake, **kwargs)
    return engine, built, fake


def test_load_builds_the_session_and_marks_the_engine_loaded(tmp_path: Path) -> None:
    """load() creates the ONNX session for the model file."""
    engine, built, _ = _engine(tmp_path)
    assert engine.is_loaded is False
    engine.load()
    assert engine.is_loaded is True
    assert built[0][0].name == "kokoro-v1.0.onnx"


def test_load_with_a_missing_model_file_raises_a_clear_error(tmp_path: Path) -> None:
    """A missing model file fails fast with KokoroLoadError."""
    engine = KokoroTTS(tmp_path / "no.onnx", tmp_path / "no.bin")
    with pytest.raises(KokoroLoadError, match="no.onnx"):
        engine.load()


def test_load_wraps_any_library_failure(tmp_path: Path) -> None:
    """Whatever the ONNX library raises becomes KokoroLoadError."""
    model, voices = _files(tmp_path)

    def boom(_path: Path, _gpu: bool) -> object:
        raise RuntimeError("cuda exploded")

    engine = KokoroTTS(model, voices, session_builder=boom)
    with pytest.raises(KokoroLoadError, match="cuda exploded"):
        engine.load()
    assert engine.is_loaded is False


def test_use_gpu_is_decided_when_loading(tmp_path: Path) -> None:
    """A callable use_gpu is evaluated at load() time."""
    flag = {"gpu": False}
    engine, built, _ = _engine(tmp_path, use_gpu=lambda: flag["gpu"])
    flag["gpu"] = True
    engine.load()
    assert built[0][1] is True


def test_synthesize_returns_flat_float32_audio_and_the_sample_rate(tmp_path: Path) -> None:
    """Audio comes back as a 1-D float32 array with Kokoro's sample rate."""
    engine, _, fake = _engine(tmp_path)
    engine.load()
    audio, rate = engine.synthesize("Hello there.")
    assert audio.dtype == np.float32 and audio.ndim == 1 and rate == 24000
    assert fake.calls == [("Hello there.", "af_heart", 1.0, "en-us")]


def test_british_voices_use_british_english(tmp_path: Path) -> None:
    """Voices starting with b use the en-gb phonemes."""
    engine, _, fake = _engine(tmp_path, voice="bf_emma")
    engine.load()
    engine.synthesize("Hello.")
    assert fake.calls[0][3] == "en-gb"


def test_the_voice_is_read_live_on_every_synthesis(tmp_path: Path) -> None:
    """A callable voice follows Settings changes without reloading."""
    current = {"voice": "af_heart"}
    engine, _, fake = _engine(tmp_path, voice=lambda: current["voice"])
    engine.load()
    engine.synthesize("a")
    current["voice"] = "am_adam"
    engine.synthesize("b")
    assert [call[1] for call in fake.calls] == ["af_heart", "am_adam"]


def test_an_unknown_voice_fails_cleanly(tmp_path: Path) -> None:
    """A voice missing from the voices file is a KokoroSynthesisError, not silence."""
    engine, _, _ = _engine(tmp_path, kokoro=_FakeKokoro(fail=True), voice="nope")
    engine.load()
    with pytest.raises(KokoroSynthesisError, match="Voice not found"):
        engine.synthesize("Hello.")


def test_synthesize_before_load_fails_cleanly(tmp_path: Path) -> None:
    """Using the engine before load() is a clear error."""
    engine, _, _ = _engine(tmp_path)
    with pytest.raises(KokoroSynthesisError, match="not loaded"):
        engine.synthesize("Hello.")


def test_unload_releases_the_model(tmp_path: Path) -> None:
    """unload() drops the model so memory can be freed."""
    engine, _, _ = _engine(tmp_path)
    engine.load()
    engine.unload()
    assert engine.is_loaded is False
```

- [ ] **Step 2: Implement** `src/vocalis/engines/kokoro_tts.py`:

```python
"""Kokoro text-to-speech engine: small, natural-sounding, fast on a normal
processor and faster on a graphics card. Wraps the `kokoro-onnx` library.

`kokoro_onnx` and `onnxruntime` are imported only inside the two small
default builder functions below (via importlib), so importing this module
is cheap and tests can replace both with fakes.
"""
import importlib
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import numpy.typing as npt

logger = logging.getLogger("vocalis")


class KokoroLoadError(Exception):
    """Raised when the Kokoro model cannot be loaded."""


class KokoroSynthesisError(Exception):
    """Raised when Kokoro cannot turn text into speech."""


def _language_for(voice: str) -> str:
    """Kokoro voices that start with 'b' speak British English, the rest American."""
    return "en-gb" if voice.startswith("b") else "en-us"


def _default_session_builder(model_path: Path, use_gpu: bool) -> Any:
    """Create the ONNX session, on the graphics card only if asked and available."""
    ort = importlib.import_module("onnxruntime")
    providers = ["CPUExecutionProvider"]
    if use_gpu and "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(str(model_path), providers=providers)


def _default_kokoro_factory(session: Any, voices_path: Path) -> Any:
    """Wrap an ONNX session in the kokoro_onnx.Kokoro synthesizer."""
    module = importlib.import_module("kokoro_onnx")
    return module.Kokoro.from_session(session, str(voices_path))


class KokoroTTS:
    """Text-to-speech engine backed by the Kokoro ONNX model."""

    def __init__(
        self,
        model_path: Path,
        voices_path: Path,
        voice: "str | Callable[[], str]" = "af_heart",
        use_gpu: "bool | Callable[[], bool]" = False,
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
            session_builder: Builds the ONNX session; injectable for tests.
            kokoro_factory: Wraps a session into a synthesizer; injectable for tests.
        """
        self._model_path = model_path
        self._voices_path = voices_path
        self._voice = voice
        self._use_gpu = use_gpu
        self._session_builder = session_builder
        self._kokoro_factory = kokoro_factory
        self._kokoro: Any | None = None

    @property
    def is_loaded(self) -> bool:
        """Whether the model is currently loaded."""
        return self._kokoro is not None

    def load(self) -> None:
        """Load the model.

        Raises:
            KokoroLoadError: If a model file is missing or loading fails.
        """
        for path in (self._model_path, self._voices_path):
            if not path.exists():
                raise KokoroLoadError(f"Kokoro file not found: {path}")
        use_gpu = self._use_gpu() if callable(self._use_gpu) else self._use_gpu
        try:
            session = self._session_builder(self._model_path, use_gpu)
            self._kokoro = self._kokoro_factory(session, self._voices_path)
        except Exception as exc:
            self._kokoro = None
            raise KokoroLoadError(f"Could not load Kokoro: {exc}") from exc
        logger.info("Kokoro loaded (graphics card: %s)", use_gpu)

    def unload(self) -> None:
        """Release the model."""
        self._kokoro = None

    def _current_voice(self) -> str:
        """The voice to use right now."""
        return self._voice() if callable(self._voice) else self._voice

    def synthesize(self, text: str) -> "tuple[npt.NDArray[np.float32], int]":
        """Turn text into speech.

        Raises:
            KokoroSynthesisError: If the engine is not loaded, the voice does
                not exist, or Kokoro fails.
        """
        if self._kokoro is None:
            raise KokoroSynthesisError("Kokoro is not loaded")
        voice = self._current_voice()
        try:
            samples, sample_rate = self._kokoro.create(text, voice=voice, speed=1.0, lang=_language_for(voice))
        except Exception as exc:
            raise KokoroSynthesisError(f"Kokoro could not speak this text: {exc}") from exc
        return np.asarray(samples, dtype=np.float32).reshape(-1), int(sample_rate)
```

- [ ] **Step 3: Run once:** `.venv/bin/python -m pytest tests/test_kokoro_tts.py -q -p no:cacheprovider --no-cov`; then `.venv/bin/python -m mypy`. Fix failures.
- [ ] **Step 4: Commit** — `git add -A && git commit -m "feat: add the Kokoro voice engine"`

---

### Task 4: Shared protocols, engine selector, and engine chain

**Files:**
- Create: `src/vocalis/engines/base.py`, `src/vocalis/engines/registry.py`, `src/vocalis/engines/chain.py`
- Test: `tests/test_engine_registry.py`, `tests/test_engine_chain.py`

**Interfaces:**
- Consumes: Task 2 (`decide`, `kokoro_should_use_gpu`, `AutoPolicy`, constants), `vocalis.model_manager.ModelEngine`/`ModelManager`, `vocalis.hardware.HardwareInfo`.
- Produces:
  - `base.py`: `class Synthesizer(Protocol)` with `synthesize(self, text: str) -> tuple[npt.NDArray[np.float32], int]`; `class TTSEngine(ModelEngine, Synthesizer, Protocol)`.
  - `registry.py`: `@dataclass(frozen=True) EnginePlan(chain: tuple[str, ...], reason: str)`; `EngineSelector(preference: Callable[[], str], hardware_probe: Callable[[], HardwareInfo], held_vram_mb: Callable[[], int] = lambda: 0, policy: AutoPolicy = AutoPolicy())` with `plan() -> EnginePlan` and `kokoro_use_gpu() -> bool`.
  - `chain.py`: `EngineChain(plan: EnginePlan, factories: Mapping[str, Callable[[], ModelEngine]], model_manager: ModelManager, fallback_engine: Synthesizer, on_downshift: Callable[[str, str | None], None])` with `synthesize(text) -> tuple[npt.NDArray[np.float32], int]`.

- [ ] **Step 1: Write the tests.** `tests/test_engine_registry.py`:

```python
"""Tests for vocalis.engines.registry.EngineSelector."""
import pytest

from vocalis.engines.auto_select import KOKORO, PIPER, QWEN3_TTS
from vocalis.engines.registry import EngineSelector
from vocalis.hardware import GpuInfo, HardwareInfo

STRONG = HardwareInfo(GpuInfo("Big", 8188, 7000), 16000, 10000, 8, 1.0)
NO_GPU = HardwareInfo(None, 16000, 10000, 8, 1.0)


def _selector(choice: str, hw: HardwareInfo = STRONG, held: int = 0) -> EngineSelector:
    return EngineSelector(preference=lambda: choice, hardware_probe=lambda: hw, held_vram_mb=lambda: held)


def test_forced_qwen3_falls_down_to_kokoro_then_piper() -> None:
    """Choosing Qwen3-TTS still keeps the lighter voices as backups."""
    plan = _selector("qwen3_tts", NO_GPU).plan()
    assert plan.chain == (QWEN3_TTS, KOKORO, PIPER)
    assert "Qwen3-TTS" in plan.reason and "Settings" in plan.reason


def test_forced_kokoro_and_piper_chains() -> None:
    """Kokoro falls back to Piper; Piper falls back to Kokoro."""
    assert _selector("kokoro").plan().chain == (KOKORO, PIPER)
    assert _selector("piper").plan().chain == (PIPER, KOKORO)


def test_auto_follows_the_hardware() -> None:
    """Auto uses the decision policy: strong PC -> Qwen3-TTS, no GPU -> Piper."""
    assert _selector("auto", STRONG).plan().chain[0] == QWEN3_TTS
    assert _selector("auto", NO_GPU).plan().chain[0] == PIPER


def test_auto_counts_the_memory_vocalis_itself_holds() -> None:
    """Held graphics memory is added back so a loaded Qwen3-TTS keeps being chosen."""
    low = HardwareInfo(GpuInfo("Big", 8188, 1000), 16000, 10000, 8, 1.0)
    assert _selector("auto", low, held=0).plan().chain[0] == KOKORO
    assert _selector("auto", low, held=4096).plan().chain[0] == QWEN3_TTS


def test_auto_survives_a_failing_hardware_probe() -> None:
    """If the probe raises, Auto picks the safe middle choice instead of crashing."""
    def boom() -> HardwareInfo:
        raise RuntimeError("probe failed")

    plan = EngineSelector(preference=lambda: "auto", hardware_probe=boom).plan()
    assert plan.chain == (KOKORO, PIPER)
    assert "could not check" in plan.reason.lower()


def test_an_unknown_preference_is_treated_as_auto() -> None:
    """A garbage stored value never breaks reading."""
    assert _selector("banana", STRONG).plan().chain[0] == QWEN3_TTS


@pytest.mark.parametrize("free,expected", [(2000, True), (500, False)])
def test_kokoro_use_gpu_follows_free_graphics_memory(free: int, expected: bool) -> None:
    """Kokoro runs on the GPU only with enough free memory."""
    hw = HardwareInfo(GpuInfo("Big", 8188, free), 16000, 10000, 8, 1.0)
    assert _selector("auto", hw).kokoro_use_gpu() is expected
```

`tests/test_engine_chain.py`:

```python
"""Tests for vocalis.engines.chain.EngineChain — walking the engine list for one reading."""
import numpy as np
import numpy.typing as npt
import pytest

from vocalis.engines.chain import EngineChain
from vocalis.engines.registry import EnginePlan
from vocalis.model_manager import ModelManager


class _Tts:
    """A fake managed engine; its audio is filled with `value` so tests can tell engines apart."""

    def __init__(self, value: float, fail_load: bool = False, fail_synth: bool = False) -> None:
        self.value, self.fail_load, self.fail_synth = value, fail_load, fail_synth
        self.loaded = False

    def load(self) -> None:
        if self.fail_load:
            raise RuntimeError("load failed")
        self.loaded = True

    def unload(self) -> None:
        self.loaded = False

    @property
    def is_loaded(self) -> bool:
        return self.loaded

    def synthesize(self, text: str) -> "tuple[npt.NDArray[np.float32], int]":
        if self.fail_synth:
            raise RuntimeError("synth failed")
        return np.full(4, self.value, dtype=np.float32), 24000


class _Piper:
    """A fake always-ready fallback engine."""

    def __init__(self, fail: bool = False) -> None:
        self.fail, self.calls = fail, 0

    def synthesize(self, text: str) -> "tuple[npt.NDArray[np.float32], int]":
        self.calls += 1
        if self.fail:
            raise RuntimeError("piper failed")
        return np.full(4, 3.0, dtype=np.float32), 22050


def _chain(chain: tuple[str, ...], qwen: _Tts, kokoro: _Tts, piper: _Piper, events: "list[tuple[str, str | None]]") -> EngineChain:
    factories = {"qwen3_tts": lambda: qwen, "kokoro": lambda: kokoro}
    return EngineChain(EnginePlan(chain, "test"), factories, ModelManager(idle_timeout_seconds=9999.0), piper,
                       lambda failed, nxt: events.append((failed, nxt)))


def test_the_first_engine_is_used_when_it_works() -> None:
    """No downshift happens when the first engine succeeds."""
    events: list[tuple[str, str | None]] = []
    chain = _chain(("qwen3_tts", "kokoro", "piper"), _Tts(1.0), _Tts(2.0), _Piper(), events)
    audio, rate = chain.synthesize("hi")
    assert audio[0] == 1.0 and rate == 24000 and events == []


def test_a_failing_load_falls_to_the_next_engine_and_is_remembered() -> None:
    """After a failure the dead engine is skipped for the rest of the reading."""
    events: list[tuple[str, str | None]] = []
    qwen, kokoro = _Tts(1.0, fail_load=True), _Tts(2.0)
    chain = _chain(("qwen3_tts", "kokoro", "piper"), qwen, kokoro, _Piper(), events)
    assert chain.synthesize("a")[0][0] == 2.0
    assert chain.synthesize("b")[0][0] == 2.0
    assert events == [("qwen3_tts", "kokoro")]


def test_a_failing_synthesis_also_falls_down_the_chain() -> None:
    """A synthesis error is handled like a load error."""
    events: list[tuple[str, str | None]] = []
    chain = _chain(("qwen3_tts", "kokoro", "piper"), _Tts(1.0, fail_synth=True), _Tts(2.0), _Piper(), events)
    assert chain.synthesize("a")[0][0] == 2.0
    assert events == [("qwen3_tts", "kokoro")]


def test_when_every_managed_engine_fails_piper_speaks() -> None:
    """Piper is the last resort."""
    events: list[tuple[str, str | None]] = []
    piper = _Piper()
    chain = _chain(("qwen3_tts", "kokoro", "piper"), _Tts(1.0, fail_load=True), _Tts(2.0, fail_load=True), piper, events)
    assert chain.synthesize("a")[0][0] == 3.0
    assert events == [("qwen3_tts", "kokoro"), ("kokoro", "piper")]
    assert piper.calls == 1


def test_a_piper_first_plan_never_touches_the_model_manager() -> None:
    """Choosing Piper (no graphics card) loads no managed model at all."""
    manager = ModelManager(idle_timeout_seconds=9999.0)
    piper = _Piper()
    chain = EngineChain(EnginePlan(("piper", "kokoro"), "t"), {"kokoro": lambda: _Tts(2.0)}, manager, piper, lambda *_: None)
    assert chain.synthesize("a")[0][0] == 3.0
    assert manager.current_engine_id is None


def test_a_forced_engine_with_no_factory_is_skipped_not_fatal() -> None:
    """A missing engine (e.g. no model installed) falls down the chain."""
    events: list[tuple[str, str | None]] = []
    chain = EngineChain(EnginePlan(("kokoro", "piper"), "t"), {}, ModelManager(idle_timeout_seconds=9999.0),
                        _Piper(), lambda failed, nxt: events.append((failed, nxt)))
    assert chain.synthesize("a")[0][0] == 3.0
    assert events == [("kokoro", "piper")]


def test_the_old_single_engine_plan_still_ends_on_piper() -> None:
    """A chain with only Qwen3-TTS falls back to Piper and reports the downshift (legacy behaviour)."""
    events: list[tuple[str, str | None]] = []
    chain = _chain(("qwen3_tts",), _Tts(1.0, fail_load=True), _Tts(2.0), _Piper(), events)
    assert chain.synthesize("a")[0][0] == 3.0
    assert events == [("qwen3_tts", None)]
    assert chain.synthesize("b")[0][0] == 3.0
    assert events == [("qwen3_tts", None)]


def test_when_everything_including_piper_fails_the_error_surfaces() -> None:
    """If nothing can speak, the caller gets an exception to turn into a notification."""
    chain = _chain(("qwen3_tts", "piper"), _Tts(1.0, fail_load=True), _Tts(2.0), _Piper(fail=True), [])
    with pytest.raises(RuntimeError, match="piper failed"):
        chain.synthesize("a")
```

- [ ] **Step 2: Implement.**

`src/vocalis/engines/base.py`:

```python
"""The shared shapes of a text-to-speech engine."""
from typing import Protocol

import numpy as np
import numpy.typing as npt

from vocalis.model_manager import ModelEngine


class Synthesizer(Protocol):
    """Anything that can turn text into (audio, sample_rate)."""

    def synthesize(self, text: str) -> "tuple[npt.NDArray[np.float32], int]":
        """Speak `text`; returns mono float32 audio and its sample rate."""
        ...


class TTSEngine(ModelEngine, Synthesizer, Protocol):
    """A text-to-speech engine that ModelManager can load and unload."""
```

`src/vocalis/engines/registry.py`:

```python
"""Chooses, for each reading, the ordered list of voice engines to try."""
import logging
from dataclasses import dataclass
from typing import Callable

from vocalis.engines.auto_select import KOKORO, PIPER, QWEN3_TTS, AutoPolicy, decide, kokoro_should_use_gpu
from vocalis.hardware import HardwareInfo

logger = logging.getLogger("vocalis")

FORCED_CHAINS: dict[str, tuple[str, ...]] = {
    QWEN3_TTS: (QWEN3_TTS, KOKORO, PIPER),
    KOKORO: (KOKORO, PIPER),
    PIPER: (PIPER, KOKORO),
}
_LABELS = {QWEN3_TTS: "Qwen3-TTS", KOKORO: "Kokoro", PIPER: "Piper"}


@dataclass(frozen=True)
class EnginePlan:
    """Engines to try, first to last, and why (plain English, for the log)."""

    chain: tuple[str, ...]
    reason: str


class EngineSelector:
    """Turns the user's choice plus the PC's current state into an EnginePlan."""

    def __init__(
        self,
        preference: Callable[[], str],
        hardware_probe: Callable[[], HardwareInfo],
        held_vram_mb: Callable[[], int] = lambda: 0,
        policy: AutoPolicy = AutoPolicy(),
    ) -> None:
        """Create a selector.

        Args:
            preference: Returns "auto" or a forced engine id (read live).
            hardware_probe: Returns a fresh HardwareInfo snapshot.
            held_vram_mb: Graphics memory Vocalis itself currently holds.
            policy: Thresholds for Auto.
        """
        self._preference = preference
        self._hardware_probe = hardware_probe
        self._held_vram_mb = held_vram_mb
        self._policy = policy

    def plan(self) -> EnginePlan:
        """The engines to try for the next reading."""
        choice = self._preference()
        if choice in FORCED_CHAINS:
            return EnginePlan(FORCED_CHAINS[choice], f"You chose {_LABELS[choice]} in Settings.")
        try:
            hardware = self._hardware_probe()
        except Exception:
            logger.exception("Could not check the PC for the Auto voice choice")
            return EnginePlan((KOKORO, PIPER), "Vocalis could not check your PC, so it is using the lighter voice.")
        decision = decide(hardware, self._policy, self._held_vram_mb())
        logger.info("Voice engine plan: %s (%s)", " -> ".join(decision.chain), decision.reason)
        return EnginePlan(decision.chain, decision.reason)

    def kokoro_use_gpu(self) -> bool:
        """Whether Kokoro should run on the graphics card right now."""
        try:
            return kokoro_should_use_gpu(self._hardware_probe(), self._policy)
        except Exception:
            logger.exception("Could not check the graphics card for Kokoro")
            return False
```

`src/vocalis/engines/chain.py`:

```python
"""Walks one reading's list of voice engines, falling down it on failure."""
import logging
from typing import Callable, Mapping, cast

import numpy as np
import numpy.typing as npt

from vocalis.engines.auto_select import PIPER
from vocalis.engines.base import Synthesizer, TTSEngine
from vocalis.engines.registry import EnginePlan
from vocalis.model_manager import ModelEngine, ModelManager

logger = logging.getLogger("vocalis")


class EngineChain:
    """Speaks text with the first working engine of a plan.

    Remembers which engines already failed, so later pieces of the same
    reading never retry a dead engine and the user is told only once.
    """

    def __init__(
        self,
        plan: EnginePlan,
        factories: Mapping[str, Callable[[], ModelEngine]],
        model_manager: ModelManager,
        fallback_engine: Synthesizer,
        on_downshift: Callable[[str, "str | None"], None],
    ) -> None:
        """Create a chain for one reading.

        Args:
            plan: The engines to try, in order.
            factories: Builds each managed engine (qwen3_tts, kokoro) by id.
            model_manager: Loads/unloads managed engines (one at a time).
            fallback_engine: The always-ready Piper engine, used for the
                "piper" step and as the very last resort.
            on_downshift: Called with (failed engine id, next id or None)
                each time an engine fails and the chain moves on.
        """
        self._plan = plan
        self._factories = factories
        self._model_manager = model_manager
        self._fallback_engine = fallback_engine
        self._on_downshift = on_downshift
        self._position = 0
        self._last_error: Exception | None = None

    def synthesize(self, text: str) -> "tuple[npt.NDArray[np.float32], int]":
        """Speak `text` with the first engine that works.

        Raises:
            Exception: The last engine's error if nothing could speak.
        """
        chain = self._plan.chain
        for index in range(self._position, len(chain)):
            engine_id = chain[index]
            try:
                return self._synthesize_with(engine_id, text)
            except Exception as exc:
                logger.exception("Voice engine %r failed", engine_id)
                self._last_error = exc
                self._position = index + 1
                self._on_downshift(engine_id, chain[index + 1] if index + 1 < len(chain) else None)
        if PIPER not in chain:
            return self._fallback_engine.synthesize(text)
        raise self._last_error if self._last_error is not None else RuntimeError("No voice engine is available")

    def _synthesize_with(self, engine_id: str, text: str) -> "tuple[npt.NDArray[np.float32], int]":
        """Speak with exactly one engine (Piper directly, others through ModelManager)."""
        if engine_id == PIPER:
            return self._fallback_engine.synthesize(text)
        factory = self._factories.get(engine_id)
        if factory is None:
            raise LookupError(f"No voice engine is set up for {engine_id!r}")
        engine = self._model_manager.acquire(engine_id, factory)
        try:
            return cast(TTSEngine, engine).synthesize(text)
        finally:
            self._model_manager.release(engine_id)
```

- [ ] **Step 3: Run once:** `.venv/bin/python -m pytest tests/test_engine_registry.py tests/test_engine_chain.py -q -p no:cacheprovider --no-cov`; then `.venv/bin/python -m mypy`. Fix failures.
- [ ] **Step 4: Commit** — `git add -A && git commit -m "feat: add the engine selector and the fall-down-the-chain runner"`

---

### Task 5: Quick Reader and Document Reader use the chain

**Files:**
- Modify: `src/vocalis/features/quick_reader.py`, `src/vocalis/features/doc_reader.py`
- Test: `tests/test_quick_reader.py`, `tests/test_doc_reader.py` (existing tests must keep passing unchanged; add the new ones below)

**Interfaces:**
- Consumes: Task 4 (`EngineChain`, `EngineSelector`, `EnginePlan`, `Synthesizer`).
- Produces: both feature constructors gain two OPTIONAL trailing parameters `engine_selector: EngineSelector | None = None` and `engine_factories: Mapping[str, Callable[[], ModelEngine]] | None = None`. With `engine_selector=None` behaviour is exactly the old Qwen3-TTS-then-Piper (existing tests rely on it). `feature._primary_engine_factory` stays (tests read it).

- [ ] **Step 1: Write the new tests.** Append to `tests/test_quick_reader.py` (reuse that file's existing `FakeSynthesizer`, `make_feature`-style helpers; build the feature directly if `make_feature` does not accept the new arguments — copy its construction and pass `engine_selector` / `engine_factories`):

```python
def test_a_selector_chooses_the_engine_for_each_reading() -> None:
    """With a selector, the planned engine is used and the primary factory is ignored."""
    from vocalis.engines.registry import EnginePlan, EngineSelector
    from vocalis.hardware import HardwareInfo

    kokoro = FakeSynthesizer()
    selector = EngineSelector(
        preference=lambda: "kokoro", hardware_probe=lambda: HardwareInfo(None, None, None, None, None)
    )
    feature, _player, _notifier = build_with_selector(  # a small local helper you add, mirroring make_feature
        selector=selector, factories={"kokoro": lambda: kokoro},
    )
    audio, rate = feature._synthesize_with_fallback("Hello.")  # noqa: SLF001
    assert kokoro.calls == ["Hello."]


def test_a_failing_planned_engine_notifies_the_user_and_uses_the_next() -> None:
    """The downshift text is unchanged: 'Primary voice unavailable — using backup voice'."""
    ...  # use a selector whose chain is kokoro->piper, a kokoro factory that raises on load,
         # a FakeSynthesizer fallback; assert the fallback spoke and the notifier saw the exact text
```

(The implementer completes the helper `build_with_selector` by copying the existing test construction in this file; the assertions above are the contract. Do the same in `tests/test_doc_reader.py`: one test that a selector-planned engine speaks, one that after a downshift the failed engine's factory is NOT called again for the next chunk and the notifier/status text `"Primary voice unavailable — using backup voice"` / `"Using backup voice"` appear exactly once.)

- [ ] **Step 2: Implement `quick_reader.py`.** Remove the local `class Synthesizer(Protocol)` block and the line `_ENGINE_ID = TTS_ENGINE_ID`; change the typing import to `from typing import Callable, Mapping`; add

```python
from vocalis.engines.base import Synthesizer
from vocalis.engines.chain import EngineChain
from vocalis.engines.registry import EnginePlan, EngineSelector
```

(`TTS_ENGINE_ID` is still imported from `vocalis.model_manager` and used in `_new_chain`.) Add the two optional constructor parameters after `speed_provider` (document them in the docstring), store `self._engine_selector = engine_selector` and `self._engine_factories = engine_factories`, and replace `_synthesize_with_fallback` with:

```python
    def _new_chain(self) -> EngineChain:
        """Build the engine chain for one reading: the selector's plan, or the old Qwen3-TTS-then-Piper."""
        if self._engine_selector is not None:
            plan = self._engine_selector.plan()
            factories = self._engine_factories or {}
        else:
            plan = EnginePlan((TTS_ENGINE_ID,), "Fixed to the built-in voice.")
            factories = {TTS_ENGINE_ID: self._primary_engine_factory}
        return EngineChain(plan, factories, self._model_manager, self._fallback_engine, self._notify_downshift)

    def _notify_downshift(self, failed_engine_id: str, next_engine_id: "str | None") -> None:
        """Tell the user the first-choice voice failed."""
        self._notifier.notify(_TITLE, "Primary voice unavailable — using backup voice")

    def _synthesize_with_fallback(self, text: str) -> "tuple[npt.NDArray[np.float32], int]":
        """Speak `text` with the planned engines, falling down the chain on failure."""
        return self._new_chain().synthesize(text)
```

- [ ] **Step 3: Implement `doc_reader.py`.** Same import changes (drop the local `Synthesizer` Protocol and `_ENGINE_ID`; keep `TTS_ENGINE_ID`; typing import becomes `Callable, Mapping`). Add the same two optional constructor parameters/attributes. Replace `self._use_fallback_for_rest = False` in `__init__` (line ~137) with `self._chain: EngineChain | None = None`, and in `_start_new_reading` (line ~285) replace `self._use_fallback_for_rest = False` with `self._chain = None`. Replace `_synthesize_with_fallback` with:

```python
    def _new_chain(self) -> EngineChain:
        """Build the engine chain for this reading (same rules as Quick Reader)."""
        if self._engine_selector is not None:
            plan = self._engine_selector.plan()
            factories = self._engine_factories or {}
        else:
            plan = EnginePlan((TTS_ENGINE_ID,), "Fixed to the built-in voice.")
            factories = {TTS_ENGINE_ID: self._primary_engine_factory}
        return EngineChain(plan, factories, self._model_manager, self._fallback_engine, self._notify_downshift)

    def _notify_downshift(self, failed_engine_id: str, next_engine_id: "str | None") -> None:
        """Tell the user, once per failure, that the first-choice voice failed."""
        self._notifier.notify(_TITLE, "Primary voice unavailable — using backup voice")
        self._on_status("Using backup voice")

    def _synthesize_with_fallback(self, text: str) -> "tuple[npt.NDArray[np.float32], int]":
        """Speak one piece; the same chain is reused for the whole document so a dead engine is not retried."""
        chain = self._chain
        if chain is None:
            chain = self._chain = self._new_chain()
        return chain.synthesize(text)
```

- [ ] **Step 4: Run once:** `.venv/bin/python -m pytest tests/test_quick_reader.py tests/test_doc_reader.py tests/test_shared_tts_engine.py tests/test_reading_hooks_integration.py -q -p no:cacheprovider --no-cov`, then the full suite once and mypy. All pre-existing tests must pass unchanged.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: Quick Reader and Document Reader choose their voice through the engine chain"`

---

### Task 6: Wire the selector into `cli.py` and keep tests off the real hardware

**Files:**
- Modify: `src/vocalis/cli.py` (imports, new helpers, `build_quick_reader`, `build_doc_reader`)
- Modify: `tests/test_cli.py`, `tests/test_cross_feature_integration.py` (autouse hardware patch + new tests)

**Interfaces:**
- Consumes: Tasks 1-5.
- Produces: `build_quick_reader(..., hardware_probe: Callable[[], HardwareInfo] | None = None)` and the same on `build_doc_reader`; module-level helpers `_make_engine_selector(preference, model_manager, hardware_probe) -> EngineSelector` and `_make_kokoro_factory(config, settings, selector) -> Callable[[], KokoroTTS]`; constant `_QWEN3_HELD_VRAM_MB = 4096`.

- [ ] **Step 1: Write the tests.** In BOTH `tests/test_cli.py` and `tests/test_cross_feature_integration.py` add an autouse fixture (next to the existing autouse fixtures) so no test ever probes the real GPU:

```python
@pytest.fixture(autouse=True)
def _strong_idle_pc() -> Iterator[None]:
    """Every builder sees a big, idle PC, so Auto picks Qwen3-TTS and no real hardware is read."""
    from vocalis.hardware import GpuInfo, HardwareInfo

    info = HardwareInfo(GpuInfo("Test GPU", 8188, 7000), 16000, 10000, 8, 1.0)
    with patch("vocalis.cli.probe_hardware", return_value=info):
        yield
```

(Import `Iterator`, `pytest`, `patch` if the file lacks them.) Append to `tests/test_cli.py`:

```python
def test_both_readers_use_the_selector_and_default_to_auto(tmp_path: Path, qtbot) -> None:  # type: ignore[no-untyped-def]
    """Built readers carry an engine selector whose preference starts as Automatic."""
    settings = _live_settings(tmp_path)
    manager = ModelManager()
    with patch("vocalis.cli.list_hotkey_capable_devices", return_value=[]):
        quick, _ = build_quick_reader(
            config=settings.config, model_manager=manager, reading_coordinator=ReadingCoordinator(), settings=settings
        )
    assert quick._engine_selector is not None  # noqa: SLF001
    assert quick._engine_selector.plan().chain[0] == "qwen3_tts"  # noqa: SLF001
    assert set(quick._engine_factories or {}) == {"qwen3_tts", "kokoro"}  # noqa: SLF001


def test_choosing_kokoro_in_settings_changes_the_next_plan(tmp_path: Path) -> None:
    """The selector reads the live Settings choice, so a change applies to the next reading."""
    settings = _live_settings(tmp_path)
    with patch("vocalis.cli.list_hotkey_capable_devices", return_value=[]):
        quick, _ = build_quick_reader(
            config=settings.config, model_manager=ModelManager(),
            reading_coordinator=ReadingCoordinator(), settings=settings,
        )
    settings.update(settings.config.model_copy(update={
        "engines": settings.config.engines.model_copy(update={"quick_reader_engine": "kokoro"})}))
    assert quick._engine_selector.plan().chain == ("kokoro", "piper")  # type: ignore[union-attr]  # noqa: SLF001


def test_the_kokoro_factory_builds_a_kokoro_engine_with_live_voice(tmp_path: Path) -> None:
    """The Kokoro factory reads the voice live and decides graphics-card use at load time."""
    from vocalis.cli import _make_kokoro_factory
    from vocalis.engines.kokoro_tts import KokoroTTS

    settings = _live_settings(tmp_path)
    selector = type("S", (), {"kokoro_use_gpu": staticmethod(lambda: True)})()
    engine = _make_kokoro_factory(settings.config, settings, selector)()  # type: ignore[arg-type]
    assert isinstance(engine, KokoroTTS)
    assert engine._current_voice() == "af_heart"  # noqa: SLF001
```

Append to `tests/test_cross_feature_integration.py` (copy the patch list of the existing "switching between Quick Reader and Document Reader reuses the loaded engine" test and keep its fakes) a test named `test_auto_picks_the_same_engine_for_both_readers_so_nothing_reloads` asserting that after a Quick Reader reading then a Document Reader reading with both on Auto, `manager.current_engine_id == "qwen3_tts"` and the fake Qwen engine's `load_count == 1`.

- [ ] **Step 2: Implement in `cli.py`.** Add imports:

```python
from vocalis.engines.auto_select import QWEN3_TTS
from vocalis.engines.kokoro_tts import KokoroTTS
from vocalis.engines.registry import EngineSelector
from vocalis.hardware import HardwareInfo, log_hardware_summary, probe_hardware
```

(extend the existing `from vocalis.hardware import log_hardware_summary` line). Add near `_make_qwen_factory`:

```python
_QWEN3_HELD_VRAM_MB = 4096


def _make_engine_selector(
    preference: Callable[[], str],
    model_manager: ModelManager,
    hardware_probe: Callable[[], HardwareInfo],
) -> EngineSelector:
    """A selector that credits Qwen3-TTS's own graphics memory back while it is loaded."""

    def held_vram_mb() -> int:
        return _QWEN3_HELD_VRAM_MB if model_manager.current_engine_id == QWEN3_TTS else 0

    return EngineSelector(preference=preference, hardware_probe=hardware_probe, held_vram_mb=held_vram_mb)


def _make_kokoro_factory(
    config: VocalisConfig, settings: LiveSettings | None, selector: EngineSelector
) -> Callable[[], KokoroTTS]:
    """A factory building KokoroTTS; the voice is read live, graphics-card use is decided at load."""
    model_path = Path(config.engines.kokoro_model_path)
    voices_path = Path(config.engines.kokoro_voices_path)
    voice: "str | Callable[[], str]" = settings.kokoro_voice if settings is not None else config.engines.kokoro_voice

    def kokoro_factory() -> KokoroTTS:
        """Build the light voice engine."""
        return KokoroTTS(model_path=model_path, voices_path=voices_path, voice=voice, use_gpu=selector.kokoro_use_gpu)

    return kokoro_factory
```

In `build_quick_reader` add parameter `hardware_probe: Callable[[], HardwareInfo] | None = None` (document it), and after `primary_engine_factory = _make_qwen_factory(config, settings)`:

```python
    preference: Callable[[], str] = (
        settings.quick_reader_engine if settings is not None else (lambda: config.engines.quick_reader_engine)
    )
    selector = _make_engine_selector(
        preference, model_manager, hardware_probe if hardware_probe is not None else probe_hardware
    )
    engine_factories = {QWEN3_TTS: primary_engine_factory, "kokoro": _make_kokoro_factory(config, settings, selector)}
```

and pass `engine_selector=selector, engine_factories=engine_factories` into `QuickReaderFeature(...)`. Do the same in `build_doc_reader` using `settings.document_reader_engine` / `config.engines.document_reader_engine`, passing both into `DocumentReaderFeature(...)`. Update the two builders' docstrings.

- [ ] **Step 3: Run once:** full suite, then mypy. Every existing integration test must still pass thanks to the autouse patch.
- [ ] **Step 4: Commit** — `git add -A && git commit -m "feat: wire the Auto voice-engine selector into the readers"`

---

### Task 7: Settings window — engine pickers and Kokoro voice

**Files:**
- Modify: `src/vocalis/ui/settings_window.py`
- Test: `tests/test_settings_window.py`

**Interfaces:**
- Consumes: Task 1 (`EnginesConfig` fields, `KOKORO_VOICES`).
- Produces: widgets `_quick_engine_combo`, `_doc_engine_combo` (item text = label, item data = engine value) and `_kokoro_voice_combo`; saved into `engines.quick_reader_engine`, `engines.document_reader_engine`, `engines.kokoro_voice`.

- [ ] **Step 1: Write the tests** (append to `tests/test_settings_window.py`; reuse its `make` helper):

```python
def test_engine_pickers_default_to_automatic(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Both voice-engine drop-downs start on Automatic."""
    _settings, window, _applied, _path = make(qtbot, tmp_path)
    assert window._quick_engine_combo.currentData() == "auto"  # noqa: SLF001
    assert window._doc_engine_combo.currentData() == "auto"  # noqa: SLF001
    assert window._kokoro_voice_combo.currentText() == "af_heart"  # noqa: SLF001


def test_engine_pickers_show_the_saved_choices(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Saved engine choices and Kokoro voice are shown when the window opens."""
    from vocalis.config import EnginesConfig

    config = VocalisConfig(
        engines=EnginesConfig(quick_reader_engine="kokoro", document_reader_engine="piper", kokoro_voice="bf_emma")
    )
    _settings, window, _applied, _path = make(qtbot, tmp_path, config=config)
    assert window._quick_engine_combo.currentData() == "kokoro"  # noqa: SLF001
    assert window._doc_engine_combo.currentData() == "piper"  # noqa: SLF001
    assert window._kokoro_voice_combo.currentText() == "bf_emma"  # noqa: SLF001


def test_saving_writes_the_chosen_engines(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Choosing engines and saving updates the live settings."""
    settings, window, _applied, _path = make(qtbot, tmp_path)
    window._quick_engine_combo.setCurrentIndex(window._quick_engine_combo.findData("qwen3_tts"))  # noqa: SLF001
    window._doc_engine_combo.setCurrentIndex(window._doc_engine_combo.findData("kokoro"))  # noqa: SLF001
    window._kokoro_voice_combo.setCurrentText("am_adam")  # noqa: SLF001
    window._on_save_clicked()  # noqa: SLF001
    assert settings.quick_reader_engine() == "qwen3_tts"
    assert settings.document_reader_engine() == "kokoro"
    assert settings.kokoro_voice() == "am_adam"


def test_an_unlisted_stored_kokoro_voice_stays_selectable(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A voice typed into the config file by hand is kept, not silently replaced."""
    from vocalis.config import EnginesConfig

    config = VocalisConfig(engines=EnginesConfig(kokoro_voice="af_river"))
    _settings, window, _applied, _path = make(qtbot, tmp_path, config=config)
    assert window._kokoro_voice_combo.currentText() == "af_river"  # noqa: SLF001
```

- [ ] **Step 2: Implement.** In `settings_window.py` import `KOKORO_VOICES` from `vocalis.settings` and define near the other module constants:

```python
_ENGINE_CHOICES: tuple[tuple[str, str], ...] = (
    ("Automatic (recommended)", "auto"),
    ("Qwen3-TTS (best quality, needs a strong PC)", "qwen3_tts"),
    ("Kokoro (light and natural)", "kokoro"),
    ("Piper (lightest)", "piper"),
)
```

In `__init__`, next to the existing `_voice_combo` creation, create the three combos (`_quick_engine_combo`, `_doc_engine_combo` filled with `addItem(label, value)` for every pair in `_ENGINE_CHOICES`; `_kokoro_voice_combo` filled with `KOKORO_VOICES`). In `_build_layout` add rows right after the existing `Voice` row: `form.addRow("Quick Reader voice engine", self._quick_engine_combo)`, `form.addRow("Document Reader voice engine", self._doc_engine_combo)`, `form.addRow("Kokoro voice", self._kokoro_voice_combo)`. In `load_from_settings` add:

```python
        for combo, stored in (
            (self._quick_engine_combo, config.engines.quick_reader_engine),
            (self._doc_engine_combo, config.engines.document_reader_engine),
        ):
            index = combo.findData(stored)
            combo.setCurrentIndex(index if index >= 0 else 0)
        kokoro_voice = config.engines.kokoro_voice
        if self._kokoro_voice_combo.findText(kokoro_voice) < 0:
            self._kokoro_voice_combo.addItem(kokoro_voice)
        self._kokoro_voice_combo.setCurrentText(kokoro_voice)
```

and replace the `engines = config.engines.model_copy(update={"qwen3_tts_speaker": ...})` line in `_config_from_widgets` with:

```python
        engines = config.engines.model_copy(
            update={
                "qwen3_tts_speaker": self._voice_combo.currentText(),
                "quick_reader_engine": self._quick_engine_combo.currentData(),
                "document_reader_engine": self._doc_engine_combo.currentData(),
                "kokoro_voice": self._kokoro_voice_combo.currentText(),
            }
        )
```

- [ ] **Step 3: Run once:** `.venv/bin/python -m pytest tests/test_settings_window.py -q -p no:cacheprovider --no-cov`, then the full suite and mypy.
- [ ] **Step 4: Commit** — `git add -A && git commit -m "feat: add voice-engine pickers and the Kokoro voice to Settings"`

---

### Task 8: Installer, startup checks, and the onnxruntime safety

**Files:**
- Modify: `pyproject.toml` (dependencies), `src/vocalis/startup_checks.py`, `packaging/installer/lib/fetch_models.py`, `packaging/installer/lib/pyenv.sh`, `packaging/installer/install.sh`, `packaging/installer/uninstall.sh`
- Test: `tests/test_startup_checks.py`, `tests/test_installer_fetch_models.py`, `tests/installer/test_pyenv.bats`, `tests/installer/test_install.bats`, `tests/installer/test_uninstall.bats`

**Interfaces:**
- Produces: `startup_checks.KOKORO_MODEL_FILES = ("kokoro-v1.0.onnx", "voices-v1.0.bin")`; `check_kokoro_model(model_path: Path, voices_path: Path) -> str | None`; `check_onnx_gpu(has_gpu: Callable[[], bool], providers: Callable[[], list[str]]) -> str | None`; `fetch_models.ensure_kokoro(model_path, voices_path, opener=urllib.request.urlopen)`; `pyenv_install_kokoro <venv_dir>` (bash).

Why `--no-deps`: `kokoro-onnx` hard-requires plain `onnxruntime`; installing it normally next to the `onnxruntime-gpu` that Parakeet needs can silently remove CUDA from Parakeet. So `pyproject.toml` lists only Kokoro's small dependencies, and the installer installs `kokoro-onnx==0.4.7` itself with `--no-deps`.

- [ ] **Step 1: Write the tests.**

`tests/test_startup_checks.py` — add `check_kokoro_model` and `check_onnx_gpu` to the import list, add an autouse fixture that fakes the two real-machine probes so no test reads the real GPU:

```python
@pytest.fixture(autouse=True)
def _fake_gpu_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_startup_checks must never ask the real machine about its GPU or onnxruntime."""
    monkeypatch.setattr("vocalis.startup_checks._gpu_present", lambda: True)
    monkeypatch.setattr("vocalis.startup_checks._onnx_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
```

(`import pytest` if missing; if an autouse `_isolated_home`-style fixture already exists in this file keep it too.) Append:

```python
def test_kokoro_model_present_is_silent(tmp_path: Path) -> None:
    """Both Kokoro files present -> no message."""
    model, voices = tmp_path / "kokoro-v1.0.onnx", tmp_path / "voices-v1.0.bin"
    _touch(model)
    _touch(voices)
    assert check_kokoro_model(model, voices) is None


def test_kokoro_model_missing_explains_the_consequence(tmp_path: Path) -> None:
    """A missing file names the folder and says another voice is used."""
    message = check_kokoro_model(tmp_path / "a.onnx", tmp_path / "b.bin")
    assert message is not None and "Kokoro" in message and "another voice" in message


def test_onnx_gpu_warning_only_when_a_gpu_exists_but_cuda_is_missing() -> None:
    """Dictation silently losing the graphics card is reported; other combinations are silent."""
    assert check_onnx_gpu(lambda: True, lambda: ["CPUExecutionProvider"]) is not None
    assert check_onnx_gpu(lambda: True, lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]) is None
    assert check_onnx_gpu(lambda: False, lambda: ["CPUExecutionProvider"]) is None
    assert "dictation" in (check_onnx_gpu(lambda: True, lambda: []) or "").lower()
```

`tests/test_installer_fetch_models.py` — append (uses the file's existing `fetch_models` fixture):

```python
class _FakeResponse:
    """A fake urlopen() result that yields fixed bytes."""

    def __init__(self, data: bytes) -> None:
        import io
        self._buffer = io.BytesIO(data)

    def read(self, size: int = -1) -> bytes:
        return self._buffer.read(size)

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_exc: object) -> None:
        return None


def test_ensure_kokoro_downloads_both_files_and_checks_their_size(fetch_models, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Both files are downloaded to the configured paths and size-checked."""
    sizes = {"kokoro-v1.0.onnx": 10, "voices-v1.0.bin": 4}
    urls: list[str] = []

    def opener(url: str) -> _FakeResponse:
        urls.append(url)
        return _FakeResponse(b"x" * sizes[url.rsplit("/", 1)[1]])

    model, voices = tmp_path / "k" / "kokoro-v1.0.onnx", tmp_path / "k" / "voices-v1.0.bin"
    fetch_models.ensure_kokoro(model, voices, opener=opener, expected_sizes=sizes)
    assert model.read_bytes() == b"x" * 10 and voices.read_bytes() == b"x" * 4
    assert all("thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/" in url for url in urls)


def test_ensure_kokoro_skips_files_that_are_already_there(fetch_models, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Nothing is downloaded when both files already exist."""
    model, voices = tmp_path / "kokoro-v1.0.onnx", tmp_path / "voices-v1.0.bin"
    model.write_bytes(b"x")
    voices.write_bytes(b"x")

    def opener(url: str) -> _FakeResponse:
        raise AssertionError("must not download")

    fetch_models.ensure_kokoro(model, voices, opener=opener)


def test_ensure_kokoro_rejects_a_truncated_download_and_leaves_no_partial_file(fetch_models, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A short download raises and never leaves a half file that would look complete."""
    model, voices = tmp_path / "kokoro-v1.0.onnx", tmp_path / "voices-v1.0.bin"
    with pytest.raises(RuntimeError, match="kokoro-v1.0.onnx"):
        fetch_models.ensure_kokoro(
            model, voices, opener=lambda url: _FakeResponse(b"xx"),
            expected_sizes={"kokoro-v1.0.onnx": 10, "voices-v1.0.bin": 4},
        )
    assert not model.exists()
    assert list(tmp_path.glob("*.part")) == []
```

`tests/installer/test_pyenv.bats` — append:

```bash
@test "pyenv_install_kokoro installs kokoro-onnx without its dependencies" {
  pyenv_run_uv() { echo "CALLED: $*" >> "$CALL_LOG"; }
  pyenv_install_kokoro "$WORKDIR/venv"
  grep -q "pip install --python $WORKDIR/venv/bin/python --no-deps kokoro-onnx==0.4.7" "$CALL_LOG"
}
```

`tests/installer/test_install.bats` — in `setup()` add the stub `pyenv_install_kokoro() { echo "pyenv_install_kokoro: $*" >> "$CALL_LOG"; }` and append:

```bash
@test "main installs kokoro-onnx right after the Vocalis wheel" {
  SCRIPT_DIR="$WORKDIR" REPO_ROOT="$WORKDIR" HOME="$WORKDIR" run main
  [ "$status" -eq 0 ]
  wheel_line="$(grep -n "pyenv_install_vocalis" "$CALL_LOG" | cut -d: -f1)"
  kokoro_line="$(grep -n "pyenv_install_kokoro" "$CALL_LOG" | cut -d: -f1)"
  [ -n "$kokoro_line" ]
  [ "$wheel_line" -lt "$kokoro_line" ]
}
```

`tests/installer/test_uninstall.bats` — append (the venv python prints one fake path for every query, as in the neighbouring `--purge` tests; copy their fixture pattern):

```bash
@test "main --purge also removes the Kokoro model directory" {
  mkdir -p "$WORKDIR/.local/share/vocalis/venv/bin" "$WORKDIR/kokoro-dir"
  cat > "$WORKDIR/.local/share/vocalis/venv/bin/python" <<EOF
#!/usr/bin/env bash
echo "$WORKDIR/kokoro-dir/kokoro-v1.0.onnx"
EOF
  chmod +x "$WORKDIR/.local/share/vocalis/venv/bin/python"
  HOME="$WORKDIR" run main --purge
  [ "$status" -eq 0 ]
  [ ! -d "$WORKDIR/kokoro-dir" ]
}
```

- [ ] **Step 2: Implement.**
  - `pyproject.toml` `dependencies`: add `"colorlog>=6.9,<7"`, `"espeakng-loader>=0.2.4,<1"`, `"phonemizer-fork==3.3.1"` (NOT `kokoro-onnx`).
  - `startup_checks.py`: add `import importlib`, `KOKORO_MODEL_FILES`, and

```python
def check_kokoro_model(model_path: Path, voices_path: Path) -> str | None:
    """Whether the Kokoro voice files are present.

    Returns:
        None if both files exist; otherwise a message naming the folder.
    """
    if model_path.exists() and voices_path.exists():
        return None
    return (
        f"The light Kokoro voice is missing at {model_path.parent}. "
        "Vocalis will use another voice until it is installed."
    )


def _gpu_present() -> bool:
    """Whether a CUDA graphics card is available (asks PyTorch)."""
    from vocalis.hardware import query_gpu_with_torch

    return query_gpu_with_torch() is not None


def _onnx_providers() -> list[str]:
    """The execution providers the installed onnxruntime offers."""
    return list(importlib.import_module("onnxruntime").get_available_providers())


def check_onnx_gpu(has_gpu: Callable[[], bool], providers: Callable[[], list[str]]) -> str | None:
    """Warn if a graphics card exists but onnxruntime cannot use it.

    A CPU-only onnxruntime installed next to the GPU one silently removes
    the graphics card from dictation; this reports that instead of letting
    dictation just feel slow.
    """
    if has_gpu() and "CUDAExecutionProvider" not in providers():
        return (
            "Your graphics card is not being used for dictation, so dictation may be slow. "
            "Re-running the Vocalis installer usually fixes this."
        )
    return None
```

  and in `run_startup_checks` insert `lambda: check_kokoro_model(Path(config.engines.kokoro_model_path), Path(config.engines.kokoro_voices_path)),` right after the Qwen3-TTS check and `lambda: check_onnx_gpu(_gpu_present, _onnx_providers),` right after the capture-tools check and BEFORE the `.asoundrc` check (which must stay last); update the docstrings listing the order.
  - `fetch_models.py`: add `import urllib.request` and `from typing import Callable, Any`, constants and function:

```python
KOKORO_BASE_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
KOKORO_FILE_SIZES = {"kokoro-v1.0.onnx": 325532387, "voices-v1.0.bin": 28214398}
_CHUNK_BYTES = 1024 * 1024


def _download(url: str, target: Path, expected_size: int, opener: Callable[[str], Any]) -> None:
    """Download url to target via a .part file, checking the size before renaming into place."""
    part = target.with_name(target.name + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with opener(url) as response, part.open("wb") as handle:
            while True:
                block = response.read(_CHUNK_BYTES)
                if not block:
                    break
                handle.write(block)
        if part.stat().st_size != expected_size:
            raise RuntimeError(f"{target.name} downloaded incompletely (expected {expected_size} bytes)")
        part.replace(target)
    finally:
        part.unlink(missing_ok=True)


def ensure_kokoro(
    model_path: Path,
    voices_path: Path,
    opener: Callable[[str], Any] = urllib.request.urlopen,
    expected_sizes: dict[str, int] = KOKORO_FILE_SIZES,
) -> None:
    """Download the Kokoro model and voices files, unless both are already present."""
    if model_path.exists() and voices_path.exists():
        print(f"Kokoro voice already present at {model_path.parent}")
        return
    print(f"Downloading the Kokoro voice to {model_path.parent} (about 350MB) ...")
    for path in (model_path, voices_path):
        if not path.exists():
            _download(KOKORO_BASE_URL + path.name, path, expected_sizes[path.name], opener)
```

  and in `main()` add `ensure_kokoro(Path(config.engines.kokoro_model_path), Path(config.engines.kokoro_voices_path))`; update the module docstring ("four" assets).
  - `pyenv.sh`: append

```bash
# Install kokoro-onnx WITHOUT its dependencies. It hard-requires plain
# onnxruntime, which would silently remove the graphics card from
# Parakeet if installed next to onnxruntime-gpu; its other dependencies
# are listed in Vocalis's own pyproject.toml.
pyenv_install_kokoro() {
  local venv_dir="$1"
  pyenv_run_uv pip install --python "${venv_dir}/bin/python" --no-deps "kokoro-onnx==0.4.7"
}
```

  - `install.sh`: directly after `pyenv_install_vocalis "$venv_dir" "$wheel_path"` add `pyenv_install_kokoro "$venv_dir"`; also add a `kokoro_model` query beside the other four (`kokoro_model="$("$venv_python" -c 'from vocalis.config import VocalisConfig; print(VocalisConfig().engines.kokoro_model_path)')"`) and an echo `Checking Kokoro voice at ${kokoro_model} ...` before `fetch_models.py` runs.
  - `uninstall.sh`: next to the `piper_voice` lookups add `kokoro_model="$("$venv_python" -c 'from vocalis.config import VocalisConfig; print(VocalisConfig().engines.kokoro_model_path)' 2>/dev/null || true)"`, derive `kokoro_dir` with the same non-empty guard (`[[ -n "$kokoro_model" ]] && kokoro_dir="$(dirname "$kokoro_model")"`), and in the purge block `[[ -n "$kokoro_dir" && -d "$kokoro_dir" ]] && rm -rf "$kokoro_dir"`; update the purge message and usage text to mention the Kokoro voice. Declare `kokoro_model="" kokoro_dir=""` with the other locals.

- [ ] **Step 3: Run once:** `.venv/bin/python -m pytest -q -p no:cacheprovider`, `.venv/bin/python -m mypy`, `shellcheck packaging/installer/*.sh packaging/installer/lib/*.sh`, `bats tests/installer/`. Fix failures.
- [ ] **Step 4: Commit** — `git add -A && git commit -m "feat: install the Kokoro voice safely and check it at startup"`

---

### Task 9: Docs and changelog

**Files:** `docs/user-docs.md`, `docs/architecture.md`, `docs/file-purpose.md`, `docs/folder-structure.md`, `docs/workflow.md`, `docs/developer-docs.md`, `packaging/installer/README.md`, `README.md` (only if it lists voices), `.claude/2026-10-16_<HHMM>_CHANGELOG.md` (use `date +%H%M`; copy the section layout of the newest existing changelog).

- [ ] **Step 1: Write the docs** (read each file first; match its style):
  - `user-docs.md`: new section "Voice engines" explaining Automatic / Qwen3-TTS / Kokoro / Piper in plain English, the Auto rule (strong idle PC -> best-quality voice; busy PC or small/used graphics card -> lighter voice; no graphics card -> lightest), the per-reader pickers and Kokoro voice in Settings, and what the "using backup voice" message means.
  - `architecture.md` + `workflow.md`: the selector -> plan -> chain flow, why `--no-deps` for kokoro-onnx, Piper is unmanaged.
  - `developer-docs.md`: "Adding a voice engine" checklist (implement `TTSEngine`, add id to `auto_select`/`registry`, factory in `cli.py`, config field, installer download, startup check).
  - `file-purpose.md` / `folder-structure.md`: rows for `engines/auto_select.py`, `engines/base.py`, `engines/chain.py`, `engines/registry.py`, `engines/kokoro_tts.py` and their tests; update the `config.py`, `settings.py`, `startup_checks.py`, `fetch_models.py`, `pyenv.sh`, `uninstall.sh`, `quick_reader.py`, `doc_reader.py` rows.
  - `packaging/installer/README.md`: Kokoro files (~350 MB) and the `--no-deps` reason.
  - Changelog: what changed, files, verification numbers, deferred items (native speed = 6c, gapless = 6b, limits/hysteresis = 6d).
- [ ] **Step 2: Run the full gates once:** pytest, mypy, shellcheck, bats.
- [ ] **Step 3: Commit** — `git add -A && git commit -m "docs: document the voice engines and the Auto choice (Stage 6a)"`

---

## Self-review notes

- **Spec coverage (6a):** TTSEngine protocol replacing duplicated `Synthesizer` (T4, T5); registry factory + `Literal` validation (T1, T4); per-feature engine id with "same engine, no reload" preserved (T5, T6 integration test); Kokoro engine via `kokoro-onnx` with the onnxruntime risk handled (T3, T8); pure auto-selection from `HardwareInfo` plus Vocalis's own VRAM (T2, T4, T6); Settings picker (T7); startup checks and `default_config.toml` sync (T1, T8); installer download + purge (T8); fall-down-the-chain with notification (T4, T5). Deliberately NOT here: hysteresis and limits (6d), native speed (6c), gapless queue (6b).
- **Rulings:** (1) `TTS_ENGINE_ID` kept as Qwen3's id instead of being removed (tests import it; renaming adds churn only). (2) New config key names `quick_reader_engine` / `document_reader_engine` instead of reusing the dead `quick_tts` / `doc_tts`, so existing user config files (which hold `"qwen3_tts"` there from the old default) become "Auto" instead of accidentally forcing Qwen3-TTS. (3) Piper stays outside `ModelManager`, as today. (4) Kokoro speed is fixed at 1.0 until 6c so the existing `paced_audio` step is not applied twice.
