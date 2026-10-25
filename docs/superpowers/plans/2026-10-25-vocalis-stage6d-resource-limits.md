# Vocalis Stage 6d — Resource Limits Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the user control over how much CPU, RAM and GPU memory Vocalis is allowed to use, and make the existing "Auto" voice-engine choice hold steady for at least 60 seconds after a downshift instead of flip-flopping.

**Architecture:** Three new/changed live settings (`max_cpu_threads`, `max_ram_mb`, and finally-enforced `gpu_vram_ceiling_mb`) flow from Settings → `LiveSettings` → each engine's constructor, exactly like the existing `speaking_speed` live-value pattern (a plain value or a zero-arg callable, resolved fresh at load/synthesis time). `max_cpu_threads` is applied via `torch.set_num_threads`/ONNX `intra_op_num_threads` for the two managed engines, and via best-effort `OMP_NUM_THREADS`/`ORT_NUM_THREADS` environment variables for Piper's subprocess (its installed binary has no thread-count flag at all — verified against its real `--help` output). `gpu_vram_ceiling_mb` is applied via `torch.cuda.set_per_process_memory_fraction` and ONNX's CUDA `gpu_mem_limit` provider option. `max_ram_mb` is deliberately a soft/advisory signal into the existing Auto-selection stress check, not an OS-level hard cap. A small amount of new state inside `EngineSelector` remembers the last downshift's tier and timestamp so a reading right after a downshift doesn't immediately try to climb back up.

**Tech Stack:** Python 3.13, pydantic (config), PySide6 (Settings UI), torch, onnxruntime, pytest/pytest-qt, mypy strict.

**Spec:** `docs/superpowers/specs/2026-09-11-vocalis-design.md`, section "6d — Resource limits" (committed at `e6e1a9c`).

## Global Constraints

- All three settings are optional-in-spirit, represented in `VocalisConfig` as plain `int` fields with a `0 = no limit` sentinel (NOT `int | None`) — `tomli_w` cannot serialize Python `None` into TOML, and the rest of the config model has no precedent for an optional field, so the sentinel keeps saving/loading trivial. `LiveSettings` getters translate the sentinel back to `int | None` at the boundary — nothing past `LiveSettings` ever sees the `0`.
- `gpu_vram_ceiling_mb` already exists as `Field(default=6144, gt=0)` — it has no "no limit" state and keeps that exact shape; only `max_cpu_threads`/`max_ram_mb` are new `0`-sentinel fields.
- Every engine-facing value follows the codebase's existing "live value" convention: a constructor parameter typed `X | Callable[[], X]`, resolved with `self._value() if callable(self._value) else self._value` at the moment it's needed (load time for the two managed engines, synthesis time for Piper, since Piper has no persistent load).
- Piper's CPU thread cap is explicitly best-effort (`OMP_NUM_THREADS`/`ORT_NUM_THREADS` env vars) — the installed `piper` binary's `--help` output has no thread flag. Docstrings/docs must say "best-effort", not imply a guarantee.
- `max_ram_mb` never raises `MemoryError` or calls `resource.setrlimit` — it only feeds `auto_select.is_stressed()` so a high-RAM moment downshifts the voice engine instead of crashing the process.
- `EngineSelector`'s new hysteresis state only applies to the "Auto" branch of `plan()` — a user-forced engine choice (`FORCED_CHAINS`) bypasses it exactly as it does today.
- Every new/changed dataclass field is added last, with a default, so every existing positional construction of `HardwareInfo` across the test suite keeps working unchanged.
- Coverage stays ≥85%, mypy strict stays clean, zero test warnings — the project's standing gates.

## Review Focus

- A user sets `max_cpu_threads` to a value larger than the machine's actual core count — Vocalis should pass it straight through to `torch.set_num_threads`/ONNX without clamping or erroring (an oversized request is harmless; both libraries already tolerate it).
- A user sets `gpu_vram_ceiling_mb` larger than the GPU's actual total memory — `Qwen3TTS._apply_vram_ceiling()` must clamp the computed fraction to `1.0` rather than pass a fraction above 1.0 to `torch.cuda.set_per_process_memory_fraction` (which raises).
- `torch.cuda.mem_get_info(0)` or `set_per_process_memory_fraction` raising (no GPU, driver hiccup) must not abort `Qwen3TTS.load()` — the ceiling is best-effort and the model must still load.
- The engine-selector hysteresis must not get stuck holding the lighter voice forever — after the hold window elapses, the very next `plan()` call must re-evaluate fully and be able to climb back to the best tier the hardware actually supports.
- A reading right after Vocalis starts (no prior decision yet) must not be treated as a "downshift" by the hysteresis logic — the very first `plan()` call for a fresh `EngineSelector` must return hardware's true first choice, never held down.

---

### Task 1: Config fields and LiveSettings getters

**Files:**
- Modify: `src/vocalis/config.py` (add `max_cpu_threads`, `max_ram_mb` to `VocalisConfig`)
- Modify: `src/vocalis/settings.py` (add `max_cpu_threads()`, `max_ram_mb()`, `gpu_vram_ceiling_mb()` getters to `LiveSettings`)
- Modify: `config/default_config.toml` (document the two new keys)
- Test: `tests/test_config.py`
- Test: `tests/test_settings.py`

**Interfaces:**
- Produces: `VocalisConfig.max_cpu_threads: int` (default `0`), `VocalisConfig.max_ram_mb: int` (default `0`); `LiveSettings.max_cpu_threads() -> int | None`, `LiveSettings.max_ram_mb() -> int | None`, `LiveSettings.gpu_vram_ceiling_mb() -> int`. `0` in config means "no limit", translated to `None` by the two `LiveSettings` getters.

- [ ] **Step 1: Write the failing config tests**

Add to `tests/test_config.py`:

```python
def test_max_cpu_threads_and_max_ram_default_to_no_limit() -> None:
    config = VocalisConfig()
    assert config.max_cpu_threads == 0
    assert config.max_ram_mb == 0


def test_max_cpu_threads_and_max_ram_accept_a_positive_value() -> None:
    config = VocalisConfig(max_cpu_threads=4, max_ram_mb=4096)
    assert config.max_cpu_threads == 4
    assert config.max_ram_mb == 4096


def test_max_cpu_threads_rejects_a_negative_value() -> None:
    with pytest.raises(ValidationError):
        VocalisConfig(max_cpu_threads=-1)


def test_max_ram_mb_rejects_a_negative_value() -> None:
    with pytest.raises(ValidationError):
        VocalisConfig(max_ram_mb=-1)
```

(`ValidationError` is already imported in `tests/test_config.py` for other fields — check the top of the file; if not, add `from pydantic import ValidationError`.)

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_config.py -k max_cpu_threads -v`
Expected: FAIL — `AttributeError: 'VocalisConfig' object has no attribute 'max_cpu_threads'`.

- [ ] **Step 3: Add the fields to `VocalisConfig`**

In `src/vocalis/config.py`, inside `class VocalisConfig`, right after the existing `gpu_vram_ceiling_mb` field:

```python
    gpu_vram_ceiling_mb: int = Field(default=6144, gt=0)
    max_cpu_threads: int = Field(default=0, ge=0)
    max_ram_mb: int = Field(default=0, ge=0)
```

Add a one-line docstring note to the class's existing docstring: `max_cpu_threads` and `max_ram_mb` are `0` when unset ("no limit"); see `LiveSettings.max_cpu_threads()`/`max_ram_mb()` for the resolved `int | None` form other code should use.

- [ ] **Step 4: Run the config tests again**

Run: `PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_config.py -v`
Expected: PASS, including the pre-existing `test_default_toml_matches_model_defaults` (the toml has no `max_cpu_threads`/`max_ram_mb` keys yet, so `load_config()` falls back to the model defaults of `0`, matching `VocalisConfig()`).

- [ ] **Step 5: Document the new keys in `config/default_config.toml`**

Add, right after the existing `gpu_vram_ceiling_mb = 6144` line:

```toml
gpu_vram_ceiling_mb = 6144
# 0 means no limit.
max_cpu_threads = 0
max_ram_mb = 0
```

Run `PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_config.py::test_default_toml_matches_model_defaults -v` to confirm this still matches (it must, since `0` is the model default).

- [ ] **Step 6: Write the failing LiveSettings tests**

Add to `tests/test_settings.py` (follow the existing `speaking_speed`/`notifications_enabled` test style — find a helper that builds a `LiveSettings` from a `VocalisConfig` and reuse it):

```python
def test_max_cpu_threads_returns_none_when_unset() -> None:
    settings = _settings(VocalisConfig(max_cpu_threads=0))
    assert settings.max_cpu_threads() is None


def test_max_cpu_threads_returns_the_configured_value() -> None:
    settings = _settings(VocalisConfig(max_cpu_threads=4))
    assert settings.max_cpu_threads() == 4


def test_max_ram_mb_returns_none_when_unset() -> None:
    settings = _settings(VocalisConfig(max_ram_mb=0))
    assert settings.max_ram_mb() is None


def test_max_ram_mb_returns_the_configured_value() -> None:
    settings = _settings(VocalisConfig(max_ram_mb=2048))
    assert settings.max_ram_mb() == 2048


def test_gpu_vram_ceiling_mb_returns_the_configured_value() -> None:
    settings = _settings(VocalisConfig(gpu_vram_ceiling_mb=3000))
    assert settings.gpu_vram_ceiling_mb() == 3000
```

Replace `_settings(...)` with whatever helper `tests/test_settings.py` already uses to build a `LiveSettings` (read the file first — do not guess its name).

- [ ] **Step 7: Run the new LiveSettings tests to verify they fail**

Run: `PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_settings.py -k "max_cpu_threads or max_ram_mb or gpu_vram_ceiling" -v`
Expected: FAIL — `AttributeError: 'LiveSettings' object has no attribute 'max_cpu_threads'`.

- [ ] **Step 8: Add the getters to `LiveSettings`**

In `src/vocalis/settings.py`, right after `speaking_speed()`:

```python
    def max_cpu_threads(self) -> "int | None":
        """The CPU thread cap to use for the next load, or None for no cap."""
        value = self.config.max_cpu_threads
        return value if value > 0 else None

    def max_ram_mb(self) -> "int | None":
        """The advisory RAM ceiling used by Auto's stress check, or None for no cap."""
        value = self.config.max_ram_mb
        return value if value > 0 else None

    def gpu_vram_ceiling_mb(self) -> int:
        """The GPU VRAM ceiling to use for the next load."""
        return self.config.gpu_vram_ceiling_mb
```

- [ ] **Step 9: Run the LiveSettings tests again**

Run: `PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_settings.py -v`
Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add src/vocalis/config.py src/vocalis/settings.py config/default_config.toml tests/test_config.py tests/test_settings.py
git commit -m "feat: add max_cpu_threads and max_ram_mb settings (0 = no limit)"
```

---

### Task 2: Hardware probe — Vocalis's own RAM use

**Files:**
- Modify: `src/vocalis/hardware.py`
- Test: `tests/test_hardware.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `HardwareInfo.own_process_ram_mb: int | None` (new last field, default `None`); `query_own_process_ram_mb() -> int | None`; `probe_hardware(..., own_ram_query: Callable[[], int | None] = query_own_process_ram_mb)`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_hardware.py`:

```python
def test_reads_own_process_ram(tmp_path: Path) -> None:
    m, l = _files(tmp_path)
    info = probe_hardware(
        m, l, cpu_count=lambda: 8, gpu_query=lambda: GpuInfo("RTX", 8188, 7000), own_ram_query=lambda: 512,
    )
    assert info.own_process_ram_mb == 512


def test_a_raising_own_ram_query_means_unknown(tmp_path: Path) -> None:
    m, l = _files(tmp_path)

    def boom() -> "int | None":
        raise RuntimeError("resource module unavailable")

    info = probe_hardware(m, l, cpu_count=lambda: 4, gpu_query=lambda: None, own_ram_query=boom)
    assert info.own_process_ram_mb is None


def test_query_own_process_ram_mb_returns_a_positive_number() -> None:
    from vocalis.hardware import query_own_process_ram_mb

    value = query_own_process_ram_mb()
    assert value is None or value > 0
```

Update the two existing calls that construct the *expected* `HardwareInfo` by hand (lines around `test_reads_ram_load_cpu_and_gpu` and `test_missing_proc_files_give_none_not_an_error`) so they stay deterministic once `probe_hardware()` has a real `own_ram_query` default — pass `own_ram_query=lambda: None` explicitly to both `probe_hardware(...)` calls that currently omit it:

```python
def test_reads_ram_load_cpu_and_gpu(tmp_path: Path) -> None:
    m, l = _files(tmp_path)
    info = probe_hardware(
        m, l, cpu_count=lambda: 8, gpu_query=lambda: GpuInfo("RTX", 8188, 7000), own_ram_query=lambda: None,
    )
    assert info == HardwareInfo(GpuInfo("RTX", 8188, 7000), 15625, 7812, 8, 1.5, None)


def test_missing_proc_files_give_none_not_an_error(tmp_path: Path) -> None:
    info = probe_hardware(
        tmp_path / "nope", tmp_path / "nope2", cpu_count=lambda: None, gpu_query=lambda: None, own_ram_query=lambda: None,
    )
    assert info == HardwareInfo(None, None, None, None, None, None)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_hardware.py -v`
Expected: FAIL — `TypeError: probe_hardware() got an unexpected keyword argument 'own_ram_query'`.

- [ ] **Step 3: Add the field and the query function**

In `src/vocalis/hardware.py`, add `own_process_ram_mb` as the new last field of `HardwareInfo`:

```python
@dataclass(frozen=True)
class HardwareInfo:
    """A snapshot of the machine; a field is None when it could not be read."""

    gpu: GpuInfo | None
    ram_total_mb: int | None
    ram_available_mb: int | None
    cpu_count: int | None
    load_1min: float | None
    own_process_ram_mb: int | None = None
```

Add the query function, right after `query_gpu_with_torch`:

```python
def query_own_process_ram_mb() -> "int | None":
    """Return Vocalis's own current resident memory in MB, or None if unreadable."""
    import resource  # slow-ish, POSIX-only import, kept local

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // _KB_PER_MB
```

- [ ] **Step 4: Wire it into `probe_hardware`**

Replace `probe_hardware`'s signature and body:

```python
def probe_hardware(
    meminfo_path: Path = Path("/proc/meminfo"),
    loadavg_path: Path = Path("/proc/loadavg"),
    cpu_count: Callable[[], int | None] = os.cpu_count,
    gpu_query: Callable[[], GpuInfo | None] | None = None,
    own_ram_query: Callable[[], "int | None"] = query_own_process_ram_mb,
) -> HardwareInfo:
    """Take a snapshot of GPU, RAM, CPU and Vocalis's own memory use; unreadable parts become None."""
    query = gpu_query if gpu_query is not None else query_gpu_with_torch
    try:
        gpu = query()
    except Exception:
        logger.debug("GPU query failed; treating as no GPU", exc_info=True)
        gpu = None
    try:
        cpus = cpu_count()
    except Exception:
        logger.debug("CPU count failed; treating as unknown", exc_info=True)
        cpus = None
    try:
        own_ram = own_ram_query()
    except Exception:
        logger.debug("Own-process RAM query failed; treating as unknown", exc_info=True)
        own_ram = None
    ram_total, ram_available = _read_meminfo(meminfo_path)
    return HardwareInfo(gpu, ram_total, ram_available, cpus, _read_load(loadavg_path), own_ram)
```

- [ ] **Step 5: Run the hardware tests**

Run: `PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_hardware.py -v`
Expected: PASS.

- [ ] **Step 6: Run the full suite once to catch any other un-updated `HardwareInfo(...)`/`probe_hardware(...)` call**

Run: `PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_auto_select.py tests/test_engine_registry.py -v`
Expected: PASS unchanged — these construct `HardwareInfo` with 5 positional args, which still works because `own_process_ram_mb` defaults to `None`.

- [ ] **Step 7: Commit**

```bash
git add src/vocalis/hardware.py tests/test_hardware.py
git commit -m "feat: probe Vocalis's own resident memory use"
```

---

### Task 3: Auto-selection RAM-stress signal + downshift hysteresis

**Files:**
- Modify: `src/vocalis/engines/auto_select.py`
- Modify: `src/vocalis/engines/registry.py`
- Test: `tests/test_auto_select.py`
- Test: `tests/test_engine_registry.py`

**Interfaces:**
- Consumes: `HardwareInfo.own_process_ram_mb` (Task 2).
- Produces: `is_stressed(hw, policy, max_own_ram_mb=None) -> bool`; `decide(hw, policy=AutoPolicy(), held_vram_mb=0, max_own_ram_mb=None) -> AutoDecision`; `EngineSelector(..., max_own_ram_mb: Callable[[], int | None] = lambda: None, now: Callable[[], float] = time.monotonic, downshift_hold_seconds: float = 60.0)`. `EngineSelector.plan()` keeps its existing return type (`EnginePlan`).

- [ ] **Step 1: Write the failing auto_select tests**

Add `from dataclasses import replace` to the top of `tests/test_auto_select.py`'s imports, then add:

```python
def test_own_ram_use_over_the_cap_counts_as_stress() -> None:
    hw = replace(_hw(), own_process_ram_mb=3000)
    assert is_stressed(hw, AutoPolicy(), max_own_ram_mb=2048) is True


def test_own_ram_use_under_the_cap_is_not_stress() -> None:
    hw = replace(_hw(), own_process_ram_mb=1000)
    assert is_stressed(hw, AutoPolicy(), max_own_ram_mb=2048) is False


def test_no_cap_means_own_ram_use_is_never_stress() -> None:
    hw = replace(_hw(), own_process_ram_mb=999999)
    assert is_stressed(hw, AutoPolicy(), max_own_ram_mb=None) is False


def test_decide_downshifts_when_own_ram_is_over_the_cap() -> None:
    hw = replace(_hw(), own_process_ram_mb=3000)
    assert decide(hw, max_own_ram_mb=2048).chain[0] == KOKORO
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_auto_select.py -v`
Expected: FAIL — `TypeError: is_stressed() got an unexpected keyword argument 'max_own_ram_mb'`.

- [ ] **Step 3: Add the parameter to `is_stressed` and `decide`**

In `src/vocalis/engines/auto_select.py`:

```python
def is_stressed(hw: HardwareInfo, policy: AutoPolicy, max_own_ram_mb: "int | None" = None) -> bool:
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
    if max_own_ram_mb is not None and hw.own_process_ram_mb is not None:
        if hw.own_process_ram_mb >= max_own_ram_mb:
            return True
    return False


def decide(
    hw: HardwareInfo,
    policy: AutoPolicy = AutoPolicy(),
    held_vram_mb: int = 0,
    max_own_ram_mb: "int | None" = None,
) -> AutoDecision:
    """Pick the engine order for the current hardware snapshot.

    Args:
        hw: What the PC looks like right now.
        policy: The thresholds.
        held_vram_mb: Graphics memory Vocalis itself currently holds, added
            back to "free" so its own loaded model never counts as stress.
        max_own_ram_mb: The user's advisory RAM ceiling for Vocalis itself
            (None means no ceiling), checked against hw.own_process_ram_mb.
    """
    if hw.gpu is None:
        return AutoDecision((PIPER, KOKORO), "No graphics card was found, so Vocalis is using the lightest voice.")
    if hw.ram_available_mb is not None and hw.ram_available_mb < policy.critical_free_ram_mb:
        return AutoDecision((PIPER, KOKORO), "Your PC is very low on memory, so Vocalis is using the lightest voice.")
    if hw.gpu.total_mb <= policy.qwen3_min_gpu_total_mb:
        return AutoDecision((KOKORO, PIPER), "Your graphics card is small, so Vocalis is using the lighter voice.")
    if is_stressed(hw, policy, max_own_ram_mb):
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
```

- [ ] **Step 4: Run the auto_select tests**

Run: `PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_auto_select.py -v`
Expected: PASS.

- [ ] **Step 5: Write the failing registry (hysteresis) tests**

Add to `tests/test_engine_registry.py`:

```python
def test_first_ever_plan_is_never_held_down() -> None:
    """A fresh selector's first decision is the hardware's true choice."""
    assert _selector("auto", STRONG).plan().chain[0] == QWEN3_TTS


def test_a_downshift_is_held_for_the_hold_window() -> None:
    hw = {"value": STRONG}
    clock = {"value": 0.0}
    selector = EngineSelector(
        preference=lambda: "auto", hardware_probe=lambda: hw["value"], now=lambda: clock["value"],
    )
    assert selector.plan().chain[0] == QWEN3_TTS  # first call: strong PC

    hw["value"] = NO_GPU  # PC gets stressed
    clock["value"] = 1.0
    assert selector.plan().chain[0] == PIPER  # real downshift, recorded

    hw["value"] = STRONG  # PC recovers immediately
    clock["value"] = 2.0  # but only 1s after the downshift
    assert selector.plan().chain[0] == PIPER  # still held down


def test_the_hold_expires_after_the_window() -> None:
    hw = {"value": STRONG}
    clock = {"value": 0.0}
    selector = EngineSelector(
        preference=lambda: "auto",
        hardware_probe=lambda: hw["value"],
        now=lambda: clock["value"],
        downshift_hold_seconds=60.0,
    )
    assert selector.plan().chain[0] == QWEN3_TTS

    hw["value"] = NO_GPU
    clock["value"] = 1.0
    assert selector.plan().chain[0] == PIPER

    hw["value"] = STRONG
    clock["value"] = 61.1  # past the 60s hold
    assert selector.plan().chain[0] == QWEN3_TTS


def test_a_further_downshift_during_the_hold_resets_the_timer() -> None:
    hw = {"value": STRONG}
    clock = {"value": 0.0}
    selector = EngineSelector(
        preference=lambda: "auto", hardware_probe=lambda: hw["value"], now=lambda: clock["value"],
    )
    assert selector.plan().chain[0] == QWEN3_TTS

    medium = HardwareInfo(GpuInfo("Big", 4096, 4096), 16000, 10000, 8, 1.0)  # -> KOKORO
    hw["value"] = medium
    clock["value"] = 1.0
    assert selector.plan().chain[0] == KOKORO

    hw["value"] = NO_GPU  # a further, worse downshift resets the hold
    clock["value"] = 2.0
    assert selector.plan().chain[0] == PIPER

    hw["value"] = medium  # trying to climb only to the middle tier, 10s later
    clock["value"] = 12.0
    assert selector.plan().chain[0] == PIPER  # still held, timer restarted at t=2.0


def test_hysteresis_does_not_apply_to_a_forced_choice() -> None:
    hw = {"value": STRONG}
    clock = {"value": 0.0}
    selector = EngineSelector(
        preference=lambda: "piper", hardware_probe=lambda: hw["value"], now=lambda: clock["value"],
    )
    assert selector.plan().chain == (PIPER, KOKORO)
    hw["value"] = STRONG
    assert selector.plan().chain == (PIPER, KOKORO)  # forced choice, no hysteresis state touched


def test_max_own_ram_mb_is_called_on_every_plan() -> None:
    from unittest.mock import Mock

    ram_cap = Mock(return_value=None)
    selector = EngineSelector(preference=lambda: "auto", hardware_probe=lambda: STRONG, max_own_ram_mb=ram_cap)
    selector.plan()
    selector.plan()
    assert ram_cap.call_count == 2
```

- [ ] **Step 6: Run the registry tests to verify they fail**

Run: `PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_engine_registry.py -v`
Expected: FAIL — `TypeError: EngineSelector.__init__() got an unexpected keyword argument 'now'`.

- [ ] **Step 7: Implement hysteresis in `EngineSelector`**

Replace the whole of `src/vocalis/engines/registry.py` with:

```python
"""Chooses, for each reading, the ordered list of voice engines to try."""
import logging
import time
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
_TIER_RANK: dict[str, int] = {QWEN3_TTS: 0, KOKORO: 1, PIPER: 2}
_TIER_ENGINE = {rank: engine for engine, rank in _TIER_RANK.items()}
_DEFAULT_DOWNSHIFT_HOLD_SECONDS = 60.0


@dataclass(frozen=True)
class EnginePlan:
    """Engines to try, first to last, and why (plain English, for the log)."""

    chain: tuple[str, ...]
    reason: str


class EngineSelector:
    """Turns the user's choice plus the PC's current state into an EnginePlan.

    For "Auto", remembers the tier of the last downshift and, for
    `downshift_hold_seconds` after it, caps any better tier the hardware
    would now qualify for back down to that tier -- so a reading moments
    after a transient slowdown does not immediately try to climb back to
    the heavier voice. A forced engine choice bypasses this entirely.
    """

    def __init__(
        self,
        preference: Callable[[], str],
        hardware_probe: Callable[[], HardwareInfo],
        held_vram_mb: Callable[[], int] = lambda: 0,
        policy: AutoPolicy = AutoPolicy(),
        max_own_ram_mb: Callable[[], "int | None"] = lambda: None,
        now: Callable[[], float] = time.monotonic,
        downshift_hold_seconds: float = _DEFAULT_DOWNSHIFT_HOLD_SECONDS,
    ) -> None:
        """Create a selector.

        Args:
            preference: Returns "auto" or a forced engine id (read live).
            hardware_probe: Returns a fresh HardwareInfo snapshot.
            held_vram_mb: Graphics memory Vocalis itself currently holds.
            policy: Thresholds for Auto.
            max_own_ram_mb: The user's live advisory RAM ceiling (None = no cap).
            now: Monotonic clock, injectable for tests.
            downshift_hold_seconds: How long a downshift is held before Auto
                is allowed to try climbing back to a better tier.
        """
        self._preference = preference
        self._hardware_probe = hardware_probe
        self._held_vram_mb = held_vram_mb
        self._policy = policy
        self._max_own_ram_mb = max_own_ram_mb
        self._now = now
        self._downshift_hold_seconds = downshift_hold_seconds
        self._last_effective_tier: int | None = None
        self._last_downshift_time: float | None = None

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
        decision = decide(hardware, self._policy, self._held_vram_mb(), self._max_own_ram_mb())
        chain, reason = self._apply_hysteresis(decision.chain, decision.reason)
        logger.info("Voice engine plan: %s (%s)", " -> ".join(chain), reason)
        return EnginePlan(chain, reason)

    def _apply_hysteresis(self, chain: tuple[str, ...], reason: str) -> "tuple[tuple[str, ...], str]":
        """Hold Auto at its last downshifted tier for downshift_hold_seconds."""
        tier = _TIER_RANK[chain[0]]
        now = self._now()
        held_recently = (
            self._last_effective_tier is not None
            and self._last_downshift_time is not None
            and now - self._last_downshift_time < self._downshift_hold_seconds
        )
        if held_recently and tier < self._last_effective_tier:  # type: ignore[operator]
            held_engine = _TIER_ENGINE[self._last_effective_tier]  # type: ignore[index]
            return FORCED_CHAINS[held_engine], f"{reason} (holding the lighter voice a little longer after a recent slowdown.)"
        if self._last_effective_tier is not None and tier > self._last_effective_tier:
            self._last_downshift_time = now
        self._last_effective_tier = tier
        return chain, reason

    def kokoro_use_gpu(self) -> bool:
        """Whether Kokoro should run on the graphics card right now."""
        try:
            return kokoro_should_use_gpu(self._hardware_probe(), self._policy)
        except Exception:
            logger.exception("Could not check the graphics card for Kokoro")
            return False
```

- [ ] **Step 8: Run the registry tests again**

Run: `PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_engine_registry.py -v`
Expected: PASS.

- [ ] **Step 9: Run the full engine test files once more**

Run: `PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_auto_select.py tests/test_engine_registry.py tests/test_cli.py -v`
Expected: PASS (confirms Task 8's later cli.py changes are not needed yet for these to pass — `cli.py` still calls `EngineSelector(...)` with only its original 3 keyword args, which remain valid since the new params all have defaults).

- [ ] **Step 10: Commit**

```bash
git add src/vocalis/engines/auto_select.py src/vocalis/engines/registry.py tests/test_auto_select.py tests/test_engine_registry.py
git commit -m "feat: advisory RAM-stress signal and downshift hysteresis for Auto voice choice"
```

---

### Task 4: Qwen3-TTS CPU thread cap and GPU VRAM ceiling

**Files:**
- Modify: `src/vocalis/engines/qwen3_tts.py`
- Test: `tests/test_qwen3_tts.py`

**Interfaces:**
- Consumes: nothing new from earlier tasks (constructor params only).
- Produces: `Qwen3TTS(..., max_cpu_threads: "int | None | Callable[[], int | None]" = None, gpu_vram_ceiling_mb: "int | None | Callable[[], int | None]" = None)`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_qwen3_tts.py`:

```python
def test_load_applies_the_cpu_thread_cap() -> None:
    engine = Qwen3TTS(model_dir=Path("/fake/model/dir"), use_gpu=False, max_cpu_threads=4)
    with (
        patch("vocalis.engines.qwen3_tts.Qwen3TTSModel") as mock_model_cls,
        patch("vocalis.engines.qwen3_tts.torch.set_num_threads") as mock_set_threads,
    ):
        mock_model_cls.from_pretrained.return_value = object()
        engine.load()
    mock_set_threads.assert_called_once_with(4)


def test_load_does_not_touch_threads_when_no_cap() -> None:
    engine = Qwen3TTS(model_dir=Path("/fake/model/dir"), use_gpu=False)
    with (
        patch("vocalis.engines.qwen3_tts.Qwen3TTSModel") as mock_model_cls,
        patch("vocalis.engines.qwen3_tts.torch.set_num_threads") as mock_set_threads,
    ):
        mock_model_cls.from_pretrained.return_value = object()
        engine.load()
    mock_set_threads.assert_not_called()


def test_cpu_thread_cap_can_be_a_live_callable() -> None:
    cap = {"value": 2}
    engine = Qwen3TTS(model_dir=Path("/fake/model/dir"), use_gpu=False, max_cpu_threads=lambda: cap["value"])
    with (
        patch("vocalis.engines.qwen3_tts.Qwen3TTSModel") as mock_model_cls,
        patch("vocalis.engines.qwen3_tts.torch.set_num_threads") as mock_set_threads,
    ):
        mock_model_cls.from_pretrained.return_value = object()
        cap["value"] = 6
        engine.load()
    mock_set_threads.assert_called_once_with(6)


def test_load_applies_the_gpu_vram_ceiling_when_using_gpu() -> None:
    engine = Qwen3TTS(model_dir=Path("/fake/model/dir"), use_gpu=True, gpu_vram_ceiling_mb=4096)
    with (
        patch("vocalis.engines.qwen3_tts.Qwen3TTSModel") as mock_model_cls,
        patch("vocalis.engines.qwen3_tts.torch.cuda.mem_get_info", return_value=(2_000_000_000, 8 * 1024 * 1024 * 1024)),
        patch("vocalis.engines.qwen3_tts.torch.cuda.set_per_process_memory_fraction") as mock_set_fraction,
    ):
        mock_model_cls.from_pretrained.return_value = object()
        engine.load()
    mock_set_fraction.assert_called_once()
    assert abs(mock_set_fraction.call_args[0][0] - 0.5) < 0.001


def test_gpu_vram_ceiling_is_clamped_to_one_when_above_total_memory() -> None:
    engine = Qwen3TTS(model_dir=Path("/fake/model/dir"), use_gpu=True, gpu_vram_ceiling_mb=999999)
    with (
        patch("vocalis.engines.qwen3_tts.Qwen3TTSModel") as mock_model_cls,
        patch("vocalis.engines.qwen3_tts.torch.cuda.mem_get_info", return_value=(2_000_000_000, 8 * 1024 * 1024 * 1024)),
        patch("vocalis.engines.qwen3_tts.torch.cuda.set_per_process_memory_fraction") as mock_set_fraction,
    ):
        mock_model_cls.from_pretrained.return_value = object()
        engine.load()
    assert mock_set_fraction.call_args[0][0] == 1.0


def test_gpu_vram_ceiling_is_skipped_when_use_gpu_is_false() -> None:
    engine = Qwen3TTS(model_dir=Path("/fake/model/dir"), use_gpu=False, gpu_vram_ceiling_mb=4096)
    with (
        patch("vocalis.engines.qwen3_tts.Qwen3TTSModel") as mock_model_cls,
        patch("vocalis.engines.qwen3_tts.torch.cuda.set_per_process_memory_fraction") as mock_set_fraction,
    ):
        mock_model_cls.from_pretrained.return_value = object()
        engine.load()
    mock_set_fraction.assert_not_called()


def test_a_failing_vram_ceiling_query_does_not_block_loading() -> None:
    fake_model = FakeQwenModel(wavs=[np.zeros(1, dtype=np.float32)], sample_rate=24000)
    engine = Qwen3TTS(model_dir=Path("/fake/model/dir"), use_gpu=True, gpu_vram_ceiling_mb=4096)
    with (
        patch("vocalis.engines.qwen3_tts.Qwen3TTSModel") as mock_model_cls,
        patch("vocalis.engines.qwen3_tts.torch.cuda.mem_get_info", side_effect=RuntimeError("no CUDA")),
    ):
        mock_model_cls.from_pretrained.return_value = fake_model
        engine.load()
    assert engine.is_loaded is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_qwen3_tts.py -v`
Expected: FAIL — `TypeError: Qwen3TTS.__init__() got an unexpected keyword argument 'max_cpu_threads'`.

- [ ] **Step 3: Implement the two caps**

In `src/vocalis/engines/qwen3_tts.py`, add a constant and update `__init__`/`load`:

```python
_BYTES_PER_MB = 1024 * 1024
```

(place it next to the other `_STRETCH_*` constants)

```python
    def __init__(
        self,
        model_dir: Path,
        speaker: str | Callable[[], str] = "Ryan",
        language: str = "English",
        use_gpu: bool = True,
        speed: float | Callable[[], float] = 1.0,
        max_cpu_threads: "int | None | Callable[[], int | None]" = None,
        gpu_vram_ceiling_mb: "int | None | Callable[[], int | None]" = None,
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
            max_cpu_threads: Cap for torch.set_num_threads(), or a function
                returning it, re-read at load time. None means no cap.
            gpu_vram_ceiling_mb: Cap for torch.cuda.set_per_process_memory_fraction(),
                or a function returning it, re-read at load time. None means
                no cap; ignored when use_gpu is False. Best-effort: a failed
                GPU query never blocks loading.
        """
        self._model_dir = model_dir
        self._speaker = speaker
        self._language = language
        self._use_gpu = use_gpu
        self._speed = speed
        self._max_cpu_threads = max_cpu_threads
        self._gpu_vram_ceiling_mb = gpu_vram_ceiling_mb
        self._model: Qwen3TTSModel | None = None

    @property
    def is_loaded(self) -> bool:
        """Whether the model is currently loaded."""
        return self._model is not None

    def _current_max_cpu_threads(self) -> "int | None":
        """The CPU thread cap to apply right now (re-read at every load)."""
        return self._max_cpu_threads() if callable(self._max_cpu_threads) else self._max_cpu_threads

    def _current_gpu_vram_ceiling_mb(self) -> "int | None":
        """The GPU VRAM ceiling to apply right now (re-read at every load)."""
        return self._gpu_vram_ceiling_mb() if callable(self._gpu_vram_ceiling_mb) else self._gpu_vram_ceiling_mb

    def _apply_vram_ceiling(self) -> None:
        """Cap this process's GPU memory fraction. Best-effort: never raises."""
        ceiling_mb = self._current_gpu_vram_ceiling_mb()
        if ceiling_mb is None:
            return
        try:
            _, total_bytes = torch.cuda.mem_get_info(0)
            fraction = min(1.0, (ceiling_mb * _BYTES_PER_MB) / total_bytes)
            torch.cuda.set_per_process_memory_fraction(fraction)
        except Exception:
            logger.warning("Could not apply the GPU VRAM ceiling", exc_info=True)

    def load(self) -> None:
        """Load the model onto the GPU.

        Raises:
            Qwen3TTSLoadError: If loading fails for any reason (missing
                files, CUDA unavailable, out-of-memory, etc.).
        """
        max_threads = self._current_max_cpu_threads()
        if max_threads is not None:
            torch.set_num_threads(max_threads)
        if self._use_gpu:
            self._apply_vram_ceiling()
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
```

Leave the rest of the file (`unload`, `_current_speaker`, `_current_speed`, `_capped_speed`, `synthesize`) unchanged.

- [ ] **Step 4: Run the tests again**

Run: `PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_qwen3_tts.py -v`
Expected: PASS.

- [ ] **Step 5: Run mypy on the file**

Run: `.venv/bin/python -m mypy src/vocalis/engines/qwen3_tts.py`
Expected: no errors (the `"int | None | Callable[[], int | None]"` string-quoted unions follow the file's existing style for forward-referenced unions).

- [ ] **Step 6: Commit**

```bash
git add src/vocalis/engines/qwen3_tts.py tests/test_qwen3_tts.py
git commit -m "feat: Qwen3-TTS CPU thread cap and GPU VRAM ceiling"
```

---

### Task 5: Kokoro CPU thread cap and GPU VRAM ceiling

**Files:**
- Modify: `src/vocalis/engines/kokoro_tts.py`
- Test: `tests/test_kokoro_tts.py`

**Interfaces:**
- Consumes: nothing new from earlier tasks.
- Produces: `KokoroTTS(..., max_cpu_threads=..., gpu_vram_ceiling_mb=...)`; `_default_session_builder(model_path, use_gpu, intra_op_num_threads=None, gpu_mem_limit_mb=None)`.

- [ ] **Step 1: Write the failing tests**

First, update the shared `_engine()` test helper near the top of `tests/test_kokoro_tts.py` so every existing test that uses it keeps working once `KokoroTTS.load()` calls `session_builder` with two extra arguments:

```python
def _engine(tmp_path: Path, kokoro: _FakeKokoro | None = None, **kwargs: Any) -> tuple[KokoroTTS, list[tuple[Path, bool, "int | None", "int | None"]], _FakeKokoro]:
    model, voices = _files(tmp_path)
    built: list[tuple[Path, bool, "int | None", "int | None"]] = []
    fake = kokoro or _FakeKokoro()

    def session_builder(
        path: Path, use_gpu: bool, intra_op_num_threads: "int | None" = None, gpu_mem_limit_mb: "int | None" = None,
    ) -> object:
        built.append((path, use_gpu, intra_op_num_threads, gpu_mem_limit_mb))
        return object()

    engine = KokoroTTS(model, voices, session_builder=session_builder, kokoro_factory=lambda _s, _v: fake, **kwargs)
    return engine, built, fake
```

Then add:

```python
def test_load_passes_the_cpu_thread_cap_to_the_session_builder(tmp_path: Path) -> None:
    engine, built, _ = _engine(tmp_path, max_cpu_threads=4)
    engine.load()
    assert built[0][2] == 4


def test_load_passes_no_thread_cap_when_unset(tmp_path: Path) -> None:
    engine, built, _ = _engine(tmp_path)
    engine.load()
    assert built[0][2] is None


def test_load_passes_the_gpu_vram_ceiling_to_the_session_builder(tmp_path: Path) -> None:
    engine, built, _ = _engine(tmp_path, gpu_vram_ceiling_mb=512)
    engine.load()
    assert built[0][3] == 512


def test_thread_cap_can_be_a_live_callable(tmp_path: Path) -> None:
    cap = {"value": 2}
    engine, built, _ = _engine(tmp_path, max_cpu_threads=lambda: cap["value"])
    cap["value"] = 6
    engine.load()
    assert built[0][2] == 6


class _FakeSessionOptions:
    def __init__(self) -> None:
        self.intra_op_num_threads: "int | None" = None


class _FakeOrt:
    def __init__(self, providers: list) -> None:  # type: ignore[type-arg]
        self._providers = providers
        self.sessions: list[dict] = []  # type: ignore[type-arg]

    def get_available_providers(self) -> list:  # type: ignore[type-arg]
        return self._providers

    def SessionOptions(self) -> _FakeSessionOptions:
        return _FakeSessionOptions()

    def InferenceSession(self, path: str, sess_options: Any, providers: list) -> Any:  # type: ignore[type-arg]
        self.sessions.append({"path": path, "sess_options": sess_options, "providers": providers})
        return object()


def test_default_session_builder_sets_intra_op_num_threads(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    from vocalis.engines.kokoro_tts import _default_session_builder

    fake_ort = _FakeOrt(["CPUExecutionProvider"])
    monkeypatch.setattr("vocalis.engines.kokoro_tts.importlib.import_module", lambda _name: fake_ort)
    _default_session_builder(tmp_path / "m.onnx", False, intra_op_num_threads=4)
    assert fake_ort.sessions[0]["sess_options"].intra_op_num_threads == 4


def test_default_session_builder_sets_gpu_mem_limit_on_cuda(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    from vocalis.engines.kokoro_tts import _default_session_builder

    fake_ort = _FakeOrt(["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr("vocalis.engines.kokoro_tts.importlib.import_module", lambda _name: fake_ort)
    _default_session_builder(tmp_path / "m.onnx", True, gpu_mem_limit_mb=512)
    providers = fake_ort.sessions[0]["providers"]
    assert providers[0] == ("CUDAExecutionProvider", {"gpu_mem_limit": 512 * 1024 * 1024})


def test_default_session_builder_skips_gpu_mem_limit_when_not_given(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    from vocalis.engines.kokoro_tts import _default_session_builder

    fake_ort = _FakeOrt(["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr("vocalis.engines.kokoro_tts.importlib.import_module", lambda _name: fake_ort)
    _default_session_builder(tmp_path / "m.onnx", True)
    providers = fake_ort.sessions[0]["providers"]
    assert providers[0] == ("CUDAExecutionProvider", {})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_kokoro_tts.py -v`
Expected: FAIL — `TypeError: KokoroTTS.__init__() got an unexpected keyword argument 'max_cpu_threads'`.

- [ ] **Step 3: Implement the two caps**

In `src/vocalis/engines/kokoro_tts.py`, replace `_default_session_builder` and `KokoroTTS.__init__`/`load`:

```python
def _default_session_builder(
    model_path: Path, use_gpu: bool, intra_op_num_threads: "int | None" = None, gpu_mem_limit_mb: "int | None" = None,
) -> Any:
    """Create the ONNX session, on the graphics card only if asked and available."""
    ort = importlib.import_module("onnxruntime")
    providers: list[Any] = ["CPUExecutionProvider"]
    if use_gpu and "CUDAExecutionProvider" in ort.get_available_providers():
        cuda_options: dict[str, Any] = {}
        if gpu_mem_limit_mb is not None:
            cuda_options["gpu_mem_limit"] = gpu_mem_limit_mb * 1024 * 1024
        providers = [("CUDAExecutionProvider", cuda_options), "CPUExecutionProvider"]
    session_options = ort.SessionOptions()
    if intra_op_num_threads is not None:
        session_options.intra_op_num_threads = intra_op_num_threads
    return ort.InferenceSession(str(model_path), sess_options=session_options, providers=providers)
```

```python
    def __init__(
        self,
        model_path: Path,
        voices_path: Path,
        voice: "str | Callable[[], str]" = "af_heart",
        use_gpu: "bool | Callable[[], bool]" = False,
        speed: "float | Callable[[], float]" = 1.0,
        max_cpu_threads: "int | None | Callable[[], int | None]" = None,
        gpu_vram_ceiling_mb: "int | None | Callable[[], int | None]" = None,
        session_builder: Callable[..., Any] = _default_session_builder,
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
            max_cpu_threads: Cap for ONNX Runtime's intra_op_num_threads, or
                a function returning it, re-read at load time. None means
                no cap.
            gpu_vram_ceiling_mb: Cap for ONNX Runtime's CUDA gpu_mem_limit
                provider option, or a function returning it, re-read at
                load time. None means no cap; ignored off the GPU.
            session_builder: Builds the ONNX session; injectable for tests.
            kokoro_factory: Wraps a session into a synthesizer; injectable for tests.
        """
        self._model_path = model_path
        self._voices_path = voices_path
        self._voice = voice
        self._use_gpu = use_gpu
        self._speed = speed
        self._max_cpu_threads = max_cpu_threads
        self._gpu_vram_ceiling_mb = gpu_vram_ceiling_mb
        self._session_builder = session_builder
        self._kokoro_factory = kokoro_factory
        self._kokoro: Any | None = None

    @property
    def is_loaded(self) -> bool:
        """Whether the model is currently loaded."""
        return self._kokoro is not None

    def _current_max_cpu_threads(self) -> "int | None":
        """The CPU thread cap to apply right now (re-read at every load)."""
        return self._max_cpu_threads() if callable(self._max_cpu_threads) else self._max_cpu_threads

    def _current_gpu_vram_ceiling_mb(self) -> "int | None":
        """The GPU VRAM ceiling to apply right now (re-read at every load)."""
        return self._gpu_vram_ceiling_mb() if callable(self._gpu_vram_ceiling_mb) else self._gpu_vram_ceiling_mb

    def load(self) -> None:
        """Load the model.

        Raises:
            KokoroLoadError: If a model file is missing or loading fails.
        """
        for path in (self._model_path, self._voices_path):
            if not path.exists():
                raise KokoroLoadError(f"Kokoro file not found: {path}")
        try:
            use_gpu = self._use_gpu() if callable(self._use_gpu) else self._use_gpu
            session = self._session_builder(
                self._model_path, use_gpu, self._current_max_cpu_threads(), self._current_gpu_vram_ceiling_mb(),
            )
            self._kokoro = self._kokoro_factory(session, self._voices_path)
        except Exception as exc:
            self._kokoro = None
            raise KokoroLoadError(f"Could not load Kokoro: {exc}") from exc
        providers: list[str] = getattr(session, "get_providers", lambda: [])()
        logger.info("Kokoro loaded (graphics card requested: %s, providers: %s)", use_gpu, providers)
```

Leave `unload`, `_current_voice`, `_current_speed`, `synthesize`, and the module-level helpers (`_split_long_part`, `_pieces_for_kokoro`, `_language_for`, `_default_kokoro_factory`) unchanged.

- [ ] **Step 4: Run the tests again**

Run: `PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_kokoro_tts.py -v`
Expected: PASS.

- [ ] **Step 5: Run mypy on the file**

Run: `.venv/bin/python -m mypy src/vocalis/engines/kokoro_tts.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/vocalis/engines/kokoro_tts.py tests/test_kokoro_tts.py
git commit -m "feat: Kokoro CPU thread cap and GPU VRAM ceiling"
```

---

### Task 6: Piper best-effort CPU thread cap

**Files:**
- Modify: `src/vocalis/engines/piper_tts.py`
- Test: `tests/test_piper_tts.py`

**Interfaces:**
- Consumes: nothing new from earlier tasks.
- Produces: `PiperTTS(..., max_cpu_threads=...)`; `CommandRunner.__call__(argv, input_bytes=None, env=None)`.

- [ ] **Step 1: Update the existing fake and its call sites**

In `tests/test_piper_tts.py`, update `FakeCommandRunner`:

```python
class FakeCommandRunner:
    def __init__(self, returncode: int = 0, stdout: bytes = b"", raise_exc: Exception | None = None) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.raise_exc = raise_exc
        self.calls: list[tuple[list[str], "bytes | None", "dict[str, str] | None"]] = []

    def __call__(
        self, argv: list[str], input_bytes: bytes | None = None, env: "dict[str, str] | None" = None
    ) -> "subprocess.CompletedProcess[bytes]":
        self.calls.append((argv, input_bytes, env))
        if self.raise_exc is not None:
            raise self.raise_exc
        return subprocess.CompletedProcess(args=argv, returncode=self.returncode, stdout=self.stdout)
```

Find the existing unpacking call sites and widen them to three elements:
- `argv, input_bytes = runner.calls[0]` → `argv, input_bytes, _env = runner.calls[0]`
- Both `argv, _ = runner.calls[0]` → `argv, _, _ = runner.calls[0]`

(`runner.calls[0][0][-2]` stays correct unchanged — it indexes into `argv`, the first element of the tuple.)

- [ ] **Step 2: Write the new failing tests**

Add:

```python
def test_synthesize_sets_thread_env_vars_when_a_cap_is_given(tmp_path: Path) -> None:
    binary = tmp_path / "piper"
    binary.write_text("")
    model = tmp_path / "voice.onnx"
    model.write_text("")
    runner = FakeCommandRunner(stdout=_pcm_bytes([0]))
    engine = PiperTTS(binary_path=binary, model_path=model, max_cpu_threads=4, command_runner=runner)
    engine.load()
    engine.synthesize("hi")
    _argv, _input_bytes, env = runner.calls[0]
    assert env is not None
    assert env["OMP_NUM_THREADS"] == "4"
    assert env["ORT_NUM_THREADS"] == "4"


def test_synthesize_passes_no_env_override_when_no_cap(tmp_path: Path) -> None:
    binary = tmp_path / "piper"
    binary.write_text("")
    model = tmp_path / "voice.onnx"
    model.write_text("")
    runner = FakeCommandRunner(stdout=_pcm_bytes([0]))
    engine = PiperTTS(binary_path=binary, model_path=model, command_runner=runner)
    engine.load()
    engine.synthesize("hi")
    _argv, _input_bytes, env = runner.calls[0]
    assert env is None


def test_thread_cap_can_be_a_live_callable(tmp_path: Path) -> None:
    binary = tmp_path / "piper"
    binary.write_text("")
    model = tmp_path / "voice.onnx"
    model.write_text("")
    runner = FakeCommandRunner(stdout=_pcm_bytes([0]))
    cap = {"value": 2}
    engine = PiperTTS(binary_path=binary, model_path=model, max_cpu_threads=lambda: cap["value"], command_runner=runner)
    engine.load()
    cap["value"] = 6
    engine.synthesize("hi")
    _argv, _input_bytes, env = runner.calls[0]
    assert env is not None
    assert env["OMP_NUM_THREADS"] == "6"
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_piper_tts.py -v`
Expected: FAIL — `TypeError: PiperTTS.__init__() got an unexpected keyword argument 'max_cpu_threads'` (plus the three unpacking lines failing until Step 1 is applied).

- [ ] **Step 4: Implement the cap**

In `src/vocalis/engines/piper_tts.py`, add `import os` to the imports, then update the `CommandRunner` protocol, `_default_command_runner`, and `PiperTTS`:

```python
class CommandRunner(Protocol):
    """Runs a command, optionally piping bytes to its stdin with a custom environment."""

    def __call__(
        self, argv: list[str], input_bytes: bytes | None = None, env: "dict[str, str] | None" = None
    ) -> "subprocess.CompletedProcess[bytes]": ...
```

```python
    def __init__(
        self,
        binary_path: Path,
        model_path: Path,
        sample_rate: int = 22050,
        speed: float | Callable[[], float] = 1.0,
        max_cpu_threads: "int | None | Callable[[], int | None]" = None,
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
            max_cpu_threads: Best-effort CPU thread cap for the piper
                subprocess, or a function returning it, re-read on every
                synthesis. Applied as OMP_NUM_THREADS/ORT_NUM_THREADS
                environment variables -- the installed piper binary has no
                CLI flag for this (verified against its own --help output),
                so this is not a guaranteed cap, only a best effort. None
                means no cap (the subprocess inherits this process's
                environment unchanged).
            command_runner: Callable(argv, input_bytes=None, env=None) ->
                CompletedProcess, used to run piper. Defaults to
                subprocess.run with a 30-second timeout. Injectable for tests.
        """
        self._binary_path = binary_path
        self._model_path = model_path
        self._sample_rate = sample_rate
        self._speed = speed
        self._max_cpu_threads = max_cpu_threads
        self._run: CommandRunner = command_runner or _default_command_runner
        self._loaded = False
```

```python
    def _current_max_cpu_threads(self) -> "int | None":
        """The CPU thread cap to apply right now (re-read on every synthesis)."""
        return self._max_cpu_threads() if callable(self._max_cpu_threads) else self._max_cpu_threads

    def _thread_cap_env(self) -> "dict[str, str] | None":
        """OMP_NUM_THREADS/ORT_NUM_THREADS for the piper subprocess, best-effort."""
        max_threads = self._current_max_cpu_threads()
        if max_threads is None:
            return None
        return {**os.environ, "OMP_NUM_THREADS": str(max_threads), "ORT_NUM_THREADS": str(max_threads)}
```

In `synthesize`, change the call to `self._run(...)`:

```python
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
                env=self._thread_cap_env(),
            )
```

And `_default_command_runner`:

```python
def _default_command_runner(
    argv: list[str], input_bytes: bytes | None = None, env: "dict[str, str] | None" = None
) -> "subprocess.CompletedProcess[bytes]":
    """Default CommandRunner: subprocess.run with a 30-second timeout."""
    return subprocess.run(argv, input=input_bytes, timeout=30.0, capture_output=True, env=env)
```

- [ ] **Step 5: Run the tests again**

Run: `PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_piper_tts.py -v`
Expected: PASS.

- [ ] **Step 6: Run mypy on the file**

Run: `.venv/bin/python -m mypy src/vocalis/engines/piper_tts.py`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add src/vocalis/engines/piper_tts.py tests/test_piper_tts.py
git commit -m "feat: best-effort CPU thread cap for the Piper subprocess"
```

---

### Task 7: Wire the live settings into cli.py's engine factories and selector

**Files:**
- Modify: `src/vocalis/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `LiveSettings.max_cpu_threads()`, `LiveSettings.max_ram_mb()`, `LiveSettings.gpu_vram_ceiling_mb()` (Task 1); `EngineSelector(max_own_ram_mb=...)` (Task 3); `Qwen3TTS(max_cpu_threads=, gpu_vram_ceiling_mb=)` (Task 4); `KokoroTTS(max_cpu_threads=, gpu_vram_ceiling_mb=)` (Task 5); `PiperTTS(max_cpu_threads=)` (Task 6).
- Produces: nothing new for later tasks — this is the integration point.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_cli.py`, next to the existing `test_primary_engine_reads_the_live_speed`/`test_kokoro_factory_reads_the_live_speed`/`test_fallback_engine_reads_the_live_speed`:

```python
def test_primary_engine_reads_the_live_cpu_and_vram_caps(tmp_path: Path) -> None:
    settings = _live_settings(tmp_path)
    with patch("vocalis.cli.list_hotkey_capable_devices", return_value=[]):
        feature, _ = build_quick_reader(
            config=settings.config, model_manager=ModelManager(),
            reading_coordinator=ReadingCoordinator(), settings=settings,
        )
    engine = feature._primary_engine_factory()  # noqa: SLF001
    assert isinstance(engine, Qwen3TTS)
    assert engine._current_max_cpu_threads() is None  # noqa: SLF001
    assert engine._current_gpu_vram_ceiling_mb() == settings.gpu_vram_ceiling_mb()  # noqa: SLF001
    settings.update(settings.config.model_copy(update={"max_cpu_threads": 4, "gpu_vram_ceiling_mb": 3000}))
    assert engine._current_max_cpu_threads() == 4  # noqa: SLF001
    assert engine._current_gpu_vram_ceiling_mb() == 3000  # noqa: SLF001


def test_kokoro_factory_reads_the_live_cpu_and_vram_caps(tmp_path: Path) -> None:
    settings = _live_settings(tmp_path)
    with patch("vocalis.cli.list_hotkey_capable_devices", return_value=[]):
        feature, _ = build_quick_reader(
            config=settings.config, model_manager=ModelManager(),
            reading_coordinator=ReadingCoordinator(), settings=settings,
        )
    kokoro = (feature._engine_factories or {})["kokoro"]()  # noqa: SLF001
    assert isinstance(kokoro, KokoroTTS)
    settings.update(settings.config.model_copy(update={"max_cpu_threads": 2, "gpu_vram_ceiling_mb": 1500}))
    assert kokoro._current_max_cpu_threads() == 2  # noqa: SLF001
    assert kokoro._current_gpu_vram_ceiling_mb() == 1500  # noqa: SLF001


def test_fallback_engine_reads_the_live_cpu_cap(tmp_path: Path) -> None:
    settings = _live_settings(tmp_path)
    with patch("vocalis.cli.list_hotkey_capable_devices", return_value=[]):
        feature, _ = build_quick_reader(
            config=settings.config, model_manager=ModelManager(),
            reading_coordinator=ReadingCoordinator(), settings=settings,
        )
    piper = feature._fallback_engine  # noqa: SLF001
    assert piper._current_max_cpu_threads() is None  # noqa: SLF001
    settings.update(settings.config.model_copy(update={"max_cpu_threads": 3}))
    assert piper._current_max_cpu_threads() == 3  # noqa: SLF001


def test_engine_selector_reads_the_live_max_ram(tmp_path: Path) -> None:
    settings = _live_settings(tmp_path)
    with patch("vocalis.cli.list_hotkey_capable_devices", return_value=[]):
        feature, _ = build_quick_reader(
            config=settings.config, model_manager=ModelManager(),
            reading_coordinator=ReadingCoordinator(), settings=settings,
        )
    selector = feature._engine_selector  # noqa: SLF001
    assert selector is not None
    assert selector._max_own_ram_mb() is None  # noqa: SLF001
    settings.update(settings.config.model_copy(update={"max_ram_mb": 2048}))
    assert selector._max_own_ram_mb() == 2048  # noqa: SLF001


def test_build_quick_reader_without_settings_uses_the_configured_caps() -> None:
    config = VocalisConfig(max_cpu_threads=5, gpu_vram_ceiling_mb=2000)
    with patch("vocalis.cli.list_hotkey_capable_devices", return_value=[]):
        feature, _ = build_quick_reader(
            config=config, model_manager=ModelManager(), reading_coordinator=ReadingCoordinator(),
        )
    engine = feature._primary_engine_factory()  # noqa: SLF001
    assert engine._current_max_cpu_threads() == 5  # noqa: SLF001
    assert engine._current_gpu_vram_ceiling_mb() == 2000  # noqa: SLF001
```

(Check whether `feature._engine_selector` is the actual private attribute name `QuickReaderFeature` stores its selector under — read `src/vocalis/features/quick_reader.py`'s `__init__` first; use whatever name it already assigns `engine_selector` to. If no such attribute exists, read the selector via the already-public `_make_engine_selector` return value in a narrower test that builds a selector directly instead of reaching into the feature — do not invent a new attribute on `QuickReaderFeature` just for this test.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_cli.py -k "cpu_and_vram or fallback_engine_reads_the_live_cpu or live_max_ram or configured_caps" -v`
Expected: FAIL — `AttributeError: 'Qwen3TTS' object has no attribute '_current_max_cpu_threads'` is already satisfied by Task 4; the real failure here is that `cli.py` never passes these values, so the assertions comparing against the *live* setting (e.g. `4` after `settings.update(...)`) fail because the engine was built with `max_cpu_threads=None` baked in instead of a live callable.

- [ ] **Step 3: Wire the live settings through `_make_qwen_factory`, `_make_kokoro_factory`, `_make_engine_selector`, and both `PiperTTS(...)` call sites**

In `src/vocalis/cli.py`, update `_make_qwen_factory`:

```python
def _make_qwen_factory(
    config: VocalisConfig, settings: LiveSettings | None
) -> Callable[[], Qwen3TTS]:
    """A factory building Qwen3TTS; voice, speed and the resource caps are read live when settings are given."""
    qwen_model_dir = Path(config.engines.qwen3_tts_model_dir)
    speaker: "str | Callable[[], str]" = (
        settings.voice if settings is not None else config.engines.qwen3_tts_speaker
    )
    speed: "float | Callable[[], float]" = settings.speaking_speed if settings is not None else config.speaking_speed
    max_cpu_threads: "int | None | Callable[[], int | None]" = (
        settings.max_cpu_threads if settings is not None else config.max_cpu_threads or None
    )
    gpu_vram_ceiling_mb: "int | None | Callable[[], int | None]" = (
        settings.gpu_vram_ceiling_mb if settings is not None else config.gpu_vram_ceiling_mb
    )

    def primary_engine_factory() -> Qwen3TTS:
        """Build the primary text-to-speech engine."""
        return Qwen3TTS(
            model_dir=qwen_model_dir,
            speaker=speaker,
            speed=speed,
            max_cpu_threads=max_cpu_threads,
            gpu_vram_ceiling_mb=gpu_vram_ceiling_mb,
        )

    return primary_engine_factory
```

Update `_make_engine_selector`:

```python
def _make_engine_selector(
    preference: Callable[[], str],
    model_manager: ModelManager,
    hardware_probe: Callable[[], HardwareInfo],
    max_own_ram_mb: Callable[[], "int | None"] = lambda: None,
) -> EngineSelector:
    """A selector that credits Qwen3-TTS's own graphics memory back while it is loaded."""

    def held_vram_mb() -> int:
        return _QWEN3_HELD_VRAM_MB if model_manager.current_engine_id == QWEN3_TTS else 0

    return EngineSelector(
        preference=preference, hardware_probe=hardware_probe, held_vram_mb=held_vram_mb, max_own_ram_mb=max_own_ram_mb,
    )
```

Update `_make_kokoro_factory`:

```python
def _make_kokoro_factory(
    config: VocalisConfig, settings: LiveSettings | None, selector: EngineSelector
) -> Callable[[], KokoroTTS]:
    """A factory building KokoroTTS; voice, speed and the resource caps are read live, graphics-card use is decided at load."""
    model_path = Path(config.engines.kokoro_model_path)
    voices_path = Path(config.engines.kokoro_voices_path)
    voice: "str | Callable[[], str]" = settings.kokoro_voice if settings is not None else config.engines.kokoro_voice
    speed: "float | Callable[[], float]" = settings.speaking_speed if settings is not None else config.speaking_speed
    max_cpu_threads: "int | None | Callable[[], int | None]" = (
        settings.max_cpu_threads if settings is not None else config.max_cpu_threads or None
    )
    gpu_vram_ceiling_mb: "int | None | Callable[[], int | None]" = (
        settings.gpu_vram_ceiling_mb if settings is not None else config.gpu_vram_ceiling_mb
    )

    def kokoro_factory() -> KokoroTTS:
        """Build the light voice engine."""
        return KokoroTTS(
            model_path=model_path,
            voices_path=voices_path,
            voice=voice,
            use_gpu=selector.kokoro_use_gpu,
            speed=speed,
            max_cpu_threads=max_cpu_threads,
            gpu_vram_ceiling_mb=gpu_vram_ceiling_mb,
        )

    return kokoro_factory
```

In both `build_quick_reader` and `build_doc_reader`, update the `_make_engine_selector(...)` call and the `PiperTTS(...)` construction identically in each function:

```python
    selector = _make_engine_selector(
        preference,
        model_manager,
        hardware_probe if hardware_probe is not None else probe_hardware,
        settings.max_ram_mb if settings is not None else (lambda: config.max_ram_mb or None),
    )
```

```python
    fallback_engine = PiperTTS(
        binary_path=Path(config.engines.piper_binary_path),
        model_path=Path(config.engines.piper_voice_model_path),
        speed=settings.speaking_speed if settings is not None else config.speaking_speed,
        max_cpu_threads=settings.max_cpu_threads if settings is not None else (config.max_cpu_threads or None),
    )
```

(`config.max_cpu_threads or None` and `config.max_ram_mb or None` translate the `0`-sentinel to `None` for the no-`settings` path, mirroring what `LiveSettings.max_cpu_threads()`/`max_ram_mb()` already do when `settings` is given — `0 or None` is `None`, any positive int is truthy and passes through unchanged.)

- [ ] **Step 4: Run the tests again**

Run: `PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_cli.py -v`
Expected: PASS (including every pre-existing `test_cli.py` test, since all new parameters have safe defaults/fallbacks).

- [ ] **Step 5: Run mypy on the file**

Run: `.venv/bin/python -m mypy src/vocalis/cli.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/vocalis/cli.py tests/test_cli.py
git commit -m "feat: wire live CPU/RAM/VRAM limits into the engine factories and selector"
```

---

### Task 8: Model unload runs a full garbage collection

**Files:**
- Modify: `src/vocalis/model_manager.py`
- Test: `tests/test_model_manager.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: nothing new for later tasks.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_model_manager.py`:

```python
def test_shutdown_runs_garbage_collection_after_unloading() -> None:
    from unittest.mock import patch

    manager = ModelManager()
    manager.acquire("engine", lambda: FakeModelEngine("engine"))
    with patch("vocalis.model_manager.gc.collect") as mock_collect:
        manager.shutdown()
    mock_collect.assert_called_once()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_model_manager.py -k garbage_collection -v`
Expected: FAIL — `AssertionError: Expected 'collect' to have been called once. Called 0 times.`

- [ ] **Step 3: Implement**

In `src/vocalis/model_manager.py`, add `import gc` next to `import logging`/`import threading`, and update `_unload_current_locked`:

```python
    def _unload_current_locked(self) -> None:
        """Unload the current engine. Caller must hold self._lock.

        Guarantees bookkeeping is reset to "nothing loaded" even if the
        engine's own unload() raises (e.g. a CUDA teardown error) — losing
        track of a resident model would be worse than losing the error.
        Runs a full garbage collection afterwards, on top of the engine's
        own torch.cuda.empty_cache(): a reference cycle holding the model
        (or its tensors) alive would otherwise delay VRAM release until
        Python's own cyclic collector happened to run on its own schedule.
        """
        try:
            if self._current_engine is not None:
                logger.info("Unloading idle engine %r", self._current_engine_id)
                self._current_engine.unload()
        except Exception:
            logger.exception("Error unloading engine %r", self._current_engine_id)
        finally:
            self._current_engine = None
            self._current_engine_id = None
            self._use_count = 0
            gc.collect()
```

- [ ] **Step 4: Run the test again**

Run: `PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_model_manager.py -v`
Expected: PASS — every pre-existing test in the file too, since `gc.collect()` is side-effect-only here and doesn't change any observable state the other tests assert on.

- [ ] **Step 5: Commit**

```bash
git add src/vocalis/model_manager.py tests/test_model_manager.py
git commit -m "feat: run gc.collect() after unloading a model"
```

---

### Task 9: systemd unit gets a Nice= line

**Files:**
- Modify: `packaging/systemd/vocalis.service`
- Test: `tests/test_systemd_unit.py`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing for later tasks.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_systemd_unit.py`:

```python
def test_nice_level_lowers_scheduling_priority() -> None:
    lines = _lines()
    assert "Nice=10" in lines
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_systemd_unit.py -v`
Expected: FAIL — `AssertionError: assert 'Nice=10' in [...]`.

- [ ] **Step 3: Add the line**

In `packaging/systemd/vocalis.service`, add `Nice=10` to the `[Service]` section, right after `RestartSec=3`:

```ini
[Service]
Type=simple
ExecStart=__VOCALIS_EXEC__
WorkingDirectory=__VOCALIS_WORKDIR__
Restart=on-failure
RestartSec=3
Nice=10
```

- [ ] **Step 4: Run the test again**

Run: `PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_systemd_unit.py -v`
Expected: PASS.

- [ ] **Step 5: Verify the unit file is still syntactically valid, if systemd-analyze is available**

Run: `systemd-analyze --user verify packaging/systemd/vocalis.service || true`
(Non-fatal if `systemd-analyze` complains about the placeholder paths — the existing README already documents that this template is not meant to verify clean until the installer substitutes real paths; only check that `Nice=10` itself does not produce a new, different complaint.)

- [ ] **Step 6: Commit**

```bash
git add packaging/systemd/vocalis.service tests/test_systemd_unit.py
git commit -m "feat: lower Vocalis's scheduling priority with Nice=10"
```

---

### Task 10: Settings window controls for CPU/RAM/VRAM limits

**Files:**
- Modify: `src/vocalis/ui/settings_window.py`
- Test: `tests/test_settings_window.py`

**Interfaces:**
- Consumes: `VocalisConfig.max_cpu_threads`, `max_ram_mb`, `gpu_vram_ceiling_mb` (Task 1).
- Produces: nothing for later tasks.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_settings_window.py`, next to the existing `test_sound_and_speed_defaults_are_shown`/`test_saving_writes_output_and_speed_and_applies_live`:

```python
def test_resource_limit_defaults_are_shown(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    _settings, window, _applied, _path = make(qtbot, tmp_path)
    assert window._cpu_threads_spin.value() == 0  # noqa: SLF001
    assert window._max_ram_spin.value() == 0  # noqa: SLF001
    assert window._vram_ceiling_spin.value() == 6144  # noqa: SLF001


def test_saving_writes_the_resource_limits(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings, window, _applied, path = make(qtbot, tmp_path)
    window._cpu_threads_spin.setValue(4)  # noqa: SLF001
    window._max_ram_spin.setValue(2048)  # noqa: SLF001
    window._vram_ceiling_spin.setValue(3000)  # noqa: SLF001
    window._on_save_clicked()  # noqa: SLF001
    saved = load_config(path)
    assert saved.max_cpu_threads == 4
    assert saved.max_ram_mb == 2048
    assert saved.gpu_vram_ceiling_mb == 3000
    assert settings.max_cpu_threads() == 4
    assert settings.max_ram_mb() == 2048
    assert settings.gpu_vram_ceiling_mb() == 3000


def test_resource_limits_round_trip_through_reload(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = VocalisConfig(max_cpu_threads=2, max_ram_mb=1024, gpu_vram_ceiling_mb=5000)
    _settings, window, _applied, _path = make(qtbot, tmp_path, config=config)
    assert window._cpu_threads_spin.value() == 2  # noqa: SLF001
    assert window._max_ram_spin.value() == 1024  # noqa: SLF001
    assert window._vram_ceiling_spin.value() == 5000  # noqa: SLF001
```

(Check the top of `tests/test_settings_window.py` for how `load_config` and `VocalisConfig` are already imported, and reuse those imports — do not add duplicate ones. The `make(qtbot, tmp_path, config=...)` helper and its `config=` keyword already exist, per the file's other tests that pass a custom `config`.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_settings_window.py -v`
Expected: FAIL — `AttributeError: 'SettingsWindow' object has no attribute '_cpu_threads_spin'`.

- [ ] **Step 3: Add the three spin boxes**

In `src/vocalis/ui/settings_window.py`, add `QSpinBox` to the PySide6 imports:

```python
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
```

Add constants near the other `_SPEED_*` ones:

```python
_CPU_THREADS_MAXIMUM = 128
_MAX_RAM_MAXIMUM_MB = 262144
_VRAM_CEILING_MINIMUM_MB = 256
_VRAM_CEILING_MAXIMUM_MB = 65536
```

In `__init__`, right after `self._speed_slider = ...` block (after its `setMaximum` call) and before `self._speed_label = ...`, add:

```python
        self._cpu_threads_spin = QSpinBox()
        self._cpu_threads_spin.setRange(0, _CPU_THREADS_MAXIMUM)
        self._max_ram_spin = QSpinBox()
        self._max_ram_spin.setRange(0, _MAX_RAM_MAXIMUM_MB)
        self._vram_ceiling_spin = QSpinBox()
        self._vram_ceiling_spin.setRange(_VRAM_CEILING_MINIMUM_MB, _VRAM_CEILING_MAXIMUM_MB)
```

In `_build_layout`, right after the `form.addRow("Speaking speed", speed_row)` line:

```python
        form.addRow("Max CPU threads (0 = no limit)", self._cpu_threads_spin)
        form.addRow("Max RAM, MB (0 = no limit)", self._max_ram_spin)
        form.addRow("GPU VRAM ceiling, MB", self._vram_ceiling_spin)
```

In `load_from_settings`, right after the `self._update_speed_label(...)` line:

```python
        self._cpu_threads_spin.setValue(config.max_cpu_threads)
        self._max_ram_spin.setValue(config.max_ram_mb)
        self._vram_ceiling_spin.setValue(config.gpu_vram_ceiling_mb)
```

In `_config_from_widgets`, add the three new keys to the final `config.model_copy(update={...})` call, alongside `"speaking_speed": ...`:

```python
        return config.model_copy(
            update={
                "notifications_enabled": self._notifications_box.isChecked(),
                "floating_widget_enabled": self._widget_box.isChecked(),
                "shortcuts": shortcuts,
                "engines": engines,
                "audio_output_device": self._selected_output_value(),
                "speaking_speed": self._speed_slider.value() / _SPEED_SCALE,
                "max_cpu_threads": self._cpu_threads_spin.value(),
                "max_ram_mb": self._max_ram_spin.value(),
                "gpu_vram_ceiling_mb": self._vram_ceiling_spin.value(),
            }
        )
```

- [ ] **Step 4: Run the tests again**

Run: `PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_settings_window.py -v`
Expected: PASS — including every pre-existing test (none of them assert on the exact set of `form` rows, only on specific widgets by name).

- [ ] **Step 5: Run mypy on the file**

Run: `.venv/bin/python -m mypy src/vocalis/ui/settings_window.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/vocalis/ui/settings_window.py tests/test_settings_window.py
git commit -m "feat: expose CPU/RAM/VRAM resource limits in the Settings window"
```

---

### Task 11: Docs and changelog

**Files:**
- Modify: `docs/architecture.md`
- Modify: `docs/algorithms.md`
- Modify: `docs/user-docs.md`
- Create: `.claude/<timestamp>_CHANGELOG.md` (use the real current timestamp, matching the format of the Stage 6c changelog at `.claude/2026-10-25_1938_CHANGELOG.md`)

**Interfaces:**
- Consumes: the finished behavior of Tasks 1-10.
- Produces: nothing — this is the last task.

- [ ] **Step 1: Read the existing docs sections for Stage 6a/6b/6c to match their style**

Run: `grep -n "Auto-select\|auto_select\|EngineSelector\|gpu_vram_ceiling" docs/architecture.md docs/algorithms.md docs/user-docs.md`

Read the surrounding sections each hit falls in before writing new text, so the new paragraphs read as part of the same document rather than a bolted-on appendix.

- [ ] **Step 2: Add a "Resource limits" subsection to `docs/architecture.md`**

Describe, in the doc's existing voice: the three new/enforced settings, which engine applies each one and how (`torch.set_num_threads`/ONNX `intra_op_num_threads`/best-effort Piper env vars for CPU; `torch.cuda.set_per_process_memory_fraction`/ONNX `gpu_mem_limit` for VRAM; the advisory RAM-stress signal for RAM), and the downshift hysteresis mechanism in `EngineSelector` (60s hold after a downshift, forced choices unaffected).

- [ ] **Step 3: Add the hysteresis rule to `docs/algorithms.md`**

Document `EngineSelector._apply_hysteresis` next to wherever the existing Auto-selection policy (`auto_select.decide`) is already documented: the tier ranking, the hold window, and that it only applies to Auto.

- [ ] **Step 4: Add the three new Settings controls to `docs/user-docs.md`**

Plain-English description for end users: what "Max CPU threads" / "Max RAM" / "GPU VRAM ceiling" do, what `0` means, and that Piper's CPU cap is best-effort (phrase this for a non-technical reader, e.g. "Piper does its best to respect this limit, but cannot always guarantee it").

- [ ] **Step 5: Write the changelog**

Create `.claude/<timestamp>_CHANGELOG.md` (timestamp format `YYYY-MM-DD_HHMM`, matching existing changelogs in `.claude/`) summarizing, in plain English: the three new/enforced settings, which engine each applies to and how, the RAM signal being advisory not a hard cap, Piper's thread cap being best-effort (with the reason: its installed binary has no thread flag), the Auto-selection hysteresis, the `Nice=10` systemd change, and the `gc.collect()` addition on unload.

- [ ] **Step 6: Commit**

```bash
git add docs/architecture.md docs/algorithms.md docs/user-docs.md .claude/*_CHANGELOG.md
git commit -m "docs: document Stage 6d resource limits"
```

---

## Final Verification (before merge)

Run the project's standing gates once, on the whole branch:

```bash
PYTHONPATH=$PWD/src .venv/bin/python -m pytest -q
.venv/bin/python -m mypy
```

Expect ≥85% coverage, zero warnings, and a clean mypy run across all source files — the same bar Stages 6a/6b/6c already met.
