"""Tests for vocalis.engines.auto_select — the pure rule table behind "Auto" voice choice."""
from dataclasses import replace

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
