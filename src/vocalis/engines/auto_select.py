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


def kokoro_should_use_gpu(hw: HardwareInfo, policy: AutoPolicy = AutoPolicy()) -> bool:
    """Whether Kokoro should run on the graphics card (enough free memory) or the processor."""
    return hw.gpu is not None and hw.gpu.free_mb >= policy.kokoro_gpu_min_free_mb
