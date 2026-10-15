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
