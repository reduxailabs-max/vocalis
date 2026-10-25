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
