"""Tests for vocalis.engines.registry.EngineSelector."""
import pytest

from vocalis.engines.auto_select import KOKORO, PIPER, QWEN3_TTS
from vocalis.engines.registry import EngineSelector
from vocalis.hardware import GpuInfo, HardwareInfo

STRONG = HardwareInfo(GpuInfo("Big", 8188, 7000), 16000, 10000, 8, 1.0)
NO_GPU = HardwareInfo(None, 16000, 10000, 8, 1.0)


def _selector(choice: str, hw: HardwareInfo = STRONG, held: int = 0) -> EngineSelector:
    """Build a selector with a fixed choice, hardware and held memory."""
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
        """Always fail, like a broken probe."""
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
