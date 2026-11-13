"""Tests for how the app behaves when the optional Qwen3-TTS add-on is not installed."""
import subprocess
from pathlib import Path

import pytest

from vocalis import hardware
from vocalis.config import VocalisConfig
from vocalis.engines.auto_select import KOKORO, PIPER, QWEN3_TTS, decide
from vocalis.engines.registry import EngineSelector
from vocalis.hardware import GpuInfo, HardwareInfo, parse_nvidia_smi_memory
from vocalis.startup_checks import _check_chosen_qwen3

STRONG = HardwareInfo(GpuInfo("Big", 8188, 7000), 16000, 10000, 8, 1.0)


def test_auto_decision_without_qwen3_uses_kokoro_on_a_strong_pc() -> None:
    decision = decide(STRONG, qwen3_installed=False)
    assert decision.chain == (KOKORO, PIPER)
    assert "not installed" in decision.reason


def test_auto_decision_keeps_the_other_rules_without_qwen3() -> None:
    no_gpu = HardwareInfo(None, 16000, 10000, 8, 1.0)
    small = HardwareInfo(GpuInfo("Small", 2048, 2000), 16000, 10000, 8, 1.0)
    assert decide(no_gpu, qwen3_installed=False).chain == (PIPER, KOKORO)
    assert decide(small, qwen3_installed=False).chain == (KOKORO, PIPER)


def test_forced_qwen3_without_the_add_on_uses_kokoro_and_says_so() -> None:
    selector = EngineSelector(
        preference=lambda: "qwen3_tts", hardware_probe=lambda: STRONG, qwen3_installed=lambda: False
    )
    plan = selector.plan()
    assert plan.chain == (KOKORO, PIPER)
    assert "not installed" in plan.reason and "Settings" in plan.reason


def test_auto_without_the_add_on_never_picks_qwen3() -> None:
    selector = EngineSelector(preference=lambda: "auto", hardware_probe=lambda: STRONG, qwen3_installed=lambda: False)
    assert selector.plan().chain == (KOKORO, PIPER)


def test_the_add_on_state_is_read_live_on_every_plan() -> None:
    """Adding Qwen3-TTS from Settings takes effect without a restart."""
    state = {"installed": False}
    selector = EngineSelector(
        preference=lambda: "auto", hardware_probe=lambda: STRONG, qwen3_installed=lambda: state["installed"]
    )
    assert selector.plan().chain[0] == KOKORO
    state["installed"] = True
    assert selector.plan().chain[0] == QWEN3_TTS


def test_a_forced_kokoro_choice_ignores_the_add_on_state() -> None:
    selector = EngineSelector(
        preference=lambda: "kokoro", hardware_probe=lambda: STRONG, qwen3_installed=lambda: False
    )
    assert selector.plan().chain == (KOKORO, PIPER)


def _config(tmp_path: Path, quick: str = "auto", document: str = "auto") -> VocalisConfig:
    config = VocalisConfig()
    config.engines.qwen3_tts_model_dir = str(tmp_path / "qwen")
    config.engines.quick_reader_engine = quick  # type: ignore[assignment]
    config.engines.document_reader_engine = document  # type: ignore[assignment]
    return config


def test_no_startup_warning_about_qwen3_when_nobody_chose_it(tmp_path: Path) -> None:
    assert _check_chosen_qwen3(_config(tmp_path)) is None
    assert _check_chosen_qwen3(_config(tmp_path, quick="kokoro", document="piper")) is None


@pytest.mark.parametrize(("quick", "document"), [("qwen3_tts", "auto"), ("auto", "qwen3_tts")])
def test_startup_warns_when_qwen3_was_chosen_but_is_missing(tmp_path: Path, quick: str, document: str) -> None:
    message = _check_chosen_qwen3(_config(tmp_path, quick, document))
    assert message is not None
    assert "Kokoro" in message and "Settings" in message


def test_no_startup_warning_when_the_chosen_model_is_there(tmp_path: Path) -> None:
    (tmp_path / "qwen").mkdir()
    (tmp_path / "qwen" / "config.json").write_text("{}")
    (tmp_path / "qwen" / "model.safetensors").write_text("x")
    assert _check_chosen_qwen3(_config(tmp_path, quick="qwen3_tts")) is None


def test_parse_nvidia_smi_memory_reads_name_total_and_free() -> None:
    assert parse_nvidia_smi_memory("NVIDIA GeForce RTX 3060, 12288, 11000\n") == GpuInfo(
        "NVIDIA GeForce RTX 3060", 12288, 11000
    )


def test_parse_nvidia_smi_memory_keeps_a_comma_in_the_name() -> None:
    assert parse_nvidia_smi_memory("Card, Ti, 8192, 100\n") == GpuInfo("Card, Ti", 8192, 100)


@pytest.mark.parametrize("text", ["", "oops", "a, b, c", "x, 0, 5", "x, 5", "x, -5, 4"])
def test_parse_nvidia_smi_memory_garbage_is_none(text: str) -> None:
    assert parse_nvidia_smi_memory(text) is None


def test_query_gpu_with_nvidia_smi_reads_the_command_output(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd: list[str], **kw: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 0, stdout="Card, 8192, 4000\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert hardware.query_gpu_with_nvidia_smi() == GpuInfo("Card", 8192, 4000)


@pytest.mark.parametrize("error", [FileNotFoundError(), subprocess.TimeoutExpired("nvidia-smi", 10)])
def test_query_gpu_with_nvidia_smi_is_none_when_the_command_fails(
    monkeypatch: pytest.MonkeyPatch, error: BaseException
) -> None:
    def boom(cmd: list[str], **kw: object) -> subprocess.CompletedProcess[str]:
        raise error

    monkeypatch.setattr(subprocess, "run", boom)
    assert hardware.query_gpu_with_nvidia_smi() is None


def test_query_gpu_uses_nvidia_smi_when_torch_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    monkeypatch.setattr(hardware, "query_gpu_with_nvidia_smi", lambda: GpuInfo("Smi", 1, 1))
    monkeypatch.setattr(hardware, "query_gpu_with_torch", lambda: GpuInfo("Torch", 2, 2))
    assert hardware.query_gpu() == GpuInfo("Smi", 1, 1)


def test_query_gpu_uses_torch_when_it_is_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())
    monkeypatch.setattr(hardware, "query_gpu_with_nvidia_smi", lambda: GpuInfo("Smi", 1, 1))
    monkeypatch.setattr(hardware, "query_gpu_with_torch", lambda: GpuInfo("Torch", 2, 2))
    assert hardware.query_gpu() == GpuInfo("Torch", 2, 2)
