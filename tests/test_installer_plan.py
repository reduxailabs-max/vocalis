"""Tests for the installer hardware plan."""
import subprocess

import pytest

from vocalis.installer import plan
from vocalis.installer.plan import InstalledGpu, make_plan, parse_nvidia_smi, read_gpu


def test_parse_nvidia_smi_reads_name_and_memory() -> None:
    gpu = parse_nvidia_smi("NVIDIA GeForce RTX 3060, 12288\n")
    assert gpu == InstalledGpu("NVIDIA GeForce RTX 3060", 12288)


@pytest.mark.parametrize("text", ["", "oops", "a, b", "x, -5", "GPU, \u00b2", "GPU, 0"])
def test_parse_nvidia_smi_garbage_is_none(text: str) -> None:
    assert parse_nvidia_smi(text) is None


def _raiser(exc: BaseException):  # type: ignore[no-untyped-def]
    def run(command: list[str]) -> str:
        raise exc

    return run


def test_read_gpu_missing_binary_is_none() -> None:
    assert read_gpu(_raiser(FileNotFoundError())) is None


def test_read_gpu_nonzero_exit_is_none() -> None:
    assert read_gpu(_raiser(subprocess.CalledProcessError(9, "nvidia-smi"))) is None


def test_read_gpu_timeout_is_none() -> None:
    assert read_gpu(_raiser(subprocess.TimeoutExpired("nvidia-smi", 10))) is None


def test_read_gpu_parses_injected_output() -> None:
    assert read_gpu(lambda cmd: "Card, 8192\n") == InstalledGpu("Card", 8192)


def test_plan_4096mb_skips_qwen3_like_the_auto_rule() -> None:
    p = make_plan(InstalledGpu("Card", 4096))
    assert p.gpu_runtime and not p.fetch_qwen3


def test_plan_4097mb_fetches_qwen3() -> None:
    p = make_plan(InstalledGpu("Card", 4097))
    assert p.gpu_runtime and p.fetch_qwen3


def test_plan_4095mb_skips_qwen3() -> None:
    p = make_plan(InstalledGpu("Card", 4095))
    assert p.gpu_runtime and not p.fetch_qwen3


def test_plan_no_gpu_uses_cpu_runtime_and_skips_qwen3() -> None:
    p = make_plan(None)
    assert not p.gpu_runtime and not p.fetch_qwen3


def test_summary_mentions_gpu_name_or_processor() -> None:
    with_gpu = make_plan(InstalledGpu("RTX 3060", 12288)).summary_lines()
    without = make_plan(None).summary_lines()
    assert "RTX 3060" in with_gpu[0] and "Qwen3-TTS (best" in with_gpu[1]
    assert "using your processor" in without[0]
    assert any("skipped" in line for line in without)


def test_run_nvidia_smi_returns_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd: list[str], **kw: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 0, stdout="Card, 1\n")

    monkeypatch.setattr("subprocess.run", fake_run)
    assert plan._run_nvidia_smi(["x"]) == "Card, 1\n"


@pytest.mark.parametrize(("mb", "expected"), [(8192, "yes"), (None, "no")])
def test_main_gpu_runtime_prints_yes_or_no(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], mb: int | None, expected: str
) -> None:
    monkeypatch.setattr(plan, "read_gpu", lambda: None if mb is None else InstalledGpu("C", mb))
    assert plan.main(["gpu-runtime"]) == 0
    assert capsys.readouterr().out.strip() == expected


@pytest.mark.parametrize(("mb", "expected"), [(8192, ""), (2048, "--skip-qwen3"), (None, "--skip-qwen3")])
def test_main_fetch_flags_prints_skip_flag_only_when_needed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], mb: int | None, expected: str
) -> None:
    monkeypatch.setattr(plan, "read_gpu", lambda: None if mb is None else InstalledGpu("C", mb))
    assert plan.main(["fetch-flags"]) == 0
    assert capsys.readouterr().out.strip() == expected


def test_read_gpu_undecodable_output_is_none() -> None:
    err = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad")
    assert read_gpu(_raiser(err)) is None


def test_parse_nvidia_smi_uses_first_gpu_of_several() -> None:
    assert parse_nvidia_smi("First, 8192\nSecond, 4096\n") == InstalledGpu("First", 8192)


def test_parse_nvidia_smi_handles_crlf() -> None:
    assert parse_nvidia_smi("Card, 8192\r\n") == InstalledGpu("Card", 8192)


def test_parse_nvidia_smi_keeps_comma_inside_name() -> None:
    assert parse_nvidia_smi("Card, Pro, 8192\n") == InstalledGpu("Card, Pro", 8192)
