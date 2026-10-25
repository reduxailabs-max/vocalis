"""Tests for vocalis.engines.piper_tts.PiperTTS.

Uses a FakeCommandRunner instead of a real subprocess, so tests never
touch the real piper binary.
"""
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from vocalis.engines.piper_tts import PiperLoadError, PiperSynthesisError, PiperTTS


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


def _pcm_bytes(samples: list[int]) -> bytes:
    return struct.pack(f"<{len(samples)}h", *samples)


def test_load_succeeds_when_binary_and_model_exist(tmp_path: Path) -> None:
    binary = tmp_path / "piper"
    binary.write_text("")
    model = tmp_path / "voice.onnx"
    model.write_text("")
    engine = PiperTTS(binary_path=binary, model_path=model)
    assert engine.is_loaded is False
    engine.load()
    assert engine.is_loaded is True


def test_load_raises_when_binary_missing(tmp_path: Path) -> None:
    model = tmp_path / "voice.onnx"
    model.write_text("")
    engine = PiperTTS(binary_path=tmp_path / "missing-piper", model_path=model)
    with pytest.raises(PiperLoadError):
        engine.load()


def test_load_raises_when_model_missing(tmp_path: Path) -> None:
    binary = tmp_path / "piper"
    binary.write_text("")
    engine = PiperTTS(binary_path=binary, model_path=tmp_path / "missing.onnx")
    with pytest.raises(PiperLoadError):
        engine.load()


def test_synthesize_before_load_raises(tmp_path: Path) -> None:
    binary = tmp_path / "piper"
    binary.write_text("")
    model = tmp_path / "voice.onnx"
    model.write_text("")
    engine = PiperTTS(binary_path=binary, model_path=model)
    with pytest.raises(RuntimeError):
        engine.synthesize("hello")


def test_synthesize_converts_raw_pcm_to_float32(tmp_path: Path) -> None:
    binary = tmp_path / "piper"
    binary.write_text("")
    model = tmp_path / "voice.onnx"
    model.write_text("")
    runner = FakeCommandRunner(returncode=0, stdout=_pcm_bytes([0, 16384, -16384, 32767]))
    engine = PiperTTS(binary_path=binary, model_path=model, sample_rate=22050, command_runner=runner)
    engine.load()

    audio, sample_rate = engine.synthesize("hello world")

    assert sample_rate == 22050
    assert audio.dtype == np.float32
    np.testing.assert_array_almost_equal(
        audio, np.array([0.0, 0.5, -0.5, 0.999969], dtype=np.float32), decimal=5
    )
    argv, input_bytes, _env = runner.calls[0]
    assert argv == [str(binary), "--model", str(model), "--length_scale", "1.0000", "--output_raw"]
    assert input_bytes == b"hello world"


def test_synthesize_raises_on_nonzero_exit(tmp_path: Path) -> None:
    binary = tmp_path / "piper"
    binary.write_text("")
    model = tmp_path / "voice.onnx"
    model.write_text("")
    runner = FakeCommandRunner(returncode=1, stdout=b"")
    engine = PiperTTS(binary_path=binary, model_path=model, command_runner=runner)
    engine.load()
    with pytest.raises(PiperSynthesisError):
        engine.synthesize("hello")


def test_synthesize_raises_when_runner_raises(tmp_path: Path) -> None:
    binary = tmp_path / "piper"
    binary.write_text("")
    model = tmp_path / "voice.onnx"
    model.write_text("")
    runner = FakeCommandRunner(raise_exc=OSError("piper not found"))
    engine = PiperTTS(binary_path=binary, model_path=model, command_runner=runner)
    engine.load()
    with pytest.raises(PiperSynthesisError):
        engine.synthesize("hello")


def test_unload_clears_loaded_state(tmp_path: Path) -> None:
    binary = tmp_path / "piper"
    binary.write_text("")
    model = tmp_path / "voice.onnx"
    model.write_text("")
    engine = PiperTTS(binary_path=binary, model_path=model)
    engine.load()
    engine.unload()
    assert engine.is_loaded is False


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

    argv, _, _ = runner.calls[0]
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

    argv, _, _ = runner.calls[0]
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
    engine = PiperTTS(
        binary_path=binary, model_path=model, max_cpu_threads=lambda: cap["value"], command_runner=runner
    )
    engine.load()
    cap["value"] = 6

    engine.synthesize("hi")

    _argv, _input_bytes, env = runner.calls[0]
    assert env is not None
    assert env["OMP_NUM_THREADS"] == "6"
