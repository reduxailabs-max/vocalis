"""Tests for vocalis.engines.qwen3_tts.Qwen3TTS.

Uses a fake model object in place of the real Qwen3TTSModel (which needs
a GPU and a multi-gigabyte download), so tests never touch real hardware.
"""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from vocalis.engines.qwen3_tts import Qwen3TTS, Qwen3TTSLoadError, _max_new_tokens_for


class FakeQwenModel:
    def __init__(self, wavs: list[np.ndarray], sample_rate: int, raise_exc: Exception | None = None) -> None:
        self.wavs = wavs
        self.sample_rate = sample_rate
        self.raise_exc = raise_exc
        self.calls: list[dict[str, object]] = []

    def generate_custom_voice(
        self, text: str, language: str, speaker: str, **kwargs: object
    ) -> tuple[list[np.ndarray], int]:
        self.calls.append({"text": text, "language": language, "speaker": speaker, **kwargs})
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.wavs, self.sample_rate


def test_is_loaded_false_before_load() -> None:
    engine = Qwen3TTS(model_dir=Path("/fake/model/dir"))
    assert engine.is_loaded is False


def test_synthesize_before_load_raises() -> None:
    engine = Qwen3TTS(model_dir=Path("/fake/model/dir"))
    with pytest.raises(RuntimeError):
        engine.synthesize("hello")


def test_load_wraps_failure_in_load_error() -> None:
    engine = Qwen3TTS(model_dir=Path("/nonexistent/model/dir"))
    with patch("vocalis.engines.qwen3_tts.Qwen3TTSModel") as mock_model_cls:
        mock_model_cls.from_pretrained.side_effect = RuntimeError("CUDA out of memory")
        with pytest.raises(Qwen3TTSLoadError):
            engine.load()
    assert engine.is_loaded is False


def test_load_then_synthesize_calls_generate_custom_voice_with_configured_speaker() -> None:
    fake_wav = np.array([0.1, 0.2, -0.1], dtype=np.float32)
    fake_model = FakeQwenModel(wavs=[fake_wav], sample_rate=24000)
    engine = Qwen3TTS(model_dir=Path("/fake/model/dir"), speaker="Ryan", language="English")
    with patch("vocalis.engines.qwen3_tts.Qwen3TTSModel") as mock_model_cls:
        mock_model_cls.from_pretrained.return_value = fake_model
        engine.load()

    assert engine.is_loaded is True
    audio, sample_rate = engine.synthesize("Hello world")

    assert sample_rate == 24000
    np.testing.assert_array_almost_equal(audio, fake_wav)
    assert len(fake_model.calls) == 1
    call = fake_model.calls[0]
    assert call["text"] == "Hello world"
    assert call["language"] == "English"
    assert call["speaker"] == "Ryan"
    assert call["temperature"] == 0.65
    assert call["top_k"] == 20
    assert call["max_new_tokens"] == 200  # "Hello world" is 11 chars * 6 = 66, floored to 200


def test_synthesize_propagates_generation_failure() -> None:
    fake_model = FakeQwenModel(wavs=[], sample_rate=24000, raise_exc=RuntimeError("generation failed"))
    engine = Qwen3TTS(model_dir=Path("/fake/model/dir"))
    with patch("vocalis.engines.qwen3_tts.Qwen3TTSModel") as mock_model_cls:
        mock_model_cls.from_pretrained.return_value = fake_model
        engine.load()
    with pytest.raises(RuntimeError):
        engine.synthesize("hello")


def test_unload_clears_loaded_state() -> None:
    fake_model = FakeQwenModel(wavs=[np.zeros(1, dtype=np.float32)], sample_rate=24000)
    engine = Qwen3TTS(model_dir=Path("/fake/model/dir"))
    with patch("vocalis.engines.qwen3_tts.Qwen3TTSModel") as mock_model_cls:
        mock_model_cls.from_pretrained.return_value = fake_model
        engine.load()
    engine.unload()
    assert engine.is_loaded is False


def test_unload_releases_gpu_memory_when_use_gpu_true() -> None:
    fake_model = FakeQwenModel(wavs=[np.zeros(1, dtype=np.float32)], sample_rate=24000)
    engine = Qwen3TTS(model_dir=Path("/fake/model/dir"), use_gpu=True)
    with patch("vocalis.engines.qwen3_tts.Qwen3TTSModel") as mock_model_cls:
        mock_model_cls.from_pretrained.return_value = fake_model
        engine.load()

    with patch("vocalis.engines.qwen3_tts.torch.cuda.empty_cache") as mock_empty_cache:
        engine.unload()

    mock_empty_cache.assert_called_once()


def test_unload_does_not_touch_cuda_when_use_gpu_false() -> None:
    engine = Qwen3TTS(model_dir=Path("/fake/model/dir"), use_gpu=False)

    with patch("vocalis.engines.qwen3_tts.torch.cuda.empty_cache") as mock_empty_cache:
        engine.unload()

    mock_empty_cache.assert_not_called()


def test_speaker_can_be_a_function_read_on_every_synthesis() -> None:
    voice = {"name": "Ryan"}
    fake_model = FakeQwenModel(wavs=[np.zeros(1, dtype=np.float32)], sample_rate=24000)
    engine = Qwen3TTS(model_dir=Path("/fake/model/dir"), speaker=lambda: voice["name"], use_gpu=False)
    with patch("vocalis.engines.qwen3_tts.Qwen3TTSModel") as mock_model_cls:
        mock_model_cls.from_pretrained.return_value = fake_model
        engine.load()
    engine.synthesize("hello")
    voice["name"] = "Dylan"
    engine.synthesize("hello")
    assert [call["speaker"] for call in fake_model.calls] == ["Ryan", "Dylan"]


def _sine(seconds: float = 2.0, rate: int = 24000) -> np.ndarray:
    """A real, long-enough-to-stretch waveform (librosa needs >4096 samples)."""
    t = np.arange(int(rate * seconds)) / rate
    return (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)


def test_max_new_tokens_scales_with_text_length_within_bounds() -> None:
    assert _max_new_tokens_for("hi") == 200  # floor
    assert _max_new_tokens_for("x" * 100) == 600  # 100 * 6, well inside bounds
    assert _max_new_tokens_for("x" * 1000) == 2048  # ceiling


def test_speed_below_one_is_not_applied() -> None:
    """Qwen3-TTS's own stretch only ever speeds up -- never slows down."""
    audio = _sine()
    fake_model = FakeQwenModel(wavs=[audio], sample_rate=24000)
    engine = Qwen3TTS(model_dir=Path("/fake/model/dir"), speed=0.7)
    with patch("vocalis.engines.qwen3_tts.Qwen3TTSModel") as mock_model_cls:
        mock_model_cls.from_pretrained.return_value = fake_model
        engine.load()

    result, _ = engine.synthesize("hello there, this sentence needs to be long enough to stretch")

    assert len(result) == len(audio)


def test_speed_above_one_point_two_is_capped() -> None:
    """A requested speed above 1.2x is capped there, not applied in full."""
    audio = _sine()
    fake_model = FakeQwenModel(wavs=[audio], sample_rate=24000)
    engine = Qwen3TTS(model_dir=Path("/fake/model/dir"), speed=1.6)
    with patch("vocalis.engines.qwen3_tts.Qwen3TTSModel") as mock_model_cls:
        mock_model_cls.from_pretrained.return_value = fake_model
        engine.load()

    result, _ = engine.synthesize("hello there, this sentence needs to be long enough to stretch")

    assert abs(len(result) - len(audio) / 1.2) < 0.03 * len(audio)


def test_speed_between_one_and_one_point_two_is_applied_as_given() -> None:
    audio = _sine()
    fake_model = FakeQwenModel(wavs=[audio], sample_rate=24000)
    engine = Qwen3TTS(model_dir=Path("/fake/model/dir"), speed=1.1)
    with patch("vocalis.engines.qwen3_tts.Qwen3TTSModel") as mock_model_cls:
        mock_model_cls.from_pretrained.return_value = fake_model
        engine.load()

    result, _ = engine.synthesize("hello there, this sentence needs to be long enough to stretch")

    assert abs(len(result) - len(audio) / 1.1) < 0.03 * len(audio)


def test_speed_can_be_a_live_callable() -> None:
    """A callable speed is re-read on every synthesis call."""
    audio = _sine()
    fake_model = FakeQwenModel(wavs=[audio], sample_rate=24000)
    speeds = iter([1.1, 1.2])
    engine = Qwen3TTS(model_dir=Path("/fake/model/dir"), speed=lambda: next(speeds))
    with patch("vocalis.engines.qwen3_tts.Qwen3TTSModel") as mock_model_cls:
        mock_model_cls.from_pretrained.return_value = fake_model
        engine.load()

    first, _ = engine.synthesize("hello there, this sentence needs to be long enough to stretch")
    second, _ = engine.synthesize("hello there, this sentence needs to be long enough to stretch")

    assert abs(len(first) - len(audio) / 1.1) < 0.03 * len(audio)
    assert abs(len(second) - len(audio) / 1.2) < 0.03 * len(audio)


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
