"""Tests for vocalis.engines.qwen3_tts.Qwen3TTS.

Uses a fake model object in place of the real Qwen3TTSModel (which needs
a GPU and a multi-gigabyte download), so tests never touch real hardware.
"""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from vocalis.engines.qwen3_tts import Qwen3TTS, Qwen3TTSLoadError


class FakeQwenModel:
    def __init__(self, wavs: list[np.ndarray], sample_rate: int, raise_exc: Exception | None = None) -> None:
        self.wavs = wavs
        self.sample_rate = sample_rate
        self.raise_exc = raise_exc
        self.calls: list[dict[str, object]] = []

    def generate_custom_voice(self, text: str, language: str, speaker: str) -> tuple[list[np.ndarray], int]:
        self.calls.append({"text": text, "language": language, "speaker": speaker})
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
    assert fake_model.calls == [{"text": "Hello world", "language": "English", "speaker": "Ryan"}]


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
