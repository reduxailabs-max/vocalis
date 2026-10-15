"""Tests for vocalis.engines.parakeet.ParakeetSTT.

The real Parakeet model files already exist on this development machine
(inherited from the existing Voxtype installation). Tests that need the
real model are skipped automatically on a machine that doesn't have them,
so this suite stays portable while still proving real correctness here.
"""
from pathlib import Path

import numpy as np
import pytest

from vocalis.engines.parakeet import ParakeetLoadError, ParakeetSTT

REAL_MODEL_DIR = Path("/home/redx/.local/share/voxtype/models/parakeet-tdt-0.6b-v3-int8")
_real_model_available = REAL_MODEL_DIR.exists()


def test_is_loaded_false_before_load() -> None:
    engine = ParakeetSTT(model_dir=REAL_MODEL_DIR)
    assert engine.is_loaded is False


def test_transcribe_before_load_raises() -> None:
    engine = ParakeetSTT(model_dir=REAL_MODEL_DIR)
    with pytest.raises(RuntimeError):
        engine.transcribe(np.zeros(16000, dtype=np.float32), sample_rate=16000)


def test_load_missing_model_dir_raises_parakeet_load_error(tmp_path: Path) -> None:
    engine = ParakeetSTT(model_dir=tmp_path / "does_not_exist", use_gpu=False)
    with pytest.raises(ParakeetLoadError):
        engine.load()


def test_unload_before_load_is_safe() -> None:
    engine = ParakeetSTT(model_dir=REAL_MODEL_DIR)
    engine.unload()  # must not raise
    assert engine.is_loaded is False


@pytest.mark.skipif(
    not _real_model_available, reason="Real Parakeet model files not present on this machine"
)
def test_load_and_transcribe_real_model_on_cpu() -> None:
    engine = ParakeetSTT(model_dir=REAL_MODEL_DIR, use_gpu=False)
    engine.load()
    try:
        assert engine.is_loaded is True
        audio = (np.random.randn(16000) * 0.001).astype(np.float32)
        result = engine.transcribe(audio, sample_rate=16000)
        assert isinstance(result, str)
    finally:
        engine.unload()
    assert engine.is_loaded is False


@pytest.mark.skipif(
    not _real_model_available, reason="Real Parakeet model files not present on this machine"
)
def test_load_and_transcribe_real_model_on_gpu() -> None:
    engine = ParakeetSTT(model_dir=REAL_MODEL_DIR, use_gpu=True)
    engine.load()
    try:
        assert engine.is_loaded is True
        audio = (np.random.randn(16000) * 0.001).astype(np.float32)
        result = engine.transcribe(audio, sample_rate=16000)
        assert isinstance(result, str)
    finally:
        engine.unload()
