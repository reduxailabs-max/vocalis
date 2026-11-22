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

REAL_MODEL_DIR = Path.home() / ".local/share/voxtype/models/parakeet-tdt-0.6b-v3-int8"
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


# ---- alphabet filter (stops accented English being heard as Russian) --------

from vocalis.engines.parakeet import _is_non_latin_token  # noqa: E402


class _FakeAsr:
    """Mimics the parts of an onnx-asr TDT model the filter wraps."""

    def __init__(self) -> None:
        self._vocab = {0: " the", 1: " это", 2: " é", 3: " καλ", 4: "<blk>"}
        self.calls = 0

    def _decode(self, *_args: object, **_kwargs: object):  # type: ignore[no-untyped-def]
        self.calls += 1
        return np.array([0.1, 0.9, 0.2, 0.8, 0.0], dtype=np.float32), 1, "state"


class _FakeModel:
    def __init__(self, asr: object) -> None:
        self.asr = asr


def _engine_with(asr: object, latin_only) -> ParakeetSTT:  # type: ignore[no-untyped-def]
    engine = ParakeetSTT(model_dir=REAL_MODEL_DIR, use_gpu=False, latin_only=latin_only)
    engine._model = _FakeModel(asr)  # type: ignore[assignment]  # noqa: SLF001
    engine._install_alphabet_filter()  # noqa: SLF001
    return engine


def test_non_latin_token_detection() -> None:
    assert _is_non_latin_token(" это")
    assert _is_non_latin_token(" καλ")
    assert not _is_non_latin_token(" the")
    assert not _is_non_latin_token(" é")
    assert not _is_non_latin_token("<blk>")


def test_filter_hides_non_latin_tokens_when_switched_on() -> None:
    asr = _FakeAsr()
    _engine_with(asr, lambda: True)
    logits, step, state = asr._decode()
    assert logits[1] == -np.inf and logits[3] == -np.inf
    assert logits[0] == np.float32(0.1) and logits[2] == np.float32(0.2) and logits[4] == 0.0
    assert (step, state) == (1, "state")


def test_filter_does_nothing_when_switched_off_and_follows_changes_live() -> None:
    asr = _FakeAsr()
    flag = {"on": False}
    _engine_with(asr, lambda: flag["on"])
    assert np.isfinite(asr._decode()[0]).all()
    flag["on"] = True
    assert not np.isfinite(asr._decode()[0]).all()


def test_filter_is_skipped_with_an_unknown_model_layout(caplog: pytest.LogCaptureFixture) -> None:
    engine = ParakeetSTT(model_dir=REAL_MODEL_DIR, use_gpu=False, latin_only=lambda: True)
    engine._model = _FakeModel(object())  # type: ignore[assignment]  # noqa: SLF001
    engine._install_alphabet_filter()  # noqa: SLF001  - must not raise
    assert "alphabet filter unavailable" in caplog.text
