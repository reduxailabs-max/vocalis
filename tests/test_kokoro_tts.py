"""Tests for vocalis.engines.kokoro_tts — the real kokoro_onnx / onnxruntime are replaced by fakes."""
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vocalis.engines.kokoro_tts import KokoroLoadError, KokoroSynthesisError, KokoroTTS


class _FakeKokoro:
    """Records create() calls and returns a short silent clip."""

    def __init__(self, fail: bool = False) -> None:
        self.calls: list[tuple[str, str, float, str]] = []
        self._fail = fail

    def create(self, text: str, voice: str, speed: float, lang: str) -> tuple[Any, int]:
        if self._fail:
            raise AssertionError("Voice not found")
        self.calls.append((text, voice, speed, lang))
        return np.zeros((2400, 1), dtype=np.float64), 24000


def _files(tmp_path: Path) -> tuple[Path, Path]:
    model, voices = tmp_path / "kokoro-v1.0.onnx", tmp_path / "voices-v1.0.bin"
    model.write_bytes(b"x")
    voices.write_bytes(b"x")
    return model, voices


def _engine(tmp_path: Path, kokoro: _FakeKokoro | None = None, **kwargs: Any) -> tuple[KokoroTTS, list[tuple[Path, bool]], _FakeKokoro]:
    model, voices = _files(tmp_path)
    built: list[tuple[Path, bool]] = []
    fake = kokoro or _FakeKokoro()

    def session_builder(path: Path, use_gpu: bool) -> object:
        built.append((path, use_gpu))
        return object()

    engine = KokoroTTS(model, voices, session_builder=session_builder, kokoro_factory=lambda _s, _v: fake, **kwargs)
    return engine, built, fake


def test_load_builds_the_session_and_marks_the_engine_loaded(tmp_path: Path) -> None:
    """load() creates the ONNX session for the model file."""
    engine, built, _ = _engine(tmp_path)
    assert engine.is_loaded is False
    engine.load()
    assert engine.is_loaded is True
    assert built[0][0].name == "kokoro-v1.0.onnx"


def test_load_with_a_missing_model_file_raises_a_clear_error(tmp_path: Path) -> None:
    """A missing model file fails fast with KokoroLoadError."""
    engine = KokoroTTS(tmp_path / "no.onnx", tmp_path / "no.bin")
    with pytest.raises(KokoroLoadError, match="no.onnx"):
        engine.load()


def test_load_wraps_any_library_failure(tmp_path: Path) -> None:
    """Whatever the ONNX library raises becomes KokoroLoadError."""
    model, voices = _files(tmp_path)

    def boom(_path: Path, _gpu: bool) -> object:
        raise RuntimeError("cuda exploded")

    engine = KokoroTTS(model, voices, session_builder=boom)
    with pytest.raises(KokoroLoadError, match="cuda exploded"):
        engine.load()
    assert engine.is_loaded is False


def test_use_gpu_is_decided_when_loading(tmp_path: Path) -> None:
    """A callable use_gpu is evaluated at load() time."""
    flag = {"gpu": False}
    engine, built, _ = _engine(tmp_path, use_gpu=lambda: flag["gpu"])
    flag["gpu"] = True
    engine.load()
    assert built[0][1] is True


def test_synthesize_returns_flat_float32_audio_and_the_sample_rate(tmp_path: Path) -> None:
    """Audio comes back as a 1-D float32 array with Kokoro's sample rate."""
    engine, _, fake = _engine(tmp_path)
    engine.load()
    audio, rate = engine.synthesize("Hello there.")
    assert audio.dtype == np.float32 and audio.ndim == 1 and rate == 24000
    assert fake.calls == [("Hello there.", "af_heart", 1.0, "en-us")]


def test_british_voices_use_british_english(tmp_path: Path) -> None:
    """Voices starting with b use the en-gb phonemes."""
    engine, _, fake = _engine(tmp_path, voice="bf_emma")
    engine.load()
    engine.synthesize("Hello.")
    assert fake.calls[0][3] == "en-gb"


def test_the_voice_is_read_live_on_every_synthesis(tmp_path: Path) -> None:
    """A callable voice follows Settings changes without reloading."""
    current = {"voice": "af_heart"}
    engine, _, fake = _engine(tmp_path, voice=lambda: current["voice"])
    engine.load()
    engine.synthesize("a")
    current["voice"] = "am_adam"
    engine.synthesize("b")
    assert [call[1] for call in fake.calls] == ["af_heart", "am_adam"]


def test_an_unknown_voice_fails_cleanly(tmp_path: Path) -> None:
    """A voice missing from the voices file is a KokoroSynthesisError, not silence."""
    engine, _, _ = _engine(tmp_path, kokoro=_FakeKokoro(fail=True), voice="nope")
    engine.load()
    with pytest.raises(KokoroSynthesisError, match="Voice not found"):
        engine.synthesize("Hello.")


def test_synthesize_before_load_fails_cleanly(tmp_path: Path) -> None:
    """Using the engine before load() is a clear error."""
    engine, _, _ = _engine(tmp_path)
    with pytest.raises(KokoroSynthesisError, match="not loaded"):
        engine.synthesize("Hello.")


def test_unload_releases_the_model(tmp_path: Path) -> None:
    """unload() drops the model so memory can be freed."""
    engine, _, _ = _engine(tmp_path)
    engine.load()
    engine.unload()
    assert engine.is_loaded is False


def _loaded(tmp_path: Path, **kwargs: Any) -> tuple[KokoroTTS, _FakeKokoro]:
    """An engine that is already loaded, plus its fake synthesizer."""
    engine, _, fake = _engine(tmp_path, **kwargs)
    engine.load()
    return engine, fake


def test_long_text_without_punctuation_is_split_and_nothing_is_lost(tmp_path: Path) -> None:
    """Pieces stay short, keep every word, and the audio is the clips plus gaps."""
    engine, fake = _loaded(tmp_path)
    words = [f"word{i}" for i in range(150)]
    text = " ".join(words)
    assert len(text) > 900

    audio, rate = engine.synthesize(text)

    assert len(fake.calls) > 1
    assert all(len(call[0]) <= 250 for call in fake.calls)
    assert " ".join(call[0] for call in fake.calls).split() == words
    gap = int(24000 * 0.06)
    assert rate == 24000
    assert len(audio) == 2400 * len(fake.calls) + gap * (len(fake.calls) - 1)


def test_text_with_no_spaces_is_hard_cut_and_terminates(tmp_path: Path) -> None:
    """One giant word is cut every 250 characters."""
    engine, fake = _loaded(tmp_path)

    engine.synthesize("x" * 600)

    assert [len(call[0]) for call in fake.calls] == [250, 250, 100]


def test_punctuated_text_splits_at_sentence_boundaries(tmp_path: Path) -> None:
    """Sentences are packed whole, never cut in the middle."""
    engine, fake = _loaded(tmp_path)
    sentence = "This is a sentence of moderate length that ends here."
    text = " ".join([sentence] * 10)

    engine.synthesize(text)

    assert len(fake.calls) > 1
    assert all(call[0].endswith(".") and len(call[0]) <= 250 for call in fake.calls)


def test_short_text_makes_exactly_one_call(tmp_path: Path) -> None:
    """Text that fits is passed through untouched."""
    engine, fake = _loaded(tmp_path)

    engine.synthesize("Hello there.")

    assert [call[0] for call in fake.calls] == ["Hello there."]


def test_a_raising_voice_callable_becomes_a_synthesis_error(tmp_path: Path) -> None:
    """A voice function that fails is reported as KokoroSynthesisError."""
    def bad_voice() -> str:
        raise RuntimeError("no voice")

    engine, _ = _loaded(tmp_path, voice=bad_voice)
    with pytest.raises(KokoroSynthesisError, match="no voice"):
        engine.synthesize("Hello.")


def test_a_raising_use_gpu_callable_becomes_a_load_error(tmp_path: Path) -> None:
    """A use_gpu function that fails is reported as KokoroLoadError."""
    def bad_gpu() -> bool:
        raise RuntimeError("no gpu answer")

    engine, _, _ = _engine(tmp_path, use_gpu=bad_gpu)
    with pytest.raises(KokoroLoadError, match="no gpu answer"):
        engine.load()


def test_missing_file_means_no_builder_is_called(tmp_path: Path) -> None:
    """The file check happens before any builder runs."""
    calls: list[str] = []

    def session_builder(_p: Path, _g: bool) -> object:
        calls.append("session")
        return object()

    def factory(_s: Any, _v: Path) -> object:
        calls.append("factory")
        return object()

    engine = KokoroTTS(tmp_path / "no.onnx", tmp_path / "no.bin", session_builder=session_builder, kokoro_factory=factory)
    with pytest.raises(KokoroLoadError):
        engine.load()
    assert calls == []
