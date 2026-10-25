"""Tests for vocalis.engines.kokoro_tts — the real kokoro_onnx / onnxruntime are replaced by fakes."""
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vocalis.audio.pace import MAX_SPEED
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


def _engine(tmp_path: Path, kokoro: _FakeKokoro | None = None, **kwargs: Any) -> tuple[KokoroTTS, list[tuple[Path, bool, "int | None", "int | None"]], _FakeKokoro]:
    model, voices = _files(tmp_path)
    built: list[tuple[Path, bool, "int | None", "int | None"]] = []
    fake = kokoro or _FakeKokoro()

    def session_builder(
        path: Path, use_gpu: bool, intra_op_num_threads: "int | None" = None, gpu_mem_limit_mb: "int | None" = None,
    ) -> object:
        built.append((path, use_gpu, intra_op_num_threads, gpu_mem_limit_mb))
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

    def boom(_path: Path, _gpu: bool, _threads: "int | None" = None, _vram: "int | None" = None) -> object:
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


def test_default_speed_is_neutral(tmp_path: Path) -> None:
    """Without a configured speed, Kokoro is asked for its natural pace."""
    engine, _, fake = _engine(tmp_path)
    engine.load()
    engine.synthesize("hello")
    assert fake.calls[0][2] == 1.0


def test_synthesize_passes_the_configured_speed(tmp_path: Path) -> None:
    """The configured speed is handed straight to Kokoro."""
    engine, _, fake = _engine(tmp_path, speed=1.3)
    engine.load()
    engine.synthesize("hello")
    assert fake.calls[0][2] == 1.3


def test_speed_is_clamped_to_the_supported_range(tmp_path: Path) -> None:
    """An out-of-range speed is clamped to MAX_SPEED before reaching Kokoro."""
    engine, _, fake = _engine(tmp_path, speed=9.0)
    engine.load()
    engine.synthesize("hello")
    assert fake.calls[0][2] == MAX_SPEED


def test_speed_can_be_a_live_callable(tmp_path: Path) -> None:
    """A callable speed is re-read on every synthesis call."""
    speeds = iter([0.8, 1.4])
    engine, _, fake = _engine(tmp_path, speed=lambda: next(speeds))
    engine.load()
    engine.synthesize("one")
    engine.synthesize("two")
    assert [call[2] for call in fake.calls] == [0.8, 1.4]


def test_load_passes_the_cpu_thread_cap_to_the_session_builder(tmp_path: Path) -> None:
    engine, built, _ = _engine(tmp_path, max_cpu_threads=4)
    engine.load()
    assert built[0][2] == 4


def test_load_passes_no_thread_cap_when_unset(tmp_path: Path) -> None:
    engine, built, _ = _engine(tmp_path)
    engine.load()
    assert built[0][2] is None


def test_load_passes_the_gpu_vram_ceiling_to_the_session_builder(tmp_path: Path) -> None:
    engine, built, _ = _engine(tmp_path, gpu_vram_ceiling_mb=512)
    engine.load()
    assert built[0][3] == 512


def test_thread_cap_can_be_a_live_callable(tmp_path: Path) -> None:
    cap = {"value": 2}
    engine, built, _ = _engine(tmp_path, max_cpu_threads=lambda: cap["value"])
    cap["value"] = 6
    engine.load()
    assert built[0][2] == 6


class _FakeSessionOptions:
    def __init__(self) -> None:
        self.intra_op_num_threads: "int | None" = None


class _FakeOrt:
    def __init__(self, providers: list) -> None:  # type: ignore[type-arg]
        self._providers = providers
        self.sessions: list[dict] = []  # type: ignore[type-arg]

    def get_available_providers(self) -> list:  # type: ignore[type-arg]
        return self._providers

    def SessionOptions(self) -> _FakeSessionOptions:
        return _FakeSessionOptions()

    def InferenceSession(self, path: str, sess_options: Any, providers: list) -> Any:  # type: ignore[type-arg]
        self.sessions.append({"path": path, "sess_options": sess_options, "providers": providers})
        return object()


def test_default_session_builder_sets_intra_op_num_threads(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    from vocalis.engines.kokoro_tts import _default_session_builder

    fake_ort = _FakeOrt(["CPUExecutionProvider"])
    monkeypatch.setattr("vocalis.engines.kokoro_tts.importlib.import_module", lambda _name: fake_ort)
    _default_session_builder(tmp_path / "m.onnx", False, intra_op_num_threads=4)
    assert fake_ort.sessions[0]["sess_options"].intra_op_num_threads == 4


def test_default_session_builder_sets_gpu_mem_limit_on_cuda(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    from vocalis.engines.kokoro_tts import _default_session_builder

    fake_ort = _FakeOrt(["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr("vocalis.engines.kokoro_tts.importlib.import_module", lambda _name: fake_ort)
    _default_session_builder(tmp_path / "m.onnx", True, gpu_mem_limit_mb=512)
    providers = fake_ort.sessions[0]["providers"]
    assert providers[0] == ("CUDAExecutionProvider", {"gpu_mem_limit": 512 * 1024 * 1024})


def test_default_session_builder_skips_gpu_mem_limit_when_not_given(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    from vocalis.engines.kokoro_tts import _default_session_builder

    fake_ort = _FakeOrt(["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr("vocalis.engines.kokoro_tts.importlib.import_module", lambda _name: fake_ort)
    _default_session_builder(tmp_path / "m.onnx", True)
    providers = fake_ort.sessions[0]["providers"]
    assert providers[0] == ("CUDAExecutionProvider", {})
