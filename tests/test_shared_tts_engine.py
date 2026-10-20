"""Quick Reader and Document Reader must share one ModelManager engine id,
so switching between them reuses the already-loaded Qwen3-TTS model instead
of unloading and reloading ~1.8 GB."""
import threading
from typing import Any

import numpy as np
import numpy.typing as npt

from tests.fake_output import OutputRecorder, pump_stream_factory
from vocalis.audio.playback import AudioPlayer
from vocalis.features.doc_reader import DocumentReaderFeature
from vocalis.features.quick_reader import QuickReaderFeature
from vocalis.features.reading_coordinator import ReadingCoordinator
from vocalis.model_manager import ModelManager


class _CountingEngine:
    """A fake TTS engine that records how often it is loaded."""

    def __init__(self) -> None:
        self.load_calls = 0
        self._loaded = False

    def load(self) -> None:
        self.load_calls += 1
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def synthesize(self, text: str) -> "tuple[npt.NDArray[np.float32], int]":
        return np.zeros(10, dtype=np.float32), 22050


class _InstantStream:
    def __init__(self, **kwargs: Any) -> None:
        self._cb = kwargs["finished_callback"]

    def start(self) -> None:
        self._cb()

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


class _Clipboard:
    def read(self) -> str:
        return "Hello there."


class _Notifier:
    """Signals `done` once Quick Reader announces it has finished."""

    def __init__(self) -> None:
        self.done = threading.Event()

    def notify(self, title: str, message: str = "") -> None:
        if message == "Done":
            self.done.set()


def test_both_features_build_the_primary_engine_only_once() -> None:
    manager = ModelManager()
    built: "list[_CountingEngine]" = []

    def factory() -> _CountingEngine:
        engine = _CountingEngine()
        built.append(engine)
        return engine

    coordinator = ReadingCoordinator()
    quick_notifier = _Notifier()
    finished = threading.Event()
    quick = QuickReaderFeature(
        model_manager=manager,
        primary_engine_factory=factory,
        fallback_engine=_CountingEngine(),
        clipboard_reader=_Clipboard(),  # type: ignore[arg-type]
        player=AudioPlayer(stream_factory=lambda **kw: _InstantStream(**kw)),
        notifier=quick_notifier,  # type: ignore[arg-type]
        reading_coordinator=coordinator,
    )
    doc = DocumentReaderFeature(
        model_manager=manager,
        primary_engine_factory=factory,
        fallback_engine=_CountingEngine(),
        player=AudioPlayer(stream_factory=pump_stream_factory(OutputRecorder())),
        notifier=_Notifier(),  # type: ignore[arg-type]
        reading_coordinator=coordinator,
        on_status=lambda message: None,
        on_progress=lambda current, total: None,
        on_finished=finished.set,
    )

    quick.on_trigger()
    assert quick_notifier.done.wait(timeout=5.0)
    doc.read_text("Some document text.")
    assert finished.wait(timeout=5.0)

    assert len(built) == 1
    assert built[0].load_calls == 1
    manager.shutdown()

