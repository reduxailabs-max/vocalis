"""Quick Reader and Document Reader sharing one real ReadingCoordinator,
reporting to a fake status hub that follows StatusHub's rule: a begin
while a reading is active is ignored; end clears it."""
import numpy as np

from vocalis.audio.playback import AudioPlayer
from vocalis.features.doc_reader import DocumentReaderFeature
from vocalis.features.quick_reader import QuickReaderFeature
from vocalis.features.reading_coordinator import ReadingCoordinator
from vocalis.model_manager import ModelManager
from tests.test_doc_reader import (
    FakeSynthesizer as DocSynth,
    RecordingNotifier,
    StreamStartRecorder,
    _endless_stream_factory,
    wait_until as doc_wait,
)
from tests.test_quick_reader import (
    FakeClipboardReader,
    FakeNotifier,
    FakeSynthesizer as QuickSynth,
    ResettablePlayer,
)


class FakeHub:
    """Mimics StatusHub: begin while active is ignored; end clears."""

    def __init__(self) -> None:
        self.source: "str | None" = None

    def begin(self, source: str) -> None:
        if self.source is None:
            self.source = source

    def end(self) -> None:
        self.source = None


def _build() -> "tuple[QuickReaderFeature, DocumentReaderFeature, FakeHub, list[str], ResettablePlayer, StreamStartRecorder]":
    log: "list[str]" = []
    hub = FakeHub()
    coordinator = ReadingCoordinator()

    def quick_changed(active: bool) -> None:
        log.append(f"quick {active}")
        hub.begin("Quick read") if active else hub.end()

    def doc_started() -> None:
        log.append("doc started")
        hub.begin("Document")

    quick_player = ResettablePlayer(delay_seconds=3.0)
    quick = QuickReaderFeature(
        model_manager=ModelManager(),
        primary_engine_factory=lambda: QuickSynth(audio=np.zeros(3, dtype=np.float32)),
        fallback_engine=QuickSynth(audio=np.zeros(1, dtype=np.float32)),
        clipboard_reader=FakeClipboardReader("quick text"),  # type: ignore[arg-type]
        player=quick_player,  # type: ignore[arg-type]
        notifier=FakeNotifier(),  # type: ignore[arg-type]
        reading_coordinator=coordinator,
        on_reading_changed=quick_changed,
    )
    recorder = StreamStartRecorder()
    doc = DocumentReaderFeature(
        model_manager=ModelManager(),
        primary_engine_factory=lambda: DocSynth(),
        fallback_engine=DocSynth(),
        player=AudioPlayer(stream_factory=_endless_stream_factory(recorder)),
        notifier=RecordingNotifier(),  # type: ignore[arg-type]
        reading_coordinator=coordinator,
        on_status=lambda _s: None,
        on_progress=lambda _c, _t: None,
        on_finished=hub.end,
        on_started=doc_started,
    )
    return quick, doc, hub, log, quick_player, recorder


def test_document_read_preempting_quick_read_leaves_hub_active_as_document() -> None:
    quick, doc, hub, log, quick_player, _rec = _build()
    quick.on_trigger()
    doc_wait(lambda: len(quick_player.play_calls) == 1)
    assert hub.source == "Quick read"

    doc.read_text("Hello there.")

    assert log == ["quick True", "quick False", "doc started"]
    assert hub.source == "Document"
    doc.stop()


def test_quick_read_preempting_document_read_leaves_hub_active_as_quick() -> None:
    quick, doc, hub, _log, quick_player, recorder = _build()
    doc.read_text("Hello there.")
    assert recorder.started.wait(timeout=5.0)
    assert hub.source == "Document"

    quick.on_trigger()

    assert hub.source == "Quick read"
    quick.stop()
    doc.stop()
