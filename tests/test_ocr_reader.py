"""Tests for vocalis.features.ocr_reader.OcrReaderFeature (fakes for every collaborator)."""
import threading
import time
from typing import Callable

from vocalis.engines.tesseract_ocr import TesseractError
from vocalis.features.ocr_reader import OcrReaderFeature
from vocalis.output.screen_capture import ScreenCaptureUnavailable


def wait_until(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out waiting for condition"
        time.sleep(0.005)


class FakeCapture:
    def __init__(self, result: bytes | None = b"PNG", raises: Exception | None = None) -> None:
        self.result = result
        self.raises = raises
        self.gate: threading.Event | None = None
        self.started = threading.Event()
        self.calls = 0
        self.threads: "list[threading.Thread]" = []

    def capture(self) -> bytes | None:
        self.calls += 1
        self.threads.append(threading.current_thread())
        self.started.set()
        if self.gate is not None:
            assert self.gate.wait(5.0), "gate never released"
        if self.raises is not None:
            raise self.raises
        return self.result


class FakeOcr:
    def __init__(self, text: str = "Hello world", raises: Exception | None = None) -> None:
        self.text = text
        self.raises = raises
        self.received: "list[bytes]" = []

    def recognize(self, png_bytes: bytes) -> str:
        self.received.append(png_bytes)
        if self.raises is not None:
            raise self.raises
        return self.text


class FakeClipboard:
    def __init__(self, ok: bool = True) -> None:
        self.ok = ok
        self.written: "list[str]" = []

    def write(self, text: str) -> bool:
        self.written.append(text)
        return self.ok


class FakeReader:
    def __init__(self, events: "list[str] | None" = None) -> None:
        self.is_reading = False
        self.generation = 0
        self.read_text_calls: "list[str]" = []
        self.stop_calls = 0
        self.events = events if events is not None else []

    def read_text(self, text: str) -> None:
        self.events.append("read_text")
        self.read_text_calls.append(text)
        self.generation += 1
        self.is_reading = True

    def stop(self) -> None:
        self.stop_calls += 1
        self.is_reading = False


class RecordingNotifier:
    def __init__(self) -> None:
        self.messages: "list[tuple[str, str]]" = []

    def notify(self, title: str, message: str = "") -> None:
        self.messages.append((title, message))

    @property
    def bodies(self) -> "list[str]":
        return [message for _title, message in self.messages]


def make(
    capture: FakeCapture | None = None,
    ocr: FakeOcr | None = None,
    clipboard: FakeClipboard | None = None,
    reader: FakeReader | None = None,
    on_reading_starting: "Callable[[str], None] | None" = None,
) -> "tuple[OcrReaderFeature, FakeCapture, FakeOcr, FakeClipboard, FakeReader, RecordingNotifier]":
    capture = capture or FakeCapture()
    ocr = ocr or FakeOcr()
    clipboard = clipboard or FakeClipboard()
    reader = reader or FakeReader()
    notifier = RecordingNotifier()
    feature = OcrReaderFeature(
        capture=capture,
        ocr=ocr,
        clipboard=clipboard,
        reader=reader,
        notifier=notifier,  # type: ignore[arg-type]
        on_reading_starting=on_reading_starting,
    )
    return feature, capture, ocr, clipboard, reader, notifier


def run_once(feature: OcrReaderFeature) -> None:
    feature.on_trigger()
    wait_until(lambda: not feature.is_busy)


def test_happy_path_captures_recognises_copies_and_reads() -> None:
    feature, capture, ocr, clipboard, reader, notifier = make()

    run_once(feature)

    assert ocr.received == [b"PNG"]
    assert clipboard.written == ["Hello world"]
    assert reader.read_text_calls == ["Hello world"]
    assert "Hello world" in notifier.bodies


def test_capture_never_runs_on_the_callers_thread() -> None:
    feature, capture, *_ = make()
    run_once(feature)
    assert capture.threads and capture.threads[0] is not threading.current_thread()


def test_cancelled_capture_says_cancelled_and_does_nothing_else() -> None:
    feature, capture, ocr, clipboard, reader, notifier = make(capture=FakeCapture(result=None))

    run_once(feature)

    assert "Cancelled" in notifier.bodies
    assert ocr.received == [] and clipboard.written == [] and reader.read_text_calls == []


def test_a_cancelled_capture_leaves_the_feature_ready_for_the_next_press() -> None:
    capture = FakeCapture(result=None)
    feature, *_ = make(capture=capture)
    run_once(feature)
    run_once(feature)
    assert capture.calls == 2


def test_missing_capture_tool_shows_its_install_hint() -> None:
    capture = FakeCapture(raises=ScreenCaptureUnavailable("No screen-capture tool found. Install one of: x"))
    feature, _c, _o, _cb, reader, notifier = make(capture=capture)

    run_once(feature)

    assert "No screen-capture tool found. Install one of: x" in notifier.bodies
    assert reader.read_text_calls == []


def test_tesseract_failure_shows_its_message() -> None:
    feature, _c, _o, clipboard, reader, notifier = make(ocr=FakeOcr(raises=TesseractError("Tesseract is not installed.")))

    run_once(feature)

    assert "Tesseract is not installed." in notifier.bodies
    assert clipboard.written == [] and reader.read_text_calls == []


def test_blank_area_says_no_text_detected() -> None:
    feature, _c, _o, clipboard, reader, notifier = make(ocr=FakeOcr(text=""))

    run_once(feature)

    assert "No text detected" in notifier.bodies
    assert clipboard.written == [] and reader.read_text_calls == []


def test_one_or_two_stray_characters_count_as_no_text() -> None:
    feature, _c, _o, clipboard, reader, notifier = make(ocr=FakeOcr(text="ab"))

    run_once(feature)

    assert "No text detected" in notifier.bodies
    assert reader.read_text_calls == []


def test_clipboard_failure_does_not_stop_the_reading() -> None:
    feature, _c, _o, clipboard, reader, _n = make(clipboard=FakeClipboard(ok=False))

    run_once(feature)

    assert clipboard.written == ["Hello world"]
    assert reader.read_text_calls == ["Hello world"]


def test_a_whole_screen_of_text_is_copied_and_read_in_full() -> None:
    long_text = "word " * 2000
    feature, _c, _o, clipboard, reader, notifier = make(ocr=FakeOcr(text=long_text))

    run_once(feature)

    assert clipboard.written == [long_text]
    assert reader.read_text_calls == [long_text]
    preview = [body for body in notifier.bodies if body.startswith("word")]
    assert len(preview) == 1 and len(preview[0]) <= 81 and preview[0].endswith("…")


def test_unexpected_error_is_reported_and_the_feature_stays_usable() -> None:
    feature, _c, ocr, _cb, reader, notifier = make(ocr=FakeOcr(raises=ValueError("boom")))

    run_once(feature)

    assert "Something went wrong reading the screen" in notifier.bodies
    assert not feature.is_busy


def test_on_reading_starting_is_called_with_the_text_before_reading_begins() -> None:
    events: "list[str]" = []
    reader = FakeReader(events=events)
    feature, *_ = make(reader=reader, on_reading_starting=lambda text: events.append(f"starting:{text}"))

    run_once(feature)

    assert events == ["starting:Hello world", "read_text"]


def test_pressing_again_while_the_box_is_being_drawn_is_ignored() -> None:
    capture = FakeCapture()
    capture.gate = threading.Event()
    feature, *_ = make(capture=capture)

    feature.on_trigger()
    assert capture.started.wait(5.0)
    feature.on_trigger()  # user presses again while still dragging
    capture.gate.set()
    wait_until(lambda: not feature.is_busy)

    assert capture.calls == 1


def test_pressing_again_while_its_own_reading_plays_stops_it() -> None:
    feature, capture, _o, _cb, reader, _n = make()
    run_once(feature)
    assert reader.is_reading is True

    feature.on_trigger()
    assert not feature.is_busy
    wait_until(lambda: reader.stop_calls == 1)

    assert capture.calls == 1  # no new capture was started


def test_pressing_while_a_document_reading_it_did_not_start_plays_starts_a_capture() -> None:
    reader = FakeReader()
    reader.is_reading = True  # e.g. a file opened from the Ctrl+Alt+D window
    reader.generation = 7
    # The user cancels the box, so the pipeline never reaches the point where
    # it replaces a playing reading: the press itself must not stop anything.
    feature, capture, *_ = make(capture=FakeCapture(result=None), reader=reader)

    run_once(feature)

    assert reader.stop_calls == 0
    assert capture.calls == 1


def test_a_later_reading_by_someone_else_is_not_stopped_by_the_next_press() -> None:
    feature, capture, _o, _cb, reader, _n = make()
    run_once(feature)  # OCR reading finishes being "its own" ...
    reader.generation += 1  # ... then the user starts a different reading
    capture.result = None  # the user cancels the box: the press must not stop anything

    run_once(feature)

    assert reader.stop_calls == 0
    assert capture.calls == 2


def test_a_reading_in_progress_is_stopped_before_the_ocr_text_starts() -> None:
    events: "list[str]" = []

    class OrderedReader(FakeReader):
        def stop(self) -> None:
            events.append("stop")
            super().stop()

    reader = OrderedReader(events=events)
    reader.is_reading = True  # a document opened elsewhere
    reader.generation = 7
    feature, *_ = make(reader=reader, on_reading_starting=lambda text: events.append(f"starting:{text}"))

    run_once(feature)

    assert reader.stop_calls == 1
    assert events == ["stop", "starting:Hello world", "read_text"]


def test_an_idle_reader_is_not_stopped_before_the_ocr_text_starts() -> None:
    feature, _c, _o, _cb, reader, _n = make()

    run_once(feature)

    assert reader.stop_calls == 0
