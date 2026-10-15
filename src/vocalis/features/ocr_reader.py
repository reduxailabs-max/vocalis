"""OCR Reader: Ctrl+Alt+O -> drag a box on screen -> read its text aloud.

This feature only captures, recognises and copies. Speaking is handed to
the existing DocumentReaderFeature.read_text(), which already provides
synthesis with fallback, chunking, pause/resume/stop and
ReadingCoordinator ownership, so no reading-aloud code is duplicated.

The interactive capture (the user dragging a box) can take many seconds,
so it never runs on the hotkey watcher thread: on_trigger() only flips
state under a short lock and starts a background thread.
"""
import logging
import threading
from typing import Callable, Protocol

from vocalis.engines.tesseract_ocr import TesseractError
from vocalis.notify import Notifier
from vocalis.output.screen_capture import ScreenCaptureUnavailable

logger = logging.getLogger("vocalis")

_TITLE = "OCR Reader"
_MIN_TEXT_LENGTH = 3
_PREVIEW_LENGTH = 80


class RegionCapture(Protocol):
    """Lets the user choose a screen region and returns it as PNG bytes."""

    def capture(self) -> bytes | None: ...


class TextRecognizer(Protocol):
    """Turns a picture into text."""

    def recognize(self, png_bytes: bytes) -> str: ...


class ClipboardSink(Protocol):
    """Puts text on the clipboard."""

    def write(self, text: str) -> bool: ...


class SpeechReader(Protocol):
    """The part of DocumentReaderFeature that the OCR Reader uses."""

    @property
    def is_reading(self) -> bool: ...

    @property
    def generation(self) -> int: ...

    def read_text(self, text: str) -> None: ...

    def stop(self) -> None: ...


class OcrReaderFeature:
    """Wires Ctrl+Alt+O to capture -> OCR -> clipboard -> read aloud."""

    def __init__(
        self,
        capture: RegionCapture,
        ocr: TextRecognizer,
        clipboard: ClipboardSink,
        reader: SpeechReader,
        notifier: Notifier,
        on_reading_starting: Callable[[str], None] | None = None,
    ) -> None:
        """Create an OcrReaderFeature.

        Args:
            capture: Lets the user drag a box on screen.
            ocr: Reads the text out of the captured picture.
            clipboard: Receives a copy of the recognised text.
            reader: The shared Document Reader that speaks the text.
            notifier: Sends the status notifications.
            on_reading_starting: Called with the text just before the
                reading starts (the app uses it to show the Document
                Reader window with Pause/Stop). Optional.
        """
        self._capture = capture
        self._ocr = ocr
        self._clipboard = clipboard
        self._reader = reader
        self._notifier = notifier
        self._on_reading_starting = on_reading_starting
        self._lock = threading.Lock()
        self._busy = False
        self._my_generation: int | None = None

    @property
    def is_busy(self) -> bool:
        """Whether a capture/recognise/start-reading run is in progress."""
        with self._lock:
            return self._busy

    def on_trigger(self) -> None:
        """Call when Ctrl+Alt+O is pressed. Never blocks the caller.

        Ignored while a capture run is in progress. If the Document
        Reader is playing a reading that this feature started, the press
        stops it; otherwise it starts a new capture on a background thread.
        """
        with self._lock:
            if self._busy:
                return
            owns_reading = (
                self._my_generation is not None
                and self._reader.is_reading
                and self._reader.generation == self._my_generation
            )
            if not owns_reading:
                self._busy = True
        if owns_reading:
            threading.Thread(target=self._reader.stop, daemon=True).start()
            return
        threading.Thread(target=self._capture_and_read, daemon=True).start()

    def _capture_and_read(self) -> None:
        """Run one full capture -> read pass on a background thread and
        always clear the busy flag, whatever happens."""
        try:
            self._run_pipeline()
        except Exception:
            logger.exception("OCR reader failed")
            self._notifier.notify(_TITLE, "Something went wrong reading the screen")
        finally:
            with self._lock:
                self._busy = False

    def _run_pipeline(self) -> None:
        """Capture a region, recognise its text, copy it, and start reading it."""
        try:
            png = self._capture.capture()
        except ScreenCaptureUnavailable as exc:
            self._notifier.notify(_TITLE, str(exc))
            return
        if png is None:
            self._notifier.notify(_TITLE, "Cancelled")
            return
        try:
            text = self._ocr.recognize(png)
        except TesseractError as exc:
            self._notifier.notify(_TITLE, str(exc))
            return
        if len(text) < _MIN_TEXT_LENGTH:
            self._notifier.notify(_TITLE, "No text detected")
            return
        if not self._clipboard.write(text):
            logger.warning("Could not copy the recognised text to the clipboard")
        self._notifier.notify(_TITLE, _preview(text))
        # A reading already playing (e.g. a document) must be stopped first,
        # otherwise the new reading's audio collides with the old chunk.
        if self._reader.is_reading:
            self._reader.stop()
        if self._on_reading_starting is not None:
            self._on_reading_starting(text)
        self._reader.read_text(text)
        with self._lock:
            self._my_generation = self._reader.generation


def _preview(text: str) -> str:
    """The first 80 characters of `text`, with an ellipsis if cut short."""
    if len(text) <= _PREVIEW_LENGTH:
        return text
    return text[:_PREVIEW_LENGTH] + "…"
