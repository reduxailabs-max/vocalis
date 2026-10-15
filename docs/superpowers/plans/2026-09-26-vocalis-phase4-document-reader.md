# Phase 4 — Document Reader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Document Reader: pressing Ctrl+Alt+D opens a Vocalis window that reads a file (.txt/.md/.pdf/.docx/.html) or pasted text aloud, chunk by chunk, with Pause/Resume/Stop and progress — sharing "one reading at a time" with the existing Quick Reader.

**Architecture:** A headless `DocumentReaderFeature` (extract → normalize → chunk → synthesize-with-fallback → play, on a background thread, reusing `AudioPlayer`/`ModelManager`/the Qwen3-TTS+Piper pair exactly as Quick Reader does) is driven by a PySide6 popup that owns all Qt code and window lifecycle. A new `ReadingCoordinator` makes Quick Reader and Document Reader mutually exclusive. The daemon's `main()` now runs the Qt event loop instead of a bare sleep loop; hotkey/worker threads reach the popup only through Qt signals.

**Tech Stack:** Python 3.13, PySide6 (Qt6), pypdf, python-docx, existing Qwen3-TTS/Piper/AudioPlayer/ModelManager/Notifier/evdev stack. Dev-only: pytest-qt, reportlab (test PDF fixture generation).

**Spec:** `docs/superpowers/specs/2026-09-11-vocalis-design.md` — see "Phase 4 amendment (confirmed via superpowers:brainstorming, 2026-09-26)".

## Global Constraints

- Fully offline/local — no network calls (spec Hard Constraint 1).
- Only one AI model resident in VRAM at a time; Document Reader's primary engine goes through `ModelManager.acquire()/release()` exactly like Quick Reader's (spec Hard Constraint 2).
- Desktop notifications for start/live/done states, toggleable via `config.notifications_enabled` (spec Hard Constraint 4).
- `old/` is never modified (spec Hard Constraint 5).
- Every function has a docstring; structured error handling; no hardcoded secrets; ≥85% test coverage (`pyproject.toml`'s `--cov-fail-under=85`); strict mypy clean; conventional-commit messages; changelog at `.claude/<timestamp>_CHANGELOG.md` after the parent task (project `CLAUDE.md`).
- One sub-task at a time — stop after each task and wait for the user's go-ahead before starting the next (project `CLAUDE.md`).
- Never hold a lock across a call into blocking/unknown code (only ever guard a plain flag/reference swap) — the hard-won rule from Phase 2 Task 9 and Phase 3 Task 6, reused throughout this plan.
- After any `pip install`/dependency change, re-run `python -c "import onnxruntime; print(onnxruntime.get_available_providers())"` and confirm `CUDAExecutionProvider` is still listed — the exact onnxruntime/onnxruntime-gpu packaging conflict hit during Phase 3's merge can recur.

## Review Focus

- **A corrupt or truncated PDF/DOCX file** — must produce a clear "damaged file" notification, never an unhandled exception that crashes the popup or the daemon. Covered by Task 3's corrupt-file fixture tests.
- **An image-only (scanned) PDF with no text layer** — must produce "no readable text" pointing at the future OCR Reader, not a blank/silent read. Covered by Task 3.
- **Pressing Ctrl+Alt+D or clicking Stop while paused** (not just while actively playing) — must actually stop, not leave the reading stuck waiting on a resume that will never come. Covered by Task 6's pause/stop-while-paused test.
- **Starting Quick Reader while Document Reader is reading (and vice versa)** — must stop the other cleanly, not play both at once or deadlock. Covered by Task 5 (coordinator) and Task 6 (integration test with a real `ReadingCoordinator`).
- **A reading that finishes, fails, or is stopped while the popup is not focused/visible** — status/progress/finished callbacks must still be delivered safely to the (possibly hidden) Qt widgets without crashing, since Qt signal delivery does not require a widget to be visible. Covered by Task 7's popup tests, which check `status_changed`/`progress_changed`/`reading_finished` handling directly without requiring `show()`.

---

## Task 1: Text normalizer

**Files:**
- Create: `src/vocalis/text/__init__.py` (empty)
- Create: `src/vocalis/text/normalizer.py`
- Test: `tests/test_normalizer.py`

**Interfaces:**
- Produces: `normalize_markdown(text: str) -> str`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for vocalis.text.normalizer.normalize_markdown."""
from vocalis.text.normalizer import normalize_markdown


def test_strips_heading_markers() -> None:
    assert normalize_markdown("# Title\n\nBody") == "Title\n\nBody"


def test_strips_bold_and_italic() -> None:
    assert normalize_markdown("**bold** and *italic* and __also bold__") == (
        "bold and italic and also bold"
    )


def test_replaces_link_with_its_label() -> None:
    assert normalize_markdown("See [the docs](https://example.com) here") == (
        "See the docs here"
    )


def test_keeps_image_alt_text() -> None:
    assert normalize_markdown("![a diagram](diagram.png)") == "a diagram"


def test_removes_code_fences_entirely() -> None:
    result = normalize_markdown("Before\n```python\nx = 1\n```\nAfter")
    assert "```" not in result
    assert "x = 1" not in result
    assert "Before" in result and "After" in result


def test_keeps_inline_code_content() -> None:
    assert normalize_markdown("Run `pytest` now") == "Run pytest now"


def test_collapses_repeated_whitespace_and_blank_lines() -> None:
    assert normalize_markdown("a   b\n\n\n\nc") == "a b\n\nc"


def test_plain_text_is_unchanged() -> None:
    assert normalize_markdown("Just plain text.") == "Just plain text."


def test_empty_string_returns_empty_string() -> None:
    assert normalize_markdown("") == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_normalizer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vocalis.text'`

- [ ] **Step 3: Write minimal implementation**

`src/vocalis/text/__init__.py`: empty file.

`src/vocalis/text/normalizer.py`:

```python
"""Converts raw markdown text into plain, speakable text for Document
Reader's TTS engines.

Strips markdown syntax that would otherwise be read aloud literally (a
TTS engine has no idea "**bold**" means emphasis — it would either speak
the asterisks or mangle the words) while keeping the underlying words,
and collapses excess whitespace so chunker.py sees clean paragraphs.

Order matters: code fences are removed before anything else looks at
their contents (code should never be read aloud), and links/images are
resolved to their label before heading/emphasis markers are stripped
generally (a link's own brackets would otherwise look like plain text
containing stray brackets).
"""
import re

_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_HEADING_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_BOLD_ITALIC_RE = re.compile(r"(\*\*\*|___)(.+?)\1")
_BOLD_RE = re.compile(r"(\*\*|__)(.+?)\1")
_ITALIC_RE = re.compile(r"(\*|_)(.+?)\1")
_INLINE_WHITESPACE_RE = re.compile(r"[ \t]+")
_EXCESS_BLANK_LINES_RE = re.compile(r"\n{3,}")


def normalize_markdown(text: str) -> str:
    """Convert markdown text into plain, speakable text.

    Args:
        text: Raw markdown source (or plain text, which passes through
            unchanged since it contains none of the syntax below).

    Returns:
        Plain text with markdown syntax removed, links/images replaced
        by their label text, and excess whitespace collapsed. Never
        raises.
    """
    result = _CODE_FENCE_RE.sub(" ", text)
    result = _IMAGE_RE.sub(r"\1", result)
    result = _LINK_RE.sub(r"\1", result)
    result = _HEADING_RE.sub("", result)
    result = _INLINE_CODE_RE.sub(r"\1", result)
    result = _BOLD_ITALIC_RE.sub(r"\2", result)
    result = _BOLD_RE.sub(r"\2", result)
    result = _ITALIC_RE.sub(r"\2", result)
    result = _INLINE_WHITESPACE_RE.sub(" ", result)
    result = _EXCESS_BLANK_LINES_RE.sub("\n\n", result)
    return result.strip()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_normalizer.py -v`
Expected: PASS (9 passed)

- [ ] **Step 5: Commit**

```bash
git add src/vocalis/text/__init__.py src/vocalis/text/normalizer.py tests/test_normalizer.py
git commit -m "feat: add markdown-to-speakable-text normalizer"
```

---

## Task 2: Text chunker

**Files:**
- Create: `src/vocalis/text/chunker.py`
- Test: `tests/test_chunker.py`

**Interfaces:**
- Consumes: nothing from Task 1 (independent of `normalize_markdown`, but always called after it in Task 6).
- Produces: `chunk_text(text: str, max_chars: int = DEFAULT_MAX_CHUNK_CHARS) -> list[str]`, `DEFAULT_MAX_CHUNK_CHARS: int`.

- [ ] **Step 1: Write the failing test**

```python
"""Tests for vocalis.text.chunker.chunk_text."""
from vocalis.text.chunker import chunk_text


def test_empty_text_returns_no_chunks() -> None:
    assert chunk_text("") == []


def test_whitespace_only_text_returns_no_chunks() -> None:
    assert chunk_text("   \n  ") == []


def test_single_short_sentence_is_one_chunk() -> None:
    assert chunk_text("Hello there.") == ["Hello there."]


def test_short_sentences_are_packed_into_one_chunk() -> None:
    text = "One. Two. Three."
    assert chunk_text(text, max_chars=300) == ["One. Two. Three."]


def test_sentences_overflow_into_a_new_chunk() -> None:
    sentence = "A" * 20 + "."
    text = " ".join([sentence] * 5)  # 5 * 21 chars = 105 chars total
    chunks = chunk_text(text, max_chars=50)
    assert len(chunks) > 1
    assert all(len(chunk) <= 50 or chunk.count(".") == 1 for chunk in chunks)


def test_never_splits_a_single_long_sentence_mid_word() -> None:
    long_sentence = "word " * 30 + "end."
    chunks = chunk_text(long_sentence, max_chars=10)
    assert len(chunks) == 1
    assert chunks[0] == long_sentence.strip()
    assert "wo rd" not in chunks[0]


def test_reassembling_chunks_preserves_reading_order() -> None:
    text = "First sentence. Second sentence. Third sentence."
    chunks = chunk_text(text, max_chars=1000)
    assert " ".join(chunks) == text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_chunker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vocalis.text.chunker'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Splits normalized text into sentence-bounded chunks for streaming
synthesis and playback, so Document Reader can start speaking before the
whole document has been processed and can show "piece N of M" progress.

A sentence is never split mid-word: a single sentence longer than
max_chars becomes its own (over-limit) chunk rather than being cut off,
since a TTS engine reading a word fragment sounds far worse than reading
one long chunk.
"""
import re

DEFAULT_MAX_CHUNK_CHARS = 300

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text: str, max_chars: int = DEFAULT_MAX_CHUNK_CHARS) -> "list[str]":
    """Split text into sentence-bounded chunks no longer than max_chars.

    Args:
        text: Normalized, speakable text (see normalizer.normalize_markdown).
        max_chars: Soft maximum characters per chunk. A single sentence
            longer than this is still returned whole, as its own chunk.

    Returns:
        A list of non-empty chunk strings, in reading order. Empty input
        (or input that is only whitespace) returns an empty list.
    """
    sentences = [s.strip() for s in _SENTENCE_BOUNDARY_RE.split(text.strip()) if s.strip()]
    if not sentences:
        return []

    chunks: "list[str]" = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}" if current else sentence
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_chunker.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add src/vocalis/text/chunker.py tests/test_chunker.py
git commit -m "feat: add sentence-bounded text chunker"
```

---

## Task 3: Document extractors

**Files:**
- Create: `src/vocalis/text/extractors.py`
- Test: `tests/test_extractors.py`
- Modify: `pyproject.toml` — add `pypdf`, `python-docx` to `dependencies`; add `reportlab` to the `dev` optional-dependencies (used only to generate a real test PDF fixture); add mypy overrides for `docx` and `reportlab`.

**Interfaces:**
- Produces: `extract_text(path: Path) -> str`; exceptions `UnsupportedFileTypeError`, `DocumentOpenError`, `NoReadableTextError`, `DocumentTooLargeError`; `SUPPORTED_EXTENSIONS: frozenset[str]`, `MAX_EXTRACTED_CHARS: int`.

- [ ] **Step 1: Add and install the new dependencies**

Edit `pyproject.toml`'s `dependencies` list to add, after `"evdev>=1.7,<2",`:

```toml
    "pypdf>=5,<6",
    "python-docx>=1.1,<2",
```

Edit the `dev` optional-dependency list to add:

```toml
    "pytest-qt>=4.4,<5",
    "reportlab>=4,<5",
```

(`pytest-qt` is added here even though it's not needed until Task 7, so this one dependency-install step covers every new package this plan needs.)

Add PySide6 to `dependencies` too, right after `python-docx`:

```toml
    "PySide6>=6.7,<7",
```

Add mypy overrides after the existing `qwen_tts` override:

```toml
[[tool.mypy.overrides]]
module = "docx"
ignore_missing_imports = true

[[tool.mypy.overrides]]
module = "docx.opc.exceptions"
ignore_missing_imports = true

[[tool.mypy.overrides]]
module = "reportlab.*"
ignore_missing_imports = true
```

Run: `.venv/bin/pip install -e ".[dev]"`
Then immediately run: `.venv/bin/python -c "import onnxruntime; print(onnxruntime.get_available_providers())"`
Expected: `CUDAExecutionProvider` is in the printed list (per Global Constraints). If it is missing, run `.venv/bin/pip install --force-reinstall --no-deps onnxruntime-gpu==1.30.0` (the exact fix used during the Phase 3 merge) and re-check before continuing.

- [ ] **Step 2: Write the failing test**

```python
"""Tests for vocalis.text.extractors.extract_text.

Uses small real files (built at test time, not committed as binaries)
for every supported format, plus a corrupt file and an image-only PDF
to exercise the error paths for real rather than through a fake.
"""
from pathlib import Path

import pypdf
from docx import Document
from reportlab.pdfgen import canvas

from vocalis.text.extractors import (
    MAX_EXTRACTED_CHARS,
    DocumentOpenError,
    DocumentTooLargeError,
    NoReadableTextError,
    UnsupportedFileTypeError,
    extract_text,
)

import pytest


def test_extracts_plain_txt(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("Hello from a text file.", encoding="utf-8")
    assert extract_text(path) == "Hello from a text file."


def test_extracts_markdown_without_normalizing_it(tmp_path: Path) -> None:
    path = tmp_path / "sample.md"
    path.write_text("# Heading\n\n**bold**", encoding="utf-8")
    # extractors.py returns raw text; normalization is doc_reader.py's job.
    assert extract_text(path) == "# Heading\n\n**bold**"


def test_extracts_pdf_text(tmp_path: Path) -> None:
    path = tmp_path / "sample.pdf"
    pdf_canvas = canvas.Canvas(str(path), pagesize=(200, 200))
    pdf_canvas.drawString(10, 100, "Hello from a PDF.")
    pdf_canvas.save()
    assert "Hello from a PDF." in extract_text(path)


def test_extracts_docx_text(tmp_path: Path) -> None:
    path = tmp_path / "sample.docx"
    document = Document()
    document.add_paragraph("Hello from a docx file.")
    document.save(str(path))
    assert "Hello from a docx file." in extract_text(path)


def test_extracts_html_and_skips_script_and_style(tmp_path: Path) -> None:
    path = tmp_path / "sample.html"
    path.write_text(
        "<html><head><style>body{color:red}</style></head>"
        "<body><script>alert('x')</script><p>Visible text</p></body></html>",
        encoding="utf-8",
    )
    text = extract_text(path)
    assert "Visible text" in text
    assert "color:red" not in text
    assert "alert" not in text


def test_unsupported_extension_raises() -> None:
    with pytest.raises(UnsupportedFileTypeError):
        extract_text(Path("/tmp/whatever.rtf"))


def test_missing_file_raises_document_open_error(tmp_path: Path) -> None:
    with pytest.raises(DocumentOpenError):
        extract_text(tmp_path / "does_not_exist.txt")


def test_corrupt_pdf_raises_document_open_error(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.pdf"
    path.write_bytes(b"this is not a real pdf file")
    with pytest.raises(DocumentOpenError):
        extract_text(path)


def test_corrupt_docx_raises_document_open_error(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.docx"
    path.write_bytes(b"this is not a real docx file")
    with pytest.raises(DocumentOpenError):
        extract_text(path)


def test_image_only_pdf_raises_no_readable_text_error(tmp_path: Path) -> None:
    path = tmp_path / "scanned.pdf"
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with path.open("wb") as handle:
        writer.write(handle)
    with pytest.raises(NoReadableTextError):
        extract_text(path)


def test_empty_txt_raises_no_readable_text_error(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("   \n  ", encoding="utf-8")
    with pytest.raises(NoReadableTextError):
        extract_text(path)


def test_oversized_text_raises_document_too_large_error(tmp_path: Path) -> None:
    path = tmp_path / "huge.txt"
    path.write_text("a" * (MAX_EXTRACTED_CHARS + 1), encoding="utf-8")
    with pytest.raises(DocumentTooLargeError):
        extract_text(path)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_extractors.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vocalis.text.extractors'`

- [ ] **Step 4: Write minimal implementation**

```python
"""Extracts raw text from a document file, dispatching by file extension.

Returns the raw text as found in the file — markdown normalization
(normalizer.normalize_markdown) is a separate step performed by
doc_reader.py, so a plain .txt file and a .md file are extracted
identically here.

Every failure mode is a typed exception rather than an unhandled one,
since a corrupt or unusual file the user picks must produce a clear
notification (see doc_reader.py), never crash the popup or the daemon.
Library-specific parsing exceptions (pypdf's, python-docx's) are caught
broadly rather than by exact class, since both libraries can raise a
range of exception types for a malformed file depending on exactly how
it is malformed, and the one thing that matters here is "this file could
not be opened," not which internal exception class reported it.
"""
import html.parser
import logging
from pathlib import Path

import pypdf
from docx import Document as DocxDocument

logger = logging.getLogger("vocalis")

SUPPORTED_EXTENSIONS = frozenset({".txt", ".md", ".pdf", ".docx", ".html", ".htm"})
MAX_EXTRACTED_CHARS = 5_000_000


class UnsupportedFileTypeError(Exception):
    """Raised when the file extension is not one Document Reader supports."""


class DocumentOpenError(Exception):
    """Raised when the file cannot be opened or parsed (corrupt/unreadable)."""


class NoReadableTextError(Exception):
    """Raised when a file opens successfully but contains no extractable
    text (e.g. an image-only scanned PDF with no text layer, or an empty
    file)."""


class DocumentTooLargeError(Exception):
    """Raised when extracted text exceeds MAX_EXTRACTED_CHARS."""


class _HtmlTextExtractor(html.parser.HTMLParser):
    """Collects visible text, skipping <script> and <style> contents."""

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self.text_parts: "list[str]" = []

    def handle_starttag(self, tag: str, attrs: "list[tuple[str, str | None]]") -> None:
        if tag in ("script", "style"):
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style") and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self.text_parts.append(data)


def extract_text(path: Path) -> str:
    """Extract raw text from a document file.

    Args:
        path: Path to the document. Dispatched by suffix (case-insensitive).

    Returns:
        The extracted raw text: non-empty and no larger than
        MAX_EXTRACTED_CHARS.

    Raises:
        UnsupportedFileTypeError: If the extension isn't supported.
        DocumentOpenError: If the file cannot be opened or parsed.
        NoReadableTextError: If extraction succeeds but yields no text.
        DocumentTooLargeError: If extracted text exceeds the size ceiling.
    """
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type {suffix!r}. Supported: "
            f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if suffix in (".txt", ".md"):
        text = _extract_plain_text(path)
    elif suffix == ".pdf":
        text = _extract_pdf(path)
    elif suffix == ".docx":
        text = _extract_docx(path)
    else:
        text = _extract_html(path)

    if not text.strip():
        raise NoReadableTextError(
            f"No readable text found in {path.name}. If this is a scanned "
            "document, try the OCR Reader feature instead."
        )
    if len(text) > MAX_EXTRACTED_CHARS:
        raise DocumentTooLargeError(
            f"{path.name} contains more than {MAX_EXTRACTED_CHARS:,} characters "
            "of text, which is too large for Document Reader."
        )
    return text


def _extract_plain_text(path: Path) -> str:
    try:
        return path.read_bytes().decode("utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Could not open %s: %s", path, exc)
        raise DocumentOpenError(f"Could not open {path.name}: {exc}") from exc


def _extract_pdf(path: Path) -> str:
    try:
        reader = pypdf.PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:
        logger.warning("Could not open PDF %s: %s", path, exc)
        raise DocumentOpenError(f"Could not open {path.name}: {exc}") from exc


def _extract_docx(path: Path) -> str:
    try:
        document = DocxDocument(str(path))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)
    except Exception as exc:
        logger.warning("Could not open docx %s: %s", path, exc)
        raise DocumentOpenError(f"Could not open {path.name}: {exc}") from exc


def _extract_html(path: Path) -> str:
    try:
        raw = path.read_bytes().decode("utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Could not open %s: %s", path, exc)
        raise DocumentOpenError(f"Could not open {path.name}: {exc}") from exc
    parser = _HtmlTextExtractor()
    parser.feed(raw)
    return "".join(parser.text_parts)
```

Note: `_extract_pdf`/`_extract_docx` catch plain `Exception` deliberately — see the module docstring. If mypy or a linter complains about the broad catch, add `# noqa` rather than narrowing it to a guessed exception class list.

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_extractors.py -v`
Expected: PASS (12 passed). If `test_extracts_pdf_text` or `test_image_only_pdf_raises_no_readable_text_error` fails because pypdf can't parse the reportlab/PdfWriter output, run with `-v` to see the real error and fix the fixture generation (not the production code) — both reportlab and pypdf.PdfWriter produce standard-compliant PDFs, so a failure here means a fixture mistake, not a pypdf limitation.

- [ ] **Step 6: Run the full test suite to confirm nothing else broke**

Run: `.venv/bin/pytest -q`
Expected: all previous tests plus the new ones pass.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/vocalis/text/extractors.py tests/test_extractors.py
git commit -m "feat: add txt/md/pdf/docx/html text extractors"
```

---

## Task 4: Generalized Ctrl+Alt+key hotkey listener

**Files:**
- Create: `src/vocalis/hotkeys/ctrl_alt_key.py`
- Modify: `src/vocalis/hotkeys/ctrl_alt_r.py` (rewritten as a thin backward-compatible wrapper)
- Test: `tests/test_ctrl_alt_key.py`
- Test: `tests/test_ctrl_alt_r.py` (existing — must keep passing unchanged; do not edit its assertions, only re-run it)

**Interfaces:**
- Produces: `CtrlAltKeyListener(key_code: int, on_trigger: Callable[[], None], device_enumerator: DeviceEnumerator)`, `list_devices_with_key(key_code: int) -> list[EvdevDevice]`, `EvdevDevice`/`EvdevEvent`/`DeviceEnumerator` (same shapes as the existing `ctrl_alt_r.py` versions).
- `CtrlAltRListener` and `list_hotkey_capable_devices` in `ctrl_alt_r.py` keep their exact existing signatures and behavior.

- [ ] **Step 1: Write the failing test**

```python
"""Tests for vocalis.hotkeys.ctrl_alt_key.CtrlAltKeyListener.

Mirrors test_ctrl_alt_r.py's approach with fake evdev devices, but
exercises the generalized key_code parameter directly (using the D key,
32, as a stand-in for a key other than R).
"""
import time
from typing import Iterator

from vocalis.hotkeys.ctrl_alt_key import CtrlAltKeyListener

_EV_KEY = 1
_KEY_LEFTCTRL = 29
_KEY_LEFTALT = 56
_KEY_D = 32


class FakeEvent:
    def __init__(self, type: int, code: int, value: int) -> None:
        self.type = type
        self.code = code
        self.value = value


class FakeDevice:
    def __init__(self, events: "list[FakeEvent]") -> None:
        self._events = events
        self._closed = False

    def read_loop(self) -> Iterator[FakeEvent]:
        for event in self._events:
            if self._closed:
                raise OSError("device closed")
            yield event
        while not self._closed:
            time.sleep(0.01)
        raise OSError("device closed")

    def close(self) -> None:
        self._closed = True


def test_fires_on_trigger_for_the_configured_key() -> None:
    triggers = []
    device = FakeDevice(
        [
            FakeEvent(_EV_KEY, _KEY_LEFTCTRL, 1),
            FakeEvent(_EV_KEY, 56, 1),  # left alt
            FakeEvent(_EV_KEY, _KEY_D, 1),
        ]
    )
    listener = CtrlAltKeyListener(
        key_code=_KEY_D, on_trigger=lambda: triggers.append(1), device_enumerator=lambda: [device]
    )
    listener.start()
    time.sleep(0.1)
    listener.stop()
    assert triggers == [1]


def test_does_not_fire_for_a_different_key() -> None:
    triggers = []
    device = FakeDevice(
        [
            FakeEvent(_EV_KEY, _KEY_LEFTCTRL, 1),
            FakeEvent(_EV_KEY, _KEY_LEFTALT, 1),
            FakeEvent(_EV_KEY, 19, 1),  # R, not the configured D
        ]
    )
    listener = CtrlAltKeyListener(
        key_code=_KEY_D, on_trigger=lambda: triggers.append(1), device_enumerator=lambda: [device]
    )
    listener.start()
    time.sleep(0.1)
    listener.stop()
    assert triggers == []


def test_a_raising_callback_does_not_kill_the_watcher_thread() -> None:
    triggers = []

    def flaky_callback() -> None:
        triggers.append(1)
        if len(triggers) == 1:
            raise RuntimeError("boom")

    device = FakeDevice(
        [
            FakeEvent(_EV_KEY, _KEY_LEFTCTRL, 1),
            FakeEvent(_EV_KEY, _KEY_LEFTALT, 1),
            FakeEvent(_EV_KEY, _KEY_D, 1),
            FakeEvent(_EV_KEY, _KEY_D, 1),
        ]
    )
    listener = CtrlAltKeyListener(
        key_code=_KEY_D, on_trigger=flaky_callback, device_enumerator=lambda: [device]
    )
    listener.start()
    time.sleep(0.1)
    listener.stop()
    assert triggers == [1, 1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_ctrl_alt_key.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vocalis.hotkeys.ctrl_alt_key'`

- [ ] **Step 3: Write the generalized listener**

```python
"""Generalized Ctrl+Alt+<key> push-to-trigger hotkey listener.

Built the same way as NumlockListener (Phase 2) and the original
CtrlAltRListener (Phase 3): watches raw evdev key events without
suppressing them, one background thread per keyboard-capable device,
firing on_trigger() when the configured key is pressed while both Ctrl
and Alt are currently held on that same device. Held-state for Ctrl/Alt
is tracked per-device (inside each device's own watcher thread) since a
real combo is normally pressed on a single physical keyboard.

CtrlAltRListener (ctrl_alt_r.py) is now a thin, backward-compatible
subclass of this class fixed to the R key, so Phase 3's existing call
sites and tests keep working unchanged after this generalization.

Two lessons from Phase 2 are applied here from the start: the
on_trigger() invocation is wrapped in its own try/except inside the read
loop (an exception must never permanently kill the watcher thread), and
Vocalis's own ydotoold virtual output device is excluded from
enumeration (the app must never watch the device it uses to type/inject
its own output).
"""
import logging
import threading
from typing import Callable, Iterable, Protocol

logger = logging.getLogger("vocalis")

_EV_KEY = 1
_KEY_LEFTCTRL = 29
_KEY_RIGHTCTRL = 97
_KEY_LEFTALT = 56
_KEY_RIGHTALT = 100
_EXCLUDED_DEVICE_NAMES = frozenset({"ydotoold virtual device"})


class EvdevEvent(Protocol):
    """The subset of an evdev InputEvent this module depends on."""

    type: int
    code: int
    value: int


class EvdevDevice(Protocol):
    """The subset of evdev.InputDevice this module depends on."""

    def read_loop(self) -> Iterable[EvdevEvent]:
        """Blocks, yielding events as they arrive, until close() is called."""
        ...

    def close(self) -> None:
        """Close the device, causing a blocked read_loop() to raise and exit."""
        ...


DeviceEnumerator = Callable[[], "list[EvdevDevice]"]


class CtrlAltKeyListener:
    """Calls on_trigger() when Ctrl+Alt+<key_code> is pressed."""

    def __init__(
        self,
        key_code: int,
        on_trigger: Callable[[], None],
        device_enumerator: DeviceEnumerator,
    ) -> None:
        """Create a CtrlAltKeyListener.

        Args:
            key_code: The evdev key code (e.g. 19 for R, 32 for D) that,
                combined with held Ctrl+Alt, fires on_trigger().
            on_trigger: Called when Ctrl+Alt+<key_code> is pressed.
            device_enumerator: Returns the list of evdev devices to watch.
                Injectable so tests can supply fake devices instead of
                real hardware.
        """
        self._key_code = key_code
        self._on_trigger = on_trigger
        self._device_enumerator = device_enumerator
        self._devices: "list[EvdevDevice]" = []
        self._threads: "list[threading.Thread]" = []

    @property
    def is_running(self) -> bool:
        """Whether any watcher thread is currently active."""
        return any(thread.is_alive() for thread in self._threads)

    def start(self) -> None:
        """Enumerate hotkey-capable devices and start watching each."""
        self._devices = self._device_enumerator()
        if not self._devices:
            logger.warning("No keyboard-capable input devices found")
            return
        self._threads = [
            threading.Thread(target=self._watch_device, args=(device,), daemon=True)
            for device in self._devices
        ]
        for thread in self._threads:
            thread.start()
        logger.info(
            "Watching %d device(s) for Ctrl+Alt+%d", len(self._devices), self._key_code
        )

    def stop(self) -> None:
        """Close all watched devices and wait for their threads to exit."""
        for device in self._devices:
            try:
                device.close()
            except Exception:
                logger.exception("Error closing input device")
        for thread in self._threads:
            thread.join(timeout=1.0)
            if thread.is_alive():
                logger.warning("Device watcher thread did not exit within timeout")
        self._devices = []
        self._threads = []

    def _watch_device(self, device: EvdevDevice) -> None:
        """Read events from one device until it is closed."""
        ctrl_held = False
        alt_held = False
        try:
            for event in device.read_loop():
                if event.type != _EV_KEY:
                    continue
                try:
                    if event.code in (_KEY_LEFTCTRL, _KEY_RIGHTCTRL):
                        ctrl_held = event.value != 0
                    elif event.code in (_KEY_LEFTALT, _KEY_RIGHTALT):
                        alt_held = event.value != 0
                    elif (
                        event.code == self._key_code
                        and event.value == 1
                        and ctrl_held
                        and alt_held
                    ):
                        self._on_trigger()
                except Exception:
                    logger.exception(
                        "Error in Ctrl+Alt+%d callback; continuing to watch device",
                        self._key_code,
                    )
        except OSError:
            pass  # device closed by stop(); expected exit path
        except Exception:
            logger.exception("Error reading input device events")


def list_devices_with_key(key_code: int) -> "list[EvdevDevice]":
    """Enumerate real input devices that report Ctrl+Alt+<key_code> capability.

    Excludes Vocalis's own ydotoold virtual output device, closing any
    device that doesn't qualify immediately since it won't be watched.

    Args:
        key_code: The evdev key code the caller wants to watch for,
            alongside Ctrl and Alt.
    """
    import evdev

    devices: "list[EvdevDevice]" = []
    needed = {_KEY_LEFTCTRL, _KEY_LEFTALT, key_code}
    for path in evdev.list_devices():
        device = evdev.InputDevice(path)
        capabilities = set(device.capabilities().get(evdev.ecodes.EV_KEY, []))
        if needed.issubset(capabilities) and device.name not in _EXCLUDED_DEVICE_NAMES:
            devices.append(device)
        else:
            device.close()
    return devices
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_ctrl_alt_key.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Rewrite ctrl_alt_r.py as a thin wrapper and re-run its existing tests unchanged**

Replace the full contents of `src/vocalis/hotkeys/ctrl_alt_r.py` with:

```python
"""Ctrl+Alt+R push-to-trigger hotkey listener for Quick Reader.

Backward-compatible wrapper around the generalized CtrlAltKeyListener
(ctrl_alt_key.py), fixed to the R key (evdev code 19). Phase 4
generalized the underlying mechanism to also support Ctrl+Alt+D for
Document Reader; this module's public names and behavior are unchanged
from Phase 3 so existing call sites and tests keep working.
"""
from typing import Callable

from vocalis.hotkeys.ctrl_alt_key import (
    CtrlAltKeyListener,
    DeviceEnumerator,
    EvdevDevice,
    EvdevEvent,
    list_devices_with_key,
)

_KEY_R = 19

__all__ = [
    "CtrlAltRListener",
    "list_hotkey_capable_devices",
    "EvdevDevice",
    "EvdevEvent",
    "DeviceEnumerator",
]


class CtrlAltRListener(CtrlAltKeyListener):
    """Calls on_trigger() when Ctrl+Alt+R is pressed."""

    def __init__(self, on_trigger: Callable[[], None], device_enumerator: DeviceEnumerator) -> None:
        """Create a CtrlAltRListener.

        Args:
            on_trigger: Called when Ctrl+Alt+R is pressed.
            device_enumerator: Returns the list of evdev devices to watch.
                Injectable so tests can supply fake devices instead of
                real hardware.
        """
        super().__init__(key_code=_KEY_R, on_trigger=on_trigger, device_enumerator=device_enumerator)


def list_hotkey_capable_devices() -> "list[EvdevDevice]":
    """Enumerate real input devices that report Ctrl+Alt+R key capability."""
    return list_devices_with_key(_KEY_R)
```

Run: `.venv/bin/pytest tests/test_ctrl_alt_r.py -v`
Expected: PASS, unchanged from before this task (if any test fails, it is because it reaches into an implementation detail this rewrite changed — fix the wrapper, not the test, unless the test is asserting something the spec never promised).

- [ ] **Step 6: Run the full test suite**

Run: `.venv/bin/pytest -q`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/vocalis/hotkeys/ctrl_alt_key.py src/vocalis/hotkeys/ctrl_alt_r.py tests/test_ctrl_alt_key.py
git commit -m "refactor: generalize Ctrl+Alt+R listener into a configurable-key listener"
```

---

## Task 5: ReadingCoordinator (one reading at a time)

**Files:**
- Create: `src/vocalis/features/reading_coordinator.py`
- Test: `tests/test_reading_coordinator.py`
- Modify: `src/vocalis/features/quick_reader.py` — wire in the coordinator.
- Modify: `tests/test_quick_reader.py` — add coverage for the new wiring (existing tests must still pass; `reading_coordinator` is an optional constructor parameter defaulting to `None`, so no existing test needs to change).

**Interfaces:**
- Produces: `ReadingCoordinator`, with `start_reading(self, owner: object, stopper: Callable[[], None]) -> None` and `finish_reading(self, owner: object) -> None`.
- Consumes (in `quick_reader.py`): none new from other Phase 4 tasks.
- Produces (from the `quick_reader.py` change, for Task 6 to reuse the same pattern): `QuickReaderFeature.__init__` gains `reading_coordinator: ReadingCoordinator | None = None`.

- [ ] **Step 1: Write the failing test**

```python
"""Tests for vocalis.features.reading_coordinator.ReadingCoordinator.

Ownership is identified by a stable `owner` object (pass the feature
instance itself), never by comparing the `stopper` callable with `is` —
a bound method obtained via `self.method` is a fresh object on every
attribute access in Python, so two calls from the very same instance
would never compare equal that way. These tests pin that down directly.
"""
from vocalis.features.reading_coordinator import ReadingCoordinator


class _Owner:
    """A stand-in for a feature instance, with a bound-method stopper."""

    def __init__(self) -> None:
        self.stop_calls = 0

    def stop(self) -> None:
        self.stop_calls += 1


def test_first_start_reading_does_not_call_its_own_stopper() -> None:
    coordinator = ReadingCoordinator()
    owner = _Owner()

    coordinator.start_reading(owner, owner.stop)

    assert owner.stop_calls == 0


def test_same_owner_starting_again_does_not_stop_itself() -> None:
    """Regression test: two bound-method accesses on the same instance
    are different objects, so ownership must be tracked by `owner`
    identity, never by comparing the stopper callable itself."""
    coordinator = ReadingCoordinator()
    owner = _Owner()

    coordinator.start_reading(owner, owner.stop)
    coordinator.start_reading(owner, owner.stop)  # fresh bound method each time

    assert owner.stop_calls == 0


def test_a_different_owner_starting_stops_the_previous_owner() -> None:
    coordinator = ReadingCoordinator()
    first = _Owner()
    second = _Owner()

    coordinator.start_reading(first, first.stop)
    coordinator.start_reading(second, second.stop)

    assert first.stop_calls == 1
    assert second.stop_calls == 0


def test_finish_reading_clears_ownership() -> None:
    coordinator = ReadingCoordinator()
    first = _Owner()
    second = _Owner()

    coordinator.start_reading(first, first.stop)
    coordinator.finish_reading(first)
    coordinator.start_reading(second, second.stop)

    assert first.stop_calls == 0  # first already released; no stop needed


def test_stale_finish_reading_does_not_clobber_a_newer_owner() -> None:
    coordinator = ReadingCoordinator()
    first = _Owner()
    second = _Owner()

    coordinator.start_reading(first, first.stop)
    coordinator.start_reading(second, second.stop)  # first is stopped, second owns it
    coordinator.finish_reading(first)  # stale: first no longer owns anything

    coordinator.start_reading(_Owner(), lambda: None)
    assert second.stop_calls == 0  # second's ownership was never cleared by the stale call
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_reading_coordinator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vocalis.features.reading_coordinator'`

- [ ] **Step 3: Write the implementation**

```python
"""Ensures only one of Quick Reader and Document Reader is reading aloud
at a time.

Each feature identifies itself with a stable `owner` object — pass the
feature instance itself (`self`) — never with its stop callback. A bound
method obtained via `self.method` is a fresh object on every attribute
access in Python, so comparing stop callbacks with `is` would never
match, even across two calls from the very same instance. `owner` is
compared with `is`; `stopper` is only ever called, never compared.

Follows the same lock discipline already hard-won in quick_reader.py:
this coordinator only ever guards a plain reference swap under its lock.
The actual stop() call — which can block briefly on AudioPlayer.stop()
or a notify-send subprocess — happens after the lock has been released.
"""
import threading
from typing import Callable

StopCallback = Callable[[], None]


class ReadingCoordinator:
    """Tracks which feature (if any) currently owns audio playback."""

    def __init__(self) -> None:
        """Create a ReadingCoordinator with no current owner."""
        self._lock = threading.Lock()
        self._owner: object | None = None
        self._stopper: StopCallback | None = None

    def start_reading(self, owner: object, stopper: StopCallback) -> None:
        """Claim ownership of playback for `owner`.

        If a different owner currently holds it, that owner's registered
        stopper is called now, after this method's own lock has been
        released, so it is free to block briefly (e.g. on AudioPlayer.stop()
        or a notification) without holding up this call's caller.

        Args:
            owner: A stable identifier for the calling feature — pass the
                feature instance itself. Compared with `is`, never `==`.
            stopper: Called (with no arguments) to stop `owner`'s own
                in-progress reading, if a later start_reading() call from
                a different owner preempts it.
        """
        with self._lock:
            previous_owner = self._owner
            previous_stopper = self._stopper
            self._owner = owner
            self._stopper = stopper
        if previous_owner is not None and previous_owner is not owner:
            assert previous_stopper is not None
            previous_stopper()

    def finish_reading(self, owner: object) -> None:
        """Release ownership for `owner`, if it is still the current owner.

        A stale call from a reading that already lost ownership to a
        later start_reading() must not clear that newer owner's claim.

        Args:
            owner: The same object passed to the matching start_reading().
        """
        with self._lock:
            if self._owner is owner:
                self._owner = None
                self._stopper = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_reading_coordinator.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Wire the coordinator into QuickReaderFeature**

In `src/vocalis/features/quick_reader.py`:

Add the import:

```python
from vocalis.features.reading_coordinator import ReadingCoordinator
```

Change the constructor signature and body (around `quick_reader.py:41-82`) to add the new optional parameter and store it:

```python
    def __init__(
        self,
        model_manager: ModelManager,
        primary_engine_factory: Callable[[], ModelEngine],
        fallback_engine: Synthesizer,
        clipboard_reader: ClipboardReader,
        player: AudioPlayer,
        notifier: Notifier,
        language: str = "English",
        reading_coordinator: ReadingCoordinator | None = None,
    ) -> None:
        """Create a QuickReaderFeature.

        Args:
            model_manager: Shared ModelManager enforcing single-residency
                (used only for the primary, GPU-backed engine).
            primary_engine_factory: Constructs a fresh primary engine
                (e.g. Qwen3TTS) when it needs to be (re)loaded. The
                constructed engine must also implement Synthesizer.
            fallback_engine: An already-loaded, CPU-only engine (e.g.
                PiperTTS) used when the primary engine fails. Never
                routed through ModelManager since it never touches the GPU.
            clipboard_reader: Reads the text to speak.
            player: Plays the generated speech and can be interrupted.
            notifier: Sends start/live/done/stopped notifications.
            language: Spoken language passed to the primary engine.
            reading_coordinator: Shared with Document Reader so only one
                feature plays audio at a time. None (the default) means
                this feature never coordinates with another — used by
                existing tests that predate Document Reader.
        """
        self._model_manager = model_manager
        self._primary_engine_factory = primary_engine_factory
        self._fallback_engine = fallback_engine
        self._clipboard_reader = clipboard_reader
        self._player = player
        self._notifier = notifier
        self._language = language
        self._reading_coordinator = reading_coordinator
        self._lock = threading.Lock()
        self._active = False
        self._generation = 0
```

Add a new method, and update `on_trigger()` and `_deactivate()` (the rest of the file — `_flip_active_state`, `_stop_and_notify`, `_is_current`, `_read_aloud`, `_synthesize_with_fallback` — is unchanged):

```python
    def on_trigger(self) -> None:
        """Call when Ctrl+Alt+R is pressed: start a new reading, or stop
        an in-progress one. Never blocks the caller (the hotkey
        listener's watcher thread): the state flip happens under a lock
        scoped to just that flip (see _flip_active_state()), and every
        potentially slow call — AudioPlayer.stop(), which can wait on the
        audio callback thread, and Notifier.notify(), a notify-send
        subprocess that can take up to its own timeout — is dispatched
        onto a short-lived background thread rather than run inline.
        """
        interrupting, generation = self._flip_active_state()
        if interrupting:
            threading.Thread(target=self._stop_and_notify, daemon=True).start()
            return
        if self._reading_coordinator is not None:
            self._reading_coordinator.start_reading(self, self._stop_for_coordinator)
        threading.Thread(target=self._read_aloud, args=(generation,), daemon=True).start()

    def _stop_for_coordinator(self) -> None:
        """Stop this feature's in-progress reading because a different
        feature (via ReadingCoordinator) is claiming playback ownership.

        Equivalent to on_trigger()'s interrupt path, but never itself
        calls back into the coordinator (that would be pointless — the
        coordinator already gave ownership away) and must not block,
        since ReadingCoordinator.start_reading() calls this synchronously
        from the preempting feature's own thread. It only flips state and
        spawns the same background stop-and-notify thread on_trigger()
        already uses.
        """
        interrupting, _ = self._flip_active_state()
        if interrupting:
            threading.Thread(target=self._stop_and_notify, daemon=True).start()
```

Change `_deactivate` (around `quick_reader.py:133-141`) to also release coordinator ownership:

```python
    def _deactivate(self, generation: int) -> None:
        """Mark `generation` as finished, but only if it is still the
        current generation — a stale worker thread whose reading was
        already superseded by a newer one must not clobber that newer
        reading's active state. Also releases ReadingCoordinator
        ownership, if one is configured; ReadingCoordinator.finish_reading()
        already no-ops safely if ownership moved on to a different
        feature in the meantime."""
        with self._lock:
            if self._generation == generation:
                self._active = False
        if self._reading_coordinator is not None:
            self._reading_coordinator.finish_reading(self)
```

- [ ] **Step 6: Write and run the new coordination tests for QuickReaderFeature**

`tests/test_quick_reader.py` already defines `FakeSynthesizer`, `FakeClipboardReader`,
`FakePlayer`, `FakeNotifier`, and a `make_feature(...)` helper — reuse these exactly
as they exist rather than introducing new fakes. Give `make_feature` one new
optional parameter:

```python
def make_feature(
    primary: FakeSynthesizer,
    fallback: FakeSynthesizer,
    clipboard_text: str,
    player: FakePlayer | None = None,
    reading_coordinator: "ReadingCoordinator | None" = None,
):
    manager = ModelManager()
    notifier = FakeNotifier()
    player = player or FakePlayer()
    feature = QuickReaderFeature(
        model_manager=manager,
        primary_engine_factory=lambda: primary,  # type: ignore[arg-type]
        fallback_engine=fallback,  # type: ignore[arg-type]
        clipboard_reader=FakeClipboardReader(clipboard_text),  # type: ignore[arg-type]
        player=player,  # type: ignore[arg-type]
        notifier=notifier,  # type: ignore[arg-type]
        reading_coordinator=reading_coordinator,
    )
    return feature, primary, fallback, player, notifier, manager
```

Add the import `from vocalis.features.reading_coordinator import ReadingCoordinator`
at the top of the file, and add these two tests:

```python
def test_starting_a_reading_claims_coordinator_ownership() -> None:
    coordinator = ReadingCoordinator()
    primary = FakeSynthesizer(audio=np.zeros(1, dtype=np.float32))
    fallback = FakeSynthesizer(audio=np.zeros(1, dtype=np.float32))
    feature, _, _, _, _, _ = make_feature(
        primary, fallback, clipboard_text="hello", reading_coordinator=coordinator
    )

    feature.on_trigger()
    wait_until(lambda: coordinator._owner is feature)  # noqa: SLF001


def test_a_different_owner_starting_stops_the_quick_reader() -> None:
    coordinator = ReadingCoordinator()
    primary = FakeSynthesizer(audio=np.zeros(1, dtype=np.float32))
    fallback = FakeSynthesizer(audio=np.zeros(1, dtype=np.float32))
    player = FakePlayer(delay_seconds=2.0)  # wait_until_done() would block a long time unless stopped
    feature, _, _, _, notifier, _ = make_feature(
        primary,
        fallback,
        clipboard_text="a long passage",
        player=player,
        reading_coordinator=coordinator,
    )

    feature.on_trigger()
    wait_until(lambda: len(player.play_calls) == 1)
    coordinator.start_reading(object(), lambda: None)  # a different owner preempts it

    wait_until(lambda: any(msg == "Stopped" for _, msg in notifier.messages))
    assert player.stop_calls == 1
```

Run: `.venv/bin/pytest tests/test_quick_reader.py -v`
Expected: PASS, including all pre-existing tests in the file (unchanged) plus the two new ones.

- [ ] **Step 7: Run the full test suite and mypy**

Run: `.venv/bin/pytest -q && .venv/bin/mypy src`
Expected: all tests pass; mypy clean.

- [ ] **Step 8: Commit**

```bash
git add src/vocalis/features/reading_coordinator.py src/vocalis/features/quick_reader.py tests/test_reading_coordinator.py tests/test_quick_reader.py
git commit -m "feat: add ReadingCoordinator and wire it into QuickReaderFeature"
```

---

## Task 6: DocumentReaderFeature

**Files:**
- Create: `src/vocalis/features/doc_reader.py`
- Test: `tests/test_doc_reader.py`

**Interfaces:**
- Consumes: `chunk_text` (Task 2), `extract_text` + its exceptions (Task 3), `normalize_markdown` (Task 1), `ReadingCoordinator` (Task 5), `AudioPlayer`/`ModelManager`/`ModelEngine`/`Notifier` (existing).
- Produces: `DocumentReaderFeature` with `read_file(path: Path) -> None`, `read_text(text: str) -> None`, `pause() -> None`, `resume() -> None`, `stop() -> None`, `is_reading: bool` (property). Constructor takes `on_status: Callable[[str], None]`, `on_progress: Callable[[int, int], None]`, `on_finished: Callable[[], None]` — Task 7's popup wires these to Qt signals; Task 8's `build_doc_reader()` constructs this feature before the popup exists (see Task 8 for how the ordering is resolved).

- [ ] **Step 1: Write the failing test**

```python
"""Tests for vocalis.features.doc_reader.DocumentReaderFeature.

Uses fakes for every collaborator, the same approach test_quick_reader.py
already established, plus a real temporary file for read_file() so
extraction genuinely runs (extractors.py itself is unit-tested
separately in test_extractors.py).
"""
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from vocalis.audio.playback import AudioPlayer
from vocalis.features.doc_reader import DocumentReaderFeature
from vocalis.features.reading_coordinator import ReadingCoordinator
from vocalis.model_manager import ModelManager


class FakeSynthesizer:
    def __init__(
        self,
        sample_rate: int = 22050,
        raise_exc: Exception | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.raise_exc = raise_exc
        self.calls: "list[str]" = []
        self._loaded = False

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def synthesize(self, text: str) -> "tuple[npt.NDArray[np.float32], int]":
        self.calls.append(text)
        if self.raise_exc is not None:
            raise self.raise_exc
        return np.zeros(100, dtype=np.float32), self.sample_rate


class FakeStream:
    def __init__(self, **kwargs: Any) -> None:
        self._finished_callback = kwargs["finished_callback"]

    def start(self) -> None:
        threading.Thread(target=self._finish_soon, daemon=True).start()

    def _finish_soon(self) -> None:
        time.sleep(0.02)
        self._finished_callback()

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


def _fake_stream_factory(**kwargs: Any) -> FakeStream:
    return FakeStream(**kwargs)


class RecordingNotifier:
    def __init__(self) -> None:
        self.messages: "list[str]" = []

    def notify(self, title: str, message: str = "") -> None:
        self.messages.append(message)


def _make_feature(
    fallback: FakeSynthesizer,
    primary: FakeSynthesizer,
    notifier: RecordingNotifier,
    coordinator: ReadingCoordinator | None = None,
) -> "tuple[DocumentReaderFeature, list[str], list[tuple[int, int]], list[None]]":
    statuses: "list[str]" = []
    progresses: "list[tuple[int, int]]" = []
    finishes: "list[None]" = []
    feature = DocumentReaderFeature(
        model_manager=ModelManager(),
        primary_engine_factory=lambda: primary,  # type: ignore[return-value]
        fallback_engine=fallback,
        player=AudioPlayer(stream_factory=_fake_stream_factory),
        notifier=notifier,  # type: ignore[arg-type]
        reading_coordinator=coordinator or ReadingCoordinator(),
        on_status=statuses.append,
        on_progress=lambda current, total: progresses.append((current, total)),
        on_finished=lambda: finishes.append(None),
    )
    return feature, statuses, progresses, finishes


def test_read_text_speaks_every_chunk_and_reports_done() -> None:
    primary = FakeSynthesizer()
    feature, statuses, progresses, finishes = _make_feature(
        fallback=FakeSynthesizer(), primary=primary, notifier=RecordingNotifier()
    )

    feature.read_text("First sentence. Second sentence.")
    for _ in range(50):
        if finishes:
            break
        time.sleep(0.05)

    assert len(primary.calls) == 1  # short text fits in one chunk
    assert progresses[0] == (1, 1)
    assert "Done" in statuses
    assert finishes == [None]
    assert feature.is_reading is False


def test_read_file_extracts_and_speaks_a_real_file(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("Some file content.", encoding="utf-8")
    primary = FakeSynthesizer()
    feature, _statuses, _progresses, finishes = _make_feature(
        fallback=FakeSynthesizer(), primary=primary, notifier=RecordingNotifier()
    )

    feature.read_file(path)
    for _ in range(50):
        if finishes:
            break
        time.sleep(0.05)

    assert primary.calls == ["Some file content."]


def test_unsupported_file_reports_status_and_finishes(tmp_path: Path) -> None:
    path = tmp_path / "sample.rtf"
    path.write_text("nope", encoding="utf-8")
    feature, statuses, _progresses, finishes = _make_feature(
        fallback=FakeSynthesizer(), primary=FakeSynthesizer(), notifier=RecordingNotifier()
    )

    feature.read_file(path)
    for _ in range(50):
        if finishes:
            break
        time.sleep(0.05)

    assert any("Unsupported file type" in status for status in statuses)
    assert feature.is_reading is False


def test_primary_failure_falls_back_to_piper_once_per_document() -> None:
    primary = FakeSynthesizer(raise_exc=RuntimeError("gpu unavailable"))
    fallback = FakeSynthesizer()
    notifier = RecordingNotifier()
    feature, _statuses, _progresses, finishes = _make_feature(
        fallback=fallback, primary=primary, notifier=notifier
    )

    feature.read_text("One. Two. Three. Four. Five. Six. Seven. Eight.")
    for _ in range(50):
        if finishes:
            break
        time.sleep(0.05)

    assert fallback.calls  # every chunk after the first failure used Piper
    assert notifier.messages.count("Primary voice unavailable — using backup voice") == 1


def test_pause_stops_audio_immediately_and_resume_replays_current_chunk() -> None:
    primary = FakeSynthesizer()
    feature, statuses, _progresses, finishes = _make_feature(
        fallback=FakeSynthesizer(), primary=primary, notifier=RecordingNotifier()
    )

    feature.read_text("Only one short sentence here to read aloud slowly.")
    time.sleep(0.01)
    feature.pause()
    assert "Paused" in statuses
    assert feature.is_reading is True  # paused, not stopped

    feature.resume()
    for _ in range(50):
        if finishes:
            break
        time.sleep(0.05)
    assert "Done" in statuses


def test_stop_while_paused_actually_stops() -> None:
    """Regression guard: stopping while paused must not leave the worker
    thread blocked forever waiting on a resume that will never come."""
    primary = FakeSynthesizer()
    feature, statuses, _progresses, finishes = _make_feature(
        fallback=FakeSynthesizer(), primary=primary, notifier=RecordingNotifier()
    )

    feature.read_text("Only one short sentence here to read aloud slowly.")
    time.sleep(0.01)
    feature.pause()
    feature.stop()

    for _ in range(50):
        if finishes:
            break
        time.sleep(0.05)
    assert finishes == [None]
    assert feature.is_reading is False


def test_a_second_reader_preempts_this_one_via_the_coordinator() -> None:
    coordinator = ReadingCoordinator()
    primary = FakeSynthesizer()
    feature, _statuses, _progresses, finishes = _make_feature(
        fallback=FakeSynthesizer(), primary=primary, notifier=RecordingNotifier(), coordinator=coordinator
    )

    feature.read_text("Only one short sentence here to read aloud slowly.")
    time.sleep(0.01)
    coordinator.start_reading(object(), lambda: None)  # a different owner preempts it

    for _ in range(50):
        if finishes:
            break
        time.sleep(0.05)
    assert finishes == [None]
    assert feature.is_reading is False


def test_empty_pasted_text_reports_nothing_to_read() -> None:
    feature, statuses, _progresses, finishes = _make_feature(
        fallback=FakeSynthesizer(), primary=FakeSynthesizer(), notifier=RecordingNotifier()
    )

    feature.read_text("   ")
    for _ in range(50):
        if finishes:
            break
        time.sleep(0.05)

    assert "Nothing to read" in statuses
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_doc_reader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vocalis.features.doc_reader'`

- [ ] **Step 3: Write the implementation**

```python
"""Orchestrates Document Reader: extract text from a file (or take
pasted text directly), normalize + chunk it, speak each chunk via
Qwen3-TTS (falling back to Piper on any failure), and play it back chunk
by chunk -- with pause/resume/stop, progress reporting, and "one reading
at a time" shared with Quick Reader via ReadingCoordinator.

Has no PySide6/Qt import at all: status, progress, and completion are
reported through plain callables supplied by the caller (the popup, see
ui/doc_popup.py), keeping this module independently testable exactly
like every other feature module. Reuses the never-block-the-caller and
generation-counter patterns already hard-won in quick_reader.py.

Pause/resume design: pause() stops audio immediately; resume() replays
the current chunk from its start (simpler than tracking a mid-chunk
playback offset, and each chunk is short by construction -- see
text/chunker.py). A worker thread that is stopped early by pause()
blocks on self._resume_event until resume() sets it, then re-checks
whether this reading is still current before replaying -- this is what
lets stop() (or a ReadingCoordinator preemption) while paused actually
end the reading, rather than leaving the worker thread stuck waiting for
a resume() that will never come.
"""
import logging
import threading
from pathlib import Path
from typing import Callable, Protocol, cast

import numpy as np
import numpy.typing as npt

from vocalis.audio.playback import AudioPlayer
from vocalis.features.reading_coordinator import ReadingCoordinator
from vocalis.model_manager import ModelEngine, ModelManager
from vocalis.notify import Notifier
from vocalis.text.chunker import chunk_text
from vocalis.text.extractors import (
    DocumentOpenError,
    DocumentTooLargeError,
    NoReadableTextError,
    UnsupportedFileTypeError,
    extract_text,
)
from vocalis.text.normalizer import normalize_markdown

logger = logging.getLogger("vocalis")

_ENGINE_ID = "document_reader_tts"
_TITLE = "Vocalis — Document Reader"

_EXTRACTION_ERRORS = (
    UnsupportedFileTypeError,
    DocumentOpenError,
    NoReadableTextError,
    DocumentTooLargeError,
)


class Synthesizer(Protocol):
    """The subset of a TTS engine this module depends on."""

    def synthesize(self, text: str) -> "tuple[npt.NDArray[np.float32], int]": ...


class DocumentReaderFeature:
    """Wires Ctrl+Alt+D / the popup's controls to extract -> chunk ->
    speak (with fallback) -> play, chunk by chunk, with pause/resume/stop."""

    def __init__(
        self,
        model_manager: ModelManager,
        primary_engine_factory: Callable[[], ModelEngine],
        fallback_engine: Synthesizer,
        player: AudioPlayer,
        notifier: Notifier,
        reading_coordinator: ReadingCoordinator,
        on_status: Callable[[str], None],
        on_progress: Callable[[int, int], None],
        on_finished: Callable[[], None],
        language: str = "English",
    ) -> None:
        """Create a DocumentReaderFeature.

        Args:
            model_manager: Shared ModelManager (same instance every
                other feature uses) enforcing single-GPU-residency; used
                only for the primary engine.
            primary_engine_factory: Constructs a fresh primary engine
                (Qwen3TTS) when it needs to be (re)loaded.
            fallback_engine: An already-loaded, CPU-only engine (Piper)
                used when the primary engine fails.
            player: Plays generated speech and can be interrupted.
            notifier: Sends desktop notifications.
            reading_coordinator: Shared with Quick Reader so only one
                feature plays audio at a time.
            on_status: Called with a human-readable status string at
                each stage (e.g. "Reading...", "Paused", "Done"). The
                popup wires this to a Qt signal to update its status line.
            on_progress: Called with (current_chunk, total_chunks), both
                1-based, at the start of each chunk. The popup wires this
                to a Qt signal to update its progress label.
            on_finished: Called once when a reading ends for any reason
                (finished, stopped, or failed). The popup wires this to
                a Qt signal that re-enables its controls.
            language: Spoken language passed to the primary engine.
        """
        self._model_manager = model_manager
        self._primary_engine_factory = primary_engine_factory
        self._fallback_engine = fallback_engine
        self._player = player
        self._notifier = notifier
        self._reading_coordinator = reading_coordinator
        self._on_status = on_status
        self._on_progress = on_progress
        self._on_finished = on_finished
        self._language = language

        self._lock = threading.Lock()
        self._active = False
        self._paused = False
        self._use_fallback_for_rest = False
        self._generation = 0
        self._resume_event = threading.Event()
        self._resume_event.set()

    @property
    def is_reading(self) -> bool:
        """Whether a reading is currently active (including while paused)."""
        with self._lock:
            return self._active

    def read_file(self, path: Path) -> None:
        """Start reading a document file aloud from the beginning.

        Extraction, normalization, and chunking all happen on the
        background worker thread spawned here, so this call returns
        immediately -- safe to call from the Qt UI thread.

        Args:
            path: Path to the file to read.
        """
        generation = self._start_new_reading()
        self._reading_coordinator.start_reading(self, self.stop)
        threading.Thread(
            target=self._prepare_and_read, args=(generation, path, None), daemon=True
        ).start()

    def read_text(self, text: str) -> None:
        """Start reading pasted text aloud from the beginning.

        Args:
            text: The raw text to read (markdown-normalized before speaking).
        """
        generation = self._start_new_reading()
        self._reading_coordinator.start_reading(self, self.stop)
        threading.Thread(
            target=self._prepare_and_read, args=(generation, None, text), daemon=True
        ).start()

    def pause(self) -> None:
        """Pause the current reading immediately. No-op if not reading
        or already paused."""
        with self._lock:
            if not self._active or self._paused:
                return
            self._paused = True
            self._resume_event.clear()
        self._player.stop()
        self._notifier.notify(_TITLE, "Paused")
        self._on_status("Paused")

    def resume(self) -> None:
        """Resume a paused reading, replaying the current chunk from its
        start. No-op if not currently paused."""
        with self._lock:
            if not self._active or not self._paused:
                return
            self._paused = False
        self._resume_event.set()
        self._notifier.notify(_TITLE, "Resuming...")
        self._on_status("Resuming...")

    def stop(self) -> None:
        """Stop the current reading immediately. No-op if not reading.

        Safe to call from any thread, including as a ReadingCoordinator
        stopper callback when Quick Reader preempts this reading.
        Unblocks a worker thread that is currently waiting on a pause,
        so a stop while paused actually ends the reading instead of
        leaving that thread stuck forever.
        """
        with self._lock:
            if not self._active:
                return
            self._active = False
            self._paused = False
        self._resume_event.set()
        self._reading_coordinator.finish_reading(self)
        self._player.stop()
        self._notifier.notify(_TITLE, "Stopped")
        self._on_status("Stopped")
        self._on_finished()

    def _start_new_reading(self) -> int:
        """Mark a fresh reading as active and return its generation
        number. Also unconditionally releases any stale pause-wait from
        a previous reading (belt-and-braces: the popup disables its Read
        button while is_reading is True, so this should not be reachable
        mid-pause in normal use, but a blocked-forever worker thread
        would be a real bug if it ever were)."""
        with self._lock:
            self._active = True
            self._paused = False
            self._use_fallback_for_rest = False
            self._generation += 1
            generation = self._generation
        self._resume_event.set()
        return generation

    def _is_current(self, generation: int) -> bool:
        """Whether `generation` is still the active reading."""
        with self._lock:
            return self._active and self._generation == generation

    def _deactivate(self, generation: int) -> None:
        """Mark `generation` as finished and release ReadingCoordinator
        ownership, but only if `generation` is still current -- a stale
        worker thread must not clobber a newer reading's state. Does not
        call on_finished() itself; every caller does that explicitly
        right after, so callers control exactly when status/notify
        callbacks fire relative to on_finished()."""
        with self._lock:
            if self._generation != generation:
                return
            self._active = False
            self._paused = False
        self._reading_coordinator.finish_reading(self)

    def _prepare_and_read(
        self, generation: int, path: "Path | None", text: "str | None"
    ) -> None:
        """Extract (if a file), normalize, and chunk the text, then read
        it chunk by chunk. Runs entirely on a background thread."""
        try:
            raw_text = extract_text(path) if path is not None else text or ""
        except _EXTRACTION_ERRORS as exc:
            logger.warning("Document extraction failed: %s", exc)
            self._notifier.notify(_TITLE, str(exc))
            self._on_status(str(exc))
            self._deactivate(generation)
            self._on_finished()
            return

        normalized = normalize_markdown(raw_text)
        chunks = chunk_text(normalized)
        if not chunks:
            self._notifier.notify(_TITLE, "Nothing to read")
            self._on_status("Nothing to read")
            self._deactivate(generation)
            self._on_finished()
            return

        if not self._is_current(generation):
            return  # stopped while extracting/chunking

        self._notifier.notify(_TITLE, "Reading...")
        self._on_status("Reading...")
        self._read_worker(generation, chunks)

    def _read_worker(self, generation: int, chunks: "list[str]") -> None:
        """Synthesize and play each chunk in order. Bails out cleanly at
        each check if this generation has been superseded by a stop, a
        coordinator preemption, or a fresh reading. On an early stop
        (stop() or wait_until_done() returning False) while paused,
        blocks on _resume_event and, once woken, either replays the
        current chunk (resumed) or exits cleanly (stopped while waiting).
        """
        total = len(chunks)
        index = 0
        while index < total:
            if not self._is_current(generation):
                return
            chunk = chunks[index]
            self._on_progress(index + 1, total)
            try:
                audio, sample_rate = self._synthesize_with_fallback(chunk)
            except Exception:
                logger.exception("Both TTS engines failed")
                self._notifier.notify(_TITLE, "Could not generate speech")
                self._on_status("Could not generate speech")
                self._deactivate(generation)
                self._on_finished()
                return
            if not self._is_current(generation):
                return

            try:
                self._player.play(audio, sample_rate)
                finished_naturally = self._player.wait_until_done()
            except Exception:
                logger.exception("Playback failed")
                self._notifier.notify(_TITLE, "Playback failed")
                self._on_status("Playback failed")
                self._deactivate(generation)
                self._on_finished()
                return

            if finished_naturally:
                index += 1
                continue

            with self._lock:
                paused = self._paused
            if not paused:
                self._deactivate(generation)
                self._on_finished()
                return
            self._resume_event.wait()
            if not self._is_current(generation):
                return
            # loop again with the same index -- replays the current chunk

        self._deactivate(generation)
        self._notifier.notify(_TITLE, "Done")
        self._on_status("Done")
        self._on_finished()

    def _synthesize_with_fallback(self, text: str) -> "tuple[npt.NDArray[np.float32], int]":
        """Try the primary engine; fall back to the secondary on any
        failure. Once fallback triggers for this reading, stays on Piper
        for the rest of the document rather than retrying Qwen3-TTS every
        chunk, and notifies the user once, not once per chunk."""
        if self._use_fallback_for_rest:
            return self._fallback_engine.synthesize(text)
        try:
            engine = self._model_manager.acquire(_ENGINE_ID, self._primary_engine_factory)
            try:
                return cast(Synthesizer, engine).synthesize(text)
            finally:
                self._model_manager.release(_ENGINE_ID)
        except Exception:
            logger.exception("Primary TTS engine failed; falling back to Piper for this document")
            self._use_fallback_for_rest = True
            self._notifier.notify(_TITLE, "Primary voice unavailable — using backup voice")
            self._on_status("Using backup voice")
            return self._fallback_engine.synthesize(text)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_doc_reader.py -v`
Expected: PASS (9 passed). If `test_stop_while_paused_actually_stops` or
`test_a_second_reader_preempts_this_one_via_the_coordinator` hangs
instead of failing cleanly, that is the exact bug this task's design
notes warn about (a worker thread stuck on `_resume_event.wait()`) —
double-check `stop()` calls `self._resume_event.set()` before returning.

- [ ] **Step 5: Run the full test suite, coverage, and mypy**

Run: `.venv/bin/pytest -q && .venv/bin/mypy src`
Expected: all tests pass; coverage still ≥85%; mypy clean.

- [ ] **Step 6: Commit**

```bash
git add src/vocalis/features/doc_reader.py tests/test_doc_reader.py
git commit -m "feat: add DocumentReaderFeature with pause/resume/stop"
```

---

## Task 7: PySide6 popup

**Files:**
- Create: `src/vocalis/ui/__init__.py` (empty)
- Create: `src/vocalis/ui/doc_popup.py`
- Create: `tests/conftest.py` — sets `QT_QPA_PLATFORM=offscreen` so PySide6 never needs a real display, in CI or on this machine.
- Test: `tests/test_doc_popup.py`

**Interfaces:**
- Consumes: any object matching `DocumentReaderFeature`'s public shape (`is_reading`, `read_file`, `read_text`, `pause`, `resume`, `stop`) — tests use a plain fake, not a real `DocumentReaderFeature`, keeping this task's tests fast and decoupled from Task 6's internals.
- Produces: `DocumentReaderPopup(feature)`, a `QWidget` with `on_hotkey_trigger() -> None` and Qt signals `status_changed(str)`, `progress_changed(int, int)`, `reading_finished()`.

- [ ] **Step 1: Add the offscreen-Qt conftest**

```python
"""Shared pytest configuration: run PySide6 without a real display.

Set before any PySide6 import happens anywhere in the test suite (a
conftest.py at the tests/ root is imported by pytest before collecting
any test module), so every test — not just the ones that mention Qt —
can safely construct widgets in CI or on a headless machine.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
```

- [ ] **Step 2: Write the failing test**

```python
"""Tests for vocalis.ui.doc_popup.DocumentReaderPopup.

Uses a plain fake in place of DocumentReaderFeature -- this suite
verifies the popup's own button-wiring and signal-handling logic only,
decoupled from the feature's internals (which test_doc_reader.py already
covers).
"""
from vocalis.ui.doc_popup import DocumentReaderPopup


class FakeFeature:
    def __init__(self) -> None:
        self.is_reading = False
        self.read_file_calls: "list[object]" = []
        self.read_text_calls: "list[str]" = []
        self.pause_calls = 0
        self.resume_calls = 0
        self.stop_calls = 0

    def read_file(self, path: object) -> None:
        self.read_file_calls.append(path)

    def read_text(self, text: str) -> None:
        self.read_text_calls.append(text)

    def pause(self) -> None:
        self.pause_calls += 1

    def resume(self) -> None:
        self.resume_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1


def test_read_button_sends_the_typed_text_to_the_feature(qtbot) -> None:  # type: ignore[no-untyped-def]
    feature = FakeFeature()
    popup = DocumentReaderPopup(feature)  # type: ignore[arg-type]
    qtbot.addWidget(popup)

    popup._text_box.setPlainText("Hello there")  # noqa: SLF001
    popup._on_read_clicked()  # noqa: SLF001

    assert feature.read_text_calls == ["Hello there"]
    assert popup._read_button.isEnabled() is False  # noqa: SLF001
    assert popup._stop_button.isEnabled() is True  # noqa: SLF001


def test_read_button_with_empty_text_does_not_call_the_feature(qtbot) -> None:  # type: ignore[no-untyped-def]
    feature = FakeFeature()
    popup = DocumentReaderPopup(feature)  # type: ignore[arg-type]
    qtbot.addWidget(popup)

    popup._text_box.setPlainText("   ")  # noqa: SLF001
    popup._on_read_clicked()  # noqa: SLF001

    assert feature.read_text_calls == []


def test_pause_button_toggles_and_calls_pause_then_resume(qtbot) -> None:  # type: ignore[no-untyped-def]
    feature = FakeFeature()
    popup = DocumentReaderPopup(feature)  # type: ignore[arg-type]
    qtbot.addWidget(popup)

    popup._on_pause_clicked()  # noqa: SLF001
    assert feature.pause_calls == 1
    assert popup._pause_button.text() == "Resume"  # noqa: SLF001

    popup._on_pause_clicked()  # noqa: SLF001
    assert feature.resume_calls == 1
    assert popup._pause_button.text() == "Pause"  # noqa: SLF001


def test_stop_button_calls_the_feature() -> None:
    feature = FakeFeature()
    popup = DocumentReaderPopup(feature)  # type: ignore[arg-type]

    popup._on_stop_clicked()  # noqa: SLF001

    assert feature.stop_calls == 1


def test_status_signal_updates_the_status_label() -> None:
    popup = DocumentReaderPopup(FakeFeature())  # type: ignore[arg-type]

    popup.status_changed.emit("Reading...")

    assert popup._status_label.text() == "Reading..."  # noqa: SLF001


def test_progress_signal_updates_the_progress_label() -> None:
    popup = DocumentReaderPopup(FakeFeature())  # type: ignore[arg-type]

    popup.progress_changed.emit(3, 10)

    assert popup._progress_label.text() == "piece 3 of 10"  # noqa: SLF001


def test_reading_finished_signal_disables_pause_and_stop_and_resets_pause_label() -> None:
    popup = DocumentReaderPopup(FakeFeature())  # type: ignore[arg-type]
    popup._on_pause_clicked()  # noqa: SLF001 -- flips to "Resume"

    popup.reading_finished.emit()

    assert popup._pause_button.isEnabled() is False  # noqa: SLF001
    assert popup._stop_button.isEnabled() is False  # noqa: SLF001
    assert popup._read_button.isEnabled() is True  # noqa: SLF001
    assert popup._pause_button.text() == "Pause"  # noqa: SLF001


def test_hotkey_trigger_while_idle_shows_the_window(qtbot) -> None:  # type: ignore[no-untyped-def]
    feature = FakeFeature()
    feature.is_reading = False
    popup = DocumentReaderPopup(feature)  # type: ignore[arg-type]
    qtbot.addWidget(popup)

    popup.on_hotkey_trigger()

    assert popup.isVisible() is True
    assert feature.stop_calls == 0


def test_hotkey_trigger_while_reading_stops_it() -> None:
    feature = FakeFeature()
    feature.is_reading = True
    popup = DocumentReaderPopup(feature)  # type: ignore[arg-type]

    popup.on_hotkey_trigger()

    assert feature.stop_calls == 1
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_doc_popup.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vocalis.ui'`

- [ ] **Step 4: Write the implementation**

`src/vocalis/ui/__init__.py`: empty file.

`src/vocalis/ui/doc_popup.py`:

```python
"""The Document Reader popup window: Open file / paste text / Read /
Pause-Resume / Stop, with a progress label.

Talks to a DocumentReaderFeature only through its public methods and the
three callables it takes at construction (on_status/on_progress/
on_finished, wired in cli.py's build_doc_reader() to this popup's *.emit
methods) -- this module is the only place PySide6 is imported for
Document Reader, keeping doc_reader.py itself independently testable
without a display.

Cross-thread signal emission: DocumentReaderFeature calls on_status/
on_progress/on_finished from its own background worker thread, never
from the Qt UI thread. Each is wired to emit a Qt signal here rather than
touch a widget directly -- Qt auto-queues a signal's delivery onto the
receiving object's own thread when sender and receiver live on different
threads, which is what makes this safe without any locking of our own.
"""
from pathlib import Path
from typing import Protocol

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

_SUPPORTED_FILE_FILTER = "Documents (*.txt *.md *.pdf *.docx *.html *.htm);;All files (*)"


class DocumentReaderLike(Protocol):
    """The subset of DocumentReaderFeature this popup depends on."""

    @property
    def is_reading(self) -> bool: ...
    def read_file(self, path: Path) -> None: ...
    def read_text(self, text: str) -> None: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...
    def stop(self) -> None: ...


class DocumentReaderPopup(QWidget):
    """The Ctrl+Alt+D popup window."""

    status_changed = Signal(str)
    progress_changed = Signal(int, int)
    reading_finished = Signal()

    def __init__(self, feature: DocumentReaderLike) -> None:
        """Create the popup, wired to the given feature.

        Args:
            feature: The DocumentReaderFeature this popup controls.
        """
        super().__init__()
        self._feature = feature
        self.setWindowTitle("Vocalis — Document Reader")

        self._text_box = QPlainTextEdit()
        self._text_box.setPlaceholderText("Paste text here, or open a file below...")

        self._open_button = QPushButton("Open file...")
        self._read_button = QPushButton("Read")
        self._pause_button = QPushButton("Pause")
        self._stop_button = QPushButton("Stop")
        self._status_label = QLabel("Idle")
        self._progress_label = QLabel("")

        self._open_button.clicked.connect(self._on_open_clicked)
        self._read_button.clicked.connect(self._on_read_clicked)
        self._pause_button.clicked.connect(self._on_pause_clicked)
        self._stop_button.clicked.connect(self._on_stop_clicked)

        self.status_changed.connect(self._status_label.setText)
        self.progress_changed.connect(self._update_progress_label)
        self.reading_finished.connect(self._on_reading_finished)

        button_row = QHBoxLayout()
        button_row.addWidget(self._open_button)
        button_row.addWidget(self._read_button)
        button_row.addWidget(self._pause_button)
        button_row.addWidget(self._stop_button)

        layout = QVBoxLayout()
        layout.addWidget(self._text_box)
        layout.addLayout(button_row)
        layout.addWidget(self._status_label)
        layout.addWidget(self._progress_label)
        self.setLayout(layout)

        self._set_reading_controls_enabled(False)

    def on_hotkey_trigger(self) -> None:
        """Call when Ctrl+Alt+D is pressed: stop an in-progress reading,
        or raise/focus this window if idle."""
        if self._feature.is_reading:
            self._feature.stop()
            return
        self.show()
        self.raise_()
        self.activateWindow()

    def _on_open_clicked(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Open document", "", _SUPPORTED_FILE_FILTER
        )
        if not path_str:
            return
        self._set_reading_controls_enabled(True)
        self._feature.read_file(Path(path_str))

    def _on_read_clicked(self) -> None:
        text = self._text_box.toPlainText()
        if not text.strip():
            self._status_label.setText("Nothing to read")
            return
        self._set_reading_controls_enabled(True)
        self._feature.read_text(text)

    def _on_pause_clicked(self) -> None:
        if self._pause_button.text() == "Pause":
            self._feature.pause()
            self._pause_button.setText("Resume")
        else:
            self._feature.resume()
            self._pause_button.setText("Pause")

    def _on_stop_clicked(self) -> None:
        self._feature.stop()

    def _on_reading_finished(self) -> None:
        self._set_reading_controls_enabled(False)
        self._pause_button.setText("Pause")

    def _update_progress_label(self, current: int, total: int) -> None:
        self._progress_label.setText(f"piece {current} of {total}")

    def _set_reading_controls_enabled(self, reading: bool) -> None:
        self._open_button.setEnabled(not reading)
        self._read_button.setEnabled(not reading)
        self._pause_button.setEnabled(reading)
        self._stop_button.setEnabled(reading)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_doc_popup.py -v`
Expected: PASS (9 passed). `pytest-qt` supplies the `qtbot` fixture automatically once installed (Task 3 already added it); no extra plugin registration is needed.

- [ ] **Step 6: Run the full test suite, coverage, and mypy**

Run: `.venv/bin/pytest -q && .venv/bin/mypy src`
Expected: all tests pass; coverage still ≥85%; mypy clean. If mypy complains about the `PySide6` or `qtbot` imports, add the narrowest override that fixes it (e.g. `[[tool.mypy.overrides]] module = "pytestqt.*"` with `ignore_missing_imports = true`) rather than disabling strict mode.

- [ ] **Step 7: Commit**

```bash
git add tests/conftest.py src/vocalis/ui/__init__.py src/vocalis/ui/doc_popup.py tests/test_doc_popup.py
git commit -m "feat: add PySide6 Document Reader popup"
```

---

## Task 8: CLI/config wiring — Qt event loop replaces the blocking wait

**Files:**
- Modify: `src/vocalis/config.py:62` — `doc_tts: str = "fish_speech"` → `doc_tts: str = "qwen3_tts"`.
- Modify: `src/vocalis/cli.py` — add `build_doc_reader()`; add a `reading_coordinator` parameter to `build_quick_reader()`; replace `_block_until_interrupted()` with the Qt event loop in `main()`.
- Test: `tests/test_config.py` — update the one assertion that checks the old default.
- Test: `tests/test_cli.py` — add coverage for `build_doc_reader()` and the updated `build_quick_reader()` signature.

**Interfaces:**
- Consumes: `DocumentReaderFeature` (Task 6), `DocumentReaderPopup` (Task 7), `CtrlAltKeyListener`/`list_devices_with_key` (Task 4), `ReadingCoordinator` (Task 5).
- Produces: `build_doc_reader(config, model_manager, reading_coordinator) -> tuple[DocumentReaderFeature, DocumentReaderPopup, CtrlAltKeyListener]`.

- [ ] **Step 1: Update the stale config default**

In `src/vocalis/config.py`, change line 62:

```python
    doc_tts: str = "qwen3_tts"
```

(was `doc_tts: str = "fish_speech"` — fish-speech was dropped for Document Reader per the Phase 4 amendment; nothing in the codebase currently branches on this field's value, same as `quick_tts`, but its label should not lie about which engine is actually used.)

In `tests/test_config.py`, find the assertion checking `VocalisConfig().engines.doc_tts` (or the equivalent default-values test) and update the expected value from `"fish_speech"` to `"qwen3_tts"`.

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: PASS.

- [ ] **Step 2: Write the failing test for build_doc_reader and the updated build_quick_reader**

Change the existing top-of-file import line in `tests/test_cli.py` from:

```python
from vocalis.cli import build_daemon, build_dictation, build_quick_reader, main
```

to:

```python
from vocalis.cli import build_daemon, build_dictation, build_doc_reader, build_quick_reader, main
```

and add one new top-of-file import alongside it:

```python
from vocalis.features.reading_coordinator import ReadingCoordinator
```

Then add to `tests/test_cli.py`:

```python
def test_build_quick_reader_accepts_a_shared_reading_coordinator() -> None:
    config = VocalisConfig()
    manager = ModelManager()
    coordinator = ReadingCoordinator()

    feature, _listener = build_quick_reader(
        config=config, model_manager=manager, reading_coordinator=coordinator
    )

    assert feature._reading_coordinator is coordinator  # noqa: SLF001


def test_build_doc_reader_wires_engines_popup_and_listener(qtbot) -> None:  # type: ignore[no-untyped-def]
    config = VocalisConfig()
    manager = ModelManager()
    coordinator = ReadingCoordinator()

    feature, popup, listener = build_doc_reader(
        config=config, model_manager=manager, reading_coordinator=coordinator
    )
    qtbot.addWidget(popup)

    assert feature._reading_coordinator is coordinator  # noqa: SLF001
    assert listener._on_trigger == popup.on_hotkey_trigger  # noqa: SLF001

    # The on_status/on_progress/on_finished callbacks wired to the popup's
    # signals must actually reach the popup's widgets, proving the
    # construction-order workaround (feature built before popup exists,
    # popup referenced by closure) really works end-to-end.
    feature._on_status("Reading...")  # noqa: SLF001
    assert popup._status_label.text() == "Reading..."  # noqa: SLF001
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_cli.py -v -k "build_doc_reader or shared_reading_coordinator"`
Expected: FAIL (`build_quick_reader() got an unexpected keyword argument 'reading_coordinator'`, then `ImportError: cannot import name 'build_doc_reader'`).

- [ ] **Step 4: Implement build_doc_reader, update build_quick_reader and main()**

In `src/vocalis/cli.py`, add these imports alongside the existing ones:

```python
import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from vocalis.features.doc_reader import DocumentReaderFeature
from vocalis.features.reading_coordinator import ReadingCoordinator
from vocalis.hotkeys.ctrl_alt_key import CtrlAltKeyListener, list_devices_with_key
from vocalis.ui.doc_popup import DocumentReaderPopup

_KEY_D = 32
```

Change `build_quick_reader`'s signature and body (`cli.py:92-132`) to accept and pass through the shared coordinator:

```python
def build_quick_reader(
    config: VocalisConfig, model_manager: ModelManager, reading_coordinator: ReadingCoordinator
) -> tuple[QuickReaderFeature, CtrlAltRListener]:
    """Construct the Quick Reader feature and its Ctrl+Alt+R listener.

    Args:
        config: Loaded, validated vocalis configuration.
        model_manager: The shared ModelManager (same instance passed to
            build_daemon() and build_dictation()) — only the primary
            (Qwen3-TTS) engine is routed through it; the Piper fallback
            never touches the GPU and is constructed/loaded directly.
        reading_coordinator: Shared with build_doc_reader() so Quick
            Reader and Document Reader never play audio at the same time.

    Returns:
        A (feature, listener) pair. The caller is responsible for calling
        listener.start()/listener.stop() around the daemon's own lifecycle.
    """
    qwen_model_dir = Path(config.engines.qwen3_tts_model_dir)
    speaker = config.engines.qwen3_tts_speaker

    def primary_engine_factory() -> Qwen3TTS:
        return Qwen3TTS(model_dir=qwen_model_dir, speaker=speaker)

    fallback_engine = PiperTTS(
        binary_path=Path(config.engines.piper_binary_path),
        model_path=Path(config.engines.piper_voice_model_path),
    )
    fallback_engine.load()

    feature = QuickReaderFeature(
        model_manager=model_manager,
        primary_engine_factory=primary_engine_factory,
        fallback_engine=fallback_engine,
        clipboard_reader=ClipboardReader(),
        player=AudioPlayer(),
        notifier=Notifier(enabled=config.notifications_enabled),
        reading_coordinator=reading_coordinator,
    )
    listener = CtrlAltRListener(
        on_trigger=feature.on_trigger,
        device_enumerator=list_hotkey_capable_devices,
    )
    return feature, listener
```

Add `build_doc_reader` right after it:

```python
def build_doc_reader(
    config: VocalisConfig, model_manager: ModelManager, reading_coordinator: ReadingCoordinator
) -> tuple[DocumentReaderFeature, DocumentReaderPopup, CtrlAltKeyListener]:
    """Construct the Document Reader feature, its popup, and its
    Ctrl+Alt+D listener.

    The feature is constructed before the popup that will display its
    status, but the feature's on_status/on_progress/on_finished
    callbacks need to call methods on that not-yet-created popup. This
    is resolved with a closure over a `popup` variable assigned after
    the feature is built: by the time any callback is actually
    *invoked* (only once a reading starts, which cannot happen before
    this function returns and the caller starts the listener), `popup`
    is already bound. Python closures capture variables by reference,
    not by value at definition time, so this ordering is safe.

    Args:
        config: Loaded, validated vocalis configuration.
        model_manager: The shared ModelManager (same instance passed to
            build_daemon(), build_dictation(), and build_quick_reader()).
        reading_coordinator: Shared with build_quick_reader() so Quick
            Reader and Document Reader never play audio at the same time.

    Returns:
        A (feature, popup, listener) triple. The caller is responsible
        for calling listener.start()/listener.stop() around the daemon's
        own lifecycle; the popup itself needs no start()/stop() — it
        stays hidden until the user first presses Ctrl+Alt+D.
    """
    qwen_model_dir = Path(config.engines.qwen3_tts_model_dir)
    speaker = config.engines.qwen3_tts_speaker

    def primary_engine_factory() -> Qwen3TTS:
        return Qwen3TTS(model_dir=qwen_model_dir, speaker=speaker)

    fallback_engine = PiperTTS(
        binary_path=Path(config.engines.piper_binary_path),
        model_path=Path(config.engines.piper_voice_model_path),
    )
    fallback_engine.load()

    popup: DocumentReaderPopup

    def on_status(message: str) -> None:
        popup.status_changed.emit(message)

    def on_progress(current: int, total: int) -> None:
        popup.progress_changed.emit(current, total)

    def on_finished() -> None:
        popup.reading_finished.emit()

    feature = DocumentReaderFeature(
        model_manager=model_manager,
        primary_engine_factory=primary_engine_factory,
        fallback_engine=fallback_engine,
        player=AudioPlayer(),
        notifier=Notifier(enabled=config.notifications_enabled),
        reading_coordinator=reading_coordinator,
        on_status=on_status,
        on_progress=on_progress,
        on_finished=on_finished,
    )
    popup = DocumentReaderPopup(feature)
    listener = CtrlAltKeyListener(
        key_code=_KEY_D,
        on_trigger=popup.on_hotkey_trigger,
        device_enumerator=lambda: list_devices_with_key(_KEY_D),
    )
    return feature, popup, listener
```

Replace `_block_until_interrupted()` and `main()` (`cli.py:135-171`):

```python
def main() -> None:
    """Entrypoint for the `vocalis` console script.

    Loads config, starts the daemon and all three hotkey listeners
    (dictation's NumlockListener, quick reader's CtrlAltRListener,
    document reader's CtrlAltKeyListener), and runs the PySide6 Qt event
    loop until interrupted, at which point it stops everything cleanly.
    Every start()/stop() call is inside the try/finally, so daemon.stop()
    (and therefore ModelManager.shutdown()) always runs on the way out.

    The Qt event loop (app.exec()) replaces the previous bare sleep loop
    -- Document Reader's popup needs one running Qt application in this
    process, and there is exactly one process, so it runs on the main
    thread here. The keepalive_timer below exists only so Ctrl+C works:
    Qt's C++ event loop does not hand control back to the Python
    interpreter between iterations unless something Python-level runs
    periodically, so without it, a signal handler's KeyboardInterrupt
    would not be able to unwind through app.exec() at all. This is a
    well-known PySide6/PyQt quirk, not a Vocalis-specific workaround.
    """
    config = load_config(default_config_path())
    model_manager = ModelManager(idle_timeout_seconds=config.model_idle_timeout_seconds)
    reading_coordinator = ReadingCoordinator()
    daemon = build_daemon(model_manager=model_manager, socket_path=_socket_path())
    _, dictation_listener = build_dictation(config=config, model_manager=model_manager)
    _, quick_reader_listener = build_quick_reader(
        config=config, model_manager=model_manager, reading_coordinator=reading_coordinator
    )
    app = QApplication.instance() or QApplication(sys.argv)
    _, _doc_popup, doc_reader_listener = build_doc_reader(
        config=config, model_manager=model_manager, reading_coordinator=reading_coordinator
    )
    keepalive_timer = QTimer()
    keepalive_timer.timeout.connect(lambda: None)
    keepalive_timer.start(200)
    try:
        daemon.start()
        dictation_listener.start()
        quick_reader_listener.start()
        doc_reader_listener.start()
        app.exec()
    except KeyboardInterrupt:
        pass
    finally:
        doc_reader_listener.stop()
        quick_reader_listener.stop()
        dictation_listener.stop()
        daemon.stop()
```

Remove the now-unused `_block_until_interrupted` function and the now-unused `import time` at the top of the file if nothing else in the file uses `time`.

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_cli.py -v`
Expected: PASS, including all pre-existing tests in the file plus the two new ones.

- [ ] **Step 6: Run the full test suite, coverage, and mypy**

Run: `.venv/bin/pytest -q && .venv/bin/mypy src`
Expected: all tests pass; coverage still ≥85%; mypy clean.

- [ ] **Step 7: Commit**

```bash
git add src/vocalis/config.py src/vocalis/cli.py tests/test_config.py tests/test_cli.py
git commit -m "feat: wire Document Reader into cli.py with a Qt event loop"
```

---

## Task 9: Docs + changelog

**Files:**
- Modify: `docs/architecture.md`, `docs/folder-structure.md`, `docs/file-purpose.md`, `docs/workflow.md`, `docs/user-docs.md` — add Document Reader.
- Create: `.claude/<timestamp>_CHANGELOG.md`.

**Interfaces:** none (docs only).

- [ ] **Step 1: Update docs/architecture.md**

Add a short "Document Reader" subsection next to the existing Quick Reader one, describing: `DocumentReaderFeature` (extract → normalize → chunk → synthesize-with-fallback → play, chunk by chunk, pause/resume/stop), the new `ReadingCoordinator` making it mutually exclusive with Quick Reader, and that the daemon's `main()` now runs a PySide6 Qt event loop instead of a bare sleep loop.

- [ ] **Step 2: Update docs/folder-structure.md**

Add entries for `src/vocalis/text/` (extractors.py, normalizer.py, chunker.py), `src/vocalis/ui/` (doc_popup.py), and the new `src/vocalis/hotkeys/ctrl_alt_key.py` and `src/vocalis/features/reading_coordinator.py`/`doc_reader.py` files.

- [ ] **Step 3: Update docs/file-purpose.md**

Add one-line purpose descriptions for every new file from Tasks 1–8: `text/normalizer.py`, `text/chunker.py`, `text/extractors.py`, `hotkeys/ctrl_alt_key.py`, `features/reading_coordinator.py`, `features/doc_reader.py`, `ui/doc_popup.py`. Update the description of `hotkeys/ctrl_alt_r.py` (now a thin wrapper) and `cli.py` (now also builds Document Reader and runs the Qt event loop).

- [ ] **Step 4: Update docs/workflow.md**

Add a "Document Reader" flow section (user perspective: Ctrl+Alt+D opens the popup, Open file/paste text, Read, Pause/Resume, Stop; bot logic: extract → normalize → chunk → synthesize-with-fallback → play per chunk, with the ReadingCoordinator note that starting Quick Reader stops Document Reader and vice versa). Double check it against the real code in `doc_reader.py`/`doc_popup.py` rather than the plan text, the same way Phase 3's Task 8 found and fixed a diagram that had drifted from the real implementation.

- [ ] **Step 5: Update docs/user-docs.md**

Add a "Document Reader (Phase 4)" section, same style as the existing "Quick Reader (Phase 3)" one: how to open it (Ctrl+Alt+D), what "Open file..." supports (.txt/.md/.pdf/.docx/.html), Pause/Resume/Stop, what "Nothing to read"/"Unsupported file type"/"No readable text found" messages mean and what to do about them, and that starting Quick Reader while Document Reader is running (or vice versa) stops the other one.

- [ ] **Step 6: Run the full verification suite one more time**

Run: `.venv/bin/pytest -q && .venv/bin/mypy src`
Expected: all tests pass; coverage ≥85%; mypy clean. (Docs changes don't affect this, but this is the last task, so it's the last checkpoint before the branch is considered done.)

- [ ] **Step 7: Write the changelog and commit**

Create `.claude/<YYYY-MM-DD_HHMM>_CHANGELOG.md` summarizing Phase 4 end to end: the 8 new/modified modules, the new dependencies (PySide6, pypdf, python-docx, dev: pytest-qt, reportlab), the ReadingCoordinator design, the Qt-event-loop change to `main()`, and final test/coverage/mypy numbers.

```bash
git add docs/architecture.md docs/folder-structure.md docs/file-purpose.md docs/workflow.md docs/user-docs.md .claude/*_CHANGELOG.md
git commit -m "docs: document phase 4 document reader feature"
```

---

## Verification (end to end)

- **Automated:** `.venv/bin/pytest -q` green at ≥85% coverage; `.venv/bin/mypy src` clean.
- **Manual (the one thing only the user can check):** press **Ctrl+Alt+D** → the popup opens → click "Open file..." and pick a real PDF (or paste text) → click Read → hear it, watch "piece N of M" update → click Pause → audio stops immediately → click Pause again (now "Resume") → the current chunk replays → click Stop → reading ends, controls reset. Then: start Quick Reader (Ctrl+Alt+R with something copied) while Document Reader is mid-sentence, and confirm Document Reader stops and Quick Reader takes over — and the reverse (start Document Reader while Quick Reader is talking).
- Once this manual check passes, this branch is ready for `superpowers:finishing-a-development-branch`, the same way Phase 3 was.
