"""Splits normalized text into sentence-bounded chunks for streaming
synthesis and playback, so a reading can start speaking before the whole
document has been processed and can show "piece N of M" progress.

Sentence breaks are found by punctuation (. ! ? followed by whitespace) and
by line breaks, but never after a known abbreviation ("Dr.", "e.g."), after
an initial ("J. Smith", "U.S."), or before a lowercase word ("etc. and").
A blank line is always a break. A single line break is a break unless the
line before it looks hard-wrapped (long, with no punctuation at its end).

A sentence is never split mid-word: a single sentence longer than
max_chars becomes its own (over-limit) chunk rather than being cut off,
since a TTS engine reading a word fragment sounds far worse than reading
one long chunk (the voice engines split very long text themselves).

Every chunk also carries the exact (start, end) character positions of the
text it came from, which a later stage uses to highlight the line being read.
"""
import re
from dataclasses import dataclass

DEFAULT_MAX_CHUNK_CHARS = 300
_HARD_WRAP_MIN_LINE_CHARS = 70

_BOUNDARY_RE = re.compile(r"[.!?][\"'”’)\]]*(?=\s)|\n")
_WORD_BEFORE_RE = re.compile(r"([A-Za-z][A-Za-z.]*)$")
_INITIALS_RE = re.compile(r"(?:[A-Za-z]\.)*[A-Za-z]")
_NEXT_CHAR_RE = re.compile(r"\s*(\S)")
_PARAGRAPH_AFTER_RE = re.compile(r"[ \t\r]*(?:\n|\Z)")
# Bounded look-behind sizes keep the chunker linear on very large input.
# No abbreviation or initial is longer than a few characters.
_PRECEDING_WORD_WINDOW_CHARS = 40
_LINE_LOOKBACK_CHARS = 200
_ABBREVIATIONS = frozenset(
    {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e",
        "inc", "ltd", "co", "no", "fig", "approx", "dept", "est", "mt",
    }
)


@dataclass(frozen=True)
class Chunk:
    """One piece of text to speak, plus where it came from in the input."""

    text: str
    start: int
    end: int


def _period_breaks(text: str, mark_index: int, match_end: int) -> bool:
    """Whether the sentence-ending mark at `mark_index` really ends a sentence."""
    if text[mark_index] != ".":
        return True
    window_start = max(0, mark_index - _PRECEDING_WORD_WINDOW_CHARS)
    word = _WORD_BEFORE_RE.search(text[window_start:mark_index])
    if word is not None:
        token = word.group(1)
        if token.lower() in _ABBREVIATIONS or _INITIALS_RE.fullmatch(token):
            return False
    next_char = _NEXT_CHAR_RE.match(text, match_end)
    following = next_char.group(1) if next_char else ""
    return not following.islower()


def _newline_breaks(text: str, index: int) -> bool:
    """Whether the line break at `index` ends a sentence (see module docstring)."""
    if _PARAGRAPH_AFTER_RE.match(text, index + 1):
        return True
    lookback_start = max(0, index - _LINE_LOOKBACK_CHARS)
    line_start = max(text.rfind("\n", lookback_start, index) + 1, lookback_start)
    line = text[line_start:index].strip()
    if not line or line[-1] in ".!?":
        return True
    if line[-1] in ",;-":
        return False
    return len(line) < _HARD_WRAP_MIN_LINE_CHARS


def _sentence_ends(text: str) -> list[int]:
    """Sorted, de-duplicated offsets where a sentence ends."""
    ends: set[int] = set()
    for match in _BOUNDARY_RE.finditer(text):
        if match.group() == "\n":
            if _newline_breaks(text, match.start()):
                ends.add(match.start())
        elif _period_breaks(text, match.start(), match.end()):
            ends.add(match.end())
    return sorted(ends)


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) of every non-empty sentence, whitespace trimmed."""
    spans: list[tuple[int, int]] = []
    begin = 0
    for end in [*_sentence_ends(text), len(text)]:
        piece = text[begin:end]
        stripped = piece.strip()
        if stripped:
            lead = len(piece) - len(piece.lstrip())
            spans.append((begin + lead, begin + lead + len(stripped)))
        begin = end
    return spans


def _make_chunk(text: str, start: int, end: int) -> Chunk:
    """Build a Chunk whose text is the slice's words joined by single spaces."""
    return Chunk(" ".join(text[start:end].split()), start, end)


def chunk_spans(text: str, max_chars: int = DEFAULT_MAX_CHUNK_CHARS) -> list[Chunk]:
    """Split text into sentence-bounded chunks no longer than max_chars.

    Args:
        text: Normalized, speakable text (see normalizer.normalize_markdown).
        max_chars: Soft maximum characters per chunk. A single sentence
            longer than this is still returned whole, as its own chunk.

    Returns:
        Chunks in reading order, each with the exact (start, end) positions
        of the input it covers. Empty or whitespace-only input gives [].
    """
    chunks: list[Chunk] = []
    current_start = current_end = -1
    current_len = 0
    for start, end in _sentence_spans(text):
        sentence_len = len(" ".join(text[start:end].split()))
        if current_start < 0:
            current_start, current_end, current_len = start, end, sentence_len
        elif current_len + 1 + sentence_len > max_chars:
            chunks.append(_make_chunk(text, current_start, current_end))
            current_start, current_end, current_len = start, end, sentence_len
        else:
            current_end = end
            current_len += 1 + sentence_len
    if current_start >= 0:
        chunks.append(_make_chunk(text, current_start, current_end))
    return chunks


def chunk_text(text: str, max_chars: int = DEFAULT_MAX_CHUNK_CHARS) -> list[str]:
    """Split text into sentence-bounded chunk strings (see chunk_spans)."""
    return [chunk.text for chunk in chunk_spans(text, max_chars)]
