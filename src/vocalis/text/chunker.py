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


def chunk_text(text: str, max_chars: int = DEFAULT_MAX_CHUNK_CHARS) -> list[str]:
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

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}" if current else sentence
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    # `sentences` is non-empty and every sentence is non-empty, so `current`
    # is always non-empty here.
    chunks.append(current)
    return chunks
