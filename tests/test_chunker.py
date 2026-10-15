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
