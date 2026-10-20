"""Tests for vocalis.text.chunker.chunk_text."""
import time

from vocalis.text.chunker import Chunk, chunk_spans, chunk_text


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


def test_a_known_abbreviation_does_not_end_a_sentence() -> None:
    chunks = chunk_text("Dr. Smith arrived. Mr. Jones left.", max_chars=25)
    assert chunks == ["Dr. Smith arrived.", "Mr. Jones left."]


def test_e_g_does_not_end_a_sentence() -> None:
    chunks = chunk_text("Use tools, e.g. a hammer. Then rest.", max_chars=20)
    assert chunks == ["Use tools, e.g. a hammer.", "Then rest."]


def test_an_initial_does_not_end_a_sentence() -> None:
    chunks = chunk_text("Ask J. Smith about it. Then go.", max_chars=24)
    assert chunks == ["Ask J. Smith about it.", "Then go."]


def test_a_lowercase_word_after_a_full_stop_keeps_the_sentence_going() -> None:
    chunks = chunk_text("We sell pens, etc. and paper. Bye.", max_chars=30)
    assert chunks == ["We sell pens, etc. and paper.", "Bye."]


def test_a_decimal_number_is_not_a_sentence_end() -> None:
    assert chunk_text("Pi is 3.14 today. Done.", max_chars=300) == ["Pi is 3.14 today. Done."]


def test_a_short_line_without_punctuation_is_its_own_sentence() -> None:
    text = "Chapter One\nIt was dark. It rained.\nThe end"
    assert chunk_text(text, max_chars=20) == ["Chapter One", "It was dark.", "It rained. The end"]


def test_a_hard_wrapped_long_line_is_joined_to_the_next_line() -> None:
    long_line = " ".join(["word"] * 15)  # 74 characters, no punctuation
    chunks = chunk_text(f"{long_line}\nmore words here.", max_chars=80)
    assert len(chunks) == 1
    assert "\n" not in chunks[0]


def test_a_blank_line_always_breaks_even_after_a_long_line() -> None:
    long_line = " ".join(["word"] * 16)  # 79 characters
    chunks = chunk_text(f"{long_line}\n\nNext.", max_chars=82)
    assert len(chunks) == 2
    assert chunks[1] == "Next."


def test_a_line_ending_in_a_comma_is_not_a_break() -> None:
    chunks = chunk_text("First part,\nsecond part.", max_chars=20)
    assert chunks == ["First part, second part."]


def test_one_enormous_unpunctuated_line_stays_one_chunk() -> None:
    text = " ".join(["word"] * 2000)
    chunks = chunk_text(text, max_chars=300)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_spans_point_at_the_exact_source_text() -> None:
    text = "  Hello there.  General Kenobi.\n"
    chunks = chunk_spans(text, max_chars=15)
    assert [text[c.start : c.end] for c in chunks] == ["Hello there.", "General Kenobi."]
    assert all(isinstance(c, Chunk) for c in chunks)


def test_a_chunk_that_joins_two_sentences_spans_both() -> None:
    text = "One. Two.\nThree."
    chunks = chunk_spans(text, max_chars=300)
    assert len(chunks) == 1
    assert (chunks[0].start, chunks[0].end) == (0, len(text))
    assert chunks[0].text == "One. Two. Three."


def test_chunk_text_matches_chunk_spans_text() -> None:
    text = "A first sentence. A second one! And a third?"
    assert chunk_text(text, max_chars=20) == [c.text for c in chunk_spans(text, max_chars=20)]


def test_many_short_lines_are_chunked_in_linear_time() -> None:
    text = "word word word.\n" * 20000  # about 320 KB
    started = time.perf_counter()
    chunk_spans(text, max_chars=300)
    assert time.perf_counter() - started < 2.0


def test_many_abbreviation_like_full_stops_are_chunked_in_linear_time() -> None:
    text = "a. " * 20000  # about 60 KB
    started = time.perf_counter()
    chunk_spans(text, max_chars=300)
    assert time.perf_counter() - started < 2.0


def test_windows_line_endings_make_a_blank_line_break() -> None:
    long_line = " ".join(["word"] * 16)  # 79 characters
    chunks = chunk_text(f"{long_line}\r\n\r\nNext.", max_chars=82)
    assert len(chunks) == 2
    assert chunks[1] == "Next."
