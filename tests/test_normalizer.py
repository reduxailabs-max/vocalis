"""Tests for vocalis.text.normalizer.normalize_markdown."""
from vocalis.text.normalizer import normalize_markdown, normalize_text


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


def test_keeps_inline_code_content_containing_underscores_intact() -> None:
    assert normalize_markdown("Set `text_input` and then `text_output`") == (
        "Set text_input and then text_output"
    )


def test_keeps_dunder_method_names_in_inline_code_intact() -> None:
    assert normalize_markdown("See `__init__.py` for details") == (
        "See __init__.py for details"
    )


def test_does_not_treat_spaced_asterisks_as_emphasis() -> None:
    assert normalize_markdown("2 * 3 * 4 equals 24") == "2 * 3 * 4 equals 24"


def test_nul_characters_in_the_input_never_collide_with_code_placeholders() -> None:
    result = normalize_markdown("Use `real` here \x00CODE0\x00 and `other`.")

    assert result == "Use real here CODE0 and other."


def test_normalize_text_speaks_urls() -> None:
    assert normalize_text("Visit https://example.com/docs for info") == "Visit example dot com slash docs for info"


def test_normalize_text_speaks_www_urls() -> None:
    assert normalize_text("See www.example.com") == "See www dot example dot com"


def test_normalize_text_strips_a_trailing_slash() -> None:
    assert normalize_text("Go to https://example.com/") == "Go to example dot com"


def test_normalize_text_speaks_currency() -> None:
    assert normalize_text("It costs $42") == "It costs 42 dollars"


def test_normalize_text_speaks_percent() -> None:
    assert normalize_text("Up 50% today") == "Up 50 percent today"


def test_normalize_text_expands_ampersand_and_at() -> None:
    assert normalize_text("Rock & Roll, email me @home") == "Rock and Roll, email me at home"


def test_normalize_text_leaves_plain_numbers_alone() -> None:
    assert normalize_text("I have 42 apples and 3.14 pie") == "I have 42 apples and 3.14 pie"


def test_normalize_text_handles_empty_string() -> None:
    assert normalize_text("") == ""


def test_normalize_markdown_also_normalizes_urls_and_symbols() -> None:
    assert normalize_markdown("Cost is $5 & visit https://example.com") == "Cost is 5 dollars and visit example dot com"
