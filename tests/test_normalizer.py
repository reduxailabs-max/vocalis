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
