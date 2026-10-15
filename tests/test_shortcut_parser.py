"""Tests for vocalis.hotkeys.shortcut_parser."""
import pytest

from vocalis.config import ShortcutsConfig
from vocalis.hotkeys.shortcut_parser import (
    CTRL_ALT_CHOICES,
    DEFAULT_SHORTCUTS,
    PUSH_TO_TALK_CHOICES,
    ShortcutError,
    ShortcutSet,
    ctrl_alt_choice,
    format_ctrl_alt_shortcut,
    parse_ctrl_alt_shortcut,
    parse_push_to_talk_key,
    parse_shortcuts,
)


def test_known_codes_match_evdev() -> None:
    assert parse_ctrl_alt_shortcut("Ctrl+Alt+R") == 19
    assert parse_ctrl_alt_shortcut("Ctrl+Alt+D") == 32
    assert parse_ctrl_alt_shortcut("Ctrl+Alt+O") == 24
    assert parse_ctrl_alt_shortcut("Ctrl+Alt+1") == 2
    assert parse_push_to_talk_key("NUMLOCK") == 69
    assert parse_push_to_talk_key("SCROLLLOCK") == 70
    assert parse_push_to_talk_key("PAUSE") == 119


def test_case_and_spaces_are_tolerated() -> None:
    assert parse_ctrl_alt_shortcut(" ctrl + alt + r ") == 19
    assert parse_push_to_talk_key("numlock") == 69


@pytest.mark.parametrize(
    "bad", ["", "Ctrl+R", "Alt+Ctrl+R", "Ctrl+Alt+", "Ctrl+Alt+Banana", "Ctrl+Alt+RR", "Ctrl+Alt+R+X", "Ctrl+Alt+-", "Shift+Alt+R"]
)
def test_bad_ctrl_alt_shortcuts_are_rejected_with_the_expected_form(bad: str) -> None:
    with pytest.raises(ShortcutError, match="Ctrl\\+Alt\\+"):
        parse_ctrl_alt_shortcut(bad)


@pytest.mark.parametrize("bad", ["", "A", "F13", "CAPSLOCK", "Ctrl+Alt+R"])
def test_bad_push_to_talk_keys_are_rejected(bad: str) -> None:
    with pytest.raises(ShortcutError, match="NUMLOCK"):
        parse_push_to_talk_key(bad)


def test_choices_round_trip() -> None:
    assert len(CTRL_ALT_CHOICES) == 36
    for choice in CTRL_ALT_CHOICES:
        text = format_ctrl_alt_shortcut(choice)
        assert ctrl_alt_choice(text) == choice
        parse_ctrl_alt_shortcut(text)
    for key in PUSH_TO_TALK_CHOICES:
        parse_push_to_talk_key(key)


def test_format_rejects_a_choice_that_is_not_offered() -> None:
    with pytest.raises(ShortcutError):
        format_ctrl_alt_shortcut("%")


def test_defaults_parse_to_the_codes_the_app_used_before() -> None:
    assert DEFAULT_SHORTCUTS == ShortcutSet(push_to_talk=69, quick_reader=19, document_reader=32, ocr_reader=24)
    assert parse_shortcuts(ShortcutsConfig()) == DEFAULT_SHORTCUTS


def test_bad_value_names_the_field() -> None:
    with pytest.raises(ShortcutError, match="ocr_reader"):
        parse_shortcuts(ShortcutsConfig(ocr_reader="Ctrl+Alt+Banana"))
    with pytest.raises(ShortcutError, match="numlock_key"):
        parse_shortcuts(ShortcutsConfig(numlock_key="F13"))


def test_two_features_may_not_share_a_shortcut() -> None:
    with pytest.raises(ShortcutError, match="quick_reader.*ocr_reader|ocr_reader.*quick_reader"):
        parse_shortcuts(ShortcutsConfig(quick_reader="Ctrl+Alt+O", ocr_reader="Ctrl+Alt+o"))


@pytest.mark.parametrize("text", ["Ctrl+Alt+ı", "Ctrl+Alt+ſ"])
def test_ctrl_alt_rejects_non_ascii_lookalike_letters(text: str) -> None:
    from vocalis.hotkeys.shortcut_parser import parse_ctrl_alt_shortcut, ShortcutError

    with pytest.raises(ShortcutError):
        parse_ctrl_alt_shortcut(text)


def test_push_to_talk_rejects_a_non_ascii_lookalike_name() -> None:
    from vocalis.hotkeys.shortcut_parser import parse_push_to_talk_key, ShortcutError

    with pytest.raises(ShortcutError):
        parse_push_to_talk_key("ſcrolllock")
