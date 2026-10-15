"""Turns shortcut text from the config file ("Ctrl+Alt+R", "NUMLOCK") into
evdev key codes, and validates it.

The three Ctrl+Alt features accept exactly `Ctrl+Alt+<letter or digit>`;
the push-to-talk key accepts a short list of safe keys. The Settings
window offers these as drop-down lists so typos cannot happen there;
values typed by hand into the config file are checked here.
"""
from dataclasses import dataclass

from evdev import ecodes

from vocalis.config import ShortcutsConfig

CTRL_ALT_CHOICES: tuple[str, ...] = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
PUSH_TO_TALK_CHOICES: tuple[str, ...] = ("NUMLOCK", "SCROLLLOCK", "PAUSE")

_EXPECTED_CTRL_ALT = "Ctrl+Alt+<one letter or digit>, for example Ctrl+Alt+R"
_EXPECTED_PUSH_TO_TALK = "one of NUMLOCK, SCROLLLOCK, PAUSE"


class ShortcutError(ValueError):
    """A shortcut could not be understood or conflicts with another one.

    The message is written for the end user (shown in the Settings window
    and in the log).
    """


@dataclass(frozen=True)
class ShortcutSet:
    """The evdev key codes for every configurable shortcut."""

    push_to_talk: int
    quick_reader: int
    document_reader: int
    ocr_reader: int


def _code_for(choice: str) -> int:
    """evdev code for a single letter/digit key name."""
    return int(getattr(ecodes, f"KEY_{choice}"))


def ctrl_alt_choice(text: str) -> str:
    """Extract the letter/digit from "Ctrl+Alt+R" text (upper-cased).

    Raises:
        ShortcutError: If the text is not `Ctrl+Alt+<one letter or digit>`.
    """
    parts = [part.strip() for part in text.split("+")]
    if (
        len(parts) != 3
        or parts[0].casefold() != "ctrl"
        or parts[1].casefold() != "alt"
        or not parts[2].isascii()
        or parts[2].upper() not in CTRL_ALT_CHOICES
    ):
        raise ShortcutError(f"'{text}' is not a valid shortcut; expected {_EXPECTED_CTRL_ALT}")
    return parts[2].upper()


def parse_ctrl_alt_shortcut(text: str) -> int:
    """Return the evdev key code for a `Ctrl+Alt+<key>` shortcut.

    Raises:
        ShortcutError: If the text is not `Ctrl+Alt+<one letter or digit>`.
    """
    return _code_for(ctrl_alt_choice(text))


def format_ctrl_alt_shortcut(choice: str) -> str:
    """Return "Ctrl+Alt+<choice>" for a letter/digit offered in the Settings window.

    Raises:
        ShortcutError: If `choice` is not one of CTRL_ALT_CHOICES.
    """
    if choice not in CTRL_ALT_CHOICES:
        raise ShortcutError(f"'{choice}' cannot be used as a shortcut key; expected {_EXPECTED_CTRL_ALT}")
    return f"Ctrl+Alt+{choice}"


def parse_push_to_talk_key(text: str) -> int:
    """Return the evdev key code for the push-to-talk key.

    Raises:
        ShortcutError: If the key is not one of PUSH_TO_TALK_CHOICES.
    """
    name = text.strip()
    if not name.isascii():
        raise ShortcutError(f"'{text}' cannot be used as the push-to-talk key; expected {_EXPECTED_PUSH_TO_TALK}")
    name = name.upper()
    if name not in PUSH_TO_TALK_CHOICES:
        raise ShortcutError(f"'{text}' cannot be used as the push-to-talk key; expected {_EXPECTED_PUSH_TO_TALK}")
    return int(getattr(ecodes, f"KEY_{name}"))


def parse_shortcuts(shortcuts: ShortcutsConfig) -> ShortcutSet:
    """Validate and convert every shortcut in the config.

    Raises:
        ShortcutError: A value is invalid (the message names the config
            field) or two Ctrl+Alt features share the same shortcut.
    """
    try:
        push_to_talk = parse_push_to_talk_key(shortcuts.numlock_key)
    except ShortcutError as exc:
        raise ShortcutError(f"numlock_key: {exc}") from exc
    codes: dict[str, int] = {}
    for name in ("quick_reader", "document_reader", "ocr_reader"):
        try:
            codes[name] = parse_ctrl_alt_shortcut(getattr(shortcuts, name))
        except ShortcutError as exc:
            raise ShortcutError(f"{name}: {exc}") from exc
    seen: dict[int, str] = {}
    for name, code in codes.items():
        if code in seen:
            raise ShortcutError(f"{seen[code]} and {name} both use the same shortcut; each feature needs its own")
        seen[code] = name
    return ShortcutSet(
        push_to_talk=push_to_talk,
        quick_reader=codes["quick_reader"],
        document_reader=codes["document_reader"],
        ocr_reader=codes["ocr_reader"],
    )


DEFAULT_SHORTCUTS = parse_shortcuts(ShortcutsConfig())
