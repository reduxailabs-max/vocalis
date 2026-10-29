"""Tests for vocalis.ui.theme: palette readability and applying the theme."""
from collections.abc import Iterator

import pytest
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QApplication, QPushButton

from vocalis.ui import theme

_AA_NORMAL_TEXT = 4.5


@pytest.fixture
def restore_app_look(qapp: QApplication) -> Iterator[QApplication]:
    """Put the shared test QApplication's look back after a test changes it."""
    style_name = qapp.style().name()
    palette = QPalette(qapp.palette())
    sheet = qapp.styleSheet()
    yield qapp
    qapp.setStyleSheet(sheet)
    qapp.setPalette(palette)
    qapp.setStyle(style_name)


def test_contrast_ratio_matches_known_values() -> None:
    assert theme.contrast_ratio("#ffffff", "#000000") == pytest.approx(21.0)
    assert theme.contrast_ratio("#000000", "#ffffff") == pytest.approx(21.0)
    assert theme.contrast_ratio("#e53935", "#e53935") == pytest.approx(1.0)


@pytest.mark.parametrize(
    "foreground,background",
    [
        (theme.TEXT, theme.BACKGROUND),
        (theme.TEXT, theme.SURFACE),
        (theme.TEXT_MUTED, theme.BACKGROUND),
        (theme.TEXT_MUTED, theme.SURFACE),
        (theme.TEXT_MUTED, theme.BORDER),
        (theme.WHITE, theme.BUTTON_PRIMARY),
        (theme.TEXT, theme.HIGHLIGHT_BG),
    ],
)
def test_every_text_pair_is_readable(foreground: str, background: str) -> None:
    """Each text/background pair the stylesheet uses passes WCAG AA for normal text."""
    assert theme.contrast_ratio(foreground, background) >= _AA_NORMAL_TEXT


def test_white_on_the_accent_red_is_too_weak_for_text() -> None:
    """Documents why buttons use BUTTON_PRIMARY, not ACCENT, behind text."""
    assert theme.contrast_ratio(theme.WHITE, theme.ACCENT) < _AA_NORMAL_TEXT


def test_stylesheet_has_every_placeholder_filled_and_uses_the_palette() -> None:
    assert "$" not in theme.STYLESHEET
    for colour in (theme.BACKGROUND, theme.SURFACE, theme.ACCENT, theme.BUTTON_PRIMARY):
        assert colour in theme.STYLESHEET
    assert f"#{theme.FLOATING_CARD_NAME}" in theme.STYLESHEET


def test_mark_primary_sets_the_role_property(qapp: QApplication) -> None:
    button = QPushButton("Save")
    theme.mark_primary(button)
    assert button.property("role") == "primary"


def test_apply_theme_sets_fusion_dark_palette_and_stylesheet(restore_app_look: QApplication) -> None:
    app = restore_app_look
    theme.apply_theme(app)
    assert app.palette().color(QPalette.ColorRole.Window).name() == theme.BACKGROUND
    assert app.palette().color(QPalette.ColorRole.WindowText).name() == theme.TEXT
    assert app.palette().color(QPalette.ColorRole.Highlight).name() == theme.BUTTON_PRIMARY
    assert app.styleSheet() == theme.STYLESHEET
    # While a stylesheet is set, Qt wraps the real style in a nameless proxy,
    # so the base style's name is only readable once the sheet is removed.
    app.setStyleSheet("")
    assert app.style().name().lower() == "fusion"
