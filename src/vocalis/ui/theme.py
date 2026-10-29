"""The Vocalis look: one red-and-black palette and one Qt stylesheet.

Every window takes its colours from here, so no other module writes a
colour value of its own. Text never sits on ACCENT (white on it is only
about 4.2:1, below the 4.5:1 readability minimum); buttons with text use
BUTTON_PRIMARY instead.
"""
from string import Template

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QPushButton

BACKGROUND = "#0d0d0f"
SURFACE = "#16161a"
BORDER = "#2a2a2e"
TEXT = "#f5f5f5"
TEXT_MUTED = "#9e9e9e"
ACCENT = "#e53935"
ACCENT_BRIGHT = "#ff1744"
BUTTON_PRIMARY = "#c62828"
HIGHLIGHT_BG = "#5c1a1d"
WHITE = "#ffffff"

FLOATING_CARD_NAME = "floatingCard"
_PRIMARY_ROLE = "primary"
_STYLE_NAME = "Fusion"

_SRGB_LINEAR_THRESHOLD = 0.03928
_LUMINANCE_OFFSET = 0.05

_STYLESHEET_TEMPLATE = Template(
    """
QWidget { background-color: $BACKGROUND; color: $TEXT; }
QWidget#$FLOATING_CARD_NAME { background: transparent; }
QLabel, QCheckBox { background: transparent; }
QCheckBox { spacing: 8px; }
QToolTip { background-color: $SURFACE; color: $TEXT; border: 1px solid $BORDER; }
QPlainTextEdit, QLineEdit, QComboBox, QSpinBox {
    background-color: $SURFACE; color: $TEXT; border: 1px solid $BORDER;
    border-radius: 4px; padding: 4px;
    selection-background-color: $BUTTON_PRIMARY; selection-color: $WHITE;
}
QPlainTextEdit:focus, QLineEdit:focus, QComboBox:focus, QSpinBox:focus { border: 1px solid $ACCENT; }
QComboBox QAbstractItemView {
    background-color: $SURFACE; color: $TEXT; selection-background-color: $BUTTON_PRIMARY;
}
QPushButton {
    background-color: $SURFACE; color: $TEXT; border: 1px solid $BORDER;
    border-radius: 4px; padding: 6px 14px;
}
QPushButton:hover { border: 1px solid $ACCENT; }
QPushButton:disabled { color: $TEXT_MUTED; }
QPushButton[role="primary"] { background-color: $BUTTON_PRIMARY; color: $WHITE; border: 1px solid $BUTTON_PRIMARY; }
QPushButton[role="primary"]:hover { border: 1px solid $ACCENT_BRIGHT; }
QPushButton[role="primary"]:disabled { background-color: $BORDER; color: $TEXT_MUTED; border: 1px solid $BORDER; }
QTabWidget::pane { border: 1px solid $BORDER; border-radius: 4px; top: -1px; }
QTabBar::tab {
    background: $BACKGROUND; color: $TEXT_MUTED; padding: 8px 16px;
    border-bottom: 2px solid transparent;
}
QTabBar::tab:selected { color: $TEXT; border-bottom: 2px solid $ACCENT; }
QTabBar::tab:hover { color: $TEXT; }
QSlider::groove:horizontal { height: 4px; background: $BORDER; border-radius: 2px; }
QSlider::sub-page:horizontal { background: $ACCENT; border-radius: 2px; }
QSlider::handle:horizontal { background: $TEXT; width: 14px; margin: -5px 0; border-radius: 7px; }
QProgressBar { background-color: $BORDER; border: none; border-radius: 2px; max-height: 4px; }
QProgressBar::chunk { background-color: $ACCENT; border-radius: 2px; }
QMenu { background-color: $SURFACE; color: $TEXT; border: 1px solid $BORDER; }
QMenu::item:selected { background-color: $BUTTON_PRIMARY; color: $WHITE; }
QMenu::item:disabled { color: $TEXT_MUTED; }
QMenu::separator { height: 1px; background: $BORDER; margin: 4px 8px; }
"""
)

STYLESHEET = _STYLESHEET_TEMPLATE.substitute(
    BACKGROUND=BACKGROUND,
    SURFACE=SURFACE,
    BORDER=BORDER,
    TEXT=TEXT,
    TEXT_MUTED=TEXT_MUTED,
    ACCENT=ACCENT,
    ACCENT_BRIGHT=ACCENT_BRIGHT,
    BUTTON_PRIMARY=BUTTON_PRIMARY,
    WHITE=WHITE,
    FLOATING_CARD_NAME=FLOATING_CARD_NAME,
)


def _relative_luminance(hex_colour: str) -> float:
    """WCAG relative luminance of a "#rrggbb" colour (0 = black, 1 = white)."""
    channels = [int(hex_colour[index:index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [
        value / 12.92 if value <= _SRGB_LINEAR_THRESHOLD else ((value + 0.055) / 1.055) ** 2.4
        for value in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast_ratio(first: str, second: str) -> float:
    """WCAG contrast ratio between two "#rrggbb" colours (1.0 to 21.0)."""
    lighter, darker = sorted((_relative_luminance(first), _relative_luminance(second)), reverse=True)
    return (lighter + _LUMINANCE_OFFSET) / (darker + _LUMINANCE_OFFSET)


def mark_primary(button: QPushButton) -> None:
    """Style a button as the window's main action (dark red, white text)."""
    button.setProperty("role", _PRIMARY_ROLE)


def _dark_palette() -> QPalette:
    """A dark palette for the parts of Qt the stylesheet does not reach."""
    palette = QPalette()
    for role, colour in (
        (QPalette.ColorRole.Window, BACKGROUND),
        (QPalette.ColorRole.WindowText, TEXT),
        (QPalette.ColorRole.Base, SURFACE),
        (QPalette.ColorRole.AlternateBase, BACKGROUND),
        (QPalette.ColorRole.Text, TEXT),
        (QPalette.ColorRole.Button, SURFACE),
        (QPalette.ColorRole.ButtonText, TEXT),
        (QPalette.ColorRole.ToolTipBase, SURFACE),
        (QPalette.ColorRole.ToolTipText, TEXT),
        (QPalette.ColorRole.PlaceholderText, TEXT_MUTED),
        (QPalette.ColorRole.Highlight, BUTTON_PRIMARY),
        (QPalette.ColorRole.HighlightedText, WHITE),
        (QPalette.ColorRole.Link, ACCENT_BRIGHT),
    ):
        palette.setColor(role, QColor(colour))
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor(TEXT_MUTED))
    return palette


def apply_theme(app: QApplication) -> None:
    """Give the whole application the Vocalis red-and-black look; call once at start-up."""
    app.setStyle(_STYLE_NAME)
    app.setPalette(_dark_palette())
    app.setStyleSheet(STYLESHEET)
