# Vocalis Stage 7 — Red-and-Black UI and Reading Highlight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One red-and-black look for every Vocalis window, a tabbed Settings window that exposes every setting, a draggable card-style floating widget, and highlighting of the piece being read in the Document Reader window.

**Architecture:** A new `ui/theme.py` owns every colour and one Qt stylesheet, applied once in `cli.main()`. Each window keeps its existing widget attribute names (tests assert them) and gains layout/visual changes only. Highlighting adds two optional callbacks to `DocumentReaderFeature` (`on_document`, `on_chunk_span`) which `cli.build_doc_reader()` wires to two new popup signals, exactly like the existing `on_progress` → `progress_changed` path.

**Tech Stack:** Python 3, PySide6 6.11 (Qt Widgets), pytest + pytest-qt (offscreen), mypy strict.

**Spec:** `docs/superpowers/specs/2026-09-11-vocalis-design.md`, section "Stage 7 amendment — Red-and-black UI and reading highlight" (commits `9c7291e`, `bea1cf4`).

## Global Constraints

- No new dependencies. PySide6 only for UI.
- Every colour value used for styling lives in `src/vocalis/ui/theme.py`; other source modules import the names, never write `#xxxxxx` literals (tests may).
- Palette (exact): `BACKGROUND #0d0d0f`, `SURFACE #16161a`, `BORDER #2a2a2e`, `TEXT #f5f5f5`, `TEXT_MUTED #9e9e9e`, `ACCENT #e53935`, `ACCENT_BRIGHT #ff1744`, `BUTTON_PRIMARY #c62828`, `HIGHLIGHT_BG #5c1a1d`, plus `WHITE #ffffff`.
- Text never sits on `ACCENT` (4.23:1 fails WCAG AA 4.5:1); text buttons use `BUTTON_PRIMARY`.
- Keep every existing widget attribute name, window title, menu text, tooltip text and label string that current tests assert.
- Every new function/method has a one-line purpose docstring (project rule).
- All tests headless (`QT_QPA_PLATFORM=offscreen`, set in `tests/conftest.py`); never touch real audio, GPU, network or real model files.
- Repo is on NTFS: `sed -i` fails — edit with the Edit/Write tools or python3.
- Gates (checked once at the end, not per task): `PYTHONPATH=$PWD/src .venv/bin/python -m pytest -q` (≥85% coverage, zero warnings), `.venv/bin/python -m mypy` (whole project, strict config in `pyproject.toml`).
- Run only the task's own test files during a task; the full suite runs at the end. Never run tests while another agent may be testing in the same worktree.
- Commits: conventional commit messages, no `--no-verify`.

## Review Focus

1. A document containing emoji (characters outside the Basic Multilingual Plane): the highlight must land on the right words, not drift — Qt counts UTF-16 units. Test in Task 7.
2. A file with Windows (`\r\n`) or old-Mac (`\r`) line endings: Qt shows them as single line breaks, so the highlight would drift by one per line unless the reader unifies line endings first. Test in Task 6.
3. A user empties a model-location field and presses Save: refused with a plain message naming the field, nothing written. Test in Task 5.
4. A user drags the floating widget, the reading ends, a new one starts: the widget reappears where they put it, not back in the corner. Test in Task 3.
5. A late highlight signal for a span past the end of the shown text (or an empty span): no crash; clamped, or highlight cleared. Test in Task 7.

---

### Task 1: Theme module and app-wide styling

**Files:**
- Create: `src/vocalis/ui/theme.py`
- Modify: `src/vocalis/cli.py` (import + one call in `main()` right after `app.setQuitOnLastWindowClosed(False)`)
- Test: `tests/test_theme.py` (new), `tests/test_cli.py` (one new test)

**Interfaces:**
- Produces (used by Tasks 2, 3, 5, 7): constants `BACKGROUND, SURFACE, BORDER, TEXT, TEXT_MUTED, ACCENT, ACCENT_BRIGHT, BUTTON_PRIMARY, HIGHLIGHT_BG, WHITE: str`; `FLOATING_CARD_NAME: str = "floatingCard"`; `STYLESHEET: str`; `contrast_ratio(first: str, second: str) -> float`; `mark_primary(button: QPushButton) -> None`; `apply_theme(app: QApplication) -> None`.

- [ ] **Step 1: Write the failing tests** — create `tests/test_theme.py`:

```python
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
    assert app.style().name().lower() == "fusion"
    assert app.palette().color(QPalette.ColorRole.Window).name() == theme.BACKGROUND
    assert app.palette().color(QPalette.ColorRole.WindowText).name() == theme.TEXT
    assert app.palette().color(QPalette.ColorRole.Highlight).name() == theme.BUTTON_PRIMARY
    assert app.styleSheet() == theme.STYLESHEET
```

Add to `tests/test_cli.py` (next to `test_main_starts_all_four_listeners_then_stops_them_in_reverse_order`, same imports already present there):

```python
def test_main_applies_the_theme_once_to_the_app(qapp) -> None:  # type: ignore[no-untyped-def]
    calls = MagicMock()
    with (
        patch("vocalis.cli.load_config", return_value=VocalisConfig()),
        patch("vocalis.cli.VocalisDaemon", return_value=calls.daemon),
        patch("vocalis.cli.build_dictation", return_value=(MagicMock(), calls.dictation)),
        patch("vocalis.cli.build_quick_reader", return_value=(MagicMock(), calls.quick)),
        patch("vocalis.cli.build_doc_reader", return_value=(MagicMock(), MagicMock(), calls.doc)),
        patch("vocalis.cli.build_ocr_reader", return_value=(MagicMock(), calls.ocr)),
        patch("vocalis.cli.apply_theme") as mock_theme,
        patch("vocalis.cli.QApplication.exec", side_effect=KeyboardInterrupt),
    ):
        main()

    mock_theme.assert_called_once_with(qapp)
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_theme.py tests/test_cli.py::test_main_applies_the_theme_once_to_the_app -q --no-cov`
Expected: FAIL — `ImportError: cannot import name 'theme'` / `AttributeError: ... has no attribute 'apply_theme'`.

- [ ] **Step 3: Implement** — create `src/vocalis/ui/theme.py`:

```python
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
```

In `src/vocalis/cli.py` add `from vocalis.ui.theme import apply_theme` with the other `vocalis.ui` imports, and in `main()`:

```python
    app = cast(QApplication, QApplication.instance() or QApplication(sys.argv))
    app.setQuitOnLastWindowClosed(False)
    apply_theme(app)
```

- [ ] **Step 4: Run to verify they pass**

Run: `PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_theme.py tests/test_cli.py -q --no-cov`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/vocalis/ui/theme.py src/vocalis/cli.py tests/test_theme.py tests/test_cli.py
git commit -m "feat: add the red-and-black theme and apply it at start-up"
```

---

### Task 2: Tray icon states by shape and colour

**Files:**
- Modify: `src/vocalis/ui/tray.py:1-28` (icon drawing), `:127-138` (`_refresh`)
- Test: `tests/test_tray.py::test_listening_changes_the_text_and_the_icon_colour` (rewrite) + one new test

**Interfaces:**
- Consumes: `theme.TEXT_MUTED`, `theme.ACCENT_BRIGHT`, `theme.ACCENT`, `theme.WHITE` (Task 1).
- Produces: nothing new for other tasks. `COLOURS` is removed (only `tray.py` uses it — verified by grep).

- [ ] **Step 1: Write the failing tests** — in `tests/test_tray.py` replace `test_listening_changes_the_text_and_the_icon_colour` with:

```python
def _pixel(tray: TrayIcon, x: int, y: int) -> str:
    """The colour of one pixel of the tray icon as "#rrggbb"."""
    return tray._tray.icon().pixmap(22, 22).toImage().pixelColor(x, y).name()  # noqa: SLF001


def test_listening_changes_the_text_and_the_icon_colour(qtbot) -> None:  # type: ignore[no-untyped-def]
    """Grey dot idle, bright red dot listening, red ring with a white centre reading."""
    hub, tray, _calls = make(qtbot)
    assert _pixel(tray, 11, 11) == "#9e9e9e"
    hub.set_listening(True)
    assert tray._tray.toolTip() == "Vocalis — listening"  # noqa: SLF001
    assert _pixel(tray, 11, 11) == "#ff1744"
    hub.set_listening(False)
    hub.begin_reading("Document", ReadingControls(stop=lambda: None))
    assert _pixel(tray, 11, 11) == "#ffffff"  # white centre dot
    assert _pixel(tray, 11, 5) == "#e53935"  # on the red ring


def test_the_reading_icon_has_a_hollow_gap_between_ring_and_dot(qtbot) -> None:  # type: ignore[no-untyped-def]
    """Shape, not just colour, tells reading apart from listening."""
    hub, tray, _calls = make(qtbot)
    hub.begin_reading("Document", ReadingControls(stop=lambda: None))
    gap = tray._tray.icon().pixmap(22, 22).toImage().pixelColor(11, 7)  # noqa: SLF001
    assert gap.alpha() == 0
```

Note: check `make()` / existing tests for whether `hub.set_listening(False)` before `begin_reading` is needed; if the original test called `begin_reading` while listening and it worked, keep the `set_listening(False)` anyway — it is harmless.

- [ ] **Step 2: Run to verify they fail**

Run: `PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_tray.py -q --no-cov`
Expected: FAIL — listening pixel is `#e53935`, reading pixel is `#1e88e5`.

- [ ] **Step 3: Implement** — in `src/vocalis/ui/tray.py` replace the imports block, `COLOURS` and `_make_icon` with:

```python
from PySide6.QtCore import QObject, QRectF, Qt
from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QMenu, QSystemTrayIcon

from vocalis.ui import theme
from vocalis.ui.status_hub import IDLE, LISTENING, READING, StatusHub

logger = logging.getLogger("vocalis")

ICON_SIZE = 22
NO_TRAY_WARNING = "No system tray available; continuing without the tray icon"
_DOT = QRectF(3, 3, 16, 16)
_RING = QRectF(5.5, 5.5, 11, 11)
_RING_WIDTH = 3.0
_CENTRE_DOT = QRectF(8.5, 8.5, 5, 5)


def _make_icon(state: str) -> QIcon:
    """Draw the 22x22 icon for a state: grey dot idle, bright red dot
    listening, red ring with a white centre while reading."""
    pixmap = QPixmap(ICON_SIZE, ICON_SIZE)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    if state == READING:
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(theme.ACCENT), _RING_WIDTH))
        painter.drawEllipse(_RING)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(theme.WHITE))
        painter.drawEllipse(_CENTRE_DOT)
    else:
        colour = theme.ACCENT_BRIGHT if state == LISTENING else theme.TEXT_MUTED
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(colour))
        painter.drawEllipse(_DOT)
    painter.end()
    return QIcon(pixmap)
```

and in `_refresh` replace the last line with `self._tray.setIcon(_make_icon(state))`. Keep the `IDLE` import if still used; if mypy/ruff flags it unused, remove it.

- [ ] **Step 4: Run to verify they pass**

Run: `PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_tray.py -q --no-cov`
Expected: all PASS. If the ring pixel is a blended shade instead of exactly `#e53935`, do NOT loosen the test to "reddish": move the sampled pixel to the middle of the 3 px stroke (the ring's top stroke spans y = 4.0 to 7.0 at x = 11).

- [ ] **Step 5: Commit**

```bash
git add src/vocalis/ui/tray.py tests/test_tray.py
git commit -m "feat: tray icon states differ by shape and use the red-and-black palette"
```

---

### Task 3: Floating widget as a draggable card with a progress bar

**Files:**
- Modify: `src/vocalis/ui/floating_widget.py` (whole file)
- Test: `tests/test_floating_widget.py` (new tests appended; existing ones must stay green unchanged)

**Interfaces:**
- Consumes: `theme.SURFACE`, `theme.BORDER`, `theme.TEXT_MUTED`, `theme.FLOATING_CARD_NAME` (Task 1).
- Produces: `FloatingWidget._progress_bar: QProgressBar` (tests), unchanged constructor `FloatingWidget(hub, enabled)`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_floating_widget.py` (add `from PySide6.QtCore import QEvent, QPoint, QPointF` and `from PySide6.QtGui import QMouseEvent` to the imports):

```python
def _mouse(kind: QEvent.Type, global_x: int, global_y: int, widget: FloatingWidget, held: bool) -> QMouseEvent:
    """A left-button mouse event at a global screen position."""
    local = QPointF(widget.mapFromGlobal(QPoint(global_x, global_y)))
    buttons = Qt.MouseButton.LeftButton if held else Qt.MouseButton.NoButton
    return QMouseEvent(
        kind, local, QPointF(global_x, global_y), Qt.MouseButton.LeftButton, buttons,
        Qt.KeyboardModifier.NoModifier,
    )


def test_is_a_translucent_card_window(qtbot) -> None:  # type: ignore[no-untyped-def]
    _hub, widget = make(qtbot)
    assert widget.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    assert widget.objectName() == "floatingCard"


def test_progress_bar_follows_the_reading(qtbot) -> None:  # type: ignore[no-untyped-def]
    hub, widget = make(qtbot)
    hub.begin_reading("Document", ReadingControls(stop=lambda: None))
    assert widget._progress_bar.value() == 0  # noqa: SLF001
    hub.report_progress(2, 5)
    assert widget._progress_bar.maximum() == 5  # noqa: SLF001
    assert widget._progress_bar.value() == 2  # noqa: SLF001
    assert widget._progress_label.text() == "piece 2 of 5"  # noqa: SLF001
    assert not widget._progress_bar.isTextVisible()  # noqa: SLF001


def test_a_new_reading_resets_the_progress_bar(qtbot) -> None:  # type: ignore[no-untyped-def]
    hub, widget = make(qtbot)
    hub.begin_reading("Document", ReadingControls(stop=lambda: None))
    hub.report_progress(3, 3)
    hub.end_reading()
    hub.begin_reading("Document", ReadingControls(stop=lambda: None))
    assert widget._progress_bar.value() == 0  # noqa: SLF001


def test_dragging_moves_the_widget_and_the_spot_is_kept_for_the_next_reading(qtbot) -> None:  # type: ignore[no-untyped-def]
    hub, widget = make(qtbot)
    hub.begin_reading("Document", ReadingControls(stop=lambda: None))
    widget.move(100, 100)
    widget.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress, 110, 110, widget, held=True))
    widget.mouseMoveEvent(_mouse(QEvent.Type.MouseMove, 310, 210, widget, held=True))
    widget.mouseReleaseEvent(_mouse(QEvent.Type.MouseButtonRelease, 310, 210, widget, held=False))
    assert widget.pos() == QPoint(300, 200)

    hub.end_reading()
    hub.begin_reading("Document", ReadingControls(stop=lambda: None))
    assert widget.pos() == QPoint(300, 200)


def test_moving_the_mouse_without_pressing_does_not_drag(qtbot) -> None:  # type: ignore[no-untyped-def]
    hub, widget = make(qtbot)
    hub.begin_reading("Document", ReadingControls(stop=lambda: None))
    widget.move(100, 100)
    widget.mouseMoveEvent(_mouse(QEvent.Type.MouseMove, 310, 210, widget, held=False))
    assert widget.pos() == QPoint(100, 100)


def test_paints_without_errors(qtbot) -> None:  # type: ignore[no-untyped-def]
    hub, widget = make(qtbot)
    hub.begin_reading("Document", ReadingControls(stop=lambda: None))
    widget.grab()  # runs paintEvent offscreen
```

Before writing these, read the existing tests to confirm the hub method names (`begin_reading`, `report_progress`, `end_reading`); they are used in `src/vocalis/cli.py`.

- [ ] **Step 2: Run to verify they fail**

Run: `PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_floating_widget.py -q --no-cov`
Expected: new tests FAIL (`AttributeError: _progress_bar`, translucent attribute false, drag does nothing).

- [ ] **Step 3: Implement** — replace `src/vocalis/ui/floating_widget.py` with:

```python
"""The floating mini-widget: a small draggable card shown only while
something is read aloud."""
from typing import Callable

from PySide6.QtCore import QPoint, QRectF, Qt
from PySide6.QtGui import QColor, QGuiApplication, QMouseEvent, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout, QWidget

from vocalis.ui import theme
from vocalis.ui.status_hub import StatusHub

_MARGIN = 16
_CORNER_RADIUS = 8.0
_PADDING_H = 14
_PADDING_V = 12


class FloatingWidget(QWidget):
    """Small always-on-top card shown only while something is read aloud.

    It can be dragged with the mouse. A dragged position is kept until
    Vocalis quits; until the first drag each reading opens bottom-right.
    """

    def __init__(self, hub: StatusHub, enabled: Callable[[], bool]) -> None:
        """Build the widget (hidden) and connect it to the status hub."""
        super().__init__(
            None,
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowDoesNotAcceptFocus,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setObjectName(theme.FLOATING_CARD_NAME)
        self._hub = hub
        self._enabled = enabled
        self._drag_offset: QPoint | None = None
        self._user_position: QPoint | None = None
        self._source_label = QLabel("")
        source_font = self._source_label.font()
        source_font.setBold(True)
        self._source_label.setFont(source_font)
        self._status_label = QLabel("")
        self._status_label.setStyleSheet(f"color: {theme.TEXT_MUTED};")
        self._progress_bar = QProgressBar()
        self._progress_bar.setTextVisible(False)
        self._reset_progress_bar()
        self._progress_label = QLabel("")
        self._pause_button = QPushButton("Pause")
        self._stop_button = QPushButton("Stop")
        for button in (self._pause_button, self._stop_button):
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        row = QHBoxLayout()
        row.addWidget(self._pause_button)
        row.addWidget(self._stop_button)
        layout = QVBoxLayout()
        layout.setContentsMargins(_PADDING_H, _PADDING_V, _PADDING_H, _PADDING_V)
        layout.addWidget(self._source_label)
        layout.addWidget(self._status_label)
        layout.addWidget(self._progress_bar)
        layout.addWidget(self._progress_label)
        layout.addLayout(row)
        self.setLayout(layout)
        hub.reading_started.connect(self._on_reading_started)
        hub.reading_finished.connect(self._on_reading_finished)
        hub.status_changed.connect(self._status_label.setText)
        hub.progress_changed.connect(self._on_progress)
        hub.paused_changed.connect(self._on_paused_changed)
        self._pause_button.clicked.connect(hub.toggle_pause)
        self._stop_button.clicked.connect(hub.stop)

    def paintEvent(self, _event: QPaintEvent) -> None:
        """Draw the rounded dark card behind the labels and buttons."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor(theme.BORDER), 1.0))
        painter.setBrush(QColor(theme.SURFACE))
        painter.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), _CORNER_RADIUS, _CORNER_RADIUS)
        painter.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Start dragging when the left button goes down on the card."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.pos()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Follow the mouse while the left button is held after a press."""
        if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            self._user_position = self.pos()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Stop dragging."""
        self._drag_offset = None
        super().mouseReleaseEvent(event)

    def _reset_progress_bar(self) -> None:
        """Show an empty bar (nothing read yet)."""
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(0)

    def _on_reading_started(self, source: str, can_pause: bool) -> None:
        """Show the new reading's name and, if switched on, the widget."""
        self._source_label.setText(source)
        self._status_label.setText("")
        self._progress_label.setText("")
        self._reset_progress_bar()
        self._pause_button.setText("Pause")
        self._pause_button.setVisible(can_pause)
        if self._enabled():
            self.adjustSize()
            if self._user_position is None:
                self._place_bottom_right()
            else:
                self.move(self._user_position)
            self.show()

    def _on_reading_finished(self) -> None:
        """Hide the widget and clear its labels."""
        self.hide()
        self._drag_offset = None
        self._source_label.setText("")
        self._status_label.setText("")
        self._progress_label.setText("")

    def _on_progress(self, current: int, total: int) -> None:
        """Show which piece of the text is being read, as text and as a bar."""
        self._progress_label.setText(f"piece {current} of {total}")
        self._progress_bar.setRange(0, max(total, 1))
        self._progress_bar.setValue(min(max(current, 0), max(total, 1)))

    def _on_paused_changed(self, paused: bool) -> None:
        """Make the button say what it will do next."""
        self._pause_button.setText("Resume" if paused else "Pause")

    def apply_enabled(self) -> None:
        """Hide the widget right away if it is visible but now switched off."""
        if self.isVisible() and not self._enabled():
            self.hide()

    def _place_bottom_right(self) -> None:
        """Move the widget to the bottom-right corner of the primary screen."""
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        area = screen.availableGeometry()
        self.move(
            area.right() + 1 - self.width() - _MARGIN,
            area.bottom() + 1 - self.height() - _MARGIN,
        )
```

- [ ] **Step 4: Run to verify they pass**

Run: `PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_floating_widget.py -q --no-cov`
Expected: all PASS, including the 8 pre-existing tests.

- [ ] **Step 5: Commit**

```bash
git add src/vocalis/ui/floating_widget.py tests/test_floating_widget.py
git commit -m "feat: floating widget becomes a draggable card with a progress bar"
```

---

### Task 4: `LiveSettings.engines_need_restart()`

**Files:**
- Modify: `src/vocalis/settings.py` (`__init__` + new method + new constant)
- Test: `tests/test_settings.py` (append)

**Interfaces:**
- Produces (used by Task 5): `MODEL_PATH_FIELDS: tuple[str, ...]` in `vocalis.settings` = `("parakeet_model_dir", "qwen3_tts_model_dir", "piper_binary_path", "piper_voice_model_path", "kokoro_model_path", "kokoro_voices_path")`; `LiveSettings.engines_need_restart(self) -> bool`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_settings.py` (reuse its existing imports; add `from vocalis.settings import MODEL_PATH_FIELDS` if not covered):

```python
def _settings_with_recorded_saves(tmp_path) -> LiveSettings:  # type: ignore[no-untyped-def]
    """LiveSettings over the defaults whose saver writes nothing."""
    return LiveSettings(VocalisConfig(), tmp_path / "config.toml", saver=lambda _config, _path: None)


def test_engines_need_no_restart_before_any_change(tmp_path) -> None:  # type: ignore[no-untyped-def]
    assert _settings_with_recorded_saves(tmp_path).engines_need_restart() is False


def test_a_new_idle_timeout_needs_a_restart(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = _settings_with_recorded_saves(tmp_path)
    settings.update(settings.config.model_copy(update={"model_idle_timeout_seconds": 120.0}))
    assert settings.engines_need_restart() is True


@pytest.mark.parametrize("field", MODEL_PATH_FIELDS)
def test_a_new_model_location_needs_a_restart(tmp_path, field: str) -> None:  # type: ignore[no-untyped-def]
    settings = _settings_with_recorded_saves(tmp_path)
    engines = settings.config.engines.model_copy(update={field: "/somewhere/else"})
    settings.update(settings.config.model_copy(update={"engines": engines}))
    assert settings.engines_need_restart() is True


def test_changing_back_to_the_start_value_needs_no_restart(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = _settings_with_recorded_saves(tmp_path)
    started = settings.config
    settings.update(started.model_copy(update={"model_idle_timeout_seconds": 120.0}))
    settings.update(started)
    assert settings.engines_need_restart() is False


def test_live_settings_like_voice_need_no_restart(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = _settings_with_recorded_saves(tmp_path)
    engines = settings.config.engines.model_copy(update={"kokoro_voice": "af_bella"})
    settings.update(settings.config.model_copy(update={"engines": engines, "speaking_speed": 1.3}))
    assert settings.engines_need_restart() is False
```

Make sure `pytest`, `LiveSettings`, `VocalisConfig` are imported at the top of `tests/test_settings.py` (add if missing).

- [ ] **Step 2: Run to verify they fail**

Run: `PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_settings.py -q --no-cov`
Expected: FAIL — `ImportError: cannot import name 'MODEL_PATH_FIELDS'`.

- [ ] **Step 3: Implement** — in `src/vocalis/settings.py` add after the imports:

```python
MODEL_PATH_FIELDS: tuple[str, ...] = (
    "parakeet_model_dir",
    "qwen3_tts_model_dir",
    "piper_binary_path",
    "piper_voice_model_path",
    "kokoro_model_path",
    "kokoro_voices_path",
)
```

In `LiveSettings.__init__`, next to `self._started_with = config.shortcuts`, add `self._started_config = config`. Add the method after `shortcuts_need_restart`:

```python
    def engines_need_restart(self) -> bool:
        """Whether a saved setting that is only read at start-up differs from
        the one in use: the voice idle-unload time or a model location."""
        current = self.config
        started = self._started_config
        if current.model_idle_timeout_seconds != started.model_idle_timeout_seconds:
            return True
        return any(
            getattr(current.engines, field) != getattr(started.engines, field)
            for field in MODEL_PATH_FIELDS
        )
```

- [ ] **Step 4: Run to verify they pass**

Run: `PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_settings.py -q --no-cov`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/vocalis/settings.py tests/test_settings.py
git commit -m "feat: tell when saved model settings need a restart"
```

---

### Task 5: Tabbed Settings window with idle timeout and Model files

**Files:**
- Modify: `src/vocalis/ui/settings_window.py` (layout, new widgets, load/save, validation)
- Test: `tests/test_settings_window.py` (append; all 29 existing tests must stay green)

**Interfaces:**
- Consumes: `MODEL_PATH_FIELDS`, `LiveSettings.engines_need_restart()` (Task 4); `theme.mark_primary` (Task 1).
- Produces: `SettingsWindow._tabs: QTabWidget`, `_idle_timeout_spin: QSpinBox`, `_path_edits: dict[str, QLineEdit]`, `_browse_buttons: dict[str, QPushButton]`. All existing attribute names unchanged.

Tab contents (exact):
- "General": `_notifications_box`, `_widget_box`, row "Sound output" (`_output_combo` + `_test_sound_button`).
- "Voice & Engine": rows "Voice", "Quick Reader voice engine", "Document Reader voice engine", "Kokoro voice", "Speaking speed" (slider + label), then `_preview_button`.
- "Performance": rows "Max CPU threads (0 = no limit)", "Max RAM, MB (0 = no limit)", "GPU VRAM ceiling, MB", "Unload the voice after (seconds idle)".
- "Shortcuts": rows "Dictation (hold)", "Quick Reader  Ctrl+Alt+", "Document Reader  Ctrl+Alt+", "OCR Reader  Ctrl+Alt+".
- "Model files": one row per `MODEL_PATH_FIELDS` entry, label from `_MODEL_PATHS`, a `QLineEdit` + "Browse…" button.
- Below the tabs: `_message_label`, then a row: stretch, `_save_button` (primary), `_close_button`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_settings_window.py` (add imports `from unittest.mock import patch` and `from vocalis.settings import MODEL_PATH_FIELDS`):

```python
_TAB_NAMES = ["General", "Voice & Engine", "Performance", "Shortcuts", "Model files"]


def _tab_of(window: SettingsWindow, widget: object) -> str:
    """The name of the tab that holds a widget."""
    tabs = window._tabs  # noqa: SLF001
    for index in range(tabs.count()):
        if tabs.widget(index).isAncestorOf(widget):  # type: ignore[arg-type]
            return tabs.tabText(index).replace("&&", "&")
    raise AssertionError("widget is on no tab")


def test_settings_are_grouped_into_five_tabs(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    _settings, window, _applied, _path = make(qtbot, tmp_path)
    tabs = window._tabs  # noqa: SLF001
    assert [tabs.tabText(i).replace("&&", "&") for i in range(tabs.count())] == _TAB_NAMES


def test_each_control_is_on_its_tab(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    _settings, w, _applied, _path = make(qtbot, tmp_path)
    expected = {
        "General": [w._notifications_box, w._widget_box, w._output_combo, w._test_sound_button],  # noqa: SLF001
        "Voice & Engine": [w._voice_combo, w._quick_engine_combo, w._doc_engine_combo,  # noqa: SLF001
                           w._kokoro_voice_combo, w._speed_slider, w._preview_button],  # noqa: SLF001
        "Performance": [w._cpu_threads_spin, w._max_ram_spin, w._vram_ceiling_spin,  # noqa: SLF001
                        w._idle_timeout_spin],  # noqa: SLF001
        "Shortcuts": [w._ptt_combo, w._quick_combo, w._doc_combo, w._ocr_combo],  # noqa: SLF001
        "Model files": [*w._path_edits.values(), *w._browse_buttons.values()],  # noqa: SLF001
    }
    for tab, widgets in expected.items():
        for widget in widgets:
            assert _tab_of(w, widget) == tab


def test_save_close_and_message_are_outside_the_tabs(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    _settings, w, _applied, _path = make(qtbot, tmp_path)
    for widget in (w._save_button, w._close_button, w._message_label):  # noqa: SLF001
        assert not w._tabs.isAncestorOf(widget)  # noqa: SLF001
    assert w._save_button.property("role") == "primary"  # noqa: SLF001


def test_idle_timeout_is_shown_and_saved(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = VocalisConfig(model_idle_timeout_seconds=45.0)
    settings, w, _applied, path = make(qtbot, tmp_path, config=config)
    assert w._idle_timeout_spin.value() == 45  # noqa: SLF001
    assert w._idle_timeout_spin.minimum() == 1 and w._idle_timeout_spin.maximum() == 3600  # noqa: SLF001
    w._idle_timeout_spin.setValue(120)  # noqa: SLF001
    w._save_button.click()  # noqa: SLF001
    assert load_config(path).model_idle_timeout_seconds == 120.0
    assert w._message_label.text() == "Saved. Restart Vocalis to use the new model settings."  # noqa: SLF001


def test_model_locations_are_shown(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = VocalisConfig()
    _settings, w, _applied, _path = make(qtbot, tmp_path, config=config)
    assert set(w._path_edits) == set(MODEL_PATH_FIELDS)  # noqa: SLF001
    for field in MODEL_PATH_FIELDS:
        assert w._path_edits[field].text() == getattr(config.engines, field)  # noqa: SLF001


def test_an_existing_new_model_file_is_saved_with_a_restart_note(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    model = tmp_path / "kokoro.onnx"
    model.write_bytes(b"fake model for test")
    _settings, w, applied, path = make(qtbot, tmp_path)
    w._path_edits["kokoro_model_path"].setText(str(model))  # noqa: SLF001
    w._save_button.click()  # noqa: SLF001
    assert load_config(path).engines.kokoro_model_path == str(model)
    assert w._message_label.text() == "Saved. Restart Vocalis to use the new model settings."  # noqa: SLF001
    assert applied == [1]


def test_an_existing_new_model_folder_is_saved(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    folder = tmp_path / "parakeet"
    folder.mkdir()
    _settings, w, _applied, path = make(qtbot, tmp_path)
    w._path_edits["parakeet_model_dir"].setText(str(folder))  # noqa: SLF001
    w._save_button.click()  # noqa: SLF001
    assert load_config(path).engines.parakeet_model_dir == str(folder)


def test_a_changed_model_location_that_does_not_exist_is_refused(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    missing = tmp_path / "nope.onnx"
    _settings, w, applied, path = make(qtbot, tmp_path)
    w._path_edits["kokoro_model_path"].setText(str(missing))  # noqa: SLF001
    w._save_button.click()  # noqa: SLF001
    assert w._message_label.text() == f"Not saved: the Kokoro model file was not found at {missing}."  # noqa: SLF001
    assert not path.exists()
    assert applied == []


def test_a_file_where_a_folder_is_needed_is_refused(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    a_file = tmp_path / "not-a-folder.txt"
    a_file.write_text("x")
    _settings, w, _applied, path = make(qtbot, tmp_path)
    w._path_edits["qwen3_tts_model_dir"].setText(str(a_file))  # noqa: SLF001
    w._save_button.click()  # noqa: SLF001
    assert w._message_label.text().startswith("Not saved: the Qwen3-TTS model folder was not found at")  # noqa: SLF001
    assert not path.exists()


def test_an_emptied_model_location_is_refused(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    _settings, w, applied, path = make(qtbot, tmp_path)
    w._path_edits["piper_binary_path"].setText("   ")  # noqa: SLF001
    w._save_button.click()  # noqa: SLF001
    assert w._message_label.text() == "Not saved: the Piper program was not found at (empty)."  # noqa: SLF001
    assert not path.exists()
    assert applied == []


def test_unchanged_missing_model_locations_do_not_block_saving(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The default locations do not exist on a test machine; saving other settings still works."""
    _settings, w, _applied, path = make(qtbot, tmp_path)
    w._notifications_box.setChecked(False)  # noqa: SLF001
    w._save_button.click()  # noqa: SLF001
    assert w._message_label.text() == "Saved."  # noqa: SLF001
    assert load_config(path).notifications_enabled is False


def test_browse_fills_a_file_location(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    _settings, w, _applied, _path = make(qtbot, tmp_path)
    with patch(
        "vocalis.ui.settings_window.QFileDialog.getOpenFileName", return_value=("/picked/voices.bin", "")
    ):
        w._browse_buttons["kokoro_voices_path"].click()  # noqa: SLF001
    assert w._path_edits["kokoro_voices_path"].text() == "/picked/voices.bin"  # noqa: SLF001


def test_browse_fills_a_folder_location(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    _settings, w, _applied, _path = make(qtbot, tmp_path)
    with patch("vocalis.ui.settings_window.QFileDialog.getExistingDirectory", return_value="/picked/dir"):
        w._browse_buttons["parakeet_model_dir"].click()  # noqa: SLF001
    assert w._path_edits["parakeet_model_dir"].text() == "/picked/dir"  # noqa: SLF001


def test_cancelling_browse_keeps_the_old_location(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    _settings, w, _applied, _path = make(qtbot, tmp_path)
    before = w._path_edits["piper_voice_model_path"].text()  # noqa: SLF001
    with patch("vocalis.ui.settings_window.QFileDialog.getOpenFileName", return_value=("", "")):
        w._browse_buttons["piper_voice_model_path"].click()  # noqa: SLF001
    assert w._path_edits["piper_voice_model_path"].text() == before  # noqa: SLF001
```

Also check the existing test that asserts the shortcut restart message: if a test expects exactly `"Saved. Restart Vocalis to use the new shortcuts."`, that must stay true (the engine note is only appended when `engines_need_restart()` is true).

- [ ] **Step 2: Run to verify they fail**

Run: `PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_settings_window.py -q --no-cov`
Expected: new tests FAIL (`AttributeError: _tabs` etc.).

- [ ] **Step 3: Implement** in `src/vocalis/ui/settings_window.py`:

Imports: add `from pathlib import Path`, add `QFileDialog, QLineEdit, QTabWidget` to the `PySide6.QtWidgets` import, add `from vocalis.settings import KOKORO_VOICES, MODEL_PATH_FIELDS, VOICES, LiveSettings` (extend the existing import), add `from vocalis.ui.theme import mark_primary`. Update the module docstring's first line to: `"""The Settings window: every setting, grouped into tabs.`

Constants (after `_VRAM_CEILING_MAXIMUM_MB`):

```python
_ENGINE_RESTART_NOTE = " Restart Vocalis to use the new model settings."
_IDLE_TIMEOUT_MINIMUM_S = 1
_IDLE_TIMEOUT_MAXIMUM_S = 3600
_EMPTY_PATH_TEXT = "(empty)"
_TAB_NAMES = ("General", "Voice & Engine", "Performance", "Shortcuts", "Model files")
# field -> (row label, noun used in the "not found" message, is a folder)
_MODEL_PATHS: dict[str, tuple[str, str, bool]] = {
    "parakeet_model_dir": ("Speech recognition model folder", "speech recognition model folder", True),
    "qwen3_tts_model_dir": ("Qwen3-TTS model folder", "Qwen3-TTS model folder", True),
    "piper_binary_path": ("Piper program", "Piper program", False),
    "piper_voice_model_path": ("Piper voice file", "Piper voice file", False),
    "kokoro_model_path": ("Kokoro model file", "Kokoro model file", False),
    "kokoro_voices_path": ("Kokoro voices file", "Kokoro voices file", False),
}
```

Add a module-level guard right after `_MODEL_PATHS` so the two lists can never drift: `assert tuple(_MODEL_PATHS) == MODEL_PATH_FIELDS`. (Rule: an `assert` at import is acceptable here because it checks two constants in this repo, not user input.)

In `__init__`, after `self._vram_ceiling_spin...` lines add:

```python
        self._idle_timeout_spin = QSpinBox()
        self._idle_timeout_spin.setRange(_IDLE_TIMEOUT_MINIMUM_S, _IDLE_TIMEOUT_MAXIMUM_S)
        self._path_edits: dict[str, QLineEdit] = {field: QLineEdit() for field in MODEL_PATH_FIELDS}
        self._browse_buttons: dict[str, QPushButton] = {
            field: QPushButton("Browse…") for field in MODEL_PATH_FIELDS
        }
        self._tabs = QTabWidget()
```

and after `self._message_label.setWordWrap(True)`: `mark_primary(self._save_button)`. After the existing `.connect` lines add:

```python
        for field, button in self._browse_buttons.items():
            button.clicked.connect(lambda _checked=False, name=field: self._on_browse_clicked(name))
```

Replace `_build_layout` with:

```python
    def _build_layout(self) -> None:
        """Place the widgets on five tabs, with the message and Save/Close below."""
        for name, page in zip(_TAB_NAMES, (
            self._general_page(), self._voice_page(), self._performance_page(),
            self._shortcuts_page(), self._model_files_page(),
        )):
            self._tabs.addTab(page, name.replace("&", "&&"))
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self._save_button)
        buttons.addWidget(self._close_button)
        layout = QVBoxLayout(self)
        layout.addWidget(self._tabs)
        layout.addWidget(self._message_label)
        layout.addLayout(buttons)

    @staticmethod
    def _page(form: QFormLayout, *extra: QWidget) -> QWidget:
        """A tab page holding a form and, under it, any extra widgets."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addLayout(form)
        for widget in extra:
            layout.addWidget(widget)
        layout.addStretch(1)
        return page

    def _general_page(self) -> QWidget:
        """Switches and the sound output."""
        form = QFormLayout()
        form.addRow(self._notifications_box)
        form.addRow(self._widget_box)
        output_row = QHBoxLayout()
        output_row.addWidget(self._output_combo)
        output_row.addWidget(self._test_sound_button)
        form.addRow("Sound output", output_row)
        return self._page(form)

    def _voice_page(self) -> QWidget:
        """Voices, engines and speaking speed."""
        form = QFormLayout()
        form.addRow("Voice", self._voice_combo)
        form.addRow("Quick Reader voice engine", self._quick_engine_combo)
        form.addRow("Document Reader voice engine", self._doc_engine_combo)
        form.addRow("Kokoro voice", self._kokoro_voice_combo)
        speed_row = QHBoxLayout()
        speed_row.addWidget(self._speed_slider)
        speed_row.addWidget(self._speed_label)
        form.addRow("Speaking speed", speed_row)
        return self._page(form, self._preview_button)

    def _performance_page(self) -> QWidget:
        """Resource limits and the idle-unload time."""
        form = QFormLayout()
        form.addRow("Max CPU threads (0 = no limit)", self._cpu_threads_spin)
        form.addRow("Max RAM, MB (0 = no limit)", self._max_ram_spin)
        form.addRow("GPU VRAM ceiling, MB", self._vram_ceiling_spin)
        form.addRow("Unload the voice after (seconds idle)", self._idle_timeout_spin)
        return self._page(form)

    def _shortcuts_page(self) -> QWidget:
        """The four keyboard shortcuts."""
        form = QFormLayout()
        form.addRow("Dictation (hold)", self._ptt_combo)
        form.addRow("Quick Reader  Ctrl+Alt+", self._quick_combo)
        form.addRow("Document Reader  Ctrl+Alt+", self._doc_combo)
        form.addRow("OCR Reader  Ctrl+Alt+", self._ocr_combo)
        return self._page(form)

    def _model_files_page(self) -> QWidget:
        """Where each model lives on disk."""
        form = QFormLayout()
        for field in MODEL_PATH_FIELDS:
            row = QHBoxLayout()
            row.addWidget(self._path_edits[field])
            row.addWidget(self._browse_buttons[field])
            form.addRow(_MODEL_PATHS[field][0], row)
        return self._page(form)
```

Note: `QFormLayout.addRow(widget)` with a single `QCheckBox` is valid (spans both columns). Check before reading on that the removed buttons row no longer holds `_preview_button` (it now lives on the Voice tab).

In `load_from_settings`, at the end:

```python
        self._idle_timeout_spin.setValue(round(config.model_idle_timeout_seconds))
        for field, edit in self._path_edits.items():
            edit.setText(getattr(config.engines, field))
```

New methods (put them after `_on_preview_clicked`):

```python
    def _on_browse_clicked(self, field: str) -> None:
        """Let the user pick a model location; cancelling keeps the old one."""
        label, _noun, is_folder = _MODEL_PATHS[field]
        current = self._path_edits[field].text()
        if is_folder:
            chosen = QFileDialog.getExistingDirectory(self, label, current)
        else:
            chosen, _filter = QFileDialog.getOpenFileName(self, label, current)
        if chosen:
            self._path_edits[field].setText(chosen)

    def _missing_model_location(self) -> str | None:
        """A "not saved" message for the first changed model location that
        does not exist, or None when every changed location is fine.

        Unchanged locations are not checked, so a PC where a model was never
        installed can still save its other settings.
        """
        engines = self._settings.config.engines
        for field in MODEL_PATH_FIELDS:
            text = self._path_edits[field].text().strip()
            if text == getattr(engines, field):
                continue
            _label, noun, is_folder = _MODEL_PATHS[field]
            path = Path(text).expanduser()
            exists = bool(text) and (path.is_dir() if is_folder else path.is_file())
            if not exists:
                return f"Not saved: the {noun} was not found at {text or _EMPTY_PATH_TEXT}."
        return None
```

In `_config_from_widgets`, extend the `engines` update dict with `**{field: self._path_edits[field].text().strip() for field in MODEL_PATH_FIELDS},` and the top-level update dict with `"model_idle_timeout_seconds": float(self._idle_timeout_spin.value()),`.

Replace `_on_save_clicked` with:

```python
    def _on_save_clicked(self) -> None:
        """Validate and save; show the outcome in the message label."""
        problem = self._missing_model_location()
        if problem is not None:
            self._message_label.setText(problem)
            return
        try:
            self._settings.update(self._config_from_widgets())
        except (ShortcutError, ConfigError) as exc:
            self._message_label.setText(f"Not saved: {exc}")
            return
        message = "Saved."
        if self._settings.shortcuts_need_restart():
            message += _RESTART_NOTE
        if self._settings.engines_need_restart():
            message += _ENGINE_RESTART_NOTE
        self._message_label.setText(message)
        self._on_applied()
```

Note on the existing behaviour: `_config_from_widgets()` may raise `pydantic.ValidationError` for out-of-range values — that behaviour is unchanged (spin boxes already clamp ranges).

- [ ] **Step 4: Run to verify they pass**

Run: `PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_settings_window.py tests/test_settings.py -q --no-cov`
Expected: all PASS (29 existing + new).

- [ ] **Step 5: Commit**

```bash
git add src/vocalis/ui/settings_window.py tests/test_settings_window.py
git commit -m "feat: tabbed Settings window with idle timeout and model file locations"
```

---

### Task 6: Document Reader reports the shown text and each piece's span

**Files:**
- Modify: `src/vocalis/features/doc_reader.py` (imports, `__init__`, `_prepare_and_read`, `_read_worker`, `_report_progress`, new helpers)
- Test: `tests/test_doc_reader.py` (extend `_make_feature`, update `_gate_chunking`, append tests)

**Interfaces:**
- Consumes: `chunk_spans(text) -> list[Chunk]`, `Chunk(text, start, end)` from `vocalis.text.chunker` (exists).
- Produces (used by Task 7): `DocumentReaderFeature.__init__(..., on_document: Callable[[str], None] | None = None, on_chunk_span: Callable[[int, int], None] | None = None)` — new keyword params added at the END of the signature; stored as `self._on_document`, `self._on_chunk_span`. `on_document(text)` is called once per reading with the text the spans refer to (markdown-normalized, line endings unified to `\n`), before the first sound; `on_chunk_span(start, end)` is called each time a piece starts being heard, right after `on_progress`.

- [ ] **Step 1: Write the failing tests** — in `tests/test_doc_reader.py`:

1. Change the chunker import to `from vocalis.text.chunker import chunk_spans, chunk_text` and add `from vocalis.text.normalizer import normalize_markdown`.

2. Extend `_make_feature` with two keyword params and pass them through:

```python
    on_document: "Callable[[str], None] | None" = None,
    on_chunk_span: "Callable[[int, int], None] | None" = None,
```
and in the `DocumentReaderFeature(...)` call add `on_document=on_document, on_chunk_span=on_chunk_span,`.

3. Replace `_gate_chunking` so it gates the function the reader now calls:

```python
def _gate_chunking(monkeypatch: pytest.MonkeyPatch) -> "tuple[threading.Event, threading.Event]":
    """Hold the reading worker inside the chunking step; returns (in_chunking, let_go) events."""
    in_chunking = threading.Event()
    let_go = threading.Event()
    real_chunk_spans = chunk_spans

    def gated(text: str) -> "list[Chunk]":
        """Wait for the test, then chunk normally."""
        in_chunking.set()
        assert let_go.wait(timeout=5.0), "the chunking gate was never opened"
        return real_chunk_spans(text)

    monkeypatch.setattr("vocalis.features.doc_reader.chunk_spans", gated)
    return in_chunking, let_go
```
(add `Chunk` to the chunker import.)

4. Append the tests:

```python
def test_reports_the_normalized_text_once_and_each_pieces_span_in_order() -> None:
    documents: "list[str]" = []
    spans: "list[tuple[int, int]]" = []
    text = "# Title\n\n" + _several_chunk_text()
    feature, _statuses, progresses, finishes = _make_feature(
        fallback=FakeSynthesizer(), primary=FakeSynthesizer(), notifier=RecordingNotifier(),
        on_document=documents.append, on_chunk_span=lambda start, end: spans.append((start, end)),
    )

    feature.read_text(text)
    wait_until(lambda: bool(finishes))

    shown = normalize_markdown(text)
    expected = [(chunk.start, chunk.end) for chunk in chunk_spans(shown)]
    assert documents == [shown]
    assert "#" not in shown  # the box will show what is actually read
    wait_until(lambda: len(spans) == len(expected))
    assert spans == expected
    assert len(progresses) == len(expected)
    for start, end in spans:
        assert shown[start:end].strip()


def test_windows_and_old_mac_line_endings_become_plain_line_breaks() -> None:
    """Qt shows \\r\\n and \\r as one break each, so spans must refer to text without \\r."""
    documents: "list[str]" = []
    spans: "list[tuple[int, int]]" = []
    feature, _statuses, _progresses, finishes = _make_feature(
        fallback=FakeSynthesizer(), primary=FakeSynthesizer(), notifier=RecordingNotifier(),
        on_document=documents.append, on_chunk_span=lambda start, end: spans.append((start, end)),
    )

    feature.read_text("First line.\r\nSecond line.\rThird line.")
    wait_until(lambda: bool(finishes))

    assert len(documents) == 1
    assert "\r" not in documents[0]
    wait_until(lambda: len(spans) == len(chunk_spans(documents[0])))
    assert spans == [(c.start, c.end) for c in chunk_spans(documents[0])]


def test_a_raising_document_hook_does_not_stop_the_reading() -> None:
    def boom(_text: str) -> None:
        """Fail like a broken window."""
        raise RuntimeError("window gone")

    feature, statuses, _progresses, finishes = _make_feature(
        fallback=FakeSynthesizer(), primary=FakeSynthesizer(), notifier=RecordingNotifier(),
        on_document=boom,
    )
    feature.read_text("Hello there.")
    wait_until(lambda: bool(finishes))
    assert "Done" in statuses


def test_a_raising_span_hook_does_not_stop_the_reading_or_progress() -> None:
    def boom(_start: int, _end: int) -> None:
        """Fail like a broken window."""
        raise RuntimeError("window gone")

    feature, statuses, progresses, finishes = _make_feature(
        fallback=FakeSynthesizer(), primary=FakeSynthesizer(), notifier=RecordingNotifier(),
        on_chunk_span=boom,
    )
    feature.read_text(_several_chunk_text())
    wait_until(lambda: bool(finishes))
    assert "Done" in statuses
    wait_until(lambda: len(progresses) == len(chunk_text(_several_chunk_text())))


def test_no_document_report_when_there_is_nothing_to_read() -> None:
    documents: "list[str]" = []
    feature, statuses, _progresses, finishes = _make_feature(
        fallback=FakeSynthesizer(), primary=FakeSynthesizer(), notifier=RecordingNotifier(),
        on_document=documents.append,
    )
    feature.read_text("   ")
    wait_until(lambda: bool(finishes))
    assert documents == []
    assert "Nothing to read" in statuses


def test_a_stale_reading_reports_no_span() -> None:
    spans: "list[tuple[int, int]]" = []
    feature, _statuses, _progresses, finishes = _make_feature(
        fallback=FakeSynthesizer(), primary=FakeSynthesizer(), notifier=RecordingNotifier(),
        on_chunk_span=lambda start, end: spans.append((start, end)),
    )
    feature.read_text("Hello there.")
    wait_until(lambda: bool(finishes))
    wait_until(lambda: len(spans) == 1)
    stale = feature.generation
    feature._report_progress(stale, 1, 1, [Chunk("Hello there.", 0, 12)])  # noqa: SLF001
    assert spans == [(0, 12)]
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_doc_reader.py -q --no-cov`
Expected: FAIL — `TypeError: ... unexpected keyword argument 'on_document'` (and the gated tests fail because `chunk_spans` is not used yet).

- [ ] **Step 3: Implement** in `src/vocalis/features/doc_reader.py`:

- Imports: `from typing import Callable, Mapping, Sequence`; replace `from vocalis.text.chunker import chunk_text` with `from vocalis.text.chunker import Chunk, chunk_spans`.
- Module helper (after `_EXTRACTION_ERRORS`):

```python
def _unify_line_endings(text: str) -> str:
    """Turn Windows (\\r\\n) and old-Mac (\\r) line breaks into \\n.

    The Document Reader window shows each as a single break, so the
    highlight positions only match text that has no \\r in it.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")
```

- `__init__`: add the two params at the end of the signature (`on_document: Callable[[str], None] | None = None, on_chunk_span: Callable[[int, int], None] | None = None`), document them in the Args section ("on_document: Called once per reading, before the first sound, with the exact text the spans refer to (markdown removed, line breaks unified). The popup shows it. A callback that raises is logged and ignored. None means no hook." / "on_chunk_span: Called with (start, end) character positions in that text each time a piece starts to be heard, right after on_progress. Same failure rule."), and store `self._on_document = on_document`, `self._on_chunk_span = on_chunk_span`.
- Generic safe-call helper (next to `_report_started`):

```python
    @staticmethod
    def _call_hook(name: str, hook: "Callable[..., None] | None", *args: object) -> None:
        """Call an optional display hook; a failing hook is logged and never breaks the reading."""
        if hook is None:
            return
        try:
            hook(*args)
        except Exception:
            logger.exception("%s hook failed", name)
```

- `_prepare_and_read`: replace the two lines `normalized = normalize_markdown(raw_text)` / `chunks = chunk_text(normalized)` with:

```python
        normalized = _unify_line_endings(normalize_markdown(raw_text))
        chunks = chunk_spans(normalized)
```
and between `if not self._is_current(generation): return` and `self._notifier.notify(_TITLE, "Reading...")` add:

```python
        self._call_hook("on_document", self._on_document, normalized)
```

- `_read_worker(self, generation: int, chunks: "list[Chunk]")`: change the speaker's callback to `on_chunk_started=lambda current, total: self._report_progress(generation, current, total, chunks),` and the run call to `outcome = speaker.run([chunk.text for chunk in chunks])`.
- `_report_progress`:

```python
    def _report_progress(
        self, generation: int, current: int, total: int, chunks: "Sequence[Chunk]"
    ) -> None:
        """Forward 'piece N of M' and the piece's span while this reading is
        still the current one.

        A stop landing between the check and the calls can let one report
        arrive just after on_finished.
        """
        if not self._is_current(generation):
            return
        self._on_progress(current, total)
        chunk = chunks[current - 1]
        self._call_hook("on_chunk_span", self._on_chunk_span, chunk.start, chunk.end)
```

Update the module docstring's reading-design paragraph with one sentence: "Two optional display hooks report the text being read and the span of each piece as it starts, which the window uses to highlight it."

- [ ] **Step 4: Run to verify they pass**

Run: `PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_doc_reader.py tests/test_reading_hooks_integration.py tests/test_ocr_reader.py -q --no-cov`
Expected: all PASS. (If either extra file does not exist, drop it from the command; `ls tests | grep -i "hook\|ocr"` first.)

- [ ] **Step 5: Commit**

```bash
git add src/vocalis/features/doc_reader.py tests/test_doc_reader.py
git commit -m "feat: Document Reader reports the text it reads and each piece's span"
```

---

### Task 7: Document Reader window — restyle, highlight, and wiring

**Files:**
- Modify: `src/vocalis/ui/doc_popup.py`, `src/vocalis/cli.py` (`build_doc_reader`, ~lines 627-676)
- Test: `tests/test_doc_popup.py` (append), `tests/test_cli.py` (extend `test_build_doc_reader_wires_engines_popup_and_listener` or add one test)

**Interfaces:**
- Consumes: `on_document`, `on_chunk_span` params (Task 6); `theme.HIGHLIGHT_BG`, `theme.TEXT`, `theme.mark_primary` (Task 1).
- Produces: `DocumentReaderPopup.document_loaded = Signal(str)`, `DocumentReaderPopup.span_started = Signal(int, int)`, `DocumentReaderPopup._progress_bar: QProgressBar`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_doc_popup.py`:

```python
def _highlighted(popup: DocumentReaderPopup) -> str:
    """The text currently highlighted in the popup's box ("" when none)."""
    selections = popup._text_box.extraSelections()  # noqa: SLF001
    if not selections:
        return ""
    assert len(selections) == 1
    return selections[0].cursor.selectedText()


def _popup(qtbot) -> DocumentReaderPopup:  # type: ignore[no-untyped-def]
    popup = DocumentReaderPopup(FakeFeature())
    qtbot.addWidget(popup)
    return popup


def test_document_loaded_shows_the_text_read_only(qtbot) -> None:  # type: ignore[no-untyped-def]
    popup = _popup(qtbot)
    popup.document_loaded.emit("Hello there. Bye now.")
    assert popup._text_box.toPlainText() == "Hello there. Bye now."  # noqa: SLF001
    assert popup._text_box.isReadOnly()  # noqa: SLF001
    assert _highlighted(popup) == ""


def test_span_started_highlights_exactly_that_piece(qtbot) -> None:  # type: ignore[no-untyped-def]
    popup = _popup(qtbot)
    popup.document_loaded.emit("Hello there. Bye now.")
    popup.span_started.emit(13, 21)
    assert _highlighted(popup) == "Bye now."
    popup.span_started.emit(0, 12)
    assert _highlighted(popup) == "Hello there."


def test_highlight_uses_the_theme_colour(qtbot) -> None:  # type: ignore[no-untyped-def]
    popup = _popup(qtbot)
    popup.document_loaded.emit("Hello there.")
    popup.span_started.emit(0, 12)
    selection = popup._text_box.extraSelections()[0]  # noqa: SLF001
    assert selection.format.background().color().name() == "#5c1a1d"


def test_highlight_is_right_after_emoji(qtbot) -> None:  # type: ignore[no-untyped-def]
    """Qt counts an emoji as two positions, Python as one."""
    popup = _popup(qtbot)
    text = "\U0001F600\U0001F600 Hi. There now."
    popup.document_loaded.emit(text)
    start = text.index("There")
    popup.span_started.emit(start, len(text))
    assert _highlighted(popup) == "There now."


def test_highlight_spans_across_lines(qtbot) -> None:  # type: ignore[no-untyped-def]
    popup = _popup(qtbot)
    text = "Line one.\nLine two."
    popup.document_loaded.emit(text)
    popup.span_started.emit(text.index("Line two"), len(text))
    assert _highlighted(popup) == "Line two."


def test_a_span_past_the_end_is_clamped_and_an_empty_span_clears(qtbot) -> None:  # type: ignore[no-untyped-def]
    popup = _popup(qtbot)
    popup.document_loaded.emit("Short.")
    popup.span_started.emit(0, 10_000)
    assert _highlighted(popup) == "Short."
    popup.span_started.emit(3, 3)
    assert _highlighted(popup) == ""
    popup.span_started.emit(50, 60)
    assert _highlighted(popup) == ""


def test_highlight_scrolls_into_view(qtbot) -> None:  # type: ignore[no-untyped-def]
    popup = _popup(qtbot)
    popup.resize(400, 300)
    popup.show()
    lines = [f"Sentence number {n}." for n in range(300)]
    text = "\n".join(lines)
    popup.document_loaded.emit(text)
    popup.span_started.emit(text.index(lines[-1]), len(text))
    assert popup._text_box.verticalScrollBar().value() > 0  # noqa: SLF001


def test_reading_finished_clears_the_highlight_and_allows_editing(qtbot) -> None:  # type: ignore[no-untyped-def]
    popup = _popup(qtbot)
    popup.document_loaded.emit("Hello there.")
    popup.span_started.emit(0, 12)
    popup.reading_finished.emit()
    assert _highlighted(popup) == ""
    assert not popup._text_box.isReadOnly()  # noqa: SLF001
    assert popup._text_box.toPlainText() == "Hello there."  # noqa: SLF001


def test_progress_bar_follows_progress(qtbot) -> None:  # type: ignore[no-untyped-def]
    popup = _popup(qtbot)
    popup.progress_changed.emit(2, 4)
    assert popup._progress_bar.maximum() == 4 and popup._progress_bar.value() == 2  # noqa: SLF001
    assert popup._progress_label.text() == "piece 2 of 4"  # noqa: SLF001
    popup.reading_finished.emit()
    assert popup._progress_bar.value() == 0  # noqa: SLF001


def test_read_is_the_primary_button(qtbot) -> None:  # type: ignore[no-untyped-def]
    popup = _popup(qtbot)
    assert popup._read_button.property("role") == "primary"  # noqa: SLF001
```

In `tests/test_cli.py::test_build_doc_reader_wires_engines_popup_and_listener`, append at the end:

```python
    # The highlight hooks must reach the popup too.
    feature._on_document("Hello there.")  # noqa: SLF001
    assert popup._text_box.toPlainText() == "Hello there."  # noqa: SLF001
    assert popup._text_box.isReadOnly()  # noqa: SLF001
    feature._on_chunk_span(0, 5)  # noqa: SLF001
    assert popup._text_box.extraSelections()[0].cursor.selectedText() == "Hello"  # noqa: SLF001
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_doc_popup.py tests/test_cli.py -q --no-cov`
Expected: FAIL — `AttributeError: ... 'document_loaded'`.

- [ ] **Step 3: Implement**

`src/vocalis/ui/doc_popup.py`:
- Imports: add `from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor`; add `QProgressBar, QTextEdit` to the widgets import; add `from vocalis.ui import theme`.
- Module docstring: add one paragraph: "While reading, the box shows the exact text being read (read-only) and highlights the piece being heard; the positions it receives are Python character positions, converted here to Qt's UTF-16 positions."
- Signals: add `document_loaded = Signal(str)` and `span_started = Signal(int, int)`.
- In `__init__`: `self._document_text = ""`; `theme.mark_primary(self._read_button)`; `self._progress_bar = QProgressBar()`, `self._progress_bar.setTextVisible(False)`, `self._reset_progress_bar()`; connect `self.document_loaded.connect(self._on_document_loaded)` and `self.span_started.connect(self._on_span_started)`; in the layout put the progress bar between the status label and the progress label.
- New/changed methods:

```python
    @staticmethod
    def _qt_position(text: str, index: int) -> int:
        """Convert a Python character position in `text` to Qt's UTF-16 position."""
        return len(text[:index].encode("utf-16-le")) // 2

    def _on_document_loaded(self, text: str) -> None:
        """Show the exact text being read, read-only, with nothing highlighted yet."""
        self._document_text = text
        self._text_box.setPlainText(text)
        self._text_box.setReadOnly(True)
        self._text_box.setExtraSelections([])

    def _on_span_started(self, start: int, end: int) -> None:
        """Highlight the piece being heard and scroll it into view (never takes focus).

        A span past the end is clamped; an empty one clears the highlight.
        """
        text = self._document_text
        start = min(max(start, 0), len(text))
        end = min(max(end, start), len(text))
        if start == end:
            self._text_box.setExtraSelections([])
            return
        document = self._text_box.document()
        qt_start = self._qt_position(text, start)
        cursor = QTextCursor(document)
        cursor.setPosition(qt_start)
        cursor.setPosition(self._qt_position(text, end), QTextCursor.MoveMode.KeepAnchor)
        char_format = QTextCharFormat()
        char_format.setBackground(QColor(theme.HIGHLIGHT_BG))
        char_format.setForeground(QColor(theme.TEXT))
        selection = QTextEdit.ExtraSelection()
        selection.cursor = cursor
        selection.format = char_format
        self._text_box.setExtraSelections([selection])
        view = QTextCursor(document)
        view.setPosition(qt_start)
        self._text_box.setTextCursor(view)
        self._text_box.ensureCursorVisible()

    def _reset_progress_bar(self) -> None:
        """Show an empty progress bar."""
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(0)
```

  `_on_reading_finished` additionally: `self._text_box.setExtraSelections([])`, `self._text_box.setReadOnly(False)`, `self._reset_progress_bar()`.
  `_update_progress_label` additionally: `self._progress_bar.setRange(0, max(total, 1))` and `self._progress_bar.setValue(min(max(current, 0), max(total, 1)))`.

  Note: `QTextEdit.ExtraSelection` field assignment (`selection.cursor = ...`, `selection.format = ...`) is how PySide6 sets these; assign whole objects, do not mutate `selection.format` in place. If the emoji test fails, check the `_qt_position` conversion before anything else.

`src/vocalis/cli.py` in `build_doc_reader`, next to `on_progress`:

```python
    def on_document(text: str) -> None:
        """Show the text being read in the popup."""
        popup.document_loaded.emit(text)

    def on_chunk_span(start: int, end: int) -> None:
        """Highlight the piece being heard in the popup."""
        popup.span_started.emit(start, end)
```
and pass `on_document=on_document, on_chunk_span=on_chunk_span,` to `DocumentReaderFeature(...)`.

- [ ] **Step 4: Run to verify they pass**

Run: `PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_doc_popup.py tests/test_cli.py tests/test_doc_reader.py -q --no-cov`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/vocalis/ui/doc_popup.py src/vocalis/cli.py tests/test_doc_popup.py tests/test_cli.py
git commit -m "feat: highlight the piece being read in the Document Reader window"
```

---

### Task 8: Documentation and changelog

**Files:**
- Modify: `docs/architecture.md` (new section `## Stage 7 addition: red-and-black UI and reading highlight` at the end), `docs/user-docs.md` (sections "## The tray icon", "## The floating widget", "## Document Reader (Phase 4)", "## Settings"), `docs/file-purpose.md` (new section `## Stage 7: red-and-black UI and reading highlight`), `docs/workflow.md` (Document Reader flow: one line on highlight), `docs/algorithms.md` (short section "Highlight position conversion")
- Create: `.claude/2026-10-29_1900_CHANGELOG.md`

- [ ] **Step 1: Write the docs** — plain English, matching each file's existing heading style. Content to cover, all verifiable against the code from Tasks 1-7:
  - architecture: `ui/theme.py` owns all colours + stylesheet, applied once in `main()`; why text never sits on `ACCENT`; the highlight path `DocumentReaderFeature.on_document/on_chunk_span` → `cli.build_doc_reader` → `document_loaded/span_started` signals → popup `ExtraSelection`; line endings unified in the reader; UTF-16 conversion in the popup; `LiveSettings.engines_need_restart()`.
  - user-docs: the three tray icon looks (grey dot / bright red dot / red ring with white centre); the widget card can be dragged and stays put until Vocalis quits; while reading, the box shows the cleaned-up text being read and highlights the current piece (word-by-word is not possible); the five Settings tabs and what is on each; model locations: only changed ones are checked, a missing one is refused with a message, changes to model locations or the idle time need a restart.
  - file-purpose: `src/vocalis/ui/theme.py` (new) and one line each for changed `tray.py`, `floating_widget.py`, `settings_window.py`, `doc_popup.py`, `doc_reader.py`, `settings.py`, `cli.py`.
  - algorithms: contrast ratio formula (WCAG relative luminance) used by the palette test; Python-index → UTF-16 position conversion (`len(text[:i].encode("utf-16-le")) // 2`, O(i) per highlight, once per piece).
- [ ] **Step 2: Write the changelog** `.claude/2026-10-29_1900_CHANGELOG.md` with: summary, files changed, user-visible changes, performance impact ("negligible: stylesheet parsed once; one selection update per piece; no new dependencies"), rollback ("revert the Stage 7 merge commit range; no config format change — the only new saved value is the existing `model_idle_timeout_seconds` and existing model path fields, so older Vocalis versions read the file unchanged").
- [ ] **Step 3: Check every code reference in the docs exists** — `grep -n "apply_theme\|engines_need_restart\|document_loaded\|span_started\|on_chunk_span\|on_document" src -r` must find each name the docs mention.
- [ ] **Step 4: Commit**

```bash
git add docs/architecture.md docs/user-docs.md docs/file-purpose.md docs/workflow.md docs/algorithms.md .claude/2026-10-29_1900_CHANGELOG.md
git commit -m "docs: document Stage 7 theme, tabbed Settings and reading highlight"
```

---

## Final verification (controller, once, after Task 8)

- `PYTHONPATH=$PWD/src .venv/bin/python -m pytest -q` → all pass, coverage ≥ 85%, zero warnings.
- `.venv/bin/python -m mypy` → 0 errors.
- `grep -rnE "#[0-9a-fA-F]{6}" src/vocalis --include=*.py | grep -v "src/vocalis/ui/theme.py"` → no styling colour literals outside `theme.py` (any hit must be justified as non-styling).
