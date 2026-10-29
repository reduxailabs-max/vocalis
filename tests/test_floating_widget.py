"""Tests for vocalis.ui.floating_widget.FloatingWidget."""
from typing import Callable

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QGuiApplication, QMouseEvent

from vocalis.ui.floating_widget import FloatingWidget
from vocalis.ui.status_hub import ReadingControls, StatusHub


def make(qtbot, enabled: Callable[[], bool] = lambda: True) -> tuple[StatusHub, FloatingWidget]:  # type: ignore[no-untyped-def]
    hub = StatusHub()
    widget = FloatingWidget(hub, enabled)
    qtbot.addWidget(widget)
    return hub, widget


def pausable(log: "list[str]") -> ReadingControls:
    return ReadingControls(stop=lambda: log.append("stop"), pause=lambda: log.append("pause"), resume=lambda: log.append("resume"))


def test_never_takes_focus_and_stays_on_top(qtbot) -> None:  # type: ignore[no-untyped-def]
    _hub, widget = make(qtbot)
    flags = widget.windowFlags()
    assert flags & Qt.WindowType.WindowStaysOnTopHint
    assert flags & Qt.WindowType.WindowDoesNotAcceptFocus
    assert flags & Qt.WindowType.FramelessWindowHint
    assert widget.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)


def test_hidden_until_a_reading_starts_and_hidden_again_when_it_ends(qtbot) -> None:  # type: ignore[no-untyped-def]
    hub, widget = make(qtbot)
    assert not widget.isVisible()
    hub.begin_reading("Document", pausable([]))
    assert widget.isVisible()
    assert widget._source_label.text() == "Document"  # noqa: SLF001
    hub.end_reading()
    assert not widget.isVisible()


def test_status_and_progress_labels_update(qtbot) -> None:  # type: ignore[no-untyped-def]
    hub, widget = make(qtbot)
    hub.begin_reading("Document", pausable([]))
    hub.report_status("Reading...")
    hub.report_progress(3, 8)
    assert widget._status_label.text() == "Reading..."  # noqa: SLF001
    assert widget._progress_label.text() == "piece 3 of 8"  # noqa: SLF001


def test_buttons_drive_the_hub_and_pause_label_follows_the_reader(qtbot) -> None:  # type: ignore[no-untyped-def]
    log: "list[str]" = []
    hub, widget = make(qtbot)
    hub.begin_reading("Document", pausable(log))
    widget._pause_button.click()  # noqa: SLF001
    assert log == ["pause"]
    assert widget._pause_button.text() == "Pause"  # noqa: SLF001  (label changes only when the reader confirms)
    hub.set_paused(True)
    assert widget._pause_button.text() == "Resume"  # noqa: SLF001
    widget._pause_button.click()  # noqa: SLF001
    assert log == ["pause", "resume"]
    widget._stop_button.click()  # noqa: SLF001
    assert log[-1] == "stop"


def test_stop_only_reading_hides_the_pause_button(qtbot) -> None:  # type: ignore[no-untyped-def]
    hub, widget = make(qtbot)
    hub.begin_reading("Quick read", ReadingControls(stop=lambda: None))
    assert widget._pause_button.isHidden()  # noqa: SLF001
    assert not widget._stop_button.isHidden()  # noqa: SLF001


def test_switched_off_means_it_never_appears(qtbot) -> None:  # type: ignore[no-untyped-def]
    hub, widget = make(qtbot, enabled=lambda: False)
    hub.begin_reading("Document", pausable([]))
    assert not widget.isVisible()


def test_turning_the_switch_off_mid_reading_hides_it(qtbot) -> None:  # type: ignore[no-untyped-def]
    state = {"on": True}
    hub, widget = make(qtbot, enabled=lambda: state["on"])
    hub.begin_reading("Document", pausable([]))
    assert widget.isVisible()
    state["on"] = False
    widget.apply_enabled()
    assert not widget.isVisible()


def test_placed_in_the_bottom_right_corner_of_the_screen(qtbot) -> None:  # type: ignore[no-untyped-def]
    from PySide6.QtGui import QGuiApplication

    hub, widget = make(qtbot)
    hub.begin_reading("Document", pausable([]))
    area = QGuiApplication.primaryScreen().availableGeometry()
    assert widget.x() + widget.width() == area.right() + 1 - 16
    assert widget.y() + widget.height() == area.bottom() + 1 - 16


def _mouse(kind: QEvent.Type, global_x: int, global_y: int, widget: FloatingWidget, held: bool) -> QMouseEvent:
    """A left-button mouse event at a global screen position."""
    local = QPointF(widget.mapFromGlobal(QPoint(global_x, global_y)))
    buttons = Qt.MouseButton.LeftButton if held else Qt.MouseButton.NoButton
    return QMouseEvent(
        kind, local, local, QPointF(global_x, global_y), Qt.MouseButton.LeftButton, buttons,
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


def test_a_saved_spot_that_is_off_every_screen_falls_back_to_bottom_right(qtbot) -> None:  # type: ignore[no-untyped-def]
    hub, widget = make(qtbot)
    widget._user_position = QPoint(-50000, -50000)  # noqa: SLF001
    hub.begin_reading("Document", ReadingControls(stop=lambda: None))
    assert widget.pos() != QPoint(-50000, -50000)
    assert QGuiApplication.screenAt(widget.pos()) is not None


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
