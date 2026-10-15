"""Tests for vocalis.ui.floating_widget.FloatingWidget."""
from typing import Callable

from PySide6.QtCore import Qt

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
