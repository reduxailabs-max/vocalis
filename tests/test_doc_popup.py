"""Tests for vocalis.ui.doc_popup.DocumentReaderPopup.

Uses a plain fake in place of DocumentReaderFeature -- this suite
verifies the popup's own button-wiring and signal-handling logic only,
decoupled from the feature's internals (which test_doc_reader.py already
covers).
"""
from vocalis.ui.doc_popup import DocumentReaderPopup


class FakeFeature:
    def __init__(self) -> None:
        self.is_reading = False
        self.read_file_calls: "list[object]" = []
        self.read_text_calls: "list[str]" = []
        self.pause_calls = 0
        self.resume_calls = 0
        self.stop_calls = 0

    def read_file(self, path: object) -> None:
        self.read_file_calls.append(path)

    def read_text(self, text: str) -> None:
        self.read_text_calls.append(text)

    def pause(self) -> None:
        self.pause_calls += 1

    def resume(self) -> None:
        self.resume_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1


def test_read_button_sends_the_typed_text_to_the_feature(qtbot) -> None:  # type: ignore[no-untyped-def]
    feature = FakeFeature()
    popup = DocumentReaderPopup(feature)
    qtbot.addWidget(popup)

    popup._text_box.setPlainText("Hello there")  # noqa: SLF001
    popup._on_read_clicked()  # noqa: SLF001

    assert feature.read_text_calls == ["Hello there"]
    assert popup._read_button.isEnabled() is False  # noqa: SLF001
    assert popup._stop_button.isEnabled() is True  # noqa: SLF001


def test_read_button_with_empty_text_does_not_call_the_feature(qtbot) -> None:  # type: ignore[no-untyped-def]
    feature = FakeFeature()
    popup = DocumentReaderPopup(feature)
    qtbot.addWidget(popup)

    popup._text_box.setPlainText("   ")  # noqa: SLF001
    popup._on_read_clicked()  # noqa: SLF001

    assert feature.read_text_calls == []


def test_pause_button_toggles_and_calls_pause_then_resume(qtbot) -> None:  # type: ignore[no-untyped-def]
    feature = FakeFeature()
    feature.is_reading = True
    popup = DocumentReaderPopup(feature)
    qtbot.addWidget(popup)

    popup._on_pause_clicked()  # noqa: SLF001
    assert feature.pause_calls == 1
    assert popup._pause_button.text() == "Resume"  # noqa: SLF001

    popup._on_pause_clicked()  # noqa: SLF001
    assert feature.resume_calls == 1
    assert popup._pause_button.text() == "Pause"  # noqa: SLF001


def test_stop_button_calls_the_feature() -> None:
    feature = FakeFeature()
    popup = DocumentReaderPopup(feature)

    popup._on_stop_clicked()  # noqa: SLF001

    assert feature.stop_calls == 1


def test_status_signal_updates_the_status_label() -> None:
    popup = DocumentReaderPopup(FakeFeature())

    popup.status_changed.emit("Reading...")

    assert popup._status_label.text() == "Reading..."  # noqa: SLF001


def test_progress_signal_updates_the_progress_label() -> None:
    popup = DocumentReaderPopup(FakeFeature())

    popup.progress_changed.emit(3, 10)

    assert popup._progress_label.text() == "piece 3 of 10"  # noqa: SLF001


def test_reading_finished_signal_disables_pause_and_stop_and_resets_pause_label() -> None:
    popup = DocumentReaderPopup(FakeFeature())
    popup._on_pause_clicked()  # noqa: SLF001 -- flips to "Resume"

    popup.reading_finished.emit()

    assert popup._pause_button.isEnabled() is False  # noqa: SLF001
    assert popup._stop_button.isEnabled() is False  # noqa: SLF001
    assert popup._read_button.isEnabled() is True  # noqa: SLF001
    assert popup._pause_button.text() == "Pause"  # noqa: SLF001


def test_hotkey_trigger_while_idle_shows_the_window(qtbot) -> None:  # type: ignore[no-untyped-def]
    feature = FakeFeature()
    feature.is_reading = False
    popup = DocumentReaderPopup(feature)
    qtbot.addWidget(popup)

    popup.on_hotkey_trigger()

    assert popup.isVisible() is True
    assert feature.stop_calls == 0


def test_hotkey_trigger_while_reading_stops_it() -> None:
    feature = FakeFeature()
    feature.is_reading = True
    popup = DocumentReaderPopup(feature)

    popup.on_hotkey_trigger()

    assert feature.stop_calls == 1


def test_pause_click_when_nothing_is_reading_is_ignored(qtbot) -> None:  # type: ignore[no-untyped-def]
    feature = FakeFeature()  # is_reading is False
    popup = DocumentReaderPopup(feature)
    qtbot.addWidget(popup)

    popup._on_pause_clicked()  # noqa: SLF001

    assert feature.pause_calls == 0
    assert feature.resume_calls == 0
    assert popup._pause_button.text() == "Pause"  # noqa: SLF001


def test_pause_click_after_reading_ended_resets_a_stale_resume_label(qtbot) -> None:  # type: ignore[no-untyped-def]
    feature = FakeFeature()
    feature.is_reading = True
    popup = DocumentReaderPopup(feature)
    qtbot.addWidget(popup)
    popup._on_pause_clicked()  # noqa: SLF001
    assert popup._pause_button.text() == "Resume"  # noqa: SLF001

    feature.is_reading = False  # reading ended without reading_finished arriving yet
    popup._on_pause_clicked()  # noqa: SLF001

    assert feature.resume_calls == 0
    assert popup._pause_button.text() == "Pause"  # noqa: SLF001


def test_a_new_reading_starts_unpaused(qtbot) -> None:  # type: ignore[no-untyped-def]
    feature = FakeFeature()
    feature.is_reading = True
    popup = DocumentReaderPopup(feature)
    qtbot.addWidget(popup)
    popup._on_pause_clicked()  # noqa: SLF001
    popup.reading_finished.emit()

    popup._text_box.setPlainText("Again")  # noqa: SLF001
    popup._on_read_clicked()  # noqa: SLF001
    popup._on_pause_clicked()  # noqa: SLF001

    assert feature.pause_calls == 2  # second click pauses, not resumes
    assert feature.resume_calls == 0


def test_external_reading_shows_the_text_with_pause_and_stop_enabled(qtbot) -> None:  # type: ignore[no-untyped-def]
    feature = FakeFeature()
    popup = DocumentReaderPopup(feature)
    qtbot.addWidget(popup)

    popup.external_reading_started.emit("Text from the screen")

    assert popup._text_box.toPlainText() == "Text from the screen"  # noqa: SLF001
    assert popup.isVisible() is True
    assert popup._pause_button.isEnabled() is True  # noqa: SLF001
    assert popup._stop_button.isEnabled() is True  # noqa: SLF001
    assert popup._read_button.isEnabled() is False  # noqa: SLF001
    assert feature.read_text_calls == []  # the popup must not start a second reading


def test_external_reading_resets_a_leftover_resume_label(qtbot) -> None:  # type: ignore[no-untyped-def]
    popup = DocumentReaderPopup(FakeFeature())
    qtbot.addWidget(popup)
    popup._paused = True  # noqa: SLF001
    popup._pause_button.setText("Resume")  # noqa: SLF001

    popup.external_reading_started.emit("x")

    assert popup._pause_button.text() == "Pause"  # noqa: SLF001
