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


def test_reading_started_signal_enables_pause_and_stop() -> None:
    popup = DocumentReaderPopup(FakeFeature())
    assert popup._pause_button.isEnabled() is False  # noqa: SLF001

    popup.reading_started.emit()

    assert popup._pause_button.isEnabled() is True  # noqa: SLF001
    assert popup._stop_button.isEnabled() is True  # noqa: SLF001


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
    popup.reading_started.emit()
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


def test_late_text_and_spans_after_stop_are_ignored(qtbot) -> None:  # type: ignore[no-untyped-def]
    popup = _popup(qtbot)
    popup.reading_finished.emit()
    popup.document_loaded.emit("Late text.")
    popup.span_started.emit(0, 4)
    assert popup._text_box.toPlainText() == ""  # noqa: SLF001
    assert not popup._text_box.isReadOnly()  # noqa: SLF001
    assert _highlighted(popup) == ""


def test_a_lone_surrogate_in_the_text_does_not_crash_the_highlight(qtbot) -> None:  # type: ignore[no-untyped-def]
    popup = _popup(qtbot)
    text = "Bad \ud800 char. Next one."
    popup.document_loaded.emit(text)
    popup.span_started.emit(text.index("Next"), len(text))
    assert _highlighted(popup) != ""
