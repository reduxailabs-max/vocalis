"""Tests for vocalis.ui.status_hub.StatusHub (synchronous: same-thread signal emits are direct)."""
import threading

from vocalis.ui.status_hub import ReadingControls, StatusHub


class Recorder:
    def __init__(self, hub: StatusHub) -> None:
        self.events: "list[tuple[object, ...]]" = []
        hub.state_changed.connect(lambda s: self.events.append(("state", s)))
        hub.reading_started.connect(lambda src, can: self.events.append(("started", src, can)))
        hub.reading_finished.connect(lambda: self.events.append(("finished",)))
        hub.paused_changed.connect(lambda p: self.events.append(("paused", p)))
        hub.status_changed.connect(lambda t: self.events.append(("status", t)))
        hub.progress_changed.connect(lambda c, t: self.events.append(("progress", c, t)))


class Calls:
    def __init__(self) -> None:
        self.log: "list[str]" = []

    def controls(self, pausable: bool = True) -> ReadingControls:
        return ReadingControls(
            stop=lambda: self.log.append("stop"),
            pause=(lambda: self.log.append("pause")) if pausable else None,
            resume=(lambda: self.log.append("resume")) if pausable else None,
        )


def test_starts_idle() -> None:
    hub = StatusHub()
    assert hub.state == "idle" and not hub.is_reading and not hub.is_paused and hub.source == ""


def test_reading_lifecycle_emits_in_order() -> None:
    hub = StatusHub()
    rec = Recorder(hub)
    hub.begin_reading("Document", Calls().controls())
    hub.report_status("Reading...")
    hub.report_progress(1, 3)
    hub.end_reading()
    assert rec.events == [
        ("state", "reading"), ("started", "Document", True),
        ("status", "Reading..."), ("progress", 1, 3),
        ("finished",), ("state", "idle"),
    ]


def test_begin_while_already_reading_and_end_while_idle_are_ignored() -> None:
    hub = StatusHub()
    rec = Recorder(hub)
    hub.end_reading()
    hub.begin_reading("Quick read", Calls().controls(pausable=False))
    hub.begin_reading("Document", Calls().controls())
    assert hub.source == "Quick read"
    assert [e for e in rec.events if e[0] == "started"] == [("started", "Quick read", False)]
    assert [e for e in rec.events if e[0] == "finished"] == []


def test_next_source_label_is_used_once() -> None:
    hub = StatusHub()
    hub.set_next_source("Screen text")
    hub.begin_reading("Document", Calls().controls())
    assert hub.source == "Screen text"
    hub.end_reading()
    hub.begin_reading("Document", Calls().controls())
    assert hub.source == "Document"


def test_listening_state_and_reading_priority() -> None:
    hub = StatusHub()
    rec = Recorder(hub)
    hub.set_listening(True)
    assert hub.state == "listening"
    hub.begin_reading("Document", Calls().controls())
    assert hub.state == "reading"
    hub.end_reading()
    assert hub.state == "listening"
    hub.set_listening(False)
    assert hub.state == "idle"
    assert [e for e in rec.events if e[0] == "state"] == [
        ("state", "listening"), ("state", "reading"), ("state", "listening"), ("state", "idle"),
    ]


def test_commands_call_the_active_readings_controls() -> None:
    hub = StatusHub()
    calls = Calls()
    hub.pause(); hub.stop()  # no active reading: harmless
    hub.begin_reading("Document", calls.controls())
    hub.pause()
    hub.resume()
    hub.stop()
    assert calls.log == ["pause", "resume", "stop"]


def test_toggle_pause_follows_the_paused_flag_and_set_paused_reports_it() -> None:
    hub = StatusHub()
    rec = Recorder(hub)
    calls = Calls()
    hub.begin_reading("Document", calls.controls())
    hub.toggle_pause()
    hub.set_paused(True)   # the reader confirms it paused
    hub.toggle_pause()
    hub.set_paused(False)
    assert calls.log == ["pause", "resume"]
    assert [e for e in rec.events if e[0] == "paused"] == [("paused", True), ("paused", False)]


def test_pause_is_ignored_for_a_stop_only_reading() -> None:
    hub = StatusHub()
    calls = Calls()
    hub.begin_reading("Quick read", calls.controls(pausable=False))
    hub.pause(); hub.toggle_pause(); hub.stop()
    assert calls.log == ["stop"]


def test_end_reading_clears_a_paused_state() -> None:
    hub = StatusHub()
    rec = Recorder(hub)
    hub.begin_reading("Document", Calls().controls())
    hub.set_paused(True)
    hub.end_reading()
    assert not hub.is_paused
    assert ("paused", False) in rec.events


def test_calls_from_another_thread_are_safe() -> None:
    hub = StatusHub()
    done = threading.Event()

    def worker() -> None:
        for _ in range(200):
            hub.begin_reading("Document", Calls().controls())
            hub.end_reading()
        done.set()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    assert done.wait(10.0)
    assert hub.state == "idle"


def test_an_ignored_begin_clears_a_pending_next_source_label() -> None:
    hub = StatusHub()
    hub.begin_reading("Quick read", ReadingControls(stop=lambda: None))
    hub.set_next_source("Screen text")
    hub.begin_reading("Document", ReadingControls(stop=lambda: None))  # ignored: one is active
    hub.end_reading()

    hub.begin_reading("Document", ReadingControls(stop=lambda: None))

    assert hub.source == "Document"
