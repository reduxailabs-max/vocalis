"""Tests for download speed/format helpers."""

from __future__ import annotations

import io

from vocalis.installer import download_progress as dp


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_format_bytes_borders() -> None:
    assert dp.format_bytes(0) == "0 B"
    assert dp.format_bytes(999) == "999 B"
    assert dp.format_bytes(1000) == "1.0 KB"
    assert dp.format_bytes(12_500_000) == "12.5 MB"
    assert dp.format_bytes(3_400_000_000) == "3.4 GB"
    assert dp.format_bytes(-5) == "0 B"


def test_format_speed() -> None:
    assert dp.format_speed(850_000) == "850.0 KB/s"
    assert dp.format_speed(1_000_000) == "1.0 MB/s"


def test_format_eta() -> None:
    assert dp.format_eta(None) == ""
    assert dp.format_eta(-1) == ""
    assert dp.format_eta(0.2) == "about 1 sec left"
    assert dp.format_eta(45) == "about 45 sec left"
    assert dp.format_eta(180) == "about 3 min left"
    assert dp.format_eta(3900) == "about 1 h 5 min left"


def test_speed_meter_speed_and_eta() -> None:
    clock = Clock()
    meter = dp.SpeedMeter(1000, clock=clock)
    clock.t = 2.0
    meter.add(200)
    assert meter.speed() == 100.0
    assert meter.eta_seconds() == 8.0


def test_resumed_bytes_count_as_done_not_speed() -> None:
    clock = Clock()
    meter = dp.SpeedMeter(1000, already_done=600, clock=clock)
    assert meter.done == 600
    assert meter.speed() == 0.0
    assert meter.eta_seconds() is None
    clock.t = 1.0
    meter.add(100)
    assert meter.speed() == 100.0


def test_meter_clamps_and_window_slides() -> None:
    clock = Clock()
    meter = dp.SpeedMeter(100, clock=clock)
    for second in range(1, 20):
        clock.t = float(second)
        meter.add(10)
    assert meter.done == 100
    assert meter.total == 100
    assert meter.speed() >= 0.0


def test_meter_allows_taking_bytes_back() -> None:
    meter = dp.SpeedMeter(100, clock=Clock())
    meter.add(50)
    meter.add(-50)
    assert meter.done == 0
    assert meter.speed() == 0.0


def test_describe_sentence() -> None:
    clock = Clock()
    meter = dp.SpeedMeter(3_400_000_000, clock=clock)
    clock.t = 1.0
    meter.add(12_500_000)
    text = dp.describe(meter)
    assert text.startswith("12.5 MB of 3.4 GB · 12.5 MB/s · about ")
    assert dp.describe(dp.SpeedMeter(10, clock=Clock())) == "0 B of 10 B"


def test_reporter_machine_line_and_throttle() -> None:
    clock = Clock()
    meter = dp.SpeedMeter(1000, clock=clock)
    out = io.StringIO()
    reporter = dp.Reporter(meter, stream=out, clock=clock)
    reporter.update()
    reporter.update()  # throttled
    clock.t = 1.0
    meter.add(500)
    reporter.update()
    lines = out.getvalue().splitlines()
    assert lines == ["PROGRESS 0 1000 0 -1", "PROGRESS 500 1000 500 1"]
    reporter.finish()
    assert out.getvalue().count("PROGRESS") == 3


def test_reporter_terminal_bar() -> None:
    class Tty(io.StringIO):
        def isatty(self) -> bool:
            return True

    clock = Clock()
    meter = dp.SpeedMeter(1000, clock=clock)
    out = Tty()
    reporter = dp.Reporter(meter, stream=out, clock=clock)
    clock.t = 1.0
    meter.add(500)
    reporter.finish()
    text = out.getvalue()
    assert " 50%" in text and "500 B of 1.0 KB" in text and text.endswith("\n")
