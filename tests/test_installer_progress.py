import pytest

from vocalis.installer.progress import ProgressEvent, parse_line


def test_parse_line_reads_a_step() -> None:
    assert parse_line("STEP 3/5 Installing models") == ProgressEvent("step", 3, 5, "Installing models")


def test_parse_line_reads_a_note() -> None:
    assert parse_line("NOTE Using the CPU only") == ProgressEvent("note", 0, 0, "Using the CPU only")


@pytest.mark.parametrize(
    "line",
    ["STEP x/y t", "STEP 0/3 t", "STEP 4/3 t", "STEP 1/3", "hello", "", "NOTE", "--- Step A: setup ---"],
)
def test_parse_line_ignores_malformed_lines(line: str) -> None:
    assert parse_line(line) is None


def test_parse_line_keeps_text_with_colons() -> None:
    event = parse_line("STEP 1/2 Download: model a: b")
    assert event is not None
    assert event.text == "Download: model a: b"


def test_parse_line_never_raises_on_huge_numbers() -> None:
    assert parse_line("STEP " + "9" * 5000 + "/3 t") is None
    assert parse_line("PROGRESS " + "9" * 5000 + " 3 1 1") is None


def test_parse_line_reads_download_progress() -> None:
    event = parse_line("PROGRESS 1200 3400 12500 180")
    assert event == ProgressEvent("progress", 0, 3400, "", done=1200, speed=12500, eta=180)


def test_parse_line_reads_unknown_eta() -> None:
    event = parse_line("PROGRESS 0 100 0 -1")
    assert event is not None and event.eta == -1


@pytest.mark.parametrize(
    "line",
    ["PROGRESS 5 4 1 1", "PROGRESS 1 0 1 1", "PROGRESS -1 4 1 1", "PROGRESS 1 4 -3 1", "PROGRESS a b c d", "PROGRESS 1 2 3"],
)
def test_parse_line_ignores_bad_progress(line: str) -> None:
    assert parse_line(line) is None
